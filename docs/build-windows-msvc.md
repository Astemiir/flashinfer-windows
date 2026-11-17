# Building FlashInfer for Windows on Ampere (sm_80 / sm_86)

A one-off build, not a project. Goal: a **portable** wheel with pre-compiled
kernels, so that someone on a 3090 (or A100) runs `pip install` and gets
`--attention-backend FLASHINFER` with fp8 KV and a 131K window — **without
nvcc, without MSVC and without hand-applied patches**.

Verified on 17 November 2026 on the machine that built the release: Python
3.13.14 (the portable interpreter from the vLLM Windows build), torch 2.14.0+cu130,
CUDA toolkit 13.3 in `C:\cuda133`, MSVC 19.44 (VS 2022 BuildTools), ninja,
RTX 3090 (sm_86).

Paths below use `C:\fi` as the build root and `C:\vllm-win` as the vLLM portable
directory. Any short root works; the root must be short, see section 2.

## Files in this repository

| path | what it is |
|---|---|
| `tools/windows-msvc/patch_win_source.py` | applies every source edit listed in 4.4 to a FlashInfer tree |
| `tools/windows-msvc/flashinfer_win_shim.cuh` | the compatibility header the patched sources include |
| `tools/windows-msvc/build_jitcache.py`, `tools/windows-msvc/build_jitcache.bat` | compile and package the kernel set |
| `tools/windows-msvc/make_wheel.py` | package the kernels into the provider wheel |
| `tools/windows-msvc/make_shim_wheel.py` | package the discovery shim wheel |
| `tools/windows-msvc/build_python_wheel.bat` | build `flashinfer_python` with the patches baked in |
| `tools/windows-msvc/verify_dist.py` | check the three installed wheels without a source tree |
| `tools/windows-msvc/provider_init.py`, `tools/windows-msvc/shim_init.py` | copies of the upstream provider and shim `__init__.py` |
| `patches/windows-msvc/windows_patch_format.patch`, `patches/windows-msvc/windows_patch_stride.patch` | submodule patches for spdlog and cutlass |

Every script reads its paths from environment variables, so the CONFIG block at
the top of the `.bat` files is the only place to edit.

---

## 1. What comes out

The delivery is three wheels plus a `README.md`:

| artifact | file | size |
|---|---|---|
| sources with the patches | `flashinfer_python-0.7.2.post1+win.1-py3-none-any.whl` | 59.5 MB |
| discovery shim | `flashinfer_jit_cache-0.7.2.post1+win.1-py3-none-any.whl` | 3 KB |
| jit-cache | `flashinfer_jit_cache_sm80-0.7.2.post1+win.1-cp39-abi3-win_amd64.whl` | 19.3 MB |
| modules inside | 29: prefill / attention / decode for head dims **128 and 256**, in **bf16, fp16, fp8_e4m3**, plus `spdlog` | ~2 MB each |
| check | provider accepted, `try_load()=Module`, `build()` with JIT off → `MissingJITCacheError`; vLLM with `FLASHINFER_DISABLE_JIT=1` answers correctly | — |

Intermediate artifacts:

| what | where |
|---|---|
| kernel wheel | `C:\fi\wheel-src\dist\` (copied to `dist-win`) |
| python wheel | `C:\fi\dist-python\` |
| shim | `C:\fi\shim-src\dist\` → `dist-win` |

Reference output of a correct build:

```
flashinfer: 0.7.2.post1+win.1
provider: sm80 ver=0.7.2.post1+win.1 modules=29
spec: ...f16qk_False_kv_stride_equal
artifacts=1  path=91 chars  ...\jit_cache\s88bfe1c0e44e\s88bfe1c0e44e.so
try_load=Module exports=20
```

---

## 2. Layout (everything on a short root)

A short root is required **for the build**: at that stage a spec name reaches 185
characters, and its directory under `b\aot\cached_ops` runs into the ANSI limit
of `nvcc`. The finished wheel no longer has this problem — kernel directories
there are `s<md5[:12]` (see 7.1) — so it installs into any `site-packages`.

| path | what |
|---|---|
| `C:\fi\src` | upstream `v0.7.2.post1` sources with submodules (cccl, cutlass, nixl, spdlog) |
| `C:\fi\src\flashinfer\data` | created by `build_backend._create_data_dir()`; the shim header is not found without it |
| `C:\fi\b` | build workspace (`FLASHINFER_WORKSPACE_BASE`); `b\aot\cached_ops` holds the specs |
| `C:\fi\pkg` | packaged modules in provider layout (`providers/sm80/jit_cache/s<md5>/s<md5>.so`) |
| `C:\fi\wheel-src` | temporary wheel tree (`flashinfer_jit_cache_provider` + `setup.py`) |
| `C:\fi\shim-src` | temporary tree for the shim |
| `C:\fi\dist-test` | clean install of all three wheels — the "as a user gets it" check |
| `C:\fi\dist-win` | **the delivery**: three wheels plus `README.md` |
| `C:\fi\tmp` | `TMP`/`TEMP`; otherwise nvcc writes `tmpxft_*` into the system temp and hurts |

---

## 3. What the machine needs

- **MSVC 2022** (`vcvars64.bat`) — the host compiler for nvcc.
- **CUDA 13.3** in `C:\cuda133` (not "Program Files": a shorter path, no spaces).
- **ninja** — the one pip puts into `Scripts`.
- **`tvm_ffi.lib`** — an import library generated from `tvm_ffi.dll`
  (`dumpbin /exports` → `.def` → `lib /def`). Example location: `C:\tvmffi`.
- **The portable Python from the vLLM build** — but note it is **embedded**:
  `python313._pth` **ignores `PYTHONPATH`**. Add the tree from inside the script:
  `sys.path.insert(0, r"C:\fi\src")`.

---

## 4. Build order

### 4.1 Clone onto a short root

```bat
git clone --single-branch --branch v0.7.2.post1 --recurse-submodules --shallow-submodules --depth 1 ^
  https://github.com/flashinfer-ai/flashinfer.git C:\fi\src
```

### 4.2 Create `flashinfer\data` (otherwise the shim and include paths are not found)

```python
import importlib.util, sys, pathlib
ROOT = pathlib.Path(r"C:\fi\src"); sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("bb", ROOT / "build_backend.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod._create_data_dir(use_symlinks=False)
```

### 4.3 Submodule patches (mandatory, the build fails without them)

Copy `windows_patch_format.patch` and `windows_patch_stride.patch` from this
repository's `patches/windows-msvc/`:

```bat
cd /d C:\fi\src\3rdparty\spdlog  && git apply --check windows_patch_format.patch && git apply windows_patch_format.patch
cd /d C:\fi\src\3rdparty\cutlass && git apply --check windows_patch_stride.patch && git apply windows_patch_stride.patch
```

### 4.4 Port FlashInfer to MSVC (`tools/windows-msvc/patch_win_source.py`)

It edits the **source tree**, not an installed package:

```bat
set "FI_SRC=C:\fi\src"
py -3.13 tools\windows-msvc\patch_win_source.py
```

What it changes:

| file | edit |
|---|---|
| `flashinfer/jit/cpp_ext.py` | `.cu` and host files through `nvcc`; drop `depfile`/`deps=gcc` on nt; `-x cu` for host files; `objN` objects; paths via `_ninja_path` (forward slashes, `E$:`); `nvcc_link` rule on nt with `-Xlinker /FORCE:UNRESOLVED`; short artifact name `m<md5[:10]>.dll`; POSIX flag filter |
| `flashinfer/jit/core.py` | `jit_library_path` plus a new `jit_library_name` → short `.dll`; `subninja` line through `_ninja_path`; `-Wno-switch-bool` only off nt |
| `flashinfer/jit/env.py` | `_ShortPath` class: a dot-less segment longer than 32 characters becomes `s<md5[:12]>`; plus reading the AOT cache by short names — `_short_aot_module`/`_aot_module_paths` (see 7.1) |
| `flashinfer/aot.py` | build directories through `_ShortPath`; module name from `jit_library_name`; `copy_built_kernels` writes short `s<md5[:12]>/s<md5[:12]>.so` (see 7.1); cuDNN and MLA off by default (`FLASHINFER_AOT_CUDNN`, `FLASHINFER_AOT_MLA`) |
| `build_backend.py` | `_create_data_dir` copies into `data` without `.git` (otherwise `rmtree` fails with `WinError 32`); `_clean_data_dir` in the three `_prepare_*` instead of a bare `rmtree` |
| `include/**` | `std::max/min` → `fi_max/fi_min` (105 and 82 places) |
| `include/flashinfer/flashinfer_win_shim.cuh` | the new shim header: `ushort`, `__attribute__`, `__restrict__`, `M_PI`, `fi_max/fi_min` over `std::common_type` |
| `csrc/**` | `#if X == A / #elif / #else / #endif` → `if constexpr` (743 blocks in 651 files): MSVC rejects directives inside a `DISPATCH_*` macro argument |

The script prints `applied:` for every anchor it found and `NOT FOUND:` for the
ones it did not. An empty `NOT FOUND` list means the port is in place.

### 4.5 Environment variables (`tools/windows-msvc/build_jitcache.bat`)

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\cuda133"
set "CUDA_PATH=C:\cuda133"
set "PATH=C:\cuda133\bin;%PATH%"
set "FLASHINFER_WORKSPACE_BASE=C:/fi/b"
set "FLASHINFER_EXTRA_CUDAFLAGS=--pre-include=C:/fi/src/include/flashinfer/flashinfer_win_shim.cuh -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING -Xcompiler=/utf-8 -Xcompiler=/wd4996"
set "FLASHINFER_EXTRA_CFLAGS="
set "FLASHINFER_WIN_LDFLAGS=-Xlinker /LIBPATH:C:/vllm-win/python/libs C:/tvmffi/tvm_ffi.lib"
set "FLASHINFER_DISABLE_VERSION_CHECK=1"
set "TMP=C:\fi\tmp"
set "TEMP=C:\fi\tmp"
set "TMPDIR=C:/fi/tmp"
set "FLASHINFER_CUDA_ARCH_LIST=8.0"
```

**Important:** keep `FLASHINFER_EXTRA_CFLAGS` **empty**. Bare MSVC flags
(`/utf-8`, `/wd4996`) without `-Xcompiler=` are taken by nvcc as a second input
file and it stops with `A single input file is required`.

### 4.6 Build the kernel set (`tools/windows-msvc/build_jitcache.py`)

```python
CONFIG = {
    "fa2_head_dim": [(128, 128), (256, 256)],
    "fa3_head_dim": [],                       # Hopper is not needed
    "f16_dtype": [torch.bfloat16, torch.float16],
    "f8_dtype": [torch.float8_e4m3fn],
    "use_sliding_window": [False],
    "use_logits_soft_cap": [False],
    "add_comm": False, "add_gemma": False, "add_oai_oss": False,
    "add_moe": False, "add_act": False, "add_misc": False, "add_xqa": False,
}
aot.compile_and_package_modules(out_dir=..., build_dir=..., project_root=...,
                                config=CONFIG, verbose=True, skip_prebuilt=False)
```

`arch 8.0` is deliberate: sm_80 SASS runs on sm_86 (same major version), so one
wheel covers both A100 and 3090. FlashInfer's own compatibility check agrees:
`cuda_binary_target_compatibility_score('sm80','sm86') = (1, 0)`.

The module list is taken from what was written — every directory holding a
`*.so` — so it stays correct after the short-name change.

### 4.7 Package the wheels (`tools/windows-msvc/make_shim_wheel.py`, `tools/windows-msvc/make_wheel.py`)

What they do and where it is easy to trip:

- package: `flashinfer_jit_cache.providers.sm80`, entry point `sm80=...:get_provider`
  in the group **`flashinfer.jit_cache.providers`**;
- **the wheel version must match the shim version and the flashinfer version**
  (`0.7.2.post1+win.1` for all three here). The local suffix is compared
  literally, so all three wheels carry the same string, not just the cache;
- **`manifest["modules"]`** is the list of directories that really hold a `*.so`
  (after the short-name change these are `s<md5[:12]>`). An empty list gives
  "Provider has no modules"; a name mismatch gives "kernel not found" and a
  silent fallback to JIT;
- wheel tag: without a custom `bdist_wheel` setuptools produces `py3-none-any` —
  a subclass is needed with `root_is_pure=False`, `py_limited_api="cp39"` and
  `get_tag() → ("cp39","abi3",platform)`;
- the files keep the `.so` extension (or `<short>.so`) but are **PE (MZ)** images;
  Windows `LoadLibrary` loads them by that name without complaint (verified:
  `MZ` signature, `tvm_ffi.load_module` returns a `Module`).

Order for the whole delivery:

```bat
py -3.13 tools\windows-msvc\make_shim_wheel.py        :: shim  (3 KB, pure Python)
py -3.13 tools\windows-msvc\make_wheel.py             :: cache (19.3 MB, cp39-abi3-win_amd64)
tools\windows-msvc\build_python_wheel.bat             :: sources (59.5 MB, py3-none-any)
```

`build_python_wheel.bat` runs `pip wheel . --no-deps --no-build-isolation` with
the vLLM interpreter, `FLASHINFER_LOCAL_VERSION=win.1`, `BUILD_NVEP=0` and
`FLASHINFER_BUILD_NO_PIP=1`; `build_backend._create_build_metadata()` writes the
version into `_build_meta.py` from `version.txt` plus that suffix.

---

## 5. Verification (mandatory, otherwise "it built" means nothing)

```python
sys.path.insert(0, r"C:\fi\dist-test")   # shim + kernel wheel installed there
os.environ["FLASHINFER_DISABLE_JIT"] = "1"
```

and use the spec builder **that vLLM itself calls**:

```python
from flashinfer.jit.attention.modules import _gen_batch_prefill_module
spec = _gen_batch_prefill_module("fa2", torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16,
                                 torch.int32, 256, 256, 0, False, False, False,
                                 paged_kv_stride_mode="equal", module_surface="full")
assert len(spec.aot_artifacts) == 1        # kernel found in the cache
assert spec.try_load() is not None         # and it loads
try: spec.build(); raise AssertionError("JIT should not have been needed")
except Exception as e: print(type(e).__name__)  # MissingJITCacheError
```

**Trap:** the public `gen_batch_prefill_module(...)` produces a name **without**
`_kv_stride_equal`, so `aot_artifacts` is 0. You need
**`paged_kv_stride_mode="equal"`** (not `"runtime"`): that is what the AOT set
builds and what vLLM requests.

`tools/windows-msvc/verify_dist.py` does exactly this against an installed set of wheels.

---

## 6. Pitfalls already hit (symptom → cause → fix)

| symptom | cause | fix |
|---|---|---|
| `PYTHONPATH` has no effect, the installed flashinfer is imported | the vLLM portable Python is embedded (`python313._pth`) | `sys.path.insert(0, ...)` inside the script |
| `C1083: cannot open flashinfer/flashinfer_win_shim.cuh` | `flashinfer\data` is missing from the tree | `build_backend._create_data_dir()` |
| `ninja: loading 'E': The system cannot find the file specified` | `subninja E:\...` without escaping | `_ninja_path` (forward slashes, `E$:`) |
| `nvcc fatal: Unknown option '-fPIC'` | upstream POSIX flags | flag filter on nt |
| `Unknown option '-Wno-switch-bool'` | a GCC warning flag in `core.py` | drop it on nt |
| `A single input file is required...` | a) host `.cc` without `-x cu`; b) bare `/utf-8 /wd4996` in CFLAGS | `-x cu` for `compile`; keep CFLAGS empty |
| `character value is out of range` in `spdlog/fmt` | no Windows patch for spdlog | `windows_patch_format.patch` |
| `LNK1104: cannot open python3.lib` | the `spdlog` module pulls in the Python C API | `-Xlinker /LIBPATH:...\python\libs` |
| `LNK2019 ... BatchPrefillWith...Dispatched<32,...>` | the generator does not instantiate unreachable variants | `-Xlinker /FORCE:UNRESOLVED` |
| `error: expression must be a pointer to a complete object type` in `cudnn_sdpa_kernel_launcher.cu` | the cuDNN module is not needed and does not build against these headers | `FLASHINFER_AOT_CUDNN=0` |
| `argument list for template ... is missing` in `mla.cuh` | the MLA (DeepSeek) code does not parse under MSVC | `FLASHINFER_AOT_MLA=0` |
| `Provider 'sm80' has no modules` | the manifest list was empty (counted `.dll`, files are `.so`) | count directories holding `*.so` |
| `version does not match flashinfer-jit-cache version` | the wheel carries a local `+cu130` tag | wheel version = shim version; here all three share `+win.1` |
| `MissingJITCacheError` from `build()` in the check | **correct**: JIT is off and not needed | — |
| `PermissionError [WinError 32] ... data\spdlog\.git` while building the python wheel | `_prepare_for_wheel` calls `rmtree(data)` and `copytree` dragged submodule gitlink files in | copy with `ignore_patterns(".git", ...)` in `_create_data_dir`, clean with `_clean_data_dir()` (symlinks and busy files no longer break the build) |
| `WinError 183: ... flashinfer_python-0.7.2.post1+win.1.dist-info already exists` | leftovers of an interrupted run in `src\build` | delete `C:\fi\src\build` and repeat |
| `ModuleNotFoundError: No module named 'setuptools'` in `make_wheel.py` | the `python` on PATH is 3.15, which has no setuptools | `py -3.13 tools\windows-msvc\make_wheel.py` (3.13 has setuptools 80.9 and wheel 0.45) |
| `ModuleNotFoundError: No module named 'tvm_ffi'` during verification | `py -3.13` has no `tvm_ffi` | verify with the vLLM interpreter (`...\python\python.exe`) |

---

## 7. Portability — what was done and what is left

### 7.1 Long names inside the cache — built with short names

**What it was.** Inside the provider there was `<spec 185 chars>/<spec 185 chars>.so`,
so the full path reached **~434 characters**:

```
<site-packages>\flashinfer_jit_cache\providers\sm80\jit_cache\
  batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_..._kv_stride_equal\
  batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_..._kv_stride_equal.so
```

On this machine it worked because `LongPathsEnabled=1` is set and Python 3.13
understands long paths. Without that, installation or loading would hit 260, and
`nvcc` with its ANSI `GetLongPathName` gives up at 248 for a directory.

**The idea.** The directory and file name is not the spec itself but its hash:
`s` plus the first 12 characters of the md5 of the full name. No collisions
across 29 modules, and the name no longer depends on spec length.

**What was built — two edits in the sources.**

1. `flashinfer/jit/env.py` — reading. Two functions were added (before `_get_aot_providers`):

```python
def _short_aot_module(name: str) -> str:
    """Directory/file of a module in the AOT cache on Windows: s<md5[:12]>."""
    if os.name != "nt":
        return name
    return "s" + hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def _aot_module_paths(base: pathlib.Path, module_name: str) -> Tuple[pathlib.Path, ...]:
    """Short path first (our cache), then the long one (a foreign upstream cache)."""
    short = _short_aot_module(module_name)
    paths = [base / short / f"{short}.so"]
    if short != module_name:
        paths.append(base / module_name / f"{module_name}.so")
    return tuple(paths)
```

and **three** places where the path used to be glued by hand were replaced:

| where | before | after |
|---|---|---|
| `get_aot_artifacts` → fallback in `data\aot` | `FLASHINFER_AOT_DIR / module_name / f"{module_name}.so"` | iterate `_aot_module_paths(FLASHINFER_AOT_DIR, module_name)` |
| `get_aot_artifacts` → provider membership | `if module_name not in provider.modules: continue` | skip only when **neither** the full nor the short name is in `manifest["modules"]` |
| `get_aot_artifacts` → provider path | `provider.jit_cache_dir / module_name / f"{module_name}.so"` | `next((p for p in _aot_module_paths(provider.jit_cache_dir, module_name) if p.exists()), None)` |

`get_aot_path()` was also switched to `_aot_module_paths(...)[0]`, otherwise its
"stable fallback" would return a path that does not exist.

2. `flashinfer/aot.py`, `copy_built_kernels` — writing:

```python
src = jit_env.FLASHINFER_JIT_DIR / jit_spec.name / jit_spec.jit_library_name
module_dir = (
    jit_env._short_aot_module(jit_spec.name) if os.name == "nt" else jit_spec.name
)
dst = out_dir / module_dir / f"{module_dir}.so"
```

`make_wheel.py` needed no change: `manifest["modules"]` was already computed from
reality — "directories that hold a `*.so`" — so short names went in by themselves.
`env` now accepts them.

**How it was rebuilt.** No compilation was needed; only packaging changes:

```bat
cd /d C:\fi
tools\windows-msvc\build_jitcache.bat      :: ninja: no work to do, only the copy into pkg
py -3.13 tools\windows-msvc\make_wheel.py  :: kernel wheel with short directories
```

`copy_built_kernels` removes `out_dir` itself, so old long directories in
`C:\fi\pkg` disappear without manual cleanup. Objects in
`C:\fi\b\aot\cached_ops` stay, so the whole rebuild takes seconds.

**What came out:**

```
...\jit_cache\s88bfe1c0e44e\s88bfe1c0e44e.so
```

- 30 characters relative to `jit_cache` (was ~380);
- worst full path (short root, long user name, full Python path) — **165**
  characters, well under the 240 where ANSI paths still work;
- the whole cache: 29 directories, 29 `.so` files, each named `s<md5[:12]>` of
  its spec.

The check is the same as before and still mandatory: `FLASHINFER_DISABLE_JIT=1`
with `paged_kv_stride_mode="equal"` — `artifacts=1`, a 91-character path,
`try_load=Module`, 20 exports. After installing the three wheels into an empty
directory it is the same, no source tree involved.

**Backward compatibility.** A cache already installed with full names (a Linux
provider, or an older build of ours) keeps working: `_aot_module_paths` tries the
short path first and the long one second. That is also why you cannot simply
rename directories inside someone else's wheel: its `manifest["modules"]` lists
the full names, and the membership check has to know both spellings — it does.

**If a cache was already built with long names and you would rather not rebuild:**
rename the directories and files by the same rule (`s` + `md5(spec_name)[:12]`)
and leave the full names in `manifest.json` — `env` will find the short path from
the manifest's long name. Rebuilding the packaging is cheaper, though: it is
copying, not compiling.

### 7.2 The delivery — three wheels

| wheel | size | contents |
|---|---|---|
| `flashinfer_python-0.7.2.post1+win.1-py3-none-any.whl` | 59.5 MB | sources with the patches |
| `flashinfer_jit_cache-0.7.2.post1+win.1-py3-none-any.whl` | 3 KB | discovery shim (entry points) |
| `flashinfer_jit_cache_sm80-0.7.2.post1+win.1-cp39-abi3-win_amd64.whl` | 19.3 MB | 29 kernels, sm_80 SASS |

The three versions must match: `_check_jit_cache_version` and
`_check_jit_cache_provider_version` compare the string literally, and the
`+win.1` suffix is part of that comparison. So the version is set in one place —
`FLASHINFER_LOCAL_VERSION=win.1` when building the python wheel — and read from
`FI_VERSION` by `make_wheel.py` and `make_shim_wheel.py`.

Verified end to end: the wheels were installed into the vLLM kit, the server was
started with `FLASHINFER_DISABLE_JIT=1` (the compiler is not reachable at all),
the answer to `17*23` was correct, and the log shows
`JIT kernel warmup finished in 0.00s` — not a single JIT call.

### 7.3 What is left

1. **A cu129 variant** for users on older drivers (CUDA 13 requires driver 580+).
2. **Publication**: a release in our fork with the three wheels and the README.
3. Other architectures (sm_89, sm_90, sm_120) — the same pipeline with a different
   `FLASHINFER_CUDA_ARCH_LIST` and an `smXX` provider, but sm_80 SASS does not
   carry over to them.

---

## 8. Why bother (numbers for context)

| configuration on RTX 3090 | max context | prefill at 33K | decode |
|---|---|---|---|
| bf16 KV, stock `TRITON_ATTN` | 65,536 | 1,038 | 85 |
| int8_per_token_head, stock | 131,072 | 828 | 76 |
| **fp8_e4m3 + `FLASHINFER`** | **131,072** | **1,182** | **82** |

So the port gives **+42.8% prefill and +7.6% decode** against the best variant
without it, and it is the only way to get fp8 KV on sm_86 (the stock backend
answers `FP8 KV cache is not supported ... requires SM89+`).

The same pair at equal KV (bf16, 65,536 window) isolates the backend: +18.0%
prefill and no decode change.

---

## 9. Rebuilding for another FlashInfer version

**Four things** change: the clone tag, the version in `version.txt`, the local
version suffix, and the version constants in the packagers. Everything else is
the same commands.

### 9.1 What to change

| where | what | example |
|---|---|---|
| `git clone --branch v0.7.2.post1` | version tag | `v0.7.4`, `v0.7.5rc3` |
| `src\version.txt` | base version (read by `_create_build_metadata`) | `0.7.4` |
| `build_python_wheel.bat` → `FLASHINFER_LOCAL_VERSION` | local suffix; becomes part of the python wheel version | `win.2`, empty |
| `FI_VERSION` for `make_wheel.py` | kernel wheel version, **exactly as the python wheel** | `0.7.4+win.2` |
| `FI_VERSION` for `make_shim_wheel.py` | shim version, **the same string** | `0.7.4+win.2` |
| `FLASHINFER_CUDA_ARCH_LIST` | `8.0` for Ampere (A100 + 3090), `8.6` for 3090 only | `8.0` |
| `CONFIG["fa2_head_dim"]` | the head dims the models need | `[(128,128),(256,256)]` |

### 9.2 Order

1. **A new clean root**, for example `C:\fi2` (leave the old one alone until the
   new build passes). Space: the tree with submodules is ~**1.2 GB**, the build
   ~0.2 GB, the packaged modules ~30 MB.
2. Clone the tag (4.1), then `_create_data_dir()` (4.2).
3. Submodule patches (4.3). If `git apply --check` fails, spdlog or cutlass moved
   and the patch has to be refreshed.
4. `patch_win_source.py` with the new `FI_SRC`. The script **tells you** what it
   did not find — it prints `NOT FOUND: ...` per anchor. An empty list means the
   port is in place; otherwise fix the anchors against the real code. Check the
   `build_backend.py` edit separately (the `.git` ignore when copying into
   `data`) — it is part of the same script.
5. The `.bat` with the variables (4.5) — pre-include and `LIBPATH` must point at
   the new root.
6. `build_jitcache.py` → a set in `C:\fi2\pkg`. Watch the log: `Total ops: N`,
   then `AOT kernels saved to: ...`, and a list of modules with sizes at the end.
7. Three wheels in the order of 4.7: shim, cache, python. Before packaging the
   cache, make sure the version matches the shim and `version.txt` plus the
   suffix, otherwise the provider is rejected.
8. Verification (5) **with** `FLASHINFER_DISABLE_JIT=1` and the
   `paged_kv_stride_mode="equal"` builder, then a vLLM smoke run
   (`FLASHINFER_DISABLE_JIT=1`, `--attention-backend FLASHINFER`,
   `--kv-cache-dtype fp8_e4m3`) with one request and a checked answer.

### 9.3 What usually breaks when the version changes

| symptom | what it means |
|---|---|
| `git apply --check` fails on spdlog/cutlass | the submodules moved; a fresh Windows patch is needed |
| `patch_win_source.py` prints `NOT FOUND: ...` | upstream rewrote that block — fix the anchor, do not ignore it |
| a new module fails (`cudnn_sdpa_kernel_launcher.cu`, `mla.cuh`, `*_blackwell_*`) | those kernels are not for Ampere: turn them off (`FLASHINFER_AOT_CUDNN`, `FLASHINFER_AOT_MLA`) or trim `CONFIG` |
| `Provider has no modules` | the manifest came out empty: count modules from directories holding a `*.so` |
| `version does not match` | wheel version ≠ shim version (including a stray `+cu130`) |
| `MissingJITCacheError` during verification | **this is success**: JIT is disabled and was not needed |

### 9.4 Should you take a newer version

For sm_80/sm_86 there is no announced speed-up in 0.7.3–0.7.5: everything there
targets Rubin (sm_107a) and Blackwell (sm_100a/103a/120). Upgrading only makes
sense when **vLLM**, the thing FlashInfer is installed for, moves to that version
itself: its code expects a specific API. While vLLM 0.33 expects
`0.7.2.post1`, the build is made for it.
