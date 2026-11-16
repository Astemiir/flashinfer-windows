# Windows JIT path: what this branch changes

Context: https://github.com/SystemPanic/flashinfer-windows/issues/14

The published wheels cover one environment (Python 3.12 + CUDA 12.4 + torch 2.6.0).
Everyone else on Windows builds kernels through the JIT path, which on this branch
was never exercised end to end. These are the three changes needed to make a
`batch_prefill_with_kv_cache` kernel with fp8 KV cache build and load on sm_86,
plus what is still missing further down.

All three are gated on `os.name == "nt"` / `_MSC_VER` and do not touch Linux.

## 1. `flashinfer/jit/env.py` — hash over-long directory names

A JIT spec name is the whole specialization and reaches 185 characters:

```
batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False_kv_stride_equal
```

Even with the flat `%SystemDrive%:\_fij` root the artifact path passes `MAX_PATH`:

| path under `C:\_fij\0.7.0.post1\86` | length |
|---|---|
| `generated\<spec>\generated_batch_prefill_paged.cu` | 252 |
| `cached_ops\<spec>\obj8.cuda.o` | 231 |
| `cached_ops\<spec>\<spec>.dll` | **409** |

The source fits by eight characters; the library never does. With the home-rooted
default layout (`%USERPROFILE%\.cache\flashinfer\<version>\<arch>`) the source path
itself is 278, and that is where nvcc failed first on the machine this was found on:

```
nvcc fatal   : Error while converting file name using Windows API function
GetLongPathName("C:/fi/.cache/flashinfer/0.7.2.post1/86/generated/<spec>/generated_batch_prefill_paged.cu")
```

`nvcc` resolves file names through the ANSI `GetLongPathName`, which fails above
260 characters regardless of `LongPathsEnabled` (it is `1` on the machine this was
found on), so the name itself has to shrink. `_ShortPath` hashes any extension-less
segment longer than 32 characters to `s` + md5 prefix, inside `__truediv__`, so all
writers — `core`, `cpp_ext`, the attention/gemm/mamba modules — agree on the path
without being changed. Generated source files keep their names because they are
`#include`d by name.

Result for the spec above: 278 → 64 for the source, 435 → 47 for the library.

## 2. `flashinfer/jit/cpp_ext.py` — `/FORCE:UNRESOLVED`, short artifact name

* `link.exe` gets `/FORCE:UNRESOLVED`. The kernel generator skips unreachable
  template variants on purpose (`CTA_TILE_Q=32` for `head_dim < 512`), and the
  host object still references them. A Linux `.so` keeps those symbols unresolved;
  `link.exe` stops with `LNK2019` on
  `BatchPrefillWithRaggedKVCacheDispatched<32,256,256,0,0,0,...>` and
  `LNK1120: 8 unresolved externals`. The paths are never entered.
* The artifact is named `m` + md5(`spec`)[:10] + `.dll`
  (`windows_artifact_name`), and `core.jit_library_path` derives the same name.
  `<spec>.dll` is 189 characters on its own and overflows together with its
  directory.

## 3. `include/flashinfer/msvc_compat.cuh` — MSVC rejects in the kernels

Force-included with `--pre-include` for CUDA compilations on Windows only:

* `ushort` (GCC gets it from `<sys/types.h>`),
* `__attribute__((...))` and `__restrict__` spelling,
* `M_PI` / `M_SQRT2`,
* `std::max` / `std::min` in `__device__` code: MSVC's `<algorithm>` versions are
  host-only, and on LLP64 `16UL / sizeof(T)` (64-bit) against `HEAD_DIM / 32U`
  (32-bit) also defeats deduction — `error: no instance of overloaded function
  "std::max" matches the argument list`, `error: cannot deduce "auto" type`.
  The header provides `fi_max` / `fi_min` over `std::common_type`, and the kernel
  sites are rewritten to those names mechanically.

## Still open

* **`#if MODE ==` inside `DISPATCH_*` arguments.** Not present in this tree; it
  appears in 0.7.2, where MSVC rejects the construction in the generated dispatch
  lambdas (744 blocks / 652 files) and the chain has to become `if constexpr`.
* **`flashinfer.jit.topk`** fails inside CCCL on MSVC
  (`cub/thread/thread_load.cuh(265): error: ThreadLoad is not a template`, ~100
  errors). Consumers can skip that path; it is unrelated to attention.
* Verified numbers on sm_86 (one RTX 3090, 27B target, 33 364-token fill,
  500-token answer): with this JIT path plus `--attention-backend FLASHINFER
  --kv-cache-dtype fp8_e4m3`, 1 187.5 tok/s prefill and 78.9 tok/s decode, against
  858.7 / 28.0 for an int4 KV cache on the stock path.

## Not verified here

This branch has not been compiled from source: it needs the 0.7.0 AOT toolchain
(CUDA 13.2, Python 3.12, the submodules). The same three edits were verified on an
installed 0.7.2.post1 package, where they are what turned a JIT failure into a
serving engine; the diff above is that change carried back to this tree. If you
want it validated here, the shortest check is one kernel:

```bat
set "FLASHINFER_WORKSPACE_BASE=%SystemDrive%"
python -c "from flashinfer.jit.attention import gen_batch_prefill_module; gen_batch_prefill_module('fa2', 'float16', 'float16', ['float16'], ['float16'], [128], [128], [0], False).build()"
```
