# Why this Windows build exists

This is the background behind the wheels in the
[v0.7.2.post1-win.1 release](https://github.com/Astemiir/flashinfer-windows/releases/tag/v0.7.2.post1-win.1):
where the work started, why it ended up on FlashInfer, and what it is worth now.
All numbers are from one RTX 3090 with a 27B model in 4 bits.

## Where it started

It did not start with FlashInfer. It started with the question "why is
Qwen3.8-27B answering so slowly?" — that is, with decode.

1. Decode on llama.cpp was stuck at about 90 tok/s (79–89 with MTP n3, 36
   without). For a 27B model in 4 bits on a 3090 that is close to the bandwidth
   limit: roughly 15.4 GB of weights against 936 GB/s of memory, so every step
   is one pass over the weights. The same engine cannot go much further — what
   is needed is an engine that produces more tokens per step.
2. On 15 November a native Windows build of vLLM 0.33 appeared (aivrar, cu130).
   We installed it and measured: int4 KV gave 234 tok/s of decode at 32K against
   94 on llama.cpp — twice as much, exactly from speculation and batching. That
   is why vLLM became interesting at all.
3. Two walls showed up at the same time. `expandable_segments` is unusable in
   that build: it fails at startup with a GPU mapping OOM. And fp8 KV was not
   available at all. fp8 was not wanted for its own sake — the DFlash2-W4A16
   model card, which promised 115 tok/s, said that on Ampere the fp8 KV cache
   for the drafter works only through FlashInfer.
4. The FlashInfer wheel did not build on Windows (its JIT is written for Linux),
   so the choice was either to drop DFlash2 or to port. We ported.

Then, during measurements, DFlash2 lost to MTP (71.4 against 78.9), so that
fourth step turned out to be needed for something other than what it was built
for. The port stayed anyway: a 131,072-token window on fp8 KV and +18% prefill.
That is what it lives for now.

In one line: we started with "we want faster answers — decode" and ended with
"we read the prompt faster — prefill — and hold a 131K context".

## Why we took on FlashInfer

Not for vLLM speed; that came as a side effect. It started as a precondition for
DFlash2.

1. The DFlash2-W4A16 model card promised 115 tok/s of decode on a single
   RTX 3090 — twice what we had with MTP. Its one caveat: "The fp8 KV cache works
   for the draft on Ampere (FlashInfer supports its non-causal attention)".
2. On Ampere, only FlashInfer can do fp8 KV. The stock Triton backend refuses
   it: requires SM89+. That was an assumption read off the card; it turned out
   to be exact.
3. The FlashInfer wheel does not build on Windows. The JIT generator assumes a
   POSIX toolchain (c++, `.so`, GCC flags), and no pre-compiled kernels are
   published for Windows. vLLM's own pin of `flashinfer-python` is marked
   `platform_system == "Linux"`, so nobody installed it on Windows in the first
   place.
4. So simply running DFlash2 the way its authors measured it required building
   FlashInfer under MSVC first. Hence the port: a compatibility shim, nvcc
   instead of c++, `.dll` artifacts, short paths, `if constexpr` instead of
   preprocessor directives inside macro arguments.

What came out of it: DFlash2 lost to MTP (71.4 against 78.9 at 32K), so the
original reason fell away — the drafter this was all for is not used. The port
remained and found other work: an fp8 window at 131,072 and +18% prefill at
equal KV.

The honest summary: had we known on 15 November that we would stay on MTP, we
would have started with int8 on the stock backend and most likely never taken on
FlashInfer at all. The interest in DFlash2 is what carried the port of FlashInfer
to Windows for the RTX 3090.

The value turned out to be different from the one it was built for, but it is
real: at 131K we now get 1,182 / 82 against 828 / 76 for the unpatched variant,
a gain of +42.8% prefill and +7.6% decode.

| configuration on RTX 3090 | max context | prefill at 33K | decode |
|---|---:|---:|---:|
| bf16 KV, stock `TRITON_ATTN` | 65,536 | 1,038 | 85 |
| int8 KV, stock `TRITON_ATTN` (no patch needed) | 131,072 | 828 | 76 |
| **fp8_e4m3 KV + `FLASHINFER`** | **131,072** | **1,182** | **82** |

The +42.8% / +7.6% pair compares the working configurations at the same window,
which is what a user actually gets; on that window the KV type changes together
with the backend (int8 → fp8), because the stock backend has no fp8 KV on
Ampere. Backend alone, at equal bf16 KV and a 65,536 window, is +18.0% prefill
and no decode change (85.4 → 82.8 is inside the ±5% run-to-run spread).

## What is in the box now

See the release page for the install command, the verification snippet and the
limits (Ampere only, CUDA 13.x only, a fixed kernel set). The build recipe,
including every MSVC blocker and how it was worked around, is written down
separately so the same pipeline can be repeated for another version.
