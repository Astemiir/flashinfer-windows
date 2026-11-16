/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MSVC compatibility layer for the JIT path.
//
// Included with `--pre-include` on Windows (see flashinfer/jit/cpp_ext.py) so
// every generated translation unit sees it before the kernels pull in their
// headers. Everything here exists because the kernels are written against a
// GCC/Clang host and MSVC rejects the spelling, or provides the name on the
// host side only.
#ifndef FLASHINFER_MSVC_COMPAT_CUH_
#define FLASHINFER_MSVC_COMPAT_CUH_

#include <type_traits>

namespace flashinfer {
namespace msvc_compat {

// `std::max` / `std::min` from <algorithm> are host-only, so a `__device__`
// function calling them does not compile with the MSVC host. They also refuse
// to deduce a type when the arguments differ, which happens easily here:
// `16UL / sizeof(T)` is 64-bit on LLP64 while `HEAD_DIM / 32U` is 32-bit.
// Promote both sides to their common type first.
//
// Defined on every platform, not only MSVC, because the kernel sites are
// rewritten to these names unconditionally; with GCC/Clang they behave exactly
// like the `std::` versions except for the extra promotion.
template <class A, class B>
__host__ __device__ constexpr typename std::common_type<A, B>::type fi_max(A a, B b) {
  using T = typename std::common_type<A, B>::type;
  return static_cast<T>(a) > static_cast<T>(b) ? static_cast<T>(a) : static_cast<T>(b);
}

template <class A, class B>
__host__ __device__ constexpr typename std::common_type<A, B>::type fi_min(A a, B b) {
  using T = typename std::common_type<A, B>::type;
  return static_cast<T>(a) < static_cast<T>(b) ? static_cast<T>(a) : static_cast<T>(b);
}

}  // namespace msvc_compat
}  // namespace flashinfer

// Kernel sites are rewritten to these two names (mechanical replacement of
// `std::max(` / `std::min(` in the headers); the macros keep them short.
#define fi_max flashinfer::msvc_compat::fi_max
#define fi_min flashinfer::msvc_compat::fi_min

#if defined(_MSC_VER)

// GCC pulls `ushort` out of <sys/types.h>; MSVC has no such name, and the
// kernels use it as a type in parameter lists.
#ifndef ushort
typedef unsigned short ushort;
#endif

// GCC attribute syntax in headers (`__attribute__((always_inline))` and
// friends). MSVC knows nothing about it and needs the declaration itself.
#ifndef __attribute__
#define __attribute__(x)
#endif

// MSVC spells the qualifier without the trailing underscores.
#ifndef __restrict__
#define __restrict__ __restrict
#endif

// Constants that <cmath>/<math.h> provide elsewhere.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

#endif  // defined(_MSC_VER)

#endif  // FLASHINFER_MSVC_COMPAT_CUH_
