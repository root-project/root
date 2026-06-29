//===--- CppInterOpThunks.h - Variadic dispatcher template ------*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Header-only variadic-template dispatcher that bindings instantiate per
// signature × slot to populate a precompiled thunk catalog. CppInterOp ships
// the template body; the binding contributes the per-instance handler lookup
// and the language-specific marshaling.
//
// The binding provides a HandlerTraits struct of this shape:
//
//   struct HandlerTraits {
//     using Handler = /* binding-specific per-instance state */;
//
//     // Per-instance lookup -- typically reads the handler pointer from an
//     // extra prefix slot reserved via MakeVTableOverlay's
//     // n_extra_prefix_slots, i.e. vptr[-(kVTableOverlayPrefixSize + 1)].
//     static Handler& From(void* self);
//
//     // Marshal Args... into the target language, call the bound callable
//     // at index Slot, and marshal the result back as R.
//     template <class R, class... Args>
//     static R Call(Handler&, std::size_t Slot, Args... args);
//   };
//
// Each instantiation of dispatch<T, Slot, R, Args...> produces a concrete
// function with the right calling convention to install in a patched vtable
// slot. The catalog is a per-binding table of such instantiations indexed by
// (signature shape, slot).
//
//===----------------------------------------------------------------------===//

#ifndef CPPINTEROP_CPPINTEROPTHUNKS_H
#define CPPINTEROP_CPPINTEROPTHUNKS_H

#include <cstddef>

namespace Cpp {
namespace Thunks {

template <class T, std::size_t Slot, class R, class... Args>
R dispatch(void* self, Args... args) {
  return T::template Call<R, Args...>(T::From(self), Slot, args...);
}

} // namespace Thunks
} // namespace Cpp

#endif // CPPINTEROP_CPPINTEROPTHUNKS_H
