//===--- Unwrap.h - wrap/unwrap for opaque handle types ---------*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal helpers for converting between CppInterOp's typed opaque handles
// (DeclRef, TypeRef, FuncRef, ObjectRef, InterpRef and their const variants)
// and raw pointers to the underlying Clang AST nodes. Not part of the public
// API — included only from lib/CppInterOp/*.cpp and unit-test TUs that need
// to materialise typed pointers for testing the primitives themselves.
//
// Downstream consumers (cppyy, xeus-cpp) call the public C++ / C APIs and
// never need wrap/unwrap; this header is deliberately not reachable from
// include/CppInterOp/CppInterOp.h.
//
//===----------------------------------------------------------------------===//

#ifndef CPPINTEROP_LIB_UNWRAP_H
#define CPPINTEROP_LIB_UNWRAP_H

#include "CppInterOp/CppInterOpTypes.h"

namespace Cpp {

// wrap<Handle>(ptr) materialises a handle from a raw pointer.
// unwrap<T>(handle) extracts the raw pointer, statically cast to T*.
// Const handles yield const T* via their const void* data member.

template <typename Handle> Handle wrap(void* P) { return Handle{P}; }

// Const handles from const pointers — no const_cast needed.
template <typename Handle> Handle wrap(const void* P) { return Handle{P}; }

// Mutable handles — unwrap to non-const pointer.
template <typename T> T* unwrap(DeclRef H) { return static_cast<T*>(H.data); }
template <typename T> T* unwrap(TypeRef H) { return static_cast<T*>(H.data); }
template <typename T> T* unwrap(FuncRef H) { return static_cast<T*>(H.data); }
template <typename T> T* unwrap(ObjectRef H) { return static_cast<T*>(H.data); }
template <typename T> T* unwrap(InterpRef H) { return static_cast<T*>(H.data); }

// Const handles — unwrap to const pointer.
template <typename T> const T* unwrap(ConstDeclRef H) {
  return static_cast<const T*>(H.data);
}
template <typename T> const T* unwrap(ConstTypeRef H) {
  return static_cast<const T*>(H.data);
}
template <typename T> const T* unwrap(ConstFuncRef H) {
  return static_cast<const T*>(H.data);
}

} // namespace Cpp

#endif // CPPINTEROP_LIB_UNWRAP_H
