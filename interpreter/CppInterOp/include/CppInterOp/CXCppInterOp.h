//===- CXCppInterOp.h - C API for the CppInterOp library --------*- C -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public C API for CppInterOp. Wraps the tablegen-generated declarations
// in CXCppInterOpDecl.inc and adds the hand-written extras that cannot
// be emitted by the generator (NoCWrapper return types, mixed return +
// out-param shapes).
//
//===----------------------------------------------------------------------===//

#ifndef CPPINTEROP_CXCPPINTEROP_H
#define CPPINTEROP_CXCPPINTEROP_H

#include "CppInterOp/CppInterOpTypes.h"

#ifdef __cplusplus
// The generated .inc spells C-API parameter and return types with the
// prefixed C-side names (CppDeclRef etc.). For C++ TUs that opt into the
// C API by including this header, alias each prefixed name to its
// namespaced Cpp::* counterpart — they're layout-identical, so this
// makes them the same C++ type and no conversion happens at call sites.
// C++ TUs that include only CppInterOp.h (the C++ API header) never see
// these aliases — the prefixed spelling stays out of their namespace.
using CppDeclRef = Cpp::DeclRef;
using CppTypeRef = Cpp::TypeRef;
using CppFuncRef = Cpp::FuncRef;
using CppObjectRef = Cpp::ObjectRef;
using CppInterpRef = Cpp::InterpRef;
using CppConstDeclRef = Cpp::ConstDeclRef;
using CppConstTypeRef = Cpp::ConstTypeRef;
using CppConstFuncRef = Cpp::ConstFuncRef;

// The C-linkage functions return Cpp::*Ref types (one-word PODs with
// inline constructors). Clang warns about returning a "user-defined
// type" with C linkage even though the ABI is fine — the C-side
// spelling is a layout-identical struct without constructors, and
// HandleTypesTest::AbiCompatibleWithVoidPtr proves the call convention
// matches. Silence the warning for downstream C++ consumers using -I.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern "C" {
#endif

#include "CppInterOp/CXCppInterOpDecl.inc"

// --- Hand-written wrappers (see lib/CppInterOp/CXCppInterOp.cpp) ---

/// C-ABI overload of Cpp::Evaluate. Returns the execution result as
/// a raw \c intptr_t bit pattern; on parse error or no-value-after-
/// success writes \c true to \c *HadError (if non-null) and returns
/// \c ~0UL. The Box-returning C++ overload is marked NoCWrapper.
CPPINTEROP_API intptr_t cppinterop_Evaluate(const char* code, bool* HadError);

/// Returns the templated method scopes inside \c parent matching
/// \c name. The bool-return-with-vector-outparam shape is not
/// expressible by the tablegen wrapper emitter; callers check
/// \c arr.size > 0 to detect "no matches".
CPPINTEROP_API CppInterOpArray
cppinterop_GetClassTemplatedMethods(const char* name, CppConstDeclRef parent);

#ifdef __cplusplus
} // extern "C"
#pragma clang diagnostic pop
#endif

#endif // CPPINTEROP_CXCPPINTEROP_H
