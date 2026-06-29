//===--- InterpreterInfo.h - Per-interpreter impl state ---------*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Impl-only header centralising the per-interpreter state struct used by
// CppInterOp.cpp and Error.cpp. Not part of the public API; never
// included from include/CppInterOp/.
//
//===----------------------------------------------------------------------===//

#ifndef CPPINTEROP_LIB_INTERPRETERINFO_H
#define CPPINTEROP_LIB_INTERPRETERINFO_H

#include "Compatibility.h"
#include "ErrorInternal.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"

#include "llvm/ADT/StringMap.h"

#include <deque>
#include <map>

namespace Cpp {

struct InterpreterInfo {
  compat::Interpreter* Interpreter = nullptr;
  bool isOwned = true;
  // Store the list of builtin types.
  llvm::StringMap<clang::QualType> BuiltinMap;
  // Per-interpreter wrapper caches. Keyed on AST nodes that belong to this
  // interpreter, so the caches must be destroyed together with it.
  std::map<const clang::FunctionDecl*, void*> WrapperStore;
  std::map<const clang::Decl*, void*> DtorWrapperStore;
  // A deque keeps element addresses stable so DiagnosticRef::data
  // survives push_back.
  std::deque<StoredDiagView> StoredDiags;

  InterpreterInfo(compat::Interpreter* I, bool Owned)
      : Interpreter(I), isOwned(Owned) {}

  InterpreterInfo(InterpreterInfo&& Other) noexcept
      : Interpreter(Other.Interpreter), isOwned(Other.isOwned) {
    Other.Interpreter = nullptr;
    Other.isOwned = false;
  }
  InterpreterInfo& operator=(InterpreterInfo&& Other) noexcept {
    if (this != &Other) {
      if (isOwned)
        delete Interpreter;
      Interpreter = Other.Interpreter;
      isOwned = Other.isOwned;
      Other.Interpreter = nullptr;
      Other.isOwned = false;
    }
    return *this;
  }

  ~InterpreterInfo() {
    if (isOwned)
      delete Interpreter;
  }

  InterpreterInfo(const InterpreterInfo&) = delete;
  InterpreterInfo& operator=(const InterpreterInfo&) = delete;
};

/// Resolve an InterpRef to the impl-side struct. When I is null,
/// returns the active (last-created) interpreter.
CPPINTEROP_API InterpreterInfo* GetInterpInfo(InterpRef I = nullptr);

/// Wire CppInterOp's DiagnosticConsumer into the interpreter's
/// DiagnosticsEngine so parser/sema diagnostics flow into II->StoredDiags
/// and continue forwarding to whatever consumer Clang already had.
CPPINTEROP_API void InstallDiagConsumer(InterpreterInfo* II);

} // namespace Cpp

#endif // CPPINTEROP_LIB_INTERPRETERINFO_H
