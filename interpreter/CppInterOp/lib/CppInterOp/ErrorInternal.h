//===--- ErrorInternal.h - Impl-side error helpers --------------*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Impl-side counterpart to CppInterOp/Error.h: the ErrorSlice body,
// the llvm::Error subclasses, and the boundary translator.
//
//===----------------------------------------------------------------------===//

#ifndef CPPINTEROP_LIB_ERRORINTERNAL_H
#define CPPINTEROP_LIB_ERRORINTERNAL_H

#include "CppInterOp/CppInterOpTypes.h"
#include "CppInterOp/Error.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <string>
#include <system_error>
#include <utility>

namespace Cpp {

struct InterpreterInfo; // defined in CppInterOp.cpp

/// Owning record for one captured diagnostic. DiagnosticRef::data
/// points at one of these.
struct StoredDiagView {
  std::string Message;
  std::string File;
  unsigned Line = 0;
  unsigned Column = 0;
  DiagnosticSeverity Sev = DiagnosticSeverity::Error;
};

/// Body of a slice-encoded error: drained diagnostics, producer
/// attribution, refcount. alignas(16) keeps the low bits of an
/// ErrorSlice* clear so ErrorRef's tag bit stays unambiguous.
struct alignas(16) ErrorSlice {
  // Mutable so Retain/Release can act on a `const ErrorSlice*` taken
  // off an ErrorRef accessor without round-tripping through const_cast.
  mutable std::atomic<uint32_t> Refcount{0};
  InterpreterInfo* Owner = nullptr;
  Status Code = Status::Ok;

  // Stable element addresses under push_back so a DiagnosticRef can
  // address an entry safely.
  std::deque<StoredDiagView> Diagnostics;

  // Pointer into static __func__ storage. No copy.
  const char* Producer = nullptr;
  // Pointer into static __PRETTY_FUNCTION__ storage. No copy.
  const char* ProducerSignature = nullptr;
};

//
// Carry the structured information the boundary translator drains
// into a slice. Impl code returns llvm::Expected<T> with these on
// the failure path; makeResult wraps them into Result<T>.

class CPPINTEROP_API StatusError : public llvm::ErrorInfo<StatusError> {
public:
  static char ID;
  Status Code;
  std::string Message;

  StatusError(Status C, std::string M) : Code(C), Message(std::move(M)) {}

  void log(llvm::raw_ostream& OS) const override;
  [[nodiscard]] std::error_code convertToErrorCode() const override;
};

/// Allocate a fresh slice on the heap, refcount 0. Caller bumps the
/// refcount on first use (the boundary translator does this when
/// wrapping the slice into a returned ErrorRef).
CPPINTEROP_API ErrorSlice* AllocSlice(InterpreterInfo* Owner);

/// Build an inline-status ErrorRef (no diagnostics, no payload).
CPPINTEROP_API ErrorRef makeError(Status S);

/// Build a slice-encoded error carrying a Clang-derived message + the
/// interp's pending diagnostics + producer attribution.
CPPINTEROP_API ErrorRef makeError(InterpreterInfo* II, Status S,
                                  std::string Message, const char* Producer,
                                  const char* ProducerSig);

/// Drain `E` into a fresh slice, returning the slice via ErrorRef.
/// Handles StatusError; any other llvm::Error is stringified via
/// llvm::ErrorInfoBase::log into the slice's first diagnostic.
CPPINTEROP_API ErrorRef drainError(InterpreterInfo* II, llvm::Error E,
                                   const char* Producer,
                                   const char* ProducerSig);

/// Convert llvm::Expected<T> into Result<T>. On the error path, drain
/// the carried llvm::Error into a slice.
template <typename T>
Result<T> makeResult(InterpreterInfo* II, llvm::Expected<T> E,
                     const char* Producer, const char* ProducerSig) {
  if (E)
    return Result<T>(std::move(*E));
  return Result<T>(drainError(II, E.takeError(), Producer, ProducerSig));
}

inline Result<void> makeResult(InterpreterInfo* II, llvm::Error E,
                               const char* Producer, const char* ProducerSig) {
  if (!E)
    return Result<void>();
  return Result<void>(drainError(II, std::move(E), Producer, ProducerSig));
}

/// Drain the consumer's per-interp buffer into a slice on failure.
/// Success paths just clear via ClearPending -- callers don't surface
/// Ok-path warnings today.
CPPINTEROP_API void DrainPendingInto(InterpreterInfo* II, ErrorSlice* S);
CPPINTEROP_API void ClearPending(InterpreterInfo* II);

//
// Production callers read diagnostics off Result<T>::error() once the
// boundary translator has drained them into a slice. These survive
// for tests of the consumer install path and as a peek for callers
// not yet migrated to Result-returning APIs.

CPPINTEROP_API unsigned GetPendingDiagnosticCount(InterpRef I = nullptr);
CPPINTEROP_API DiagnosticRef GetPendingDiagnostic(unsigned Idx,
                                                  InterpRef I = nullptr);
CPPINTEROP_API void ClearPendingDiagnostics(InterpRef I = nullptr);

} // namespace Cpp

#endif // CPPINTEROP_LIB_ERRORINTERNAL_H
