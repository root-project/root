//===- ErrorInternal.cpp - Impl-side Result<T> + ErrorRef + Diag --- C++ -===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CppInterOp/Error.h"

#include "CppInterOp/CppInterOpTypes.h"

#include "Compatibility.h"
#include "ErrorInternal.h"
#include "InterpreterInfo.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>

namespace Cpp {

// A null DiagnosticRef returns defaults so callers don't have to
// branch on isNull() before every access.
static const StoredDiagView* AsView(DiagnosticRef D) {
  return static_cast<const StoredDiagView*>(D.data);
}

CPPINTEROP_API DiagnosticSeverity GetDiagnosticSeverity(DiagnosticRef D) {
  if (const StoredDiagView* V = AsView(D))
    return V->Sev;
  return DiagnosticSeverity::Note;
}

CPPINTEROP_API const char* GetDiagnosticMessage(DiagnosticRef D) {
  if (const StoredDiagView* V = AsView(D))
    return V->Message.c_str();
  return "";
}

CPPINTEROP_API const char* GetDiagnosticFile(DiagnosticRef D) {
  if (const StoredDiagView* V = AsView(D))
    return V->File.c_str();
  return "";
}

CPPINTEROP_API unsigned GetDiagnosticLine(DiagnosticRef D) {
  if (const StoredDiagView* V = AsView(D))
    return V->Line;
  return 0;
}

CPPINTEROP_API unsigned GetDiagnosticColumn(DiagnosticRef D) {
  if (const StoredDiagView* V = AsView(D))
    return V->Column;
  return 0;
}

// Member-function forwarders.
DiagnosticSeverity DiagnosticRef::severity() const {
  return GetDiagnosticSeverity(*this);
}
const char* DiagnosticRef::message() const {
  return GetDiagnosticMessage(*this);
}
const char* DiagnosticRef::file() const { return GetDiagnosticFile(*this); }
unsigned DiagnosticRef::line() const { return GetDiagnosticLine(*this); }
unsigned DiagnosticRef::column() const { return GetDiagnosticColumn(*this); }

CPPINTEROP_API Status GetStatus(ErrorRef E) {
  if (E.isOk())
    return Status::Ok;
  if (E.isInline())
    return E.inlineStatus();
  if (const ErrorSlice* S = E.slice())
    return S->Code;
  return Status::Ok;
}

CPPINTEROP_API ArrayView<DiagnosticRef> GetDiagnostics(ErrorRef E) {
  const ErrorSlice* S = E.slice();
  if (!S || S->Diagnostics.empty())
    return {};
  // DiagnosticRefs are constructed lazily into a thread-local buffer
  // so the returned ArrayView<DiagnosticRef> can alias contiguous
  // storage (slice.Diagnostics is a std::deque -- not contiguous).
  // The buffer survives until the next GetDiagnostics call on the
  // same thread; callers that need to retain the view should copy it.
  static thread_local std::vector<DiagnosticRef> Buf;
  Buf.clear();
  Buf.reserve(S->Diagnostics.size());
  for (const StoredDiagView& Dv : S->Diagnostics)
    Buf.push_back(DiagnosticRef{&Dv});
  return ArrayView<DiagnosticRef>{Buf.data(), Buf.size()};
}

// Member-function forwarders.
Status ErrorRef::status() const { return GetStatus(*this); }
ArrayView<DiagnosticRef> ErrorRef::diagnostics() const {
  return GetDiagnostics(*this);
}
const char* ErrorRef::producer() const {
  const ErrorSlice* S = slice();
  return S ? S->Producer : nullptr;
}
const char* ErrorRef::producerSignature() const {
  const ErrorSlice* S = slice();
  return S ? S->ProducerSignature : nullptr;
}

ErrorRecord ErrorRef::record() const {
  ErrorRecord R;
  R.Code = status();
  const ErrorSlice* S = slice();
  if (!S)
    return R;
  R.Diagnostics.reserve(S->Diagnostics.size());
  for (const StoredDiagView& Dv : S->Diagnostics) {
    DiagnosticInfo Di;
    Di.Severity = Dv.Sev;
    Di.Message = Dv.Message;
    Di.File = Dv.File;
    Di.Line = Dv.Line;
    Di.Column = Dv.Column;
    R.Diagnostics.push_back(std::move(Di));
  }
  R.Producer = S->Producer;
  R.ProducerSignature = S->ProducerSignature;
  return R;
}

CPPINTEROP_API void RetainErrorRef(ErrorRef E) {
  // Inline-encoded errors carry no refcount; the bits are pure value.
  if (!E.isSlice())
    return;
  E.slice()->Refcount.fetch_add(1, std::memory_order_acq_rel);
}

CPPINTEROP_API void ReleaseErrorRef(ErrorRef E) {
  if (!E.isSlice())
    return;
  const ErrorSlice* S = E.slice();
  if (S->Refcount.fetch_sub(1, std::memory_order_acq_rel) == 1)
    delete S;
}

char StatusError::ID = 0;

void StatusError::log(llvm::raw_ostream& OS) const {
  OS << "CppInterOp status error: " << Message;
}
std::error_code StatusError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

CPPINTEROP_API ErrorSlice* AllocSlice(InterpreterInfo* Owner) {
  auto* S = new ErrorSlice();
  S->Owner = Owner;
  return S;
}

CPPINTEROP_API ErrorRef makeError(Status S) { return ErrorRef::makeInline(S); }

CPPINTEROP_API ErrorRef makeError(InterpreterInfo* II, Status S,
                                  std::string Message, const char* Producer,
                                  const char* ProducerSig) {
  ErrorSlice* Slc = AllocSlice(II);
  Slc->Code = S;
  Slc->Producer = Producer;
  Slc->ProducerSignature = ProducerSig;
  if (!Message.empty()) {
    StoredDiagView Dv;
    Dv.Message = std::move(Message);
    Dv.Sev = DiagnosticSeverity::Error;
    Slc->Diagnostics.push_back(std::move(Dv));
  }
  DrainPendingInto(II, Slc);
  // Refcount stays at the default 0; the wrapping Result<T> bumps to 1
  // via its ErrorRef ctor.
  return ErrorRef::makeSlice(Slc);
}

CPPINTEROP_API ErrorRef drainError(InterpreterInfo* II, llvm::Error E,
                                   const char* Producer,
                                   const char* ProducerSig) {
  // Decompose the llvm::Error chain into a single slice. StatusError
  // sets the Status directly; any other ErrorInfo subclass is
  // stringified via its log() and surfaced as Status::CompileError.
  // Typed payloads (overload / deduction) and their llvm::Error
  // subclasses arrive with the migrations that need them.
  Status Code = Status::CompileError;
  std::string Msg;

  llvm::handleAllErrors(
      std::move(E),
      [&](const StatusError& SE) {
        Code = SE.Code;
        if (Msg.empty())
          Msg = SE.Message;
      },
      [&](const llvm::ErrorInfoBase& EIB) {
        // Unknown subclass: stringify and surface as CompileError.
        if (Msg.empty()) {
          llvm::raw_string_ostream OS(Msg);
          EIB.log(OS);
        }
      });

  ErrorSlice* Slc = AllocSlice(II);
  Slc->Code = Code;
  Slc->Producer = Producer;
  Slc->ProducerSignature = ProducerSig;
  if (!Msg.empty()) {
    StoredDiagView Dv;
    Dv.Message = std::move(Msg);
    Dv.Sev = DiagnosticSeverity::Error;
    Slc->Diagnostics.push_back(std::move(Dv));
  }
  DrainPendingInto(II, Slc);
  return ErrorRef::makeSlice(Slc);
}

namespace {
/// Captures every diagnostic the parser/sema emits into the owning
/// interpreter's StoredDiags and forwards to the previously installed
/// consumer (typically Clang's TextDiagnosticPrinter) so existing
/// stderr output is preserved. Owned holds the chained consumer when
/// the engine owned it (clang-REPL); Raw points at it regardless of
/// ownership so cling's externally-owned consumer also forwards.
class CppInteropDiagConsumer : public clang::DiagnosticConsumer {
public:
  CppInteropDiagConsumer(InterpreterInfo* II,
                         std::unique_ptr<clang::DiagnosticConsumer> Owned,
                         clang::DiagnosticConsumer* Raw)
      : II(II), Owned(std::move(Owned)), Raw(Raw) {}

  void BeginSourceFile(const clang::LangOptions& LO,
                       const clang::Preprocessor* PP) override {
    if (Raw)
      Raw->BeginSourceFile(LO, PP);
  }
  void EndSourceFile() override {
    if (Raw)
      Raw->EndSourceFile();
  }
  void HandleDiagnostic(clang::DiagnosticsEngine::Level Level,
                        const clang::Diagnostic& Info) override;

private:
  InterpreterInfo* II;
  std::unique_ptr<clang::DiagnosticConsumer> Owned;
  clang::DiagnosticConsumer* Raw;
};

DiagnosticSeverity MapClangSeverity(clang::DiagnosticsEngine::Level L) {
  switch (L) {
  case clang::DiagnosticsEngine::Ignored:
  case clang::DiagnosticsEngine::Note:
  case clang::DiagnosticsEngine::Remark:
    return DiagnosticSeverity::Note;
  case clang::DiagnosticsEngine::Warning:
    return DiagnosticSeverity::Warning;
  case clang::DiagnosticsEngine::Error:
    return DiagnosticSeverity::Error;
  case clang::DiagnosticsEngine::Fatal:
    return DiagnosticSeverity::Fatal;
  }
  return DiagnosticSeverity::Error;
}
} // namespace

void CppInteropDiagConsumer::HandleDiagnostic(
    clang::DiagnosticsEngine::Level Level, const clang::Diagnostic& Info) {
  // Update the base consumer's NumErrors so hasErrorOccurred() keeps
  // working for callers that still rely on it.
  clang::DiagnosticConsumer::HandleDiagnostic(Level, Info);
  if (Raw)
    Raw->HandleDiagnostic(Level, Info);

  llvm::SmallString<128> Buf;
  Info.FormatDiagnostic(Buf);

  StoredDiagView Dv;
  Dv.Message = Buf.str().str();
  Dv.Sev = MapClangSeverity(Level);

  if (Info.hasSourceManager() && Info.getLocation().isValid()) {
    clang::SourceManager& SM = Info.getSourceManager();
    clang::PresumedLoc PLoc = SM.getPresumedLoc(Info.getLocation());
    if (PLoc.isValid()) {
      if (const char* F = PLoc.getFilename())
        Dv.File = F;
      Dv.Line = PLoc.getLine();
      Dv.Column = PLoc.getColumn();
    }
  }

  II->StoredDiags.push_back(std::move(Dv));
}

CPPINTEROP_API void InstallDiagConsumer(InterpreterInfo* II) {
  // Chain to whatever consumer Clang already installed so existing
  // stderr behaviour is preserved. takeClient transfers ownership when
  // the engine owns it (clang-REPL); cling installs its consumer with
  // ShouldOwnClient=false, so takeClient returns null there and we
  // forward through the raw pointer instead. getClient still returns
  // the original between takeClient and setClient.
  clang::DiagnosticsEngine& Diag = II->Interpreter->getCI()->getDiagnostics();
  std::unique_ptr<clang::DiagnosticConsumer> PrevOwned = Diag.takeClient();
  clang::DiagnosticConsumer* PrevRaw = Diag.getClient();
  auto* New = new CppInteropDiagConsumer(II, std::move(PrevOwned), PrevRaw);
  Diag.setClient(New, /*ShouldOwnClient=*/true);
}

CPPINTEROP_API void DrainPendingInto(InterpreterInfo* II, ErrorSlice* S) {
  for (auto& Dv : II->StoredDiags)
    S->Diagnostics.push_back(std::move(Dv));
  II->StoredDiags.clear();
}

CPPINTEROP_API void ClearPending(InterpreterInfo* II) {
  II->StoredDiags.clear();
}

CPPINTEROP_API unsigned GetPendingDiagnosticCount(InterpRef I) {
  return static_cast<unsigned>(GetInterpInfo(I)->StoredDiags.size());
}

CPPINTEROP_API DiagnosticRef GetPendingDiagnostic(unsigned Idx, InterpRef I) {
  InterpreterInfo* II = GetInterpInfo(I);
  if (Idx >= II->StoredDiags.size())
    return DiagnosticRef{};
  return DiagnosticRef{&II->StoredDiags[Idx]};
}

CPPINTEROP_API void ClearPendingDiagnostics(InterpRef I) {
  GetInterpInfo(I)->StoredDiags.clear();
}

[[noreturn]] CPPINTEROP_API void
ResultAbort_ValueOnError(const ErrorRef& /*Err*/) {
  std::fputs("Cpp::Result<T>::value() called on an error-bearing "
             "Result. Use value_or(fallback) for lenient semantics, "
             "or branch on .ok() / .error() before calling .value().\n",
             stderr);
  std::abort();
}

#ifndef NDEBUG
[[noreturn]] CPPINTEROP_API void
ResultAbort_UncheckedOnDtor(const ErrorRef& /*Err*/) {
  std::fputs("Cpp::Result destroyed without check (likely a dropped "
             "error). Call .ok() / .error() / .value() to inspect, "
             "or .ignore() to acknowledge.\n",
             stderr);
  std::abort();
}
#endif

} // namespace Cpp
