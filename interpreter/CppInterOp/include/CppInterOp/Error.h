//===--- Error.h - Result<T>, ErrorRef, DiagnosticRef -----------*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cpp::Result<T> carries either a T or an ErrorRef. ErrorRef encodes
// either an inline Status code (no allocation) or a pointer to an
// ErrorSlice owning drained Clang diagnostics, an optional structured
// payload, and producer attribution.
//
//===----------------------------------------------------------------------===//

#ifndef CPPINTEROP_ERROR_H
#define CPPINTEROP_ERROR_H

#include "CppInterOp/CppInterOpTypes.h"

#ifdef __cplusplus

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Cpp {

/// Outcome of a fallible API call.
enum class Status : uint8_t {
  Ok = 0,
  NotFound,        // name lookup found nothing
  InvalidArgument, // precondition violated by caller
  Ambiguous,       // name lookup returned multiple matches
  ParseError,      // Clang parser rejected the input
  CompileError,    // Sema/codegen rejected the input
};

/// Severity of a captured diagnostic.
enum class DiagnosticSeverity : uint8_t { Note, Warning, Error, Fatal };

/// Non-owning view over a contiguous range. Returned from accessors so
/// callers can iterate without committing to a container type.
template <typename T> struct ArrayView {
  const T* Data = nullptr;
  size_t Size = 0;

  [[nodiscard]] const T* begin() const { return Data; }
  [[nodiscard]] const T* end() const { return Data + Size; }
  [[nodiscard]] size_t size() const { return Size; }
  [[nodiscard]] bool empty() const { return Size == 0; }
  const T& operator[](size_t I) const { return Data[I]; }
};

/// 8-byte opaque handle to one captured diagnostic. Valid while the
/// owning error (or interpreter) keeps its storage alive.
struct DiagnosticRef {
  const void* data = nullptr;

  [[nodiscard]] bool isNull() const { return data == nullptr; }

  // Member-function surface (forwarders defined in ErrorInternal.cpp).
  CPPINTEROP_API DiagnosticSeverity severity() const;
  CPPINTEROP_API const char* message() const;
  CPPINTEROP_API const char* file() const;
  CPPINTEROP_API unsigned line() const;
  CPPINTEROP_API unsigned column() const;
};

/// Field accessors. The TableGen binding surface; member functions
/// above forward here. Safe to call on a null DiagnosticRef: empty
/// strings, zero positions, Severity::Note.
CPPINTEROP_API DiagnosticSeverity GetDiagnosticSeverity(DiagnosticRef D);
CPPINTEROP_API const char* GetDiagnosticMessage(DiagnosticRef D);
CPPINTEROP_API const char* GetDiagnosticFile(DiagnosticRef D);
CPPINTEROP_API unsigned GetDiagnosticLine(DiagnosticRef D);
CPPINTEROP_API unsigned GetDiagnosticColumn(DiagnosticRef D);

//
// Encoding (8 bytes, single uintptr_t):
//   bits == 0      : Ok.
//   bits & 1 == 1  : inline Status in bits[1..7]. No allocation.
//   bits & 1 == 0,
//   bits != 0      : pointer to ErrorSlice (diagnostics, payload,
//                    refcount). Result<T> manages the refcount.

struct ErrorSlice; // defined in lib/CppInterOp/ErrorInternal.h

struct ErrorRef {
  uintptr_t bits = 0;

  [[nodiscard]] bool isOk() const { return bits == 0; }
  [[nodiscard]] bool isInline() const { return (bits & 1U) == 1U; }
  [[nodiscard]] bool isSlice() const { return bits != 0 && (bits & 1U) == 0u; }

  /// Pack a Status code into the ErrorRef bits. Status::Ok produces
  /// the canonical null encoding so isOk() stays a single comparison.
  static ErrorRef makeInline(Status S) {
    if (S == Status::Ok)
      return ErrorRef{};
    return ErrorRef{(static_cast<uintptr_t>(S) << 1) | 1U};
  }

  /// The inline-encoded Status. Only meaningful when isInline() is
  /// true; callers should prefer status() which handles both shapes.
  [[nodiscard]] Status inlineStatus() const {
    return static_cast<Status>(bits >> 1);
  }

  /// Wrap a slice pointer. The slice's alignment must keep bit 0 clear
  /// (ErrorSlice is alignas(16) for this reason).
  static ErrorRef makeSlice(const ErrorSlice* S) {
    return ErrorRef{reinterpret_cast<uintptr_t>(S)};
  }
  [[nodiscard]] const ErrorSlice* slice() const {
    return isSlice() ? reinterpret_cast<const ErrorSlice*>(bits) : nullptr;
  }

  // Member-function surface (defined in Error.cpp).
  CPPINTEROP_API Status status() const;
  CPPINTEROP_API ArrayView<DiagnosticRef> diagnostics() const;
  CPPINTEROP_API const char* producer() const;
  CPPINTEROP_API const char* producerSignature() const;
  CPPINTEROP_API class ErrorRecord record() const;
};

/// Free-function aliases over the ErrorRef accessors. The TableGen
/// binding surface; the member functions above forward here.
CPPINTEROP_API Status GetStatus(ErrorRef E);
CPPINTEROP_API ArrayView<DiagnosticRef> GetDiagnostics(ErrorRef E);

/// Refcount hooks. Result<T> calls these around copy/move/dtor; they
/// are no-ops on Ok and inline encodings.
CPPINTEROP_API void RetainErrorRef(ErrorRef E);
CPPINTEROP_API void ReleaseErrorRef(ErrorRef E);

//
// Deep-copy snapshot for callers that want a plain value-type and do
// not need the originating slice kept alive. Strings are copied;
// nothing points back into slice storage.

struct DiagnosticInfo {
  DiagnosticSeverity Severity = DiagnosticSeverity::Error;
  std::string Message;
  std::string File;
  unsigned Line = 0;
  unsigned Column = 0;
};

class ErrorRecord {
public:
  Status Code = Status::Ok;
  std::vector<DiagnosticInfo> Diagnostics;
  // Pointers into static __func__ / __PRETTY_FUNCTION__ storage. No
  // copying needed; the storage outlives every record.
  const char* Producer = nullptr;
  const char* ProducerSignature = nullptr;
};

//
// Extends a slice's lifetime past the originating Result<T> via one
// refcount bump on construction and one decrement on destruction.
// Inline errors are pure value -- no refcount work.

class [[nodiscard]] CapturedError {
  ErrorRef Ref;

public:
  CapturedError() = default;
  explicit CapturedError(ErrorRef E) : Ref(E) { RetainErrorRef(Ref); }

  CapturedError(const CapturedError& O) : Ref(O.Ref) { RetainErrorRef(Ref); }
  CapturedError(CapturedError&& O) noexcept : Ref(O.Ref) { O.Ref = ErrorRef{}; }

  CapturedError& operator=(const CapturedError& O) {
    if (this != &O) {
      ReleaseErrorRef(Ref);
      Ref = O.Ref;
      RetainErrorRef(Ref);
    }
    return *this;
  }
  CapturedError& operator=(CapturedError&& O) noexcept {
    if (this != &O) {
      ReleaseErrorRef(Ref);
      Ref = O.Ref;
      O.Ref = ErrorRef{};
    }
    return *this;
  }

  ~CapturedError() { ReleaseErrorRef(Ref); }

  [[nodiscard]] bool ok() const { return Ref.isOk(); }
  explicit operator bool() const { return ok(); }
  [[nodiscard]] Status status() const { return Ref.status(); }
  [[nodiscard]] ErrorRef ref() const { return Ref; }

  [[nodiscard]] ArrayView<DiagnosticRef> diagnostics() const {
    return Ref.diagnostics();
  }
  [[nodiscard]] const char* producer() const { return Ref.producer(); }
  [[nodiscard]] const char* producerSignature() const {
    return Ref.producerSignature();
  }
  [[nodiscard]] ErrorRecord record() const { return Ref.record(); }
};

[[noreturn]] CPPINTEROP_API void ResultAbort_ValueOnError(const ErrorRef& Err);

#ifndef NDEBUG
[[noreturn]] CPPINTEROP_API void
ResultAbort_UncheckedOnDtor(const ErrorRef& Err);
#endif

template <typename T> class [[nodiscard]] Result {
  static_assert(!std::is_same<T, void>::value,
                "Result<void> is provided via specialization below");

  static constexpr size_t kStorageSize = sizeof(T) > sizeof(ErrorRef)
                                             ? sizeof(T)
                                             : sizeof(ErrorRef);
  static constexpr size_t kStorageAlign = alignof(T) > alignof(ErrorRef)
                                              ? alignof(T)
                                              : alignof(ErrorRef);

  union {
    alignas(kStorageAlign) unsigned char m_ValueBytes[kStorageSize];
    ErrorRef m_ErrState;
  };
  bool m_HasError = false;

#ifndef NDEBUG
  mutable bool m_Unchecked = true;
  void markChecked() const { m_Unchecked = false; }
#else
  void markChecked() const {}
#endif

  T* valuePtr() { return reinterpret_cast<T*>(m_ValueBytes); }
  const T* valuePtr() const { return reinterpret_cast<const T*>(m_ValueBytes); }

public:
  Result() { new (m_ValueBytes) T(); }

  Result(T V) { new (m_ValueBytes) T(std::move(V)); }

  Result(ErrorRef E) : m_ErrState(E), m_HasError(!E.isOk()) {
    if (m_HasError)
      RetainErrorRef(m_ErrState);
  }

  Result(const Result& Other) : m_HasError(Other.m_HasError) {
    if (m_HasError) {
      m_ErrState = Other.m_ErrState;
      RetainErrorRef(m_ErrState);
    } else {
      new (m_ValueBytes) T(*Other.valuePtr());
    }
#ifndef NDEBUG
    m_Unchecked = Other.m_Unchecked;
    Other.m_Unchecked = false;
#endif
  }

  Result(Result&& Other) noexcept : m_HasError(Other.m_HasError) {
    if (m_HasError) {
      m_ErrState = Other.m_ErrState;
      Other.m_ErrState = ErrorRef{};
      Other.m_HasError = false;
    } else {
      new (m_ValueBytes) T(std::move(*Other.valuePtr()));
    }
#ifndef NDEBUG
    m_Unchecked = Other.m_Unchecked;
    Other.m_Unchecked = false;
#endif
  }

  ~Result() {
#ifndef NDEBUG
    if (m_Unchecked && m_HasError)
      ResultAbort_UncheckedOnDtor(m_ErrState);
#endif
    if (m_HasError)
      ReleaseErrorRef(m_ErrState);
    else
      valuePtr()->~T();
  }

  Result& operator=(const Result&) = delete;
  Result& operator=(Result&&) = delete;

  void ignore() const { markChecked(); }

  [[nodiscard]] bool ok() const {
    markChecked();
    return !m_HasError;
  }
  explicit operator bool() const { return ok(); }

  [[nodiscard]] Status status() const {
    markChecked();
    return m_HasError ? m_ErrState.status() : Status::Ok;
  }

  [[nodiscard]] ErrorRef error() const {
    markChecked();
    return m_HasError ? m_ErrState : ErrorRef{};
  }

  /// Share ownership of the carried ErrorRef with a CapturedError so
  /// the error outlives this Result. One refcount bump on slice
  /// errors; no-op for inline ones.
  [[nodiscard]] CapturedError share() const {
    markChecked();
    return m_HasError ? CapturedError(m_ErrState) : CapturedError();
  }

  T value() const {
    markChecked();
    if (m_HasError)
      ResultAbort_ValueOnError(m_ErrState);
    return *valuePtr();
  }

  T value_or(T fallback) const {
    markChecked();
    return m_HasError ? std::move(fallback) : *valuePtr();
  }
};

// Void specialization.

template <> class [[nodiscard]] Result<void> {
  ErrorRef m_ErrState;
#ifndef NDEBUG
  mutable bool m_Unchecked = true;
  void markChecked() const { m_Unchecked = false; }
#else
  void markChecked() const {}
#endif

public:
  Result() = default;
  Result(ErrorRef E) : m_ErrState(E) {
    if (!E.isOk())
      RetainErrorRef(m_ErrState);
  }

  Result(const Result& O) : m_ErrState(O.m_ErrState) {
    if (!m_ErrState.isOk())
      RetainErrorRef(m_ErrState);
#ifndef NDEBUG
    m_Unchecked = O.m_Unchecked;
    O.m_Unchecked = false;
#endif
  }

  Result(Result&& O) noexcept : m_ErrState(O.m_ErrState) {
#ifndef NDEBUG
    m_Unchecked = O.m_Unchecked;
    O.m_Unchecked = false;
#endif
    O.m_ErrState = ErrorRef{};
  }

  ~Result() {
#ifndef NDEBUG
    if (m_Unchecked && !m_ErrState.isOk())
      ResultAbort_UncheckedOnDtor(m_ErrState);
#endif
    if (!m_ErrState.isOk())
      ReleaseErrorRef(m_ErrState);
  }

  Result& operator=(const Result&) = delete;
  Result& operator=(Result&&) = delete;

  void ignore() const { markChecked(); }

  [[nodiscard]] bool ok() const {
    markChecked();
    return m_ErrState.isOk();
  }
  explicit operator bool() const { return ok(); }

  [[nodiscard]] Status status() const {
    markChecked();
    return m_ErrState.status();
  }

  [[nodiscard]] ErrorRef error() const {
    markChecked();
    return m_ErrState;
  }

  [[nodiscard]] CapturedError share() const {
    markChecked();
    return CapturedError(m_ErrState);
  }
};

} // namespace Cpp

#endif // __cplusplus

#endif // CPPINTEROP_ERROR_H
