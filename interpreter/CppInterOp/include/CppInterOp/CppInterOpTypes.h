//===--- CppInterOpTypes.h - Types for CppInterOp API -----------*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Opaque types, enums, and helper classes used by the CppInterOp API.
// This header is self-contained and does not declare any API functions,
// making it safe to include from both CppInterOp.h and Dispatch.h.
//
//===----------------------------------------------------------------------===//

#ifndef CPPINTEROP_CPPINTEROPTYPES_H
#define CPPINTEROP_CPPINTEROPTYPES_H

// The cross-platform CPPINTEROP_API macro definition
#if defined _WIN32 || defined __CYGWIN__
#define CPPINTEROP_API __declspec(dllexport)
#else
#ifdef __GNUC__
#define CPPINTEROP_API __attribute__((__visibility__("default")))
#else
#define CPPINTEROP_API
#endif
#endif

// Cross-platform deprecation attribute. Older Clang versions (Cling)
// mis-parse C++11 `[[deprecated]]` near the return type, so we use
// the vendor-specific spelling on each compiler.
#if defined(_MSC_VER)
#define CPPINTEROP_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined(__GNUC__) || defined(__clang__)
#define CPPINTEROP_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define CPPINTEROP_DEPRECATED(msg)
#endif

// C-compatible headers — usable from both C and C++.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/// Opaque handle types for the CppInterOp C and C++ API.
///
/// Each handle wraps a single void* and is a distinct type so the
/// compiler rejects cross-kind misuse (e.g. passing a TypeRef where a
/// DeclRef is expected). Layout is identical across all handles — a
/// single pointer — so they are trivially copyable and ABI-stable.
///
/// The C and C++ views are declared independently:
///   * C sees `struct CppDeclRef { void* data; }` etc. at global scope,
///     prefixed so they don't collide with generic names from other
///     C libraries.
///   * C++ sees `Cpp::DeclRef` etc. in namespace `Cpp`, no prefix —
///     the namespace already disambiguates.
/// Both have identical layout, so a C-linkage function exchanging these
/// structs across the boundary is byte-identical regardless of which
/// view names it. The TableGen-emitted C-API .inc adds C++ typedef
/// aliases (`using CppDeclRef = Cpp::DeclRef;`) so the generated C
/// wrappers compile under C++ using the prefixed C-side names.

#ifndef __cplusplus

typedef struct CppDeclRef {
  void* data;
} CppDeclRef;
typedef struct CppTypeRef {
  void* data;
} CppTypeRef;
typedef struct CppFuncRef {
  void* data;
} CppFuncRef;
typedef struct CppObjectRef {
  void* data;
} CppObjectRef;
typedef struct CppInterpRef {
  void* data;
} CppInterpRef;
typedef struct CppConstDeclRef {
  const void* data;
} CppConstDeclRef;
typedef struct CppConstTypeRef {
  const void* data;
} CppConstTypeRef;
typedef struct CppConstFuncRef {
  const void* data;
} CppConstFuncRef;

#else // __cplusplus

namespace Cpp {

struct DeclRef {
  void* data;
  DeclRef() : data(nullptr) {}
  DeclRef(void* P) : data(P) {}
  DeclRef(decltype(nullptr)) : data(nullptr) {}
  explicit operator bool() const { return data != nullptr; }
  friend bool operator==(DeclRef a, DeclRef b) { return a.data == b.data; }
  friend bool operator!=(DeclRef a, DeclRef b) { return !(a == b); }
};

struct TypeRef {
  void* data;
  TypeRef() : data(nullptr) {}
  TypeRef(void* P) : data(P) {}
  TypeRef(decltype(nullptr)) : data(nullptr) {}
  explicit operator bool() const { return data != nullptr; }
  friend bool operator==(TypeRef a, TypeRef b) { return a.data == b.data; }
  friend bool operator!=(TypeRef a, TypeRef b) { return !(a == b); }
};

struct FuncRef {
  void* data;
  FuncRef() : data(nullptr) {}
  FuncRef(void* P) : data(P) {}
  FuncRef(decltype(nullptr)) : data(nullptr) {}
  explicit operator bool() const { return data != nullptr; }
  friend bool operator==(FuncRef a, FuncRef b) { return a.data == b.data; }
  friend bool operator!=(FuncRef a, FuncRef b) { return !(a == b); }
};

struct ObjectRef {
  void* data;
  ObjectRef() : data(nullptr) {}
  ObjectRef(void* P) : data(P) {}
  ObjectRef(decltype(nullptr)) : data(nullptr) {}
  explicit operator bool() const { return data != nullptr; }
  friend bool operator==(ObjectRef a, ObjectRef b) { return a.data == b.data; }
  friend bool operator!=(ObjectRef a, ObjectRef b) { return !(a == b); }
};

struct InterpRef {
  void* data;
  InterpRef() : data(nullptr) {}
  InterpRef(void* P) : data(P) {}
  InterpRef(decltype(nullptr)) : data(nullptr) {}
  explicit operator bool() const { return data != nullptr; }
  friend bool operator==(InterpRef a, InterpRef b) { return a.data == b.data; }
  friend bool operator!=(InterpRef a, InterpRef b) { return !(a == b); }
};

/// Const handle variants — read-only access to the underlying AST node.
/// Implicit widening from mutable to const is allowed; narrowing is not.

struct ConstDeclRef {
  const void* data;
  ConstDeclRef() : data(nullptr) {}
  ConstDeclRef(const void* P) : data(P) {}
  ConstDeclRef(DeclRef d) : data(d.data) {}
  ConstDeclRef(decltype(nullptr)) : data(nullptr) {}
  explicit operator bool() const { return data != nullptr; }
  friend bool operator==(ConstDeclRef a, ConstDeclRef b) {
    return a.data == b.data;
  }
  friend bool operator!=(ConstDeclRef a, ConstDeclRef b) { return !(a == b); }
};

struct ConstTypeRef {
  const void* data;
  ConstTypeRef() : data(nullptr) {}
  ConstTypeRef(const void* P) : data(P) {}
  ConstTypeRef(TypeRef d) : data(d.data) {}
  ConstTypeRef(decltype(nullptr)) : data(nullptr) {}
  explicit operator bool() const { return data != nullptr; }
  friend bool operator==(ConstTypeRef a, ConstTypeRef b) {
    return a.data == b.data;
  }
  friend bool operator!=(ConstTypeRef a, ConstTypeRef b) { return !(a == b); }
};

struct ConstFuncRef {
  const void* data;
  ConstFuncRef() : data(nullptr) {}
  ConstFuncRef(const void* P) : data(P) {}
  ConstFuncRef(FuncRef d) : data(d.data) {}
  ConstFuncRef(decltype(nullptr)) : data(nullptr) {}
  explicit operator bool() const { return data != nullptr; }
  friend bool operator==(ConstFuncRef a, ConstFuncRef b) {
    return a.data == b.data;
  }
  friend bool operator!=(ConstFuncRef a, ConstFuncRef b) { return !(a == b); }
};

} // namespace Cpp

#endif // __cplusplus

#ifdef __cplusplus
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

template <> struct std::hash<Cpp::DeclRef> {
  std::size_t operator()(const Cpp::DeclRef& obj) const {
    return std::hash<void*>{}(obj.data);
  }
};
template <> struct std::hash<Cpp::TypeRef> {
  std::size_t operator()(const Cpp::TypeRef& obj) const {
    return std::hash<void*>{}(obj.data);
  }
};
template <> struct std::hash<Cpp::FuncRef> {
  std::size_t operator()(const Cpp::FuncRef& obj) const {
    return std::hash<void*>{}(obj.data);
  }
};
template <> struct std::hash<Cpp::ObjectRef> {
  std::size_t operator()(const Cpp::ObjectRef& obj) const {
    return std::hash<void*>{}(obj.data);
  }
};

namespace Cpp {
class JitCall;
}
namespace CppInternal {
namespace DispatchRaw {
// Trace-hook slot forward decls; the X-macro expansion of
// CppInterOpAPI.inc re-declares these with identical types. They're
// here so JitCall::Invoke's inline body below can reference them.
extern CPPINTEROP_API void (*CppInterOpTraceJitCallInvokeImpl)(
    const Cpp::JitCall* JC, void* result, void** args, std::size_t nargs,
    void* self);
extern CPPINTEROP_API void (*CppInterOpTraceJitCallInvokeDestructorImpl)(
    const Cpp::JitCall* JC, void* object, unsigned long nary, int withFree);
extern CPPINTEROP_API void (*CppInterOpTraceJitCallInvokeReturnImpl)(
    const Cpp::JitCall* JC, void* result);
} // namespace DispatchRaw
} // namespace CppInternal

#endif // __cplusplus

/// C-compatible array of opaque pointers, returned by generated C API
/// wrappers for functions that produce collections. The caller must free
/// the array by calling cppinterop_DisposeArray().
typedef struct CppInterOpArray {
  void** data;
  size_t size;
} CppInterOpArray;

/// C-compatible array of strings, returned by generated C API wrappers
/// for functions that produce string collections. Each string is
/// individually allocated with strdup(). The caller must free the array
/// by calling cppinterop_DisposeStringArray().
typedef struct CppInterOpStringArray {
  char** data;
  size_t size;
} CppInterOpStringArray;

/// Holds information for instantiating a template.
/// Standard-layout, C-compatible.
typedef struct TemplateArgInfo {
  void* m_Type;
  const char* m_IntegralValue;
#ifdef __cplusplus
  TemplateArgInfo(void* type, const char* integral_value = nullptr)
      : m_Type(type), m_IntegralValue(integral_value) {}
#endif
} TemplateArgInfo;

#ifdef __cplusplus
namespace Cpp {

// Pull file-scope C-compatible structs into the namespace.
// The handle types (DeclRef etc.) are already defined in this namespace
// above; only the genuinely-C-visible structs need pulling in.
using ::CppInterOpArray;
using ::CppInterOpStringArray;
using ::TemplateArgInfo;

static_assert(sizeof(DeclRef) == sizeof(ConstDeclRef),
              "Const/mutable handle ABI mismatch");
static_assert(sizeof(TypeRef) == sizeof(ConstTypeRef),
              "Const/mutable handle ABI mismatch");
static_assert(sizeof(FuncRef) == sizeof(ConstFuncRef),
              "Const/mutable handle ABI mismatch");

enum Operator : unsigned char {
  OP_None,
  OP_New,
  OP_Delete,
  OP_Array_New,
  OP_Array_Delete,
  OP_Plus,
  OP_Minus,
  OP_Star,
  OP_Slash,
  OP_Percent,
  OP_Caret,
  OP_Amp,
  OP_Pipe,
  OP_Tilde,
  OP_Exclaim,
  OP_Equal,
  OP_Less,
  OP_Greater,
  OP_PlusEqual,
  OP_MinusEqual,
  OP_StarEqual,
  OP_SlashEqual,
  OP_PercentEqual,
  OP_CaretEqual,
  OP_AmpEqual,
  OP_PipeEqual,
  OP_LessLess,
  OP_GreaterGreater,
  OP_LessLessEqual,
  OP_GreaterGreaterEqual,
  OP_EqualEqual,
  OP_ExclaimEqual,
  OP_LessEqual,
  OP_GreaterEqual,
  OP_Spaceship,
  OP_AmpAmp,
  OP_PipePipe,
  OP_PlusPlus,
  OP_MinusMinus,
  OP_Comma,
  OP_ArrowStar,
  OP_Arrow,
  OP_Call,
  OP_Subscript,
  OP_Conditional,
  OP_Coawait,
};

enum OperatorArity : unsigned char { kUnary = 1, kBinary, kBoth };
enum Signedness : unsigned char { kSigned = 1, kUnsigned };

/// Enum modelling CVR qualifiers.
enum QualKind : unsigned char {
  Const = 1 << 0,
  Volatile = 1 << 1,
  Restrict = 1 << 2,
  All = Const | Volatile | Restrict
};

/// Enum modelling programming languages.
enum class InterpreterLanguage : unsigned char {
  Unknown,
  Asm,
  CIR,
  LLVM_IR,
  C,
  CPlusPlus,
  ObjC,
  ObjCPlusPlus,
  OpenCL,
  OpenCLCXX,
  CUDA,
  HIP,
  HLSL
};

/// Enum modelling language standards.
enum class InterpreterLanguageStandard : unsigned char {
  c89,
  c94,
  gnu89,
  c99,
  gnu99,
  c11,
  gnu11,
  c17,
  gnu17,
  c23,
  gnu23,
  c2y,
  gnu2y,
  cxx98,
  gnucxx98,
  cxx11,
  gnucxx11,
  cxx14,
  gnucxx14,
  cxx17,
  gnucxx17,
  cxx20,
  gnucxx20,
  cxx23,
  gnucxx23,
  cxx26,
  gnucxx26,
  opencl10,
  opencl11,
  opencl12,
  opencl20,
  opencl30,
  openclcpp10,
  openclcpp2021,
  hlsl,
  hlsl2015,
  hlsl2016,
  hlsl2017,
  hlsl2018,
  hlsl2021,
  hlsl202x,
  hlsl202y,
  lang_unspecified
};

enum class AllocType : unsigned char {
  None,
  New,
  NewArr,
  Malloc,
  Unknown,
  CustomAlloc
};

inline QualKind operator|(QualKind a, QualKind b) {
  return static_cast<QualKind>(static_cast<unsigned char>(a) |
                               static_cast<unsigned char>(b));
}

enum class ValueKind : std::uint8_t {
  None,
  LValue,
  RValue,
};

/// A class modeling function calls for functions produced by the interpreter
/// in compiled code. It provides an information if we are calling a standard
/// function, constructor or destructor.
class JitCall {
public:
  friend CPPINTEROP_API JitCall MakeFunctionCallable(InterpRef I,
                                                     ConstFuncRef func);
  enum Kind : char {
    kUnknown = 0,
    kGenericCall,
    kConstructorCall,
    kDestructorCall,
  };
  struct ArgList {
    void** m_Args = nullptr;
    size_t m_ArgSize = 0;
    // Clang struggles with =default...
    ArgList() {}
    ArgList(void** Args, size_t ArgSize) : m_Args(Args), m_ArgSize(ArgSize) {}
  };
  // FIXME: Figure out how to unify the wrapper signatures.
  // FIXME: Hide these implementation details by moving wrapper generation in
  // this class.
  // (self, nargs, args, result, nary)
  using GenericCall = void (*)(void*, size_t, void**, void*);
  // (result, nary, nargs, args, is_arena)
  using ConstructorCall = void (*)(void*, size_t, size_t, void**, void*);
  // (self, nary, withFree)
  using DestructorCall = void (*)(void*, size_t, int);

private:
  union {
    GenericCall m_GenericCall;
    ConstructorCall m_ConstructorCall;
    DestructorCall m_DestructorCall;
  };
  Kind m_Kind;
  ConstFuncRef m_FD;
  JitCall() : m_GenericCall(nullptr), m_Kind(kUnknown), m_FD(nullptr) {}
  JitCall(Kind K, GenericCall C, ConstFuncRef FD)
      : m_GenericCall(C), m_Kind(K), m_FD(FD) {}
  JitCall(Kind K, ConstructorCall C, ConstFuncRef Ctor)
      : m_ConstructorCall(C), m_Kind(K), m_FD(Ctor) {}
  JitCall(Kind K, DestructorCall C, ConstFuncRef Dtor)
      : m_DestructorCall(C), m_Kind(K), m_FD(Dtor) {}

  // Trace-hook impls need private m_FD for the function-name lookup.
  // CPPINTEROP_API matches the X-macro-generated decl in CppInterOpDecl.inc;
  // MSVC treats a friend decl without the dllimport/dllexport attribute as
  // a different-linkage redeclaration.
  friend CPPINTEROP_API void
  CppInterOpTraceJitCallInvokeImpl(const JitCall*, void* result, void** args,
                                   std::size_t nargs, void* self);
  friend CPPINTEROP_API void
  CppInterOpTraceJitCallInvokeDestructorImpl(const JitCall*, void* object,
                                             unsigned long nary, int withFree);
  friend CPPINTEROP_API void
  CppInterOpTraceJitCallInvokeReturnImpl(const JitCall*, void* result);

  /// Checks if the passed arguments are valid for the given function.
  CPPINTEROP_API bool AreArgumentsValid(void* result, ArgList args, void* self,
                                        size_t nary) const;

public:
  [[nodiscard]] Kind getKind() const { return m_Kind; }
  bool isValid() const { return getKind() != kUnknown; }
  bool isInvalid() const { return !isValid(); }
  explicit operator bool() const { return isValid(); }

  // Specialized for calling void functions.
  void Invoke(ArgList args = {}, void* self = nullptr) const {
    Invoke(/*result=*/nullptr, args, self);
  }

  /// Makes a call to a generic function or method.
  ///\param[in] result - the location where the return result will be placed.
  ///\param[in] args - a pointer to a argument list and argument size.
  ///\param[in] self - the 'this pointer' of the object.
  // FIXME: Adjust the arguments and their types: args_size can be unsigned;
  // self can go in the end and be nullptr by default; result can be a nullptr
  // by default. These changes should be synchronized with the wrapper if we
  // decide to directly.
  void Invoke(void* result, ArgList args = {}, void* self = nullptr) const {
    // NOLINTBEGIN(*-type-union-access)
    // Its possible the JitCall object deals with structor decls but went
    // through Invoke

    switch (m_Kind) {
    case kUnknown:
      assert(0 && "Attempted to call an invalid function declaration");
      break;

    case kGenericCall:
      // We pass 1UL to nary which is only relevant for structors
      assert(AreArgumentsValid(result, args, self, 1UL) && "Invalid args!");
      if (auto fn =
              ::CppInternal::DispatchRaw::CppInterOpTraceJitCallInvokeImpl)
        fn(this, result, args.m_Args, args.m_ArgSize, self);
      m_GenericCall(self, args.m_ArgSize, args.m_Args, result);
      if (auto fn = ::CppInternal::DispatchRaw::
              CppInterOpTraceJitCallInvokeReturnImpl)
        fn(this, result);
      break;

    case kConstructorCall:
      // Forward if we intended to call a constructor (nary cannot be inferred,
      // so we stick to constructing a single object)
      InvokeConstructor(result, /*nary=*/1UL, args, self);
      break;
    case kDestructorCall:
      // Forward if we intended to call a dtor with only 1 parameter.
      assert(!args.m_Args && "Destructor called with arguments");
      InvokeDestructor(result, /*nary=*/0UL, /*withFree=*/true);
      break;
    }
    // NOLINTEND(*-type-union-access)
  }
  /// Makes a call to a destructor.
  ///\param[in] object - the pointer of the object whose destructor we call.
  ///\param[in] nary - the count of the objects we destruct if we deal with an
  ///           array of objects.
  ///\param[in] withFree - true if we should call operator delete or false if
  ///           we should call only the destructor.
  // FIXME: Change the type of withFree from int to bool in the wrapper code.
  void InvokeDestructor(void* object, unsigned long nary = 0,
                        int withFree = true) const {
    assert(m_Kind == kDestructorCall && "Wrong overload!");
    if (auto fn = ::CppInternal::DispatchRaw::
            CppInterOpTraceJitCallInvokeDestructorImpl)
      fn(this, object, nary, withFree);
    m_DestructorCall(object, nary, withFree);
  }

  /// Makes a call to a constructor.
  ///\param[in] result - the memory address at which we construct the object
  ///           (placement new).
  ///\param[in] nary - Use array new if we have to construct an array of
  ///           objects (nary > 1).
  ///\param[in] args - a pointer to a argument list and argument size.
  ///\param[in] is_arena - a pointer that indicates if placement new is to be
  /// used
  // FIXME: Change the type of withFree from int to bool in the wrapper code.
  void InvokeConstructor(void* result, unsigned long nary = 1,
                         ArgList args = {}, void* is_arena = nullptr) const {
    assert(m_Kind == kConstructorCall && "Wrong overload!");
    assert(AreArgumentsValid(result, args, /*self=*/nullptr, nary) &&
           "Invalid args!");
    if (auto fn = ::CppInternal::DispatchRaw::CppInterOpTraceJitCallInvokeImpl)
      fn(this, result, args.m_Args, args.m_ArgSize, nullptr);
    m_ConstructorCall(result, nary, args.m_ArgSize, args.m_Args, is_arena);
    if (auto fn =
            ::CppInternal::DispatchRaw::CppInterOpTraceJitCallInvokeReturnImpl)
      fn(this, result);
  }
};

/// Opaque handle returned by MakeVTableOverlay. Owns a writable copy
/// of an instance's vtable with selected slots replaced, and the
/// original vptr so the overlay can be undone. Itanium ABI, single
/// inheritance: the offset-to-top + type_info prefix and all
/// non-overlaid slots are copied verbatim from the original vtable.
struct VTableOverlay;

// Cleanup callback fired by VTableOverlay's destructor hook (see
// MakeVTableOverlay). Receives the original instance pointer (possibly
// dangling -- do not dereference) and the cleanup_data passed at install.
using VTableOverlayDtorHook = void (*)(void* inst, void* cleanup_data);

// FIXME: Rework GetDimensions to make this enum redundant.
namespace DimensionValue {
enum : long int {
  UNKNOWN_SIZE = -1,
};
} // namespace DimensionValue

/// @name Stream Redirection
///@{

enum CaptureStreamKind : char {
  kStdOut = 1, ///< stdout
  kStdErr,     ///< stderr
};

///@}

} // namespace Cpp

#endif // __cplusplus
#endif // CPPINTEROP_CPPINTEROPTYPES_H
