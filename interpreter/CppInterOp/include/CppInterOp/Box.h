//===--- Box.h - Typed result carrier across the AOT/JIT boundary -*- C++ -*-=//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cpp::Box is the typed result vocabulary that crosses the binding-AOT /
// clang-repl-JIT boundary in both directions: catalog thunks Create<T> a
// Box to hand to the runtime, Evaluate returns one to hand back. Storage
// is hybrid -- fundamentals (K_Bool ... K_LongDouble) live inline in the
// union; K_PtrOrObj keeps an owned pointer plus a type-erased
// {retain, release} pair so the public header sees only `void*` and
// function pointers, never the concrete payload type. Copy is shallow +
// refcounted on K_PtrOrObj (matching clang::Value's model); the bridge
// composes only when both layers share that semantic.
//
// The AOT path `Box::Create<T>(x).visit(v)` is designed to fold to a
// direct `v(x)` -- the always_inline annotations below preserve this
// against compiler inliner heuristics (validated separately on clang 17
// and gcc 12 -O3; this commit does not pin the property in-tree).
//
// CPP_BOX_BUILTIN_TYPES is the single source of truth: enum positions,
// storage union slots, KindOf<T> specialisations, and visit's switch are
// all generated from this one X-macro -- one edit per new fundamental.
// Mirrors clang's REPL_BUILTIN_TYPES so the Evaluate bridge in
// CppInterOp.cpp is a Kind-to-Kind switch with no translation table.
//
//===----------------------------------------------------------------------===//

#ifndef CPPINTEROP_BOX_H
#define CPPINTEROP_BOX_H

// Box.h is meant to be cheap to parse: the JIT pulls it whenever a
// catalog thunk or evaluated snippet references Cpp::Box, so every
// pulled-in standard header is paid at interpretation time. We avoid
// CppInterOpTypes.h (which drags in <vector> / <string> / <set>) and
// keep <cassert> behind NDEBUG. Under NDEBUG, this file pulls in zero
// standard or project headers.

#ifndef NDEBUG
#include <cassert>
#endif

// Cross-platform always_inline. Duplicated from CppInterOpTypes.h's
// CPPINTEROP_ALWAYS_INLINE under a guard so consumers don't have to
// pull that (heavier) header just to use Box.
#ifndef CPPINTEROP_ALWAYS_INLINE
#if defined(_MSC_VER)
#define CPPINTEROP_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define CPPINTEROP_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define CPPINTEROP_ALWAYS_INLINE inline
#endif
#endif

// Cross-platform unreachable hint used in visit's default arms.
#ifndef CPPINTEROP_UNREACHABLE
#if defined(_MSC_VER)
#define CPPINTEROP_UNREACHABLE() __assume(0)
#elif defined(__GNUC__) || defined(__clang__)
#define CPPINTEROP_UNREACHABLE() __builtin_unreachable()
#else
#define CPPINTEROP_UNREACHABLE() ((void)0)
#endif
#endif

namespace Cpp {

// FIXME: clang's REPL_BUILTIN_TYPES lists `unsigned char` twice (Char_U +
// UChar) to disambiguate the platform-default signedness of `char` from an
// explicit `signed char` / `unsigned char`. The proper KindOf<char>()
// resolution uses std::is_signed_v<char> to pick K_Char_S vs K_Char_U;
// the duplicate Char_U entry is reached only via the runtime bridge
// (Evaluate). We keep this table single-valued per C++ type for unambiguous
// per-T specializations; Evaluate adds the K_Char_U case explicitly.
#define CPP_BOX_BUILTIN_TYPES                                                  \
  X(bool, Bool)                                                                \
  X(char, Char_S)                                                              \
  X(signed char, SChar)                                                        \
  X(unsigned char, UChar)                                                      \
  X(short, Short)                                                              \
  X(unsigned short, UShort)                                                    \
  X(int, Int)                                                                  \
  X(unsigned int, UInt)                                                        \
  X(long, Long)                                                                \
  X(unsigned long, ULong)                                                      \
  X(long long, LongLong)                                                       \
  X(unsigned long long, ULongLong)                                             \
  X(float, Float)                                                              \
  X(double, Double)                                                            \
  X(long double, LongDouble)

class Box {
public:
  // `int` (not `unsigned char`) to work around compiler-research/cppyy#223.
  enum Kind : int {
#define X(type, name) K_##name,
    CPP_BOX_BUILTIN_TYPES
#undef X
        K_Char_U, // alias of UChar storage; reached only by runtime bridge
    K_Void,
    K_PtrOrObj,
    K_Unspecified,
  };

  /// Operations vtable for a K_PtrOrObj payload. Defined once per concrete
  /// payload type in the TU that knows the type (see Evaluate's
  /// kCompatValueOps in CppInterOp.cpp). The Box stores a pointer to this
  /// const-static struct; the storage slot is 2 pointers (16 bytes).
  ///
  /// retain/release implement intrusive ref-counting: AdoptObject installs
  /// the payload with refcount 1; copy ctor / copy assign call retain;
  /// destructor calls release; the producer's release fires the payload's
  /// destructor when the last ref drops.
  struct ObjectOps {
    /// Increment the payload's refcount. Must be noexcept; called from the
    /// (noexcept) copy ctor / copy assign.
    void (*retain)(void*) noexcept;
    /// Decrement the payload's refcount; on the last drop, run the payload's
    /// destructor and free storage. Must be noexcept; called from ~Box.
    void (*release)(void*) noexcept;
  };

private:
  union Storage {
#define X(type, name) type m_##name;
    CPP_BOX_BUILTIN_TYPES
#undef X
    struct {
      void* m_Ptr;
      const ObjectOps* m_Ops;
    } m_Object;
  };

  Kind m_kind = K_Unspecified;
  void* m_Type = nullptr;
  Storage m_storage = {};

  template <class T> static constexpr Kind KindOf() noexcept;
  template <class T> static T& slot(Storage& s) noexcept;
  template <class T> static const T& slot(const Storage& s) noexcept;

public:
  // -- rule of five: refcount-shared on K_PtrOrObj, bitwise on fundamentals --
  Box() = default;
  Box(const Box& o) noexcept
      : m_kind(o.m_kind), m_Type(o.m_Type), m_storage(o.m_storage) {
    // Kind check first so fundamentals constant-fold the branch away
    // (AOT-fold preserved). K_PtrOrObj bumps the payload refcount.
    if (m_kind == K_PtrOrObj && m_storage.m_Object.m_Ops)
      m_storage.m_Object.m_Ops->retain(m_storage.m_Object.m_Ptr);
  }
  Box& operator=(const Box& o) noexcept {
    if (this != &o) {
      this->~Box();
      m_kind = o.m_kind;
      m_Type = o.m_Type;
      m_storage = o.m_storage;
      if (m_kind == K_PtrOrObj && m_storage.m_Object.m_Ops)
        m_storage.m_Object.m_Ops->retain(m_storage.m_Object.m_Ptr);
    }
    return *this;
  }
  Box(Box&& o) noexcept
      : m_kind(o.m_kind), m_Type(o.m_Type), m_storage(o.m_storage) {
    // moved-from: dtor sees K_Unspecified → no release of the stolen ref.
    o.m_kind = K_Unspecified;
  }
  Box& operator=(Box&& o) noexcept {
    if (this != &o) {
      this->~Box();
      m_kind = o.m_kind;
      m_Type = o.m_Type;
      m_storage = o.m_storage;
      o.m_kind = K_Unspecified;
    }
    return *this;
  }
  // always_inline so a static Kind constant-propagates into the dtor
  // body and the K_PtrOrObj branch dead-strips on the AOT path.
  CPPINTEROP_ALWAYS_INLINE ~Box() noexcept {
    if (m_kind == K_PtrOrObj && m_storage.m_Object.m_Ops)
      m_storage.m_Object.m_Ops->release(m_storage.m_Object.m_Ptr);
  }

  Kind getKind() const noexcept { return m_kind; }
  void* getType() const noexcept { return m_Type; }

  /// AOT-typed extraction. \c T must match the runtime \c Kind exactly
  /// (UB to read an inactive union member otherwise -- e.g. calling
  /// \c unbox<int>() on a \c K_Long Box). For unknown-kind extraction
  /// use \c visit() or check \c getKind() against \c KindOf<T>() first.
  /// The assert is a no-op under \c NDEBUG so the AOT-fold path stays
  /// zero-overhead.
  template <class T> T unbox() const noexcept {
#ifndef NDEBUG
    assert(m_kind == KindOf<T>() &&
           "Cpp::Box::unbox<T>(): T does not match the runtime Kind");
#endif
    return slot<T>(m_storage);
  }

  /// AOT-typed construction for fundamentals. Sets Kind = KindOf<T>(),
  /// stores x. Optional QualType preserves typedef sugar (int8_t vs
  /// signed char) the Kind enum collapses.
  // always_inline so `Create<T>(x).visit(toPy)` folds bit-identical to
  // a direct `toPy(x)` on gcc-12 (default inliner gives up otherwise).
  template <class T>
  CPPINTEROP_ALWAYS_INLINE static Box Create(T x,
                                             void* type = nullptr) noexcept {
    Box v;
    v.m_kind = KindOf<T>();
    v.m_Type = type;
    slot<T>(v.m_storage) = x;
    return v;
  }

  /// Object-payload construction. `obj` enters with refcount 1 (the
  /// producer's invariant); ~Box calls `ops->release(obj)` which on the
  /// last drop runs the payload's destructor. The ops table is const-static,
  /// defined in the TU that knows the concrete payload type.
  static Box AdoptObject(void* obj, const ObjectOps* ops, void* type) noexcept {
    Box v;
    v.m_kind = K_PtrOrObj;
    v.m_Type = type;
    v.m_storage.m_Object.m_Ptr = obj;
    v.m_storage.m_Object.m_Ops = ops;
    return v;
  }

  /// Object payload accessor; only valid for K_PtrOrObj. Returns the
  /// raw pointer set at AdoptObject time. Layout is producer-defined --
  /// the producer that installed the ObjectOps knows how to extract the
  /// underlying object.
  void* getObjectPtr() const noexcept {
    return m_kind == K_PtrOrObj ? m_storage.m_Object.m_Ptr : nullptr;
  }

  /// Runtime convert across Kinds: dispatches on the actual Kind and
  /// returns the value reinterpreted (via \c static_cast) as \c T.
  /// Mirrors clang::Value::convertTo<T>. Only valid for fundamental
  /// Kinds (K_Bool ... K_LongDouble); K_PtrOrObj / K_Void /
  /// K_Unspecified are caller-must-check-Kind cases.
  template <class T> T convertTo() const noexcept {
    return visit([](auto x) -> T { return static_cast<T>(x); });
  }

  /// Runtime-typed dispatch via visitor. Switch over Kind, call visitor
  /// with the typed T extracted from storage. K_PtrOrObj / K_Void / K_*
  /// non-fundamental kinds are not dispatched -- caller checks Kind first.
  // always_inline so the switch reduces to the single case the AOT Kind
  // resolves to; the other arms then dead-strip.
  template <class V>
  CPPINTEROP_ALWAYS_INLINE auto visit(V&& vis) const -> decltype(vis(int{})) {
    switch (m_kind) {
#define X(type, name)                                                          \
  case K_##name:                                                               \
    return vis(unbox<type>());
      CPP_BOX_BUILTIN_TYPES
#undef X
    case K_Char_U:
    case K_Void:
    case K_PtrOrObj:
    case K_Unspecified:
      CPPINTEROP_UNREACHABLE();
    }
    CPPINTEROP_UNREACHABLE();
  }
};

// --- Per-T specializations, generated via the same X-macro -------------------

#define X(type, name)                                                          \
  template <> constexpr Box::Kind Box::KindOf<type>() noexcept {               \
    return K_##name;                                                           \
  }                                                                            \
  template <> inline type& Box::slot<type>(Storage & s) noexcept {             \
    return s.m_##name;                                                         \
  }                                                                            \
  template <> inline const type& Box::slot<type>(const Storage& s) noexcept {  \
    return s.m_##name;                                                         \
  }
CPP_BOX_BUILTIN_TYPES
#undef X

} // namespace Cpp

#endif // CPPINTEROP_BOX_H
