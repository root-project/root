// Tests for the opaque handle types in CppInterOpTypes.h:
// ABI invariants, type safety (const widening, cross-kind rejection),
// and wrap/unwrap round-trip. No interpreter required.

#include "../../lib/CppInterOp/Unwrap.h"
#include "CppInterOp/CppInterOp.h"
#include "gtest/gtest.h"

#include <functional>
#include <type_traits>
#include <unordered_set>

using namespace Cpp;

namespace {
// SFINAE traits used to assert that an expression does NOT compile.
template <typename A, typename B, typename = void>
struct is_eq_comparable : std::false_type {};
template <typename A, typename B>
struct is_eq_comparable<
    A, B, std::void_t<decltype(std::declval<A>() == std::declval<B>())>>
    : std::true_type {};

bool void_ptr_takes_nonnull(void* p) { return p != nullptr; }
} // namespace

// -- ABI: handles must be layout-compatible with void* ------------------

static_assert(sizeof(DeclRef) == sizeof(void*));
static_assert(alignof(DeclRef) == alignof(void*));
static_assert(std::is_standard_layout_v<DeclRef>);
static_assert(std::is_trivially_copyable_v<DeclRef>);

// Const and mutable variants must share ABI so a dispatch wrapper can
// reinterpret-cast between them without marshalling.
static_assert(sizeof(DeclRef) == sizeof(ConstDeclRef));
static_assert(sizeof(TypeRef) == sizeof(ConstTypeRef));
static_assert(sizeof(FuncRef) == sizeof(ConstFuncRef));

// -- Type safety: implicit conversions ----------------------------------

// Mutable → const widening is allowed.
static_assert(std::is_convertible_v<DeclRef, ConstDeclRef>);
static_assert(std::is_convertible_v<TypeRef, ConstTypeRef>);
static_assert(std::is_convertible_v<FuncRef, ConstFuncRef>);

// Const → mutable narrowing is rejected.
static_assert(!std::is_convertible_v<ConstDeclRef, DeclRef>);
static_assert(!std::is_assignable_v<DeclRef&, ConstDeclRef>);

// Cross-kind conversions are rejected (Decl ≠ Type ≠ Func).
static_assert(!std::is_convertible_v<TypeRef, DeclRef>);
static_assert(!std::is_convertible_v<DeclRef, TypeRef>);
static_assert(!std::is_convertible_v<FuncRef, DeclRef>);

// Cross-kind equality is rejected — the entire point of distinct types.
static_assert(is_eq_comparable<DeclRef, DeclRef>::value);
static_assert(!is_eq_comparable<DeclRef, TypeRef>::value);
static_assert(!is_eq_comparable<TypeRef, FuncRef>::value);

// unwrap preserves const: T* for mutable handles, const T* for const.
static_assert(std::is_same_v<decltype(unwrap<int>(DeclRef{})), int*>);
static_assert(
    std::is_same_v<decltype(unwrap<int>(ConstDeclRef{})), const int*>);

// -- TemplateArgInfo: C-ABI layout cannot drift -------------------------

static_assert(std::is_standard_layout_v<TemplateArgInfo>);
static_assert(sizeof(TemplateArgInfo) == 2 * sizeof(void*));
static_assert(offsetof(TemplateArgInfo, m_Type) == 0);
static_assert(offsetof(TemplateArgInfo, m_IntegralValue) == sizeof(void*));

// -- Runtime --------------------------------------------------------------

TEST(HandleTypes, NullSemantics) {
  EXPECT_FALSE(DeclRef{});
  EXPECT_FALSE(ConstDeclRef{});
  EXPECT_EQ(DeclRef{}, nullptr);
  EXPECT_TRUE(DeclRef(nullptr) == nullptr);
}

TEST(HandleTypes, WrapUnwrapRoundTrip) {
  int x = 0;
  DeclRef d = wrap<DeclRef>(static_cast<void*>(&x));
  EXPECT_EQ(unwrap<int>(d), &x);

  // Mutable → const widening is value-preserving.
  ConstDeclRef cd = d;
  EXPECT_EQ(unwrap<int>(cd), &x);
}

TEST(HandleTypes, Equality) {
  int x = 0, y = 0;
  DeclRef a(&x), b(&x), c(&y);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

// std::hash specializations let handles key unordered containers. The
// hash must agree with std::hash<void*> on the underlying pointer and
// distinguish distinct pointers.
TEST(HandleTypes, Hash) {
  int x = 0, y = 0;
  DeclRef a(&x), b(&x), c(&y);
  std::hash<DeclRef> h;
  EXPECT_EQ(h(a), h(b));
  EXPECT_EQ(h(a), std::hash<void*>{}(&x));
  EXPECT_NE(h(a), h(c));

  std::unordered_set<DeclRef> s{a, b, c};
  EXPECT_EQ(s.size(), 2u);
  EXPECT_TRUE(s.count(DeclRef(&x)));
  EXPECT_FALSE(s.count(DeclRef{}));
}

// The dispatch ABI cppyy uses reinterpret-casts a dlsym'd void*-taking
// function pointer to a handle-taking one. Verify that the struct{void*}
// calling convention is byte-identical to passing void*. A union swaps
// the function-pointer type without triggering -Wcast-function-type.
TEST(HandleTypes, AbiCompatibleWithVoidPtr) {
  union {
    bool (*void_ptr_fn)(void*);
    bool (*handle_fn)(DeclRef);
  } pun{&void_ptr_takes_nonnull};

  // The handle-taking pointer comes from dlsym at runtime, so the compiler
  // cannot see through it. If it is a compile-time constant, GCC 11 exploits
  // the type mismatch (UB) to constant-fold the call giving a wrong
  // reslt. Route the call through a volatile local so the compiler treats
  // the pointer as runtime-opaque
  volatile auto handle_fn = pun.handle_fn;
  int x = 0;
  EXPECT_TRUE(handle_fn(DeclRef(&x)));
  EXPECT_FALSE(handle_fn(DeclRef{}));
}
