# Author: Jonas Rembser CERN 08/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

"""The C++ runtime support library used by the generated code.

Everything in here exists for one reason: Python's operator semantics are not
C++'s.  Rather than relying on the transpiler's type inference to be complete
enough to pick the right C++ spelling, the semantics are pushed into small
templates that get it right for whatever type the compiler actually deduces.
Inference then only has to be good enough to choose *which* helper to call.

The header is declared to cling exactly once per process, lazily.
"""

_DECLARED = False

CPP_SUPPORT = r"""
#ifndef ROOT_PYDECLARE_SUPPORT
#define ROOT_PYDECLARE_SUPPORT

#include <ROOT/RVec.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ROOT {
namespace Internal {
namespace PyDeclare {

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

template <class T>
struct IsRVec : std::false_type {};
template <class T>
struct IsRVec<ROOT::VecOps::RVec<T>> : std::true_type {};

template <class T>
struct IsStdVector : std::false_type {};
template <class T, class A>
struct IsStdVector<std::vector<T, A>> : std::true_type {};

template <class T>
struct IsStdArray : std::false_type {};
template <class T, std::size_t N>
struct IsStdArray<std::array<T, N>> : std::true_type {};

template <class T>
struct IsVecLike : std::integral_constant<bool, IsRVec<T>::value || IsStdVector<T>::value || IsStdArray<T>::value> {};

/// The element type of a container, or the type itself for scalars.
template <class T, bool = IsVecLike<typename std::decay<T>::type>::value>
struct ElemTypeImpl {
   using type = typename std::decay<T>::type::value_type;
};
template <class T>
struct ElemTypeImpl<T, false> {
   using type = typename std::decay<T>::type;
};
template <class T>
using ElemType = typename ElemTypeImpl<T>::type;

/// Negativity test that does not warn for unsigned types.
template <class T>
constexpr bool IsNeg(T x)
{
   if constexpr (std::is_signed<T>::value) {
      return x < T(0);
   } else {
      (void)x;
      return false;
   }
}

/// The type an integer operation is carried out in.
///
/// Python integers do not overflow and have no unsigned flavour, so the usual
/// C++ arithmetic conversions are wrong for us in one specific place: a mix of
/// a signed and an unsigned operand converts the signed one, turning -7 into a
/// huge positive value before the operation ever runs.  Working in a signed
/// 64-bit type whenever either operand is signed reproduces Python for every
/// pair of operands that is narrower than 64 bits, which is all of them except
/// long long itself.
template <class A, class B>
using IntOpType =
   typename std::conditional<std::is_signed<A>::value || std::is_signed<B>::value,
                             typename std::conditional<(sizeof(A) < sizeof(long long) &&
                                                        sizeof(B) < sizeof(long long)),
                                                       long long, typename std::common_type<A, B>::type>::type,
                             typename std::common_type<A, B>::type>::type;

/// The type a Python arithmetic expression on A and B produces.
template <class A, class B>
using ArithResult =
   typename std::conditional<std::is_integral<A>::value && std::is_integral<B>::value, IntOpType<A, B>,
                             typename std::common_type<A, B>::type>::type;

/// '<' that is correct across a signed/unsigned mix, where C++'s own '<' is not.
template <class A, class B>
constexpr bool Less(A a, B b)
{
   if constexpr (std::is_integral<A>::value && std::is_integral<B>::value &&
                 std::is_signed<A>::value != std::is_signed<B>::value) {
      if constexpr (std::is_signed<A>::value) {
         if (a < A(0)) {
            return true;
         }
         return static_cast<typename std::make_unsigned<A>::type>(a) < b;
      } else {
         if (b < B(0)) {
            return false;
         }
         return a < static_cast<typename std::make_unsigned<B>::type>(b);
      }
   } else {
      return a < b;
   }
}

/// Python's unary minus.  Negating an unsigned value in C++ wraps around;
/// Python has no unsigned integers and simply produces a negative number.
template <class T>
auto NegScalar(T x)
{
   if constexpr (std::is_integral<T>::value && !std::is_signed<T>::value) {
      return -static_cast<IntOpType<T, long long>>(x);
   } else {
      return -x;
   }
}

template <class T>
auto Neg(const T &x)
{
   if constexpr (IsVecLike<typename std::decay<T>::type>::value) {
      using E = ElemType<T>;
      using R = typename std::decay<decltype(NegScalar(std::declval<E>()))>::type;
      ROOT::VecOps::RVec<R> out(x.size());
      for (std::size_t i = 0; i < x.size(); ++i) {
         out[i] = NegScalar(static_cast<E>(x[i]));
      }
      return out;
   } else {
      return NegScalar(x);
   }
}

/// Python's 'and' and 'or' evaluate to one of their *operands*, not to a bool,
/// and they short-circuit.  The right-hand side is passed as a callable so
/// that it is only evaluated when Python would evaluate it.
template <class A, class F>
auto And(const A &a, F &&rhs)
{
   using R = typename std::common_type<A, decltype(rhs())>::type;
   return a ? static_cast<R>(rhs()) : static_cast<R>(a);
}

template <class A, class F>
auto Or(const A &a, F &&rhs)
{
   using R = typename std::common_type<A, decltype(rhs())>::type;
   return a ? static_cast<R>(a) : static_cast<R>(rhs());
}

// ---------------------------------------------------------------------------
// Adapting the supported container types to RVec
//
// RVec is the only container with the full set of element-wise operators, so
// std::vector and std::array arguments are viewed as RVecs on entry.  The
// adopting RVec constructor makes this a non-owning view: no copy per event.
// ---------------------------------------------------------------------------

template <class T>
const ROOT::VecOps::RVec<T> &AsRVec(const ROOT::VecOps::RVec<T> &v)
{
   return v;
}

template <class T, class A>
ROOT::VecOps::RVec<T> AsRVec(const std::vector<T, A> &v)
{
   return ROOT::VecOps::RVec<T>(const_cast<T *>(v.data()), v.size());
}

template <class A>
ROOT::VecOps::RVec<bool> AsRVec(const std::vector<bool, A> &v)
{
   // std::vector<bool> is a bitfield and has no data(); this one has to copy.
   return ROOT::VecOps::RVec<bool>(v.begin(), v.end());
}

template <class T, std::size_t N>
ROOT::VecOps::RVec<T> AsRVec(const std::array<T, N> &v)
{
   return ROOT::VecOps::RVec<T>(const_cast<T *>(v.data()), N);
}

// The views above are non-owning, so a view over a temporary would dangle as
// soon as the full expression ends.  Generated code never does this, but the
// header is declared into a live interpreter where anyone can call AsRVec.
template <class T, class A>
ROOT::VecOps::RVec<T> AsRVec(std::vector<T, A> &&) = delete;
template <class T, std::size_t N>
ROOT::VecOps::RVec<T> AsRVec(std::array<T, N> &&) = delete;

// ---------------------------------------------------------------------------
// Element-wise application over a mix of containers and scalars
// ---------------------------------------------------------------------------

template <class T>
std::size_t VecSizeOf(const T &x)
{
   if constexpr (IsVecLike<typename std::decay<T>::type>::value) {
      return x.size();
   } else {
      (void)x;
      return 0u;
   }
}

template <class T>
decltype(auto) ElemAt(const T &x, std::size_t i)
{
   if constexpr (IsVecLike<typename std::decay<T>::type>::value) {
      return x[i];
   } else {
      (void)i;
      return x;
   }
}

template <class A, class B, class F>
auto ZipMap(const A &a, const B &b, F f)
{
   constexpr bool aVec = IsVecLike<typename std::decay<A>::type>::value;
   constexpr bool bVec = IsVecLike<typename std::decay<B>::type>::value;
   static_assert(aVec || bVec, "ZipMap needs at least one container argument");
   const std::size_t na = VecSizeOf(a);
   const std::size_t nb = VecSizeOf(b);
   if (aVec && bVec && na != nb) {
      throw std::runtime_error("PyDeclare: operands could not be broadcast together");
   }
   const std::size_t n = aVec ? na : nb;
   using R = typename std::decay<decltype(f(ElemAt(a, 0), ElemAt(b, 0)))>::type;
   ROOT::VecOps::RVec<R> out(n);
   for (std::size_t i = 0; i < n; ++i) {
      out[i] = f(ElemAt(a, i), ElemAt(b, i));
   }
   return out;
}

// ---------------------------------------------------------------------------
// Python arithmetic semantics
// ---------------------------------------------------------------------------

/// Python's '/' is always true division, even between integers.
template <class T>
auto AsDoubleVec(const T &x)
{
   using E = ElemType<T>;
   ROOT::VecOps::RVec<double> out(x.size());
   for (std::size_t i = 0; i < x.size(); ++i) {
      out[i] = static_cast<double>(static_cast<E>(x[i]));
   }
   return out;
}

template <class A, class B>
auto Div(const A &a, const B &b)
{
   using EA = ElemType<A>;
   using EB = ElemType<B>;
   if constexpr (std::is_floating_point<EA>::value || std::is_floating_point<EB>::value) {
      // C++ already promotes to floating point here, no fixup needed.
      return a / b;
   } else if constexpr (IsVecLike<typename std::decay<A>::type>::value) {
      if constexpr (IsVecLike<typename std::decay<B>::type>::value) {
         return AsDoubleVec(a) / AsDoubleVec(b);
      } else {
         return AsDoubleVec(a) / static_cast<double>(b);
      }
   } else if constexpr (IsVecLike<typename std::decay<B>::type>::value) {
      return static_cast<double>(a) / AsDoubleVec(b);
   } else {
      if constexpr (std::is_integral<EA>::value && std::is_integral<EB>::value) {
         if (b == B(0)) {
            throw std::domain_error("PyDeclare: division by zero");
         }
      }
      return static_cast<double>(a) / static_cast<double>(b);
   }
}

/// Guard the two integer divisions C++ leaves undefined.  Both of them abort
/// the process on x86 rather than producing a wrong number, so there is no
/// choice about paying for the check; Python raises here in any case.
template <class T>
void CheckIntDivisor(T a, T b)
{
   if (b == T(0)) {
      throw std::domain_error("PyDeclare: integer division or modulo by zero");
   }
   if constexpr (std::is_signed<T>::value) {
      if (b == T(-1) && a == std::numeric_limits<T>::min()) {
         throw std::overflow_error("PyDeclare: integer division overflow; the result of "
                                   "-(-2**63) is not representable as a C++ integer");
      }
   }
}

/// Python's '//' floors; C++ integer division truncates towards zero.
template <class A, class B>
auto FloorDivScalar(A a, B b)
{
   if constexpr (std::is_integral<A>::value && std::is_integral<B>::value) {
      using T = IntOpType<A, B>;
      const T x = static_cast<T>(a);
      const T y = static_cast<T>(b);
      CheckIntDivisor(x, y);
      T q = x / y;
      const T r = x % y;
      if (r != 0 && (IsNeg(r) != IsNeg(y))) {
         q -= 1;
      }
      return q;
   } else {
      return std::floor(static_cast<double>(a) / static_cast<double>(b));
   }
}

template <class A, class B>
auto FloorDiv(const A &a, const B &b)
{
   if constexpr (IsVecLike<typename std::decay<A>::type>::value ||
                 IsVecLike<typename std::decay<B>::type>::value) {
      return ZipMap(a, b, [](auto x, auto y) { return FloorDivScalar(x, y); });
   } else {
      return FloorDivScalar(a, b);
   }
}

/// Python's '%' takes the sign of the divisor; C++'s takes the sign of the
/// dividend.  -7 % 3 is 2 in Python and -1 in C++.
template <class A, class B>
auto ModScalar(A a, B b)
{
   if constexpr (std::is_integral<A>::value && std::is_integral<B>::value) {
      using T = IntOpType<A, B>;
      const T x = static_cast<T>(a);
      const T y = static_cast<T>(b);
      CheckIntDivisor(x, y);
      T r = x % y;
      if (r != 0 && (IsNeg(r) != IsNeg(y))) {
         r += y;
      }
      return r;
   } else {
      const double bd = static_cast<double>(b);
      double r = std::fmod(static_cast<double>(a), bd);
      if (r != 0.0 && ((r < 0.0) != (bd < 0.0))) {
         r += bd;
      }
      return r;
   }
}

template <class A, class B>
auto Mod(const A &a, const B &b)
{
   if constexpr (IsVecLike<typename std::decay<A>::type>::value ||
                 IsVecLike<typename std::decay<B>::type>::value) {
      return ZipMap(a, b, [](auto x, auto y) { return ModScalar(x, y); });
   } else {
      return ModScalar(a, b);
   }
}

/// Python's '**'.  int ** non-negative int stays integral, as in Python.
template <class A, class B>
auto PowScalar(A a, B b)
{
   if constexpr (std::is_integral<A>::value && std::is_integral<B>::value) {
      if (IsNeg(b)) {
         throw std::runtime_error("PyDeclare: integer ** negative integer would return a float in Python; "
                                  "cast a base or exponent to a floating point type");
      }
      // Accumulating in A would truncate: a short base overflows at 2**15
      // even though both Python and C++'s own integer promotion would not.
      using T = IntOpType<A, long long>;
      T result = T(1);
      T base = static_cast<T>(a);
      auto e = b;
      while (e > 0) {
         if (e & 1) {
            result *= base;
         }
         e >>= 1;
         if (e) {
            base *= base;
         }
      }
      return result;
   } else {
      return std::pow(a, b);
   }
}

template <class A, class B>
auto Pow(const A &a, const B &b)
{
   if constexpr (IsVecLike<typename std::decay<A>::type>::value ||
                 IsVecLike<typename std::decay<B>::type>::value) {
      return ZipMap(a, b, [](auto x, auto y) { return PowScalar(x, y); });
   } else {
      return PowScalar(a, b);
   }
}

// ---------------------------------------------------------------------------
// Indexing and slicing with Python semantics
// ---------------------------------------------------------------------------

/// Resolve a Python index against a size, wrapping negatives and rejecting
/// anything out of range.  Python raises IndexError rather than reading past
/// the end, and an out-of-bounds read is not something to hand to an analysis.
template <class I>
std::size_t IndexOf(I i, std::size_t size)
{
   const long long n = static_cast<long long>(size);
   long long j = static_cast<long long>(i);
   if constexpr (std::is_signed<I>::value) {
      if (j < 0) {
         j += n;
      }
   }
   if (j < 0 || j >= n) {
      throw std::out_of_range("PyDeclare: index out of range");
   }
   return static_cast<std::size_t>(j);
}

template <class V, class I>
decltype(auto) Index(const V &v, I i)
{
   return v[IndexOf(i, v.size())];
}

template <class V, class I>
decltype(auto) Index(V &v, I i)
{
   return v[IndexOf(i, v.size())];
}

/// Full Python slice semantics, including negative bounds and negative steps.
template <class V>
auto Slice(const V &v, long long start, long long stop, long long step, bool hasStart, bool hasStop)
{
   using E = ElemType<V>;
   if (step == 0) {
      throw std::runtime_error("PyDeclare: slice step cannot be zero");
   }
   const long long n = static_cast<long long>(v.size());
   long long b = 0;
   long long e = 0;
   if (step > 0) {
      b = hasStart ? (start < 0 ? std::max(0LL, n + start) : std::min(start, n)) : 0;
      e = hasStop ? (stop < 0 ? std::max(0LL, n + stop) : std::min(stop, n)) : n;
   } else {
      b = hasStart ? (start < 0 ? std::max(-1LL, n + start) : std::min(start, n - 1)) : n - 1;
      e = hasStop ? (stop < 0 ? std::max(-1LL, n + stop) : std::min(stop, n - 1)) : -1;
   }
   ROOT::VecOps::RVec<E> out;
   if (step > 0) {
      if (e > b) {
         out.reserve(static_cast<std::size_t>((e - b + step - 1) / step));
      }
      for (long long i = b; i < e; i += step) {
         out.push_back(v[static_cast<std::size_t>(i)]);
      }
   } else {
      if (b > e) {
         out.reserve(static_cast<std::size_t>((b - e - step - 1) / (-step)));
      }
      for (long long i = b; i > e; i += step) {
         out.push_back(v[static_cast<std::size_t>(i)]);
      }
   }
   return out;
}

template <class V>
long Len(const V &v)
{
   return static_cast<long>(v.size());
}

// ---------------------------------------------------------------------------
// Reductions, with numpy's result conventions
// ---------------------------------------------------------------------------

/// The type numpy accumulates a sum or a product into: at least 64 bits for
/// any integer input, so that summing an array of small integers does not
/// overflow the element type.  ROOT::VecOps::Sum accumulates into the element
/// type instead, which is why the integer cases below are spelled out.
template <class E>
using AccType =
   typename std::conditional<std::is_integral<E>::value,
                             typename std::conditional<std::is_signed<E>::value ||
                                                          std::is_same<E, bool>::value,
                                                       long long, unsigned long long>::type,
                             E>::type;

/// numpy sums booleans and narrow integers as wide integers.  For floating
/// point the two agree, and the VecOps implementation is used unchanged so
/// that the generated code is exactly as fast as hand-written C++.
template <class V>
auto Sum(const V &v)
{
   using E = ElemType<V>;
   if constexpr (std::is_integral<E>::value) {
      using Acc = AccType<E>;
      Acc acc = 0;
      for (std::size_t i = 0; i < v.size(); ++i) {
         acc += static_cast<Acc>(v[i]);
      }
      return acc;
   } else {
      return ROOT::VecOps::Sum(v);
   }
}

template <class V>
auto Prod(const V &v)
{
   using E = ElemType<V>;
   if constexpr (std::is_integral<E>::value) {
      using Acc = AccType<E>;
      Acc acc = 1;
      for (std::size_t i = 0; i < v.size(); ++i) {
         acc *= static_cast<Acc>(v[i]);
      }
      return acc;
   } else {
      return ROOT::VecOps::Product(v);
   }
}

template <class V>
double Mean(const V &v)
{
   if (v.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
   }
   return ROOT::VecOps::Mean(v);
}

/// numpy's std(), i.e. the population standard deviation (ddof = 0).  Note
/// that ROOT::VecOps::StdDev is the *sample* standard deviation (ddof = 1).
template <class V>
double Std(const V &v)
{
   if (v.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
   }
   const double m = Mean(v);
   double acc = 0.;
   for (std::size_t i = 0; i < v.size(); ++i) {
      const double d = static_cast<double>(v[i]) - m;
      acc += d * d;
   }
   return std::sqrt(acc / static_cast<double>(v.size()));
}

template <class V>
auto Min(const V &v)
{
   if (v.empty()) {
      throw std::runtime_error("PyDeclare: min() of an empty sequence");
   }
   return ROOT::VecOps::Min(v);
}

template <class V>
auto Max(const V &v)
{
   if (v.empty()) {
      throw std::runtime_error("PyDeclare: max() of an empty sequence");
   }
   return ROOT::VecOps::Max(v);
}

template <class V>
long ArgMin(const V &v)
{
   if (v.empty()) {
      throw std::runtime_error("PyDeclare: argmin() of an empty sequence");
   }
   std::size_t best = 0;
   for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] < v[best]) {
         best = i;
      }
   }
   return static_cast<long>(best);
}

template <class V>
long ArgMax(const V &v)
{
   if (v.empty()) {
      throw std::runtime_error("PyDeclare: argmax() of an empty sequence");
   }
   std::size_t best = 0;
   for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[best] < v[i]) {
         best = i;
      }
   }
   return static_cast<long>(best);
}

template <class V>
bool Any(const V &v)
{
   for (std::size_t i = 0; i < v.size(); ++i) {
      if (v[i]) {
         return true;
      }
   }
   return false;
}

template <class V>
bool All(const V &v)
{
   for (std::size_t i = 0; i < v.size(); ++i) {
      if (!v[i]) {
         return false;
      }
   }
   return true;
}

// ---------------------------------------------------------------------------
// Element-wise maths.  Unqualified lookup picks std::f for scalars and
// ROOT::VecOps::f for RVecs.
// ---------------------------------------------------------------------------

#define PYDECLARE_UNARY_MATH(NAME, FN)      \
   template <class T>                       \
   auto NAME(const T &x)                    \
   {                                        \
      using std::FN;                        \
      using namespace ROOT::VecOps;         \
      return FN(x);                         \
   }

PYDECLARE_UNARY_MATH(Sqrt, sqrt)
PYDECLARE_UNARY_MATH(Cbrt, cbrt)
PYDECLARE_UNARY_MATH(Exp, exp)
PYDECLARE_UNARY_MATH(Exp2, exp2)
PYDECLARE_UNARY_MATH(Expm1, expm1)
PYDECLARE_UNARY_MATH(Log, log)
PYDECLARE_UNARY_MATH(Log2, log2)
PYDECLARE_UNARY_MATH(Log10, log10)
PYDECLARE_UNARY_MATH(Log1p, log1p)
PYDECLARE_UNARY_MATH(Sin, sin)
PYDECLARE_UNARY_MATH(Cos, cos)
PYDECLARE_UNARY_MATH(Tan, tan)
PYDECLARE_UNARY_MATH(Asin, asin)
PYDECLARE_UNARY_MATH(Acos, acos)
PYDECLARE_UNARY_MATH(Atan, atan)
PYDECLARE_UNARY_MATH(Sinh, sinh)
PYDECLARE_UNARY_MATH(Cosh, cosh)
PYDECLARE_UNARY_MATH(Tanh, tanh)
PYDECLARE_UNARY_MATH(Asinh, asinh)
PYDECLARE_UNARY_MATH(Acosh, acosh)
PYDECLARE_UNARY_MATH(Atanh, atanh)
PYDECLARE_UNARY_MATH(Floor, floor)
PYDECLARE_UNARY_MATH(Ceil, ceil)
PYDECLARE_UNARY_MATH(Trunc, trunc)
PYDECLARE_UNARY_MATH(Erf, erf)
PYDECLARE_UNARY_MATH(Erfc, erfc)
PYDECLARE_UNARY_MATH(Lgamma, lgamma)
PYDECLARE_UNARY_MATH(Tgamma, tgamma)

#undef PYDECLARE_UNARY_MATH

/// abs() is the one function here with no unsigned overload in <cstdlib>: for
/// an unsigned argument every candidate is an equally good conversion and the
/// call is ambiguous.  Python's abs() of a non-negative number is the identity.
template <class T>
auto AbsScalar(T x)
{
   if constexpr (std::is_integral<T>::value && !std::is_signed<T>::value) {
      return x;
   } else {
      using std::abs;
      return abs(x);
   }
}

template <class T>
auto Abs(const T &x)
{
   if constexpr (IsVecLike<typename std::decay<T>::type>::value) {
      using E = ElemType<T>;
      if constexpr (std::is_integral<E>::value && !std::is_signed<E>::value) {
         return x;
      } else {
         using namespace ROOT::VecOps;
         return abs(x);
      }
   } else {
      return AbsScalar(x);
   }
}

/// Python's round() and numpy's round() both round halves to even, while
/// std::round rounds halves away from zero.  std::nearbyint under the default
/// rounding mode does what Python does.
inline double RoundScalar(double x)
{
   const double r = std::nearbyint(x);
   return r == 0. ? 0. : r; // normalise -0.
}

template <class T>
auto Round(const T &x)
{
   if constexpr (IsVecLike<typename std::decay<T>::type>::value) {
      ROOT::VecOps::RVec<double> out(x.size());
      for (std::size_t i = 0; i < x.size(); ++i) {
         out[i] = RoundScalar(static_cast<double>(x[i]));
      }
      return out;
   } else {
      return RoundScalar(static_cast<double>(x));
   }
}

#define PYDECLARE_BINARY_MATH(NAME, FN)     \
   template <class A, class B>              \
   auto NAME(const A &a, const B &b)        \
   {                                        \
      using std::FN;                        \
      using namespace ROOT::VecOps;         \
      return FN(a, b);                      \
   }

PYDECLARE_BINARY_MATH(Atan2, atan2)
PYDECLARE_BINARY_MATH(Hypot, hypot)
PYDECLARE_BINARY_MATH(Fmod, fmod)

#undef PYDECLARE_BINARY_MATH

template <class A, class B>
auto MaximumScalar(A a, B b)
{
   using R = ArithResult<A, B>;
   return Less(a, b) ? static_cast<R>(b) : static_cast<R>(a);
}

template <class A, class B>
auto MinimumScalar(A a, B b)
{
   using R = ArithResult<A, B>;
   return Less(b, a) ? static_cast<R>(b) : static_cast<R>(a);
}

template <class A, class B>
auto Maximum(const A &a, const B &b)
{
   if constexpr (IsVecLike<typename std::decay<A>::type>::value ||
                 IsVecLike<typename std::decay<B>::type>::value) {
      return ZipMap(a, b, [](auto x, auto y) { return MaximumScalar(x, y); });
   } else {
      return MaximumScalar(a, b);
   }
}

template <class A, class B>
auto Minimum(const A &a, const B &b)
{
   if constexpr (IsVecLike<typename std::decay<A>::type>::value ||
                 IsVecLike<typename std::decay<B>::type>::value) {
      return ZipMap(a, b, [](auto x, auto y) { return MinimumScalar(x, y); });
   } else {
      return MinimumScalar(a, b);
   }
}

template <class C, class A, class B>
auto Where(const C &c, const A &a, const B &b)
{
   if constexpr (IsVecLike<typename std::decay<C>::type>::value) {
      const std::size_t n = c.size();
      if constexpr (IsVecLike<typename std::decay<A>::type>::value) {
         if (a.size() != n) {
            throw std::runtime_error("PyDeclare: operands could not be broadcast together");
         }
      }
      if constexpr (IsVecLike<typename std::decay<B>::type>::value) {
         if (b.size() != n) {
            throw std::runtime_error("PyDeclare: operands could not be broadcast together");
         }
      }
      using R = typename std::decay<decltype(true ? ElemAt(a, 0) : ElemAt(b, 0))>::type;
      ROOT::VecOps::RVec<R> out(n);
      for (std::size_t i = 0; i < n; ++i) {
         out[i] = c[i] ? static_cast<R>(ElemAt(a, i)) : static_cast<R>(ElemAt(b, i));
      }
      return out;
   } else {
      return c ? a : b;
   }
}

// ---------------------------------------------------------------------------
// Returning
// ---------------------------------------------------------------------------

/// Convert the value of the translated return expression to the declared
/// return type.  Anything that cannot be converted is a compile error, which
/// is exactly the loud failure we want.
template <class Out, class T>
Out Return(T &&x)
{
   using D = typename std::decay<T>::type;
   if constexpr (std::is_same<Out, D>::value) {
      return std::forward<T>(x);
   } else if constexpr (IsStdArray<Out>::value && IsVecLike<D>::value) {
      Out out{};
      if (x.size() != out.size()) {
         throw std::runtime_error("PyDeclare: returned sequence has the wrong length for the declared std::array");
      }
      std::copy(x.begin(), x.end(), out.begin());
      return out;
   } else if constexpr (IsVecLike<Out>::value && IsVecLike<D>::value) {
      return Out(x.begin(), x.end());
   } else {
      return static_cast<Out>(x);
   }
}

} // namespace PyDeclare
} // namespace Internal
} // namespace ROOT

#endif // ROOT_PYDECLARE_SUPPORT
"""


def ensure_declared():
    """Declare the C++ support library to cling, once per process."""
    global _DECLARED
    if _DECLARED:
        return
    import ROOT

    # RVec entities are not in the global module index, and gInterpreter.Declare
    # is not allowed to autoload libROOTVecOps, so preload it here.
    ROOT.gSystem.Load("libROOTVecOps")
    if not ROOT.gInterpreter.Declare(CPP_SUPPORT):
        raise RuntimeError("PyDeclare: failed to declare the C++ support library to cling")
    _DECLARED = True
