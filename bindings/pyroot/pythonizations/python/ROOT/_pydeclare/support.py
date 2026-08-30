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
#include <limits>
#include <stdexcept>
#include <type_traits>
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
      return static_cast<double>(a) / static_cast<double>(b);
   }
}

/// Python's '//' floors; C++ integer division truncates towards zero.
template <class A, class B>
auto FloorDivScalar(A a, B b)
{
   if constexpr (std::is_integral<A>::value && std::is_integral<B>::value) {
      auto q = a / b;
      auto r = a % b;
      if (r != 0 && (IsNeg(r) != IsNeg(b))) {
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
      auto r = a % b;
      if (r != 0 && (IsNeg(r) != IsNeg(b))) {
         r += b;
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
      A result = A(1);
      A base = a;
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

template <class V, class I>
decltype(auto) Index(const V &v, I i)
{
   if constexpr (std::is_signed<I>::value) {
      if (i < 0) {
         const long long n = static_cast<long long>(v.size());
         const long long j = n + static_cast<long long>(i);
         if (j < 0) {
            throw std::out_of_range("PyDeclare: index out of range");
         }
         return v[static_cast<std::size_t>(j)];
      }
   }
   return v[static_cast<std::size_t>(i)];
}

template <class V, class I>
decltype(auto) Index(V &v, I i)
{
   if constexpr (std::is_signed<I>::value) {
      if (i < 0) {
         const long long n = static_cast<long long>(v.size());
         const long long j = n + static_cast<long long>(i);
         if (j < 0) {
            throw std::out_of_range("PyDeclare: index out of range");
         }
         return v[static_cast<std::size_t>(j)];
      }
   }
   return v[static_cast<std::size_t>(i)];
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

template <class V>
auto Sum(const V &v)
{
   using E = ElemType<V>;
   using R = typename std::conditional<std::is_same<E, bool>::value, long, E>::type;
   R acc = R(0);
   for (std::size_t i = 0; i < v.size(); ++i) {
      acc += static_cast<R>(v[i]);
   }
   return acc;
}

template <class V>
auto Prod(const V &v)
{
   using E = ElemType<V>;
   using R = typename std::conditional<std::is_same<E, bool>::value, long, E>::type;
   R acc = R(1);
   for (std::size_t i = 0; i < v.size(); ++i) {
      acc *= static_cast<R>(v[i]);
   }
   return acc;
}

template <class V>
double Mean(const V &v)
{
   if (v.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
   }
   double acc = 0.;
   for (std::size_t i = 0; i < v.size(); ++i) {
      acc += static_cast<double>(v[i]);
   }
   return acc / static_cast<double>(v.size());
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
   using E = ElemType<V>;
   if (v.empty()) {
      throw std::runtime_error("PyDeclare: min() of an empty sequence");
   }
   E best = v[0];
   for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] < best) {
         best = v[i];
      }
   }
   return best;
}

template <class V>
auto Max(const V &v)
{
   using E = ElemType<V>;
   if (v.empty()) {
      throw std::runtime_error("PyDeclare: max() of an empty sequence");
   }
   E best = v[0];
   for (std::size_t i = 1; i < v.size(); ++i) {
      if (best < v[i]) {
         best = v[i];
      }
   }
   return best;
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

PYDECLARE_UNARY_MATH(Abs, abs)
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
auto Maximum(const A &a, const B &b)
{
   if constexpr (IsVecLike<typename std::decay<A>::type>::value ||
                 IsVecLike<typename std::decay<B>::type>::value) {
      return ZipMap(a, b, [](auto x, auto y) { return x > y ? x : y; });
   } else {
      return a > b ? a : b;
   }
}

template <class A, class B>
auto Minimum(const A &a, const B &b)
{
   if constexpr (IsVecLike<typename std::decay<A>::type>::value ||
                 IsVecLike<typename std::decay<B>::type>::value) {
      return ZipMap(a, b, [](auto x, auto y) { return x < y ? x : y; });
   } else {
      return a < b ? a : b;
   }
}

template <class C, class A, class B>
auto Where(const C &c, const A &a, const B &b)
{
   if constexpr (IsVecLike<typename std::decay<C>::type>::value) {
      const std::size_t n = c.size();
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
