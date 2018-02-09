// Author: Enrico Guiraud, Enric Tejedor, Danilo Piparo CERN  01/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVEC
#define ROOT_TVEC

#include <ROOT/TVecAllocator.hxx>
#include <ROOT/TypeTraits.hxx>

#include <algorithm>
#include <cmath>   // for sqrt
#include <numeric> // for inner_product
#include <sstream>
#include <vector>

namespace ROOT {

namespace Experimental {

namespace VecOps {

template<typename T>
using TCallTraits = typename ROOT::TypeTraits::CallableTraits<T>;

template <typename T>
using TVec = std::vector<T, ROOT::Detail::VecOps::TVecAllocator<T>>;

} // End of Experimental NS

} // End of VecOps NS

// Other helpers
namespace Internal {

namespace VecOps {

using namespace ROOT::Experimental::VecOps;

void CheckSizes(size_t s0, size_t s1, const char *opName)
{
   if (s0 != s1) {
      std::stringstream err;
      err << "Cannot perform operation " << opName
          << ". The array sizes differ: "
          << s0 << " and " << s1 << std::endl;
      throw std::runtime_error(err.str());
   }
}

template <typename T, typename V, typename F>
TVec<typename TCallTraits<F>::ret_type> Operate(const TVec<T> &v0, const TVec<V> &v1, const char* opName, F f)
{
   CheckSizes(v0.size(), v1.size(), opName);
   TVec<typename TCallTraits<F>::ret_type> w;
   w.resize(v0.size());
   std::transform(v0.begin(), v0.end(), v1.begin(), w.begin(), f);
   return w;
}

template <typename T, typename F>
TVec<typename TCallTraits<F>::ret_type> Operate(const TVec<T> &v0, F f)
{
   TVec<typename TCallTraits<F>::ret_type> w;
   w.resize(v0.size());
   std::transform(v0.begin(), v0.end(), w.begin(), f);
   return w;
}


template <typename...>
struct TIsOneOf {
   static constexpr bool value = false;
};

template <typename F, typename S, typename... T>
struct TIsOneOf<F, S, T...> {
   static constexpr bool value = std::is_same<F, S>::value || TIsOneOf<F, T...>::value;
};

template <typename T>
struct TIsChar {
   static constexpr bool value =
      TIsOneOf<typename std::decay<T>::type, char, signed char, unsigned char, wchar_t, char16_t, char32_t>::value;
};

} // End of VecOps NS

} // End of Internal NS


namespace Detail {

namespace VecOps {

   template <typename>
   struct TIsTVec {
      static const bool value = false;
   };

   template <typename T>
   struct TIsTVec<ROOT::Experimental::VecOps::TVec<T>> {
      static const bool value = true;
   };

} // End of VecOps NS

} // End of Detail NS

namespace Experimental {

namespace VecOps {

/** @name Math Operators with scalars
 *  Math operators involving TVec
*/
///@{
template <typename T, typename V>
auto operator+(const TVec<T> &v, const V &c) -> TVec<decltype(v[0] + c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T& t) {return t+c;});
}

template <typename T, typename V>
auto operator-(const TVec<T> &v, const V &c) -> TVec<decltype(v[0] - c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T& t) {return t-c;});
}

template <typename T, typename V>
auto operator*(const TVec<T> &v, const V &c) -> TVec<decltype(v[0] * c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T& t) {return t*c;});
}

template <typename T, typename V>
auto operator/(const TVec<T> &v, const V &c) -> TVec<decltype(v[0] / c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T& t) {return t/c;});
}

template <typename T, typename V, typename D = typename std::enable_if<!ROOT::Detail::VecOps::TIsTVec<V>::value, int>::type>
TVec<char> operator>(const TVec<T> &v, const V &c)
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T& t) -> char {return t>c;});
}

template <typename T, typename V, typename D = typename std::enable_if<!ROOT::Detail::VecOps::TIsTVec<V>::value, int>::type>
TVec<char> operator>=(const TVec<T> &v, const V &c)
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T& t) -> char {return t>=c;});
}

template <typename T, typename V, typename D = typename std::enable_if<!ROOT::Detail::VecOps::TIsTVec<V>::value, int>::type>
TVec<char> operator==(const TVec<T> &v, const V &c)
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T& t) -> char {return t==c;});
}


template <typename T, typename V, typename D = typename std::enable_if<!ROOT::Detail::VecOps::TIsTVec<V>::value, int>::type>
TVec<char> operator!=(const TVec<T> &v, const V &c)
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T& t) -> char {return t!=c;});
}

template <typename T, typename V, typename D = typename std::enable_if<!ROOT::Detail::VecOps::TIsTVec<V>::value, int>::type>
TVec<char> operator<=(const TVec<T> &v, const V &c)
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T& t) -> char {return t<=c;});
}

template <typename T, typename V, typename D = typename std::enable_if<!ROOT::Detail::VecOps::TIsTVec<V>::value, int>::type>
TVec<char> operator<(const TVec<T> &v, const V &c)
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T& t) -> char {return t<c;});
}
///@}

/** @name Math Operators with TVecs
 *  Math operators involving TVecs
*/
///@{

template <typename T, typename V>
auto operator+(const TVec<T> &v0, const TVec<V> &v1) -> TVec<decltype(v0[0] + v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "+", [](const T& t, const V &v) {return t+v;});
}

template <typename T, typename V>
auto operator-(const TVec<T> &v0, const TVec<V> &v1) -> TVec<decltype(v0[0] - v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "-", [](const T& t, const V &v) {return t-v;});
}

template <typename T, typename V>
auto operator*(const TVec<T> &v0, const TVec<V> &v1) -> TVec<decltype(v0[0] * v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "*", [](const T& t, const V &v) {return t*v;});
}

template <typename T, typename V>
auto operator/(const TVec<T> &v0, const TVec<V> &v1) -> TVec<decltype(v0[0] / v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "/", [](const T& t, const V &v) {return t/v;});
}

template <typename T, typename V>
TVec<char> operator>(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, ">", [](const T& t, const V &v) -> char {return t>v;});
}

template <typename T, typename V>
TVec<char> operator>=(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, ">=", [](const T& t, const V &v) -> char {return t>=v;});
}

template <typename T, typename V>
TVec<char> operator==(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "==", [](const T& t, const V &v) -> char {return t==v;});
}

template <typename T, typename V>
TVec<char> operator!=(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "!=", [](const T& t, const V &v) -> char {return t!=v;});
}
template <typename T, typename V>
TVec<char> operator<=(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "<=", [](const T& t, const V &v) -> char {return t<=v;});
}

template <typename T, typename V>
TVec<char> operator<(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "<", [](const T& t, const V &v) -> char {return t<v;});
}

// Workaround for operators >&co on vectors
template <typename T>
TVec<char> operator>(const TVec<T> &v0, const TVec<T> &v1)
{
   return ROOT::Experimental::VecOps::operator><T, T>(v0, v1);
}
template <typename T>
TVec<char> operator>=(const TVec<T> &v0, const TVec<T> &v1)
{
   return ROOT::Experimental::VecOps::operator>=<T, T>(v0, v1);
}
template <typename T>
TVec<char> operator==(const TVec<T> &v0, const TVec<T> &v1)
{
   return ROOT::Experimental::VecOps::operator==<T, T>(v0, v1);
}
template <typename T>
TVec<char> operator!=(const TVec<T> &v0, const TVec<T> &v1)
{
   return ROOT::Experimental::VecOps::operator!=<T, T>(v0, v1);
}
template <typename T>
TVec<char> operator<=(const TVec<T> &v0, const TVec<T> &v1)
{
   return ROOT::Experimental::VecOps::operator<=<T, T>(v0, v1);
}
template <typename T>
TVec<char> operator<(const TVec<T> &v0, const TVec<T> &v1)
{
   // clang-format off
   return ROOT::Experimental::VecOps::operator< <T, T>(v0, v1);
   // clang-format on
}

///@}

/** @name Math Functions
 *  Math functions on TVecs
*/
///@{
#define MATH_FUNC(FUNCNAME)\
template <typename T>\
auto FUNCNAME(const TVec<T> &v) -> TVec<decltype(std::FUNCNAME(v[0]))>\
{return ROOT::Internal::VecOps::Operate(v, [](const T& t) {return std::FUNCNAME(t);});}

MATH_FUNC(sqrt)
MATH_FUNC(log)
MATH_FUNC(sin)
MATH_FUNC(cos)
MATH_FUNC(asin)
MATH_FUNC(acos)
MATH_FUNC(tan)
MATH_FUNC(atan)
MATH_FUNC(sinh)
MATH_FUNC(cosh)
MATH_FUNC(asinh)
MATH_FUNC(acosh)
MATH_FUNC(tanh)
MATH_FUNC(atanh)


///@}

/// Inner product
template <typename T, typename V>
auto Dot(const TVec<T> v0, const TVec<V> v1) -> decltype(v0[0] * v1[0])
{
   ROOT::Internal::VecOps::CheckSizes(v0.size(), v1.size(), "Dot");
   return std::inner_product(v0.begin(), v0.end(), v1.begin(), decltype(v0[0] * v1[0])(0));
}

/// Sum elements
template <typename T>
T Sum(const TVec<T> v)
{
   return std::accumulate(v.begin(), v.end(), 0);
}

template <class T>
std::ostream &operator<<(std::ostream &os, const TVec<T> &v)
{
   // In order to print properly, convert to 64 bit int if this is a char
   using Print_t = typename std::conditional<ROOT::Internal::VecOps::TIsChar<T>::value, long long int, T>::type;
   os << "{ ";
   auto size = v.size();
   if (size) {
      for (size_t i = 0; i < size - 1; ++i) {
         os << (Print_t)v[i] << ", ";
      }
      os << (Print_t)v[size - 1];
   }
   os << " }";
   return os;
}

template <typename T>
TVec<T> Filter(const TVec<T> &v, bool cond)
{
   return cond ? v : TVec<T>();
}

template <typename T, typename V>
TVec<T> Filter(const TVec<T> &v, const TVec<V> &conds)
{
   ROOT::Internal::VecOps::CheckSizes(v.size(), conds.size(), "Filter");
   TVec<T> w;
   auto size = v.size();
   w.reserve(size);
   for (size_t i = 0; i< size; i++) {
      if (conds[i]) w.emplace_back(v[i]);
   }
   return w;
}

} // End of VecOps NS

} // End of Experimental NS

} // End of ROOT NS

namespace cling {
template <typename T>
inline std::string printValue(ROOT::Experimental::VecOps::TVec<T> *tvec)
{
   std::stringstream ss;
   ss << *tvec;
   return ss.str();
}

} // namespace cling

#endif
