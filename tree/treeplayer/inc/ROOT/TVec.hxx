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

#include "RStringView.h"

#include <ROOT/TAdoptAllocator.hxx>
#include <ROOT/TypeTraits.hxx>

#include <algorithm>
#include <cmath>
#include <numeric> // for inner_product
#include <sstream>
#include <stdexcept>
#include <vector>
#include <utility>

namespace ROOT {

namespace Detail {

namespace VecOps {

template <typename T>
using TVecImpl = std::vector<T, ROOT::Detail::VecOps::TAdoptAllocator<T>>;

} // End of VecOps NS

} // End of Detail NS

namespace Experimental {

namespace VecOps {

template <typename T>
class TVec;

} // End of Experimental NS

} // End of VecOps NS

// Other helpers
namespace Internal {

namespace VecOps {

using namespace ROOT::Experimental::VecOps;

void CheckSizes(std::size_t s0, std::size_t s1, std::string_view opName)
{
   if (s0 != s1) {
      std::stringstream err;
      err << "Cannot perform operation " << opName << ". The array sizes differ: " << s0 << " and " << s1 << std::endl;
      throw std::runtime_error(err.str());
   }
}

template <typename T, typename V, typename F>
auto Operate(const TVec<T> &v0, const TVec<V> &v1, std::string_view opName, F &&f) -> TVec<decltype(f(v0[0], v1[1]))>
{
   CheckSizes(v0.size(), v1.size(), opName);
   TVec<decltype(f(v0[0], v1[1]))> w(v0.size());
   std::transform(v0.begin(), v0.end(), v1.begin(), w.begin(), std::forward<F>(f));
   return w;
}

template <typename T, typename V, typename F>
TVec<T> &OperateInPlace(TVec<T> &v0, const TVec<V> &v1, std::string_view opName, F &&f)
{
   const auto v0size = v0.size();
   CheckSizes(v0size, v1.size(), opName);
   std::transform(v0.begin(), v0.end(), v1.begin(), v0.begin(), std::forward<F>(f));
   return v0;
}

template <typename T, typename F>
auto Operate(const TVec<T> &v, F &&f) -> TVec<decltype(f(v[0]))>
{
   TVec<decltype(f(v[0]))> w(v.size());
   std::transform(v.begin(), v.end(), w.begin(), std::forward<F>(f));
   return w;
}

template <typename T, typename F>
TVec<T> &OperateInPlace(TVec<T> &v, F &&f)
{
   std::transform(v.begin(), v.end(), v.begin(), std::forward<F>(f));
   return v;
}

} // End of VecOps NS

} // End of Internal NS

namespace Experimental {

namespace VecOps {

template <typename T>
class TVec {
private:
   ROOT::Detail::VecOps::TVecImpl<T> fData;

public:
   using Impl_t = typename ROOT::Detail::VecOps::TVecImpl<T>;
   using value_type = typename Impl_t::value_type;
   using size_type = typename Impl_t::size_type;
   using difference_type = typename Impl_t::difference_type;
   using reference = typename Impl_t::reference;
   using const_reference = typename Impl_t::const_reference;
   using pointer = typename Impl_t::pointer;
   using const_pointer = typename Impl_t::const_pointer;
   using iterator = typename Impl_t::iterator;
   using const_iterator = typename Impl_t::const_iterator;
   using reverse_iterator = typename Impl_t::reverse_iterator;

   // ctors
   TVec() {}
   TVec(size_type count, const T &value) : fData(count, value) {}
   explicit TVec(size_type count) : fData(count) {}
   TVec(const std::vector<T> &other) { std::copy(other.begin(), other.end(), fData.begin()); }
   TVec(std::initializer_list<T> init) : fData(init) {}
   TVec(pointer p, size_type n) : fData(n, T(), ROOT::Detail::VecOps::TAdoptAllocator<T>(p, n)) {}
   // assignment
   TVec<T> &operator=(std::initializer_list<T> ilist) { return fData = ilist; }
   // accessors
   reference at(size_type pos) { return fData.at(pos); }
   const_reference at(size_type pos) const { return fData.at(pos); }
   reference operator[](size_type pos) { return fData[pos]; }
   const_reference operator[](size_type pos) const { return fData[pos]; }
   template <typename V>
   TVec<T> operator[](const TVec<V>& conds)
   {
      const auto thisSize = size();
      ROOT::Internal::VecOps::CheckSizes(thisSize, conds.size(), "operator[]");
      TVec<T> w;
      w.reserve(thisSize);
      for (std::size_t i = 0; i < thisSize; i++) {
         if (conds[i])
            w.emplace_back(fData[i]);
      }
      return w;
   }
   reference front() { return fData.front(); }
   const_reference front() const { return fData.front(); }
   reference back() { return fData.back(); }
   const_reference back() const { return fData.back(); }
   T *data() noexcept { return fData.data(); }
   const T *data() const noexcept { return fData.data(); }
   // iterators
   iterator begin() noexcept { return fData.begin(); }
   const_iterator begin() const noexcept { return fData.begin(); }
   const_iterator cbegin() const noexcept { return fData.cbegin(); }
   iterator end() noexcept { return fData.end(); }
   const_iterator end() const noexcept { return fData.end(); }
   const_iterator cend() const noexcept { return fData.cend(); }
   iterator rbegin() noexcept { return fData.rbegin(); }
   const_iterator rbegin() const noexcept { return fData.rbegin(); }
   const_iterator crbegin() const noexcept { return fData.crbegin(); }
   iterator rend() noexcept { return fData.rend(); }
   const_iterator rend() const noexcept { return fData.rend(); }
   const_iterator crend() const noexcept { return fData.crend(); }
   // capacity
   bool empty() const noexcept { return fData.empty(); }
   size_type size() const noexcept { return fData.size(); }
   size_type max_size() const noexcept { return fData.size(); }
   void reserve(size_type new_cap) { fData.reserve(new_cap); }
   size_type capacity() const noexcept { return fData.capacity(); }
   void shrink_to_fit() { fData.shrink_to_fit(); };
   // modifiers
   void clear() noexcept { fData.clear(); }
   iterator erase(const_iterator pos) { return fData.erase(pos); }
   iterator erase(const_iterator first, const_iterator last) { return fData.erase(first, last); }
   void push_back(T &&value) { fData.push_back(std::forward<T>(value)); }
   template <class... Args>
   reference emplace_back(Args &&... args)
   {
      fData.emplace_back(std::forward<Args>(args)...);
      return fData.back();
   }
   template <class... Args>
   iterator emplace(const_iterator pos, Args &&... args)
   {
      return fData.emplace(pos, std::forward<Args...>(args...));
   }
   void pop_back() { fData.pop_back(); }
   void resize(size_type count) { fData.resize(count); }
   void resize(size_type count, const value_type &value) { fData.resize(count, value); }
   void swap(TVec<T> &other) { std::swap(fData, other.fData); }
   // arithmetic operators
   template <typename V>
   TVec<T> &operator+=(const V &c)
   {
      return ROOT::Internal::VecOps::OperateInPlace(*this, [&c](const T &t) { return t + c; });
   }

   template <typename V>
   TVec<T> &operator-=(const V &c)
   {
      return ROOT::Internal::VecOps::OperateInPlace(*this, [&c](const T &t) { return t - c; });
   }

   template <typename V>
   TVec<T> &operator*=(const V &c)
   {
      return ROOT::Internal::VecOps::OperateInPlace(*this, [&c](const T &t) { return t * c; });
   }

   template <typename V>
   TVec<T> &operator/=(const V &c)
   {
      return ROOT::Internal::VecOps::OperateInPlace(*this, [&c](const T &t) { return t / c; });
   }

   template <typename V>
   TVec<T> &operator%=(const V &c)
   {
      return ROOT::Internal::VecOps::OperateInPlace(*this, [&c](const T &t) { return t % c; });
   }

   template <typename V>
   TVec<T> &operator+=(const TVec<V> &v0)
   {
      return ROOT::Internal::VecOps::OperateInPlace(*this, v0, "+", [](const T &t, const V &v) { return t + v; });
   }

   template <typename V>
   TVec<T> &operator-=(const TVec<V> &v0)
   {
      return ROOT::Internal::VecOps::OperateInPlace(*this, v0, "-", [](const T &t, const V &v) { return t - v; });
   }

   template <typename V>
   TVec<T> &operator*=(const TVec<V> &v0)
   {
      return ROOT::Internal::VecOps::OperateInPlace(*this, v0, "*", [](const T &t, const V &v) { return t * v; });
   }

   template <typename V>
   TVec<T> &operator/=(const TVec<V> &v0)
   {
      return ROOT::Internal::VecOps::OperateInPlace(*this, v0, "/", [](const T &t, const V &v) { return t / v; });
   }

   template <typename V>
   TVec<T> &operator%=(const TVec<V> &v0)
   {
      return ROOT::Internal::VecOps::OperateInPlace(*this, v0, "%", [](const T &t, const V &v) { return t % v; });
   }
};

/** @name Math Operators with scalars
 *  Math operators involving TVec
*/
///@{
template <typename T, typename V>
auto operator+(const TVec<T> &v, const V &c) -> TVec<decltype(v[0] + c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) { return t + c; });
}

template <typename T, typename V>
auto operator-(const TVec<T> &v, const V &c) -> TVec<decltype(v[0] - c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) { return t - c; });
}

template <typename T, typename V>
auto operator*(const TVec<T> &v, const V &c) -> TVec<decltype(v[0] * c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) { return t * c; });
}

template <typename T, typename V>
auto operator/(const TVec<T> &v, const V &c) -> TVec<decltype(v[0] / c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) { return t / c; });
}

template <typename T, typename V>
auto operator%(const TVec<T> &v, const V &c) -> TVec<decltype(v[0] % c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) { return t % c; });
}

using Boolean_t = int;

template <typename T, typename V>
auto operator>(const TVec<T> &v, const V &c) -> decltype(v[0] > c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return t > c; });
}

template <typename T, typename V>
auto operator>=(const TVec<T> &v, const V &c) -> decltype(v[0] >= c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return t >= c; });
}

template <typename T, typename V>
auto operator==(const TVec<T> &v, const V &c) -> decltype(v[0] == c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return t == c; });
}

template <typename T, typename V>
auto operator!=(const TVec<T> &v, const V &c) -> decltype(v[0] != c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return t != c; });
}

template <typename T, typename V>
auto operator<=(const TVec<T> &v, const V &c) -> decltype(v[0] <= c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return t <= c; });
}

template <typename T, typename V>
auto operator<(const TVec<T> &v, const V &c) -> decltype(v[0] < c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return t < c; });
}
///@}

/** @name Math Operators with TVecs
 *  Math operators involving TVecs
*/
///@{

template <typename T, typename V>
auto operator+(const TVec<T> &v0, const TVec<V> &v1) -> TVec<decltype(v0[0] + v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "+", [](const T &t, const V &v) { return t + v; });
}

template <typename T, typename V>
auto operator-(const TVec<T> &v0, const TVec<V> &v1) -> TVec<decltype(v0[0] - v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "-", [](const T &t, const V &v) { return t - v; });
}

template <typename T, typename V>
auto operator*(const TVec<T> &v0, const TVec<V> &v1) -> TVec<decltype(v0[0] * v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "*", [](const T &t, const V &v) { return t * v; });
}

template <typename T, typename V>
auto operator/(const TVec<T> &v0, const TVec<V> &v1) -> TVec<decltype(v0[0] / v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "/", [](const T &t, const V &v) { return t / v; });
}

template <typename T, typename V>
auto operator%(const TVec<T> &v0, const TVec<V> &v1) -> TVec<decltype(v0[0] % v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "%", [](const T &t, const V &v) { return t % v; });
}

template <typename T, typename V>
TVec<Boolean_t> operator>(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, ">", [](const T &t, const V &v) -> Boolean_t { return t > v; });
}

template <typename T, typename V>
TVec<Boolean_t> operator>=(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, ">=", [](const T &t, const V &v) -> Boolean_t { return t >= v; });
}

template <typename T, typename V>
TVec<Boolean_t> operator==(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "==", [](const T &t, const V &v) -> Boolean_t { return t == v; });
}

template <typename T, typename V>
TVec<Boolean_t> operator!=(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "!=", [](const T &t, const V &v) -> Boolean_t { return t != v; });
}
template <typename T, typename V>
TVec<Boolean_t> operator<=(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "<=", [](const T &t, const V &v) -> Boolean_t { return t <= v; });
}

template <typename T, typename V>
TVec<Boolean_t> operator<(const TVec<T> &v0, const TVec<V> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "<", [](const T &t, const V &v) -> Boolean_t { return t < v; });
}
///@}

/** @name Math Functions
 *  Math functions on TVecs
*/
///@{
#define MATH_FUNC(FUNCNAME)                                                                   \
   template <typename T>                                                                      \
   auto FUNCNAME(const TVec<T> &v)->TVec<decltype(std::FUNCNAME(v[0]))>                       \
   {                                                                                          \
      return ROOT::Internal::VecOps::Operate(v, [](const T &t) { return std::FUNCNAME(t); }); \
   }

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

template <typename T, typename F>
auto Map(const TVec<T> &v, F &&f) -> TVec<decltype(f(v[0]))>
{
   return ROOT::Internal::VecOps::Operate(v, std::forward<F>(f));
}

template <typename T, typename F>
TVec<T> Filter(const TVec<T> &v, F &&f)
{
   const auto thisSize = v.size();
   TVec<T> w;
   w.reserve(thisSize);
   for (auto &&val : v) {
      if (f(val))
         w.emplace_back(val);
   }
   return w;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const TVec<T> &v)
{
   // In order to print properly, convert to 64 bit int if this is a char
   constexpr bool mustConvert = std::is_same<char, T>::value ||
                                std::is_same<signed char, T>::value ||
                                std::is_same<unsigned char, T>::value ||
                                std::is_same<wchar_t, T>::value ||
                                std::is_same<char16_t, T>::value ||
                                std::is_same<char32_t, T>::value;
   using Print_t = typename std::conditional<mustConvert, long long int ,T>::type;
   os << "{ ";
   auto size = v.size();
   if (size) {
      for (std::size_t i = 0; i < size - 1; ++i) {
         os << (Print_t)v[i] << ", ";
      }
      os << (Print_t)v[size - 1];
   }
   os << " }";
   return os;
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
