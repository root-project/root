// Author: Enrico Guiraud, Enric Tejedor, Danilo Piparo CERN  01/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
  \defgroup vecops VecOps
ROOT's TDataFrame allows to analyse data stored in TTrees with a high level interface.
*/

#ifndef ROOT_TVEC
#define ROOT_TVEC

#include <ROOT/RStringView.hxx>
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

inline void CheckSizes(std::size_t s0, std::size_t s1, std::string_view opName)
{
   if (s0 != s1) {
      std::stringstream err;
      err << "Cannot perform operation " << opName << ". The array sizes differ: " << s0 << " and " << s1 << std::endl;
      throw std::runtime_error(err.str());
   }
}

template <typename T0, typename T1, typename F>
inline auto Operate(const TVec<T0> &v0, const TVec<T1> &v1, std::string_view opName, F &&f) -> TVec<decltype(f(v0[0], v1[1]))>
{
   CheckSizes(v0.size(), v1.size(), opName);
   TVec<decltype(f(v0[0], v1[1]))> w(v0.size());
   std::transform(v0.begin(), v0.end(), v1.begin(), w.begin(), std::forward<F>(f));
   return w;
}

template <typename T, typename F>
inline auto Operate(const TVec<T> &v, F &&f) -> TVec<decltype(f(v[0]))>
{
   TVec<decltype(f(v[0]))> w(v.size());
   std::transform(v.begin(), v.end(), w.begin(), std::forward<F>(f));
   return w;
}

} // End of VecOps NS

} // End of Internal NS

namespace Experimental {

namespace VecOps {
// clang-format off
/*
\class ROOT::Experimental::VecOps::TVec
\ingroup vecops
\brief A "std:::vector"-like collection of values implementing handy operation to analyse them
\tparam T The type of the contained objects

A TVec is a container designed to make analysis of values' collections fast and easy.
Its storage is contiguous in memory and its interface is designed such to resemble to the one
of the stl vector. In addition the interface features methods and external functions to ease
the manipulation and analysis of the data in the TVec.

For example, suppose to have in hands an event featuring eight muons with a certain pseudorapidity,
momentum and charge. Suppose you want to extract the transverse momenta of the muons satisfying certain
criteria. Something which would require, among the other things, the management of an explicit loop,
becomes straightforward with TVec:
~~~{.cpp}
TVec<short> mu_charge {1, 1, -1, -1, -1, 1, 1, -1};
TVec<float> mu_pt {56, 45, 32, 24, 12, 8, 7, 6.2};
TVec<float> mu_eta {3.1, -.2, -1.1, 1, 4.1, 1.6, 2.4, -.5};
auto goodMuons_pt = mu_pt[ (mu_pt > 10.f && abs(mu_eta) <= 2.f && mu_charge == -1)
~~~
Now the clean collection of transverse momenta can be used within the rest of the data analysis, for
example to fill a histogram.

## Owning and adopting memory
TVec has contiguous memory associated to it. It can own it or simply adopt it. In the latter case,
it can be constructed with the address of the memory associated to it and its lenght. For example:
~~~{.cpp}
std::vector<int> myStlVec {1,2,3};
TVec<int> myTVec(myStlVec.data(), myStlVec.size());
~~~
In this case, the memory associated to myStlVec and myTVec is the same, myTVec simply "adopted it".
If any method which implies a re-allocation is called, e.g. *emplace_back* or *resize*, the adopted
memory is released and new one is allocated. The previous content is copied in the new memory and
preserved.
*/
// clang-format on
template <typename T>
class TVec {
public:
   using Impl_t = typename std::vector<T, ROOT::Detail::VecOps::TAdoptAllocator<T>>;
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

private:
   Impl_t fData;

   template <typename V, typename F>
   TVec<T> &OperateInPlace(const TVec<V> &v, std::string_view opName, F &&f)
   {
      ROOT::Internal::VecOps::CheckSizes(size(), v.size(), opName);
      std::transform(begin(), end(), v.begin(), begin(), std::forward<F>(f));
      return *this;
   }

   template <typename F>
   TVec<T> &OperateInPlace(F &&f)
   {
      std::transform(begin(), end(), begin(), std::forward<F>(f));
      return *this;
   }

public:
   // ctors
   TVec() = default;
   TVec(const TVec<T>& ) = default;
   TVec(TVec<T>&&) = default;
   template< class InputIt >
   TVec(InputIt first, InputIt last) : fData(first, last) {}
   TVec(size_type count, const T &value) : fData(count, value) {}
   explicit TVec(size_type count) : fData(count) {}
   TVec(const std::vector<T> &other) { std::copy(other.begin(), other.end(), fData.begin()); }
   TVec(std::initializer_list<T> init) : fData(init) {}
   TVec(pointer p, size_type n) : fData(n, T(), ROOT::Detail::VecOps::TAdoptAllocator<T>(p, n)) {}
   // assignment
   TVec<T> &operator=(const TVec<T>& ) = default;
   TVec<T> &operator=(TVec<T>&& ) = default;
   TVec<T> &operator=(std::initializer_list<T> ilist) { return fData = ilist; }
   // accessors
   reference at(size_type pos) { return fData.at(pos); }
   const_reference at(size_type pos) const { return fData.at(pos); }
   reference operator[](size_type pos) { return fData[pos]; }
   const_reference operator[](size_type pos) const { return fData[pos]; }
   template <typename V>
   TVec<T> operator[](const TVec<V> &conds) const
   {
      const auto thisSize = size();
      ROOT::Internal::VecOps::CheckSizes(thisSize, conds.size(), "operator[]");
      TVec<T> w;
      w.reserve(thisSize);
      for (std::size_t i = 0; i < thisSize; i++) {
         if (conds[i]) {
            w.emplace_back(fData[i]);
         }
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
      return OperateInPlace([&c](const T &t) { return t + c; });
   }

   template <typename V>
   TVec<T> &operator-=(const V &c)
   {
      return OperateInPlace([&c](const T &t) { return t - c; });
   }

   template <typename V>
   TVec<T> &operator*=(const V &c)
   {
      return OperateInPlace([&c](const T &t) { return t * c; });
   }

   template <typename V>
   TVec<T> &operator/=(const V &c)
   {
      return OperateInPlace([&c](const T &t) { return t / c; });
   }

   template <typename V>
   TVec<T> &operator%=(const V &c)
   {
      return OperateInPlace([&c](const T &t) { return t % c; });
   }

   template <typename V>
   TVec<T> &operator+=(const TVec<V> &v0)
   {
      return OperateInPlace(v0, "+", [](const T &t, const V &v) { return t + v; });
   }

   template <typename V>
   TVec<T> &operator-=(const TVec<V> &v0)
   {
      return OperateInPlace(v0, "-", [](const T &t, const V &v) { return t - v; });
   }

   template <typename V>
   TVec<T> &operator*=(const TVec<V> &v0)
   {
      return OperateInPlace(v0, "*", [](const T &t, const V &v) { return t * v; });
   }

   template <typename V>
   TVec<T> &operator/=(const TVec<V> &v0)
   {
      return OperateInPlace(v0, "/", [](const T &t, const V &v) { return t / v; });
   }

   template <typename V>
   TVec<T> &operator%=(const TVec<V> &v0)
   {
      return OperateInPlace(v0, "%", [](const T &t, const V &v) { return t % v; });
   }

// Friends for the ADL-lookup

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

template <typename T, typename V>
auto operator+(const V &c, const TVec<T> &v) -> TVec<decltype(v[0] + c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) { return c + t; });
}

template <typename T, typename V>
auto operator-(const V &c, const TVec<T> &v) -> TVec<decltype(v[0] - c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) { return c - t; });
}

template <typename T, typename V>
auto operator*(const V &c, const TVec<T> &v) -> TVec<decltype(v[0] * c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) { return c * t; });
}

template <typename T, typename V>
auto operator/(const V &c, const TVec<T> &v) -> TVec<decltype(v[0] / c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) { return c / t; });
}

template <typename T, typename V>
auto operator%(const V &c, const TVec<T> &v) -> TVec<decltype(v[0] % c)>
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) { return c % t; });
}

// This has been defined to avoid to use the specialisation of vector<bool>
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

template <typename T, typename V>
auto operator&&(const TVec<T> &v, const V &c) -> decltype(v[0] && c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return t && c; });
}

template <typename T, typename V>
auto operator||(const TVec<T> &v, const V &c) -> decltype(v[0] || c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return t || c; });
}

template <typename T, typename V>
auto operator>(const V &c, const TVec<T> &v) -> decltype(v[0] > c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return c > t; });
}

template <typename T, typename V>
auto operator>=(const V &c, const TVec<T> &v) -> decltype(v[0] >= c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return c >= t; });
}

template <typename T, typename V>
auto operator==(const V &c, const TVec<T> &v) -> decltype(v[0] == c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return c == t; });
}

template <typename T, typename V>
auto operator!=(const V &c, const TVec<T> &v) -> decltype(v[0] != c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return c != t; });
}

template <typename T, typename V>
auto operator<=(const V &c, const TVec<T> &v) -> decltype(v[0] <= c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return c <= t; });
}

template <typename T, typename V>
auto operator<(const V &c, const TVec<T> &v) -> decltype(v[0] < c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return c < t; });
}

template <typename T, typename V>
auto operator&&(const V &c, const TVec<T> &v) -> decltype(v[0] && c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return c && t; });
}

template <typename T, typename V>
auto operator||(const V &c, const TVec<T> &v) -> decltype(v[0] || c, TVec<Boolean_t>())
{
   return ROOT::Internal::VecOps::Operate(v, [&c](const T &t) -> Boolean_t { return c || t; });
}

///@}

/** @name Math Operators with TVecs
 *  Math operators involving TVecs
*/
///@{

template <typename T0, typename T1>
auto operator+(const TVec<T0> &v0, const TVec<T1> &v1) -> TVec<decltype(v0[0] + v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "+", [](const T0 &v0, const T1 &v1) { return v0 + v1; });
}

template <typename T0, typename T1>
auto operator-(const TVec<T0> &v0, const TVec<T1> &v1) -> TVec<decltype(v0[0] - v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "-", [](const T0 &v0, const T1 &v1) { return v0 - v1; });
}

template <typename T0, typename T1>
auto operator*(const TVec<T0> &v0, const TVec<T1> &v1) -> TVec<decltype(v0[0] * v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "*", [](const T0 &v0, const T1 &v1) { return v0 * v1; });
}

template <typename T0, typename T1>
auto operator/(const TVec<T0> &v0, const TVec<T1> &v1) -> TVec<decltype(v0[0] / v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "/", [](const T0 &v0, const T1 &v1) { return v0 / v1; });
}

template <typename T0, typename T1>
auto operator%(const TVec<T0> &v0, const TVec<T1> &v1) -> TVec<decltype(v0[0] % v1[0])>
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "%", [](const T0 &v0, const T1 &v1) { return v0 % v1; });
}

template <typename T0, typename T1>
TVec<Boolean_t> operator>(const TVec<T0> &v0, const TVec<T1> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, ">", [](const T0 &v0, const T1 &v1) -> Boolean_t { return v0 > v1; });
}

template <typename T0, typename T1>
TVec<Boolean_t> operator>=(const TVec<T0> &v0, const TVec<T1> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, ">=", [](const T0 &v0, const T1 &v1) -> Boolean_t { return v0 >= v1; });
}

template <typename T0, typename T1>
TVec<Boolean_t> operator==(const TVec<T0> &v0, const TVec<T1> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "==", [](const T0 &v0, const T1 &v1) -> Boolean_t { return v0 == v1; });
}

template <typename T0, typename T1>
TVec<Boolean_t> operator!=(const TVec<T0> &v0, const TVec<T1> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "!=", [](const T0 &v0, const T1 &v1) -> Boolean_t { return v0 != v1; });
}
template <typename T0, typename T1>
TVec<Boolean_t> operator<=(const TVec<T0> &v0, const TVec<T1> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "<=", [](const T0 &v0, const T1 &v1) -> Boolean_t { return v0 <= v1; });
}

template <typename T0, typename T1>
TVec<Boolean_t> operator<(const TVec<T0> &v0, const TVec<T1> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "<", [](const T0 &v0, const T1 &v1) -> Boolean_t { return v0 < v1; });
}

template <typename T0, typename T1>
TVec<Boolean_t> operator&&(const TVec<T0> &v0, const TVec<T1> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "&&", [](const T0 &v0, const T1 &v1) -> Boolean_t { return v0 && v1; });
}

template <typename T0, typename T1>
TVec<Boolean_t> operator||(const TVec<T0> &v0, const TVec<T1> &v1)
{
   return ROOT::Internal::VecOps::Operate(v0, v1, "||", [](const T0 &v0, const T1 &v1) -> Boolean_t { return v0 || v1; });
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
MATH_FUNC(abs)
#undef MATH_FUNC

///@}

/// Inner product
template <typename T, typename V>
auto Dot(const TVec<T> &v0, const TVec<V> &v1) -> decltype(v0[0] * v1[0])
{
   ROOT::Internal::VecOps::CheckSizes(v0.size(), v1.size(), "Dot");
   return std::inner_product(v0.begin(), v0.end(), v1.begin(), decltype(v0[0] * v1[0])(0));
}

/// Sum elements
template <typename T>
T Sum(const TVec<T> &v)
{
   return std::accumulate(v.begin(), v.end(), T(0));
}

/// Create new collection applying a callable to the elements of the input collection
template <typename T, typename F>
auto Map(const TVec<T> &v, F &&f) -> TVec<decltype(f(v[0]))>
{
   return ROOT::Internal::VecOps::Operate(v, std::forward<F>(f));
}

/// Create a new collection with the elements passing the filter expressed by the predicate
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

template <typename T>
void swap(TVec<T> &lhs, TVec<T> &rhs)
{
   lhs.swap(rhs);
}

////////////////////////////////////////////////////////////////////////////////
/// Print a TVec at the prompt:
template <class T>
std::ostream &operator<<(std::ostream &os, const TVec<T> &v)
{
   // In order to print properly, convert to 64 bit int if this is a char
   constexpr bool mustConvert = std::is_same<char, T>::value || std::is_same<signed char, T>::value ||
                                std::is_same<unsigned char, T>::value || std::is_same<wchar_t, T>::value ||
                                std::is_same<char16_t, T>::value || std::is_same<char32_t, T>::value;
   using Print_t = typename std::conditional<mustConvert, long long int, T>::type;
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
