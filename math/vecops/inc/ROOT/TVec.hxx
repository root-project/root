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
*/

#ifndef ROOT_TVEC
#define ROOT_TVEC

#define _VECOPS_USE_EXTERN_TEMPLATES true

// We do not support extern templates on Win
#ifdef _WIN32
#undef _VECOPS_USE_EXTERN_TEMPLATES
#define _VECOPS_USE_EXTERN_TEMPLATES false
#endif // _WIN32

// We do not support extern templates on Linux if the compiler is old
#ifdef R_LINUX
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) <= 40800
#undef _VECOPS_USE_EXTERN_TEMPLATES
#define _VECOPS_USE_EXTERN_TEMPLATES false
#endif // GCC version
#ifdef __clang__
#undef _VECOPS_USE_EXTERN_TEMPLATES
#define _VECOPS_USE_EXTERN_TEMPLATES false
#endif
#endif // R__LINUX


#include <ROOT/TAdoptAllocator.hxx>
#include <ROOT/TypeTraits.hxx>

#include <algorithm>
#include <cmath>
#include <numeric> // for inner_product
#include <sstream>
#include <stdexcept>
#include <vector>
#include <utility>

#ifdef R__HAS_VDT
#include <vdt/vdtMath.h>
#endif

namespace ROOT {

namespace Experimental {

namespace VecOps {
// clang-format off
/**
\class ROOT::Experimental::VecOps::TVec
\ingroup vecops
\brief A "std::vector"-like collection of values implementing handy operation to analyse them
\tparam T The type of the contained objects

A TVec is a container designed to make analysis of values' collections fast and easy.
Its storage is contiguous in memory and its interface is designed such to resemble to the one
of the stl vector. In addition the interface features methods and external functions to ease
the manipulation and analysis of the data in the TVec.

## Table of Contents
- [Example](#example)
- [Owning and adopting memory](#owningandadoptingmemory)
- [Usage in combination with TDataFrame](#usagetdataframe)

## <a name="example"></a>Example
Suppose to have an event featuring a collection of muons with a certain pseudorapidity,
momentum and charge, e.g.:
~~~{.cpp}
std::vector<short> mu_charge {1, 1, -1, -1, -1, 1, 1, -1};
std::vector<float> mu_pt {56, 45, 32, 24, 12, 8, 7, 6.2};
std::vector<float> mu_eta {3.1, -.2, -1.1, 1, 4.1, 1.6, 2.4, -.5};
~~~
Suppose you want to extract the transverse momenta of the muons satisfying certain
criteria, for example consider only negatively charged muons with a pseudorapidity
smaller or equal to 2 and with a transverse momentum greater than 10 GeV.
Such a selection would require, among the other things, the management of an explicit
loop, for example:
~~~{.cpp}
std::vector<float> goodMuons_pt;
const auto size = mu_charge.size();
for (size_t i=0; i < size; ++i) {
   if (mu_pt[i] > 10 && abs(mu_eta[i]) <= 2. &&  mu_charge[i] == -1) {
      goodMuons_pt.emplace_back(mu_pt[i]);
   }
}
~~~
These operations become straightforward with TVec - we just need to *write what
we mean*:
~~~{.cpp}
TVec<short> mu_charge {1, 1, -1, -1, -1, 1, 1, -1};
TVec<float> mu_pt {56, 45, 32, 24, 12, 8, 7, 6.2};
TVec<float> mu_eta {3.1, -.2, -1.1, 1, 4.1, 1.6, 2.4, -.5};

auto goodMuons_pt = mu_pt[ (mu_pt > 10.f && abs(mu_eta) <= 2.f && mu_charge == -1)
~~~
Now the clean collection of transverse momenta can be used within the rest of the data analysis, for
example to fill a histogram.

## <a name="owningandadoptingmemory"></a>Owning and adopting memory
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

## <a name="usagetdataframe"></a>Usage in combination with TDataFrame
TDataFrame leverages internally TVecs. Suppose to have a dataset stored in a
TTree which holds these columns (here we choose C arrays to represent the
collections, they could be as well std::vector instances):
~~~{.bash}
  nPart            "nPart/I"            An integer representing the number of particles
  px               "px[nPart]/D"        The C array of the particles' x component of the momentum
  py               "py[nPart]/D"        The C array of the particles' y component of the momentum
  E                "E[nPart]/D"         The C array of the particles' Energy
~~~
Suppose you'd like to plot in a histogram the transverse momenta of all particles
for which the energy is greater than 200 MeV.
The code required would just be:
~~~{.cpp}
TDataFrame d("mytree", "myfile.root");
using doubles = TVec<double>;
auto cutPt = [](doubles &pxs, doubles &pys, doubles &Es) {
   auto all_pts = sqrt(pxs * pxs + pys * pys);
   auto good_pts = all_pts[Es > 200.];
   return good_pts;
   };

auto hpt = d.Define("pt", cutPt, {"px", "py", "E"})
            .Histo1D("pt");
hpt->Draw();
~~~
And if you'd like to express your selection as a string:
~~~{.cpp}
TDataFrame d("mytree", "myfile.root");
auto hpt = d.Define("pt", "sqrt(pxs * pxs + pys * pys)[E>200]")
            .Histo1D("pt");
hpt->Draw();
~~~

**/
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
   using const_reverse_iterator = typename Impl_t::const_reverse_iterator;

private:
   Impl_t fData;

public:
   // constructors
   TVec() {}

   explicit TVec(size_type count) : fData(count) {}

   TVec(size_type count, const T &value) : fData(count, value) {}

   TVec(const TVec<T> &v) : fData(v.fData) {}

   TVec(TVec<T> &&v) : fData(std::move(v.fData)) {}

   TVec(const std::vector<T> &v) : fData(v.cbegin(), v.cend()) {}

   TVec(pointer p, size_type n) : fData(n, T(), ROOT::Detail::VecOps::TAdoptAllocator<T>(p)) {}

   template <class InputIt>
   TVec(InputIt first, InputIt last) : fData(first, last) {}

   TVec(std::initializer_list<T> init) : fData(init) {}

   // assignment
   TVec<T> &operator=(const TVec<T> &v)
   {
      fData = v.fData;
      return *this;
   }

   TVec<T> &operator=(TVec<T> &&v)
   {
      std::swap(fData, v.fData);
      return *this;
   }

   TVec<T> &operator=(std::initializer_list<T> ilist)
   {
      fData = ilist;
      return *this;
   }

   // conversion
   template <typename U, typename = std::enable_if<std::is_convertible<T, U>::value>>
   operator TVec<U>() const
   {
      TVec<U> ret(size());
      std::copy(begin(), end(), ret.begin());
      return ret;
   }

   // accessors
   reference at(size_type pos) { return fData.at(pos); }
   const_reference at(size_type pos) const { return fData.at(pos); }
   reference operator[](size_type pos) { return fData[pos]; }
   const_reference operator[](size_type pos) const { return fData[pos]; }

   template <typename V, typename = std::enable_if<std::is_convertible<V, bool>::value>>
   TVec operator[](const TVec<V> &conds) const
   {
      const size_type n = conds.size();

      if (n != size())
         throw std::runtime_error("Cannot index TVec with condition vector of different size");

      TVec<T> ret;
      ret.reserve(n);
      for (size_type i = 0; i < n; ++i)
         if (conds[i])
            ret.emplace_back(fData[i]);
      return ret;
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
   reverse_iterator rbegin() noexcept { return fData.rbegin(); }
   const_reverse_iterator rbegin() const noexcept { return fData.rbegin(); }
   const_reverse_iterator crbegin() const noexcept { return fData.crbegin(); }
   reverse_iterator rend() noexcept { return fData.rend(); }
   const_reverse_iterator rend() const noexcept { return fData.rend(); }
   const_reverse_iterator crend() const noexcept { return fData.crend(); }
   // capacity
   bool empty() const noexcept { return fData.empty(); }
   size_type size() const noexcept { return fData.size(); }
   size_type max_size() const noexcept { return fData.size(); }
   void reserve(size_type new_cap) { fData.reserve(new_cap); }
   size_type capacity() const noexcept { return fData.capacity(); }
   void shrink_to_fit() { fData.shrink_to_fit(); };
   // modifiers
   void clear() noexcept { fData.clear(); }
   iterator erase(iterator pos) { return fData.erase(pos); }
   iterator erase(iterator first, iterator last) { return fData.erase(first, last); }
   void push_back(T &&value) { fData.push_back(std::forward<T>(value)); }
   template <class... Args>
   reference emplace_back(Args &&... args)
   {
      fData.emplace_back(std::forward<Args>(args)...);
      return fData.back();
   }
   /// This method is intended only for arithmetic types unlike the std::vector
   /// corresponding one which is generic.
   template<typename U = T, typename std::enable_if<std::is_arithmetic<U>::value, int>* = nullptr>
   iterator emplace(const_iterator pos, U value)
   {
      return fData.emplace(pos, value);
   }
   void pop_back() { fData.pop_back(); }
   void resize(size_type count) { fData.resize(count); }
   void resize(size_type count, const value_type &value) { fData.resize(count, value); }
   void swap(TVec<T> &other) { std::swap(fData, other.fData); }
};

///@name TVec Unary Arithmetic Operators
///@{

#define TVEC_UNARY_OPERATOR(OP)                                                \
template <typename T>                                                          \
TVec<T> operator OP(const TVec<T> &v)                                          \
{                                                                              \
   TVec<T> ret(v);                                                             \
   for (auto &x : ret)                                                         \
      x = OP x;                                                                \
return ret;                                                                    \
}                                                                              \

TVEC_UNARY_OPERATOR(+)
TVEC_UNARY_OPERATOR(-)
TVEC_UNARY_OPERATOR(~)
TVEC_UNARY_OPERATOR(!)
#undef TVEC_UNARY_OPERATOR

///@}
///@name TVec Binary Arithmetic Operators
///@{

#define ERROR_MESSAGE(OP) \
 "Cannot call operator " #OP " on vectors of different sizes."

#define TVEC_BINARY_OPERATOR(OP)                                               \
template <typename T0, typename T1>                                            \
auto operator OP(const TVec<T0> &v, const T1 &y)                               \
  -> TVec<decltype(T0() OP T1())>                                              \
{                                                                              \
   TVec<decltype(T0() OP T1())> ret(v.size());                                 \
   auto op = [&y](const T0 &x) { return x OP y; };                             \
   std::transform(v.begin(), v.end(), ret.begin(), op);                        \
   return ret;                                                                 \
}                                                                              \
                                                                               \
template <typename T0, typename T1>                                            \
auto operator OP(const T0 &x, const TVec<T1> &v)                               \
  -> TVec<decltype(T0() OP T1())>                                              \
{                                                                              \
   TVec<decltype(T0() OP T1())> ret(v.size());                                 \
   auto op = [&x](const T1 &y) { return x OP y; };                             \
   std::transform(v.begin(), v.end(), ret.begin(), op);                        \
   return ret;                                                                 \
}                                                                              \
                                                                               \
template <typename T0, typename T1>                                            \
auto operator OP(const TVec<T0> &v0, const TVec<T1> &v1)                       \
  -> TVec<decltype(T0() OP T1())>                                              \
{                                                                              \
   if (v0.size() != v1.size())                                                 \
      throw std::runtime_error(ERROR_MESSAGE(OP));                             \
                                                                               \
   TVec<decltype(T0() OP T1())> ret(v0.size());                                \
   auto op = [](const T0 &x, const T1 &y) { return x OP y; };                  \
   std::transform(v0.begin(), v0.end(), v1.begin(), ret.begin(), op);          \
   return ret;                                                                 \
}                                                                              \

TVEC_BINARY_OPERATOR(+)
TVEC_BINARY_OPERATOR(-)
TVEC_BINARY_OPERATOR(*)
TVEC_BINARY_OPERATOR(/)
TVEC_BINARY_OPERATOR(%)
TVEC_BINARY_OPERATOR(^)
TVEC_BINARY_OPERATOR(|)
TVEC_BINARY_OPERATOR(&)
#undef TVEC_BINARY_OPERATOR

///@}
///@name TVec Assignment Arithmetic Operators
///@{

#define TVEC_ASSIGNMENT_OPERATOR(OP)                                           \
template <typename T0, typename T1>                                            \
TVec<T0>& operator OP(TVec<T0> &v, const T1 &y)                                \
{                                                                              \
   auto op = [&y](T0 &x) { return x OP y; };                                   \
   std::transform(v.begin(), v.end(), v.begin(), op);                          \
   return v;                                                                   \
}                                                                              \
                                                                               \
template <typename T0, typename T1>                                            \
TVec<T0>& operator OP(TVec<T0> &v0, const TVec<T1> &v1)                        \
{                                                                              \
   if (v0.size() != v1.size())                                                 \
      throw std::runtime_error(ERROR_MESSAGE(OP));                             \
                                                                               \
   auto op = [](T0 &x, const T1 &y) { return x OP y; };                        \
   std::transform(v0.begin(), v0.end(), v1.begin(), v0.begin(), op);           \
   return v0;                                                                  \
}                                                                              \

TVEC_ASSIGNMENT_OPERATOR(+=)
TVEC_ASSIGNMENT_OPERATOR(-=)
TVEC_ASSIGNMENT_OPERATOR(*=)
TVEC_ASSIGNMENT_OPERATOR(/=)
TVEC_ASSIGNMENT_OPERATOR(%=)
TVEC_ASSIGNMENT_OPERATOR(^=)
TVEC_ASSIGNMENT_OPERATOR(|=)
TVEC_ASSIGNMENT_OPERATOR(&=)
TVEC_ASSIGNMENT_OPERATOR(>>=)
TVEC_ASSIGNMENT_OPERATOR(<<=)
#undef TVEC_ASSIGNMENT_OPERATOR

///@}
///@name TVec Comparison and Logical Operators
///@{

#define TVEC_LOGICAL_OPERATOR(OP)                                              \
template <typename T0, typename T1>                                            \
auto operator OP(const TVec<T0> &v, const T1 &y)                               \
  -> TVec<int> /* avoid std::vector<bool> */                                   \
{                                                                              \
   TVec<int> ret(v.size());                                                    \
   auto op = [y](const T0 &x) -> int { return x OP y; };                       \
   std::transform(v.begin(), v.end(), ret.begin(), op);                        \
   return ret;                                                                 \
}                                                                              \
                                                                               \
template <typename T0, typename T1>                                            \
auto operator OP(const T0 &x, const TVec<T1> &v)                               \
  -> TVec<int> /* avoid std::vector<bool> */                                   \
{                                                                              \
   TVec<int> ret(v.size());                                                    \
   auto op = [x](const T1 &y) -> int { return x OP y; };                       \
   std::transform(v.begin(), v.end(), ret.begin(), op);                        \
   return ret;                                                                 \
}                                                                              \
                                                                               \
template <typename T0, typename T1>                                            \
auto operator OP(const TVec<T0> &v0, const TVec<T1> &v1)                       \
  -> TVec<int> /* avoid std::vector<bool> */                                   \
{                                                                              \
   if (v0.size() != v1.size())                                                 \
      throw std::runtime_error(ERROR_MESSAGE(OP));                             \
                                                                               \
   TVec<int> ret(v0.size());                                                   \
   auto op = [](const T0 &x, const T1 &y) -> int { return x OP y; };           \
   std::transform(v0.begin(), v0.end(), v1.begin(), ret.begin(), op);          \
   return ret;                                                                 \
}                                                                              \

TVEC_LOGICAL_OPERATOR(<)
TVEC_LOGICAL_OPERATOR(>)
TVEC_LOGICAL_OPERATOR(==)
TVEC_LOGICAL_OPERATOR(!=)
TVEC_LOGICAL_OPERATOR(<=)
TVEC_LOGICAL_OPERATOR(>=)
TVEC_LOGICAL_OPERATOR(&&)
TVEC_LOGICAL_OPERATOR(||)
#undef TVEC_LOGICAL_OPERATOR

///@}
///@name TVec Standard Mathematical Functions
///@{

#define TVEC_UNARY_FUNCTION(NAME, FUNC)                                        \
   template <typename T>                                                       \
   TVec<T> NAME(const TVec<T> &v)                                              \
   {                                                                           \
      TVec<T> ret(v.size());                                                   \
      auto f = [](const T &x) { return FUNC(x); };                             \
      std::transform(v.begin(), v.end(), ret.begin(), f);                      \
      return ret;                                                              \
   }

#define TVEC_BINARY_FUNCTION(NAME, FUNC)                                       \
   template <typename T0, typename T1>                                         \
   TVec<decltype(T0() + T1())> NAME(const T0 &x, const TVec<T1> &v)            \
   {                                                                           \
      TVec<decltype(T0() + T1())> ret(v.size());                               \
      auto f = [&x](const T1 &y) { return FUNC(x, y); };                       \
      std::transform(v.begin(), v.end(), ret.begin(), f);                      \
      return ret;                                                              \
   }                                                                           \
                                                                               \
   template <typename T0, typename T1>                                         \
   TVec<decltype(T0() + T1())> NAME(const TVec<T0> &v, const T1 &y)            \
   {                                                                           \
      TVec<decltype(T0() + T1())> ret(v.size());                               \
      auto f = [&y](const T1 &x) { return FUNC(x, y); };                       \
      std::transform(v.begin(), v.end(), ret.begin(), f);                      \
      return ret;                                                              \
   }                                                                           \
                                                                               \
   template <typename T0, typename T1>                                         \
   TVec<decltype(T0() + T1())> NAME(const TVec<T0> &v0, const TVec<T1> &v1)    \
   {                                                                           \
      if (v0.size() != v1.size())                                              \
         throw std::runtime_error(ERROR_MESSAGE(NAME));                        \
                                                                               \
      TVec<decltype(T0() + T1())> ret(v0.size());                              \
      auto f = [](const T0 &x, const T1 &y) { return FUNC(x, y); };            \
      std::transform(v0.begin(), v0.end(), v1.begin(), ret.begin(), f);        \
      return ret;                                                              \
   }                                                                           \

#define TVEC_STD_UNARY_FUNCTION(F) TVEC_UNARY_FUNCTION(F, std::F)
#define TVEC_STD_BINARY_FUNCTION(F) TVEC_BINARY_FUNCTION(F, std::F)

TVEC_STD_UNARY_FUNCTION(abs)
TVEC_STD_BINARY_FUNCTION(fdim)
TVEC_STD_BINARY_FUNCTION(fmod)
TVEC_STD_BINARY_FUNCTION(remainder)

TVEC_STD_UNARY_FUNCTION(exp)
TVEC_STD_UNARY_FUNCTION(exp2)
TVEC_STD_UNARY_FUNCTION(expm1)

TVEC_STD_UNARY_FUNCTION(log)
TVEC_STD_UNARY_FUNCTION(log10)
TVEC_STD_UNARY_FUNCTION(log2)
TVEC_STD_UNARY_FUNCTION(log1p)

TVEC_STD_BINARY_FUNCTION(pow)
TVEC_STD_UNARY_FUNCTION(sqrt)
TVEC_STD_UNARY_FUNCTION(cbrt)
TVEC_STD_BINARY_FUNCTION(hypot)

TVEC_STD_UNARY_FUNCTION(sin)
TVEC_STD_UNARY_FUNCTION(cos)
TVEC_STD_UNARY_FUNCTION(tan)
TVEC_STD_UNARY_FUNCTION(asin)
TVEC_STD_UNARY_FUNCTION(acos)
TVEC_STD_UNARY_FUNCTION(atan)
TVEC_STD_BINARY_FUNCTION(atan2)

TVEC_STD_UNARY_FUNCTION(sinh)
TVEC_STD_UNARY_FUNCTION(cosh)
TVEC_STD_UNARY_FUNCTION(tanh)
TVEC_STD_UNARY_FUNCTION(asinh)
TVEC_STD_UNARY_FUNCTION(acosh)
TVEC_STD_UNARY_FUNCTION(atanh)

TVEC_STD_UNARY_FUNCTION(floor)
TVEC_STD_UNARY_FUNCTION(ceil)
TVEC_STD_UNARY_FUNCTION(trunc)
TVEC_STD_UNARY_FUNCTION(round)
TVEC_STD_UNARY_FUNCTION(lround)
TVEC_STD_UNARY_FUNCTION(llround)

TVEC_STD_UNARY_FUNCTION(erf)
TVEC_STD_UNARY_FUNCTION(erfc)
TVEC_STD_UNARY_FUNCTION(lgamma)
TVEC_STD_UNARY_FUNCTION(tgamma)
#undef TVEC_STD_UNARY_FUNCTION

///@}
///@name TVec Fast Mathematical Functions with Vdt
///@{

#ifdef R__HAS_VDT
#define TVEC_VDT_UNARY_FUNCTION(F) TVEC_UNARY_FUNCTION(F, vdt::F)

TVEC_VDT_UNARY_FUNCTION(fast_expf)
TVEC_VDT_UNARY_FUNCTION(fast_logf)
TVEC_VDT_UNARY_FUNCTION(fast_sinf)
TVEC_VDT_UNARY_FUNCTION(fast_cosf)
TVEC_VDT_UNARY_FUNCTION(fast_tanf)
TVEC_VDT_UNARY_FUNCTION(fast_asinf)
TVEC_VDT_UNARY_FUNCTION(fast_acosf)
TVEC_VDT_UNARY_FUNCTION(fast_atanf)

TVEC_VDT_UNARY_FUNCTION(fast_exp)
TVEC_VDT_UNARY_FUNCTION(fast_log)
TVEC_VDT_UNARY_FUNCTION(fast_sin)
TVEC_VDT_UNARY_FUNCTION(fast_cos)
TVEC_VDT_UNARY_FUNCTION(fast_tan)
TVEC_VDT_UNARY_FUNCTION(fast_asin)
TVEC_VDT_UNARY_FUNCTION(fast_acos)
TVEC_VDT_UNARY_FUNCTION(fast_atan)
#undef TVEC_VDT_UNARY_FUNCTION

#endif // R__HAS_VDT

#undef TVEC_UNARY_FUNCTION

///@}

/// Inner product
template <typename T, typename V>
auto Dot(const TVec<T> &v0, const TVec<V> &v1) -> decltype(v0[0] * v1[0])
{
   if (v0.size() != v1.size())
      throw std::runtime_error("Cannot compute inner product of vectors of different sizes");
   return std::inner_product(v0.begin(), v0.end(), v1.begin(), decltype(v0[0] * v1[0])(0));
}

/// Sum elements
template <typename T>
T Sum(const TVec<T> &v)
{
   return std::accumulate(v.begin(), v.end(), T(0));
}

/// Get Mean
template <typename T>
double Mean(const TVec<T> &v)
{
   if (v.empty()) return 0.;
   return double(Sum(v)) / v.size();
}

/// Get variance
template <typename T>
double Var(const TVec<T> &v)
{
   const std::size_t size = v.size();
   if (size < std::size_t(2)) return 0.;
   T sum_squares(0), squared_sum(0);
   auto pred = [&sum_squares, &squared_sum](const T& x) {sum_squares+=x*x; squared_sum+=x;};
   std::for_each(v.begin(), v.end(), pred);
   squared_sum *= squared_sum;
   const auto dsize = (double) size;
   return 1. / (dsize - 1.) * (sum_squares - squared_sum / dsize );
}

/// Get standard deviation
template <typename T>
double StdDev(const TVec<T> &v)
{
   return std::sqrt(Var(v));
}

/// Create new collection applying a callable to the elements of the input collection
template <typename T, typename F>
auto Map(const TVec<T> &v, F &&f) -> TVec<decltype(f(v[0]))>
{
   TVec<decltype(f(v[0]))> ret(v.size());
   std::transform(v.begin(), v.end(), ret.begin(), f);
   return ret;
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

#if(_VECOPS_USE_EXTERN_TEMPLATES)

#define TVEC_EXTERN_UNARY_OPERATOR(T, OP) \
   extern template TVec<T> operator OP<T>(const TVec<T> &);

#define TVEC_EXTERN_BINARY_OPERATOR(T, OP)                                                       \
   extern template TVec<decltype((T){} OP (T){})> operator OP<T, T>(const T &, const TVec<T> &); \
   extern template TVec<decltype((T){} OP (T){})> operator OP<T, T>(const TVec<T> &, const T &); \
   extern template TVec<decltype((T){} OP (T){})> operator OP<T, T>(const TVec<T> &, const TVec<T> &);

#define TVEC_EXTERN_ASSIGN_OPERATOR(T, OP)                           \
   extern template TVec<T> &operator OP<T, T>(TVec<T> &, const T &); \
   extern template TVec<T> &operator OP<T, T>(TVec<T> &, const TVec<T> &);

#define TVEC_EXTERN_LOGICAL_OPERATOR(T, OP)                                 \
   extern template TVec<int> operator OP<T, T>(const TVec<T> &, const T &); \
   extern template TVec<int> operator OP<T, T>(const T &, const TVec<T> &); \
   extern template TVec<int> operator OP<T, T>(const TVec<T> &, const TVec<T> &);

#define TVEC_EXTERN_FLOAT_TEMPLATE(T)   \
   extern template class TVec<T>;       \
   TVEC_EXTERN_UNARY_OPERATOR(T, +)     \
   TVEC_EXTERN_UNARY_OPERATOR(T, -)     \
   TVEC_EXTERN_UNARY_OPERATOR(T, !)     \
   TVEC_EXTERN_BINARY_OPERATOR(T, +)    \
   TVEC_EXTERN_BINARY_OPERATOR(T, -)    \
   TVEC_EXTERN_BINARY_OPERATOR(T, *)    \
   TVEC_EXTERN_BINARY_OPERATOR(T, /)    \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, +=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, -=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, *=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, /=)   \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, <)   \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, >)   \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, ==)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, !=)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, <=)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, >=)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, &&)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, ||)

#define TVEC_EXTERN_INTEGER_TEMPLATE(T) \
   extern template class TVec<T>;       \
   TVEC_EXTERN_UNARY_OPERATOR(T, +)     \
   TVEC_EXTERN_UNARY_OPERATOR(T, -)     \
   TVEC_EXTERN_UNARY_OPERATOR(T, ~)     \
   TVEC_EXTERN_UNARY_OPERATOR(T, !)     \
   TVEC_EXTERN_BINARY_OPERATOR(T, +)    \
   TVEC_EXTERN_BINARY_OPERATOR(T, -)    \
   TVEC_EXTERN_BINARY_OPERATOR(T, *)    \
   TVEC_EXTERN_BINARY_OPERATOR(T, /)    \
   TVEC_EXTERN_BINARY_OPERATOR(T, %)    \
   TVEC_EXTERN_BINARY_OPERATOR(T, &)    \
   TVEC_EXTERN_BINARY_OPERATOR(T, |)    \
   TVEC_EXTERN_BINARY_OPERATOR(T, ^)    \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, +=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, -=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, *=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, /=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, %=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, &=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, |=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, ^=)   \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, >>=)  \
   TVEC_EXTERN_ASSIGN_OPERATOR(T, <<=)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, <)   \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, >)   \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, ==)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, !=)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, <=)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, >=)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, &&)  \
   TVEC_EXTERN_LOGICAL_OPERATOR(T, ||)

TVEC_EXTERN_INTEGER_TEMPLATE(char)
TVEC_EXTERN_INTEGER_TEMPLATE(short)
TVEC_EXTERN_INTEGER_TEMPLATE(int)
TVEC_EXTERN_INTEGER_TEMPLATE(long)
//TVEC_EXTERN_INTEGER_TEMPLATE(long long)

TVEC_EXTERN_INTEGER_TEMPLATE(unsigned char)
TVEC_EXTERN_INTEGER_TEMPLATE(unsigned short)
TVEC_EXTERN_INTEGER_TEMPLATE(unsigned int)
TVEC_EXTERN_INTEGER_TEMPLATE(unsigned long)
//TVEC_EXTERN_INTEGER_TEMPLATE(unsigned long long)

TVEC_EXTERN_FLOAT_TEMPLATE(float)
TVEC_EXTERN_FLOAT_TEMPLATE(double)

#undef TVEC_EXTERN_UNARY_OPERATOR
#undef TVEC_EXTERN_BINARY_OPERATOR
#undef TVEC_EXTERN_ASSIGN_OPERATOR
#undef TVEC_EXTERN_LOGICAL_OPERATOR
#undef TVEC_EXTERN_INTEGER_TEMPLATE
#undef TVEC_EXTERN_FLOAT_TEMPLATE

#define TVEC_EXTERN_UNARY_FUNCTION(T, NAME, FUNC) \
   extern template TVec<T> NAME(const TVec<T> &);

#define TVEC_EXTERN_STD_UNARY_FUNCTION(T, F) TVEC_EXTERN_UNARY_FUNCTION(T, F, std::F)

#define TVEC_EXTERN_BINARY_FUNCTION(T0, T1, NAME, FUNC)                            \
   extern template TVec<decltype(T0() + T1())> NAME(const TVec<T0> &, const T1 &); \
   extern template TVec<decltype(T0() + T1())> NAME(const T0 &, const TVec<T1> &); \
   extern template TVec<decltype(T0() + T1())> NAME(const TVec<T0> &, const TVec<T1> &);

#define TVEC_EXTERN_STD_BINARY_FUNCTION(T, F) TVEC_EXTERN_BINARY_FUNCTION(T, T, F, std::F)

#define TVEC_EXTERN_STD_FUNCTIONS(T)             \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, abs)        \
   TVEC_EXTERN_STD_BINARY_FUNCTION(T, fdim)      \
   TVEC_EXTERN_STD_BINARY_FUNCTION(T, fmod)      \
   TVEC_EXTERN_STD_BINARY_FUNCTION(T, remainder) \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, exp)        \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, exp2)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, expm1)      \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, log)        \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, log10)      \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, log2)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, log1p)      \
   TVEC_EXTERN_STD_BINARY_FUNCTION(T, pow)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, sqrt)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, cbrt)       \
   TVEC_EXTERN_STD_BINARY_FUNCTION(T, hypot)     \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, sin)        \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, cos)        \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, tan)        \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, asin)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, acos)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, atan)       \
   TVEC_EXTERN_STD_BINARY_FUNCTION(T, atan2)     \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, sinh)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, cosh)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, tanh)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, asinh)      \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, acosh)      \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, atanh)      \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, floor)      \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, ceil)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, trunc)      \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, round)      \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, erf)        \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, erfc)       \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, lgamma)     \
   TVEC_EXTERN_STD_UNARY_FUNCTION(T, tgamma)     \

TVEC_EXTERN_STD_FUNCTIONS(float)
TVEC_EXTERN_STD_FUNCTIONS(double)
#undef TVEC_EXTERN_STD_UNARY_FUNCTION
#undef TVEC_EXTERN_STD_BINARY_FUNCTION
#undef TVEC_EXTERN_STD_UNARY_FUNCTIONS

#ifdef R__HAS_VDT

#define TVEC_EXTERN_VDT_UNARY_FUNCTION(T, F) TVEC_EXTERN_UNARY_FUNCTION(T, F, vdt::F)

TVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_expf)
TVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_logf)
TVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_sinf)
TVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_cosf)
TVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_tanf)
TVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_asinf)
TVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_acosf)
TVEC_EXTERN_VDT_UNARY_FUNCTION(float, fast_atanf)

TVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_exp)
TVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_log)
TVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_sin)
TVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_cos)
TVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_tan)
TVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_asin)
TVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_acos)
TVEC_EXTERN_VDT_UNARY_FUNCTION(double, fast_atan)

#endif // R__HAS_VDT

#endif // _VECOPS_USE_EXTERN_TEMPLATES

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

#undef _VECOPS_USE_EXTERN_TEMPLATES

#endif
