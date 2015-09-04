/// \file rhysd_array_view.h
/// \ingroup Base StdExt
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-06

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RHYSD_ARRAY_VIEW_H
#define ROOT7_RHYSD_ARRAY_VIEW_H

// From https://github.com/rhysd/array_view/blob/master/include/array_view.hpp

#include <cstddef>
#include <iterator>
#include <array>
#include <vector>
#include <stdexcept>
#include <memory>
#include <type_traits>
#include <vector>
#include <initializer_list>

namespace std {

using std::size_t;

namespace detail {
// detail meta functions {{{
template<class Array>
struct is_array_class {
  static bool const value = false;
};
template<class T, size_t N>
struct is_array_class<std::array<T, N>> {
  static bool const value = true;
};
template<class T>
struct is_array_class<std::vector<T>> {
  static bool const value = true;
};
template<class T>
struct is_array_class<std::initializer_list<T>> {
  static bool const value = true;
};
// }}}

// index sequences {{{
template< size_t... Indices >
struct indices{
  static constexpr size_t value[sizeof...(Indices)] = {Indices...};
};

template<class IndicesType, size_t Next>
struct make_indices_next;

template<size_t... Indices, size_t Next>
struct make_indices_next<indices<Indices...>, Next> {
  typedef indices<Indices..., (Indices + Next)...> type;
};

template<class IndicesType, size_t Next, size_t Tail>
struct make_indices_next2;

template<size_t... Indices, size_t Next, size_t Tail>
struct make_indices_next2<indices<Indices...>, Next, Tail> {
  typedef indices<Indices..., (Indices + Next)..., Tail> type;
};

template<size_t First, size_t Step, size_t N, class = void>
struct make_indices_impl;

template<size_t First, size_t Step, size_t N>
struct make_indices_impl<
   First,
   Step,
   N,
   typename std::enable_if<(N == 0)>::type
> {
  typedef indices<> type;
};

template<size_t First, size_t Step, size_t N>
struct make_indices_impl<
   First,
   Step,
   N,
   typename std::enable_if<(N == 1)>::type
> {
  typedef indices<First> type;
};

template<size_t First, size_t Step, size_t N>
struct make_indices_impl<
   First,
   Step,
   N,
   typename std::enable_if<(N > 1 && N % 2 == 0)>::type
>
   : detail::make_indices_next<
      typename detail::make_indices_impl<First, Step, N / 2>::type,
      First + N / 2 * Step
   >
{};

template<size_t First, size_t Step, size_t N>
struct make_indices_impl<
   First,
   Step,
   N,
   typename std::enable_if<(N > 1 && N % 2 == 1)>::type
>
   : detail::make_indices_next2<
      typename detail::make_indices_impl<First, Step, N / 2>::type,
      First + N / 2 * Step,
      First + (N - 1) * Step
   >
{};

template<size_t First, size_t Last, size_t Step = 1>
struct make_indices_
   : detail::make_indices_impl<
      First,
      Step,
      ((Last - First) + (Step - 1)) / Step
   >
{};

template < size_t Start, size_t Last, size_t Step = 1 >
using make_indices = typename make_indices_< Start, Last, Step >::type;
// }}}
} // namespace detail

// array_view {{{

struct check_bound_t {};
static constexpr check_bound_t check_bound{};

template<class T>
class array_view {
public:
  /*
   * types
   */
  typedef T value_type;
  typedef value_type const* pointer;
  typedef value_type const* const_pointer;
  typedef value_type const& reference;
  typedef value_type const& const_reference;
  typedef value_type const* iterator;
  typedef value_type const* const_iterator;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  /*
   * ctors and assign operators
   */
  constexpr array_view() noexcept
     : length_(0), data_(nullptr)
  {}

  constexpr array_view(array_view const&) noexcept = default;
  constexpr array_view(array_view &&) noexcept = default;

  // Note:
  // This constructor can't be constexpr because & operator can't be constexpr.
  template<size_type N>
  /*implicit*/ array_view(std::array<T, N> const& a) noexcept
     : length_(N), data_(N > 0 ? a.data() : nullptr)
  {}

  // Note:
  // This constructor can't be constexpr because & operator can't be constexpr.
  template<size_type N>
  /*implicit*/ array_view(T const (& a)[N]) noexcept
     : length_(N), data_(N > 0 ? std::addressof(a[0]) : nullptr)
  {
    static_assert(N > 0, "Zero-length array is not permitted in ISO C++.");
  }

  /*implicit*/ array_view(std::vector<T> const& v) noexcept
     : length_(v.size()), data_(v.empty() ? nullptr : v.data())
  {}

  /*implicit*/ constexpr array_view(T const* a, size_type const n) noexcept
     : length_(n), data_(a)
  {}

  template<
     class InputIterator,
     class = typename std::enable_if<
        std::is_same<
           T,
           typename std::iterator_traits<InputIterator>::value_type
        >::value
     >::type
  >
  explicit array_view(InputIterator start, InputIterator last)
     : length_(std::distance(start, last)), data_(start)
  {}

  array_view(std::initializer_list<T> const& l)
     : length_(l.size()), data_(std::begin(l))
  {}

  array_view& operator=(array_view const&) noexcept = delete;
  array_view& operator=(array_view &&) noexcept = delete;

  /*
   * iterator interfaces
   */
  constexpr const_iterator begin() const noexcept
  {
    return data_;
  }
  constexpr const_iterator end() const noexcept
  {
    return data_ + length_;
  }
  constexpr const_iterator cbegin() const noexcept
  {
    return begin();
  }
  constexpr const_iterator cend() const noexcept
  {
    return end();
  }
  const_reverse_iterator rbegin() const
  {
    return {end()};
  }
  const_reverse_iterator rend() const
  {
    return {begin()};
  }
  const_reverse_iterator crbegin() const
  {
    return rbegin();
  }
  const_reverse_iterator crend() const
  {
    return rend();
  }

  /*
   * access
   */
  constexpr size_type size() const noexcept
  {
    return length_;
  }
  constexpr size_type length() const noexcept
  {
    return size();
  }
  constexpr size_type max_size() const noexcept
  {
    return size();
  }
  constexpr bool empty() const noexcept
  {
    return length_ == 0;
  }
  constexpr const_reference operator[](size_type const n) const noexcept
  {
    return *(data_ + n);
  }
  constexpr const_reference at(size_type const n) const
  {
    if (n >= length_) throw std::out_of_range("array_view::at()");
    return *(data_ + n);
  }
  constexpr const_pointer data() const noexcept
  {
    return data_;
  }
  constexpr const_reference front() const noexcept
  {
    return *data_;
  }
  constexpr const_reference back() const noexcept
  {
    return *(data_ + length_ - 1);
  }

  /*
   * slices
   */
  // slice with indices {{{
  // check bound {{{
  constexpr array_view<T> slice(check_bound_t, size_type const pos, size_type const length) const
  {
    if (pos >= length_ || pos + length >= length_) {
      throw std::out_of_range("array_view::slice()");
    }
    return array_view<T>{begin() + pos, begin() + pos + length};
  }
  constexpr array_view<T> slice_before(check_bound_t, size_type const pos) const
  {
    if (pos >= length_) {
      throw std::out_of_range("array_view::slice()");
    }
    return array_view<T>{begin(), begin() + pos};
  }
  constexpr array_view<T> slice_after(check_bound_t, size_type const pos) const
  {
    if (pos >= length_) {
      throw std::out_of_range("array_view::slice()");
    }
    return array_view<T>{begin() + pos, end()};
  }
  // }}}
  // not check bound {{{
  constexpr array_view<T> slice(size_type const pos, size_type const length) const
  {
    return array_view<T>{begin() + pos, begin() + pos + length};
  }
  constexpr array_view<T> slice_before(size_type const pos) const
  {
    return array_view<T>{begin(), begin() + pos};
  }
  constexpr array_view<T> slice_after(size_type const pos) const
  {
    return array_view<T>{begin() + pos, end()};
  }
  // }}}
  // }}}
  // slice with iterators {{{
  // check bound {{{
  constexpr array_view<T> slice(check_bound_t, iterator start, iterator last) const
  {
    if ( start >= end() ||
         last > end() ||
         start > last ||
         static_cast<size_t>(std::distance(start, last > end() ? end() : last)) > length_ - std::distance(begin(), start) ) {
      throw std::out_of_range("array_view::slice()");
    }
    return array_view<T>{start, last > end() ? end() : last};
  }
  constexpr array_view<T> slice_before(check_bound_t, iterator const pos) const
  {
    if (pos < begin() || pos > end()) {
      throw std::out_of_range("array_view::slice()");
    }
    return array_view<T>{begin(), pos > end() ? end() : pos};
  }
  constexpr array_view<T> slice_after(check_bound_t, iterator const pos) const
  {
    if (pos < begin() || pos > end()) {
      throw std::out_of_range("array_view::slice()");
    }
    return array_view<T>{pos < begin() ? begin() : pos, end()};
  }
  // }}}
  // not check bound {{{
  constexpr array_view<T> slice(iterator start, iterator last) const
  {
    return array_view<T>{start, last};
  }
  constexpr array_view<T> slice_before(iterator const pos) const
  {
    return array_view<T>{begin(), pos};
  }
  constexpr array_view<T> slice_after(iterator const pos) const
  {
    return array_view<T>{pos, end()};
  }
  // }}}
  // }}}

  /*
   * others
   */
  template<class Allocator = std::allocator<T>>
  auto to_vector(Allocator const& alloc = Allocator{}) const
  -> std::vector<T, Allocator>
  {
    return {begin(), end(), alloc};
  }

  template<size_t N>
  auto to_array() const
  -> std::array<T, N>
  {
    return to_array_impl(detail::make_indices<0, N>{});
  }
private:
  template<size_t... I>
  auto to_array_impl(detail::indices<I...>) const
  -> std::array<T, sizeof...(I)>
  {
    return {{(I < length_ ? *(data_ + I) : T{} )...}};
  }

private:
  size_type const length_;
  const_pointer const data_;
};
// }}}

// compare operators {{{
namespace detail {
template< class ArrayL, class ArrayR >
inline constexpr
bool operator_equal_impl(ArrayL const& lhs, size_t const lhs_size, ArrayR const& rhs, size_t const rhs_size)
{
  if (lhs_size != rhs_size) {
    return false;
  }

  auto litr = std::begin(lhs);
  auto ritr = std::begin(rhs);
  for (; litr != std::end(lhs); ++litr, ++ritr) {
    if (!(*litr == *ritr)) {
      return false;
    }
  }

  return true;
}
} // namespace detail

template<class T1, class T2>
inline constexpr
bool operator==(array_view<T1> const& lhs, array_view<T2> const& rhs)
{
  return detail::operator_equal_impl(lhs, lhs.length(), rhs, rhs.length());
}

template<
   class T,
   class Array,
   class = typename std::enable_if<
      detail::is_array_class<Array>::value
   >::type
>
inline constexpr
bool operator==(array_view<T> const& lhs, Array const& rhs)
{
  return detail::operator_equal_impl(lhs, lhs.length(), rhs, rhs.size());
}

template<class T1, class T2, size_t N>
inline constexpr
bool operator==(array_view<T1> const& lhs, T2 const (& rhs)[N])
{
  return detail::operator_equal_impl(lhs, lhs.length(), rhs, N);
}

template<
   class T,
   class Array,
   class = typename std::enable_if<
      is_array<Array>::value
   >::type
>
inline constexpr
bool operator!=(array_view<T> const& lhs, Array const& rhs)
{
  return !(lhs == rhs);
}

template<
   class Array,
   class T,
   class = typename std::enable_if<
      is_array<Array>::value
   >::type
>
inline constexpr
bool operator==(Array const& lhs, array_view<T> const& rhs)
{
  return rhs == lhs;
}

template<
   class Array,
   class T,
   class = typename std::enable_if<
      is_array<Array>::value,
      Array
   >::type
>
inline constexpr
bool operator!=(Array const& lhs, array_view<T> const& rhs)
{
  return !(rhs == lhs);
}
// }}}

// helpers to construct view {{{
template<
   class Array,
   class = typename std::enable_if<
      detail::is_array_class<Array>::value
   >::type
>
inline constexpr
auto make_view(Array const& a)
-> array_view<typename Array::value_type>
{
  return {a};
}

template< class T, size_t N>
inline constexpr
array_view<T> make_view(T const (&a)[N])
{
  return {a};
}

template<class T>
inline constexpr
array_view<T> make_view(T const* p, typename array_view<T>::size_type const n)
{
  return array_view<T>{p, n};
}

template<class InputIterator, class Result = array_view<typename std::iterator_traits<InputIterator>::value_type>>
inline constexpr
Result make_view(InputIterator begin, InputIterator end)
{
  return Result{begin, end};
}

template<class T>
inline constexpr
array_view<T> make_view(std::initializer_list<T> const& l)
{
  return {l};
}
// }}}

} // namespace std





#if 0
// This stuff is too complex for our simple use case!

#include <cstddef>
#include <array>
#include <type_traits>

// See N3851

namespace std {

template<int Rank>
class index;

template<int Rank>
class bounds {
public:
  static constexpr int rank = Rank;
  using reference = ptrdiff_t &;
  using const_reference = const ptrdiff_t &;
  using size_type = size_t;
  using value_type = ptrdiff_t;

private:
  std::array<value_type, Rank> m_B;

public:
  constexpr bounds() noexcept;

  constexpr bounds(value_type b) noexcept: m_B{{b}} { };
  //constexpr bounds(const initializer_list<value_type>&) noexcept;
  //constexpr bounds(const bounds&) noexcept;
  //bounds& operator=(const bounds&) noexcept;

  reference operator[](size_type i) noexcept { return m_B[i]; }

  constexpr const_reference operator[](
     size_type i) const noexcept { return m_B[i]; };


  bool operator==(const bounds &rhs) const noexcept;

  bool operator!=(const bounds &rhs) const noexcept;

  bounds operator+(const index<rank> &rhs) const noexcept;

  bounds operator-(const index<rank> &rhs) const noexcept;

  bounds &operator+=(const index<rank> &rhs) noexcept;

  bounds &operator-=(const index<rank> &rhs) noexcept;

  constexpr size_type size() const noexcept;

  bool contains(const index<rank> &idx) const noexcept;
  //bounds_iterator<rank> begin() const noexcept;
  //bounds_iterator<rank> end() const noexcept;

};

//bounds operator+(const index<rank>& lhs, const bounds& rhs) noexcept;

template<int Rank>
class index {
public:
  static constexpr int rank = Rank;
  using reference = ptrdiff_t &;
  using const_reference = const ptrdiff_t &;
  using size_type = size_t;
  using value_type = ptrdiff_t;

// For index<rank>:
  constexpr index() noexcept;

  constexpr index(value_type) noexcept;

  constexpr index(const initializer_list<value_type> &) noexcept;

  constexpr index(const index &) noexcept;

  index &operator=(const index &) noexcept;

  reference operator[](size_type component_idx) noexcept;

  constexpr const_reference operator[](size_type component_idx) const noexcept;

  bool operator==(const index &rhs) const noexcept;

  bool operator!=(const index &rhs) const noexcept;

  index operator+(const index &rhs) const noexcept;

  index operator-(const index &rhs) const noexcept;

  index &operator+=(const index &rhs) noexcept;

  index &operator-=(const index &rhs) noexcept;

  index &operator++() noexcept;

  index operator++(int) noexcept;

  index &operator--() noexcept;

  index operator--(int) noexcept;

  index operator+() const noexcept;

  index operator-() const noexcept;
};

/// Mock-up of future atd::(experimental::)array_view.
/// Supports only what we need for THist, e.g. Rank := 1.
template<typename ValueType, int Rank = 1>
class array_view {
public:
  static constexpr int rank = Rank;
  using index_type = index<rank>;
  using bounds_type = bounds<rank>;
  using size_type = typename bounds_type::size_type;
  using value_type = ValueType;
  using pointer = typename std::add_pointer_t<value_type>;
  using reference = typename std::add_lvalue_reference_t<value_type>;

  constexpr array_view() noexcept;

  constexpr explicit array_view(std::vector<ValueType> &cont) noexcept;

  template<typename ArrayType>
  constexpr explicit array_view(ArrayType &data) noexcept;

  template<typename ViewValueType>
  constexpr array_view(const array_view<ViewValueType, rank> &rhs) noexcept;

  template<typename Container>
  constexpr array_view(bounds_type bounds, Container &cont) noexcept;

  constexpr array_view(bounds_type bounds, pointer data) noexcept;

  template<typename ViewValueType>
  array_view &operator=(const array_view<ViewValueType, rank> &rhs) noexcept;

  constexpr bounds_type bounds() const noexcept;
  constexpr size_type size() const noexcept;
  constexpr index_type stride() const noexcept;

  constexpr pointer data() const noexcept;
  constexpr reference operator[](const index_type& idx) const noexcept;
};

}
#endif // too complex!
#endif
