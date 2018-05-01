/// \file ROOT/RIntegerSequence.hxx
/// \ingroup Base StdExt
/// \author Enrico Guiraud <enrico.guiraud@cern.ch>
/// \date 2018-04-06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RIntegerSequence
#define ROOT_RIntegerSequence

#include <RConfigure.h>
#include "ROOT/RSpan.hxx"

#ifdef R__HAS_STD_INDEX_SEQUENCE
// no need to backport

#include <utility>

#else
// backport libc++'s implementation (master branch, commit ece1de8)

#include <cstddef>
#include <type_traits>

namespace ROOT {
namespace Detail {

template <class _IdxType, _IdxType... _Values>
struct __integer_sequence {
  template <template <class _OIdxType, _OIdxType...> class _ToIndexSeq, class _ToIndexType>
  using __convert = _ToIndexSeq<_ToIndexType, _Values...>;
};

template<typename _Tp, size_t ..._Extra>
struct __repeat;

template<typename _Tp, _Tp ..._Np, size_t ..._Extra>
struct __repeat<__integer_sequence<_Tp, _Np...>, _Extra...> {
  typedef __integer_sequence<_Tp,
                           _Np...,
                           sizeof...(_Np) + _Np...,
                           2 * sizeof...(_Np) + _Np...,
                           3 * sizeof...(_Np) + _Np...,
                           4 * sizeof...(_Np) + _Np...,
                           5 * sizeof...(_Np) + _Np...,
                           6 * sizeof...(_Np) + _Np...,
                           7 * sizeof...(_Np) + _Np...,
                           _Extra...> type;
};

template<size_t _Np> struct __parity;
template<size_t _Np> struct __make : __parity<_Np % 8>::template __pmake<_Np> {};

template<> struct __make<0> { typedef __integer_sequence<size_t> type; };
template<> struct __make<1> { typedef __integer_sequence<size_t, 0> type; };
template<> struct __make<2> { typedef __integer_sequence<size_t, 0, 1> type; };
template<> struct __make<3> { typedef __integer_sequence<size_t, 0, 1, 2> type; };
template<> struct __make<4> { typedef __integer_sequence<size_t, 0, 1, 2, 3> type; };
template<> struct __make<5> { typedef __integer_sequence<size_t, 0, 1, 2, 3, 4> type; };
template<> struct __make<6> { typedef __integer_sequence<size_t, 0, 1, 2, 3, 4, 5> type; };
template<> struct __make<7> { typedef __integer_sequence<size_t, 0, 1, 2, 3, 4, 5, 6> type; };

template <>
struct __parity<0> {
   template <size_t _Np>
   struct __pmake : __repeat<typename __make<_Np / 8>::type> {
   };
};
template<> struct __parity<1> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 1> {}; };
template<> struct __parity<2> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 2, _Np - 1> {}; };
template<> struct __parity<3> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 3, _Np - 2, _Np - 1> {}; };
template<> struct __parity<4> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 4, _Np - 3, _Np - 2, _Np - 1> {}; };
template<> struct __parity<5> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 5, _Np - 4, _Np - 3, _Np - 2, _Np - 1> {}; };
template<> struct __parity<6> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 6, _Np - 5, _Np - 4, _Np - 3, _Np - 2, _Np - 1> {}; };
template<> struct __parity<7> { template<size_t _Np> struct __pmake : __repeat<typename __make<_Np / 8>::type, _Np - 7, _Np - 6, _Np - 5, _Np - 4, _Np - 3, _Np - 2, _Np - 1> {}; };

} // namespace Detail
} // namespace ROOT

namespace std {

template <class _Tp, _Tp... _Ip>
struct integer_sequence {
   typedef _Tp value_type;
   static_assert(is_integral<_Tp>::value, "std::integer_sequence can only be instantiated with an integral type");
   static constexpr size_t size() noexcept { return sizeof...(_Ip); }
};

template <size_t... _Ip>
using index_sequence = integer_sequence<size_t, _Ip...>;

template <typename _Tp, _Tp _Np>
using __make_integer_sequence_unchecked =
   typename ROOT::Detail::__make<_Np>::type::template __convert<integer_sequence, _Tp>;

template <class _Tp, _Tp _Ep>
struct __make_integer_sequence_checked {
   static_assert(is_integral<_Tp>::value, "std::make_integer_sequence can only be instantiated with an integral type");
   static_assert(0 <= _Ep, "std::make_integer_sequence must have a non-negative sequence length");
   // Workaround GCC bug by preventing bad installations when 0 <= _Ep
   // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=68929
   typedef __make_integer_sequence_unchecked<_Tp, 0 <= _Ep ? _Ep : 0> type;
};

template <class _Tp, _Tp _Ep>
using __make_integer_sequence = typename __make_integer_sequence_checked<_Tp, _Ep>::type;

template <class _Tp, _Tp _Np>
using make_integer_sequence = __make_integer_sequence<_Tp, _Np>;

template <size_t _Np>
using make_index_sequence = make_integer_sequence<size_t, _Np>;

template <class... _Tp>
using index_sequence_for = make_index_sequence<sizeof...(_Tp)>;

} // namespace std

#endif // R__HAS_STD_INDEX_SEQUENCE
#endif // ROOT_RIntegerSequence
