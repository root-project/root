// -*- C++ -*-
// Author: Philippe Canal, March 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//===---------------------wrap string_view ----------------------------===//
// Wrapper header adapting the snapshot of sring_view.h to build without
// the libc++ infrastructure header files.

#ifndef RWrap_libcpp_string_view_h
#define RWrap_libcpp_string_view_h

// In case we are connected with a libc++ which defines those, we need
// to include it first, so we avoid being silently over-ridden.

// To import a new version of the original file do:
//
/*
 cat original/string_view | \
 sed -e 's:_LIBCPP_BEGIN_NAMESPACE_LFTS:_ROOT_LIBCPP_BEGIN_NAMESPACE_LFTS:' \
     -e 's:_LIBCPP_END_NAMESPACE_LFTS:_ROOT_LIBCPP_END_NAMESPACE_LFTS:' \
     -e 's:#include <__debug>://#include <__debug>:' \
     -e 's:#include <experimental/__config>://#include <experimental/__config>:' \
     -e 's:__put_character_sequence:R__put_character_sequence:g' \
 > core/base/inc/libcpp_string_view.h
*/

#include <string>
#include <algorithm>
#include <iterator>
#include <ostream>
#include <iomanip>

#define _ROOT_LIBCPP_BEGIN_NAMESPACE_LFTS \
namespace std { \
namespace experimental { inline namespace __ROOT {
#define _ROOT_LIBCPP_END_NAMESPACE_LFTS } } }

#ifndef _LIBCPP_BEGIN_NAMESPACE_STD
#define _LOCAL_LIBCPP_BEGIN_NAMESPACE_STD
#define _LIBCPP_BEGIN_NAMESPACE_STD namespace std {
#define _LIBCPP_END_NAMESPACE_STD }
#endif

#ifndef _LIBCPP_CONSTEXPR
#define _LOCAL_LIBCPP_CONSTEXPR
#define _LIBCPP_CONSTEXPR constexpr
#endif

#ifndef _VSTD
#define _LOCAL_VSTD
#define _VSTD ::std
#endif

#ifndef _LIBCPP_INLINE_VISIBILITY
#define _LOCAL_LIBCPP_INLINE_VISIBILITY
#define _LIBCPP_INLINE_VISIBILITY inline
#endif

#ifndef _LIBCPP_EXPLICIT
#define _LOCAL_LIBCPP_EXPLICIT
#define _LIBCPP_EXPLICIT explicit
#endif

//#ifndef _LIBCPP_CONSTEXPR_AFTER_CXX11
//#define _LOCAL_LIBCPP_CONSTEXPR_AFTER_CXX11
//#define _LIBCPP_CONSTEXPR_AFTER_CXX11 constexpr
//#endif

#ifdef _LIBCPP_STD_VER
#define _LOCAL_LIBCPP_STD_VER
#define _LIBCPP_STD_VER 11
#endif

#ifndef _LIBCPP_TYPE_VIS_ONLY
#define _LOCAL_LIBCPP_TYPE_VIS_ONLY
#define _LIBCPP_TYPE_VIS_ONLY
#endif

#ifndef _LIBCPP_CONSTEXPR_AFTER_CXX11
#define _LOCAL_LIBCPP_CONSTEXPR_AFTER_CXX11
#define _LIBCPP_CONSTEXPR_AFTER_CXX11
#endif

#ifndef _NOEXCEPT
#define _LOCAL_NOEXCEPT
#define _NOEXCEPT
#endif

/* Also used:
 _LIBCPP_TYPE_VIS_ONLY
 _LIBCPP_CONSTEXPR_AFTER_CXX11
 */

namespace std {
inline namespace __1 {

   //   template <class _Traits>
   //   struct _LIBCPP_HIDDEN __traits_eq
   //   {
   //      typedef typename _Traits::char_type char_type;
   //      _LIBCPP_INLINE_VISIBILITY
   //      bool operator()(const char_type& __x, const char_type& __y) _NOEXCEPT
   //      {return _Traits::eq(__x, __y);}
   //   };

   template<class _CharT, class _Traits>
   basic_ostream<_CharT, _Traits>&
   R__put_character_sequence(basic_ostream<_CharT, _Traits>& __os,
                            const _CharT* __str, size_t __len)
   {
#if 0
//#ifndef _LIBCPP_NO_EXCEPTIONS
      try
      {
#endif  // _LIBCPP_NO_EXCEPTIONS
         typename basic_ostream<_CharT, _Traits>::sentry __s(__os);
         if (__s)
         {
            typedef ostreambuf_iterator<_CharT, _Traits> _Ip;
            if (__pad_and_output(_Ip(__os),
                                 __str,
                                 (__os.flags() & ios_base::adjustfield) == ios_base::left ?
                                 __str + __len :
                                 __str,
                                 __str + __len,
                                 __os,
                                 __os.fill()).failed())
               __os.setstate(ios_base::badbit | ios_base::failbit);
         }
#if 0
//#ifndef _LIBCPP_NO_EXCEPTIONS
      }
      catch (...)
      {
         __os.__set_badbit_and_consider_rethrow();
      }
#endif  // _LIBCPP_NO_EXCEPTIONS
      return __os;
   }

   template <typename _CharT, typename _SizeT, typename _Traits, _SizeT __npos>
   _SizeT
   __str_find(_CharT* __p, _SizeT __sz, _CharT *__s, _SizeT __pos, _SizeT __n)
   {
      //_LIBCPP_ASSERT(__n == 0 || __s != nullptr, "string::find(): recieved nullptr");
      if (__pos > __sz || __sz - __pos < __n)
         return __npos;
      if (__n == 0)
         return __pos;
      const _CharT* __r = _VSTD::search(__p + __pos, __p + __sz, __s, __s + __n,
                                        _Traits::eq);
      if (__r == __p + __sz)
         return __npos;
      return static_cast<_SizeT>(__r - __p);
   }


   template <typename _CharT, typename _SizeT, typename _Traits, _SizeT __npos>
   _SizeT
   __str_rfind(_CharT* __p, _SizeT __sz, _CharT *__s, _SizeT __pos, _SizeT __n)
   {
      //_LIBCPP_ASSERT(__n == 0 || __s != nullptr, "string::rfind(): recieved nullptr");
      __pos = _VSTD::min(__pos, __sz);
      if (__n < __sz - __pos)
         __pos += __n;
      else
         __pos = __sz;
      const _CharT* __r = _VSTD::find_end(__p, __p + __pos, __s, __s + __n,
                                          _Traits::eq());
      if (__n > 0 && __r == __p + __pos)
         return __npos;
      return static_cast<_SizeT>(__r - __p);
   }

   template <typename _CharT, typename _SizeT, typename _Traits, size_t __npos>
   _SizeT
   __str_find_first_of(_CharT* __p, _SizeT __sz, _CharT *__s, _SizeT __pos, _SizeT __n)
   {
      if (__pos >= __sz || __n == 0)
         return __npos;
      const _CharT* __r = _VSTD::find_first_of
      (__p + __pos, __p + __sz, __s, __s + __n, _Traits::eq );
      if (__r == __p + __sz)
         return __npos;
      return static_cast<_SizeT>(__r - __p);
   }

   template <typename _CharT, typename _SizeT, typename _Traits, _SizeT __npos>
   _SizeT
   __str_find_last_of(_CharT* __p, _SizeT __sz, _CharT *__s, _SizeT __pos, _SizeT __n)
   {
      if (__n != 0)
      {
         if (__pos < __sz)
            ++__pos;
         else
            __pos = __sz;
         for (const _CharT* __ps = __p + __pos; __ps != __p;)
         {
            const _CharT* __r = _Traits::find(__s, __n, *--__ps);
            if (__r)
               return static_cast<_SizeT>(__ps - __p);
         }
      }
      return __npos;
   }

   template <typename _CharT, typename _SizeT, typename _Traits, _SizeT __npos>
   _SizeT
   __str_find_first_not_of(_CharT* __p, _SizeT __sz, _CharT *__s, _SizeT __pos, _SizeT __n)
   {
      if (__pos < __sz)
      {
         const _CharT* __pe = __p + __sz;
         for (const _CharT* __ps = __p + __pos; __ps != __pe; ++__ps)
            if (_Traits::find(__s, __n, *__ps) == 0)
               return static_cast<_SizeT>(__ps - __p);
      }
      return __npos;
   }

   template <typename _CharT, typename _SizeT, typename _Traits, _SizeT __npos>
   _SizeT
   __str_find_last_not_of(_CharT* __p, _SizeT __sz, _CharT *__s, _SizeT __pos, _SizeT __n)
   {
      if (__pos < __sz)
         ++__pos;
      else
         __pos = __sz;
      for (const _CharT* __ps = __p + __pos; __ps != __p;)
         if (_Traits::find(__s, __n, *--__ps) == 0)
            return static_cast<_SizeT>(__ps - __p);
      return __npos;
   }

}
}

// Now include the main meat
#include "libcpp_string_view.h"


// And undo
#ifdef _LOCAL_LIBCPP_BEGIN_NAMESPACE_LFTS
#undef _LIBCPP_BEGIN_NAMESPACE_LFTS
#undef _LIBCPP_END_NAMESPACE_LFTS
#endif

#ifdef _LOCAL_LIBCPP_BEGIN_NAMESPACE_STD
#undef _LIBCPP_BEGIN_NAMESPACE_STD
#undef _LIBCPP_END_NAMESPACE_STD
#endif

#ifdef _LOCAL_LIBCPP_CONSTEXPR
#undef _LIBCPP_CONSTEXPR
#endif

#ifdef _LOCAL_VSTD
#undef _VSTD
#endif

#ifdef _LOCAL_LIBCPP_INLINE_VISIBILITY
#undef _LIBCPP_INLINE_VISIBILITY
#endif

#ifdef _LOCAL_LIBCPP_STD_VER
#undef _LIBCPP_STD_VER
#endif

#ifdef _LOCAL_LIBCPP_TYPE_VIS_ONLY
#undef _LIBCPP_TYPE_VIS_ONLY
#endif

#ifdef _LOCAL_LIBCPP_CONSTEXPR_AFTER_CXX11
#undef _LIBCPP_CONSTEXPR_AFTER_CXX11
#endif

#ifdef _LOCAL_NOEXCEPT
#undef _NOEXCEPT
#endif

#endif // RWrap_libcpp_string_view_h
