// -*- C++ -*-
// Author: Philippe Canal, March 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RStringView_H
#define RStringView_H

// Make sure the C++ standard library 'detection' macros are visible.
#include <string>

#if defined(__GLIBCXX__)

// With glibcxx, we use our version unless we are being compiled in C++14 mode
// (or higher)
#if __cplusplus > 201103L

#if !defined(__has_include)
#define R_HAS_std_experimental_string_view
#else
#define R__CHECK_FOR_STRING_VIEW_HEADER
#endif

#endif

#elif defined(_LIBCPP_VERSION)

#if !defined(__has_include)
#if __cplusplus <= 201402L
#define R_HAS_std_experimental_string_view
#else
#define R_HAS_std_string_view_header
#endif
#else
#define R__CHECK_FOR_STRING_VIEW_HEADER
#endif

#else

#if defined(__has_include)
#define R__CHECK_FOR_STRING_VIEW_HEADER
#endif

#endif

#if defined(R__CHECK_FOR_STRING_VIEW_HEADER)
#if __has_include("string_view")

#define R_HAS_std_string_view_header

#elif __has_include("experimental/string_view")

#define R_HAS_lib_std_experimental_string_view

#endif
#endif // R__CHECK_FOR_STRING_VIEW

#ifdef R_HAS_std_string_view_header
#include <string_view>
#elif defined(R_HAS_lib_std_experimental_string_view)
#define R_HAS_std_experimental_string_view
#include <experimental/string_view>
#else
#include "RWrap_libcpp_string_view.h"
#endif

#if defined( R_HAS_std_experimental_string_view )

namespace std {

   template<class _CharT, class _Traits = std::char_traits<_CharT> >
   using basic_string_view = ::std::experimental::basic_string_view<_CharT,_Traits>;

   // basic_string_view typedef names
   typedef basic_string_view<char> string_view;
   typedef basic_string_view<char16_t> u16string_view;
   typedef basic_string_view<char32_t> u32string_view;
   typedef basic_string_view<wchar_t> wstring_view;

//   template<class _CharT, class _Traits = std::char_traits<_CharT> >
//   basic_string_view<_CharT,_Traits>
//   &operator=(basic_string_view<_CharT,_Traits> &lhs, const TString &rsh) {
//      *lhs = basic_string_view<_CharT,_Traits>(rsh);
//      return *lhs;
//   }
}

#endif // R_HAS_std_experimental_string_view

#endif // RStringView_H
