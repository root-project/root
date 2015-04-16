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

#include "RConfigure.h"

#ifdef R__HAS_STD_STRING_VIEW

#include <string_view>

#else

# if defined(R__HAS_STD_EXPERIMENTAL_STRING_VIEW)
#  include <experimental/string_view>
# else
#  include "RWrap_libcpp_string_view.h"
# endif

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

#endif // ifdef else R__HAS_STD_STRING_VIEW

#endif // RStringView_H
