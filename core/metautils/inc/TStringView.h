// -*- C++ -*-
// Author: Philippe Canal, March 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TStringView_H
#define TStringView_H

#ifdef R_HAS_std_string_view

#include <string_view>

namespace ROOT {

   template<class _CharT, class _Traits = std::char_traits<_CharT> >
   using TBasicStringView = ::std::basic_string_view<_CharT,_Traits>;

}

#else // R_HAS_std_string_view

#include "RWrap_libcpp_string_view.h"

namespace ROOT {

   template<class _CharT, class _Traits = std::char_traits<_CharT> >
   using TBasicStringView = ::std::experimental::basic_string_view<_CharT,_Traits>;

//   template<class _CharT, class _Traits = std::char_traits<_CharT> >
//   TBasicStringView<_CharT,_Traits>
//   &operator=(TBasicStringView<_CharT,_Traits> &lhs, const TString &rsh) {
//      *lhs = TBasicStringView<_CharT,_Traits>(rsh);
//      return *lhs;
//   }
}

#endif // R_HAS_std_string_view

namespace ROOT {

   // basic_string_view typedef names
   typedef TBasicStringView<char> TStringView;
   typedef TBasicStringView<char16_t> TStringViewU16;
   typedef TBasicStringView<char32_t> TStringViewU32;
   typedef TBasicStringView<wchar_t> TStringViewW;

}

#endif // TStringView_H
