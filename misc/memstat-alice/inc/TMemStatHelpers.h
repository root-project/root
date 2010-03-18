// @(#)root/memstat:$Name$:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 09/05/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemStatHelpers
#define ROOT_TMemStatHelpers

// ROOT
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TObjString
#include "TObjString.h"
#endif
#ifndef ROOT_TCollection
#include "TCollection.h"
#endif
// STD
#include <string>
#include <functional>
#include <algorithm>
#include <cctype>


class TObject;

namespace Memstat
{
std::string dig2bytes(Long64_t bytes);

//______________________________________________________________________________
struct SFind_t : std::binary_function<TObject*, TString, bool> {
   bool operator()(TObject *_Obj, const TString &_ToFind) const
   {
      TObjString *str(dynamic_cast<TObjString*>(_Obj));
      if (!str)
         return false;
      return !str->String().CompareTo(_ToFind);
   }
};

//______________________________________________________________________________
template<class T>
Int_t find_string( const T &_Container,  const TString &_ToFind )
{
   // This function retuns an index in _Container of found element.
   // Returns -1 if there was no element found.

   typedef TIterCategory<T> iterator_t;

   iterator_t iter(&_Container);
   iterator_t found(
      std::find_if(iter.Begin(), iterator_t::End(), bind2nd(SFind_t(), _ToFind))
   );
   return ( ( !(*found) )? -1: std::distance(iter.Begin(), found) );
}

//______________________________________________________________________________
// HACK: because of the bug in gcc 3.3 we need to use this nasty ToLower and ToUpper instead of direct calls of tolower (tolower.. is inline in this version of std lib)...
struct ToLower_t
{
   char operator() ( char c ) const
   {
      return std::tolower( c );
   }
};

}
#endif
