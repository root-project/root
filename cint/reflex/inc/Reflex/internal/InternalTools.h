// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.


// Include Files

#ifndef Reflex_InternalTools
#define Reflex_InternalTools

namespace Reflex {
namespace OTools {
template <typename TO> class ToIter {
public:
   template <typename CONT>
   static typename std::vector<TO>::iterator
   Begin(const CONT& cont) {
      return ((typename std::vector<TO> &) const_cast<CONT&>(cont)).begin();
   }


   template <typename CONT>
   static typename std::vector<TO>::iterator
   End(const CONT& cont) {
      return ((typename std::vector<TO> &) const_cast<CONT&>(cont)).end();
   }


   template <typename CONT>
   static typename std::vector<TO>::const_reverse_iterator
   RBegin(const CONT& cont) {
      return ((const typename std::vector<TO> &)cont).rbegin();
   }


   template <typename CONT>
   static typename std::vector<TO>::const_reverse_iterator
   REnd(const CONT& cont) {
      return ((const typename std::vector<TO> &)cont).rend();
   }


};

}    // namespace OTools
} // namespace Reflex

#endif // Reflex_InternalTools
