// @(#)root/reflex:$Name:  $:$Id: TypeName.cxx,v 1.12 2006/08/01 09:14:33 roiser Exp $
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.


// Include Files
// Note: the Owned*.h stuff is needed for Solaris CC
#include "Reflex/internal/OwnedMember.h"


namespace ROOT {

   namespace Reflex {

      namespace OTools {

         // internal stuff (can be moved)

         template< typename TO > class ToIter {
            
         public:

            template < typename FROM > 
               static typename std::vector<TO>::iterator Forward( const FROM & iter ) {
               return typename std::vector<TO>::iterator(iter.base());
            }


            template < typename FROM > 
               static typename std::vector<TO>::reverse_iterator Reverse( const FROM & iter ) {
               return typename std::vector<TO>::reverse_iterator(typename std::vector<TO>::iterator(iter.base().base()));
            }

         };

      } // namespace OTools
   } // namespace Reflex
} // namespace ROOT
