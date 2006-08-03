// @(#)root/reflex:$Name:  $:$Id: InternalTools.h,v 1.2 2006/08/01 15:04:59 roiser Exp $
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.


// Include Files

namespace ROOT {

   namespace Reflex {

      namespace OTools {

         template< typename TO > class ToIter {
            
         public:

            template < typename CONT > 
               static typename std::vector<TO>::iterator Begin( const CONT & cont ) {
               return typename std::vector<TO>::iterator((TO*)&cont[0]);
            }

            template < typename CONT >
               static typename std::vector<TO>::iterator End( const CONT & cont ) {
               typename std::vector<TO>::iterator it = typename std::vector<TO>::iterator((TO*)&cont[cont.size()-1]);
               return it++;
            }

            template < typename CONT > 
               static typename std::vector<TO>::reverse_iterator RBegin( const CONT & cont ) {
               return typename std::vector<TO>::reverse_iterator(typename std::vector<TO>::iterator((TO*)&cont[cont.size()-1]));
            }

            template < typename CONT >
               static typename std::vector<TO>::reverse_iterator REnd( const CONT & cont ) {
               typename std::vector<TO>::iterator it = typename std::vector<TO>::iterator((TO*)&cont[0]);
               return typename std::vector<TO>::reverse_iterator(it--);
            }

         };

      } // namespace OTools
   } // namespace Reflex
} // namespace ROOT
