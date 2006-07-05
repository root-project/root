// @(#)root/reflex:$Name: HEAD $:$Id: Fundamental.h,v 1.4 2006/03/06 12:51:46 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Fundamental
#define ROOT_Reflex_Fundamental

// Include files
#include "Reflex/TypeBase.h"

namespace ROOT {
   namespace Reflex {


      /**
       * @class Fundamental Fundamental.h Reflex/Fundamental.h
       *  @author Stefan Roiser
       *  @date 24/11/2003
       */
      class Fundamental : public TypeBase {
      public:

         /** default constructor */
         Fundamental( const char * typ,
                      size_t size,
                      const std::type_info & ti ) ;


         /** destructor */
         virtual ~Fundamental() {}

      }; // class Fundamental
   } //namespace Reflex
} //namespace ROOT

#endif // ROOT_Reflex_Fundamental

