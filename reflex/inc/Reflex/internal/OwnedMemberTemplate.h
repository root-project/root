// @(#)root/reflex:$Name:  $:$Id: OwnedMemberTemplate.h,v 1.1 2006/08/01 09:14:32 roiser Exp $
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_OwnedMemberTemplate
#define ROOT_Reflex_OwnedMemberTemplate

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/MemberTemplate.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class MemberTemplateImpl;

      /**
       * @class OwnedMemberTemplate OwnedMemberTemplate.h OwnedMemberTemplate.h
       * @author Stefan Roiser
       * @date 21/07/2006
       * @ingroup Ref
       */
      class RFLX_API OwnedMemberTemplate : public MemberTemplate {

      public:

         /** constructor */
         OwnedMemberTemplate( const MemberTemplateImpl * memberTemplateImpl )
            : MemberTemplate( memberTemplateImpl ) {}


         /** take ownership */
         OwnedMemberTemplate( const MemberTemplate & rh )
            : MemberTemplate( rh ) {}

         
         /** delete info */
         void Delete() {
            delete fMemberTemplateImpl;
            fMemberTemplateImpl = 0;
         }

      }; // class OwnedMemberTemplate
   
   } // namespace Reflex
} // namespace ROOT


#endif // ROOT_Reflex_OwnedMemberTemplate
