// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_OwnedMemberTemplate
#define Reflex_OwnedMemberTemplate

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/MemberTemplate.h"

namespace Reflex {
// forward declarations
class MemberTemplateImpl;

/**
 * @class OwnedMemberTemplate OwnedMemberTemplate.h OwnedMemberTemplate.h
 * @author Stefan Roiser
 * @date 21/07/2006
 * @ingroup Ref
 */
class RFLX_API OwnedMemberTemplate: public MemberTemplate {
public:
   /** constructor */
   OwnedMemberTemplate(const MemberTemplateName * memberTemplateName):
      MemberTemplate(memberTemplateName) {}


   /** take ownership */
   OwnedMemberTemplate(const MemberTemplate &rh):
      MemberTemplate(rh) {}


   /** delete info */
   void
   Delete() {
      fMemberTemplateName->DeleteMemberTemplate();
   }


};    // class OwnedMemberTemplate

} // namespace Reflex

#endif // Reflex_OwnedMemberTemplate
