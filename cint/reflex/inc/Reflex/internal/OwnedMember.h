// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_OwnedMember
#define Reflex_OwnedMember

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Member.h"
#include <vector>

namespace Reflex {
// forward declarations
class MemberBase;

/**
 * @class OwnedMember OwnedMember.h OwnedMember.h
 * @author Stefan Roiser
 * @date 21/07/2006
 * @ingroup Ref
 */
class RFLX_API OwnedMember: public Member {
public:
   /** constructor */
   OwnedMember(const MemberBase * memberBase = 0):
      Member(memberBase) {
   }

   /** take ownership */
   OwnedMember(const Member &rh):
      Member(rh) {}


   /** delete info */
   void
   Delete() {
      Member::Delete();
      /*             delete fMemberBase; */
      /*             fMemberBase = 0; */
   }


};    // class OwnedMember

} // namespace Reflex


#endif // Reflex_OwnedMember
