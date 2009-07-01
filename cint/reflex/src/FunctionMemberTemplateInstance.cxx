// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "FunctionMemberTemplateInstance.h"

#include "Reflex/MemberTemplate.h"
#include "Reflex/internal/MemberTemplateImpl.h"
#include "Reflex/internal/OwnedMember.h"

#include "Reflex/Tools.h"

#include <iostream>

//-------------------------------------------------------------------------------
Reflex::FunctionMemberTemplateInstance::
FunctionMemberTemplateInstance(const char* nam,
                               const Type& typ,
                               StubFunction stubFP,
                               void* stubCtx,
                               const char* params,
                               unsigned int modifiers,
                               const Scope& scop):
   FunctionMember(nam,
                  typ,
                  stubFP,
                  stubCtx,
                  params,
                  modifiers,
                  MEMBERTEMPLATEINSTANCE),
   TemplateInstance(Tools::GetTemplateArguments(nam)),
   fTemplateFamily(MemberTemplate()) {
//-------------------------------------------------------------------------------
// Create the dictionary information for a templated function member.
   std::string templateName = Tools::GetTemplateName(nam);
   std::string scopeName = scop.Name(SCOPED);
   std::string scopedTemplateName = "";

   if (scopeName != "") {
      scopedTemplateName = scopeName + "::" + templateName;
   } else { scopedTemplateName = templateName; }

//    for ( size_t i = 0; i < scop.MemberTemplateSize(); ++i ) {
//       MemberTemplate mtl = scop.MemberTemplateAt( i );
//       if ( mtl.Name(SCOPED) == scopedTemplateName &&
//            mtl.TemplateParameterSize() == TemplateArgumentSize()) {
//          fTemplateFamily = mtl;
//          break;
//       }
//    }

   fTemplateFamily = MemberTemplate::ByName(scopedTemplateName, TemplateArgumentSize());

   if (!fTemplateFamily) {
      std::vector<std::string> parameterNames = std::vector<std::string>();

      for (size_t i = 65; i < 65 + TemplateArgumentSize(); ++i) {
         parameterNames.push_back("typename " + std::string(1, char (i)));
      }
      MemberTemplateImpl* mti = new MemberTemplateImpl(scopedTemplateName.c_str(),
                                                       scop,
                                                       parameterNames);
      fTemplateFamily = mti->ThisMemberTemplate();
      scop.AddMemberTemplate(fTemplateFamily);
   }

   fTemplateFamily.AddTemplateInstance((Member) (*this));
}


//-------------------------------------------------------------------------------
std::string
Reflex::FunctionMemberTemplateInstance::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of the function member.
   return FunctionMember::Name(mod);
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionMemberTemplateInstance::TemplateArgumentAt(size_t nth) const {
//-------------------------------------------------------------------------------
// Return nth template argument of this function member.
   return TemplateInstance::TemplateArgumentAt(nth);
}
