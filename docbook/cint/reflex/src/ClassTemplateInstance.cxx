// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "ClassTemplateInstance.h"

#include "Reflex/Scope.h"

#include "TemplateInstance.h"
#include "Reflex/Tools.h"

#include <vector>
#include <string>
#include <sstream>

//-------------------------------------------------------------------------------
Reflex::ClassTemplateInstance::
ClassTemplateInstance(const char* typ,
                      size_t size,
                      const std::type_info& ti,
                      unsigned int modifiers):
   Class(typ,
         size,
         ti,
         modifiers,
         TYPETEMPLATEINSTANCE),
   TemplateInstance(Tools::GetTemplateArguments(typ)),
   fTemplateFamily(TypeTemplate()) {
//-------------------------------------------------------------------------------
// Construct a template class instance dictionary information. This constructor
// takes case of deducing the template parameter names and generates the info for
// a template family if necessary.
   Scope s = DeclaringScope();

   std::string templateName = Tools::GetTemplateName(typ);

//    for ( size_t i = 0; i < s.SubTypeTemplateSize(); ++i ) {
//       TypeTemplate ttl = s.SubTypeTemplateAt( i );
//       if ( ttl.Name(SCOPED) == templateName ) {
//          fTemplateFamily = ttl;
//          break;
//       }
//    }

   fTemplateFamily = TypeTemplate::ByName(templateName, TemplateArgumentSize());

   if (!fTemplateFamily) {
      std::vector<std::string> parameterNames = std::vector<std::string>();

      std::string typenameP("typename X");
      for (size_t i = 65; i < 65 + TemplateArgumentSize(); ++i) {
         typenameP[9] = (char) i;
         parameterNames.push_back(typenameP);
      }
      TypeTemplateImpl* tti = new TypeTemplateImpl(templateName.c_str(),
                                                   s,
                                                   parameterNames);
      fTemplateFamily = tti->ThisTypeTemplate();
      s.AddSubTypeTemplate(fTemplateFamily);
   }

   fTemplateFamily.AddTemplateInstance((Type) (*this));
}


//-------------------------------------------------------------------------------
std::string
Reflex::ClassTemplateInstance::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of the template class.
   return Class::Name(mod);
}


//-------------------------------------------------------------------------------
const char*
Reflex::ClassTemplateInstance::SimpleName(size_t& pos,
                                          unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of the template class.
   return Class::SimpleName(pos, mod);
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::ClassTemplateInstance::TemplateArgumentAt(size_t nth) const {
//-------------------------------------------------------------------------------
// Return the nth template argument type.
   return TemplateInstance::TemplateArgumentAt(nth);
}
