// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "ClassTemplateInstance.h"

#include "Reflex/TypeTemplate.h"
#include "Reflex/Scope.h"

#include "TemplateInstance.h"
#include "Reflex/Tools.h"

#include <vector>
#include <string>
#include <sstream>

//-------------------------------------------------------------------------------
ROOT::Reflex::ClassTemplateInstance::
ClassTemplateInstance( const char * TypeNth, 
                       size_t size, 
                       const std::type_info & TypeInfo, 
                       unsigned int modifiers )
//-------------------------------------------------------------------------------
  : Class( TypeNth, 
           size, 
           TypeInfo, 
           modifiers,
           TYPETEMPLATEINSTANCE ),
    TemplateInstance( Tools::GetTemplateArguments( TypeNth )),
    fTemplateFamily( TypeTemplate()) {

  Scope s = DeclaringScope();

  std::string templateName = Tools::GetTemplateName( TypeNth );

  for ( size_t i = 0; i < s.TypeTemplateCount(); ++i ) {
    TypeTemplate ttl = s.TypeTemplateNth( i );
    if ( ttl.Name(SCOPED) == templateName ) {
      fTemplateFamily = ttl;
      break;
    }
  }
  
  if ( ! fTemplateFamily ) {
    std::vector < std::string > parameterNames = std::vector < std::string > ();
    for ( size_t i = 65; i < 65 + TemplateArgumentCount(); ++i ) {
      std::ostringstream o; 
      o << char(i); 
      parameterNames.push_back("typename " + o.str());      
    }
    TypeTemplateImpl * tti = new TypeTemplateImpl( Tools::GetBaseName(templateName),
                                                   s,
                                                   parameterNames );
    fTemplateFamily = TypeTemplate(tti);
    s.AddTypeTemplate( fTemplateFamily );
  }
  
  fTemplateFamily.AddTemplateInstance((Type)(*this));
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::ClassTemplateInstance::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  return Class::Name( mod );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::ClassTemplateInstance::TemplateArgumentNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return TemplateInstance::TemplateArgumentNth( nth );
}
