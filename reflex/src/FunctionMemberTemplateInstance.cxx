// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "FunctionMemberTemplateInstance.h"

#include "Reflex/MemberTemplate.h"
#include "Reflex/MemberTemplateImpl.h"

#include "Reflex/Tools.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::FunctionMemberTemplateInstance::
FunctionMemberTemplateInstance( const char * Name,
                                const Type & TypeNth,
                                StubFunction stubFP,
                                void * stubCtx,
                                const char * params,
                                unsigned int modifiers,
                                const Scope & ScopeNth )
//-------------------------------------------------------------------------------
  : FunctionMember( Name,
                    TypeNth,
                    stubFP,
                    stubCtx,
                    params,
                    modifiers,
                    MEMBERTEMPLATEINSTANCE ),
    TemplateInstance( Tools::GetTemplateArguments( Name )),
    fTemplateFamily( MemberTemplate()) {
  
  std::string templateName = Tools::GetTemplateName( Name );
  std::string scopeName = ScopeNth.Name(SCOPED);
  std::string scopedTemplateName = "";
  if ( scopeName != "" ) scopedTemplateName = scopeName + "::" + templateName;
  else                   scopedTemplateName = templateName;

  for ( size_t i = 0; i < ScopeNth.MemberTemplateCount(); ++i ) {
    MemberTemplate mtl = ScopeNth.MemberTemplateNth( i );
    if ( mtl.Name(SCOPED) == scopedTemplateName && 
         mtl.ParameterCount() == TemplateArgumentCount()) {
      fTemplateFamily = mtl;
      break;
    }
  }

  if ( ! fTemplateFamily ) {
    std::vector < std::string > parameterNames = std::vector < std::string > ();
    for ( size_t i = 65; i < 65 + TemplateArgumentCount(); ++i ) 
      parameterNames.push_back("typename " + std::string(new char(i)));
    MemberTemplateImpl * mti = new MemberTemplateImpl( Tools::GetBaseName(templateName),
                                                       ScopeNth,
                                                       parameterNames );
    fTemplateFamily = MemberTemplate( mti );
    ScopeNth.AddMemberTemplate( fTemplateFamily );
  }
  
  fTemplateFamily.AddTemplateInstance((Member)(*this));
}


//-------------------------------------------------------------------------------
std::string 
ROOT::Reflex::FunctionMemberTemplateInstance::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
  return FunctionMember::Name( mod );
}



//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionMemberTemplateInstance::TemplateArgumentNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return TemplateInstance::TemplateArgumentNth( nth );
}
