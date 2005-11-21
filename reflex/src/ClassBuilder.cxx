// @(#)root/reflex:$Name:  $:$Id: ClassBuilder.cxx,v 1.3 2005/11/11 07:18:06 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Builder/ClassBuilder.h"

#include "Reflex/Type.h"
#include "Reflex/Member.h"

#include "Class.h"
#include "ClassTemplateInstance.h"
#include "Reflex/Tools.h"
#include "Typedef.h"
#include "Enum.h"
#include "DataMember.h"
#include "FunctionMemberTemplateInstance.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilderImpl::ClassBuilderImpl( const char * nam, 
                                                  const std::type_info & ti, 
                                                  size_t size, 
                                                  unsigned int modifiers )
//-------------------------------------------------------------------------------
  : fClass( 0 ),
    fLastMember( 0 )
{
  Type c = Type::ByName(nam);
  if ( c ) { 
    // Class already exists. Check if it was a class.
    if (! c.IsClass() ) throw RuntimeError("Attempt to replace a non-Class At with a Class At"); 
  }

  if ( Tools::IsTemplated( nam))  fClass = new ClassTemplateInstance( nam,
                                                                       size,
                                                                       ti,
                                                                       modifiers );                    
  else                             fClass = new Class( nam, 
                                                       size, 
                                                       ti, 
                                                       modifiers );
}

    
//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilderImpl::~ClassBuilderImpl() {
//-------------------------------------------------------------------------------
  FireClassCallback( fClass->ThisType() );
}

    
//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddBase( const Type & bas,
                                              OffsetFunction offsFP,
                                              unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fClass->AddBase( bas, offsFP, modifiers );
}
    
    
//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddDataMember( const char * nam,
                                                    const Type & typ,
                                                    size_t offs,
                                                    unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fLastMember = Member(new DataMember( nam, typ, offs, modifiers ));
  fClass->AddDataMember( fLastMember );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddFunctionMember( const char * nam,
                                                        const Type & typ,
                                                        StubFunction stubFP,
                                                        void*        stubCtx,
                                                        const char * params,
                                                        unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  if ( Tools::IsTemplated( nam )) 
    fLastMember = Member(new FunctionMemberTemplateInstance( nam, 
                                                             typ, 
                                                             stubFP, 
                                                             stubCtx, 
                                                             params, 
                                                             modifiers,
                                                             *(dynamic_cast<ScopeBase*>(fClass))));
  else                            
    fLastMember = Member(new FunctionMember( nam, 
                                             typ, 
                                             stubFP, 
                                             stubCtx, 
                                             params, 
                                             modifiers ));
  fClass->AddFunctionMember( fLastMember );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddTypedef( const Type & typ,
                                                 const char * def ) {
//-------------------------------------------------------------------------------
  new Typedef( def, typ );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddEnum( const char * nam,
                                              const char * values,
                                              const std::type_info * ti ) {
//-------------------------------------------------------------------------------
  
  Enum * e = new Enum(nam, *ti);

  std::vector<std::string> valVec = std::vector<std::string>();
  Tools::StringSplit(valVec, values, ";");

  for (std::vector<std::string>::const_iterator it = valVec.begin(); 
       it != valVec.end(); ++it ) {
    std::string name = "";
    std::string value = "";
    Tools::StringSplitPair(name, value, *it, "=");
    unsigned long valInt = atol(value.c_str());
    e->AddDataMember( Member( new DataMember( name.c_str(),
                                              Type::ByName("int"),
                                              valInt,
                                              0 )));
  }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddProperty( const char * key, 
                                                  const char * value ) {
//-------------------------------------------------------------------------------
  AddProperty( key, Any(value) );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddProperty( const char * key, 
                                                  Any value ) {
//-------------------------------------------------------------------------------
  if ( fLastMember ) fLastMember.Properties().AddProperty( key, value );
  else                fClass->Properties().AddProperty(key, value); 
}

