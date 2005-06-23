// @(#)root/reflex:$Name:$:$Id:$
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
ROOT::Reflex::ClassBuilderImpl::ClassBuilderImpl( const char * Name, 
                                                  const std::type_info & ti, 
                                                  size_t size, 
                                                  unsigned int modifiers )
//-------------------------------------------------------------------------------
  : fClass( 0 ),
    fLastMember( 0 )
{
  Type c = Type::ByName(Name);
  if ( c ) { 
    // Class already exists. Check if it was a class.
    if (! c.IsClass() ) throw RuntimeError("Attempt to replace a non-Class TypeNth with a Class TypeNth"); 
  }

  if ( Tools::IsTemplated(Name) )  fClass = new ClassTemplateInstance( Name,
                                                                        size,
                                                                        ti,
                                                                        modifiers );                    
  else                             fClass = new Class( Name, 
                                                        size, 
                                                        ti, 
                                                        modifiers );
}

    
//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilderImpl::~ClassBuilderImpl() {
//-------------------------------------------------------------------------------
  FireClassCallback( fClass->TypeGet() );
}

    
//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddBase( const Type & BaseNth,
                                              OffsetFunction OffsetFP,
                                              unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fClass->AddBase( BaseNth, OffsetFP, modifiers );
}
    
    
//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddDataMember( const char * Name,
                                                    const Type & TypeNth,
                                                    size_t Offset,
                                                    unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fLastMember = Member(new DataMember( Name, TypeNth, Offset, modifiers ));
  fClass->AddDataMember( fLastMember );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddFunctionMember( const char * Name,
                                                        const Type & TypeNth,
                                                        StubFunction stubFP,
                                                        void*        stubCtx,
                                                        const char * params,
                                                        unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  if ( Tools::IsTemplated(Name) ) 
    fLastMember = Member(new FunctionMemberTemplateInstance( Name, 
                                                              TypeNth, 
                                                              stubFP, 
                                                              stubCtx, 
                                                              params, 
                                                              modifiers,
                                                              (Scope)(*fClass)));
  else                            
    fLastMember = Member(new FunctionMember( Name, 
                                              TypeNth, 
                                              stubFP, 
                                              stubCtx, 
                                              params, 
                                              modifiers ));
  fClass->AddFunctionMember( fLastMember );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddTypedef( const Type & TypeNth,
                                                 const char * def ) {
//-------------------------------------------------------------------------------
  new Typedef( def, TypeNth );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddEnum( const char * Name,
                                              const char * values,
                                              const std::type_info * typeinfo ) {
//-------------------------------------------------------------------------------
  
  Enum * e = new Enum(Name, *typeinfo);

  std::vector<std::string> valVec = std::vector<std::string>();
  Tools::StringSplit(valVec, values, ";");

  for (std::vector<std::string>::const_iterator it = valVec.begin(); 
       it != valVec.end(); ++it ) {
    std::string Name = "";
    std::string value = "";
    Tools::StringSplitPair(Name, value, *it, "=");
    unsigned long valInt = atol(value.c_str());
    e->AddDataMember( Member( new DataMember( Name.c_str(),
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
  if ( fLastMember ) fLastMember.PropertyListGet().AddProperty( key, value );
  else                fClass->PropertyListGet().AddProperty(key, value); 
}

