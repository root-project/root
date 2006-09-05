// @(#)root/reflex:$Name:  $:$Id: ClassBuilder.cxx,v 1.13 2006/08/16 14:04:10 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
#define REFLEX_BUILD
#endif

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
ROOT::Reflex::ClassBuilder::ClassBuilder( const char * nam, 
                                          const std::type_info & ti,
                                          size_t size,
                                          unsigned int modifiers ) 
//-------------------------------------------------------------------------------
// ClassBuilder constructor. 
  : fClassBuilderImpl( nam, ti, size, modifiers ) { }


//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilder & 
ROOT::Reflex::ClassBuilder::AddBase( const Type & bas,
                                     OffsetFunction offsFP, 
                                     unsigned int modifiers ) {
//-------------------------------------------------------------------------------
// Add base class information to this class.
   fClassBuilderImpl.AddBase( bas, 
                              offsFP,
                              modifiers );
   return * this;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilder & 
ROOT::Reflex::ClassBuilder::AddDataMember( const Type &  typ,
                                           const char *  nam,
                                           size_t        offs,
                                           unsigned int modifiers ) {
//-------------------------------------------------------------------------------
// Add data member info to this class.
   fClassBuilderImpl.AddDataMember( nam,
                                    typ,
                                    offs,
                                    modifiers );
   return * this;
}
    
    
//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilder & 
ROOT::Reflex::ClassBuilder::AddFunctionMember( const Type & typ,
                                               const char * nam,
                                               StubFunction stubFP,
                                               void * stubCtx,
                                               const char * params, 
                                               unsigned int modifiers ) {
//-------------------------------------------------------------------------------
// Add function member info to this class.
   fClassBuilderImpl.AddFunctionMember( nam,
                                        typ,
                                        stubFP,
                                        stubCtx,
                                        params,
                                        modifiers );
   return * this;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilder & 
ROOT::Reflex::ClassBuilder::AddTypedef( const char * typ,
                                        const char * def ) {
//-------------------------------------------------------------------------------
// Add typedef info to this class.
   fClassBuilderImpl.AddTypedef( TypeBuilder( typ ),
                                 def );
   return * this;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilder & 
ROOT::Reflex::ClassBuilder::AddTypedef( const Type & typ,
                                        const char * def ) {
//-------------------------------------------------------------------------------
// Add typedef info to this class.
   fClassBuilderImpl.AddTypedef( typ,
                                 def );
   return * this;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilder &
ROOT::Reflex::ClassBuilder::AddEnum( const char * nam,
                                     const char * values,
                                     const std::type_info * ti,
                                     unsigned int modifiers ) {
//-------------------------------------------------------------------------------
// Add enum info to this class.
   fClassBuilderImpl.AddEnum( nam, 
                              values, 
                              ti,
                              modifiers );
   return * this;
}


/*/-------------------------------------------------------------------------------
  ROOT::Reflex::ClassBuilder &
  ROOT::Reflex::ClassBuilder::addUnion( const char * nam,
  const char * values,
  unsigned int modifiers ) {
//-------------------------------------------------------------------------------
  fClassBuilderImpl.addUnion( nam, values, modifiers );
  return * this;
  }
*/


//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilderImpl::ClassBuilderImpl( const char * nam, 
                                                  const std::type_info & ti, 
                                                  size_t size, 
                                                  unsigned int modifiers )
   : fClass( 0 ),
     fLastMember( 0 )
{
//-------------------------------------------------------------------------------
// Construct a class information (internal).

   std::string nam2(nam);

   const Type & c = Type::ByName(nam2);
   if ( c ) { 
      // We found a typedef to a class with the same name
      if ( c.IsTypedef() ) nam2 += " @HIDDEN@";
      // Class already exists. Check if it was a class.
      else if (! c.IsClass() ) throw RuntimeError("Attempt to replace a non-class type with a class"); 
   }

   if ( Tools::IsTemplated( nam ))  fClass = new ClassTemplateInstance( nam2.c_str(),
                                                                        size,
                                                                        ti,
                                                                        modifiers );                    
   else                             fClass = new Class( nam2.c_str(), 
                                                        size, 
                                                        ti, 
                                                        modifiers );
}

    
//-------------------------------------------------------------------------------
ROOT::Reflex::ClassBuilderImpl::~ClassBuilderImpl() {
//-------------------------------------------------------------------------------
// ClassBuilderImpl destructor. Used for call back functions (e.g. Cintex).
   FireClassCallback( fClass->ThisType() );
}

    
//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddBase( const Type & bas,
                                              OffsetFunction offsFP,
                                              unsigned int modifiers ) {
//-------------------------------------------------------------------------------
// Add base class information (internal).
   fClass->AddBase( bas, offsFP, modifiers );
}
    
    
//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddDataMember( const char * nam,
                                                    const Type & typ,
                                                    size_t offs,
                                                    unsigned int modifiers ) {
//-------------------------------------------------------------------------------
// Add data member info (internal).
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
// Add function member info (internal).
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
// Add typedef info (internal).
   TypedefTypeBuilder( def, typ );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddEnum( const char * nam,
                                              const char * values,
                                              const std::type_info * ti,
                                              unsigned int modifiers ) {
//-------------------------------------------------------------------------------
// Add enum info (internal).  

   EnumTypeBuilder(nam, values, *ti, modifiers);
//    Enum * e = new Enum(nam, *ti, modifiers);

//    std::vector<std::string> valVec = std::vector<std::string>();
//    Tools::StringSplit(valVec, values, ";");

//    for (std::vector<std::string>::const_iterator it = valVec.begin(); 
//         it != valVec.end(); ++it ) {
//       std::string name = "";
//       std::string value = "";
//       Tools::StringSplitPair(name, value, *it, "=");
//       unsigned long int valInt = atol(value.c_str());
//       e->AddDataMember( Member( new DataMember( name.c_str(),
//                                                 Type::ByName("int"),
//                                                 valInt,
//                                                 0 )));
//    }
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddProperty( const char * key, 
                                                  const char * value ) {
//-------------------------------------------------------------------------------
// Add property info (internal).
   AddProperty( key, Any(value) );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::ClassBuilderImpl::AddProperty( const char * key, 
                                                  Any value ) {
//-------------------------------------------------------------------------------
// Add property info (internal).
   if ( fLastMember ) fLastMember.Properties().AddProperty( key, value );
   else                fClass->Properties().AddProperty(key, value); 
}

