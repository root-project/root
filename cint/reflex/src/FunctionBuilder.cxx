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

#include "Reflex/Builder/FunctionBuilder.h"

#include "Reflex/PropertyList.h"
#include "Reflex/Scope.h"
#include "Reflex/Any.h"
#include "Reflex/Type.h"
#include "Reflex/Tools.h"
#include "Reflex/internal/OwnedMember.h"

#include "FunctionMember.h"
#include "FunctionMemberTemplateInstance.h"
#include "Namespace.h"


//-------------------------------------------------------------------------------
Reflex::FunctionBuilder::~FunctionBuilder() {
//-------------------------------------------------------------------------------
// Functionbuilder destructor used for call backs.
   FireFunctionCallback(fFunction);
}


//-------------------------------------------------------------------------------
Reflex::FunctionBuilder&
Reflex::FunctionBuilder::AddProperty(const char* key,
                                     const char* value) {
//-------------------------------------------------------------------------------
// Add property info to this function as string.
   fFunction.Properties().AddProperty(key, value);
   return *this;
}


//-------------------------------------------------------------------------------
Reflex::FunctionBuilder&
Reflex::FunctionBuilder::AddProperty(const char* key,
                                     Any value) {
//-------------------------------------------------------------------------------
// Add property info to this function as Any object.
   fFunction.Properties().AddProperty(key, value);
   return *this;
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::FunctionBuilder::ToMember() {
//-------------------------------------------------------------------------------
// Return the function currently being built.
   return fFunction;
}


//-------------------------------------------------------------------------------
Reflex::FunctionBuilderImpl::FunctionBuilderImpl(const char* nam,
                                                 const Type& typ,
                                                 StubFunction stubFP,
                                                 void* stubCtx,
                                                 const char* params,
                                                 unsigned char modifiers):
   fFunction(Member(0)) {
//-------------------------------------------------------------------------------
// Create function type dictionary info (internal).
   std::string fullname(nam);
   std::string declScope;
   std::string funcName;
   size_t pos = Tools::GetTemplateName(nam).rfind("::");

   // Name contains declaring At
   if (pos != std::string::npos) {
      funcName = fullname.substr(pos + 2);
      declScope = fullname.substr(0, pos);
   } else {
      funcName = nam;
      declScope = "";
   }

   Scope sc = Scope::ByName(declScope);

   if (!sc) {
      // Let's create the namespace here
      sc = (new Namespace(declScope.c_str()))->ThisScope();
   }

   if (!sc.IsNamespace()) {
      throw RuntimeError("Declaring scope is not a namespace");
   }

   if (Tools::IsTemplated(funcName.c_str())) {
      fFunction = Member(new FunctionMemberTemplateInstance(funcName.c_str(),
                                                            typ,
                                                            stubFP,
                                                            stubCtx,
                                                            params,
                                                            modifiers,
                                                            sc));
   } else { fFunction = Member(new FunctionMember(funcName.c_str(),
                                                  typ,
                                                  stubFP,
                                                  stubCtx,
                                                  params,
                                                  modifiers)); }
   sc.AddFunctionMember(fFunction);
}


//-------------------------------------------------------------------------------
Reflex::FunctionBuilderImpl::~FunctionBuilderImpl() {
//-------------------------------------------------------------------------------
// FunctionBuilder destructor.
   FireFunctionCallback(fFunction);
}


//-------------------------------------------------------------------------------
void
Reflex::FunctionBuilderImpl::AddProperty(const char* key,
                                         const char* value) {
//-------------------------------------------------------------------------------
// Add property info to this function type.
   fFunction.Properties().AddProperty(key, value);
}


//-------------------------------------------------------------------------------
void
Reflex::FunctionBuilderImpl::AddProperty(const char* key,
                                         Any value) {
//-------------------------------------------------------------------------------
// Add property info to this function type.
   fFunction.Properties().AddProperty(key, value);
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::FunctionBuilderImpl::ToMember() {
//-------------------------------------------------------------------------------
// Return the function member currently being built.
   return fFunction;
}


//-------------------------------------------------------------------------------
Reflex::FunctionBuilder::FunctionBuilder(const Type& typ, const char* nam, StubFunction stubFP, void* stubCtx, const char* params, unsigned char modifiers): fFunction(Member(0)) {
   // Create function dictionary type information.
   std::string declScope(Tools::GetScopeName(nam));
   std::string funcName(Tools::GetBaseName(nam));
   Scope sc = Scope::ByName(declScope);

   if (!sc) {
      // Let's create the namespace here
      sc = (new Namespace(declScope.c_str()))->ThisScope();
   }

   if (!sc.IsNamespace()) {
      throw RuntimeError("Declaring scope is not a namespace");
   }

   if (Tools::IsTemplated(funcName.c_str())) {
      fFunction = Member(new FunctionMemberTemplateInstance(funcName.c_str(), typ, stubFP, stubCtx, params, modifiers, sc));
   } else {
      fFunction = Member(new FunctionMember(funcName.c_str(), typ, stubFP, stubCtx, params, modifiers));
   }
   sc.AddFunctionMember(fFunction);
}
