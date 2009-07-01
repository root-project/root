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

#include "Reflex/Builder/VariableBuilder.h"

#include "Reflex/internal/OwnedMember.h"

#include "Namespace.h"
#include "DataMember.h"


//-------------------------------------------------------------------------------
Reflex::VariableBuilderImpl::VariableBuilderImpl(const char* nam,
                                                 const Type& typ,
                                                 size_t offs,
                                                 unsigned int modifiers):
   fDataMember(Member()) {
//-------------------------------------------------------------------------------
// Construct the info for a variable.
   std::string declScope = "";
   std::string memName = std::string(nam);
   size_t pos = memName.rfind("::");

   if (pos != std::string::npos) {
      declScope = memName.substr(0, pos);
      memName = memName.substr(pos + 2);
   }

   Scope sc = Scope::ByName(declScope);

   if (!sc) {
      sc = (new Namespace(declScope.c_str()))->ThisScope();
   }

   if (!sc.IsNamespace()) {
      throw RuntimeError("Declaring At is not a namespace");
   }

   sc.AddDataMember(memName.c_str(),
                    typ,
                    offs,
                    modifiers);
}


//-------------------------------------------------------------------------------
Reflex::VariableBuilderImpl::~VariableBuilderImpl() {
//-------------------------------------------------------------------------------
// Destructor.
   FireFunctionCallback(fDataMember);
}


//-------------------------------------------------------------------------------
void
Reflex::VariableBuilderImpl::AddProperty(const char* key,
                                         const char* value) {
//-------------------------------------------------------------------------------
// Attach a property to this variable as string.
   fDataMember.Properties().AddProperty(key, value);
}


//-------------------------------------------------------------------------------
void
Reflex::VariableBuilderImpl::AddProperty(const char* key,
                                         Any value) {
//-------------------------------------------------------------------------------
// Attach a property to this variable as Any object.
   fDataMember.Properties().AddProperty(key, value);
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::VariableBuilderImpl::ToMember() {
//-------------------------------------------------------------------------------
// Return the member currently being built.
   return fDataMember;
}


//-------------------------------------------------------------------------------
Reflex::VariableBuilder::VariableBuilder(const char* nam,
                                         const Type& typ,
                                         size_t offs,
                                         unsigned int modifiers):
   fDataMember(Member()) {
//-------------------------------------------------------------------------------
// Construct the variable info.
   std::string declScope = Tools::GetScopeName(nam);
   std::string memName = Tools::GetBaseName(nam);

   Scope sc = Scope::ByName(declScope);

   if (!sc) {
      sc = (new Namespace(declScope.c_str()))->ThisScope();
   }

   if (!sc.IsNamespace()) {
      throw RuntimeError("Declaring scope is not a namespace");
   }

   DataMember* dm = new DataMember(memName.c_str(),
                                   typ,
                                   offs,
                                   modifiers);
   sc.AddDataMember(Member(dm));
   fDataMember = Member(dm);
}


//-------------------------------------------------------------------------------
Reflex::VariableBuilder::~VariableBuilder() {
//-------------------------------------------------------------------------------
// Destructor.
   FireFunctionCallback(fDataMember);
}


//-------------------------------------------------------------------------------
Reflex::VariableBuilder&
Reflex::VariableBuilder::AddProperty(const char* key,
                                     const char* value) {
//-------------------------------------------------------------------------------
// Attach a property to this variable as a string.
   fDataMember.Properties().AddProperty(key, value);
   return *this;
}


//-------------------------------------------------------------------------------
Reflex::VariableBuilder&
Reflex::VariableBuilder::AddProperty(const char* key,
                                     Any value) {
//-------------------------------------------------------------------------------
// Attach a property to this variable as Any object.
   fDataMember.Properties().AddProperty(key, value);
   return *this;
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::VariableBuilder::ToMember() {
//-------------------------------------------------------------------------------
// Return the member currently being built.
   return fDataMember;
}
