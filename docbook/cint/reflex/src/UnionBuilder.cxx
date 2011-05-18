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

#include "Reflex/Builder/UnionBuilder.h"

#include "Reflex/Any.h"
#include "Reflex/Callback.h"
#include "Reflex/Member.h"
#include "DataMember.h"
#include "FunctionMemberTemplateInstance.h"
#include "Union.h"

//______________________________________________________________________________
//______________________________________________________________________________
//
//
//  UnionBuilderImpl
//
//

//______________________________________________________________________________
Reflex::UnionBuilderImpl::UnionBuilderImpl(const char* nam, size_t size, const std::type_info& ti, unsigned int modifiers /*= 0*/, TYPE typ /*=UNION*/):
   fUnion(0)
   ,
   fLastMember()
   ,
   fCallbackEnabled(true) {
// Construct union info.
   std::string nam2(nam);
   const Type& c = Type::ByName(nam2);

   if (c) {
      // We found a typedef to a class with the same name
      if (c.IsTypedef()) {
         nam2 += " @HIDDEN@";
      }
      // Type already exists. Check if it was a class, struct, or union.
      else if (!c.IsClass()) {
         throw RuntimeError("Attempt to replace a non-class type with a union"); // FIXME: We should not throw!
      }
   }
   fUnion = new Union(nam2.c_str(), size, ti, modifiers, typ);
}


//______________________________________________________________________________
Reflex::UnionBuilderImpl::~UnionBuilderImpl() {
// UnionBuilderImpl destructor. Used for call back functions (e.g. Cintex).
   if (fCallbackEnabled) {
      FireClassCallback(fUnion->ThisType());
   }
}


//______________________________________________________________________________
void
Reflex::UnionBuilderImpl::AddItem(const char* nam,
                                  const Type& typ) {
// Add data member info (internal).  !!! Obsolete, do not use.
   fLastMember = Member(new DataMember(nam, typ, 0, 0));
   fUnion->AddDataMember(fLastMember);
}


//______________________________________________________________________________
void
Reflex::UnionBuilderImpl::AddDataMember(const char* nam,
                                        const Type& typ,
                                        size_t offs,
                                        unsigned int modifiers /*= 0*/) {
// Add data member info (internal).
   fLastMember = Member(new DataMember(nam, typ, offs, modifiers));
   fUnion->AddDataMember(fLastMember);
}


//______________________________________________________________________________
void
Reflex::UnionBuilderImpl::AddFunctionMember(const char* nam,
                                            const Type& typ,
                                            StubFunction stubFP,
                                            void* stubCtx /*= 0*/,
                                            const char* params /*= 0*/,
                                            unsigned int modifiers /*= 0*/) {
// Add function member info (internal).
   if (Tools::IsTemplated(nam)) {
      fLastMember = Member(new FunctionMemberTemplateInstance(nam, typ, stubFP, stubCtx, params, modifiers, *(dynamic_cast<ScopeBase*>(fUnion))));
   } else {
      fLastMember = Member(new FunctionMember(nam, typ, stubFP, stubCtx, params, modifiers));
   }
   fUnion->AddFunctionMember(fLastMember);
}


//______________________________________________________________________________
void
Reflex::UnionBuilderImpl::AddProperty(const char* key,
                                      const char* value) {
// Attach property to this union as string.
   AddProperty(key, Any(value));
}


//______________________________________________________________________________
void
Reflex::UnionBuilderImpl::AddProperty(const char* key,
                                      Any value) {
// Attach property to this union as Any object.
   if (fLastMember) {
      fLastMember.Properties().AddProperty(key, value);
   } else {
      fUnion->Properties().AddProperty(key, value);
   }
}


//______________________________________________________________________________
void
Reflex::UnionBuilderImpl::EnableCallback(const bool enable /*= true*/) {
// Enable callback call in the destructor.
   fCallbackEnabled = enable;
}


//______________________________________________________________________________
void
Reflex::UnionBuilderImpl::SetSizeOf(size_t size) {
// Set the size of the union (internal).
   fUnion->SetSize(size);
}


//______________________________________________________________________________
Reflex::Type
Reflex::UnionBuilderImpl::ToType() {
// Return the type currently being built.
   return fUnion->ThisType();
}


//______________________________________________________________________________
//______________________________________________________________________________
//
//
//  UnionBuilder
//
//

//______________________________________________________________________________
Reflex::UnionBuilder::UnionBuilder(const char* nam, const std::type_info& ti, size_t size, unsigned int modifiers /*= 0*/, TYPE typ /*= UNION*/):
   fUnionBuilderImpl(nam, size, ti, modifiers, typ) {
// Constructor.
}


//______________________________________________________________________________
Reflex::UnionBuilder::~UnionBuilder() {
// Destructor.
}


//______________________________________________________________________________
Reflex::UnionBuilder&
Reflex::UnionBuilder::AddItem(const char* nam,
                              const char* typ) {
// !!! Obsolete, do not use.
   fUnionBuilderImpl.AddItem(nam, TypeBuilder(typ));
   return *this;
}


//______________________________________________________________________________
Reflex::UnionBuilder&
Reflex::UnionBuilder::AddDataMember(const Type& typ,
                                    const char* nam,
                                    size_t offs,
                                    unsigned int modifiers /*= 0*/) {
// Add data member info to this union.
   fUnionBuilderImpl.AddDataMember(nam, typ, offs, modifiers);
   return *this;
}


//______________________________________________________________________________
Reflex::UnionBuilder&
Reflex::UnionBuilder::AddFunctionMember(const Type& typ,
                                        const char* nam,
                                        StubFunction stubFP,
                                        void* stubCtx,
                                        const char* params,
                                        unsigned int modifiers) {
// Add function member info to this union.
   fUnionBuilderImpl.AddFunctionMember(nam, typ, stubFP, stubCtx, params, modifiers);
   return *this;
}


//______________________________________________________________________________
Reflex::UnionBuilder&
Reflex::UnionBuilder::EnableCallback(const bool enable /*= true*/) {
// Enable callback call in the destructor.
   fUnionBuilderImpl.EnableCallback(enable);
   return *this;
}


//______________________________________________________________________________
Reflex::UnionBuilder&
Reflex::UnionBuilder::SetSizeOf(size_t size) {
// Set the object / memory size of the union.
   fUnionBuilderImpl.SetSizeOf(size);
   return *this;
}


//______________________________________________________________________________
Reflex::Type
Reflex::UnionBuilder::ToType() {
// Get the union's Type object.
   return fUnionBuilderImpl.ToType();
}
