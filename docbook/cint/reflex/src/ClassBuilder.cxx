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
#include "Reflex/Builder/OnDemandBuilderForScope.h"


//______________________________________________________________________________
//______________________________________________________________________________
//
//
//  ClassBuilderImpl
//
//

//______________________________________________________________________________
Reflex::ClassBuilderImpl::ClassBuilderImpl(const char* nam, const std::type_info& ti, size_t size, unsigned int modifiers, TYPE typ):
   fClass(0),
   fLastMember(),
   fNewClass(true),
   fCallbackEnabled(true) {
   // -- Construct a class information (internal).
   std::string nam2(nam);
   Type c = Type::ByName(nam2);

   if (c) {
      // We found a typedef to a class with the same name
      if (c.IsTypedef()) {
         nam2 += " @HIDDEN@";
         nam = nam2.c_str();
         c = Dummy::Type();
      }
      // Class already exists. Check if it was a class.
      else if (!c.IsClass()) {
         throw RuntimeError("Attempt to replace a non-class type with a class");
      }
   }

   if (!c) {
      if (Tools::IsTemplated(nam)) {
         fClass = new ClassTemplateInstance(nam, size, ti, modifiers);
      } else {
         fClass = new Class(nam, size, ti, modifiers, typ);
      }
   } else {
      fNewClass = false;
      fClass = const_cast<Class*>(dynamic_cast<const Class*>(c.ToTypeBase()));

      if (!fClass) {
         throw RuntimeError("Attempt to replace a non-class type with a class");
      }

      if (fClass->SizeOf() == 0) {
         fClass->SetSize(size);
      } else if (size != 0 && fClass->SizeOf() != size) {
         throw RuntimeError(std::string("Attempt to change the size of the class ") + std::string(nam));
      }

      if (!strcmp(fClass->TypeInfo().name(), typeid(Reflex::UnknownType).name())) {
         fClass->SetTypeInfo(ti);
      } else if (strcmp(fClass->TypeInfo().name(), ti.name())) {
         throw RuntimeError(std::string("Attempt to change the type_info of the class ") + std::string(nam));
      }

      if (modifiers != 0) {
         if (0 == fClass->Modifiers()) {
            fClass->SetModifiers(modifiers);
         } else if (fClass->Modifiers() != modifiers) {
            throw RuntimeError(std::string("Attempt to change the modifiers of the class ") + std::string(nam));
         }
      }
   }
}


//______________________________________________________________________________
Reflex::ClassBuilderImpl::ClassBuilderImpl(Class* cl):
   fClass(cl),
   fLastMember(),
   fNewClass(false),
   fCallbackEnabled(true) {  
}

//______________________________________________________________________________
Reflex::ClassBuilderImpl::~ClassBuilderImpl() {
   // -- ClassBuilderImpl destructor. Used for call back functions (e.g. Cintex).
   if (fCallbackEnabled) {
      FireClassCallback(fClass->ThisType());
   }
   
}


//______________________________________________________________________________
void
Reflex::ClassBuilderImpl::AddBase(const Type& bas,
                                  OffsetFunction offsFP,
                                  unsigned int modifiers) {
   // -- Add base class information (internal).
   if (!fNewClass) {
      for (Reflex::Base_Iterator iter = fClass->Base_Begin(); iter != fClass->Base_End(); ++iter) {
         if (iter->Name() == bas.Name()) {
            // Already entered, ignore.
            return;
//            if ( offsFP != iter->Offset() ) {
//               throw RuntimeError(std::string("Attempt to change the offset of a base class (")+bas.Name()+" of the class \"")+fClass.Name());
//            }
         }
      }
   }
   fClass->AddBase(bas, offsFP, modifiers);
}


//______________________________________________________________________________
void
Reflex::ClassBuilderImpl::AddDataMember(const char* nam,
                                        const Type& typ,
                                        size_t offs,
                                        unsigned int modifiers) {
   // -- Add data member info (internal).
   if (!fNewClass) {
      for (Reflex::Member_Iterator iter = fClass->DataMember_Begin(); iter != fClass->DataMember_End(); ++iter) {
         if (iter->Name() == nam) {
            if (offs != 0 && iter->Offset() != offs) {
               throw RuntimeError(std::string("Attempt to change the offset of a data member (") + nam + ") of the class " + fClass->Name());
            }

            if (typ && typ != iter->TypeOf()) {
               throw RuntimeError(std::string("Attempt to change the type of a data member (") + nam + ") of the class " + fClass->Name());
            }
            return;
         }
      }
   }
   fLastMember = Member(new DataMember(nam, typ, offs, modifiers));
   fClass->AddDataMember(fLastMember);
} // AddDataMember


//______________________________________________________________________________
void
Reflex::ClassBuilderImpl::AddFunctionMember(const char* nam,
                                            const Type& typ,
                                            StubFunction stubFP,
                                            void* stubCtx,
                                            const char* params,
                                            unsigned int modifiers) {
   // -- Add function member info (internal).
   if (!fNewClass) {
      for (Reflex::Member_Iterator iter = fClass->DataMember_Begin(); iter != fClass->DataMember_End(); ++iter) {
         if (iter->Name() == nam && typ && typ == iter->TypeOf()) {
            return;
         }
      }
   }

   if (Tools::IsTemplated(nam)) {
      fLastMember = Member(new FunctionMemberTemplateInstance(nam, typ, stubFP, stubCtx, params, modifiers, *(dynamic_cast<ScopeBase*>(fClass))));
   } else {
      fLastMember = Member(new FunctionMember(nam, typ, stubFP, stubCtx, params, modifiers));
   }
   fClass->AddFunctionMember(fLastMember);
} // AddFunctionMember


//______________________________________________________________________________
void
Reflex::ClassBuilderImpl::AddTypedef(const Type& typ,
                                     const char* def) {
   // -- Add typedef info (internal).
   Type ret = Type::ByName(def);

   // Check for typedef AA AA;
   if (ret == typ && !typ.IsTypedef()) {
      if (typ) {
         typ.ToTypeBase()->HideName();
      } else {
         ((TypeName*) typ.Id())->HideName();
      }
   }
   // We found the typedef type
   else if (ret) {
      fClass->AddSubType(ret);
   }
   // Create a new typedef
   else {
      new Typedef(def, typ);
   }
} // AddTypedef


//______________________________________________________________________________
void
Reflex::ClassBuilderImpl::AddEnum(const char* nam,
                                  const char* values,
                                  const std::type_info* ti,
                                  unsigned int modifiers) {
   // -- Add enum info (internal).

   // This does not work because the EnumTypeBuilder does a definition of the enum
   // not only a declaration. (It is called in the dictionary header already)
   //   EnumTypeBuilder(nam, values, *ti, modifiers);

   Enum* e = new Enum(nam, *ti, modifiers);

   std::vector<std::string> valVec = std::vector<std::string>();
   Tools::StringSplit(valVec, values, ";");

   for (
      std::vector<std::string>::const_iterator it = valVec.begin();
      it != valVec.end();
      ++it
   ) {
      std::string name;
      std::string value;
      Tools::StringSplitPair(name, value, *it, "=");
      unsigned long int valInt = atol(value.c_str());
      e->AddDataMember(Member(new DataMember(name.c_str(), Type::ByName("int"), valInt, 0)));
   }
} // AddEnum


//______________________________________________________________________________
void
Reflex::ClassBuilderImpl::AddProperty(const char* key,
                                      const char* value) {
   // -- Add property info (internal).
   AddProperty(key, Any(value));
}


//______________________________________________________________________________
void
Reflex::ClassBuilderImpl::AddProperty(const char* key,
                                      Any value) {
   // -- Add property info (internal).
   if (fLastMember) {
      fLastMember.Properties().AddProperty(key, value);
   } else {
      fClass->Properties().AddProperty(key, value);
   }
}


//______________________________________________________________________________
void
Reflex::ClassBuilderImpl::EnableCallback(bool enable /*= true*/) {
   // Enable (or disable) the calling of the callback when this object is
   // destructed.

   fCallbackEnabled = enable;
}


//-------------------------------------------------------------------------------
void
Reflex::ClassBuilderImpl::AddOnDemandFunctionMemberBuilder(OnDemandBuilderForScope* odb) {
//-------------------------------------------------------------------------------
// Register an on demand builder with this class.
// The builder odb is able to provide on demand building for elements
// specified by kind.
   fClass->RegisterOnDemandBuilder(odb, ScopeBase::kBuildFunctionMembers);
   odb->SetContext(fClass);
}


//-------------------------------------------------------------------------------
void
Reflex::ClassBuilderImpl::AddOnDemandDataMemberBuilder(OnDemandBuilderForScope* odb) {
//-------------------------------------------------------------------------------
// Register an on demand builder with this class.
// The builder odb is able to provide on demand building for elements
// specified by kind.
   fClass->RegisterOnDemandBuilder(odb, ScopeBase::kBuildDataMembers);
   odb->SetContext(fClass);
}


//______________________________________________________________________________
void
Reflex::ClassBuilderImpl::SetSizeOf(size_t size) {
   // -- Set the size of the class (internal).
   fClass->SetSize(size);
}


//______________________________________________________________________________
Reflex::Type
Reflex::ClassBuilderImpl::ToType() {
   // -- Return the type currently being built.
   return fClass->ThisType();
}


//______________________________________________________________________________
//______________________________________________________________________________
//
//
//  ClassBuilder
//
//

//______________________________________________________________________________
Reflex::ClassBuilder::ClassBuilder(const char* nam, const std::type_info& ti, size_t size, unsigned int modifiers, TYPE typ):
   fClassBuilderImpl(nam, ti, size, modifiers, typ) {
   // -- Constructor
}


//______________________________________________________________________________
Reflex::ClassBuilder::ClassBuilder(Class* cl):
   fClassBuilderImpl(cl) {
   // -- Constructor
}


//______________________________________________________________________________
Reflex::ClassBuilder::~ClassBuilder() {
   // -- Destructor
}


//______________________________________________________________________________
Reflex::ClassBuilder&
Reflex::ClassBuilder::AddBase(const Type& bas,
                              OffsetFunction offsFP,
                              unsigned int modifiers) {
   // -- Add base class information to this class.
   fClassBuilderImpl.AddBase(bas, offsFP, modifiers);
   return *this;
}


//______________________________________________________________________________
Reflex::ClassBuilder&
Reflex::ClassBuilder::AddDataMember(const Type& typ,
                                    const char* nam,
                                    size_t offs,
                                    unsigned int modifiers) {
   // -- Add data member info to this class.
   fClassBuilderImpl.AddDataMember(nam, typ, offs, modifiers);
   return *this;
}


//______________________________________________________________________________
Reflex::ClassBuilder&
Reflex::ClassBuilder::AddFunctionMember(const Type& typ,
                                        const char* nam,
                                        StubFunction stubFP,
                                        void* stubCtx,
                                        const char* params,
                                        unsigned int modifiers) {
   // -- Add function member info to this class.
   fClassBuilderImpl.AddFunctionMember(nam, typ, stubFP, stubCtx, params, modifiers);
   return *this;
}


//______________________________________________________________________________
Reflex::ClassBuilder&
Reflex::ClassBuilder::AddTypedef(const char* typ,
                                 const char* def) {
   // -- Add typedef info to this class.
   fClassBuilderImpl.AddTypedef(TypeBuilder(typ), def);
   return *this;
}


//______________________________________________________________________________
Reflex::ClassBuilder&
Reflex::ClassBuilder::AddTypedef(const Type& typ,
                                 const char* def) {
   // -- Add typedef info to this class.
   fClassBuilderImpl.AddTypedef(typ, def);
   return *this;
}


//______________________________________________________________________________
Reflex::ClassBuilder&
Reflex::ClassBuilder::AddEnum(const char* nam,
                              const char* values,
                              const std::type_info* ti,
                              unsigned int modifiers) {
   // -- Add enum info to this class.
   fClassBuilderImpl.AddEnum(nam, values, ti, modifiers);
   return *this;
}


/*
   //______________________________________________________________________________
   Reflex::ClassBuilder& Reflex::ClassBuilder::addUnion(const char* nam, const char* values, unsigned int modifiers)
   {
   fClassBuilderImpl.addUnion(nam, values, modifiers);
   return *this;
   }
 */

//-------------------------------------------------------------------------------
Reflex::ClassBuilder&
Reflex::ClassBuilder::AddOnDemandDataMemberBuilder(OnDemandBuilderForScope* odb) {
//-------------------------------------------------------------------------------
// Register an on demand builder with this class.
   fClassBuilderImpl.AddOnDemandDataMemberBuilder(odb);
   return *this;
}

//-------------------------------------------------------------------------------
Reflex::ClassBuilder&
Reflex::ClassBuilder::AddOnDemandFunctionMemberBuilder(OnDemandBuilderForScope* odb) {
//-------------------------------------------------------------------------------
// Register an on demand builder with this class.
   fClassBuilderImpl.AddOnDemandFunctionMemberBuilder(odb);
   return *this;
}

//______________________________________________________________________________
Reflex::ClassBuilder&
Reflex::ClassBuilder::EnableCallback(bool enable /*=true*/) {
   // Enable (or disable) the calling of the callback when this object is
   // destructed.

   fClassBuilderImpl.EnableCallback(enable);
   return *this;
}


//______________________________________________________________________________
Reflex::ClassBuilder&
Reflex::ClassBuilder::SetSizeOf(size_t size) {
// Set the class's size.
   fClassBuilderImpl.SetSizeOf(size);
   return *this;
}


//______________________________________________________________________________
Reflex::Type
Reflex::ClassBuilder::ToType() {
   // -- Return the type currently being built.
   return fClassBuilderImpl.ToType();
}
