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

#include "Reflex/internal/TypeBase.h"

#include "Reflex/Type.h"
#include "Reflex/internal/OwnedPropertyList.h"
#include "Reflex/Object.h"
#include "Reflex/Scope.h"
#include "Reflex/internal/TypeName.h"
#include "Reflex/Base.h"
#include "Reflex/TypeTemplate.h"
#include "Reflex/DictionaryGenerator.h"


#include "Array.h"
#include "Pointer.h"
#include "PointerToMember.h"
#include "Union.h"
#include "Enum.h"
#include "Fundamental.h"
#include "Function.h"
#include "Class.h"
#include "Typedef.h"
#include "ClassTemplateInstance.h"
#include "FunctionMemberTemplateInstance.h"

#include "Reflex/Tools.h"

#include "Reflex/Builder/TypeBuilder.h"

//-------------------------------------------------------------------------------
Reflex::TypeBase::TypeBase(const char* nam,
                           size_t size,
                           TYPE typeTyp,
                           const std::type_info& ti,
                           const Type& finalType,
                           REPRESTYPE represType /*= REPRES_NOTYPE */):
   fTypeInfo(&ti),
   fRepresType(represType),
   fScope(Scope::__NIRVANA__()),
   fSize(size),
   fTypeType(typeTyp),
   fPropertyList(OwnedPropertyList(new PropertyListImpl())),
   fBasePosition(Tools::GetBasePosition(nam)),
   fFinalType(finalType.Id() ? new Type(finalType): 0),
   fRawType(0) {
//-------------------------------------------------------------------------------
// Construct the dictinary info for a type.
   Type t = TypeName::ByName(nam);

   if (t.Id() == 0) {
      fTypeName = new TypeName(nam, this, &ti);
   } else {
      fTypeName = (TypeName*) t.Id();

      if (t.Id() != TypeName::ByTypeInfo(ti).Id()) {
         fTypeName->SetTypeId(ti);
      }

      if (fTypeName->fTypeBase) {
         delete fTypeName->fTypeBase;
      }
      fTypeName->fTypeBase = this;
   }

   if (typeTyp != FUNDAMENTAL &&
       typeTyp != FUNCTION &&
       typeTyp != POINTER) {
      std::string sname = Tools::GetScopeName(nam);
      fScope = Scope::ByName(sname);

      if (fScope.Id() == 0) {
         ScopeName* sn = 0;
         Type scopeType = Type::ByName(sname);
         if (scopeType.Id()) {
            TypeName* scopeTypeName = (TypeName*) scopeType.Id();
            if (scopeTypeName->LiteralName().IsLiteral()) {
               sn = new ScopeName(Literal(scopeTypeName->Name()), 0);
            } else {
               sn = new ScopeName(sname.c_str(), 0);
            }
         } else {
            sn = new ScopeName(sname.c_str(), 0);
         }
         fScope = sn->ThisScope();
      }

      // Set declaring At
      if (fScope) {
         fScope.AddSubType(ThisType());
      }
   }
}


//-------------------------------------------------------------------------------
Reflex::TypeBase::~TypeBase() {
//-------------------------------------------------------------------------------
// Destructor.
   fPropertyList.Delete();

   if (fFinalType) {
      delete fFinalType;
   }

   if (fRawType) {
      delete fRawType;
   }

   if (fTypeName->fTypeBase == this) {
      fTypeName->fTypeBase = 0;
   }
}


//-------------------------------------------------------------------------------
Reflex::TypeBase::operator
Reflex::Scope() const {
//-------------------------------------------------------------------------------
// Conversion operator to Scope.
   switch (fTypeType) {
   case CLASS :
   case STRUCT:
   case TYPETEMPLATEINSTANCE:
   case UNION:
   case ENUM:
      {
         const ScopeBase* sb = dynamic_cast<const ScopeBase*>(this);
         if (!sb) return Dummy::Scope();
         return sb->ThisScope();
      }
   case TYPEDEF:
      return FinalType();
   default:
      return Dummy::Scope();
   }
}


//-------------------------------------------------------------------------------
Reflex::TypeBase::operator
Reflex::Type() const {
//-------------------------------------------------------------------------------
// Converison operator to Type.
   return ThisType();
}


//-------------------------------------------------------------------------------
void*
Reflex::TypeBase::Allocate() const {
//-------------------------------------------------------------------------------
// Allocate memory for this type.
   return new char[fSize];
}


//-------------------------------------------------------------------------------
size_t
Reflex::TypeBase::ArrayLength() const {
//-------------------------------------------------------------------------------
// Return the length of the array type.
   return 0;
}


//-------------------------------------------------------------------------------
void
Reflex::TypeBase::Deallocate(void* instance) const {
//-------------------------------------------------------------------------------
// Deallocate the memory for this type from instance.
   delete [] ((char*)instance);
}


//-------------------------------------------------------------------------------
Reflex::Scope
Reflex::TypeBase::DeclaringScope() const {
//-------------------------------------------------------------------------------
// Return the declaring scope of this type.
   return fScope;
}


//-------------------------------------------------------------------------------
Reflex::Object
Reflex::TypeBase::CastObject(const Type& /* to */,
                             const Object& /* obj */) const {
//-------------------------------------------------------------------------------
// Cast this type into "to" using object "obj"
   throw RuntimeError("This function can only be called on Class/Struct");
   return Dummy::Object();
}


//-------------------------------------------------------------------------------
//const Reflex::Object &
//Reflex::TypeBase::Construct( const Type &  /*signature*/,
//                                   const std::vector < Object > & /*values*/,
//                                   void * /*mem*/ ) const {
//-------------------------------------------------------------------------------
//  return Object(ThisType(), Allocate());
//}


//-------------------------------------------------------------------------------
Reflex::Object
Reflex::TypeBase::Construct(const Type& /*signature*/,
                            const std::vector<void*>& /*values*/,
                            void* /*mem*/) const {
//-------------------------------------------------------------------------------
// Construct this type.
   return Object(ThisType(), Allocate());
}


//-------------------------------------------------------------------------------
void
Reflex::TypeBase::Destruct(void* instance,
                           bool dealloc) const {
//-------------------------------------------------------------------------------
// Destruct this type.
   if (dealloc) {
      Deallocate(instance);
   }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeBase::DynamicType(const Object& /* obj */) const {
//-------------------------------------------------------------------------------
// Return the dynamic type info of this type.
   throw RuntimeError("Type::DynamicType can only be called on Class/Struct");
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeBase::FinalType() const {
//-------------------------------------------------------------------------------
// Return the type without typedefs information.
   if (fFinalType) {
      return *fFinalType;
   }

   fFinalType = new Type(DetermineFinalType(ThisType()));
   return *fFinalType;
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeBase::DetermineFinalType(const Type& t) const {
//-------------------------------------------------------------------------------
// Return the type t without typedefs information.

   Type retType(t);

   switch (t.TypeType()) {
   case TYPEDEF:
      retType = t.ToType().FinalType();
      break;
   case POINTER:
      retType = PointerBuilder(t.ToType().FinalType(), t.TypeInfo());
      break;
   case POINTERTOMEMBER:
      retType = PointerToMemberBuilder(t.ToType().FinalType(), t.PointerToMemberScope(), t.TypeInfo());
      break;
   case ARRAY:
      retType = ArrayBuilder(t.ToType().FinalType(), t.ArrayLength(), t.TypeInfo());
      break;
   case FUNCTION:
   {
      std::vector<Type> vecParFinal(t.FunctionParameterSize());

      for (size_t iPar = 0; iPar < t.FunctionParameterSize(); ++iPar) {
         vecParFinal[iPar] = t.FunctionParameterAt(iPar).FinalType();
      }
      retType = FunctionTypeBuilder(t.ReturnType().FinalType(), vecParFinal, t.TypeInfo());
      break;
   }
   case UNRESOLVED:
      return Dummy::Type();
   default:
      return t;
   } // switch

   // copy fModifiers
   if (t.IsConst()) {
      retType = ConstBuilder(retType);
   }

   if (t.IsReference()) {
      retType = ReferenceBuilder(retType);
   }

   if (t.IsVolatile()) {
      retType = VolatileBuilder(retType);
   }

   return retType;
} // DetermineFinalType


//-------------------------------------------------------------------------------
void
Reflex::TypeBase::HideName() const {
//-------------------------------------------------------------------------------
// Append the string " @HIDDEN@" to a type name.
   fTypeName->HideName();
}


//-------------------------------------------------------------------------------
void
Reflex::TypeBase::UnhideName() const {
   //-------------------------------------------------------------------------------
   // Remove the string " @HIDDEN@" to a type name.
   fTypeName->UnhideName();
}


//-------------------------------------------------------------------------------
std::string
Reflex::TypeBase::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of the type.
   if (0 != (mod & (SCOPED | S))) {
      return fTypeName->Name();
   }
   return fTypeName->Name() + fBasePosition;
}


//-------------------------------------------------------------------------------
const char*
Reflex::TypeBase::SimpleName(size_t& pos,
                             unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of the type.
   if (0 != (mod & (SCOPED | S))) {
      pos = 0;
   } else {
      pos = fBasePosition;
   }
   return fTypeName->Name();
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeBase::FunctionParameterAt(size_t /* nth */) const {
//-------------------------------------------------------------------------------
// Return the nth function parameter type.
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
size_t
Reflex::TypeBase::FunctionParameterSize() const {
//-------------------------------------------------------------------------------
// Return the number of function parameters.
   return 0;
}


//-------------------------------------------------------------------------------
Reflex::Scope
Reflex::TypeBase::PointerToMemberScope() const {
//-------------------------------------------------------------------------------
// Return the scope of a pointer to member type.
   return Dummy::Scope();
}


//-------------------------------------------------------------------------------
Reflex::PropertyList
Reflex::TypeBase::Properties() const {
//-------------------------------------------------------------------------------
// Return the property list attached to this type.
   return fPropertyList;
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeBase::RawType() const {
//-------------------------------------------------------------------------------
// Return the raw type of this type, removing all info of pointers, arrays, typedefs.
   if (fRawType) {
      return *fRawType;
   }

   Type rawType = ThisType();

   while (true) {
      switch (rawType.TypeType()) {
      case POINTER:
      case POINTERTOMEMBER:
      case TYPEDEF:
      case ARRAY:
         rawType = rawType.ToType();
         break;
      case UNRESOLVED:
         return Dummy::Type();
      default:
         fRawType = new Type(*rawType.ToTypeBase());
         return *fRawType;
      }
   }
} // RawType


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeBase::ReturnType() const {
//-------------------------------------------------------------------------------
// Return the function return type.
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeBase::TemplateArgumentAt(size_t /* nth */) const {
//-------------------------------------------------------------------------------
// Return the nth template argument.
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeBase::ToType() const {
//-------------------------------------------------------------------------------
// Return the underlying type.
   return Dummy::Type();
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeBase::ThisType() const {
//-------------------------------------------------------------------------------
// Return the Type object pointing to this TypeBase.
   return fTypeName->ThisType();
}


//-------------------------------------------------------------------------------
std::string
Reflex::TypeBase::TypeTypeAsString() const {
//-------------------------------------------------------------------------------
// Return the kind of type as a string.
   switch (fTypeType) {
   case CLASS:
      return "CLASS";
      break;
   case STRUCT:
      return "STRUCT";
      break;
   case ENUM:
      return "ENUM";
      break;
   case FUNCTION:
      return "FUNCTION";
      break;
   case ARRAY:
      return "ARRAY";
      break;
   case FUNDAMENTAL:
      return "FUNDAMENTAL";
      break;
   case POINTER:
      return "POINTER";
      break;
   case TYPEDEF:
      return "TYPEDEF";
      break;
   case TYPETEMPLATEINSTANCE:
      return "TYPETEMPLATEINSTANCE";
      break;
   case MEMBERTEMPLATEINSTANCE:
      return "MEMBERTEMPLATEINSTANCE";
      break;
   case UNRESOLVED:
      return "UNRESOLVED";
      break;
   default:
      return "Type " + Name() + "is not assigned to a TYPE";
   } // switch
} // TypeTypeAsString


//-------------------------------------------------------------------------------
Reflex::TYPE
Reflex::TypeBase::TypeType() const {
//-------------------------------------------------------------------------------
// Return the kind of type as an enum.
   return fTypeType;
}


//-------------------------------------------------------------------------------
void
Reflex::TypeBase::GenerateDict(DictionaryGenerator& /* generator */) const {
//-------------------------------------------------------------------------------
// Generate Dictionary information about itself.
}
