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

#include "DataMember.h"

#include "Reflex/Scope.h"
#include "Reflex/Object.h"
#include "Reflex/Member.h"
#include "Reflex/DictionaryGenerator.h"

#include "Reflex/Tools.h"
#include "Class.h"
#include <cstring>


//-------------------------------------------------------------------------------
Reflex::DataMember::DataMember(const char* nam, const Type& typ, size_t offs, unsigned int modifiers, char* interpreterOffset):
   MemberBase(nam, typ, DATAMEMBER, modifiers),
   fOffset(offs),
   fInterpreterOffset(interpreterOffset) {
   // Construct the dictionary information for a data member.
}


//-------------------------------------------------------------------------------
Reflex::DataMember::~DataMember() {
//-------------------------------------------------------------------------------
// Data member destructor.
}


//-------------------------------------------------------------------------------
std::string
Reflex::DataMember::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the scoped and qualified (if requested with mod) name of the data member
   std::string s;
   s.reserve(30); // an arbitrary, reasonable number

   if (0 != (mod & (QUALIFIED | Q))) {
      if (IsPublic()) {
         s += "public ";
      }

      if (IsProtected()) {
         s += "protected ";
      }

      if (IsPrivate()) {
         s += "private ";
      }

      if (IsExtern()) {
         s += "extern ";
      }

      if (IsStatic()) {
         s += "static ";
      }

      if (IsAuto()) {
         s += "auto ";
      }

      if (IsRegister()) {
         s += "register ";
      }

      if (IsMutable()) {
         s += "mutable ";
      }
   }

   if (mod & SCOPED && DeclaringScope().IsEnum()) {
      if (DeclaringScope().DeclaringScope()) {
         std::string sc = DeclaringScope().DeclaringScope().Name(SCOPED);

         if (sc != "::") {
            s += sc + "::";
         }
      }
      s += MemberBase::Name(mod & ~SCOPED);
   } else {
      s += MemberBase::Name(mod);
   }

   return s;
} // Name


//-------------------------------------------------------------------------------
Reflex::Object
Reflex::DataMember::Get(const Object& obj) const {
//-------------------------------------------------------------------------------
// Get the value of this data member as stored in object obj.
   if (DeclaringScope().ScopeType() == ENUM) {
      return Object(Type::ByName("int"), (void*) &fOffset);
   } else {
      void* mem = CalculateBaseObject(obj);
      mem = (char*) mem + Offset();
      return Object(TypeOf(), mem);
   }
}


/*/-------------------------------------------------------------------------------
   void Reflex::DataMember::Set( const Object & instance,
   const Object & value ) const {
   //-------------------------------------------------------------------------------
   void * mem = CalculateBaseObject( instance );
   mem = (char*)mem + Offset();
   if (TypeOf().IsClass() ) {
   // Should use the asigment operator if exists (FIX-ME)
   memcpy( mem, value.Address(), TypeOf().SizeOf());
   }
   else {
   memcpy( mem, value.Address(), TypeOf().SizeOf() );
   }
   }
 */


//-------------------------------------------------------------------------------
void
Reflex::DataMember::Set(const Object& instance,
                        const void* value) const {
//-------------------------------------------------------------------------------
// Set the data member value in object instance.
   void* mem = CalculateBaseObject(instance);
   mem = (char*) mem + Offset();

   if (TypeOf().IsClass()) {
      // Should use the asigment operator if exists (FIX-ME)
      memcpy(mem, value, TypeOf().SizeOf());
   } else {
      memcpy(mem, value, TypeOf().SizeOf());
   }
}


//-------------------------------------------------------------------------------
void
Reflex::DataMember::GenerateDict(DictionaryGenerator& generator) const {
//-------------------------------------------------------------------------------
// Generate Dictionary information about itself.

   const Scope& declScope = DeclaringScope();

   if (declScope.IsUnion()) {
      // FIXME

   } else if (declScope.IsEnum()) {
      std::stringstream tmp;
      tmp << Offset();

      if (declScope.DeclaringScope().IsNamespace()) {
         generator.AddIntoInstances("\n.AddItem(\"" + Name() + "\", " + tmp.str() + ")");
      } else { // class, struct
         generator.AddIntoFree(Name() + "=" + tmp.str());
      }
   } else { // class, struct
      const Type& rType = TypeOf().RawType();

      if (TypeOf().IsArray()) {
         Type t = TypeOf();

         std::stringstream temp;
         temp << t.ArrayLength();

         generator.AddIntoShadow(t.ToType().Name(SCOPED) + " " + Name() + "[" + temp.str() + "];\n");

      } else if (TypeOf().IsPointer() && TypeOf().RawType().IsFunction()) {
         Type t = TypeOf().ToType();
         generator.AddIntoShadow(t.ReturnType().Name(SCOPED) + "(");

         if (t.DeclaringScope().IsClass()) {
            generator.AddIntoShadow(t.DeclaringScope().Name(SCOPED) + "::");
         }

         generator.AddIntoShadow("*" + t.Name() + ")(");

         for (size_t parameters = 0; parameters < t.FunctionParameterSize();
              ++parameters) {
            generator.AddIntoShadow(t.FunctionParameterAt(parameters).Name());

            if (t.FunctionParameterSize() > parameters) {
               generator.AddIntoShadow(",");
            }
         }

         generator.AddIntoShadow(");\n");

      } else {
         std::string tname = TypeOf().Name(SCOPED);

         if ((rType.IsClass() || rType.IsStruct()) && (!rType.IsPublic())) {
            tname = generator.Replace_colon(rType.Name(SCOPED));

            if (rType != TypeOf()) {
               tname = tname + TypeOf().Name(SCOPED).substr(tname.length());
            }
         }
         generator.AddIntoShadow(tname + " " + Name() + ";\n");
      }

      //register type and get its number
      std::string typenumber = generator.GetTypeNumber(TypeOf());

      generator.AddIntoFree(".AddDataMember(type_" + typenumber + ", \"" + Name() + "\", ");
      generator.AddIntoFree("OffsetOf (__shadow__::" +
                            generator.Replace_colon((*this).DeclaringScope().Name(SCOPED)));
      generator.AddIntoFree(", " + Name() + "), ");

      if (IsPublic()) {
         generator.AddIntoFree("PUBLIC");
      } else if (IsPrivate()) {
         generator.AddIntoFree("PRIVATE");
      } else if (IsProtected()) {
         generator.AddIntoFree("PROTECTED");
      }

      if (IsVirtual()) {
         generator.AddIntoFree(" | VIRTUAL");
      }

      if (IsArtificial()) {
         generator.AddIntoFree(" | ARTIFICIAL");
      }

      generator.AddIntoFree(")\n");

   }
} // GenerateDict
