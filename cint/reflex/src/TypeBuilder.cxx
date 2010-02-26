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

#include "Reflex/Builder/TypeBuilder.h"

#include "Reflex/Type.h"
#include "Reflex/internal/TypeName.h"

#include "Pointer.h"
#include "Function.h"
#include "Array.h"
#include "Enum.h"
#include "Typedef.h"
#include "PointerToMember.h"
#include "Reflex/Tools.h"
#include "Reflex/internal/LiteralString.h"

//-------------------------------------------------------------------------------
Reflex::Literal::Literal(const char* s): fPtr(s) {
//-------------------------------------------------------------------------------
// Construct a temporary Literal, adding s to LiteralString's list of string
// literals.
   LiteralString::Add(s);
}

//-------------------------------------------------------------------------------
Reflex::Literal::~Literal() {
//-------------------------------------------------------------------------------
// Destruct a temporary Literal, removing s from LiteralString's list of string
// literals.
   LiteralString::Remove(fPtr);
}

//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypeBuilder(const char* n,
                    unsigned int modifiers) {
//-------------------------------------------------------------------------------
// Construct the type information for a type.
   Reflex::Instance initialize_reflex;
   const Type& ret = Type::ByName(n);

   if (ret.Id()) {
      return Type(ret, modifiers);
   } else {
      TypeName* tname = new TypeName(n, 0);
      std::string sname = Tools::GetScopeName(n);

      if (!Scope::ByName(sname).Id()) {
         Type scopeType = Type::ByName(sname);
         if (scopeType.Id()) {
            TypeName* scopeTypeName = (TypeName*) scopeType.Id();
            if (scopeTypeName->LiteralName().IsLiteral()) {
               new ScopeName(Literal(scopeTypeName->Name()), 0);
            } else {
               new ScopeName(sname.c_str(), 0);
            }
         } else {
            new ScopeName(sname.c_str(), 0);
         }
      }
      return Type(tname, modifiers);
   }
} // TypeBuilder


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::ConstBuilder(const Type& t) {
//-------------------------------------------------------------------------------
// Construct a const qualified type.
   unsigned int mod = CONST;

   if (t.IsVolatile()) {
      mod |= VOLATILE;
   }
   return Type(t, mod);
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::VolatileBuilder(const Type& t) {
//-------------------------------------------------------------------------------
// Construct a volatile qualified type.
   unsigned int mod = VOLATILE;

   if (t.IsConst()) {
      mod |= CONST;
   }
   return Type(t, mod);
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::PointerBuilder(const Type& t,
                       const std::type_info& ti) {
//-------------------------------------------------------------------------------
// Construct a pointer type.
   const Type& ret = Type::ByName(Pointer::BuildTypeName(t));

   if (ret) {
      return ret;
   } else { return (new Pointer(t, ti))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::PointerToMemberBuilder(const Type& t,
                               const Scope& s,
                               const std::type_info& ti) {
//-------------------------------------------------------------------------------
// Construct a pointer type.
   const Type& ret = Type::ByName(PointerToMember::BuildTypeName(t, s));

   if (ret) {
      return ret;
   } else { return (new PointerToMember(t, s, ti))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::ReferenceBuilder(const Type& t) {
//-------------------------------------------------------------------------------
// Construct a "reference qualified" type.
   unsigned int mod = REFERENCE;

   if (t.IsConst()) {
      mod |= CONST;
   }

   if (t.IsVolatile()) {
      mod |= VOLATILE;
   }
   return Type(t, mod);
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::ArrayBuilder(const Type& t,
                     size_t n,
                     const std::type_info& ti) {
//-------------------------------------------------------------------------------
// Construct an array type.
   const Type& ret = Type::ByName(Array::BuildTypeName(t, n));

   if (ret) {
      return ret;
   } else { return (new Array(t, n, ti))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::EnumTypeBuilder(const char* nam,
                        const char* values,
                        const std::type_info& ti,
                        unsigned int modifiers) {
//-------------------------------------------------------------------------------
// Construct an enum type.

   std::string nam2(nam);

   const Type& ret = Type::ByName(nam2);

   if (ret) {
      if (ret.IsTypedef()) {
         nam2 += " @HIDDEN@";
         nam = nam2.c_str();
      } else { return ret; }
   }

   Enum* e = new Enum(nam, ti, modifiers);

   std::vector<std::string> valVec;
   Tools::StringSplit(valVec, values, ";");

   const Type& int_t = Type::ByName("int");

   for (std::vector<std::string>::const_iterator it = valVec.begin();
        it != valVec.end(); ++it) {
      std::string iname, ivalue;
      Tools::StringSplitPair(iname, ivalue, *it, "=");
      long val = atol(ivalue.c_str());
      e->AddDataMember(iname.c_str(), int_t, val, 0);
   }
   return e->ThisType();
} // EnumTypeBuilder


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::TypedefTypeBuilder(const char* nam,
                           const Type& t,
                           REPRESTYPE represType) {
//-------------------------------------------------------------------------------
// Construct a typedef type.
   Type ret = Type::ByName(nam);

   // Check for typedef AA AA;
   if (ret == t && !t.IsTypedef()) {
      if (t) {
         t.ToTypeBase()->HideName();
      } else { ((TypeName*) t.Id())->HideName(); }
   }
   // We found the typedef type
   else if (ret) {
      return ret;
   }
   // Create a new typedef
   return (new Typedef(nam, t, Reflex::TYPEDEF, Reflex::Dummy::Type(), represType))->ThisType();
} // TypedefTypeBuilder


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const std::vector<Type>& p,
                            const std::type_info& ti) {
//-------------------------------------------------------------------------------
// Construct a function type.
   const Type& ret = Type::ByName(Function::BuildTypeName(r, p));

   if (ret && ret.TypeInfo() == ti) {
      return ret;
   } else {
      return (new Function(r, p, ti))->ThisType();
   }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v;
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21,
                            const Type& t22) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21,
                            const Type& t22,
                            const Type& t23) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21,
                            const Type& t22,
                            const Type& t23,
                            const Type& t24) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21,
                            const Type& t22,
                            const Type& t23,
                            const Type& t24,
                            const Type& t25) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21,
                            const Type& t22,
                            const Type& t23,
                            const Type& t24,
                            const Type& t25,
                            const Type& t26) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21,
                            const Type& t22,
                            const Type& t23,
                            const Type& t24,
                            const Type& t25,
                            const Type& t26,
                            const Type& t27) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21,
                            const Type& t22,
                            const Type& t23,
                            const Type& t24,
                            const Type& t25,
                            const Type& t26,
                            const Type& t27,
                            const Type& t28) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21,
                            const Type& t22,
                            const Type& t23,
                            const Type& t24,
                            const Type& t25,
                            const Type& t26,
                            const Type& t27,
                            const Type& t28,
                            const Type& t29) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21,
                            const Type& t22,
                            const Type& t23,
                            const Type& t24,
                            const Type& t25,
                            const Type& t26,
                            const Type& t27,
                            const Type& t28,
                            const Type& t29,
                            const Type& t30) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
                                           t30);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::FunctionTypeBuilder(const Type& r,
                            const Type& t0,
                            const Type& t1,
                            const Type& t2,
                            const Type& t3,
                            const Type& t4,
                            const Type& t5,
                            const Type& t6,
                            const Type& t7,
                            const Type& t8,
                            const Type& t9,
                            const Type& t10,
                            const Type& t11,
                            const Type& t12,
                            const Type& t13,
                            const Type& t14,
                            const Type& t15,
                            const Type& t16,
                            const Type& t17,
                            const Type& t18,
                            const Type& t19,
                            const Type& t20,
                            const Type& t21,
                            const Type& t22,
                            const Type& t23,
                            const Type& t24,
                            const Type& t25,
                            const Type& t26,
                            const Type& t27,
                            const Type& t28,
                            const Type& t29,
                            const Type& t30,
                            const Type& t31) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
                                           t30, t31);
   const Type& ret = Type::ByName(Function::BuildTypeName(r, v));

   if (ret) {
      return ret;
   } else { return (new Function(r, v, typeid(UnknownType)))->ThisType(); }
}
