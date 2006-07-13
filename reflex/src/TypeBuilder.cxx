// @(#)root/reflex:$Name:  $:$Id: TypeBuilder.cxx,v 1.10 2006/07/04 15:02:55 roiser Exp $
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

#include "Reflex/Builder/TypeBuilder.h"

#include "Reflex/Type.h"
#include "Reflex/TypeName.h"

#include "Pointer.h"
#include "Function.h"
#include "Array.h"
#include "Enum.h"
#include "Typedef.h"
#include "PointerToMember.h"
#include "Reflex/Tools.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypeBuilder( const char * n, 
                                              unsigned int modifiers ) {
//-------------------------------------------------------------------------------
// Construct the type information for a type.
   Type ret = Type::ByName(n);
   if ( ret.Id() ) return Type(ret, modifiers);
   else {
      TypeName* tname = new TypeName(n, 0);
      std::string sname = Tools::GetScopeName(n);
      if ( ! Scope::ByName( sname ).Id() )  new ScopeName( sname.c_str(), 0 );
      return Type(tname, modifiers);
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::ConstBuilder(const Type & t) {
//-------------------------------------------------------------------------------
// Construct a const qualified type.
   unsigned int mod = CONST;
   if ( t.IsVolatile() ) mod |= VOLATILE;
   return Type(t,mod);    
}

//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::VolatileBuilder(const Type & t) {
//-------------------------------------------------------------------------------
// Construct a volatile qualified type.
   unsigned int mod = VOLATILE;
   if ( t.IsConst() )    mod |= CONST;
   return Type(t,mod);    
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::PointerBuilder( const Type & t,
                                                 const std::type_info & ti ) {
//-------------------------------------------------------------------------------
// Construct a pointer type.
   Type ret = Type::ByName( Pointer::BuildTypeName(t));
   if ( ret ) return ret;
   else       return (new Pointer(t, ti))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::PointerToMemberBuilder( const Type & t,
                                                         const Scope & s,
                                                         const std::type_info & ti ) {
//-------------------------------------------------------------------------------
// Construct a pointer type.
   Type ret = Type::ByName( PointerToMember::BuildTypeName(t,s));
   if ( ret ) return ret;
   else       return (new PointerToMember(t, s, ti))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::ReferenceBuilder(const Type & t) {
//-------------------------------------------------------------------------------
// Construct a "reference qualified" type.
   unsigned int mod = REFERENCE;
   if ( t.IsConst() )    mod |= CONST;
   if ( t.IsVolatile() ) mod |= VOLATILE;
   return Type(t,mod);    
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::ArrayBuilder( const Type & t, 
                                               size_t n,
                                               const std::type_info & ti ) {
//-------------------------------------------------------------------------------
// Construct an array type.
   Type ret = Type::ByName(Array::BuildTypeName(t,n));
   if ( ret ) return ret;
   else       return (new Array(t, n, ti))->ThisType();
}

//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::EnumTypeBuilder( const char * Name, 
                                                  const char * values,
                                                  const std::type_info & ti ) {
//-------------------------------------------------------------------------------
// Construct an enum type.
   Type ret = Type::ByName(Name);
   if ( ret ) return ret;
   Enum * e = new Enum(Name, ti);

   std::vector<std::string> valVec;
   Tools::StringSplit(valVec, values, ";");

   Type int_t = Type::ByName("int");
   for (std::vector<std::string>::const_iterator it = valVec.begin(); 
        it != valVec.end(); ++it ) {
      std::string iname, ivalue;
      Tools::StringSplitPair(iname, ivalue, *it, "=");
      long val = atol(ivalue.c_str());
      e->AddDataMember( iname.c_str(), int_t, val, 0);
   }  
   return e->ThisType();
}

//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::TypedefTypeBuilder(const char * Name, 
                                                    const Type & t) {
//-------------------------------------------------------------------------------
// Construct a typedef type.
   Type ret = Type::ByName(Name);
   if ( ret ) return ret;
   else       return (new Typedef( Name , t ))->ThisType();        
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder( const Type & r,
                                                      const std::vector<Type> & p,
                                                      const std::type_info & ti ) {
//-------------------------------------------------------------------------------
// Construct a function type.
   return (new Function( r, p, ti))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector< Type > v;
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}
 

//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}
 

//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}
 

//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector< Type > v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}
 

//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14) {
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}
 

//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21,
                                                     const Type & t22) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21,
                                                     const Type & t22,
                                                     const Type & t23) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21,
                                                     const Type & t22,
                                                     const Type & t23,
                                                     const Type & t24) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21,
                                                     const Type & t22,
                                                     const Type & t23,
                                                     const Type & t24,
                                                     const Type & t25) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21,
                                                     const Type & t22,
                                                     const Type & t23,
                                                     const Type & t24,
                                                     const Type & t25,
                                                     const Type & t26) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21,
                                                     const Type & t22,
                                                     const Type & t23,
                                                     const Type & t24,
                                                     const Type & t25,
                                                     const Type & t26,
                                                     const Type & t27) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21,
                                                     const Type & t22,
                                                     const Type & t23,
                                                     const Type & t24,
                                                     const Type & t25,
                                                     const Type & t26,
                                                     const Type & t27,
                                                     const Type & t28) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21,
                                                     const Type & t22,
                                                     const Type & t23,
                                                     const Type & t24,
                                                     const Type & t25,
                                                     const Type & t26,
                                                     const Type & t27,
                                                     const Type & t28,
                                                     const Type & t29) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21,
                                                     const Type & t22,
                                                     const Type & t23,
                                                     const Type & t24,
                                                     const Type & t25,
                                                     const Type & t26,
                                                     const Type & t27,
                                                     const Type & t28,
                                                     const Type & t29,
                                                     const Type & t30) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
                                           t30);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type ROOT::Reflex::FunctionTypeBuilder(const Type & r, 
                                                     const Type & t0, 
                                                     const Type & t1,
                                                     const Type & t2,
                                                     const Type & t3,
                                                     const Type & t4,
                                                     const Type & t5,
                                                     const Type & t6,
                                                     const Type & t7,
                                                     const Type & t8,
                                                     const Type & t9,
                                                     const Type & t10,
                                                     const Type & t11,
                                                     const Type & t12,
                                                     const Type & t13,
                                                     const Type & t14,
                                                     const Type & t15,
                                                     const Type & t16,
                                                     const Type & t17,
                                                     const Type & t18,
                                                     const Type & t19,
                                                     const Type & t20,
                                                     const Type & t21,
                                                     const Type & t22,
                                                     const Type & t23,
                                                     const Type & t24,
                                                     const Type & t25,
                                                     const Type & t26,
                                                     const Type & t27,
                                                     const Type & t28,
                                                     const Type & t29,
                                                     const Type & t30,
                                                     const Type & t31) { 
//-------------------------------------------------------------------------------
// Construct a function type.
   std::vector<Type> v = Tools::MakeVector(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
                                           t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
                                           t30, t31);
   Type ret = Type::ByName(Function::BuildTypeName(r,v));
   if ( ret ) return ret;
   else       return (new Function(r, v, typeid(UnknownType)))->ThisType();
}



