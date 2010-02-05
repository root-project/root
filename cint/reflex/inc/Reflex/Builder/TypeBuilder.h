// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_TypeBuilder
#define Reflex_TypeBuilder

// Include files
#include "Reflex/Type.h"
#include "Reflex/Tools.h"

#include <vector>

#if defined(__ICC)
# define OffsetOf(c1, mem) (long (&((volatile const char &)((c1*) 0)->mem)))
#else
# define OffsetOf(c1, mem) ((size_t) (&reinterpret_cast<const volatile char&>(((c1*) 64)->mem)) - 64)
#endif

namespace Reflex {
template<typename FUNC>
void* FuncToVoidPtr(FUNC f) {
   union Cnv_t {
      Cnv_t(FUNC ff): fFunc(ff) {}
      FUNC fFunc;
      void* fPtr;
   } u(f);
   return u.fPtr;
}
template <typename FUNC>
FUNC VoidPtrToFunc(void* p) {
   union Cnv_t {
      Cnv_t(void* pp): fPtr(pp) {}
      FUNC fFunc;
      void* fPtr;
   } u(p);
   return u.fFunc;
}

class RFLX_API Literal {
public:
   Literal(const char*);
   ~Literal();
   operator const char*() const { return fPtr; }
 private:
   const char* fPtr;
};

RFLX_API Type TypeBuilder(const char* n,
                          unsigned int modifiers = 0);


RFLX_API Type ConstBuilder(const Type& t);


RFLX_API Type VolatileBuilder(const Type& t);


RFLX_API Type PointerBuilder(const Type& t,
                             const std::type_info& ti = typeid(UnknownType));


RFLX_API Type PointerToMemberBuilder(const Type& t,
                                     const Scope& s,
                                     const std::type_info& ti = typeid(UnknownType));


RFLX_API Type ReferenceBuilder(const Type& t);


RFLX_API Type ArrayBuilder(const Type& t,
                           size_t n,
                           const std::type_info& ti = typeid(UnknownType));

RFLX_API Type EnumTypeBuilder(const char*,
                              const char* items = "",
                              const std::type_info& ti = typeid(UnknownType),
                              unsigned int modifiers = 0);

RFLX_API Type TypedefTypeBuilder(const char* Name,
                                 const Type& t,
                                 REPRESTYPE represType = REPRES_NOTYPE);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const std::vector<Reflex::Type>& p,
                                  const std::type_info& ti = typeid(UnknownType));


RFLX_API Type FunctionTypeBuilder(const Type& r);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const Type& t0);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const Type& t0,
                                  const Type& t1);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const Type& t0,
                                  const Type& t1,
                                  const Type& t2);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const Type& t0,
                                  const Type& t1,
                                  const Type& t2,
                                  const Type& t3);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const Type& t0,
                                  const Type& t1,
                                  const Type& t2,
                                  const Type& t3,
                                  const Type& t4);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const Type& t0,
                                  const Type& t1,
                                  const Type& t2,
                                  const Type& t3,
                                  const Type& t4,
                                  const Type& t5);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const Type& t0,
                                  const Type& t1,
                                  const Type& t2,
                                  const Type& t3,
                                  const Type& t4,
                                  const Type& t5,
                                  const Type& t6);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const Type& t0,
                                  const Type& t1,
                                  const Type& t2,
                                  const Type& t3,
                                  const Type& t4,
                                  const Type& t5,
                                  const Type& t6,
                                  const Type& t7);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const Type& t0,
                                  const Type& t1,
                                  const Type& t2,
                                  const Type& t3,
                                  const Type& t4,
                                  const Type& t5,
                                  const Type& t6,
                                  const Type& t7,
                                  const Type& t8);


RFLX_API Type FunctionTypeBuilder(const Type& r,
                                  const Type& t0,
                                  const Type& t1,
                                  const Type& t2,
                                  const Type& t3,
                                  const Type& t4,
                                  const Type& t5,
                                  const Type& t6,
                                  const Type& t7,
                                  const Type& t8,
                                  const Type& t9);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t10);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t11);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t12);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t13);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t14);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t15);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t16);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t17);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t18);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t19);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t20);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t21);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t22);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t23);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t24);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t25);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t26);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t27);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t28);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t29);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t30);


RFLX_API Type FunctionTypeBuilder(const Type& r,
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
                                  const Type& t31);


/**
 * offsetOf will calculate the Offset of a data MemberAt relative
 * to the start of the class
 * @param MemberAt the pointer to the data MemberAt
 * @return the Offset of the data MemberAt
 */
template <typename C, typename M>
size_t
offsetOf(M C::* member) {
   return (size_t) &((((C*) 0)->*member));
}


/**
 * @struct BaseOffset TypeBuilder.h Reflex/Builder/TypeBuilder.h
 * provide the static function that calculates the Offset between  BaseAt classes
 */
template <typename C, typename B>
class BaseOffset {
public:
   static size_t
   Offset(void* o) { return (size_t) (B*) (C*) o - (size_t) (C*) o; }

   static OffsetFunction
   Get() { return &BaseOffset::Offset; }

};


/**
 * @struct TypeDistiller TypeBuilder.h Reflex/Builder/TypeBuilder.h
 * @author Pere Mato
 * @date 29/07/2004
 * @ingroup RefBld
 */
template <typename T> class TypeDistiller {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(T));

      if (!t.Id()) { t = Type::ByName(Tools::Demangle(typeid(T))); }

      if (t.Id()) { return t; } else { return TypeBuilder(Tools::Demangle(typeid(T)).c_str()); }
   }


};


/** */
template <typename T> class TypeDistiller<T*> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(T*));

      if (t) { return t; } else { return PointerBuilder(TypeDistiller<T>::Get(), typeid(T*)); }
   }


};


/** */
template <typename T, size_t N> class TypeDistiller<T[N]> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(T*));

      if (t) { return t; } else { return ArrayBuilder(TypeDistiller<T>::Get(), N, typeid(NullType)); }
   }


};


/**  */
template <typename T> class TypeDistiller<const T> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(T));

      if (t) { return Type(t, CONST); } else { return Type(TypeDistiller<T>::Get(), CONST); }
   }


};


/**  */
template <typename T> class TypeDistiller<volatile T> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(T));

      if (t) { return Type(t, VOLATILE); } else { return Type(TypeDistiller<T>::Get(), VOLATILE); }
   }


};


/** */
template <typename T> class TypeDistiller<const volatile T> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(T));

      if (t) { return Type(t, CONST | VOLATILE); } else { return Type(TypeDistiller<T>::Get(), CONST | VOLATILE); }
   }


};


/** */
template <typename T> class TypeDistiller<T&> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(T));

      if (t) { return Type(t, REFERENCE); } else { return Type(TypeDistiller<T>::Get(), REFERENCE); }
   }


};


/** */
template <typename T> class TypeDistiller<const T&> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(T));

      if (t) { return Type(t, CONST | REFERENCE); } else { return Type(TypeDistiller<T>::Get(), CONST | REFERENCE); }
   }


};


/** */
template <typename T> class TypeDistiller<volatile T&> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(T));

      if (t) { return Type(t, VOLATILE | REFERENCE); } else { return Type(TypeDistiller<T>::Get(), VOLATILE | REFERENCE); }
   }


};


/** */
template <typename T> class TypeDistiller<const volatile T&> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(T));

      if (t) { return Type(t, CONST | VOLATILE | REFERENCE); } else { return Type(TypeDistiller<T>::Get(), CONST | VOLATILE | REFERENCE); }
   }


};

#ifndef TYPEDISTILLER_STRING_SPECIALIZATION
# define TYPEDISTILLER_STRING_SPECIALIZATION
template <> class TypeDistiller<std::string> {
public:
   static Type
   Get() {
      return TypeBuilder("std::basic_string<char>");
   }


};
#endif

/**
 * getType will return a reference to a Type (create it if necessery)
 * representating the type of the template parameter
 * @return reference to Type
 */
template <typename T>
const Type&
GetType() {
   static Type t = TypeDistiller<T>::Get();
   return t;
}


/**
 * @struct FunctionDistiller TypeBuilder.h Reflex/Builder/TypeBuilder.h
 * @author Pere Mato
 * @date 29/07/2004
 * @ingroup RefBld
 */
template <typename S> class FunctionDistiller;

// This define is necessary for all Sun Forte compilers with version < 5.5 (SunWSpro8)
#if ((defined(__SUNPRO_CC)) && (__SUNPRO_CC < 0x550))
# define __R_TN__ typename
#else
# define __R_TN__
#endif

/** */
template <typename R>
class FunctionDistiller<R(void)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(void)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             std::vector<Type>(),
                                                             typeid(R(void))); }
   }


};

/** */
template <typename R, typename T0>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get()),
                                                             typeid(R(T0))); }
   }


};

/** */
template <typename R, typename T0, typename T1>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get()),
                                                             typeid(R(T0, T1))); }
   }


};

/** */
template <typename R, typename T0, typename T1, typename T2>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get()),
                                                             typeid(R(T0, T1, T2))); }
   }


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get()),
                                                             typeid(R(T0, T1, T2, T3))); }
   }


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4))); }
   }


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5))); }
   }


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5,
                                   __R_TN__ T6)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get(),
                                                                               TypeDistiller<T6>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5, T6))); }
   }


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5,
                                   __R_TN__ T6, __R_TN__ T7)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get(),
                                                                               TypeDistiller<T6>::Get(),
                                                                               TypeDistiller<T7>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5, T6, T7))); }
   }


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5,
                                   __R_TN__ T6, __R_TN__ T7, __R_TN__ T8)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get(),
                                                                               TypeDistiller<T6>::Get(),
                                                                               TypeDistiller<T7>::Get(),
                                                                               TypeDistiller<T8>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8))); }
   }


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5,
                                   __R_TN__ T6, __R_TN__ T7, __R_TN__ T8,
                                   __R_TN__ T9)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get(),
                                                                               TypeDistiller<T6>::Get(),
                                                                               TypeDistiller<T7>::Get(),
                                                                               TypeDistiller<T8>::Get(),
                                                                               TypeDistiller<T9>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9))); }
   }


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename T10>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5,
                                   __R_TN__ T6, __R_TN__ T7, __R_TN__ T8,
                                   __R_TN__ T9, __R_TN__ T10)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get(),
                                                                               TypeDistiller<T6>::Get(),
                                                                               TypeDistiller<T7>::Get(),
                                                                               TypeDistiller<T8>::Get(),
                                                                               TypeDistiller<T9>::Get(),
                                                                               TypeDistiller<T10>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10))); }
   } // Get


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename T10, typename T11>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5,
                                   __R_TN__ T6, __R_TN__ T7, __R_TN__ T8,
                                   __R_TN__ T9, __R_TN__ T10, __R_TN__ T11)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get(),
                                                                               TypeDistiller<T6>::Get(),
                                                                               TypeDistiller<T7>::Get(),
                                                                               TypeDistiller<T8>::Get(),
                                                                               TypeDistiller<T9>::Get(),
                                                                               TypeDistiller<T10>::Get(),
                                                                               TypeDistiller<T11>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11))); }
   } // Get


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename T10, typename T11,
          typename T12>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5,
                                   __R_TN__ T6, __R_TN__ T7, __R_TN__ T8,
                                   __R_TN__ T9, __R_TN__ T10, __R_TN__ T11,
                                   __R_TN__ T12)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get(),
                                                                               TypeDistiller<T6>::Get(),
                                                                               TypeDistiller<T7>::Get(),
                                                                               TypeDistiller<T8>::Get(),
                                                                               TypeDistiller<T9>::Get(),
                                                                               TypeDistiller<T10>::Get(),
                                                                               TypeDistiller<T11>::Get(),
                                                                               TypeDistiller<T12>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12))); }
   } // Get


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename T10, typename T11,
          typename T12, typename T13>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5,
                                   __R_TN__ T6, __R_TN__ T7, __R_TN__ T8,
                                   __R_TN__ T9, __R_TN__ T10, __R_TN__ T11,
                                   __R_TN__ T12, __R_TN__ T13)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get(),
                                                                               TypeDistiller<T6>::Get(),
                                                                               TypeDistiller<T7>::Get(),
                                                                               TypeDistiller<T8>::Get(),
                                                                               TypeDistiller<T9>::Get(),
                                                                               TypeDistiller<T10>::Get(),
                                                                               TypeDistiller<T11>::Get(),
                                                                               TypeDistiller<T12>::Get(),
                                                                               TypeDistiller<T13>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13))); }
   } // Get


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename T10, typename T11,
          typename T12, typename T13, typename T14>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5,
                                   __R_TN__ T6, __R_TN__ T7, __R_TN__ T8,
                                   __R_TN__ T9, __R_TN__ T10, __R_TN__ T11,
                                   __R_TN__ T12, __R_TN__ T13, __R_TN__ T14)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get(),
                                                                               TypeDistiller<T6>::Get(),
                                                                               TypeDistiller<T7>::Get(),
                                                                               TypeDistiller<T8>::Get(),
                                                                               TypeDistiller<T9>::Get(),
                                                                               TypeDistiller<T10>::Get(),
                                                                               TypeDistiller<T11>::Get(),
                                                                               TypeDistiller<T12>::Get(),
                                                                               TypeDistiller<T13>::Get(),
                                                                               TypeDistiller<T14>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14))); }
   } // Get


};

/** */
template <typename R, typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename T10, typename T11,
          typename T12, typename T13, typename T14, typename T15>
class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2,
                                   __R_TN__ T3, __R_TN__ T4, __R_TN__ T5,
                                   __R_TN__ T6, __R_TN__ T7, __R_TN__ T8,
                                   __R_TN__ T9, __R_TN__ T10, __R_TN__ T11,
                                   __R_TN__ T12, __R_TN__ T13, __R_TN__ T14,
                                   __R_TN__ T15)> {
public:
   static Type
   Get() {
      Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15)));

      if (t) { return t; } else { return FunctionTypeBuilder(TypeDistiller<R>::Get(),
                                                             Tools::MakeVector(TypeDistiller<T0>::Get(),
                                                                               TypeDistiller<T1>::Get(),
                                                                               TypeDistiller<T2>::Get(),
                                                                               TypeDistiller<T3>::Get(),
                                                                               TypeDistiller<T4>::Get(),
                                                                               TypeDistiller<T5>::Get(),
                                                                               TypeDistiller<T6>::Get(),
                                                                               TypeDistiller<T7>::Get(),
                                                                               TypeDistiller<T8>::Get(),
                                                                               TypeDistiller<T9>::Get(),
                                                                               TypeDistiller<T10>::Get(),
                                                                               TypeDistiller<T11>::Get(),
                                                                               TypeDistiller<T12>::Get(),
                                                                               TypeDistiller<T13>::Get(),
                                                                               TypeDistiller<T14>::Get(),
                                                                               TypeDistiller<T15>::Get()),
                                                             typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15))); }
   } // Get


};

#undef __R_TN__
// end of the Sun Forte CC fix

} // namespace Reflex

#endif
