// @(#)root/reflex:$Name:  $:$Id: TypeBuilder.h,v 1.5 2005/11/30 13:22:05 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_TypeBuilder
#define ROOT_Reflex_TypeBuilder

// Include files
#include "Reflex/Type.h"
#include "Reflex/Tools.h"

#include <vector>

#if defined(__ICC)
#define OffsetOf(c1,mem) (int(&(((c1*)0)->mem)))
#define OffsetOf2(c1,c2,mem) (int(&(((c1,c2*)0)->mem)))
#define OffsetOf3(c1,c2,c3,mem) (int(&(((c1,c2,c3*)0)->mem)))
#define OffsetOf4(c1,c2,c3,c4,mem) (int(&(((c1,c2,c3,c4*)0)->mem)))
#define OffsetOf5(c1,c2,c3,c4,c5,mem) (int(&(((c1,c2,c3,c4,c5*)0)->mem)))
#define OffsetOf6(c1,c2,c3,c4,c5,c6,mem) (int(&(((c1,c2,c3,c4,c5,c6*)0)->mem)))
#define OffsetOf7(c1,c2,c3,c4,c5,c6,c7,mem) (int(&(((c1,c2,c3,c4,c5,c6,c7*)0)->mem)))
#define OffsetOf8(c1,c2,c3,c4,c5,c6,c7,c8,mem) (int(&(((c1,c2,c3,c4,c5,c6,c7,c8*)0)->mem)))
#define OffsetOf9(c1,c2,c3,c4,c5,c6,c7,c8,c9,mem) (int(&(((c1,c2,c3,c4,c5,c6,c7,c8,c9*)0)->mem)))
#define OffsetOf10(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,mem) (int(&(((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10*)0)->mem)))
#define OffsetOf11(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,mem) (int(&(((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11*)0)->mem)))
#define OffsetOf12(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,mem) (int(&(((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12*)0)->mem)))
#define OffsetOf13(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,mem) (int(&(((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13*)0)->mem)))
#define OffsetOf14(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,mem) (int(&(((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14*)0)->mem)))
#define OffsetOf15(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,mem) (int(&(((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15*)0)->mem)))
#define OffsetOf16(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,mem) (int(&(((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16*)0)->mem)))
#else
#define OffsetOf(c1,mem) ((size_t)(&((c1*)64)->mem)-64)
#define OffsetOf2(c1,c2,mem) ((size_t)(&((c1,c2*)64)->mem)-64)
#define OffsetOf3(c1,c2,c3,mem) ((size_t)(&((c1,c2,c3*)64)->mem)-64)
#define OffsetOf4(c1,c2,c3,c4,mem) ((size_t)(&((c1,c2,c3,c4*)64)->mem)-64)
#define OffsetOf5(c1,c2,c3,c4,c5,mem) ((size_t)(&((c1,c2,c3,c4,c5*)64)->mem)-64)
#define OffsetOf6(c1,c2,c3,c4,c5,c6,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6*)64)->mem)-64)
#define OffsetOf7(c1,c2,c3,c4,c5,c6,c7,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6,c7*)64)->mem)-64)
#define OffsetOf8(c1,c2,c3,c4,c5,c6,c7,c8,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6,c7,c8*)64)->mem)-64)
#define OffsetOf9(c1,c2,c3,c4,c5,c6,c7,c8,c9,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6,c7,c8,c9*)64)->mem)-64)
#define OffsetOf10(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10*)64)->mem)-64)
#define OffsetOf11(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11*)64)->mem)-64)
#define OffsetOf12(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12*)64)->mem)-64)
#define OffsetOf13(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13*)64)->mem)-64)
#define OffsetOf14(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14*)64)->mem)-64)
#define OffsetOf15(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15*)64)->mem)-64)
#define OffsetOf16(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,mem) ((size_t)(&((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16*)64)->mem)-64)
#endif

namespace ROOT{ 
   namespace Reflex{

      Type TypeBuilder( const char * n,
                        unsigned int modifiers = 0 );


      Type ConstBuilder( const Type & t );


      Type VolatileBuilder( const Type & t );


      Type PointerBuilder( const Type & t,
                           const std::type_info & ti = typeid(UnknownType));


      Type ReferenceBuilder( const Type & t );


      Type ArrayBuilder( const Type & t, 
                         size_t n,
                         const std::type_info & ti = typeid(UnknownType));

      Type EnumTypeBuilder( const char *, 
                            const char * items = "",
                            const std::type_info & ti = typeid(UnknownType));

      Type TypedefTypeBuilder( const char * Name, 
                               const Type & t );


      Type FunctionTypeBuilder( const Type & r,
                                const std::vector<Type> & p,
                                const std::type_info & ti = typeid(UnknownType));


    Type FunctionTypeBuilder(const Type & r);


    Type FunctionTypeBuilder(const Type & r,   const Type & t0);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21, const Type & t22);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21, const Type & t22,
                             const Type & t23);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21, const Type & t22,
                             const Type & t23, const Type & t24);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21, const Type & t22,
                             const Type & t23, const Type & t24, const Type & t25);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21, const Type & t22,
                             const Type & t23, const Type & t24, const Type & t25,
                             const Type & t26);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21, const Type & t22,
                             const Type & t23, const Type & t24, const Type & t25,
                             const Type & t26, const Type & t27);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21, const Type & t22,
                             const Type & t23, const Type & t24, const Type & t25,
                             const Type & t26, const Type & t27, const Type & t28);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21, const Type & t22,
                             const Type & t23, const Type & t24, const Type & t25,
                             const Type & t26, const Type & t27, const Type & t28,
                             const Type & t29);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21, const Type & t22,
                             const Type & t23, const Type & t24, const Type & t25,
                             const Type & t26, const Type & t27, const Type & t28,
                             const Type & t29, const Type & t30);
  

    Type FunctionTypeBuilder(const Type & r,   const Type & t0,  const Type & t1, 
                             const Type & t2,  const Type & t3,  const Type & t4, 
                             const Type & t5,  const Type & t6,  const Type & t7,
                             const Type & t8,  const Type & t9,  const Type & t10, 
                             const Type & t11, const Type & t12, const Type & t13, 
                             const Type & t14, const Type & t15, const Type & t16,
                             const Type & t17, const Type & t18, const Type & t19,
                             const Type & t20, const Type & t21, const Type & t22,
                             const Type & t23, const Type & t24, const Type & t25,
                             const Type & t26, const Type & t27, const Type & t28,
                             const Type & t29, const Type & t30, const Type & t31);
  

      /**
       * offsetOf will calculate the Offset of a data MemberAt relative
       * to the start of the class
       * @param MemberAt the pointer to the data MemberAt
       * @return the Offset of the data MemberAt
       */
      template < typename C, typename M >
         size_t offsetOf( M C::* member )  {
         return (size_t) & (((C*)0)->*member); 
      }


      /**
       * @struct BaseOffset TypeBuilder.h Reflex/Builder/TypeBuilder.h
       * provide the static function that calculates the Offset between  BaseAt classes
       */
      template < typename C, typename B >
         class BaseOffset {
         public:
            static size_t Offset (void * o ) { return (size_t)(B*)(C*)o - (size_t)(C*)o; } 
            static OffsetFunction Get() { return  & BaseOffset::Offset; }
         };

    
      /** 
       * @struct TypeDistiller TypeBuilder.h Reflex/Builder/TypeBuilder.h
       * @author Pere Mato
       * @date 29/07/2004
       * @ingroup RefBld
       */
      template<typename T> class TypeDistiller {
      public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(T));
            if ( ! t.Id() ) t = Type::ByName(Tools::Demangle(typeid(T)));
            if ( t.Id() ) return t;
            else return TypeBuilder(Tools::Demangle(typeid(T)).c_str());
         }
      };


      /** */
      template<typename T> class TypeDistiller<T *> {
      public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(T*));
            if ( t ) return t;
            else return PointerBuilder(TypeDistiller<T>::Get(),typeid(T *));
         }
      };


      /** */
      template<typename T, size_t N > class TypeDistiller<T[N]> {
      public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(T*));
            if ( t ) return t;
            else return ArrayBuilder(TypeDistiller<T>::Get(),N,typeid(NullType));
         }
      };


      /**  */
      template<typename T> class TypeDistiller<const T> {
      public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(T));
            if ( t ) return Type( t, CONST );
            else return TypeBuilder(TypeDistiller<T>::Get().Name().c_str(),CONST);
         }
      };


      /**  */
      template<typename T> class TypeDistiller<volatile T> {
      public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(T));
            if ( t ) return Type( t, VOLATILE );
            else return TypeBuilder(TypeDistiller<T>::Get().Name().c_str(),VOLATILE);
         }
      };


      /** */
      template<typename T> class TypeDistiller<const volatile T> {
      public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(T));
            if ( t ) return Type( t, CONST | VOLATILE );
            else return TypeBuilder(TypeDistiller<T>::Get().Name().c_str(),CONST|VOLATILE);
         }
      };


      /** */
      template<typename T> class TypeDistiller<T &> {
      public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(T));
            if ( t ) return Type( t, REFERENCE );
            else return TypeBuilder(TypeDistiller<T>::Get().Name().c_str(),REFERENCE);
         }
      };


      /** */
      template<typename T> class TypeDistiller<const T &> {
      public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(T));
            if ( t ) return Type( t, CONST | REFERENCE );
            else return TypeBuilder(TypeDistiller<T>::Get().Name().c_str(),CONST|REFERENCE);
         }
      };


      /** */
      template<typename T> class TypeDistiller<volatile T &> {
      public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(T));
            if ( t ) return Type( t, VOLATILE | REFERENCE );
            else return TypeBuilder(TypeDistiller<T>::Get().Name().c_str(),VOLATILE|REFERENCE);
         }
      };


      /** */
      template<typename T> class TypeDistiller<const volatile T &> {
      public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(T));
            if ( t ) return Type( t, CONST | VOLATILE | REFERENCE );
            else return TypeBuilder(TypeDistiller<T>::Get().Name().c_str(),CONST|VOLATILE|REFERENCE);
         }
      };


      /**
       * getType will return a pointer to a Type (create it if necessery) 
       * representating the At of the template FunctionParameterAt
       * @return pointer to Type
       */
      template < typename T > 
         Type GetType() {
         return TypeDistiller<T>::Get();
      }


      /** 
       * @struct FuntionDistiller TypeBuilder.h Reflex/Builder/TypeBuilder.h
       * @author Pere Mato
       * @date 29/07/2004
       * @ingroup RefBld
       */
      template< typename S > class FunctionDistiller;

      // This define is necessary for all Sun Forte compilers with version < 5.5 (SunWSpro8)
#if ( (defined(__SUNPRO_CC)) && (__SUNPRO_CC<0x550) )
#define __R_TN__ typename
#else
#define __R_TN__
#endif

      /** */
      template< typename R > 
         class FunctionDistiller<R(void)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(void)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(), 
                                                 std::vector<Type>(), 
                                                 typeid(R(void))); 
         }
      };

      /** */
      template < typename R, typename T0 > 
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(), 
                                                 Tools::MakeVector( TypeDistiller<T0>::Get() ), 
                                                 typeid(R(T0))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1 > 
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1)));
            if ( t ) return t;
            else return FunctionTypeBuilder( TypeDistiller<R>::Get(), 
                                             Tools::MakeVector( TypeDistiller<T0>::Get(), 
                                                                TypeDistiller<T1>::Get()),
                                             typeid(R(T0, T1))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
                                                                    TypeDistiller<T1>::Get(),
                                                                    TypeDistiller<T2>::Get()), 
                                                 typeid(R(T0, T1, T2))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
                                                                    TypeDistiller<T1>::Get(),
                                                                    TypeDistiller<T2>::Get(), 
                                                                    TypeDistiller<T3>::Get()), 
                                                 typeid(R(T0, T1, T2, T3))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
                                                                    TypeDistiller<T1>::Get(),
                                                                    TypeDistiller<T2>::Get(), 
                                                                    TypeDistiller<T3>::Get(), 
                                                                    TypeDistiller<T4>::Get()), 
                                                 typeid(R(T0, T1, T2, T3, T4))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
                                                                    TypeDistiller<T1>::Get(),
                                                                    TypeDistiller<T2>::Get(), 
                                                                    TypeDistiller<T3>::Get(), 
                                                                    TypeDistiller<T4>::Get(), 
                                                                    TypeDistiller<T5>::Get()), 
                                                 typeid(R(T0, T1, T2, T3, T4, T5))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5, 
                                             __R_TN__ T6)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
                                                                    TypeDistiller<T1>::Get(),
                                                                    TypeDistiller<T2>::Get(), 
                                                                    TypeDistiller<T3>::Get(), 
                                                                    TypeDistiller<T4>::Get(), 
                                                                    TypeDistiller<T5>::Get(), 
                                                                    TypeDistiller<T6>::Get()), 
                                                 typeid(R(T0, T1, T2, T3, T4, T5, T6))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6, typename T7 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5, 
                                             __R_TN__ T6, __R_TN__ T7)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
                                                                    TypeDistiller<T1>::Get(),
                                                                    TypeDistiller<T2>::Get(), 
                                                                    TypeDistiller<T3>::Get(), 
                                                                    TypeDistiller<T4>::Get(), 
                                                                    TypeDistiller<T5>::Get(), 
                                                                    TypeDistiller<T6>::Get(),
                                                                    TypeDistiller<T7>::Get()), 
                                                 typeid(R( T0, T1, T2, T3, T4, T5, T6, T7))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6, typename T7,
         typename T8 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5, 
                                             __R_TN__ T6, __R_TN__ T7, __R_TN__ T8)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
                                                                    TypeDistiller<T1>::Get(),
                                                                    TypeDistiller<T2>::Get(), 
                                                                    TypeDistiller<T3>::Get(), 
                                                                    TypeDistiller<T4>::Get(), 
                                                                    TypeDistiller<T5>::Get(), 
                                                                    TypeDistiller<T6>::Get(),
                                                                    TypeDistiller<T7>::Get(), 
                                                                    TypeDistiller<T8>::Get()), 
                                                 typeid(R( T0, T1, T2, T3, T4, T5, T6, T7, T8)));
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6, typename T7,
         typename T8, typename T9 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5, 
                                             __R_TN__ T6, __R_TN__ T7, __R_TN__ T8, 
                                             __R_TN__ T9)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
                                                                    TypeDistiller<T1>::Get(),
                                                                    TypeDistiller<T2>::Get(), 
                                                                    TypeDistiller<T3>::Get(), 
                                                                    TypeDistiller<T4>::Get(), 
                                                                    TypeDistiller<T5>::Get(), 
                                                                    TypeDistiller<T6>::Get(),
                                                                    TypeDistiller<T7>::Get(), 
                                                                    TypeDistiller<T8>::Get(), 
                                                                    TypeDistiller<T9>::Get()), 
                                                 typeid(R( T0, T1, T2, T3, T4, T5, T6, T7, T8, T9))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6, typename T7,
         typename T8, typename T9, typename T10 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5, 
                                             __R_TN__ T6, __R_TN__ T7, __R_TN__ T8, 
                                             __R_TN__ T9, __R_TN__ T10)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
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
                                                 typeid(R( T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6, typename T7,
         typename T8, typename T9, typename T10, typename T11 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5, 
                                             __R_TN__ T6, __R_TN__ T7, __R_TN__ T8, 
                                             __R_TN__ T9, __R_TN__ T10, __R_TN__ T11)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
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
                                                 typeid(R( T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6, typename T7,
         typename T8, typename T9, typename T10, typename T11,
         typename T12 >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5, 
                                             __R_TN__ T6, __R_TN__ T7, __R_TN__ T8, 
                                             __R_TN__ T9, __R_TN__ T10, __R_TN__ T11, 
                                             __R_TN__ T12)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
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
                                                 typeid(R( T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6, typename T7,
         typename T8, typename T9, typename T10, typename T11,
         typename T12, typename T13  >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5, 
                                             __R_TN__ T6, __R_TN__ T7, __R_TN__ T8, 
                                             __R_TN__ T9, __R_TN__ T10, __R_TN__ T11, 
                                             __R_TN__ T12, __R_TN__ T13)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
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
                                                 typeid(R( T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6, typename T7,
         typename T8, typename T9, typename T10, typename T11,
         typename T12, typename T13, typename T14  >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5, 
                                             __R_TN__ T6, __R_TN__ T7, __R_TN__ T8, 
                                             __R_TN__ T9, __R_TN__ T10, __R_TN__ T11, 
                                             __R_TN__ T12, __R_TN__ T13, __R_TN__ T14)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
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
                                                 typeid(R( T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14))); 
         }
      };

      /** */
      template < typename R, typename T0, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6, typename T7,
         typename T8, typename T9, typename T10, typename T11,
         typename T12, typename T13, typename T14, typename T15  >
         class FunctionDistiller<__R_TN__ R(__R_TN__ T0, __R_TN__ T1, __R_TN__ T2, 
                                             __R_TN__ T3, __R_TN__ T4, __R_TN__ T5, 
                                             __R_TN__ T6, __R_TN__ T7, __R_TN__ T8, 
                                             __R_TN__ T9, __R_TN__ T10, __R_TN__ T11, 
                                             __R_TN__ T12, __R_TN__ T13, __R_TN__ T14, 
                                             __R_TN__ T15)> {
         public:
         static Type Get() {
            Type t = Type::ByTypeInfo(typeid(R(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15)));
            if ( t ) return t;
            else     return FunctionTypeBuilder( TypeDistiller<R>::Get(),
                                                 Tools::MakeVector( TypeDistiller<T0>::Get(),
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
                                                 typeid(R( T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15))); 
         }
      };

#undef __R_TN__
      // end of the Sun Forte CC fix

   } // namespace Reflex 
} // namespace ROOT

#endif // ROOT_Reflex_TypeBuilder
