// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Object
#define Reflex_Object

// Include files
#include "Reflex/Type.h"
#include <string>
#include <vector>

namespace Reflex {
// forward declarations

/**
 * @class Object Object.h Reflex/Object.h
 * @author Stefan Roiser
 * @date 24/06/2004
 * @ingroup Ref
 */
class RFLX_API Object {
public:
   /** constructor */
   Object(const Type& type = Type(),
          void* mem = 0);


   /** constructor */
   Object(const Object &);


   /** destructor */
   ~Object() {}


   template <typename T>
   static Object
   Create(T& v) {
      return Object(Type::ByTypeInfo(typeid(T)), &v);
   }


   /**
    * operator assigment
    */
   Object operator =(const Object& obj);


   /**
    * operator ==
    */
   bool operator ==(const Object& obj);


   /**
    * inequal operator
    */
   bool operator !=(const Object& obj);


   /**
    * operator bool
    */
   operator bool() const;


   /**
    * Address will return the memory address of the object
    * @return memory address of object
    */
   void* Address() const;


   /**
    * CastObject an object from this class type to another one
    * @param  to is the class type to cast into
    * @param  obj the memory address of the object to be casted
    */
   Object CastObject(const Type& to) const;


   /**
    * Destruct will call the destructor of a type and remove its memory
    * allocation if desired
    */
   void Destruct() const;


   /**
    * DynamicType is used to discover the dynamic type (useful in
    * case of polymorphism)
    * @return the actual class of the object
    */
   Type DynamicType() const;


   /**
    * Get the data member value
    * @param dm name of the data member to get
    * @return member value as object
    */
   Object Get(const std::string& dm) const;


   /**
    * Invoke a member function of the object
    * @param fm name of the member function
    * @param ret Object to put the return value into (can be 0 for function returning void)
    * @param args a vector of memory addresses to parameter values
    * @return the return value of the function as object
    */
   void Invoke(const std::string& fm,
               Object* ret = 0,
               const std::vector<void*>& args = std::vector<void*>()) const;


   /**
    * Invoke a member function of the object
    * @param fm name of the member function
    * @param ret Object to put the return value into (can be 0 for function returning void)
    * @param args a vector of memory addresses to parameter values
    * @return the return value of the function as object
    */
   template <typename T>
   void
   Invoke(const std::string& fm,
          T& ret,
          const std::vector<void*>& args = std::vector<void*>()) const {
      Object retO(Type::ByTypeInfo(typeid(T)), &ret);
      Invoke(fm, &retO, args);
   }


   /**
    * Invoke a member function of the object
    * @param fm name of the member function
    * @param sign the signature of the member function (for overloads)
    * @param ret Object to put the return value into (can be 0 for function returning void)
    * @param args a vector of memory addresses to parameter values
    * @return the return value of the function as object
    */
   void Invoke(const std::string& fm,
               const Type& sign,
               Object* ret = 0,
               const std::vector<void*>& args = std::vector<void*>()) const;


   /**
    * Invoke a member function of the object
    * @param fm name of the member function
    * @param sign the signature of the member function (for overloads)
    * @param ret Object to put the return value into (can be 0 for function returning void)
    * @param args a vector of memory addresses to parameter values
    * @return the return value of the function as object
    */
   template <typename T>
   void
   Invoke(const std::string& fm,
          const Type& sign,
          T& ret,
          const std::vector<void*>& args = std::vector<void*>()) const {
      Object retO(Type::ByTypeInfo(typeid(T)), &ret);
      Invoke(fm, sign, &retO, args);
   }


   /**
    * Set will set a data member value of this object
    * @param dm the name of the data member
    * @param value the memory address of the value to set
    */
   void Set(const std::string& dm,
            const void* value) const;


   /**
    * Set will set a data member value of this object
    * @param dm the name of the data member
    * @param value the memory address of the value to set
    */
   template <class T>
   void Set(const std::string& dm,
            const T& value) const;


   /**
    * TypeOf will return the type of the object
    * @return type of the object
    */
   Type TypeOf() const;

private:
   friend class ValueObject;

   /** */
   void Set2(const std::string& dm,
             const void* value) const;

   /**
    * the type of the object
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 1
    * @label object type
    **/
   Type fType;


   /**
    * the address of the object
    */
   mutable
   void* fAddress;

};    // class Object


/**
 * Object_Cast can be used to cast an object into a given type
 * (no additional checks are performed for the time being)
 * @param o the object to be casted
 * @return the address of the object casted into type T
 */
template <class T> T Object_Cast(const Object& o);


} // namespace Reflex

#include "Reflex/Tools.h"

//-------------------------------------------------------------------------------
template <class T>
inline T
Reflex::Object_Cast(const Object& o) {
//-------------------------------------------------------------------------------
   return *(T*) o.Address();
}


//-------------------------------------------------------------------------------
inline Reflex::Object::Object(const Type& type,
                              void* mem)
//-------------------------------------------------------------------------------
   : fType(type),
   fAddress(mem) {
}


//-------------------------------------------------------------------------------
inline Reflex::Object::Object(const Object& obj)
//-------------------------------------------------------------------------------
   : fType(obj.fType),
   fAddress(obj.fAddress) {
}


//-------------------------------------------------------------------------------
inline Reflex::Object
Reflex::Object::operator =(const Object& obj) {
//-------------------------------------------------------------------------------
   fType = obj.fType;
   fAddress = obj.fAddress;
   return *this;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Object::operator ==(const Object& obj) {
//-------------------------------------------------------------------------------
   return fType == obj.fType && fAddress == obj.fAddress;
}


//-------------------------------------------------------------------------------
inline bool
Reflex::Object::operator !=(const Object& obj) {
//-------------------------------------------------------------------------------
   return fType != obj.fType || fAddress != obj.fAddress;
}


//-------------------------------------------------------------------------------
inline
Reflex::Object::operator bool() const {
//-------------------------------------------------------------------------------
   if (fType && fAddress) {
      return true;
   }
   return false;
}


//-------------------------------------------------------------------------------
inline void*
Reflex::Object::Address() const {
//-------------------------------------------------------------------------------
   return fAddress;
}


//-------------------------------------------------------------------------------
inline Reflex::Object
Reflex::Object::CastObject(const Type& to) const {
//-------------------------------------------------------------------------------
   if (*this) {
      return fType.CastObject(to, *this);
   }
   return Object();
}


//-------------------------------------------------------------------------------
inline void
Reflex::Object::Destruct() const {
//-------------------------------------------------------------------------------
   if (*this) {
      fType.Destruct(fAddress);
      fAddress = 0;
   }
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Object::DynamicType() const {
//-------------------------------------------------------------------------------
   return fType.DynamicType(*this);
}


//-------------------------------------------------------------------------------
inline void
Reflex::Object::Set(const std::string& dm,
                    const void* value) const {
//-------------------------------------------------------------------------------
   Set2(dm, value);
}


//-------------------------------------------------------------------------------
template <class T>
inline void
Reflex::Object::Set(const std::string& dm,
                    const T& value) const {
//-------------------------------------------------------------------------------
   Set2(dm, &value);
}


//-------------------------------------------------------------------------------
inline Reflex::Type
Reflex::Object::TypeOf() const {
//-------------------------------------------------------------------------------
   return fType;
}


#endif // Reflex_Object
