// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// See http://www.boost.org/libs/any for Documentation.

// Copyright Kevlin Henney, 2000, 2001, 2002. All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Any
#define Reflex_Any

// What:  variant At boost::any
// who:   contributed by Kevlin Henney,
//        with features contributed and bugs found by
//        Ed Brey, Mark Rodgers, Peter Dimov, and James Curran
// when:  July 2001
// where: tested with BCC 5.5, MSVC 6.0, and g++ 2.95

#include "Reflex/Kernel.h"
#include <algorithm>
#include <typeinfo>
#include <iostream>

namespace Reflex {
/**
 * @class Any Any.h Reflex/Any.h
 * @author K. Henney
 */
class RFLX_API Any {
   friend RFLX_API std::ostream& operator <<(std::ostream&,
                                             const Any&);

public:
   /** Constructor */
   Any():
      fContent(0) {}

   /** Constructor */
   template <typename ValueType> Any(const ValueType &value):
      fContent(new Holder<ValueType>(value)) {}

   /** Copy Constructor */
   Any(const Any &other):
      fContent(other.fContent ? other.fContent->Clone() : 0) {}

   /** Dtor */
   ~Any() {
      delete fContent;
   }

   /** Clear the content */
   void
   Clear() {
      if (!Empty()) {
         delete fContent;
         fContent = 0;
      }
   }


   /** bool operator */
   operator bool() {
      return !Empty();
   }

   /** Modifier */
   Any&
   Swap(Any& rhs) {
      std::swap(fContent, rhs.fContent);
      return *this;
   }


   /** Modifier */
   template <typename ValueType> Any&
   operator =(const ValueType& rhs) {
      Any(rhs).Swap(*this);
      return *this;
   }


   /** Modifier */
   Any&
   operator =(const Any& rhs) {
      Any(rhs).Swap(*this);
      return *this;
   }


   /** Query */
   bool
   Empty() const {
      return !fContent;
   }


   /** Query */
   const std::type_info&
   TypeInfo() const {
      return fContent ? fContent->TypeInfo() : typeid(void);
   }


   /** Adress */
   void*
   Address() const {
      return fContent ? fContent->Address() : 0;
   }


private:
   // or public: ?
   /**
    * @class Placeholder BoostAny.h Reflex/BoostAny.h
    * @author K. Henney
    */
   class Placeholder {
   public:
      /** Constructor */
      Placeholder() {}

      /** Destructor */
      virtual ~Placeholder() {}

      /** Query */
      virtual const std::type_info& TypeInfo() const = 0;

      /** Query */
      virtual Placeholder* Clone() const = 0;

      /** Query */
      virtual void* Address() const = 0;

   };

   /**
    * @class Holder BoostAny.h Reflex/BoostAny.h
    * @author K. Henney
    */
   template <typename ValueType> class Holder: public Placeholder {
   public:
      /** Constructor */
      Holder(const ValueType& value):
         fHeld(value) {}

      /** Query */
      virtual const std::type_info&
      TypeInfo() const {
         return typeid(ValueType);
      }


      /** Clone */
      virtual Placeholder*
      Clone() const {
         return new Holder(fHeld);
      }


      /** Address */
      virtual void*
      Address() const {
         return (void*) (&fHeld);
      }


      /** representation */
      ValueType fHeld;

   };


   /** representation */
   template <typename ValueType> friend ValueType* any_cast(Any*);

   // or  public:

   /** representation */
   Placeholder* fContent;

};


/**
 * @class BadAnyCast Any.h Reflex/Any.h
 * @author K. Henney
 */
class BadAnyCast: public std::bad_cast {
public:
   /** Constructor */
   BadAnyCast() {}

   /** Query */
   virtual const char*
   what() const throw() {
      return "BadAnyCast: failed conversion using any_cast";
   }


};

/** throw */
template <class E> void
throw_exception(const E& e) {
   throw e;
}


/** value */
template <typename ValueType> ValueType*
any_cast(Any* operand) {
   return operand && operand->TypeInfo() == typeid(ValueType)
          ? &static_cast<Any::Holder<ValueType>*>(operand->fContent)->fHeld : 0;
}


/** value */
template <typename ValueType> const ValueType*
any_cast(const Any* operand) {
   return any_cast<ValueType>(const_cast<Any*>(operand));
}


/** value */
template <typename ValueType> ValueType
any_cast(const Any& operand) {
   const ValueType* result = any_cast<ValueType>(&operand);

   if (!result) {
      throw_exception(BadAnyCast());
   }
   return *result;
}


/** stream operator */
RFLX_API std::ostream& operator <<(std::ostream&,
                                   const Any&);

} // namespace Reflex

#endif // Reflex_Any
