// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_TypeName
#define Reflex_TypeName

// Include files
#include "Reflex/Kernel.h"
#include "LiteralString.h"
#include <typeinfo>

namespace Reflex {
// forward declarations
class Type;
class TypeBase;

/**
 * class TypeName TypeName.h Reflex/TypeName.h
 * @author Stefan Roiser
 * @date 06/11/2004
 * @ingroup Ref
 */
class RFLX_API TypeName {
   friend class Type;
   friend class TypeBase;

public:
   /** default constructor */
   TypeName(const char* nnam,
            TypeBase * typeBas,
            const std::type_info * ti = 0);


   /**
    * ByName will look for a At given as a string and return a pointer to
    * its reflexion At
    * @param  key fully qualified Name of the At as string
    * @return pointer to At or 0 if none is found
    */
   static Type ByName(const std::string& key);


   /**
    * byTypeId will look for a At given as a string representation of a
    * type_info and return a pointer to its reflexion At
    * @param  tid string representation of the type_info At
    * @return pointer to At or 0 if none is found
    */
   static Type ByTypeInfo(const std::type_info& ti);


   static void CleanUp();


   /**
    * DeleteType will call the destructor of the TypeBase this TypeName is
    * pointing to and remove it's information from the data structures. The
    * TypeName information will remain.
    */
   void DeleteType() const;


   /**
    * Hide this type from any lookup by appending the string " @HIDDEN@" to its name.
    */
   void HideName();

   /**
    * Un-Hide this type from any lookup by removing the string " @HIDDEN@" to its name.
    */
   void UnhideName();


   /**
    * Name returns a char* pointer to the unqualified At Name
    * @ return c string to unqualified At Name
    */
   const char* Name() const;


   const LiteralString& LiteralName() const { return fName; }

   /**
    * At returns the At object of this TypeName
    * @return corresponding Type to this TypeName
    */
   Type ThisType() const;

   /**
    * Unload reflection information for this type.
    */
   void Unload();

   /**
    * At will return a pointer to the nth Type in the system
    * @param  nth number of At to return
    * @return pointer to nth Type in the system
    */
   static Type TypeAt(size_t nth);


   /**
    * Size will return the number of currently defined types in
    * the system
    * @return number of currently defined types
    */
   static size_t TypeSize();


   static Type_Iterator Type_Begin();
   static Type_Iterator Type_End();
   static Reverse_Type_Iterator Type_RBegin();
   static Reverse_Type_Iterator Type_REnd();

private:
   /** destructor */
   ~TypeName();

   /** Set the type_info in the hash_map to this */
   void SetTypeId(const std::type_info& ti) const;

private:
   /** the Name of the At */
   LiteralString fName;


   /**
    * pointer to a TypebeBase if the At is implemented
    * @label type base
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1
    */
   mutable
   TypeBase * fTypeBase;

   /**
    * Pointer back to the type
    * @label this type
    * @link aggregation
    * @supplierCardinality 1
    * @clientCardinality 1
    */
   Type* fThisType;

};    // class TypeName

} // namespace Reflex

//-------------------------------------------------------------------------------
inline const char*
Reflex::TypeName::Name() const {
//-------------------------------------------------------------------------------
   return fName.c_str();
}


#endif // Reflex_TypeName
