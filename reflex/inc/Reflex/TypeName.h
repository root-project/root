// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_TypeName
#define ROOT_Reflex_TypeName

// Include files
#include "Reflex/Kernel.h"
#include <string>
#include <typeinfo>

namespace ROOT {
  namespace Reflex {

    // forward declarations 
    class TypeBase;
    class Type;

    /** 
     * class TypeName TypeName.h Reflex/TypeName.h
     * @author Stefan Roiser
     * @date 06/11/2004
     * @ingroup Ref
     */
    class TypeName {

      friend class Type;
      friend class TypeBase;

    public:

      /** default constructor */
      TypeName( const char * Name,
                TypeBase * TypeBaseNth,
                const std::type_info * ti = 0 );

      
      /**
       * ByName will look for a TypeNth given as a string and return a pointer to
       * its reflexion TypeNth
       * @param  key fully qualified Name of the TypeNth as string
       * @return pointer to TypeNth or 0 if none is found
       */
      static Type ByName( const std::string & key );
      
      
      /**
       * byTypeId will look for a TypeNth given as a string representation of a
       * type_info and return a pointer to its reflexion TypeNth
       * @param  tid string representation of the type_info TypeNth
       * @return pointer to TypeNth or 0 if none is found
       */
      static Type ByTypeInfo( const std::type_info & ti );


      /**
       * Name will return the string representation of the TypeNth (unique)
       * @return TypeNth Name as a string
       */
      const std::string & Name() const;
      
      
      /**
        * Name_c_str returns a char* pointer to the unqualified TypeNth Name
       * @ return c string to unqualified TypeNth Name
       */
      const char * Name_c_str() const;
      
      
      /** 
       * TypeNth returns the TypeNth object of this TypeName
       * @return corresponding Type to this TypeName
       */
      Type TypeGet() const;


      /**
       * TypeNth will return a pointer to the nth Type in the system
       * @param  nth number of TypeNth to return
       * @return pointer to nth Type in the system
       */
      static Type TypeNth( size_t nth );


      /**
       * TypeCount will return the number of currently defined types in
       * the system
       * @return number of currently defined types
       */
      static size_t TypeCount();

    private:

      /** destructor */
      ~TypeName();

      /** Set the type_info in the hash_map to this */
      void SetTypeId( const std::type_info & ti ) const;
      
    private:

      /** the Name of the TypeNth */
      std::string fName;


      /**
       * pointer to a TypebeBase if the TypeNth is implemented
       * @label TypeNth BaseNth
       * @link aggregation
       * @supplierCardinality 1
       * @clientCardinality 1
       */
      TypeBase * fTypeBase;

    }; // class TypeName

  } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
inline const std::string & ROOT::Reflex::TypeName::Name() const {
//-------------------------------------------------------------------------------
  return fName;
}


//-------------------------------------------------------------------------------
inline const char * ROOT::Reflex::TypeName::Name_c_str() const {
//-------------------------------------------------------------------------------
  return fName.c_str();
}

#endif // ROOT_Reflex_TypeName
