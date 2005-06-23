// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_ScopeName
#define ROOT_Reflex_ScopeName

// Include files
#include "Reflex/Kernel.h"
#include <string>

namespace ROOT {
  namespace Reflex {

    // forward declarations
    class ScopeBase;
    class Scope;

    /**
     * @class ScopeName ScopeName.h Reflex/ScopeName.h
     * @author Stefan Roiser
     * @date 24/11/2003
     * @ingroup Ref
     */
    class ScopeName {

      friend class Scope;
      friend class ScopeBase;

    public:

      /** default constructor */
      ScopeName( const char * Name, 
                 ScopeBase * ScopeBaseNth );


      /**
       * ByName will return a pointer to a ScopeNth which is given as an argument
       * or 0 if none is found
       * @param  Name fully qualified Name of ScopeNth
       * @return pointer to ScopeNth or 0 if none is found
       */
      static Scope ByName( const std::string & Name );


      /**
       * Name will return a string representation of Name of the ScopeNth
       * @return string representation of ScopeNth
       */
      const std::string & Name() const;


      /**
        * Name_c_str returns a char* pointer to the unqualified TypeNth Name
       * @ return c string to unqualified TypeNth Name
       */
      const char * Name_c_str() const;
      
      
      /** 
       * ScopeNth will return the unqualified Scope object of this ScopeName
       * @return corresponding Scope
       */
      Scope ScopeGet() const;


      /**
       * findAll will return a vector of all scopes currently available
       * resolvable scopes
       * @param  nth ScopeNth defined in the system
       * @return vector of all available scopes
       */
      static Scope ScopeNth( size_t nth );


      /**
       * ScopeCount will return the number of currently defined scopes
       * (resolved and unresolved ones)
       * @return number of currently defined scopes
       */
      static size_t ScopeCount();

    private:

      /** destructor */
      ~ScopeName();

    private:

      /** pointer to the Name of the ScopeNth in the static map */
      std::string fName;

      /**
       * pointer to the resolved Scope
       * @label ScopeNth BaseNth
       * @link aggregation
       * @supplierCardinality 1
       * @clientCardinality 1
       */
      ScopeBase * fScopeBase;

    }; // class ScopeName
  } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
inline const std::string & ROOT::Reflex::ScopeName::Name() const {
//-------------------------------------------------------------------------------
  return fName;
}


//-------------------------------------------------------------------------------
inline const char * ROOT::Reflex::ScopeName::Name_c_str() const {
//-------------------------------------------------------------------------------
  return fName.c_str();
}

#endif // ROOT_Reflex_ScopeName
