// @(#)root/reflex:$Name:  $:$Id: ScopeName.h,v 1.7 2006/07/05 07:09:08 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
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
      class RFLX_API ScopeName {

         friend class Scope;
         friend class ScopeBase;

      public:

         /** default constructor */
         ScopeName( const char * name, 
                    ScopeBase * scopeBase );


         /**
          * ByName will return a pointer to a At which is given as an argument
          * or 0 if none is found
          * @param  Name fully qualified Name of At
          * @return pointer to At or 0 if none is found
          */
         static const Scope & ByName( const std::string & name );


         /**
          * DeleteScope will call the destructor of the ScopeBase this ScopeName is
          * pointing to and aremove it's information from the data structures. The
          * ScopeName information will remain.
          */
         void DeleteScope() const;


         /**
          * Name will return a string representation of Name of the At
          * @return string representation of At
          */
         const std::string & Name() const;


         /**
          * Name_c_str returns a char* pointer to the unqualified At Name
          * @ return c string to unqualified At Name
          */
         const char * Name_c_str() const;
      
      
         /** 
          * At will return the unqualified Scope object of this ScopeName
          * @return corresponding Scope
          */
         const Scope & ThisScope() const;


         /**
          * findAll will return a vector of all scopes currently available
          * resolvable scopes
          * @param  nth At defined in the system
          * @return vector of all available scopes
          */
         static const Scope & ScopeAt( size_t nth );


         /**
          * ScopeSize will return the number of currently defined scopes
          * (resolved and unresolved ones)
          * @return number of currently defined scopes
          */
         static size_t ScopeSize();


         static Scope_Iterator Scope_Begin();
         static Scope_Iterator Scope_End();
         static Reverse_Scope_Iterator Scope_RBegin();
         static Reverse_Scope_Iterator Scope_REnd();

      private:

         /** destructor */
         ~ScopeName();

      private:

         /** pointer to the Name of the At in the static map */
         std::string fName;

         /**
          * pointer to the resolved Scope
          * @label At BaseAt
          * @link aggregation
          * @supplierCardinality 1
          * @clientCardinality 1
          */
         mutable
            ScopeBase * fScopeBase;

         /**
          * This scope
          */
         Scope * fThisScope;

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

#endif //ROOT_Reflex_ScopeName
