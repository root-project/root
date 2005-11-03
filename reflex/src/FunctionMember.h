// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_FunctionMember
#define ROOT_Reflex_FunctionMember

// Include files
#include "Reflex/MemberBase.h"

namespace ROOT {
  namespace Reflex {

    // forward declarations
    class Type;
    class Object;

    /**
     * @class FunctionMember FunctionMember.h Reflex/FunctionMember.h
     * @author Stefan Roiser
     * @date 24/11/2003
     * @ingroup Ref
     */
    class FunctionMember : public MemberBase {

    public:

      /** default constructor */
      FunctionMember( const char *   nam,
                      const Type &   typ,
                      StubFunction   stubFP,
                      void *         stubCtx = 0,
                      const char *   params = 0,
                      unsigned int   modifiers = 0,
                      TYPE           memType = FUNCTIONMEMBER );


      /** destructor */
      virtual ~FunctionMember() {}


      /** return full Name of function MemberNth */
      std::string Name( unsigned int mod = 0 ) const;


      /** Invoke the function (if return TypeNth as void*) */
      /*Object Invoke( const Object & obj, 
        const std::vector < Object > & paramList ) const;*/
      Object Invoke( const Object & obj, 
                     const std::vector < void * > & paramList = 
                     std::vector<void*>()) const;


      /** Invoke the function (for static functions) */
      //Object Invoke( const std::vector < Object > & paramList ) const;
      Object Invoke( const std::vector < void * > & paramList = 
                     std::vector<void*>()) const;


      /** number of parameters */
      size_t ParameterCount( bool required = false ) const;


      /** ParameterNth nth default value if declared*/
      std::string ParameterDefault( size_t nth ) const;


      /** ParameterNth nth Name if declared*/
      std::string ParameterName( size_t nth ) const;


      /** return a pointer to the context */
      void * Stubcontext() const;


      /** return the pointer to the stub function */
      StubFunction Stubfunction() const;

    private:

      /** pointer to the stub function */
      StubFunction fStubFP;


      /** user data for the stub function */
      void*  fStubCtx;


      /** ParameterNth names */
      std::vector < std::string > fParameterNames;


      /** ParameterNth names */
      std::vector < std::string > fParameterDefaults;


      /** number of required parameters */
      size_t fReqParameters;
     
    }; // class FunctionMember
  } //namespace Reflex
} //namespace ROOT



//-------------------------------------------------------------------------------
inline std::string 
ROOT::Reflex::FunctionMember::ParameterDefault( size_t nth ) const {
//-------------------------------------------------------------------------------
  return fParameterDefaults[nth];
}


//-------------------------------------------------------------------------------
inline std::string 
ROOT::Reflex::FunctionMember::ParameterName( size_t nth ) const {
//-------------------------------------------------------------------------------
  return fParameterNames[nth];
}


//-------------------------------------------------------------------------------
inline void * ROOT::Reflex::FunctionMember::Stubcontext() const {
//-------------------------------------------------------------------------------
  return fStubCtx;
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::StubFunction 
ROOT::Reflex::FunctionMember::Stubfunction() const {
//-------------------------------------------------------------------------------
  return fStubFP;
}

#endif // ROOT_Reflex_FunctionMember
