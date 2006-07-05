// @(#)root/reflex:$Name: HEAD $:$Id: FunctionMember.cxx,v 1.9 2006/07/04 15:02:55 roiser Exp $
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

#include "FunctionMember.h"

#include "Reflex/Scope.h"
#include "Reflex/Object.h"
#include "Reflex/Member.h"

#include "Function.h"
#include "Reflex/Tools.h"

//-------------------------------------------------------------------------------
ROOT::Reflex::FunctionMember::FunctionMember( const char *  nam,
                                              const Type &  typ,
                                              StubFunction  stubFP,
                                              void*         stubCtx,
                                              const char *  parameters,
                                              unsigned int  modifiers,
                                              TYPE          memType )
//-------------------------------------------------------------------------------
   : MemberBase( nam, typ, memType, modifiers ),
     fStubFP( stubFP ), 
     fStubCtx( stubCtx ),
     fParameterNames( std::vector<std::string>()),
     fParameterDefaults( std::vector<std::string>()),
     fReqParameters( 0 )
{
   // Obtain the names and default values of the function parameters
   // The "real" number of parameters is obtained from the function type
   size_t numDefaultParams = 0;
   size_t type_npar = typ.FunctionParameterSize();
   std::vector<std::string> params;
   if ( parameters ) Tools::StringSplit(params, parameters, ";");
   size_t npar = std::min(type_npar,params.size());
   for ( size_t i = 0; i < npar ; ++i ) {
      size_t pos = params[i].find( "=" );
      fParameterNames.push_back(params[i].substr(0,pos));
      if ( pos != std::string::npos ) {
         fParameterDefaults.push_back(params[i].substr(pos+1));
         ++numDefaultParams;
      }
      else {
         fParameterDefaults.push_back("");
      }
   }
   // padding with blanks
   for ( size_t i = npar; i < type_npar; ++i ) {
      fParameterNames.push_back("");
      fParameterDefaults.push_back("");
   }
   fReqParameters = type_npar - numDefaultParams;
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::FunctionMember::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
// Construct the qualified (if requested) name of the function member.
   std::string s = "";

   if ( 0 != ( mod & ( QUALIFIED | Q ))) {
      if ( IsPublic())          { s += "public ";    }
      if ( IsProtected())       { s += "protected "; }
      if ( IsPrivate())         { s += "private ";   }  
      if ( IsExtern())          { s += "extern ";    }
      if ( IsStatic())          { s += "static ";    }
      if ( IsInline())          { s += "inline ";    }
      if ( IsVirtual())         { s += "virtual ";   }
      if ( IsExplicit())        { s += "explicit ";  }
   }

   s += MemberBase::Name( mod ); 

   return s;
}


/*/-------------------------------------------------------------------------------
  ROOT::Reflex::Object
  ROOT::Reflex::FunctionMember::Invoke( const Object & obj,
  const std::vector < Object > & paramList ) const {
//-----------------------------------------------------------------------------
  if ( paramList.size() < FunctionParameterSize(true)) {
  throw RuntimeError("Not enough parameters given to function ");
  return Object();
  }
  void * mem = CalculateBaseObject( obj );
  std::vector < void * > paramValues;
  // needs more checking FIXME
  for (std::vector<Object>::const_iterator it = paramList.begin();
  it != paramList.end(); ++it ) paramValues.push_back(it->Address());
  return Object(TypeOf().ReturnType(), fStubFP( mem, paramValues, fStubCtx ));
  }
*/


//-------------------------------------------------------------------------------
ROOT::Reflex::Object
ROOT::Reflex::FunctionMember::Invoke( const Object & obj,
                                      const std::vector < void * > & paramList ) const {
//-----------------------------------------------------------------------------
// Invoke this function member with object obj. 
   if ( paramList.size() < FunctionParameterSize(true)) {
      throw RuntimeError("Not enough parameters given to function ");
      return Object();
   }
   void * mem = CalculateBaseObject( obj );
   // parameters need more checking FIXME
   return Object(TypeOf().ReturnType(), fStubFP( mem, paramList, fStubCtx ));
}


/*/-------------------------------------------------------------------------------
  ROOT::Reflex::Object
  ROOT::Reflex::FunctionMember::Invoke( const std::vector < Object > & paramList ) const {
//-------------------------------------------------------------------------------
  std::vector < void * > paramValues;
  // needs more checking FIXME
  for (std::vector<Object>::const_iterator it = paramList.begin();
  it != paramList.end(); ++it ) paramValues.push_back(it->Address());
  return Object(TypeOf().ReturnType(), fStubFP( 0, paramValues, fStubCtx ));
  }
*/


//-------------------------------------------------------------------------------
ROOT::Reflex::Object
ROOT::Reflex::FunctionMember::Invoke( const std::vector < void * > & paramList ) const {
//-------------------------------------------------------------------------------
// Call static function 
   // parameters need more checking FIXME
   return Object(TypeOf().ReturnType(), fStubFP( 0, paramList, fStubCtx ));
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::FunctionMember::FunctionParameterSize( bool required ) const {
//-------------------------------------------------------------------------------
// Return number of function parameters. If required = true return number without default params.
   if ( required ) return fReqParameters;
   else            return TypeOf().FunctionParameterSize();
}
