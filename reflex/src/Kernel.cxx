// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Kernel.h"

#include "Reflex/Scope.h"
#include "Reflex/ScopeName.h"
#include "Reflex/PropertyList.h"
#include "Reflex/Type.h"
#include "Reflex/Base.h"
#include "Reflex/Member.h"
#include "Reflex/Object.h"
#include "Reflex/PropertyList.h"

#include "Fundamental.h"
#include "Namespace.h"
#include "Typedef.h"
#include "Class.h"
#include <typeinfo>


ROOT::Reflex::Scope ROOT::Reflex::Scope::__NIRVANA__ = 
  ROOT::Reflex::Scope( new ScopeName( "@N@I@R@V@A@N@A@", 0 ));

//-------------------------------------------------------------------------------
ROOT::Reflex::Reflex::Reflex() {
//-------------------------------------------------------------------------------

  /** initialisation of the global namespace */
  Namespace::InitGlobalNamespace();

  // initialising fundamental types
  Fundamental * tb = 0;
  Type t = Type();
 
  // char [3.9.1.1]
  tb = new Fundamental( "char", 
                        sizeof( char ), 
                        typeid( char ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );

  // signed integer types [3.9.1.2]
  tb = new Fundamental( "signed char", 
                        sizeof( signed char ), 
                        typeid( signed char ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );

  tb = new Fundamental( "short int", 
                        sizeof( short int ), 
                        typeid( short int ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );
  t = (*tb);
  new Typedef( "short", t, FUNDAMENTAL );
  new Typedef( "signed short", t, FUNDAMENTAL );
  new Typedef( "short signed", t, FUNDAMENTAL );
  new Typedef( "signed short int", t, FUNDAMENTAL );
  new Typedef( "short signed int", t, FUNDAMENTAL );

  tb = new Fundamental( "int", 
                        sizeof( int ), 
                        typeid( int ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );
  t = *tb;
  new Typedef( "signed", t, FUNDAMENTAL );
  new Typedef( "signed int", t, FUNDAMENTAL );

  tb = new Fundamental( "long int", 
                        sizeof( long int ), 
                        typeid( long int ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );
  t = *tb;
  new Typedef( "long", t, FUNDAMENTAL );
  new Typedef( "signed long", t, FUNDAMENTAL );
  new Typedef( "long signed", t, FUNDAMENTAL );
  new Typedef( "signed long int", t, FUNDAMENTAL );
  new Typedef( "long signed int", t, FUNDAMENTAL );

  // unsigned integer types [3.9.1.3]
  tb = new Fundamental( "unsigned char", 
                        sizeof( unsigned char ), 
                        typeid( unsigned char ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );

  tb = new Fundamental( "unsigned short int", 
                        sizeof( unsigned short int ), 
                        typeid( unsigned short int ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );
  t = *tb;
  new Typedef( "unsigned short", t, FUNDAMENTAL );
  new Typedef( "short unsigned int", t, FUNDAMENTAL );

  tb = new Fundamental( "unsigned int", 
                        sizeof( unsigned int ), 
                        typeid( unsigned int ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );
  t = *tb;
  new Typedef( "unsigned", t, FUNDAMENTAL );

  tb = new Fundamental( "unsigned long int", 
                        sizeof( unsigned long int ), 
                        typeid( unsigned long int ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );
  t = *tb;
  new Typedef( "unsigned long", t, FUNDAMENTAL );
  new Typedef( "long unsigned", t, FUNDAMENTAL );
  new Typedef( "long unsigned int", t, FUNDAMENTAL );

  /*/ w_chart [3.9.1.5]
  tb = new Fundamental( "w_chart", 
                        sizeof( w_chart ), 
                        & typeid( w_chart ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );
  */

  // bool [3.9.1.6]
  tb = new Fundamental( "bool", 
                        sizeof( bool ), 
                        typeid( bool ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );

  // floating point types [3.9.1.8]
  tb = new Fundamental( "float", 
                        sizeof( float ), 
                        typeid( float ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );

  tb = new Fundamental( "double", 
                        sizeof( double ), 
                        typeid( double ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );

  tb = new Fundamental( "long double", 
                        sizeof( long double ), 
                        typeid( long double ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );

  // void [3.9.1.9]
  tb = new Fundamental( "void", 
                        0, 
                        typeid( void ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );

  // non fundamental types but also supported at initialisation
  tb = new Fundamental( "longlong", 
                        sizeof( longlong ), 
                        typeid( longlong ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );
  t = *tb;
  new Typedef( "long long", t, FUNDAMENTAL );
  new Typedef( "long long int", t, FUNDAMENTAL );

  tb = new Fundamental( "ulonglong", 
                        sizeof( ulonglong ), 
                        typeid( ulonglong ));
  tb->PropertyListGet().AddProperty( "desc", "fundamental TypeNth" );
  t = *tb;
  new Typedef( "long long unsigned", t, FUNDAMENTAL );
  new Typedef( "unsigned long long", t, FUNDAMENTAL );
  new Typedef( "unsigned long long int", t, FUNDAMENTAL );
  new Typedef( "long long unsigned int", t, FUNDAMENTAL );
}


//-------------------------------------------------------------------------------
const std::string & ROOT::Reflex::Reflex::Argv0() {
//-------------------------------------------------------------------------------
  static std::string str = "SEAL REFLEX";
  return str;
}


//-------------------------------------------------------------------------------
namespace {
  static ROOT::Reflex::Reflex initialise;
}
//-------------------------------------------------------------------------------



