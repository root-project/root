// @(#)root/reflex:$Name:  $:$Id: Kernel.cxx,v 1.13 2006/08/01 09:14:33 roiser Exp $
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

#include "Reflex/Kernel.h"

#include "Reflex/Scope.h"
#include "Reflex/internal/ScopeName.h"
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


//-------------------------------------------------------------------------------
ROOT::Reflex::Reflex::Reflex() {
//-------------------------------------------------------------------------------
// Initialisation of Reflex.Setup of global scope, fundamental types.

   /** initialisation of the global namespace */
   Namespace::InitGlobalNamespace();

   // initialising fundamental types
   Fundamental * tb = 0;
   Type t = Type();
 
   // char [3.9.1.1]
   tb = new Fundamental( "char", 
                         sizeof( char ), 
                         typeid( char ));
   tb->Properties().AddProperty( "desc", "fundamental type" );

   // signed integer types [3.9.1.2]
   tb = new Fundamental( "signed char", 
                         sizeof( signed char ), 
                         typeid( signed char ));
   tb->Properties().AddProperty( "desc", "fundamental type" );

   tb = new Fundamental( "short int", 
                         sizeof( short int ), 
                         typeid( short int ));
   tb->Properties().AddProperty( "desc", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "short", t, FUNDAMENTAL );
   new Typedef( "signed short", t, FUNDAMENTAL );
   new Typedef( "short signed", t, FUNDAMENTAL );
   new Typedef( "signed short int", t, FUNDAMENTAL );
   new Typedef( "short signed int", t, FUNDAMENTAL );

   tb = new Fundamental( "int", 
                         sizeof( int ), 
                         typeid( int ));
   tb->Properties().AddProperty( "desc", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "signed", t, FUNDAMENTAL );
   new Typedef( "signed int", t, FUNDAMENTAL );

   tb = new Fundamental( "long int", 
                         sizeof( long int ), 
                         typeid( long int ));
   tb->Properties().AddProperty( "desc", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "long", t, FUNDAMENTAL );
   new Typedef( "signed long", t, FUNDAMENTAL );
   new Typedef( "long signed", t, FUNDAMENTAL );
   new Typedef( "signed long int", t, FUNDAMENTAL );
   new Typedef( "long signed int", t, FUNDAMENTAL );

   // unsigned integer types [3.9.1.3]
   tb = new Fundamental( "unsigned char", 
                         sizeof( unsigned char ), 
                         typeid( unsigned char ));
   tb->Properties().AddProperty( "desc", "fundamental type" );

   tb = new Fundamental( "unsigned short int", 
                         sizeof( unsigned short int ), 
                         typeid( unsigned short int ));
   tb->Properties().AddProperty( "desc", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "unsigned short", t, FUNDAMENTAL );
   new Typedef( "short unsigned int", t, FUNDAMENTAL );

   tb = new Fundamental( "unsigned int", 
                         sizeof( unsigned int ), 
                         typeid( unsigned int ));
   tb->Properties().AddProperty( "desc", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "unsigned", t, FUNDAMENTAL );

   tb = new Fundamental( "unsigned long int", 
                         sizeof( unsigned long int ), 
                         typeid( unsigned long int ));
   tb->Properties().AddProperty( "desc", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "unsigned long", t, FUNDAMENTAL );
   new Typedef( "long unsigned", t, FUNDAMENTAL );
   new Typedef( "long unsigned int", t, FUNDAMENTAL );

   /*/ w_chart [3.9.1.5]
     tb = new Fundamental( "w_chart", 
     sizeof( w_chart ), 
     & typeid( w_chart ));
     tb->Properties().AddProperty( "desc", "fundamental type" );
   */

   // bool [3.9.1.6]
   tb = new Fundamental( "bool", 
                         sizeof( bool ), 
                         typeid( bool ));
   tb->Properties().AddProperty( "desc", "fundamental type" );

   // floating point types [3.9.1.8]
   tb = new Fundamental( "float", 
                         sizeof( float ), 
                         typeid( float ));
   tb->Properties().AddProperty( "desc", "fundamental type" );

   tb = new Fundamental( "double", 
                         sizeof( double ), 
                         typeid( double ));
   tb->Properties().AddProperty( "desc", "fundamental type" );

   tb = new Fundamental( "long double", 
                         sizeof( long double ), 
                         typeid( long double ));
   tb->Properties().AddProperty( "desc", "fundamental type" );

   // void [3.9.1.9]
   tb = new Fundamental( "void", 
                         0, 
                         typeid( void ));
   tb->Properties().AddProperty( "desc", "fundamental type" );

   // non fundamental types but also supported at initialisation
   tb = new Fundamental( "longlong", 
                         sizeof( longlong ), 
                         typeid( longlong ));
   tb->Properties().AddProperty( "desc", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "long long", t, FUNDAMENTAL );
   new Typedef( "long long int", t, FUNDAMENTAL );

   tb = new Fundamental( "ulonglong", 
                         sizeof( ulonglong ), 
                         typeid( ulonglong ));
   tb->Properties().AddProperty( "desc", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "long long unsigned", t, FUNDAMENTAL );
   new Typedef( "unsigned long long", t, FUNDAMENTAL );
   new Typedef( "unsigned long long int", t, FUNDAMENTAL );
   new Typedef( "long long unsigned int", t, FUNDAMENTAL );
}


//-------------------------------------------------------------------------------
ROOT::Reflex::StdString_Cont_Type_t & ROOT::Reflex::Dummy::StdStringCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of std strings.
   static StdString_Cont_Type_t c;
   return c;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Type_Cont_Type_t & ROOT::Reflex::Dummy::TypeCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Types.
   static Type_Cont_Type_t c;
   return c;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Base_Cont_Type_t & ROOT::Reflex::Dummy::BaseCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Bases.
   static Base_Cont_Type_t c;
   return c;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Scope_Cont_Type_t & ROOT::Reflex::Dummy::ScopeCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Scopes.
   static Scope_Cont_Type_t c;
   return c;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Object_Cont_Type_t & ROOT::Reflex::Dummy::ObjectCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Objects.
   static Object_Cont_Type_t c;
   return c;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::Member_Cont_Type_t & ROOT::Reflex::Dummy::MemberCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Members.
   static Member_Cont_Type_t c;
   return c;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::TypeTemplate_Cont_Type_t & ROOT::Reflex::Dummy::TypeTemplateCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of TypeTemplates.
   static TypeTemplate_Cont_Type_t c;
   return c;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::MemberTemplate_Cont_Type_t & ROOT::Reflex::Dummy::MemberTemplateCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of MemberTemplates.
   static MemberTemplate_Cont_Type_t c;
   return c;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Object & ROOT::Reflex::Dummy::Object() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Object 
   static ROOT::Reflex::Object i;
   return i;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Type & ROOT::Reflex::Dummy::Type() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Type 
   static ROOT::Reflex::Type i;
   return i;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::TypeTemplate & ROOT::Reflex::Dummy::TypeTemplate() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty TypeTemplate 
   static ROOT::Reflex::TypeTemplate i;
   return i;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Base & ROOT::Reflex::Dummy::Base() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Base 
   static ROOT::Reflex::Base i;
   return i;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::PropertyList & ROOT::Reflex::Dummy::PropertyList() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty PropertyList 
   static ROOT::Reflex::PropertyList i;
   return i;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Member & ROOT::Reflex::Dummy::Member() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Member 
   static ROOT::Reflex::Member i;
   return i;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::MemberTemplate & ROOT::Reflex::Dummy::MemberTemplate() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty MemberTemplate 
   static ROOT::Reflex::MemberTemplate i;
   return i;
}


//-------------------------------------------------------------------------------
const ROOT::Reflex::Scope & ROOT::Reflex::Dummy::Scope() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Scope 
   static ROOT::Reflex::Scope i;
   return i;
}


//-------------------------------------------------------------------------------
const std::string & ROOT::Reflex::Reflex::Argv0() {
//-------------------------------------------------------------------------------
// Return the name of the package.
   static std::string str = "REFLEX";
   return str;
}


//-------------------------------------------------------------------------------
namespace {
   ROOT::Reflex::Reflex initialise;
}
//-------------------------------------------------------------------------------



