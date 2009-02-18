// @(#)root/reflex:$Id$
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
#include "Reflex/MemberTemplate.h"
#include "Reflex/Any.h"

#include "Fundamental.h"
#include "Namespace.h"
#include "Typedef.h"
#include "Class.h"
#include <typeinfo>


//-------------------------------------------------------------------------------
Reflex::Instance::Instance() {
//-------------------------------------------------------------------------------
// Initialisation of Reflex.Setup of global scope, fundamental types.

   static bool initialized = false;

   if (initialized) {
      return;
   }

   initialized = true;

   /** initialisation of the global namespace */
   Namespace::GlobalScope();

   // initialising fundamental types
   TypeBase* tb = 0;
   Type t = Type();
 
   // char [3.9.1.1]
   tb = new TypeBase( "char", sizeof( char ), FUNDAMENTAL, typeid( char ), 'c');
   tb->Properties().AddProperty( "Description", "fundamental type" );

   // signed integer types [3.9.1.2]
   tb = new TypeBase( "signed char", sizeof( signed char ), FUNDAMENTAL, typeid( signed char ), 'c');
   tb->Properties().AddProperty( "Description", "fundamental type" );

   tb = new TypeBase( "short int", sizeof( short int ), FUNDAMENTAL, typeid( short int ), 's');
   tb->Properties().AddProperty( "Description", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "short", t, FUNDAMENTAL, t );
   new Typedef( "signed short", t, FUNDAMENTAL, t );
   new Typedef( "short signed", t, FUNDAMENTAL, t );
   new Typedef( "signed short int", t, FUNDAMENTAL, t );
   new Typedef( "short signed int", t, FUNDAMENTAL, t );

   tb = new TypeBase( "int", sizeof( int ), FUNDAMENTAL, typeid( int ), 'i');
   tb->Properties().AddProperty( "Description", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "signed", t, FUNDAMENTAL, t );
   new Typedef( "signed int", t, FUNDAMENTAL, t );

   tb = new TypeBase( "long int", sizeof( long int ), FUNDAMENTAL, typeid( long int ), 'l');
   tb->Properties().AddProperty( "Description", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "long", t, FUNDAMENTAL, t );
   new Typedef( "signed long", t, FUNDAMENTAL, t );
   new Typedef( "long signed", t, FUNDAMENTAL, t );
   new Typedef( "signed long int", t, FUNDAMENTAL, t );
   new Typedef( "long signed int", t, FUNDAMENTAL, t );

   // unsigned integer types [3.9.1.3]
   tb = new TypeBase( "unsigned char", sizeof( unsigned char ), FUNDAMENTAL, typeid( unsigned char ), 'b');
   tb->Properties().AddProperty( "Description", "fundamental type" );

   tb = new TypeBase( "unsigned short int", sizeof( unsigned short int ), FUNDAMENTAL, typeid( unsigned short int ), 'r');
   tb->Properties().AddProperty( "Description", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "unsigned short", t, FUNDAMENTAL, t );
   new Typedef( "short unsigned int", t, FUNDAMENTAL, t );

   tb = new TypeBase( "unsigned int", sizeof( unsigned int ), FUNDAMENTAL, typeid( unsigned int ), 'h');
   tb->Properties().AddProperty( "Description", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "unsigned", t, FUNDAMENTAL, t );

   tb = new TypeBase( "unsigned long int", sizeof( unsigned long int ), FUNDAMENTAL, typeid( unsigned long int ), 'k');
   tb->Properties().AddProperty( "Description", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "unsigned long", t, FUNDAMENTAL, t );
   new Typedef( "long unsigned", t, FUNDAMENTAL, t );
   new Typedef( "long unsigned int", t, FUNDAMENTAL, t );

   /*/ w_chart [3.9.1.5]
     tb = new TypeBase( "w_chart", 
     sizeof( w_chart ), 
     & typeid( w_chart ));
     tb->Properties().AddProperty( "Description", "fundamental type" );
   */

   // bool [3.9.1.6]
   tb = new TypeBase( "bool", sizeof( bool ), FUNDAMENTAL, typeid( bool ), 'g');
   tb->Properties().AddProperty( "Description", "fundamental type" );

   // floating point types [3.9.1.8]
   tb = new TypeBase( "float", sizeof( float ), FUNDAMENTAL, typeid( float ), 'f');
   tb->Properties().AddProperty( "Description", "fundamental type" );

   tb = new TypeBase( "double", sizeof( double ), FUNDAMENTAL, typeid( double ), 'd');
   tb->Properties().AddProperty( "Description", "fundamental type" );

   tb = new TypeBase( "long double", sizeof( long double ), FUNDAMENTAL, typeid( long double ), 'q');
   tb->Properties().AddProperty( "Description", "fundamental type" );

   // void [3.9.1.9]
   tb = new TypeBase( "void", 0, FUNDAMENTAL, typeid( void ), 'y');
   tb->Properties().AddProperty( "Description", "fundamental type" );

      // Large integer definition depends of the platform
#if defined(_WIN32) && !defined(__CINT__)
   typedef __int64 longlong;
   typedef unsigned __int64 ulonglong;
#else
   typedef long long int longlong; /* */
   typedef unsigned long long int /**/ ulonglong;
#endif

   // non fundamental types but also supported at initialisation
   tb = new TypeBase( "long long", sizeof( longlong ), FUNDAMENTAL, typeid( longlong ), 'n');
   tb->Properties().AddProperty( "Description", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "long long int", t, FUNDAMENTAL, t );

   tb = new TypeBase( "unsigned long long", sizeof( ulonglong ), FUNDAMENTAL, typeid( ulonglong ), 'm');
   tb->Properties().AddProperty( "Description", "fundamental type" );
   t = tb->ThisType();
   new Typedef( "long long unsigned", t, FUNDAMENTAL, t );
   new Typedef( "unsigned long long int", t, FUNDAMENTAL, t );
   new Typedef( "long long unsigned int", t, FUNDAMENTAL, t );

   new TypeBase("rootSpecial$", sizeof(void*) * 2, FUNDAMENTAL, typeid(::Reflex::UnknownType), 'Z'); // 'Z' type
   new TypeBase("blockBreakContinueGoto$", sizeof(int), FUNDAMENTAL, typeid(::Reflex::UnknownType), '\001'); // was also 'Z' type (confusing)

}


//-------------------------------------------------------------------------------
void Reflex::Instance::Shutdown() {
//-------------------------------------------------------------------------------
   // function to be called at tear down of Reflex, removes all memory allocations
   MemberTemplateName::CleanUp();
   TypeTemplateName::CleanUp();
   TypeName::CleanUp();
   ScopeName::CleanUp();
}


//-------------------------------------------------------------------------------

Reflex::Instance::~Instance() {
//-------------------------------------------------------------------------------
   // Destructor

   // Uncomment this once Unload work:
   // Shutdown;
}


//-------------------------------------------------------------------------------
const Reflex::StdString_Cont_Type_t & Reflex::Dummy::StdStringCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of std strings.
   return Get< StdString_Cont_Type_t >();
}


//-------------------------------------------------------------------------------
const Reflex::Type_Cont_Type_t & Reflex::Dummy::TypeCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Types.
   return Get< Type_Cont_Type_t >();
}


//-------------------------------------------------------------------------------
const Reflex::Base_Cont_Type_t & Reflex::Dummy::BaseCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Bases.
   return Get< Base_Cont_Type_t >();
}


//-------------------------------------------------------------------------------
const Reflex::Scope_Cont_Type_t & Reflex::Dummy::ScopeCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Scopes.
   return Get< Scope_Cont_Type_t >();
}


//-------------------------------------------------------------------------------
const Reflex::Object_Cont_Type_t & Reflex::Dummy::ObjectCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Objects.
   return Get< Object_Cont_Type_t >();
}


//-------------------------------------------------------------------------------
const Reflex::Member_Cont_Type_t & Reflex::Dummy::MemberCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Members.
   return Get< Member_Cont_Type_t >();
}


//-------------------------------------------------------------------------------
const Reflex::TypeTemplate_Cont_Type_t & Reflex::Dummy::TypeTemplateCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of TypeTemplates.
   return Get< TypeTemplate_Cont_Type_t >();
}


//-------------------------------------------------------------------------------
const Reflex::MemberTemplate_Cont_Type_t & Reflex::Dummy::MemberTemplateCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of MemberTemplates.
   return Get< MemberTemplate_Cont_Type_t >();
}


//-------------------------------------------------------------------------------
Reflex::Any & Reflex::Dummy::Any() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Any object 
   static Reflex::Any i;
   if ( i ) i.Clear();
   return i;
}


//-------------------------------------------------------------------------------
const Reflex::Object & Reflex::Dummy::Object() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Object 
   return Get< Reflex::Object >();
}


//-------------------------------------------------------------------------------
const Reflex::Type & Reflex::Dummy::Type() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Type 
   return Get< Reflex::Type >();
}


//-------------------------------------------------------------------------------
const Reflex::TypeTemplate & Reflex::Dummy::TypeTemplate() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty TypeTemplate 
   return Get< Reflex::TypeTemplate >();
}


//-------------------------------------------------------------------------------
const Reflex::Base & Reflex::Dummy::Base() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Base 
   return Get< Reflex::Base >();
}


//-------------------------------------------------------------------------------
const Reflex::PropertyList & Reflex::Dummy::PropertyList() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty PropertyList 
   return Get< Reflex::PropertyList >();
}


//-------------------------------------------------------------------------------
const Reflex::Member & Reflex::Dummy::Member() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Member 
   return Get< Reflex::Member >();
}


//-------------------------------------------------------------------------------
const Reflex::MemberTemplate & Reflex::Dummy::MemberTemplate() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty MemberTemplate 
   return Get< Reflex::MemberTemplate >();
}


//-------------------------------------------------------------------------------
const Reflex::Scope & Reflex::Dummy::Scope() {
//-------------------------------------------------------------------------------
   // static wrapper for an empty Scope 
   return Get< Reflex::Scope >();
}


//-------------------------------------------------------------------------------
const std::string & Reflex::Argv0() {
//-------------------------------------------------------------------------------
// Return the name of the package.
   static std::string str = "REFLEX";
   return str;
}


//-------------------------------------------------------------------------------
namespace {
   Reflex::Instance initialise;
}
//-------------------------------------------------------------------------------



