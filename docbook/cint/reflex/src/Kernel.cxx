// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
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
#include "Reflex/Builder/TypeBuilder.h"

#include "Fundamental.h"
#include "Namespace.h"
#include "Typedef.h"
#include "Class.h"
#include <typeinfo>

namespace {
// Helper to factor out common code
class TFundamentalDeclarator {
public:
   TFundamentalDeclarator(const char* name, size_t size, const std::type_info& ti,
                         Reflex::REPRESTYPE repres) {
      Reflex::TypeBase* tb = new Reflex::TypeBase(name, size, Reflex::FUNDAMENTAL,
                                                  ti, Reflex::Type(), repres);
      tb->Properties().AddProperty("Description", "fundamental type");
      fType = tb->ThisType();
   }


   TFundamentalDeclarator&
   Typedef(const char* name) {
      new Reflex::Typedef(Reflex::Literal(name), fType, Reflex::FUNDAMENTAL, fType);
      return *this;
   }


private:
   Reflex::Type fType;
};

// sizeof(void) doesn't work; we want it to return 0.
// This template with the specialization does just that.
template <typename T>
struct GetSizeOf {
   size_t
   operator ()() const { return sizeof(T); }

};
template <>
struct GetSizeOf<void> {
   size_t
   operator ()() const { return 0; }

};

// Helper function constructing the declarator
template <typename T>
TFundamentalDeclarator
DeclFundamental(const char* name,
                Reflex::REPRESTYPE repres) {
   return TFundamentalDeclarator(Reflex::Literal(name),
                                 GetSizeOf<T>() (), typeid(T), repres);
}


Reflex::Instance instantiate;

}

//-------------------------------------------------------------------------------
Reflex::Instance* Reflex::Instance::fgSingleton = 0;
Reflex::Instance::EState Reflex::Instance::fgState = Reflex::Instance::kUninitialized;
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
Reflex::Instance&
Reflex::Instance::CreateReflexInstance() {
//-------------------------------------------------------------------------------
// Initialize the singleton.

   static Reflex::Instance instance((Reflex::Instance*) 0);
   return instance;
}


//-------------------------------------------------------------------------------
Reflex::Instance::Instance() {
//-------------------------------------------------------------------------------
// Ensure that Reflex is properly initialized.
   CreateReflexInstance();
}


//-------------------------------------------------------------------------------
bool
Reflex::Instance::HasShutdown() {
//-------------------------------------------------------------------------------
// Return true, if we shutdown Reflex (i.e. delete all the containers)
   return fgState == kHasShutDown;
}


//-------------------------------------------------------------------------------
Reflex::Instance::Instance(Instance*) {
//-------------------------------------------------------------------------------
// Initialisation of Reflex.Setup of global scope, fundamental types.

   fgSingleton = this;
   fgState = kInitializing;

   /** initialisation of the global namespace */
   Namespace::GlobalScope();

   // initialising fundamental types
   // char [3.9.1.1]
   DeclFundamental<char>("char", REPRES_CHAR);

   // signed integer types [3.9.1.2]
   DeclFundamental<signed char>("signed char", REPRES_SIGNED_CHAR);

   DeclFundamental<short int>("short int", REPRES_SHORT_INT)
   .Typedef("short")
   .Typedef("signed short")
   .Typedef("short signed")
   .Typedef("signed short int")
   .Typedef("short signed int");

   DeclFundamental<int>("int", REPRES_INT)
   .Typedef("signed")
   .Typedef("signed int");

   DeclFundamental<long int>("long int", REPRES_LONG_INT)
   .Typedef("long")
   .Typedef("signed long")
   .Typedef("long signed")
   .Typedef("signed long int")
   .Typedef("long signed int");

   // unsigned integer types [3.9.1.3]
   DeclFundamental<unsigned char>("unsigned char", REPRES_UNSIGNED_CHAR);

   DeclFundamental<unsigned short int>("unsigned short int", REPRES_UNSIGNED_SHORT_INT)
   .Typedef("unsigned short")
   .Typedef("short unsigned int");

   DeclFundamental<unsigned int>("unsigned int", REPRES_UNSIGNED_INT)
   .Typedef("unsigned");

   DeclFundamental<unsigned long int>("unsigned long int", REPRES_UNSIGNED_LONG_INT)
   .Typedef("unsigned long")
   .Typedef("long unsigned")
   .Typedef("long unsigned int");

   /* w_chart [3.9.1.5]
      DeclFundamental<w_chart>("w_chart", REPRES_WCHART);
    */

   // bool [3.9.1.6]
   DeclFundamental<bool>("bool", REPRES_BOOL);

   // floating point types [3.9.1.8]
   DeclFundamental<float>("float", REPRES_FLOAT);
   DeclFundamental<double>("double", REPRES_DOUBLE);
   DeclFundamental<long double>("long double", REPRES_LONG_DOUBLE);

   // void [3.9.1.9]
   DeclFundamental<void>("void", REPRES_VOID);

   // Large integer definition depends of the platform
#if defined(_WIN32) && !defined(__CINT__)
   typedef __int64 longlong;
   typedef unsigned __int64 ulonglong;
#else
   typedef long long int longlong; /* */
   typedef unsigned long long int /**/ ulonglong;
#endif

   // non fundamental types but also supported at initialisation
   DeclFundamental<longlong>("long long", REPRES_LONGLONG)
   .Typedef("long long int");

   DeclFundamental<ulonglong>("unsigned long long", REPRES_ULONGLONG)
   .Typedef("long long unsigned")
   .Typedef("unsigned long long int")
   .Typedef("long long unsigned int");

   fgState = kActive;
}


//-------------------------------------------------------------------------------
void
Reflex::Instance::Shutdown() {
//-------------------------------------------------------------------------------
// Function to be called at tear down of Reflex, removes all memory allocations.

   fgState = kTearingDown;

   MemberTemplateName::CleanUp();
   TypeTemplateName::CleanUp();
   TypeName::CleanUp();
   ScopeName::CleanUp();

   fgState = kHasShutDown;
}


//-------------------------------------------------------------------------------

Reflex::Instance::~Instance() {
//-------------------------------------------------------------------------------
// Destructor.  This will shutdown Reflex only if this instance is the 'main'
// instance.

   if (fgSingleton == this) {
      Shutdown();
   }
}


//-------------------------------------------------------------------------------
Reflex::Instance::EState
Reflex::Instance::State() {
//-------------------------------------------------------------------------------
// return Reflex instance state.
   return fgState;
}


//-------------------------------------------------------------------------------
const Reflex::StdString_Cont_Type_t&
Reflex::Dummy::StdStringCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of std strings.
   return Get<StdString_Cont_Type_t>();
}


//-------------------------------------------------------------------------------
const Reflex::Type_Cont_Type_t&
Reflex::Dummy::TypeCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Types.
   return Get<Type_Cont_Type_t>();
}


//-------------------------------------------------------------------------------
const Reflex::Base_Cont_Type_t&
Reflex::Dummy::BaseCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Bases.
   return Get<Base_Cont_Type_t>();
}


//-------------------------------------------------------------------------------
const Reflex::Scope_Cont_Type_t&
Reflex::Dummy::ScopeCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Scopes.
   return Get<Scope_Cont_Type_t>();
}


//-------------------------------------------------------------------------------
const Reflex::Object_Cont_Type_t&
Reflex::Dummy::ObjectCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Objects.
   return Get<Object_Cont_Type_t>();
}


//-------------------------------------------------------------------------------
const Reflex::Member_Cont_Type_t&
Reflex::Dummy::MemberCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of Members.
   return Get<Member_Cont_Type_t>();
}


//-------------------------------------------------------------------------------
const Reflex::TypeTemplate_Cont_Type_t&
Reflex::Dummy::TypeTemplateCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of TypeTemplates.
   return Get<TypeTemplate_Cont_Type_t>();
}


//-------------------------------------------------------------------------------
const Reflex::MemberTemplate_Cont_Type_t&
Reflex::Dummy::MemberTemplateCont() {
//-------------------------------------------------------------------------------
// static wrapper for an empty container of MemberTemplates.
   return Get<MemberTemplate_Cont_Type_t>();
}


//-------------------------------------------------------------------------------
Reflex::Any&
Reflex::Dummy::Any() {
//-------------------------------------------------------------------------------
// static wrapper for an empty Any object
   static Reflex::Any i;

   if (i) {
      i.Clear();
   }
   return i;
}


//-------------------------------------------------------------------------------
const Reflex::Object&
Reflex::Dummy::Object() {
//-------------------------------------------------------------------------------
// static wrapper for an empty Object
   return Get<Reflex::Object>();
}


//-------------------------------------------------------------------------------
const Reflex::Type&
Reflex::Dummy::Type() {
//-------------------------------------------------------------------------------
// static wrapper for an empty Type
   return Get<Reflex::Type>();
}


//-------------------------------------------------------------------------------
const Reflex::TypeTemplate&
Reflex::Dummy::TypeTemplate() {
//-------------------------------------------------------------------------------
// static wrapper for an empty TypeTemplate
   return Get<Reflex::TypeTemplate>();
}


//-------------------------------------------------------------------------------
const Reflex::Base&
Reflex::Dummy::Base() {
//-------------------------------------------------------------------------------
// static wrapper for an empty Base
   return Get<Reflex::Base>();
}


//-------------------------------------------------------------------------------
const Reflex::PropertyList&
Reflex::Dummy::PropertyList() {
//-------------------------------------------------------------------------------
// static wrapper for an empty PropertyList
   return Get<Reflex::PropertyList>();
}


//-------------------------------------------------------------------------------
const Reflex::Member&
Reflex::Dummy::Member() {
//-------------------------------------------------------------------------------
// static wrapper for an empty Member
   return Get<Reflex::Member>();
}


//-------------------------------------------------------------------------------
const Reflex::MemberTemplate&
Reflex::Dummy::MemberTemplate() {
//-------------------------------------------------------------------------------
// static wrapper for an empty MemberTemplate
   return Get<Reflex::MemberTemplate>();
}


//-------------------------------------------------------------------------------
const Reflex::Scope&
Reflex::Dummy::Scope() {
//-------------------------------------------------------------------------------
// static wrapper for an empty Scope
   return Get<Reflex::Scope>();
}


//-------------------------------------------------------------------------------
const std::string&
Reflex::Argv0() {
//-------------------------------------------------------------------------------
// Return the name of the package.
   static std::string str = "REFLEX";
   return str;
}
