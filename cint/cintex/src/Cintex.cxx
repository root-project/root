// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Callback.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"
#include "Reflex/Builder/ReflexBuilder.h"

#include "Cintex/Cintex.h"
#include "CINTClassBuilder.h"
#include "CINTFunctionBuilder.h"
#include "CINTVariableBuilder.h"
#include "CINTTypedefBuilder.h"
#include "CINTEnumBuilder.h"
#include "ROOTClassEnhancer.h"
#include "CINTSourceFile.h"
#include <iostream>

#include "TROOT.h"

using namespace ROOT::Reflex;
using namespace ROOT::Cintex;
using namespace std;

//Build the Cintex dictionary on Reflex and convert it to CINT
namespace {
   struct Cintex_dict_t { 
   public:
      Cintex_dict_t() {
         //--Reflex class builder
         //NamespaceBuilder( "ROOT::Cintex" );
         Type t_void = TypeBuilder("void");
         Type t_int  = TypeBuilder("int");
         Type t_bool = TypeBuilder("bool");
         ClassBuilderT< Cintex >("Cintex", PUBLIC)
            .AddFunctionMember(FunctionTypeBuilder(t_void), "Enable", Enable, 0, 0, PUBLIC | STATIC)
            .AddFunctionMember(FunctionTypeBuilder(t_void, t_int), "SetDebug", SetDebug, 0, 0, PUBLIC | STATIC)
            .AddFunctionMember(FunctionTypeBuilder(t_int), "Debug", Debug, 0, 0, PUBLIC | STATIC)
            .AddFunctionMember(FunctionTypeBuilder(t_bool), "PropagateClassTypedefs", PropagateClassTypedefs, 0, 0, PUBLIC | STATIC)
            .AddFunctionMember(FunctionTypeBuilder(t_void, t_bool), "SetPropagateClassTypedefs", SetPropagateClassTypedefs, 0, 0, PUBLIC | STATIC);
          
         //--CINT class builder
         Type t = Type::ByName("Cintex");

         // Before loading in CINT, let's make sure that CINT (and ROOT) are initialized
         ROOT::GetROOT();
         ROOT::Cintex::CINTClassBuilder::Get(t).Setup();
      }
      static void Enable(void*, void*, const std::vector<void*>&, void*) {
         Cintex::Enable();
      }
      static void SetDebug(void*, void*, const std::vector<void*>& arg, void*) {
         Cintex::SetDebug(*(bool*)arg[0]);
      }
      static void Debug(void*, void* ret, const std::vector<void*>&, void*) {
         if (ret) *(int*)ret = Cintex::Debug();
         else Cintex::Debug();
      }

      static void PropagateClassTypedefs(void*, void* ret, const std::vector<void*>&, void*) {
         if (ret) *(bool*)ret = Cintex::PropagateClassTypedefs();
         else Cintex::PropagateClassTypedefs();
      }

      static void SetPropagateClassTypedefs(void*, void*, const std::vector<void*>& arg, void*) {
         Cintex::SetPropagateClassTypedefs(*(bool*)arg[0]);
      }

   };
   static Cintex_dict_t s_dict;
   
   static const char* btypes[] = { "bool", "char", "unsigned char", "short", "unsigned short", "int", "unsigned int",
     "long", "unsigned long", "float", "double", "string" };

   void Declare_additional_CINT_typedefs() {
      // as the function name says
      std::string name;
      std::string value;
      int autoload = G__set_class_autoloading(0); // To avoid recursive loads
      for ( size_t i = 0; i < sizeof(btypes)/sizeof(char*); i ++ ) {
         //--- vector ---
         name = std::string("vector<") + btypes[i];
         value = name;
         name += ">";
         value += std::string(",allocator<") + btypes[i] + "> >";
         CINTTypedefBuilder::Set(name.c_str(), value.c_str());
      }
      // Now that genreflex always translates basic_string<char> to string
      // we need a "typedef" (the wrong way!) for backward compatibility:
      CINTTypedefBuilder::Set("basic_string<char>", "string");
      G__set_class_autoloading(autoload);
   }
}


namespace ROOT {
   namespace Cintex {
      
      Cintex& Cintex::Instance() {
         static Cintex s_instance;
         return s_instance;
      }

      Cintex::Cintex() {
         fCallback = new Callback();
         fRootcreator = 0;
         fDbglevel = 0;
         fPropagateClassTypedefs = true;
         fPropagateClassEnums = true;
         fEnabled = false;
      }

      Cintex::~Cintex() {
         if( fCallback ) UninstallClassCallback( fCallback );
         delete fCallback;
      }

      void Cintex::Enable() {
         if ( Instance().fEnabled ) return;
         //---Declare some extra typdefs to please CINT
         Declare_additional_CINT_typedefs();
         //---Install the callback to fothcoming classes ----//
         InstallClassCallback( Instance().fCallback );        
         //---Convert to CINT all existing classes ---//
         for( size_t i = 0; i < Type::TypeSize(); i++ ) {

            ( * Instance().fCallback)( Type::TypeAt(i) );
         }
         //---Convert to CINT all existing free functions and variables
         for ( size_t n = 0; n < Scope::ScopeSize(); n++ ) {
            Scope ns = Scope::ScopeAt(n);
            if ( ns.IsNamespace() ) {
               for( size_t m = 0; m < ns.MemberSize(); m++ ) {
                  ( * Instance().fCallback)( ns.MemberAt(m) );
               }
            }
         }
         Instance().fEnabled = true;
      } 

      void Cintex::SetROOTCreator(ROOTCreator_t c) {
         Instance().fRootcreator = c;
      }

      ROOTCreator_t Cintex::GetROOTCreator() {
         return Instance().fRootcreator;
      }

      int Cintex::Debug() {
         return Instance().fDbglevel;
      }

      void Cintex::SetDebug(int l) {
         Instance().fDbglevel = l;
      }


      bool Cintex::PropagateClassTypedefs() {
         return Instance().fPropagateClassTypedefs;
      }

      void Cintex::SetPropagateClassTypedefs(bool val) {
         Instance().fPropagateClassTypedefs = val;
      }

      bool Cintex::PropagateClassEnums() {
         return Instance().fPropagateClassEnums;
      }

      void Cintex::SetPropagateClassEnums(bool val) {
         Instance().fPropagateClassEnums = val;
      }
  
      void Cintex::Default_CreateClass(const char* name, TGenericClassInfo* gci) {
         // Create a TClass object from the Reflex data; forward to ROOTClassEnhancer.
         ROOTClassEnhancer::Default_CreateClass(Reflex::Type::ByName(name), gci);
      }

      void Callback::operator () ( const Type& t ) {
         ArtificialSourceFile asf;
         int autoload = G__set_class_autoloading(0); // To avoid recursive loads
         if ( t.IsClass() || t.IsStruct() ) {
            ROOTClassEnhancer enhancer(t);
            enhancer.Setup();
            CINTClassBuilder::Get(t).Setup();
            enhancer.CreateInfo();
         }
         else if ( t.IsTypedef() ) {
            CINTTypedefBuilder::Setup(t);
         }
         else if ( t.IsEnum() ) {
            CINTEnumBuilder::Setup(t);
         } 
         G__set_class_autoloading(autoload);
      }
  
      void Callback::operator () ( const Member& m ) {
         ArtificialSourceFile asf;
         int autoload = G__set_class_autoloading(0); // To avoid recursive loads
         if ( m.IsFunctionMember() ) {
            if( Cintex::Debug() ) cout << "Cintex: Building function " << m.Name(SCOPED|QUALIFIED) << endl; 
            CINTFunctionBuilder(m).Setup();
         }
         else if ( m.IsDataMember() ) {
            if( Cintex::Debug() ) cout << "Cintex: Building variable " << m.Name(SCOPED|QUALIFIED) << endl; 
            CINTVariableBuilder(m).Setup();
         } 
         G__set_class_autoloading(autoload);
      }
   }
}
