// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Cintex/Cintex.h"

// Cintex
#include "CINTClassBuilder.h"
#include "CINTEnumBuilder.h"
#include "CINTFunctionBuilder.h"
#include "CINTSourceFile.h"
#include "CINTTypedefBuilder.h"
#include "CINTVariableBuilder.h"
#include "ROOTClassEnhancer.h"

// Reflex
#include "Reflex/Builder/ReflexBuilder.h"
#include "Reflex/Callback.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"

// Cint
#include "G__ci.h"

// Internal CINT
#include "../../cint7/src/common.h"

#include <iostream>

using namespace ROOT::Reflex;
using namespace ROOT::Cintex;
using namespace std;

namespace {

//______________________________________________________________________________
struct Cintex_dict_t {

   // --

public: // Public Static Interface

   static void Enable(void*, void*, const std::vector<void*>&, void*)
   {
      ::ROOT::Cintex::Cintex::Enable();
   }

   static void SetDebug(void*, void*, const std::vector<void*>& arg, void*)
   {
      ::ROOT::Cintex::Cintex::SetDebug(*(bool*)arg[0]);
   }

   static void Debug(void*, void* ret, const std::vector<void*>&, void*)
   {
      if (ret) {
         *(int*)ret = ::ROOT::Cintex::Cintex::Debug();
      }
      else {
         ::ROOT::Cintex::Cintex::Debug();
      }
   }

   static void PropagateClassTypedefs(void*, void* ret, const std::vector<void*>&, void*)
   {
      if (ret) {
         *(bool*)ret = ::ROOT::Cintex::Cintex::PropagateClassTypedefs();
      }
      else {
         ::ROOT::Cintex::Cintex::PropagateClassTypedefs();
      }
   }

   static void SetPropagateClassTypedefs(void*, void*, const std::vector<void*>& arg, void*)
   {
      ::ROOT::Cintex::Cintex::SetPropagateClassTypedefs(*(bool*)arg[0]);
   }

public: // Public Interface.

   Cintex_dict_t()
   {
      Type t_void = TypeBuilder("void");
      Type t_int = TypeBuilder("int");
      Type t_bool = TypeBuilder("bool");

      // No need to disable the CallBack here since we should be ran before Cintex is enable
      // (i.e. to is run at library load time while the Cintex::Enable needs to be explicit)

      ClassBuilderT<ROOT::Cintex::Cintex>("Cintex", PUBLIC)
         .AddFunctionMember(FunctionTypeBuilder(t_void), "Enable", Enable, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_void, t_int), "SetDebug", SetDebug, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_int), "Debug", Debug, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_bool), "PropagateClassTypedefs", PropagateClassTypedefs, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_void, t_bool), "SetPropagateClassTypedefs", SetPropagateClassTypedefs, 0, 0, PUBLIC | STATIC);

      ClassBuilderT<ROOT::Cintex::Cintex>("ROOT::Cintex", PUBLIC)
         .AddFunctionMember(FunctionTypeBuilder(t_void), "Enable", Enable, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_void, t_int), "SetDebug", SetDebug, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_int), "Debug", Debug, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_bool), "PropagateClassTypedefs", PropagateClassTypedefs, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_void, t_bool), "SetPropagateClassTypedefs", SetPropagateClassTypedefs, 0, 0, PUBLIC | STATIC);

      ClassBuilderT<ROOT::Cintex::Cintex>("ROOT::Cintex::Cintex", PUBLIC)
         .AddFunctionMember(FunctionTypeBuilder(t_void), "Enable", Enable, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_void, t_int), "SetDebug", SetDebug, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_int), "Debug", Debug, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_bool), "PropagateClassTypedefs", PropagateClassTypedefs, 0, 0, PUBLIC | STATIC)
         .AddFunctionMember(FunctionTypeBuilder(t_void, t_bool), "SetPropagateClassTypedefs", SetPropagateClassTypedefs, 0, 0, PUBLIC | STATIC);

      G__linked_taginfo mydictLN_ROOT = { "ROOT" , 110 , -1 };
      mydictLN_ROOT.tagnum = -1 ;
      G__get_linked_tagnum_fwd(&mydictLN_ROOT);

      G__linked_taginfo mydictLN_Cintex = { "Cintex" , 110 , -1 };
      mydictLN_Cintex.tagnum = -1 ;
      G__get_linked_tagnum_fwd(&mydictLN_Cintex);

      G__linked_taginfo mydictLN_ROOTcLcLCintex = { "ROOT::Cintex" , 110 , -1 };
      mydictLN_ROOTcLcLCintex.tagnum = -1 ;
      G__get_linked_tagnum_fwd(&mydictLN_ROOTcLcLCintex);

      G__linked_taginfo mydictLN_ROOTcLcLCintexcLcLCintex = { "ROOT::Cintex::Cintex" , 99 , -1 };
      mydictLN_ROOTcLcLCintexcLcLCintex.tagnum = -1 ;
      G__get_linked_tagnum_fwd(&mydictLN_ROOTcLcLCintexcLcLCintex);

      Type outer = Type::ByName("Cintex");
      ::ROOT::Cintex::CINTClassBuilder::Get(outer).Setup();

      Type t = Type::ByName("ROOT::Cintex");
      ::ROOT::Cintex::CINTClassBuilder::Get(t).Setup();

      Type nested = Type::ByName("ROOT::Cintex::Cintex");
      ::ROOT::Cintex::CINTClassBuilder::Get(nested).Setup();
   }

};

//______________________________________________________________________________
static Cintex_dict_t s_dict;

//______________________________________________________________________________
static const char* btypes[] = {
   "bool"
   , "char"
   , "unsigned char"
   , "short"
   , "unsigned short"
   , "int"
   , "unsigned int"
   , "long"
   , "unsigned long"
   , "float"
   , "double"
   , "string"
};

//______________________________________________________________________________
void Declare_additional_CINT_typedefs()
{
   char name[4096];
   char value[8192];
   int autoload = G__set_class_autoloading(0); // To avoid recursive loads
   for (size_t i = 0; i < sizeof(btypes) / sizeof(char*); ++i) {
      sprintf(name, "vector<%s>", btypes[i]);
      sprintf(value, "vector<%s,allocator<%s> >", btypes[i], btypes[i]);
      CINTTypedefBuilder::Set(name, value);
   }
   // Now that genreflex always translates basic_string<char> to string
   // we need a "typedef" (the wrong way!) for backward compatibility:
   CINTTypedefBuilder::Set("basic_string<char>", "string");
   G__set_class_autoloading(autoload);
}

} // unnamed namespace


//void ::Cintex::Enable()
//{
//   ::ROOT::Cintex::Cintex::Enable();
//}

namespace ROOT {
namespace Cintex {

//______________________________________________________________________________
Cintex& Cintex::Instance()
{
   static Cintex s_instance;
   return s_instance;
}

//______________________________________________________________________________
Cintex::Cintex()
{
   fCallback = new Callback();
   fRootcreator = 0;
   fDbglevel = 0;
   fPropagateClassTypedefs = true;
   fPropagateClassEnums = true;
   fEnabled = false;
}

//______________________________________________________________________________
Cintex::~Cintex()
{
   if (fCallback) {
      UninstallClassCallback(fCallback);
   }
   delete fCallback;
}

//______________________________________________________________________________
void Cintex::Enable()
{
   if (Instance().fEnabled) {
      return;
   }
   Declare_additional_CINT_typedefs(); // Declare some extra typdefs to please CINT
   InstallClassCallback(Instance().fCallback);
   //
   //
   //
   static size_t pid = Reflex::PropertyList::KeyByName("Cint Properties", true);
   //
   //  Notify cint of all classes, structs, enums,
   //  and typedefs in the reflex dictionary.
   //
   for (size_t i = 0; i < Type::TypeSize(); ++i) {
      Type ty = Type::TypeAt(i);
      if (!ty.IsClass() && !ty.IsEnum() && !ty.IsTypedef()) {
         continue;
      }
      const TypeBase* tb = ty.ToTypeBase();
      G__RflxProperties* p = 0;
      if (tb) {
         p = (G__RflxProperties*) tb->Properties().PropertyValue(pid).Address();
         // If we have a properties, this type has been seen by CINT but possibly
         // only has a forward declaration like operation.
         if (p && p->filenum == -1 && ty.SizeOf()>0 && p->tagnum!=-1 && (Cint::Internal::G__struct.size[p->tagnum]==0 &&  Cint::Internal::G__struct.type[p->tagnum]!='e')) {
            // Force the update
            p = 0;
            //fprintf(stderr,"The type %s has been only partially setup %d %d \n",ty.Name(Reflex::SCOPED).c_str(),ty.SizeOf(),Cint::Internal::G__struct.size[p->tagnum]);
         }
      }
      if (!p) { // No cint properties, only reflex knows this type, inform cint.
         (*Instance().fCallback)(Type::TypeAt(i));
      }
   }
   //
   //  Notify cint of all namespace members in
   //  the reflex dictionary.
   //
   //  Note: These can only be data members or
   //        function members.
   //
   for (size_t i = 0; i < Scope::ScopeSize(); ++i) {
      Scope ns = Scope::ScopeAt(i);
      if (ns.IsNamespace()) { // Skip classes, structs, unions, and enums.
         for (size_t j = 0; j < ns.MemberSize(); ++j) {
            Member mbr = ns.MemberAt(j);
            const MemberBase* mb = mbr.ToMemberBase();
            void* p = 0;
            if (mb) {
               p = mb->Properties().PropertyValue(pid).Address();
            }
            if (!p) { // No cint properties, only reflex knows this member, inform cint.
               (*Instance().fCallback)(mbr);
            }
         }
      }
   }
   Instance().fEnabled = true;
}

//______________________________________________________________________________
void Cintex::SetROOTCreator(ROOTCreator_t c)
{
   Instance().fRootcreator = c;
}

//______________________________________________________________________________
ROOTCreator_t Cintex::GetROOTCreator()
{
   return Instance().fRootcreator;
}

//______________________________________________________________________________
int Cintex::Debug()
{
   return Instance().fDbglevel;
}

//______________________________________________________________________________
void Cintex::SetDebug(int l)
{
   Instance().fDbglevel = l;
}

//______________________________________________________________________________
bool Cintex::PropagateClassTypedefs()
{
   return Instance().fPropagateClassTypedefs;
}

//______________________________________________________________________________
void Cintex::SetPropagateClassTypedefs(bool val)
{
   Instance().fPropagateClassTypedefs = val;
}

//______________________________________________________________________________
bool Cintex::PropagateClassEnums()
{
   return Instance().fPropagateClassEnums;
}

//______________________________________________________________________________
void Cintex::SetPropagateClassEnums(bool val)
{
   Instance().fPropagateClassEnums = val;
}

//______________________________________________________________________________
void Cintex::Default_CreateClass(const char* name, TGenericClassInfo* gci)
{
   // Create a TClass object from the Reflex data; forward to ROOTClassEnhancer.
   ROOTClassEnhancer::Default_CreateClass(Reflex::Type::ByName(name), gci);
}

//______________________________________________________________________________
class ArtificialSourceFile {
   // --
private:
   G__input_file fOldIFile;
public:
   ArtificialSourceFile()
   {
      G__setfilecontext("{CINTEX dictionary translator}", &fOldIFile);
   }
   ~ArtificialSourceFile()
   {
      G__input_file* ifile = G__get_ifile();
      if (ifile) {
         *ifile = fOldIFile;
      }
   }
};

//______________________________________________________________________________
void Callback::operator()(const Type& t)
{
   ArtificialSourceFile asf;
   int autoload = G__set_class_autoloading(0); // To avoid recursive loads
   //cerr << "Callback::operator()(const Type&): " << "class: " << t.Name(SCOPED) << " with " << t.IsClass() << endl;
   if (t.IsClass()) {
      ROOTClassEnhancer enhancer(t);
      enhancer.Setup();
      CINTClassBuilder::Get(t).Setup();
      enhancer.CreateInfo();
   }
   else if (t.IsEnum()) {
      CINTEnumBuilder::Setup(t);
   }
   else if (t.IsTypedef()) {
      CINTTypedefBuilder::Setup(t);
   }
   G__set_class_autoloading(autoload);
}

//______________________________________________________________________________
void Callback::operator()(const Member& mbr)
{
   ArtificialSourceFile asf;
   int autoload = G__set_class_autoloading(0); // To avoid recursive loads
   //
   //
   //
   static size_t pid = Reflex::PropertyList::KeyByName("Cint Properties", true);
   const MemberBase* mb = mbr.ToMemberBase();
   void* p = 0;
   if (mb) {
      p = mb->Properties().PropertyValue(pid).Address();
   }
   if (!p) { // Not known to cint
      if (mbr.IsFunctionMember() || mbr.IsTemplateInstance()) {
         if (Cintex::Debug()) {
            cout << "Cintex: Building function " << mbr.Name(SCOPED | QUALIFIED) << endl;
         }
         if (!mbr.Stubfunction()) {
            cout << "Error: ROOT::Cintex::Callback:operator()(const Member&): member '" << mbr.Name(SCOPED | QUALIFIED) << "' has no dictionary stub function!" << endl;
         }
         CINTFunctionBuilder(mbr).Setup();
      }
      else if (mbr.IsDataMember()) {
         if (Cintex::Debug()) {
            cout << "Cintex: Building variable " << mbr.Name(SCOPED | QUALIFIED) << endl;
         }
         CINTVariableBuilder(mbr).Setup();
      }
      else {
         assert(0); // Cannot happen!
      }
   }
   else {
      cout << "Error: ROOT::Cintex::Callback:operator()(const Member&): member '" << mbr.Name(SCOPED | QUALIFIED) << "' has cint properties!" << endl;
   }
   G__set_class_autoloading(autoload);
}

} // namespace Cintex
} // namespace ROOT
