// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Reflex.h"
#include "Reflex/Tools.h"
#include "Cintex/Cintex.h"
#include "CINTdefs.h"
#include "CINTClassBuilder.h"
#include "CINTScopeBuilder.h"
#include "CINTFunctionBuilder.h"
#include "CINTVariableBuilder.h"
#include "CINTTypedefBuilder.h"
#include "CINTEnumBuilder.h"
#include "CINTFunctional.h"
#include "Api.h"
#include "TError.h"
#include <list>
#include <set>
#include <iomanip>
#include <sstream>

using namespace ROOT::Reflex;
using namespace std;

namespace ROOT { namespace Cintex {
  
   void* CINTClassBuilder::fgFakeObject  = 0;
   void* CINTClassBuilder::fgFakeAddress = &CINTClassBuilder::fgFakeObject;

   struct PendingBase_t {
      Type   fBasetype;
      int    fTagnum;
      size_t fOffset;
      PendingBase_t( const Type& t, int n, size_t o) : fBasetype(t), fTagnum(n), fOffset(o) {}
   };


   //   static list<PendingBase>& pendingBases() {
   //      static list<PendingBase> s_pendingBases;
   //      return s_pendingBases;
   //   }  

   CINTClassBuilder& CINTClassBuilder::Get(const Type& cl) {
      // Run all builders.
      CINTClassBuilders& builders = CINTClassBuilders::Instance();
      CINTClassBuilders::iterator it = builders.find(cl);
      if( it != builders.end() )  return *(*it).second;
      CINTClassBuilder* builder = new CINTClassBuilder(cl);
      builders[cl] = builder;
      return *builder;
   }

   CINTClassBuilder::CINTClassBuilder(const Type& cl)
      : fClass(cl), fName(CintName(cl)), fPending(true), 
        fSetup_memvar(0), fSetup_memfunc(0), fBases(0)
   {
      // CINTClassBuilder constructor.
      fTaginfo = new G__linked_taginfo;
      fTaginfo->tagnum  = -1;   // >> need to be pre-initialized to be understood by CINT
      fTaginfo->tagtype = 'c';
      fTaginfo->tagname = fName.c_str();
      fTaginfo->tagnum = G__defined_tagname(fTaginfo->tagname, 2);

      if ( fTaginfo->tagnum < 0 )  {
         Setup_tagtable();
      }
      else   {
         G__ClassInfo info(fTaginfo->tagnum);
         // if the scope is a class and was used before it might happen that
         // it was assumed to be a namespace, reset to class if this was the case
         if (!(info.Property() & (G__BIT_ISSTRUCT | G__BIT_ISCLASS))) {
            // update from 'n' or 'a'
            G__search_tagname(fTaginfo->tagname, cl.IsClass() ? 'c' : 's'); 
            Setup_tagtable();
         }
         else if ( !info.IsLoaded() )  {
            Setup_tagtable();
         }
         else  {
            fPending = false;
            if( Cintex::Debug() > 1 ) std::cout << "Cintex: Precompiled class:" << fName << std::endl;
         }
      }
   }

   CINTClassBuilder::~CINTClassBuilder() {
      // CINTClassBuilder destructor.
      delete fTaginfo;
      Free_function((void*)fSetup_memfunc);
      Free_function((void*)fSetup_memvar);
   }

   void CINTClassBuilder::Setup() {
      // Setup a Cint class.
      if ( fPending ) {
         if ( Cintex::Debug() ) std::cout << "Cintex: Building class " << fName << std::endl;
         fPending = false;

         // Setup_memfunc();        // It is delayed
         // Setup_memvar();         // It is delayed
         Setup_inheritance();
         Setup_typetable();
      }
      return;
   }

   void CINTClassBuilder::Setup_tagtable() {
      // Setup scope.
      static bool alreadyWarnedAboutTooManyClasses = false;
      Scope scope = fClass.DeclaringScope();
      if ( scope ) CINTScopeBuilder::Setup(scope);
      else {
         scope = Scope::ByName(Tools::GetScopeName(fClass.Name(SCOPED)));
         if( scope.Id() ) CINTScopeBuilder::Setup(scope);
      }

      // Setup tag number
      fTaginfo->tagnum = G__get_linked_tagnum(fTaginfo);
      if (!alreadyWarnedAboutTooManyClasses
          && Cint::G__ClassInfo::GetNumClasses() > 0.9 * G__MAXSTRUCT) {
         alreadyWarnedAboutTooManyClasses = true;
         Warning("CINTClassBuilder::Setup_tagtable()",
                 "%d out of %d possible entries are in use!",
                 (int)Cint::G__ClassInfo::GetNumClasses(), (int)G__MAXSTRUCT);
      }
      std::string comment = fClass.Properties().HasProperty("comment") ? 
         fClass.Properties().PropertyAsString("comment").c_str() :
         "";
      // Assume some minimal class functionality; see below for explanation
      int rootFlag = 0;
      rootFlag   += 0x00020000;         // No operator >> ()
      //if ( 0 != streamer() )    {     // If a customized streamer exists
      //  if ( !(isIntrinsic() || getFlag(ISSTRING)) )  {
      //    rootFlag += 0x00010000;     // Bit 1 Set
      //  }                             //
      //}                               //
      //else  {                         // Automatic schema evolution
      //  rootFlag += 0x00040000;       //
      //}                               //
      if ( fClass.IsAbstract() ) {
         rootFlag += G__BIT_ISABSTRACT;  // Abstract class
      }                                 //
      //  rootFlag += 0x00000100;       // Class has Empty Constructor
      //  rootFlag += 0x00000400;       // Class has destructor
      //if ( 0 != fCopyConstructor ) { //
      //  rootFlag += 0x00000200;       // Class has copy Constructor
      //}                               //
      if ( fClass.HasBase(Type::ByName("TObject")))   {
         rootFlag += 0x00007000;       // Class has inherits from TObject
      }                               //
      if ( fClass.TypeInfo() == typeid(std::string) )  {
         rootFlag = 0x48F00;
      }

      fSetup_memvar  = Allocate_void_function(this, Setup_memvar_with_context );
      fSetup_memfunc = Allocate_void_function(this, Setup_memfunc_with_context );

      G__tagtable_setup( fTaginfo->tagnum,    // tag number
                         fClass.SizeOf(),     // size
                         G__CPPLINK,          // cpplink
                         rootFlag,            // isabstract
                         comment.empty() ? 0 : comment.c_str(), // comment
                         fSetup_memvar,       // G__setup_memvarMyClass
                         fSetup_memfunc);     // G__setup_memfuncMyClass
   }

   void CINTClassBuilder::Setup_memfunc_with_context(void* ctx) {
      // Setup a CINT member function.
      int autoload = G__set_class_autoloading(0); // To avoid recursive loads
      ((CINTClassBuilder*)ctx)->Setup_memfunc();
      G__set_class_autoloading(autoload);
   }
   void CINTClassBuilder::Setup_memvar_with_context(void* ctx) {
      // Setup a CINT data member.
      int autoload = G__set_class_autoloading(0); // To avoid recursive loads
      ((CINTClassBuilder*)ctx)->Setup_memvar();
      G__set_class_autoloading(autoload);
   }

   void CINTClassBuilder::Setup_memfunc() {
      // Setup a CINT member function.
      for ( size_t i = 0; i < fClass.FunctionMemberSize(INHERITEDMEMBERS_NO); i++ ) 
         CINTScopeBuilder::Setup(fClass.FunctionMemberAt(i, INHERITEDMEMBERS_NO).TypeOf());

      G__tag_memfunc_setup(fTaginfo->tagnum);
      for ( size_t i = 0; i < fClass.FunctionMemberSize(INHERITEDMEMBERS_NO); i++ ) {
         Member method = fClass.FunctionMemberAt(i, INHERITEDMEMBERS_NO); 
         std::string n = method.Name();
         CINTFunctionBuilder::Setup(method);
      }
      ::G__tag_memfunc_reset();

   }

   void CINTClassBuilder::Setup_memvar() {
      // Setup a CINT data member.
      for ( size_t i = 0; i < fClass.DataMemberSize(INHERITEDMEMBERS_NO); i++ ) 
         CINTScopeBuilder::Setup(fClass.DataMemberAt(i, INHERITEDMEMBERS_NO).TypeOf());

      G__tag_memvar_setup(fTaginfo->tagnum);

      // Set placeholder for virtual function table if the class is virtual
      if ( fClass.IsVirtual() ) {
         G__memvar_setup((void*)0,'l',0,0,-1,-1,-1,4,"G__virtualinfo=",0,0);
      }

      if ( ! IsSTL(fClass.Name(SCOPED)) )  {
         for ( size_t i = 0; i < fClass.DataMemberSize(INHERITEDMEMBERS_NO); i++ ) {

            Member dm = fClass.DataMemberAt(i, INHERITEDMEMBERS_NO);
            CINTVariableBuilder::Setup(dm);
         }
      }
      G__tag_memvar_reset();

   }

   CINTClassBuilder::Bases* CINTClassBuilder::GetBases() {
      // Get base class info.
      if ( fBases ) return fBases;
      Member getbases = fClass.FunctionMemberByName("__getBasesTable", Reflex::Type(), 0, INHERITEDMEMBERS_NO, DELAYEDLOAD_OFF);
      if ( !getbases ) getbases = fClass.FunctionMemberByName("getBasesTable", Reflex::Type(), 0, INHERITEDMEMBERS_NO, DELAYEDLOAD_OFF);
      if( getbases ) {
         static Type tBases = Type::ByTypeInfo(typeid(Bases));
         Object ret(tBases, &fBases);
         getbases.Invoke(&ret);
      }
      else {
         static Bases s_bases;
         fBases = &s_bases;
      }
      return fBases;
   }

   void CINTClassBuilder::Setup_inheritance() {
      // Setup inheritance info.
      if ( 0 == ::G__getnumbaseclass(fTaginfo->tagnum) )  {     
         bool isVirtual = false; 
         for ( Bases::iterator it = GetBases()->begin(); it != GetBases()->end(); it++ )
            if( (*it).first.IsVirtual() ) isVirtual = true;

         if ( isVirtual ) {
            if ( !fClass.IsAbstract() )  {
               Member ctor, dtor;
               for ( size_t i = 0; i < fClass.FunctionMemberSize(INHERITEDMEMBERS_NO); i++ ) {
                  Member method = fClass.FunctionMemberAt(i, INHERITEDMEMBERS_NO); 
                  if( method.IsConstructor() && method.FunctionParameterSize() == 0 )  ctor = method;
                  else if ( method.IsDestructor() )  dtor = method;
               }
               if ( ctor )  {
                  Object obj = fClass.Construct();
                  Setup_inheritance(obj);
                  /*if ( dtor )*/ fClass.Destruct(obj.Address());
               }
               // There is no default constructor. So, it is not a I/O class
               else {  
                  Object obj(fClass, 0);
                  Setup_inheritance(obj);
               }
            }
            // Special case of "pure abstract". The offsets will be Set to 0.
            // All that is necessary because ROOT does not handle virtual inheritance correctly.
            // ROOT always wants to Get a real offset between base classes ans this is not 
            // possible without having an Instance of the object. In case of pure abstract classes
            // we can not do it. So, in case the abstract class has no data members then we assume
            // offsets to base class to be 0.
            else if ( fClass.IsAbstract() && fClass.DataMemberSize(INHERITEDMEMBERS_NO) == 0) {
               Object obj(fClass, 0);
               Setup_inheritance(obj);
            }
            // The above fails for Gaudi Algorithms (virutal inheritance, abstract and with
            // data members. Do not know What to do.
            else {  
               Object obj(fClass, 0);
               Setup_inheritance(obj);
            }
         }
         else {
            Object obj(fClass, fgFakeAddress);
            Setup_inheritance(obj);
         }
      }
   }
   void CINTClassBuilder::Setup_inheritance(Object& obj) {
      // Setup inheritance info.
      if ( ! IsSTL(fClass.Name(SCOPED)) )    {
         if ( 0 == ::G__getnumbaseclass(fTaginfo->tagnum) )  {
            for ( Bases::iterator it = GetBases()->begin(); it != GetBases()->end(); it++ ) {
               Base base  = it->first;
               int  level = it->second;
               Type btype = base.ToType();
               CINTScopeBuilder::Setup(btype);
               std::string b_nam = CintName(btype);
               int b_tagnum = G__search_tagname(b_nam.c_str(), 'a');
               // Get the Offset. Treat differently virtual and non-virtual inheritance
               size_t offset;
               long int type = (level == 0) ?  G__ISDIRECTINHERIT : 0;
               if ( base.IsVirtual() ) {
                  if (obj.Address())  {
                     offset = ( * base.OffsetFP())(obj.Address());
                  }
                  else {
                     offset = (size_t)base.OffsetFP();
                     type = type | G__ISVIRTUALBASE;
                  }
               }
               else {
                  offset = base.Offset(fgFakeAddress);
               }
               if( Cintex::Debug() > 1 )  {
                  std::cout << "Cintex: " << fClass.Name(SCOPED) << " Base:" << btype.Name(SCOPED) << " Offset:" << offset << std::endl;
               }
               int mod = base.IsPublic() ? G__PUBLIC : ( base.IsPrivate() ? G__PRIVATE : G__PROTECTED );
               ::G__inheritance_setup(fTaginfo->tagnum, b_tagnum, offset, mod, type );
               Object bobj(btype,(char*)obj.Address() + offset);
            }
         }
      }    
   }

   void CINTClassBuilder::Setup_typetable() {
      // Setup types.
      for (Type_Iterator ti = fClass.SubType_Begin(); ti != fClass.SubType_End(); ++ti) {
         if (Cintex::PropagateClassTypedefs() && ti->IsTypedef() ) {
            CINTTypedefBuilder::Setup(*ti);
            CINTScopeBuilder::Setup(ti->ToType());
         }
         else if (Cintex::PropagateClassEnums() && ti->IsEnum() ) {
            CINTEnumBuilder::Setup(*ti);
         }
      }

   }

}}
