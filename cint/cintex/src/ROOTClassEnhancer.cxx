// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "ROOTClassEnhancer.h"
#include "CINTdefs.h"
#include "CINTFunctional.h"
#include "Cintex/Cintex.h"
#include "TROOT.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClassTable.h"
#include "TClassStreamer.h"
#include "TCollectionProxyInfo.h"
#include "TVirtualCollectionProxy.h"
#include "TMemberInspector.h"
#include "RVersion.h"
#include "Reflex/Reflex.h"
#include "Reflex/Tools.h"
#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Builder/CollectionProxy.h"
#include "Api.h"
#define G__DICTIONARY
#include "RtypesImp.h"
#undef  G__DICTIONARY

#include <sstream>
#include <memory>

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
#include "TVirtualIsAProxy.h"
#endif

using namespace ROOT::Reflex;
using namespace ROOT::Cintex;
using namespace std;

namespace ROOT { namespace Cintex {

   class IsAProxy;

   class ROOTClassEnhancerInfo  {

      Type                     fType;
      string                   fName;
      TClass*                  fTclass;
      TClass*                  fLastClass;
      std::map<const std::type_info*,TClass*> fSub_types;
      const  std::type_info*   fLastType;
      const  std::type_info*   fMyType;
      bool                     fIsVirtual;
      ROOT::TGenericClassInfo* fClassInfo;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
      IsAProxy*                fIsa_func;
#else
      IsAFunc_t                fIsa_func;
#endif
      VoidFuncPtr_t            fDictionary_func;
      Int_t                    fVersion;

   public:
      ROOTClassEnhancerInfo(Type& t);
      virtual ~ROOTClassEnhancerInfo();

      virtual void Setup(void);
      virtual void CreateInfo(void);
      TClass* Tclass() {
         if ( fTclass == 0 ) {
            fTclass = ROOT::GetROOT()->GetClass( Name().c_str() /*, kFALSE */);
         }
         return fTclass;
      }
      const Type&   TypeGet() const { return fType; }
      const string& Name() const { return fName; }
      ROOT::TGenericClassInfo* Info() const { return fClassInfo; }
      Int_t         Version() const {return fVersion;}

      void AddFunction( const std::string& Name, const ROOT::Reflex::Type& sig,
                        ROOT::Reflex::StubFunction stubFP, void*  stubCtx, int );
      TClass* IsA(const void* obj);
      static void* Stub_IsA2(void* ctxt, void* obj);
      static void Stub_IsA(void* ret, void*, const std::vector<void*>&, void*);
      static void Stub_Streamer(void*, void*, const std::vector<void*>&, void*);
      static void Stub_StreamerNVirtual(void*, void*, const std::vector<void*>&, void*);
      static void Stub_Dictionary(void* ret, void*, const std::vector<void*>&, void*);
      static void Stub_ShowMembers(void*, void*, const std::vector<void*>&, void*);
      static void Stub_ShowMembers(TClass*, const ROOT::Reflex::Type&, void*, TMemberInspector&);
      static void Stub_Dictionary( void* ctx );
      static TClass* Default_CreateClass(Type typ, ROOT::TGenericClassInfo* info);
   };

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
   class IsAProxy : public TVirtualIsAProxy {
   public:
      IsAProxy( ROOTClassEnhancerInfo* info ) : fInfo(info), fClass(0) {}
      void SetClass(TClass *cl) { fClass = cl;}
      TClass* operator()(const void *obj){ return obj == 0 ? fClass : fInfo->IsA(obj); }
   private:
      ROOTClassEnhancerInfo* fInfo;
      TClass* fClass;
   };
#endif

   class ROOTEnhancerCont : public std::vector<ROOTClassEnhancerInfo*>  {
   public:
      ROOTEnhancerCont() {}
      ~ROOTEnhancerCont()  {
         for(std::vector<ROOTClassEnhancerInfo*>::iterator j=begin(); j!= end(); ++j)
            delete (*j);
         clear();
      }
   };

   std::vector<ROOTClassEnhancerInfo*>& rootEnhancerInfos()  {
      static ROOTEnhancerCont s_cont;
      return s_cont;
   }

   ROOTClassEnhancer::ROOTClassEnhancer(const ROOT::Reflex::Type& cl):
      fEnhancerinfo(0)
   {
      // Constructor.
      fClass = CleanType(cl);
      fName  = CintName(fClass);
   }

   ROOTClassEnhancer::~ROOTClassEnhancer() {
      // Destructor.
   }

   void ROOTClassEnhancer::Setup() {
      // Enhance root class info.
      VoidFuncPtr_t dict_func = TClassTable::GetDict(fName.c_str());
      if (dict_func) {
         fEnhancerinfo = 0; // Prevent adding the TClass to root twice.
      }
      else {
         ROOTClassEnhancerInfo* p = new ROOTClassEnhancerInfo(fClass);
         fEnhancerinfo = p;
         p->Setup();
      }
   }

   void ROOTClassEnhancer::CreateInfo() {
      // Enhance root class info.
      if ( fEnhancerinfo ) {
         ROOTClassEnhancerInfo* p = (ROOTClassEnhancerInfo*)fEnhancerinfo;
         p->CreateInfo();
      }
   }

   TClass* ROOTClassEnhancer::Default_CreateClass(Type typ, ROOT::TGenericClassInfo* info) {
      // forward to ROOTClassEnhancerInfo
      return ROOTClassEnhancerInfo::Default_CreateClass(typ, info);
   }

   /// Access streamer info from a void (polymorph) pointer
   TClass* accessType(const TClass* cl, const void* /* ptr */)  {
      return (TClass*)cl;
   }

   ROOTClassEnhancerInfo::ROOTClassEnhancerInfo(Type& t) :
      fTclass(0), fLastClass(0), fLastType(0)
   {
      // Constructor.
      fType = CleanType(t);
      fName = CintName(fType);
      rootEnhancerInfos().push_back(this);
      fMyType = &t.TypeInfo();
      fIsVirtual = TypeGet().IsVirtual();
      fClassInfo = 0;
      fIsa_func = 0;
      fDictionary_func = 0;
      fVersion = 0;
   }

   ROOTClassEnhancerInfo::~ROOTClassEnhancerInfo() {
      // Destructor.
      fSub_types.clear();
      if ( fClassInfo ) delete fClassInfo;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
      // fIsa_func is deleted by ROOT
#else
      if ( fIsa_func ) Free_function((void*)fIsa_func);
#endif
      if ( fDictionary_func ) Free_function((void*)fDictionary_func);
   }

   void ROOTClassEnhancerInfo::Setup() {
      // Setup root class enhance.
      std::string nam = TypeGet().Name(SCOPED);
      if ( Cintex::Debug() > 1 )  {
         std::cout << "Cintex: Enhancing:" << nam << std::endl;
      }
      fVersion = 1;
      if (TypeGet().Properties().HasProperty("ClassVersion")) {
         std::stringstream ssVersion(TypeGet().Properties().PropertyAsString("ClassVersion"));
         if (ssVersion.good())
            ssVersion >> fVersion;
         if ( Cintex::Debug() > 2 )  {
            cout << "Cintex: ROOTClassEnhancer: setting class version of " << nam << " to " << fVersion << endl;
         }
      }
      if ( ! IsSTLext(nam) && (IsSTL(nam) || IsSTLinternal(nam))) {
         //--- create TGenericClassInfo Instance
         //createInfo();
         return;
      }
      else if (TypeGet().Properties().HasProperty("ClassDef")) {
         return;
      }
      else    {
         Type void_t = Type::ByName("void");
         Type char_t = Type::ByName("char");
         Type signature;
         void* ctxt = this;

         signature = FunctionTypeBuilder( void_t, ReferenceBuilder(TypeBuilder("TBuffer")));
         Member exists = fType.FunctionMemberByName("StreamerNVirtual", signature,
                                                    0, INHERITEDMEMBERS_NO, DELAYEDLOAD_OFF);
         if (!exists) {
            //AddFunction("Streamer", signature, Stub_Streamer, ctxt, VIRTUAL);
            //AddFunction("StreamerNVirtual", signature, Stub_StreamerNVirtual, ctxt, 0);

            //--- adding TClass* IsA()
            signature = FunctionTypeBuilder( PointerBuilder(TypeBuilder("TClass")));
            AddFunction("IsA", signature, Stub_IsA, ctxt, 0);
            //--- adding void Data_ShowMembers(void *, TMemberInspector&, char*)
            signature = FunctionTypeBuilder( void_t,
                                             ReferenceBuilder(TypeBuilder("TMemberInspector")));
            
            AddFunction("ShowMembers", signature, Stub_ShowMembers, ctxt,
                        /*should be VIRTUAL but avoid vtable creation:*/
                        fType.IsVirtual() ? VIRTUAL : 0);

            //--- create TGenericClassInfo Instance
            //createInfo();
         }
      }
   }

   void ROOTClassEnhancerInfo::CreateInfo() {
      //---Check is the the dictionary is already defined for the class
      VoidFuncPtr_t dict = TClassTable::GetDict(Name().c_str());
      if ( dict ) return;

      ::ROOT::TGenericClassInfo* info = 0;
      void* context = this;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
      fIsa_func = new IsAProxy(this);
#else
      fIsa_func = (IsAFunc_t)Allocate_1arg_function(context, Stub_IsA2);
#endif
      fDictionary_func = Allocate_void_function(context, Stub_Dictionary);

      info = new ::ROOT::TGenericClassInfo(
                                           Name().c_str(),           // Class Name
                                           Version(),                // class version
                                           "",                       // declaration file Name
                                           1,                        // declaration line number
                                           TypeGet().TypeInfo(),     // typeid
                                           ROOT::DefineBehavior(0,0),// default behavior
                                           0,                        // show members function
                                           fDictionary_func,         // dictionary function
                                           fIsa_func,                // IsA function
                                           0,                        // pragma bits
                                           TypeGet().SizeOf()        // sizeof
                                           );

      if (info) info->SetImplFile("", 1);
      //----Fill the New and Deletete functions
      Member getfuncs = TypeGet().FunctionMemberByName("__getNewDelFunctions", Reflex::Type(),
                                                       0, INHERITEDMEMBERS_NO, DELAYEDLOAD_OFF);
      if( getfuncs ) {
         NewDelFunctions_t* newdelfunc = 0;
         getfuncs.Invoke(newdelfunc);

         if ( newdelfunc ) {
            info->SetNew(newdelfunc->fNew);
            info->SetNewArray(newdelfunc->fNewArray);
            info->SetDelete(newdelfunc->fDelete);
            info->SetDeleteArray(newdelfunc->fDeleteArray);
            info->SetDestructor(newdelfunc->fDestructor);
         }
      }


      //------------------------------------------------------------------------
      // Deal with the schema evolution rules
      //------------------------------------------------------------------------
      if( TypeGet().Properties().HasProperty( "ioread" ) ) {
         Any& obj = TypeGet().Properties().PropertyValue( "ioread" );
         std::vector<ROOT::TSchemaHelper> rules = any_cast<std::vector<ROOT::TSchemaHelper> >( obj );
         info->SetReadRules( rules );
      }

      if( TypeGet().Properties().HasProperty( "ioreadraw" ) ) {
         Any& obj = TypeGet().Properties().PropertyValue( "ioreadraw" );
         std::vector<ROOT::TSchemaHelper> rules = any_cast<std::vector<ROOT::TSchemaHelper> >( obj );
         info->SetReadRawRules( rules );
      }


      fClassInfo = info;
   }

   struct Fornamespace_t {};
   void ROOTClassEnhancer::CreateClassForNamespace(const std::string& Name ) {
      // Create root class.
      ROOT::CreateClass(Name.c_str(),         // Name
                        0,                    // version
                        typeid(Fornamespace_t), // typeid
                        0,                    // TVirtualIsAProxy *isa,
                        0,                    // ShowMembersFunc_t show,
                        "",                // definition file
                        "",                // implementation file
                        1,                    // definition line number
                        1 );                  // implementation line number
   }

   void ROOTClassEnhancerInfo::AddFunction( const std::string& name,
                                            const Type & sig,
                                            StubFunction stubFP,
                                            void*  stubCtx,
                                            int mods)
   {
      // Add function info.
      fType.AddFunctionMember( name.c_str(), sig, stubFP, stubCtx, 0, PUBLIC | mods );
   }

   inline static ROOTClassEnhancerInfo& context(void* ctxt)  {
      if ( ctxt )  {
         return *(ROOTClassEnhancerInfo*)ctxt;
      }
      throw std::runtime_error("Invalid stub context passes to emultated function!");
   }


   void ROOTClassEnhancerInfo::Stub_IsA(void* ret, void* obj, const vector<void*>&, void* ctx) {
      // Root IsA.
      *((TClass**)ret) = context(ctx).IsA(obj);
   }
   void* ROOTClassEnhancerInfo::Stub_IsA2(void* ctx, void* obj) {
      // Root IsA.
      return context(ctx).IsA(obj);
   }

   struct DynamicStruct_t  {    virtual ~DynamicStruct_t() {}  };

   TClass* ROOTClassEnhancerInfo::IsA(const void* obj)  {
      // Root IsA.
      if ( ! obj || ! fIsVirtual )  {
         return Tclass();
      }
      else  {
         // Avoid the case that the first word is a virtual_base_offset_table instead of
         // a virtual_function_table
         long Offset = **(long**)obj;
         if ( Offset == 0 ) return Tclass();

         DynamicStruct_t* p = (DynamicStruct_t*)obj;
         const std::type_info& typ = typeid(*p);

         if ( &typ == fMyType )  {
            return Tclass();
         }
         else if ( &typ == fLastType )  {
            return fLastClass;
         }
         // Check if TypeNth is already in sub-class cache
         else if ( 0 != (fLastClass=fSub_types[&typ]) )  {
            fLastType = &typ;
         }
         // Last resort: lookup root class
         else   {
            std::string nam;
            Type t = Type::ByTypeInfo(typ);
            if (t) nam = CintName(t);
            else   nam = CintName(Tools::Demangle(typ));
            fLastClass = ROOT::GetROOT()->GetClass(nam.c_str());
            fSub_types[fLastType=&typ] = fLastClass;
         }
      }
      //std::cout << "Cintex: IsA:" << TypeNth.Name(SCOPED) << " dynamic:" << dtype.Name(SCOPED) << std::endl;
      return fLastClass;
   }

   TClass* ROOTClassEnhancerInfo::Default_CreateClass( Type typ, ROOT::TGenericClassInfo* info)  {
      // Create root class.
      TClass* root_class = 0;
      std::string Name = typ.Name(SCOPED);
      int kind = TClassEdit::IsSTLCont(Name.c_str());
      if ( kind < 0 ) kind = -kind;

      const std::type_info& tid = typ.TypeInfo();
      root_class = info->GetClass();

      if ( 0 != root_class )   {
         root_class->Size();
         if ( ! typ.IsVirtual() ) root_class->SetGlobalIsA(accessType);
         auto_ptr<TClassStreamer> str;
         switch(kind)  {
         case TClassEdit::kVector:
         case TClassEdit::kList:
         case TClassEdit::kDeque:
         case TClassEdit::kMap:
         case TClassEdit::kMultiMap:
         case TClassEdit::kSet:
         case TClassEdit::kMultiSet:
         case TClassEdit::kBitSet:
            {
               Member method = typ.FunctionMemberByName("createCollFuncTable", Reflex::Type(), 0,
                                                        INHERITEDMEMBERS_NO, DELAYEDLOAD_OFF);
               if ( !method )   {
                  if ( Cintex::Debug() )  {
                     cout << "Cintex: " << Name << "' Setup failed to create this class! "
                          << "The function createCollFuncTable is not availible."
                          << endl;
                  }
                  return 0;
               }
               CollFuncTable* m = 0;
               method.Invoke(m);

               ::ROOT::TCollectionProxyInfo cpinfo(tid,
                                                   m->iter_size,
                                                   m->value_diff,
                                                   m->value_offset,
                                                   m->size_func,
                                                   m->resize_func,
                                                   m->clear_func,
                                                   m->first_func,
                                                   m->next_func,
                                                   m->construct_func,
                                                   m->destruct_func,
                                                   m->feed_func,
                                                   m->collect_func,
                                                   m->create_env,
                                                   m->fCreateIterators,
                                                   m->fCopyIterator,
                                                   m->fNext,
                                                   m->fDeleteSingleIterator,
                                                   m->fDeleteTwoIterators
                                                   );
               root_class->SetCollectionProxy(cpinfo);

               root_class->SetBit(TClass::kIsForeign);
            }
            break;
         case TClassEdit::kNotSTL:
         case TClassEdit::kEnd:
         default:
            if (!typ.Properties().HasProperty("ClassDef")) {
               root_class->SetBit(TClass::kIsForeign);
            }
         }
      }
      return root_class;
   }

   void ROOTClassEnhancerInfo::Stub_Dictionary(void* ctx )
   {
      // Create class info.
      if( Cintex::GetROOTCreator() ) {
         (*Cintex::GetROOTCreator())( context(ctx).TypeGet(), context(ctx).Info() );
      }
      else {
         //context(ctx).Info()->GetClass();
         Default_CreateClass( context(ctx).TypeGet(), context(ctx).Info() );
      }
   }


   void ROOTClassEnhancerInfo::Stub_Streamer(void*, void* obj, const vector<void*>& args, void* ctx) {
      //  Create streamer info.
      TBuffer& b = *(TBuffer*)args[0];
      TClass* cl = context(ctx).Tclass();
      TClassStreamer* s = cl->GetStreamer();
      if ( s )  {
         (*s)(b, obj);
      }
      else if ( b.IsWriting() )  {
         cl->WriteBuffer(b, obj);
      }
      else {
         UInt_t start, count;
         Version_t version = b.ReadVersion(&start, &count, cl);
         cl->ReadBuffer(b, obj, version, start, count);
      }
   }

   void ROOTClassEnhancerInfo::Stub_StreamerNVirtual(void*, void* obj, const vector<void*>& args, void* ctx) {
      // Create streamer info.
      TBuffer& b = *(TBuffer*)args[0];
      TClass* cl = context(ctx).Tclass();
      TClassStreamer* s = cl->GetStreamer();
      if ( s )  {
         (*s)(b, obj);
      }
      else if ( b.IsWriting() )  {
         cl->WriteBuffer(b, obj);
      }
      else {
         UInt_t start, count;
         Version_t version = b.ReadVersion(&start, &count, cl);
         cl->ReadBuffer(b, obj, version, start, count);
      }
   }

   void ROOTClassEnhancerInfo::Stub_ShowMembers(void*, void* obj, const vector<void*>& args, void* ctx) {
      // Create show members.
      Type typ = context(ctx).TypeGet();
      TClass* tcl = context(ctx).Tclass();
      TMemberInspector& insp = *(TMemberInspector*)args[0];
      if( tcl ) Stub_ShowMembers( tcl, typ, obj, insp);
   }

   void ROOTClassEnhancerInfo::Stub_ShowMembers(TClass* tcl, const Type& cl, void* obj, TMemberInspector& insp) {
      if ( tcl->GetShowMembersWrapper() )    {
         tcl->GetShowMembersWrapper()(obj, insp);
         return;
      }

      // Create show members.
      // Loop over data members
      if ( IsSTL(cl.Name(SCOPED)) || cl.IsArray() ) return;
      for ( size_t m = 0; m < cl.DataMemberSize(INHERITEDMEMBERS_NO); m++) {
         Member mem = cl.DataMemberAt(m, INHERITEDMEMBERS_NO);
         if ( ! mem.IsStatic() ) {
            Type typ = mem.TypeOf();
            string nam = mem.Properties().HasProperty("ioname") ?
               mem.Properties().PropertyAsString("ioname") : mem.Name();
            if( typ.IsPointer() ) nam = "*" + nam;
            if( typ.IsArray() ) {
               std::stringstream s;
               s << typ.ArrayLength();
               nam += "[" + s.str() + "]";
            }
            char*  add = (char*)obj + mem.Offset();
            if ( Cintex::Debug() > 2 )  {
               cout << "Cintex: Showmembers: ("<< tcl->GetName() << ") " << nam.c_str()
                    << " = " << (void*)add << " Offset:" << mem.Offset() << endl;
            }
            insp.Inspect(tcl, insp.GetParent(), nam.c_str(), add);
            if ( !typ.IsFundamental() && !typ.IsPointer() ) {
               string tnam  = mem.Properties().HasProperty("iotype") ? CintName(mem.Properties().PropertyAsString("iotype")) : CintName(typ);
               TClass* tmcl = ROOT::GetROOT()->GetClass(tnam.c_str(), kTRUE, mem.IsTransient());
               if ( tmcl ) {
                  insp.InspectMember(tmcl, add, (nam + ".").c_str());
               }
            }
         }
      }
      // Loop over bases
      for ( size_t b = 0; b < cl.BaseSize(); b++ ) {
         Base BaseNth = cl.BaseAt(b);
         string bname = CintName(BaseNth.ToType());
         char* ptr = (char*)obj + BaseNth.Offset(obj);
         TClass* bcl = ROOT::GetROOT()->GetClass(bname.c_str());
         if( bcl ) Stub_ShowMembers( bcl, BaseNth.ToType(), ptr, insp);
      }
   }
}}
