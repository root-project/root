// @(#)root/cintex:$Name:  $:$Id: ROOTClassEnhancer.cxx,v 1.7 2006/05/03 15:49:40 axel Exp $
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
#include "TCollectionProxy.h"
#include "TVirtualCollectionProxy.h"
#include "TMemberInspector.h"
#include "RVersion.h"
#include "Reflex/Reflex.h"
#include "Reflex/Tools.h"
#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Builder/CollectionProxy.h"
#include "Api.h"

#include <sstream>

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

  public:
    ROOTClassEnhancerInfo(Type& t);
    virtual ~ROOTClassEnhancerInfo();

    virtual void Setup(void);
    virtual void CreateInfo(void);
    TClass* tclass() {  
      if ( fTclass == 0 ) {
        fTclass = ROOT::GetROOT()->GetClass( Name().c_str() /*, kFALSE */);
      }
      return fTclass; 
    }
    const Type&   TypeGet() const { return fType; }
    const string& Name() const { return fName; }
    ROOT::TGenericClassInfo* info() const { return fClassInfo; }

    void addFunction( const std::string& Name, const ROOT::Reflex::Type& sig,
                        ROOT::Reflex::StubFunction stubFP, void*  stubCtx, int );
    TClass* IsA(const void* obj);
    static void* stub_IsA2(void* ctxt, void* obj);
    static void* stub_IsA(void*, const std::vector<void*>&, void*);
    static void* stub_Streamer(void*, const std::vector<void*>&, void*);
    static void* stub_StreamerNVirtual(void*, const std::vector<void*>&, void*);
    static void* stub_Dictionary(void*, const std::vector<void*>&, void*);
    static void* stub_ShowMembers(void*, const std::vector<void*>&, void*);
    static void  stub_ShowMembers(TClass*, const ROOT::Reflex::Type&, void*, TMemberInspector&, char*);
    static void  stub_Dictionary( void* ctx );
    static TClass* default_CreateClass(Type typ, ROOT::TGenericClassInfo* info);
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

  ROOTClassEnhancer::ROOTClassEnhancer(const ROOT::Reflex::Type& cl)  {
    fClass = CleanType(cl);
    fName  = CintName(fClass);
  }

  ROOTClassEnhancer::~ROOTClassEnhancer() {
  }

  void ROOTClassEnhancer::Setup() {
    ROOTClassEnhancerInfo* p = new ROOTClassEnhancerInfo(fClass);
    fEnhancerinfo = p;
    p->Setup();
  }

  void ROOTClassEnhancer::CreateInfo() {
    if ( fEnhancerinfo ) {
     ROOTClassEnhancerInfo* p = (ROOTClassEnhancerInfo*)fEnhancerinfo;
     p->CreateInfo();
    }
  }


  /// Access streamer info from a void (polymorph) pointer
  TClass* accessType(const TClass* cl, const void* /* ptr */)  {
    return (TClass*)cl;
  }

  ROOTClassEnhancerInfo::ROOTClassEnhancerInfo(Type& t) : 
    fTclass(0), fLastClass(0), fLastType(0)
  {
    fType = CleanType(t);
    fName = CintName(fType);
    rootEnhancerInfos().push_back(this);
    fMyType = &t.TypeInfo();
    fIsVirtual = TypeGet().IsVirtual();
    fClassInfo = 0;
    fIsa_func = 0;
    fDictionary_func = 0;
 }

  ROOTClassEnhancerInfo::~ROOTClassEnhancerInfo() {
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
    std::string nam = TypeGet().Name(SCOPED);
    if ( Cintex::Debug() > 1 )  {
      std::cout << "Enhancing:" << nam << std::endl;
    }
    if ( ! IsSTLext(nam) && (IsSTL(nam) || IsSTLinternal(nam)) )  {
      //--- create TGenericClassInfo Instance
      //createInfo();
      return;
    }
    else    {
      Type int_t  = Type::ByName("int");
      Type void_t = Type::ByName("void");
      Type char_t = Type::ByName("char");
      Type signature;
      void* ctxt = this;

      //--- adding TClass* IsA()
      signature = FunctionTypeBuilder( PointerBuilder(TypeBuilder("TClass")));
      addFunction("IsA", signature, stub_IsA, ctxt, 0);
      //--- adding void Data_ShowMembers(void *, TMemberInspector&, char*)
      signature = FunctionTypeBuilder( void_t, 
                                      ReferenceBuilder(TypeBuilder("TMemberInspector")),
                                      PointerBuilder(char_t));
      addFunction("ShowMembers", signature, stub_ShowMembers, ctxt, VIRTUAL);
      signature = FunctionTypeBuilder( void_t, ReferenceBuilder(TypeBuilder("TBuffer")));
      addFunction("Streamer", signature, stub_Streamer, ctxt, VIRTUAL);
      addFunction("StreamerNVirtual", signature, stub_StreamerNVirtual, ctxt, 0);
    }
    //--- create TGenericClassInfo Instance
    //createInfo();
  }

  void ROOTClassEnhancerInfo::CreateInfo() {
    //---Check is the the dictionary is already defined for the class
    VoidFuncPtr_t dict = TClassTable::GetDict(Name().c_str());
    if ( dict ) return;

    void* context = this;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
    fIsa_func = new IsAProxy(this);
#else
    fIsa_func = (IsAFunc_t)Allocate_1arg_function(context, stub_IsA2);
#endif
    fDictionary_func = Allocate_void_function(context, stub_Dictionary);

    ::ROOT::TGenericClassInfo* info = new ::ROOT::TGenericClassInfo( 
              Name().c_str(),                     // Class Name 
              "",                              // declaration file Name 
              1,                                  // declaration line number
              TypeGet().TypeInfo(),                  // typeid 
              ROOT::DefineBehavior(0,0),          // default behavior
              0,                                  // show members function 
              fDictionary_func,                  // dictionary function 
              fIsa_func,                         // IsA function
              0,                                  // pragma bits
              TypeGet().SizeOf()                     // sizeof
           );
    info->SetImplFile("", 1);
    //----Fill the New and Deletete functions
    Member getfuncs = TypeGet().MemberByName("__getNewDelFunctions");
    if( getfuncs ) {
      NewDelFunctions* newdelfunc = (NewDelFunctions*)( getfuncs.Invoke().Address() );
      if ( newdelfunc ) {
        info->SetNew(newdelfunc->New);
        info->SetNewArray(newdelfunc->NewArray);
        info->SetDelete(newdelfunc->Delete);
        info->SetDeleteArray(newdelfunc->DeleteArray);
        info->SetDestructor(newdelfunc->Destructor);
      }
    }
    fClassInfo = info;
  }

  struct fornamespace {};
  void ROOTClassEnhancer::CreateClassForNamespace(const std::string& Name ) {

    ROOT::CreateClass(Name.c_str(),         // Name
                      0,                    // version
                      typeid(fornamespace), // typeid
                      0,                    // TVirtualIsAProxy *isa,
                      0,                    // ShowMembersFunc_t show,
                      "",                // definition file
                      "",                // implementation file
                      1,                    // definition line number
                      1 );                  // implementation line number
  }

  void ROOTClassEnhancerInfo::addFunction( const std::string& Name, 
                                           const Type & sig,
                                           StubFunction stubFP, 
                                           void*  stubCtx,
                                           int mods)
  {
    fType.AddFunctionMember( Name.c_str(), sig, stubFP, stubCtx, 0, PUBLIC | mods );
  }

  inline static ROOTClassEnhancerInfo& context(void* ctxt)  {
    if ( ctxt )  {
      return *(ROOTClassEnhancerInfo*)ctxt;
    }
    throw std::runtime_error("Invalid stub context passes to emultated function!");
  }


  void* ROOTClassEnhancerInfo::stub_IsA(void* obj, const vector<void*>&, void* ctx) {
    return context(ctx).IsA(obj);
  }
  void* ROOTClassEnhancerInfo::stub_IsA2(void* ctx, void* obj) {
    return context(ctx).IsA(obj);
  }

  struct DynamicStruct  {    virtual ~DynamicStruct() {}  };

  TClass* ROOTClassEnhancerInfo::IsA(const void* obj)  {
    if ( ! obj || ! fIsVirtual )  {
      return tclass();
    }
    else  {
      // Avoid the case that the first word is a virtual_base_offset_table instead of
      // a virtual_function_table  
      long Offset = **(long**)obj;
      if ( Offset == 0 ) return tclass();
      
      DynamicStruct* p = (DynamicStruct*)obj;
      const std::type_info& typ = typeid(*p);
      
      if ( &typ == fMyType )  {
        return tclass();
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
        Type t = Type::ByTypeInfo(typ);
        std::string nam = CintName(t);
        fLastClass = ROOT::GetROOT()->GetClass(nam.c_str());
        fSub_types[fLastType=&typ] = fLastClass;
      }
    }
    //std::cout << "IsA:" << TypeNth.Name(SCOPED) << " dynamic:" << dtype.Name(SCOPED) << std::endl;
    return fLastClass;
  }
  
  TClass* ROOTClassEnhancerInfo::default_CreateClass( Type typ, ROOT::TGenericClassInfo* info)  {
    TClass* root_class = 0;
    std::string Name = typ.Name(SCOPED);
    int kind = TClassEdit::IsSTLCont(Name.c_str());
    if ( kind < 0 ) kind = -kind;
    const char* tagname = Name.c_str();
    int tagnum = ::G__defined_tagname(tagname, 2);
    G__ClassInfo cl_info(tagnum);
    if ( cl_info.IsValid() )  {
      switch(kind)  {
        case TClassEdit::kVector:
        case TClassEdit::kList:
        case TClassEdit::kDeque:
        case TClassEdit::kMap:
        case TClassEdit::kMultiMap:
        case TClassEdit::kSet:
        case TClassEdit::kMultiSet:
          cl_info.SetVersion(4);
          break;
        case TClassEdit::kNotSTL:
        case TClassEdit::kEnd:
          cl_info.SetVersion(1);
          break;
      }
    }

    const std::type_info& tid = typ.TypeInfo();
    root_class = info->GetClass();

    if ( 0 != root_class )   {
      root_class->Size();
      if ( ! typ.IsVirtual() ) root_class->SetGlobalIsA(accessType);
      std::auto_ptr<TClassStreamer> str;
      switch(kind)  {
        case TClassEdit::kVector:
        case TClassEdit::kList:
        case TClassEdit::kDeque:
        case TClassEdit::kMap:
        case TClassEdit::kMultiMap:
        case TClassEdit::kSet:
        case TClassEdit::kMultiSet:
          {
            Member method = typ.MemberByName("createCollFuncTable");
            if ( !method )   {
              if ( Cintex::Debug() )  {
                cout << Name << "' Setup failed to create this class! "
                << "The function createCollFuncTable is not availible."
                << endl;
              }
              return 0;
            }
            std::auto_ptr<CollFuncTable> m((CollFuncTable*)method.Invoke().Address());
            std::auto_ptr<TCollectionProxy::Proxy_t> proxy(
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,4,0)
              TCollectionProxy::GenExplicitProxy(tid,
#else
              TCollectionProxy::genExplicitProxy(tid,
#endif
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
              m->collect_func
              ));
              std::auto_ptr<TClassStreamer> str(
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,4,0)
              TCollectionProxy::GenExplicitClassStreamer(tid,
#else
              TCollectionProxy::genExplicitClassStreamer(tid,
#endif
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
              m->collect_func
              ));
            root_class->CopyCollectionProxy(*(proxy.get()));
            root_class->SetBit(TClass::kIsForeign);
            if ( str.get() )  {
              root_class->AdoptStreamer(str.release());
            }
          }
          break;
        case TClassEdit::kNotSTL:
        case TClassEdit::kEnd:
        default:
          root_class->SetBit(TClass::kIsForeign);
      }
    }
    return root_class;
  }

  void ROOTClassEnhancerInfo::stub_Dictionary(void* ctx )
  {
    if( Cintex::GetROOTCreator() ) {
      (*Cintex::GetROOTCreator())( context(ctx).TypeGet(), context(ctx).info() );
    }
    else {
       //context(ctx).info()->GetClass();
       default_CreateClass( context(ctx).TypeGet(), context(ctx).info() );
    }  
  }


  void* ROOTClassEnhancerInfo::stub_Streamer(void* obj, const vector<void*>& args, void* ctx) {
    TBuffer& b = *(TBuffer*)args[0];
    TClass* cl = context(ctx).tclass();
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
    return 0;
  }

  void* ROOTClassEnhancerInfo::stub_StreamerNVirtual(void* obj, const vector<void*>& args, void* ctx) {
    TBuffer& b = *(TBuffer*)args[0];
    TClass* cl = context(ctx).tclass();
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
    return 0;
  }

  void* ROOTClassEnhancerInfo::stub_ShowMembers(void* obj, const vector<void*>& args, void* ctx) {
    const Type& TypeNth = context(ctx).TypeGet();
    TClass* tcl = context(ctx).tclass();
    TMemberInspector& insp = *(TMemberInspector*)args[0];
    char* par = (char*)args[1];
    if( tcl ) stub_ShowMembers( tcl, TypeNth, obj, insp, par);
    return 0;
  }

  void ROOTClassEnhancerInfo::stub_ShowMembers(TClass* tcl, const Type& cl, void* obj, TMemberInspector& insp, char* par) {
    int ncp = ::strlen(par);
    // Loop over data members
    if ( IsSTL(cl.Name(SCOPED)) || cl.IsArray() ) return;
    for ( size_t m = 0; m < cl.DataMemberSize(); m++) {
      Member mem = cl.DataMemberAt(m);
      if ( ! mem.IsTransient() ) {
        Type typ = mem.TypeOf();
        string nam = mem.Properties().HasKey("ioname") ? 
                     mem.Properties().PropertyAsString("ioname") : mem.Name();
        if( typ.IsPointer() ) nam = "*" + nam;
        if( typ.IsArray() ) {
          std::stringstream s;
          s << typ.ArrayLength();
          nam += "[" + s.str() + "]";
        }
        char*  add = (char*)obj + mem.Offset();
        if ( Cintex::Debug() > 2 )  {
          cout << "Showmembers: ("<< tcl->GetName() << ") " << par << nam.c_str() 
            << " = " << (void*)add << " Offset:" << mem.Offset() << endl;
        }
        insp.Inspect(tcl, par, nam.c_str(), add);
        if ( !typ.IsFundamental() && !typ.IsPointer() ) {
          string tnam  = CintName(typ);
          TClass* tmcl = ROOT::GetROOT()->GetClass(tnam.c_str());
          if ( tmcl ) {
            ::strcat(par,nam.c_str());
            ::strcat(par,".");
            stub_ShowMembers(tmcl, typ, add, insp, par);
            par[ncp] = 0;
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
      if( bcl ) stub_ShowMembers( bcl, BaseNth.ToType(), ptr, insp, par);
    }
  }
}}
