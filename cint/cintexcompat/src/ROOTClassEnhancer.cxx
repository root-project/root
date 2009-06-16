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

namespace ROOT {
namespace Cintex {

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
class IsAProxy;
#endif // ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)

//______________________________________________________________________________
class ROOTClassEnhancerInfo {
   Type fType;
   string fName;
   TClass* fTclass;
   TClass* fLastClass;
   std::map<const std::type_info*, TClass*> fSub_types;
   const std::type_info* fLastType;
   const std::type_info* fMyType;
   bool fIsVirtual;
   ROOT::TGenericClassInfo* fClassInfo;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
   IsAProxy* fIsa_func;
#else // ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
   IsAFunc_t fIsa_func;
#endif // ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
   VoidFuncPtr_t fDictionary_func;
   ShowMembersFunc_t fShowMembers_func;
   Int_t fVersion;
public:
   ROOTClassEnhancerInfo(Type&);
   virtual ~ROOTClassEnhancerInfo();
   virtual void Setup(void);
   virtual void CreateInfo(void);
   TClass* Tclass() {
      if (!fTclass) {
         fTclass = ROOT::GetROOT()->GetClass(Name().c_str());
      }
      return fTclass;
   }
   const Type& TypeGet() const {
      return fType;
   }
   const string& Name() const {
      return fName;
   }
   ROOT::TGenericClassInfo* Info() const {
      return fClassInfo;
   }
   Int_t Version() const {
      return fVersion;
   }
   void AddFunction(const std::string& Name, const ROOT::Reflex::Type& sig, ROOT::Reflex::StubFunction stubFP, void* stubCtx, int mods);
   TClass* IsA(const void* obj);
   static void* Stub_IsA2(void* ctxt, void* obj);
   static void Stub_IsA(void* ret, void*, const std::vector<void*>&, void*);
   static void Stub_Streamer(void*, void*, const std::vector<void*>&, void*);
   static void Stub_StreamerNVirtual(void*, void*, const std::vector<void*>&, void*);
   static void Stub_Dictionary(void* ret, void*, const std::vector<void*>&, void*); // NOT IMPLEMENTED
   static void Stub_ShowMembers2(void*, void*, TMemberInspector&, char*);
   static void Stub_ShowMembers(void*, void*, const std::vector<void*>&, void*);
   static void Stub_ShowMembers(TClass*, const ROOT::Reflex::Type&, void*, TMemberInspector&, char*);
   static void Stub_Dictionary(void* ctx);
   static TClass* Default_CreateClass(Type typ, ROOT::TGenericClassInfo* info);
};

//______________________________________________________________________________
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
class IsAProxy : public TVirtualIsAProxy {
   // --
private:
   ROOTClassEnhancerInfo* fInfo;
   TClass* fClass;
public:
   IsAProxy(ROOTClassEnhancerInfo* info) : fInfo(info), fClass(0) {}
   void SetClass(TClass* cl) { fClass = cl; }
   TClass* operator()(const void* obj) { return !obj ? fClass : fInfo->IsA(obj); }
};
#endif // ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)

//______________________________________________________________________________
//
//  rootEnchancerInfos singleton class
//

class ROOTEnhancerCont : public std::vector<ROOTClassEnhancerInfo*> {
public:
   ROOTEnhancerCont() {}
   ~ROOTEnhancerCont() {
      for (std::vector<ROOTClassEnhancerInfo*>::iterator j = begin(); j != end(); ++j) {
         delete *j;
      }
      clear();
   }
};

std::vector<ROOTClassEnhancerInfo*>& rootEnhancerInfos()
{
   static ROOTEnhancerCont s_cont;
   return s_cont;
}

//______________________________________________________________________________
//
//  Class ROOTClassEnchancer
//

//______________________________________________________________________________
ROOTClassEnhancer::ROOTClassEnhancer(const ROOT::Reflex::Type& cl)
{
   fClass = CleanType(cl);
   fName  = TClassEdit::GetLong64_Name( CintName(fClass) );
}

//______________________________________________________________________________
ROOTClassEnhancer::~ROOTClassEnhancer()
{
}

//______________________________________________________________________________
void ROOTClassEnhancer::Setup()
{
   ROOTClassEnhancerInfo* p = new ROOTClassEnhancerInfo(fClass);
   fEnhancerinfo = p;
   p->Setup();
}

//______________________________________________________________________________
void ROOTClassEnhancer::CreateInfo()
{
   if (fEnhancerinfo) {
      ROOTClassEnhancerInfo* p = (ROOTClassEnhancerInfo*)fEnhancerinfo;
      p->CreateInfo();
   }
}

//______________________________________________________________________________
TClass* ROOTClassEnhancer::Default_CreateClass(Type type, ROOT::TGenericClassInfo* info)
{
   return ROOTClassEnhancerInfo::Default_CreateClass(type, info);
}

//______________________________________________________________________________
//
//  Class ROOTClassEnhancerInfo
//

//______________________________________________________________________________
ROOTClassEnhancerInfo::ROOTClassEnhancerInfo(Type& t)
: fTclass(0)
, fLastClass(0)
, fLastType(0)
, fMyType(0)
, fIsVirtual(false)
, fClassInfo(0)
, fIsa_func(0)
, fDictionary_func(0)
, fShowMembers_func(0)
, fVersion(0)
{
   fType = CleanType(t);
   fName = CintName(fType);
   fMyType = &t.TypeInfo();
   fIsVirtual = TypeGet().IsVirtual();
   rootEnhancerInfos().push_back(this);
}

//______________________________________________________________________________
ROOTClassEnhancerInfo::~ROOTClassEnhancerInfo()
{
   delete fClassInfo;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
   // fIsa_func is deleted by ROOT
#else // ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
   if (fIsa_func) {
      Free_code((void*)fIsa_func);
   }
#endif // ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
   if (fDictionary_func) {
      Free_code((void*)fDictionary_func);
   }
   if (fShowMembers_func) {
      Free_code((void*)fShowMembers_func);
   }
}

//______________________________________________________________________________
void ROOTClassEnhancerInfo::Setup()
{
   std::string nam = TypeGet().Name(SCOPED);
   if (Cintex::Debug() > 1)  {
      std::cout << "Cintex: Enhancing:" << nam << std::endl;
   }
   fVersion = 1;
   if (TypeGet().Properties().HasProperty("ClassVersion")) {
      std::stringstream ssVersion(TypeGet().Properties().PropertyAsString("ClassVersion"));
      if (ssVersion.good()) {
         ssVersion >> fVersion;
      }
      if (Cintex::Debug() > 2)  {
         cout << "Cintex: ROOTClassEnhancer: setting class version of " << nam << " to " << fVersion << endl;
      }
   }
   if (!IsSTLext(nam) && (IsSTL(nam) || IsSTLinternal(nam))) {
      return;
   }
   if (TypeGet().Properties().HasProperty("ClassDef")) {
      return;
   }
   Type int_t = Type::ByName("int");
   Type void_t = Type::ByName("void");
   Type char_t = Type::ByName("char");
   Type signature;
   void* ctxt = this;
   //--- adding TClass* IsA()
   signature = FunctionTypeBuilder(PointerBuilder(TypeBuilder("TClass")));
   AddFunction("IsA", signature, Stub_IsA, ctxt, 0);
   //--- adding void Data_ShowMembers(void *, TMemberInspector&, char*)
   signature = FunctionTypeBuilder(void_t, ReferenceBuilder(TypeBuilder("TMemberInspector")), PointerBuilder(char_t));
   AddFunction("ShowMembers", signature, Stub_ShowMembers, ctxt, VIRTUAL);
   signature = FunctionTypeBuilder(void_t, ReferenceBuilder(TypeBuilder("TBuffer")));
   AddFunction("Streamer", signature, Stub_Streamer, ctxt, VIRTUAL);
   AddFunction("StreamerNVirtual", signature, Stub_StreamerNVirtual, ctxt, 0);
}

//______________________________________________________________________________
void ROOTClassEnhancerInfo::CreateInfo()
{
   VoidFuncPtr_t dict = TClassTable::GetDict(Name().c_str());
   if (dict) {
      return;
   }
   ::ROOT::TGenericClassInfo* info = 0;

   void* context = this;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
   fIsa_func = new IsAProxy(this);
#else // ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
   fIsa_func = (IsAFunc_t) Allocate_1arg_function(context, Stub_IsA2);
#endif // ROOT_VERSION_CODE >= ROOT_VERSION(5,1,1)
   fDictionary_func = Allocate_void_function(context, Stub_Dictionary);
   fShowMembers_func = Allocate_3arg_function(context, Stub_ShowMembers2);
   info = new ::ROOT::TGenericClassInfo(
                                        Name().c_str() // fullClassname, class name
                                        , Version() // version, class version
                                        , "" // declFileName, declaration file Name
                                        , 1 // declFileLine, declaration line number
                                        , TypeGet().TypeInfo() // info, typeid
                                        , ROOT::DefineBehavior(0, 0) // action, default behavior
                                        , 0 // (void*)&fShowMembers_func // showmembers, show members function
                                        , fDictionary_func // dictionary, dictionary function
                                        , fIsa_func // isa, IsA function
                                        , 0 // pragmabits, pragma bits
                                        , TypeGet().SizeOf() // sizof, size of
                                        );
   info->SetImplFile("", 1);
   Member getfuncs = TypeGet().MemberByName("__getNewDelFunctions", Reflex::Type(), INHERITEDMEMBERS_NO);
   if (getfuncs) {
      NewDelFunctions_t* newdelfunc = 0;
      getfuncs.Invoke(newdelfunc);
      if (newdelfunc) {
         info->SetNew(newdelfunc->fNew);
         info->SetNewArray(newdelfunc->fNewArray);
         info->SetDelete(newdelfunc->fDelete);
         info->SetDeleteArray(newdelfunc->fDeleteArray);
         info->SetDestructor(newdelfunc->fDestructor);
      }
   }
   //
   //  Deal with the schema evolution rules.
   //
   if (TypeGet().Properties().HasProperty("ioread")) {
      Any& obj = TypeGet().Properties().PropertyValue("ioread");
      std::vector<ROOT::TSchemaHelper> rules = any_cast<std::vector<ROOT::TSchemaHelper> >(obj);
      info->SetReadRules(rules);
   }
   if (TypeGet().Properties().HasProperty("ioreadraw")) {
      Any& obj = TypeGet().Properties().PropertyValue("ioreadraw");
      std::vector<ROOT::TSchemaHelper> rules = any_cast<std::vector<ROOT::TSchemaHelper> >(obj);
      info->SetReadRawRules(rules);
   }
   fClassInfo = info;
}

//______________________________________________________________________________
void ROOTClassEnhancerInfo::AddFunction(const std::string& name, const Type& sig, StubFunction stubFP, void* stubCtx, int mods)
{
   fType.AddFunctionMember(name.c_str(), sig, stubFP, stubCtx, 0, mods | PUBLIC);
}

//______________________________________________________________________________
struct DynamicStruct_t {
   virtual ~DynamicStruct_t();
};

//______________________________________________________________________________
DynamicStruct_t::~DynamicStruct_t()
{
}

//______________________________________________________________________________
TClass* ROOTClassEnhancerInfo::IsA(const void* obj)
{
   // Root IsA.
   if (! obj || !fIsVirtual)  {
      return Tclass();
   }
   // Avoid the case that the first word is a
   // virtual_base_offset_table instead of
   // a virtual_function_table.
   long Offset = **(long**)obj;
   if (!Offset) {
      return Tclass();
   }
   DynamicStruct_t* p = (DynamicStruct_t*) obj;
   const std::type_info& typ = typeid(*p);
   if (&typ == fMyType)  {
      return Tclass();
   }
   if (&typ == fLastType)  {
      return fLastClass;
   }
   // Check if type is already in sub-class cache
   fLastClass = fSub_types[&typ];
   if (fLastClass) {
      fLastType = &typ;
      return fLastClass;
   }
   // Last resort, lookup root class
   Type t = Type::ByTypeInfo(typ);
   std::string nam;
   if (t) {
      nam = CintName(t);
   }
   else {
      nam = CintName(Tools::Demangle(typ));
   }
   fLastClass = ROOT::GetROOT()->GetClass(nam.c_str());
   fLastType = &typ;
   fSub_types[fLastType] = fLastClass;
   return fLastClass;
}

//______________________________________________________________________________
static inline ROOTClassEnhancerInfo& context(void* ctxt)
{
   if (ctxt)  {
      return *(ROOTClassEnhancerInfo*)ctxt;
   }
   throw std::runtime_error("Invalid stub context passed to emulated function!");
}

//______________________________________________________________________________
void* ROOTClassEnhancerInfo::Stub_IsA2(void* ctx, void* obj)
{
   return context(ctx).IsA(obj);
}

//______________________________________________________________________________
void ROOTClassEnhancerInfo::Stub_IsA(void* ret, void* obj, const vector<void*>&, void* ctx)
{
   *((TClass**)ret) = context(ctx).IsA(obj);
}

//______________________________________________________________________________
void ROOTClassEnhancerInfo::Stub_Streamer(void*, void* obj, const vector<void*>& args, void* ctx)
{
   TBuffer& b = *(TBuffer*)args[0];
   TClass* cl = context(ctx).Tclass();
   TClassStreamer* s = cl->GetStreamer();
   if (s) {
      (*s)(b, obj);
   }
   else if (b.IsWriting()) {
      cl->WriteBuffer(b, obj);
   }
   else {
      UInt_t start = 0;
      UInt_t count = 0;
      Version_t version = b.ReadVersion(&start, &count, cl);
      cl->ReadBuffer(b, obj, version, start, count);
   }
}

//______________________________________________________________________________
void ROOTClassEnhancerInfo::Stub_StreamerNVirtual(void*, void* obj, const vector<void*>& args, void* ctx)
{
   TBuffer& b = *(TBuffer*)args[0];
   TClass* cl = context(ctx).Tclass();
   TClassStreamer* s = cl->GetStreamer();
   if (s) {
      (*s)(b, obj);
   }
   else if (b.IsWriting())  {
      cl->WriteBuffer(b, obj);
   }
   else {
      UInt_t start = 0;
      UInt_t count = 0;
      Version_t version = b.ReadVersion(&start, &count, cl);
      cl->ReadBuffer(b, obj, version, start, count);
   }
}

//______________________________________________________________________________
void ROOTClassEnhancerInfo::Stub_ShowMembers2(void* ctx, void* obj, TMemberInspector& insp, char* parent)
{
   TClass* tcl = context(ctx).Tclass();
   if (!tcl) {
      return;
   }
   Type typ = context(ctx).TypeGet();
   Stub_ShowMembers(tcl, typ, obj, insp, parent);
}

//______________________________________________________________________________
void ROOTClassEnhancerInfo::Stub_ShowMembers(void*, void* obj, const vector<void*>& args, void* ctx)
{
   TClass* tcl = context(ctx).Tclass();
   if (!tcl) {
      return;
   }
   Type typ = context(ctx).TypeGet();
   TMemberInspector& insp = *(TMemberInspector*)args[0];
   char* par = (char*) args[1];
   Stub_ShowMembers(tcl, typ, obj, insp, par);
}

//______________________________________________________________________________
void ROOTClassEnhancerInfo::Stub_ShowMembers(TClass* tcl, const Type& cl, void* obj, TMemberInspector& insp, char* par)
{
   if (tcl->GetShowMembersWrapper())    {
      tcl->GetShowMembersWrapper()(obj, insp, par);
      return;
   }
   int ncp = ::strlen(par);
   if (IsSTL(cl.Name(SCOPED)) || cl.IsArray()) {
      return;
   }
   for (size_t m = 0; m < cl.DataMemberSize(INHERITEDMEMBERS_NO); ++m) {
      Member mem = cl.DataMemberAt(m, INHERITEDMEMBERS_NO);
      if (mem.IsTransient()) {
         continue;
      }
      Type typ = mem.TypeOf();
      string nam = mem.Properties().HasProperty("ioname") ?
                   mem.Properties().PropertyAsString("ioname") : mem.Name();
      if (typ.IsPointer()) {
         nam = "*" + nam;
      }
      if (typ.IsArray()) {
         std::stringstream s;
         s << typ.ArrayLength();
         nam += "[" + s.str() + "]";
      }
      char* add = (char*) obj + mem.Offset();
      if (Cintex::Debug() > 2)  {
         cout << "Cintex: Showmembers: ("
              << tcl->GetName()
              << ") "
              << par
              << nam.c_str()
              << " = "
              << (void*) add
              << " Offset:"
              << mem.Offset()
              << endl;
      }
      insp.Inspect(tcl, par, nam.c_str(), add);
      if (!typ.IsFundamental() && !typ.IsPointer()) {
         string tnam = mem.Properties().HasProperty("iotype") ? CintName(mem.Properties().PropertyAsString("iotype")) : CintName(typ);
         TClass* tmcl = ROOT::GetROOT()->GetClass(tnam.c_str());
         if (tmcl) {
            strcat(par, nam.c_str());
            strcat(par, ".");
            Stub_ShowMembers(tmcl, typ, add, insp, par);
            par[ncp] = 0;
         }
      }
   }
   for (size_t b = 0; b < cl.BaseSize(); ++b) {
      Base BaseNth = cl.BaseAt(b);
      string bname = CintName(BaseNth.ToType());
      char* ptr = (char*) obj + BaseNth.Offset(obj);
      TClass* bcl = ROOT::GetROOT()->GetClass(bname.c_str());
      if (bcl) {
         Stub_ShowMembers(bcl, BaseNth.ToType(), ptr, insp, par);
      }
   }
}

//______________________________________________________________________________
void ROOTClassEnhancerInfo::Stub_Dictionary(void* ctx)
{
   if (Cintex::GetROOTCreator()) {
      (*Cintex::GetROOTCreator())(context(ctx).TypeGet(), context(ctx).Info());
      return;
   }
   Default_CreateClass(context(ctx).TypeGet(), context(ctx).Info());
}

//______________________________________________________________________________
static TClass* accessType(const TClass* cl, const void* /*ptr*/)
{
   return (TClass*) cl;
}

//______________________________________________________________________________
TClass* ROOTClassEnhancerInfo::Default_CreateClass(Type typ, ROOT::TGenericClassInfo* info)
{
   TClass* root_class = 0;
   std::string Name = typ.Name(SCOPED);
   int kind = TClassEdit::IsSTLCont(Name.c_str());
   if (kind < 0) {
      kind = -kind;
   }
   const char* tagname = Name.c_str();
   int tagnum = G__defined_tagname(tagname, 2);
   G__ClassInfo cl_info(tagnum);
   if (cl_info.IsValid()) {
      switch (kind)  {
         case TClassEdit::kVector:
         case TClassEdit::kList:
         case TClassEdit::kDeque:
         case TClassEdit::kMap:
         case TClassEdit::kMultiMap:
         case TClassEdit::kSet:
         case TClassEdit::kMultiSet:
            cl_info.SetVersion(4);
            break;
         case TClassEdit::kBitSet:
            cl_info.SetVersion(2);
         case TClassEdit::kNotSTL:
         case TClassEdit::kEnd:
            cl_info.SetVersion(1);
            break;
      }
   }
   const std::type_info& tid = typ.TypeInfo();
   root_class = info->GetClass();
   if (!root_class) {
      return 0;
   }
   root_class->Size();
   if (!typ.IsVirtual()) {
      root_class->SetGlobalIsA(accessType);
   }
   switch (kind)  {
      case TClassEdit::kVector:
      case TClassEdit::kList:
      case TClassEdit::kDeque:
      case TClassEdit::kMap:
      case TClassEdit::kMultiMap:
      case TClassEdit::kSet:
      case TClassEdit::kMultiSet:
      case TClassEdit::kBitSet:
         {
            Member method = typ.MemberByName("createCollFuncTable", Reflex::Type(), INHERITEDMEMBERS_NO);
            if (!method) {
               if (Cintex::Debug())  {
                  cout << "Cintex: "
                       << Name
                       << "' Setup failed to create this class! "
                       << "The function createCollFuncTable is not availible."
                       << endl;
               }
               return 0;
            }
            CollFuncTable* m = 0;
            method.Invoke(m);
            ROOT::TCollectionProxyInfo cpinfo(
                 tid
               , m->iter_size
               , m->value_diff
               , m->value_offset
               , m->size_func
               , m->resize_func
               , m->clear_func
               , m->first_func
               , m->next_func
               , m->construct_func
               , m->destruct_func
               , m->feed_func
               , m->collect_func
               , m->create_env
            );
            root_class->SetCollectionProxy(cpinfo);
            root_class->SetBit(TClass::kIsForeign);
         }
         break;
      case TClassEdit::kNotSTL:
      case TClassEdit::kEnd:
      default:
         root_class->SetBit(TClass::kIsForeign);
   }
   return root_class;
}

} // namespace Cintex
} // namespace ROOT
