// @(#)root/meta:$Id$
// Author: Rene Brun   07/01/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClass
#define ROOT_TClass


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClass                                                               //
//                                                                      //
// Dictionary of a class.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#ifndef ROOT_TObjString
#include "TObjString.h"
#endif
#include <map>
#include <string>

class TBaseClass;
class TBrowser;
class TDataMember;
class TClassRef;
class TMethod;
class TRealData;
class TCint;
class TBuffer;
class TVirtualStreamerInfo;
class TVirtualCollectionProxy;
class TMethodCall;
class TVirtualIsAProxy;
class TVirtualRefProxy;
class THashTable;

namespace clang {
   class Decl;
}

namespace ROOT {
   class TGenericClassInfo;
   class TCollectionProxyInfo;
   class TSchemaRuleSet;
}

namespace ROOT {
   class TMapTypeToTClass;
}
typedef ROOT::TMapTypeToTClass IdMap_t;

class TClass : public TDictionary {

friend class TCint;
friend class TCintWithCling;
friend void ROOT::ResetClassVersion(TClass*, const char*, Short_t);
friend class ROOT::TGenericClassInfo;

public:
   // TClass status bits
   enum { kClassSaved  = BIT(12), kIgnoreTObjectStreamer = BIT(15), 
          kUnloaded    = BIT(16), kIsTObject = BIT(17),
          kIsForeign   = BIT(18), kIsEmulation = BIT(19),
          kStartWithTObject = BIT(20),  // see comments for IsStartingWithTObject()
          kWarned      = BIT(21),
          kHasNameMapNode = BIT(22)
   };
   enum ENewType { kRealNew = 0, kClassNew, kDummyNew };

private:

   mutable TObjArray *fStreamerInfo;    //Array of TVirtualStreamerInfo
   mutable std::map<std::string, TObjArray*> *fConversionStreamerInfo; //Array of the streamer infos derived from another class.
   TList             *fRealData;        //linked list for persistent members including base classes
   TList             *fBase;            //linked list for base classes
   TList             *fData;            //linked list for data members
   TList             *fMethod;          //linked list for methods
   TList             *fAllPubData;      //all public data members (including from base classes)
   TList             *fAllPubMethod;    //all public methods (including from base classes)
   mutable TList     *fClassMenuList;   //list of class menu items

   const char        *fDeclFileName;    //name of class declaration file
   const char        *fImplFileName;    //name of class implementation file
   Short_t            fDeclFileLine;    //line of class declaration
   Short_t            fImplFileLine;    //line of class implementation
   UInt_t             fInstanceCount;   //number of instances of this class
   UInt_t             fOnHeap;          //number of instances on heap
   mutable UInt_t     fCheckSum;        //checksum of data members and base classes
   TVirtualCollectionProxy *fCollectionProxy; //Collection interface
   Version_t          fClassVersion;    //Class version Identifier
   ClassInfo_t       *fClassInfo;       //pointer to CINT class info class
   TString            fContextMenuTitle;//context menu title
   const type_info   *fTypeInfo;        //pointer to the C++ type information.
   ShowMembersFunc_t  fShowMembers;     //pointer to the class's ShowMembers function
   mutable void      *fInterShowMembers;//Interpreter call setup for ShowMembers
   TClassStreamer    *fStreamer;        //pointer to streamer function
   TString            fSharedLibs;      //shared libraries containing class code

   TVirtualIsAProxy  *fIsA;             //!pointer to the class's IsA proxy.
   IsAGlobalFunc_t    fGlobalIsA;       //pointer to a global IsA function.
   mutable TMethodCall *fIsAMethod;       //!saved info to call a IsA member function

   ROOT::MergeFunc_t   fMerge;          //pointer to a function implementing Merging objects of this class.
   ROOT::ResetAfterMergeFunc_t fResetAfterMerge; //pointer to a function implementing Merging objects of this class.
   ROOT::NewFunc_t     fNew;            //pointer to a function newing one object.
   ROOT::NewArrFunc_t  fNewArray;       //pointer to a function newing an array of objects.
   ROOT::DelFunc_t     fDelete;         //pointer to a function deleting one object.
   ROOT::DelArrFunc_t  fDeleteArray;    //pointer to a function deleting an array of objects.
   ROOT::DesFunc_t     fDestructor;     //pointer to a function call an object's destructor.
   ROOT::DirAutoAdd_t  fDirAutoAdd;     //pointer which implements the Directory Auto Add feature for this class.']'
   ClassStreamerFunc_t fStreamerFunc;   //Wrapper around this class custom Streamer member function.
   Int_t               fSizeof;         //Sizeof the class.

   mutable Int_t      fCanSplit;        //!Indicates whether this class can be split or not.
   mutable Long_t     fProperty;        //!Property
   mutable Bool_t     fVersionUsed;     //!Indicates whether GetClassVersion has been called

   mutable Bool_t     fIsOffsetStreamerSet; //!saved remember if fOffsetStreamer has been set.
   mutable Long_t     fOffsetStreamer;  //!saved info to call Streamer
   Int_t              fStreamerType;    //!cached of the streaming method to use
   mutable TVirtualStreamerInfo     *fCurrentInfo;     //!cached current streamer info.
   TClassRef         *fRefStart;        //!List of references to this object
   TVirtualRefProxy  *fRefProxy;        //!Pointer to reference proxy if this class represents a reference
   ROOT::TSchemaRuleSet *fSchemaRules;  //! Schema evolution rules

   typedef void (TClass::*StreamerImpl_t)(void *obj, TBuffer &b, const TClass *onfile_class) const;
   mutable StreamerImpl_t fStreamerImpl;//! Pointer to the function implementing the right streaming behavior for the class represented by this object.

   TMethod           *GetClassMethod(Long_t faddr);
   TMethod           *GetClassMethod(const char *name, const char *signature);
   Int_t              GetBaseClassOffsetRecurse(const TClass *base);
   void Init(const char *name, Version_t cversion, const type_info *info,
             TVirtualIsAProxy *isa, ShowMembersFunc_t showmember,
             const char *dfil, const char *ifil,
             Int_t dl, Int_t il,
             Bool_t silent);
   void ForceReload (TClass* oldcl);

   void               SetClassVersion(Version_t version);
   void               SetClassSize(Int_t sizof) { fSizeof = sizof; }
   
   // Various implementation for TClass::Stramer
   void StreamerExternal(void *object, TBuffer &b, const TClass *onfile_class) const;
   void StreamerTObject(void *object, TBuffer &b, const TClass *onfile_class) const;
   void StreamerTObjectInitialized(void *object, TBuffer &b, const TClass *onfile_class) const;
   void StreamerTObjectEmulated(void *object, TBuffer &b, const TClass *onfile_class) const;
   void StreamerInstrumented(void *object, TBuffer &b, const TClass *onfile_class) const;
   void StreamerStreamerInfo(void *object, TBuffer &b, const TClass *onfile_class) const;
   void StreamerDefault(void *object, TBuffer &b, const TClass *onfile_class) const;
   
   static IdMap_t    *GetIdMap();       //Map from typeid to TClass pointer
   static ENewType    fgCallingNew;     //Intent of why/how TClass::New() is called
   static Int_t       fgClassCount;     //provides unique id for a each class
                                        //stored in TObject::fUniqueID
   // Internal status bits
   enum { kLoading = BIT(14) };
   // Internal streamer type.
   enum EStreamerType {kDefault=0, kEmulated=1, kTObject=2, kInstrumented=4, kForeign=8, kExternal=16};

   // When a new class is created, we need to be able to find
   // if there are any existing classes that have the same name
   // after any typedefs are expanded.  (This only really affects
   // template arguments.)  To avoid having to search through all classes
   // in that case, we keep a hash table mapping from the fully
   // typedef-expanded names to the original class names.
   // An entry is made in the table only if they are actually different.
   //
   // In these objects, the TObjString base holds the typedef-expanded
   // name (the hash key), and fOrigName holds the original class name
   // (the value to which the key maps).
   //
   class TNameMapNode
     : public TObjString
   {
   public:
     TNameMapNode (const char* typedf, const char* orig);
     TString fOrigName;
   };

   // These are the above-referenced hash tables.  (The pointers are null
   // if no entries have been made.)  There are actually two variants.
   // In the first, the typedef names are resolved with
   // TClassEdit::ResolveTypedef; in the second, the class names
   // are first massaged with TClassEdit::ShortType with kDropStlDefault.
   // (??? Are the two distinct tables really needed?)
   static THashTable* fgClassTypedefHash;
   static THashTable* fgClassShortTypedefHash;

private:
   TClass(const TClass& tc);
   TClass& operator=(const TClass&);   

protected:
   TVirtualStreamerInfo     *FindStreamerInfo(TObjArray* arr, UInt_t checksum) const;
   static THashTable        *GetClassShortTypedefHash();

public:
   TClass();
   TClass(const char *name, Bool_t silent = kFALSE);
   TClass(const char *name, Version_t cversion,
          const char *dfil = 0, const char *ifil = 0,
          Int_t dl = 0, Int_t il = 0, Bool_t silent = kFALSE);
   TClass(const char *name, Version_t cversion,
          const type_info &info, TVirtualIsAProxy *isa,
          ShowMembersFunc_t showmember,
          const char *dfil, const char *ifil,
          Int_t dl, Int_t il, Bool_t silent = kFALSE);
   virtual           ~TClass();

   void               AddInstance(Bool_t heap = kFALSE) { fInstanceCount++; if (heap) fOnHeap++; }
   void               AddImplFile(const char *filename, int line);
   void               AddRef(TClassRef *ref);
   static Bool_t      AddRule(const char *rule);
   static Int_t       ReadRules(const char *filename);
   static Int_t       ReadRules();
   void               AdoptSchemaRules( ROOT::TSchemaRuleSet *rules );
   virtual void       Browse(TBrowser *b);
   void               BuildRealData(void *pointer=0, Bool_t isTransient = kFALSE);
   void               BuildEmulatedRealData(const char *name, Long_t offset, TClass *cl);
   void               CalculateStreamerOffset() const;
   Bool_t             CallShowMembers(void* obj, TMemberInspector &insp,
                                      Int_t isATObject = -1) const;
   Bool_t             CanSplit() const;
   Bool_t             CanIgnoreTObjectStreamer() { return TestBit(kIgnoreTObjectStreamer);}
   TObject           *Clone(const char *newname="") const;
   void               CopyCollectionProxy(const TVirtualCollectionProxy&);
   void               Draw(Option_t *option="");
   void               Dump() const { TDictionary::Dump(); }
   void               Dump(void *obj) const;
   char              *EscapeChars(const char *text) const;
   TVirtualStreamerInfo     *FindStreamerInfo(UInt_t checksum) const;
   TVirtualStreamerInfo     *GetConversionStreamerInfo( const char* onfile_classname, Int_t version ) const;
   TVirtualStreamerInfo     *FindConversionStreamerInfo( const char* onfile_classname, UInt_t checksum ) const;
   TVirtualStreamerInfo     *GetConversionStreamerInfo( const TClass* onfile_cl, Int_t version ) const;
   TVirtualStreamerInfo     *FindConversionStreamerInfo( const TClass* onfile_cl, UInt_t checksum ) const;
   Bool_t             HasDefaultConstructor() const;
   UInt_t             GetCheckSum(UInt_t code=0) const;
   TVirtualCollectionProxy *GetCollectionProxy() const;
   TVirtualIsAProxy  *GetIsAProxy() const;
   Version_t          GetClassVersion() const { fVersionUsed = kTRUE; return fClassVersion; }
   TDataMember       *GetDataMember(const char *datamember) const;
   Long_t              GetDataMemberOffset(const char *membername) const;
   const char        *GetDeclFileName() const { return fDeclFileName; }
   Short_t            GetDeclFileLine() const { return fDeclFileLine; }
   ROOT::DelFunc_t    GetDelete() const;
   ROOT::DesFunc_t    GetDestructor() const;
   ROOT::DelArrFunc_t GetDeleteArray() const;
   ClassInfo_t       *GetClassInfo() const { return fClassInfo; }
   const char        *GetContextMenuTitle() const { return fContextMenuTitle; }
   TVirtualStreamerInfo     *GetCurrentStreamerInfo() {
      if (fCurrentInfo) return fCurrentInfo;
      else return (fCurrentInfo=(TVirtualStreamerInfo*)(fStreamerInfo->At(fClassVersion)));
   }
   TList             *GetListOfDataMembers();
   TList             *GetListOfBases();
   TList             *GetListOfMethods();
   TList             *GetListOfRealData() const { return fRealData; }
   TList             *GetListOfAllPublicMethods();
   TList             *GetListOfAllPublicDataMembers();
   const char        *GetImplFileName() const { return fImplFileName; }
   Short_t            GetImplFileLine() const { return fImplFileLine; }
   TClass            *GetActualClass(const void *object) const;
   TClass            *GetBaseClass(const char *classname);
   TClass            *GetBaseClass(const TClass *base);
   Int_t              GetBaseClassOffset(const TClass *base);
   TClass            *GetBaseDataMember(const char *datamember);
   ROOT::DirAutoAdd_t GetDirectoryAutoAdd() const;
   UInt_t             GetInstanceCount() const { return fInstanceCount; }
   UInt_t             GetHeapInstanceCount() const { return fOnHeap; }
   void               GetMenuItems(TList *listitems);
   TList             *GetMenuList() const;
   TMethod           *GetMethod(const char *method, const char *params);
   TMethod           *GetMethodWithPrototype(const char *method, const char *proto);
   TMethod           *GetMethodAny(const char *method);
   TMethod           *GetMethodAllAny(const char *method);
   Int_t              GetNdata();
   ROOT::MergeFunc_t  GetMerge() const;
   ROOT::ResetAfterMergeFunc_t  GetResetAfterMerge() const;
   ROOT::NewFunc_t    GetNew() const;
   ROOT::NewArrFunc_t GetNewArray() const;
   Int_t              GetNmethods();
   TRealData         *GetRealData(const char *name) const;
   TVirtualRefProxy  *GetReferenceProxy()  const   {  return fRefProxy; }
   const ROOT::TSchemaRuleSet *GetSchemaRules() const;
   ROOT::TSchemaRuleSet *GetSchemaRules(Bool_t create = kFALSE);
   const char        *GetSharedLibs();
   ShowMembersFunc_t  GetShowMembersWrapper() const { return fShowMembers; }
   TClassStreamer    *GetStreamer() const; 
   ClassStreamerFunc_t GetStreamerFunc() const;
   TObjArray         *GetStreamerInfos() const { return fStreamerInfo; }
   TVirtualStreamerInfo     *GetStreamerInfo(Int_t version=0) const;
   TVirtualStreamerInfo     *GetStreamerInfoAbstractEmulated(Int_t version=0) const;
   const type_info   *GetTypeInfo() const { return fTypeInfo; };
   void               IgnoreTObjectStreamer(Bool_t ignore=kTRUE);
   Bool_t             InheritsFrom(const char *cl) const;
   Bool_t             InheritsFrom(const TClass *cl) const;
   void               InterpretedShowMembers(void* obj, TMemberInspector &insp);
   Bool_t             IsFolder() const { return kTRUE; }
   Bool_t             IsLoaded() const;
   Bool_t             IsForeign() const;
   Bool_t             IsStartingWithTObject() const;
   Bool_t             IsTObject() const;
   void               ls(Option_t *opt="") const;
   void               MakeCustomMenuList();
   void               Move(void *arenaFrom, void *arenaTo) const;
   void              *New(ENewType defConstructor = kClassNew) const;
   void              *New(void *arena, ENewType defConstructor = kClassNew) const;
   void              *NewArray(Long_t nElements, ENewType defConstructor = kClassNew) const;
   void              *NewArray(Long_t nElements, void *arena, ENewType defConstructor = kClassNew) const;
   virtual void       PostLoadCheck();
   Long_t             Property() const;
   Int_t              ReadBuffer(TBuffer &b, void *pointer, Int_t version, UInt_t start, UInt_t count);
   Int_t              ReadBuffer(TBuffer &b, void *pointer);
   void               RemoveRef(TClassRef *ref); 
   void               ReplaceWith(TClass *newcl, Bool_t recurse = kTRUE) const;
   void               ResetClassInfo(Long_t tagnum);
   void               ResetInstanceCount() { fInstanceCount = fOnHeap = 0; }
   void               ResetMenuList();
   Int_t              Size() const;
   void               SetCanSplit(Int_t splitmode);
   void               SetCollectionProxy(const ROOT::TCollectionProxyInfo&);
   void               SetContextMenuTitle(const char *title);
   void               SetCurrentStreamerInfo(TVirtualStreamerInfo *info);
   void               SetGlobalIsA(IsAGlobalFunc_t);
   void               SetDeclFile(const char *name, int line) { fDeclFileName = name; fDeclFileLine = line; }
   void               SetDelete(ROOT::DelFunc_t deleteFunc);
   void               SetDeleteArray(ROOT::DelArrFunc_t deleteArrayFunc);
   void               SetDirectoryAutoAdd(ROOT::DirAutoAdd_t dirAutoAddFunc);
   void               SetDestructor(ROOT::DesFunc_t destructorFunc);
   void               SetImplFileName(const char *implFileName) { fImplFileName = implFileName; }
   void               SetMerge(ROOT::MergeFunc_t mergeFunc);
   void               SetResetAfterMerge(ROOT::ResetAfterMergeFunc_t resetFunc);
   void               SetNew(ROOT::NewFunc_t newFunc);
   void               SetNewArray(ROOT::NewArrFunc_t newArrayFunc);
   TVirtualStreamerInfo     *SetStreamerInfo(Int_t version, const char *info="");
   void               SetUnloaded();
   Int_t              WriteBuffer(TBuffer &b, void *pointer, const char *info="");

   void               AdoptReferenceProxy(TVirtualRefProxy* proxy);
   void               AdoptStreamer(TClassStreamer *strm);
   void               AdoptMemberStreamer(const char *name, TMemberStreamer *strm);
   void               SetMemberStreamer(const char *name, MemberStreamerFunc_t strm);
   void               SetStreamerFunc(ClassStreamerFunc_t strm);

   // Function to retrieve the TClass object and dictionary function
   static void           AddClass(TClass *cl);
   static void           RemoveClass(TClass *cl);
   static TClass        *GetClass(const char *name, Bool_t load = kTRUE, Bool_t silent = kFALSE);
   static TClass        *GetClass(const type_info &typeinfo, Bool_t load = kTRUE, Bool_t silent = kFALSE);
   static VoidFuncPtr_t  GetDict (const char *cname);
   static VoidFuncPtr_t  GetDict (const type_info &info);

   static Int_t       AutoBrowse(TObject *obj, TBrowser *browser);
   static ENewType    IsCallingNew();
   static TClass     *Load(TBuffer &b);
   void               Store(TBuffer &b) const;

   // Pseudo-method apply to the 'obj'. In particular those are used to
   // implement TObject like methods for non-TObject classes

   Int_t              Browse(void *obj, TBrowser *b) const;
   void               DeleteArray(void *ary, Bool_t dtorOnly = kFALSE);
   void               Destructor(void *obj, Bool_t dtorOnly = kFALSE);
   void              *DynamicCast(const TClass *base, void *obj, Bool_t up = kTRUE);
   Bool_t             IsFolder(void *obj) const;
   inline void        Streamer(void *obj, TBuffer &b, const TClass *onfile_class = 0) const
   {
      // Inline for performance, skipping one function call.
       (this->*fStreamerImpl)(obj,b,onfile_class);
   }

   ClassDef(TClass,0)  //Dictionary containing class information
};

namespace ROOT {

   #ifndef R__NO_CLASS_TEMPLATE_SPECIALIZATION
      template <typename T> struct IsPointer { enum { kVal = 0 }; };
      template <typename T> struct IsPointer<T*> { enum { kVal = 1 }; };
   #else
      template <typename T> Bool_t IsPointer(const T* /* dummy */) { return false; };
      template <typename T> Bool_t IsPointer(const T** /* dummy */) { return true; };
   #endif

   template <typename T> TClass* GetClass(      T* /* dummy */)        { return TClass::GetClass(typeid(T)); }
   template <typename T> TClass* GetClass(const T* /* dummy */)        { return TClass::GetClass(typeid(T)); }

   #ifndef R__NO_CLASS_TEMPLATE_SPECIALIZATION
      // This can only be used when the template overload resolution can distringuish between
      // T* and T**
      template <typename T> TClass* GetClass(      T**       /* dummy */) { return GetClass((T*)0); }
      template <typename T> TClass* GetClass(const T**       /* dummy */) { return GetClass((T*)0); }
      template <typename T> TClass* GetClass(      T* const* /* dummy */) { return GetClass((T*)0); }
      template <typename T> TClass* GetClass(const T* const* /* dummy */) { return GetClass((T*)0); }
   #endif

   extern TClass *CreateClass(const char *cname, Version_t id,
                              const char *dfil, const char *ifil,
                              Int_t dl, Int_t il);
}

#endif // ROOT_TClass
