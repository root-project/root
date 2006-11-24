// @(#)root/meta:$Name:  $:$Id: TClass.h,v 1.68 2006/10/05 17:10:09 pcanal Exp $
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
#ifndef ROOT_TROOT
#include "TROOT.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#ifndef ROOT_TStreamerInfo
// This include is no longer necessary but is kept to be 
// backward compatible with user code.
#include "TStreamerInfo.h"
#endif

#include <map>

class TBaseClass;
class TBrowser;
class TDataMember;
class TClassRef;
class TMethod;
class TRealData;
class TCint;
class TBuffer;
namespace Cint {
class G__ClassInfo;
}
using namespace Cint;
class TStreamerInfo;
class TVirtualCollectionProxy;
class TMethodCall;
class TVirtualIsAProxy;
class TVirtualRefProxy;

namespace ROOT { class TGenericClassInfo; }

class TClass : public TDictionary {

friend class TCint;
friend void ROOT::ResetClassVersion(TClass*, const char*, Short_t);
friend class ROOT::TGenericClassInfo;

public:
   // TClass status bits
   enum { kClassSaved  = BIT(12), kIgnoreTObjectStreamer = BIT(15), 
          kUnloaded    = BIT(16), kIsTObject = BIT(17),
          kIsForeign   = BIT(18), kIsEmulation = BIT(19),
          kStartWithTObject = BIT(20),  // see comments for IsStartingWithTObject()
          kWarned      = BIT(21)
   };
   enum ENewType { kRealNew = 0, kClassNew, kDummyNew };

private:
   TObjArray         *fStreamerInfo;    //Array of TStreamerInfo
   TList             *fRealData;        //linked list for persistent members including base classes
   TList             *fBase;            //linked list for base classes
   TList             *fData;            //linked list for data members
   TList             *fMethod;          //linked list for methods
   TList             *fAllPubData;      //all public data members (including from base classes)
   TList             *fAllPubMethod;    //all public methods (including from base classes)
   const char        *fDeclFileName;    //name of class declaration file
   const char        *fImplFileName;    //name of class implementation file
   Short_t            fDeclFileLine;    //line of class declaration
   Short_t            fImplFileLine;    //line of class implementation
   UInt_t             fInstanceCount;   //number of instances of this class
   UInt_t             fOnHeap;          //number of instances on heap
   UInt_t             fCheckSum;        //checksum of data members and base classes
   TVirtualCollectionProxy *fCollectionProxy; //Collection interface
   Version_t          fClassVersion;    //Class version Identifier
   G__ClassInfo      *fClassInfo;       //pointer to CINT class info class
   TString            fContextMenuTitle;//context menu title
   TList             *fClassMenuList;   //list of class menu items
   const type_info   *fTypeInfo;        //pointer to the C++ type information.
   ShowMembersFunc_t  fShowMembers;     //pointer to the class's ShowMembers function
   TClassStreamer    *fStreamer;        //pointer to streamer function
   TString            fSharedLibs;      //shared libraries containing class code

   TVirtualIsAProxy  *fIsA;             //!pointer to the class's IsA proxy.
   IsAGlobalFunc_t    fGlobalIsA;       //pointer to a global IsA function.
   TMethodCall       *fIsAMethod;       //!saved info to call a IsA member function

   ROOT::NewFunc_t    fNew;             //pointer to a function newing one object.
   ROOT::NewArrFunc_t fNewArray;        //pointer to a function newing an array of objects.
   ROOT::DelFunc_t    fDelete;          //pointer to a function deleting one object.
   ROOT::DelArrFunc_t fDeleteArray;     //pointer to a function deleting an array of objects.
   ROOT::DesFunc_t    fDestructor;      //pointer to a function call an object's destructor.
   Int_t              fSizeof;          //Sizeof the class.

   Bool_t             fVersionUsed;     //!Indicates whether GetClassVersion has been called
   Long_t             fProperty;        //!Property

   void              *fInterStreamer;   //!saved info to call Streamer
   Long_t             fOffsetStreamer;  //!saved info to call Streamer
   Int_t              fStreamerType;    //!cached of the streaming method to use
   TStreamerInfo     *fCurrentInfo;     //!cached current streamer info.
   TClassRef         *fRefStart;        //!List of references to this object
   TVirtualRefProxy  *fRefProxy;        //!Pointer to reference proxy if this class represents a reference
   TMethod           *GetClassMethod(Long_t faddr);
   TMethod           *GetClassMethod(const char *name, const char *signature);
   Int_t              GetBaseClassOffsetRecurse(const TClass *base);
   TStreamerInfo     *GetCurrentStreamerInfo() {
      if (fCurrentInfo) return fCurrentInfo;
      else return (fCurrentInfo=(TStreamerInfo*)(fStreamerInfo->At(fClassVersion)));
   }
   void Init(const char *name, Version_t cversion, const type_info *info,
             TVirtualIsAProxy *isa, ShowMembersFunc_t showmember,
             const char *dfil, const char *ifil,
             Int_t dl, Int_t il);

   void               SetClassVersion(Version_t version) { fClassVersion = version; fCurrentInfo = 0; }
   void               SetClassSize(Int_t sizof) { fSizeof = sizof; }

   static ENewType    fgCallingNew;     //Intent of why/how TClass::New() is called
   static Int_t       fgClassCount;     //provides unique id for a each class
                                        //stored in TObject::fUniqueID
#if 0
   // FIXME: Turn off for now, trouble when ptr is reallocated to
   //        some different type and we don't know.
#endif
   static std::multimap<void*, Version_t> fgObjectVersionRepository; // Record what version of a class was used to construct an object
//#endif

   // Internal status bits
   enum { kLoading = BIT(14) };
   // Internal streamer type.
   enum {kDefault=0, kEmulated=1, kTObject=2, kInstrumented=4, kForeign=8, kExternal=16};

protected:
   TClass(const TClass& tc);
   TClass& operator=(const TClass&);   

public:
   TClass();
   TClass(const char *name);
   TClass(const char *name, Version_t cversion,
          const char *dfil = 0, const char *ifil = 0,
          Int_t dl = 0, Int_t il = 0);
   TClass(const char *name, Version_t cversion,
          const type_info &info, TVirtualIsAProxy *isa,
          ShowMembersFunc_t showmember,
          const char *dfil, const char *ifil,
          Int_t dl, Int_t il);
   virtual           ~TClass();

   void               AddInstance(Bool_t heap = kFALSE) { fInstanceCount++; if (heap) fOnHeap++; }
   void               AddImplFile(const char *filename, int line);
   void               AddRef(TClassRef *ref); 
   virtual void       Browse(TBrowser *b);
   void               BuildRealData(void *pointer=0);
   void               BuildEmulatedRealData(const char *name, Int_t offset, TClass *cl);
   Bool_t             CanSplit() const;
   Bool_t             CanIgnoreTObjectStreamer() { return TestBit(kIgnoreTObjectStreamer);}
   void               CopyCollectionProxy(const TVirtualCollectionProxy&);
   void               Draw(Option_t *option="");
   void               Dump() const { TDictionary::Dump(); }
   void               Dump(void *obj) const;
   char              *EscapeChars(const char *text) const;
   TStreamerInfo     *FindStreamerInfo(UInt_t checksum) const;
   Bool_t             HasDefaultConstructor() const;
   UInt_t             GetCheckSum(UInt_t code=0) const;
   TVirtualCollectionProxy *GetCollectionProxy() const;
   TVirtualIsAProxy  *GetIsAProxy() const;
   Version_t          GetClassVersion() const { ((TClass*)this)->fVersionUsed = kTRUE; return (fClassVersion>=0) ? fClassVersion : (-fClassVersion); }
   TDataMember       *GetDataMember(const char *datamember) const;
   Int_t              GetDataMemberOffset(const char *membername) const;
   const char        *GetDeclFileName() const { return fDeclFileName; }
   Short_t            GetDeclFileLine() const { return fDeclFileLine; }
   ROOT::DelFunc_t    GetDelete() const;
   ROOT::DesFunc_t    GetDestructor() const;
   ROOT::DelArrFunc_t GetDeleteArray() const;
   G__ClassInfo      *GetClassInfo() const { return fClassInfo; }
   const char        *GetContextMenuTitle() const { return fContextMenuTitle; }
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
   UInt_t             GetInstanceCount() const { return fInstanceCount; }
   UInt_t             GetHeapInstanceCount() const { return fOnHeap; }
   void               GetMenuItems(TList *listitems);
   TList             *GetMenuList() const { return fClassMenuList; }
   TMethod           *GetMethod(const char *method, const char *params);
   TMethod           *GetMethodWithPrototype(const char *method, const char *proto);
   TMethod           *GetMethodAny(const char *method);
   TMethod           *GetMethodAllAny(const char *method);
   Int_t              GetNdata();
   ROOT::NewFunc_t    GetNew() const;
   ROOT::NewArrFunc_t GetNewArray() const;
   Int_t              GetNmethods();
   TRealData         *GetRealData(const char *name) const;
   TVirtualRefProxy  *GetReferenceProxy()  const   {  return fRefProxy; }
   const char        *GetSharedLibs();
   ShowMembersFunc_t  GetShowMembersWrapper() const { return fShowMembers; }
   TClassStreamer    *GetStreamer() const; 
   TObjArray         *GetStreamerInfos() const { return fStreamerInfo; }
   TStreamerInfo     *GetStreamerInfo(Int_t version=0);
   const type_info   *GetTypeInfo() const { return fTypeInfo; };
   void               IgnoreTObjectStreamer(Bool_t ignore=kTRUE);
   Bool_t             InheritsFrom(const char *cl) const;
   Bool_t             InheritsFrom(const TClass *cl) const;
   Bool_t             IsFolder() const { return kTRUE; }
   Bool_t             IsLoaded() const;
   Bool_t             IsForeign() const;
   Bool_t             IsStartingWithTObject() const;
   Bool_t             IsTObject() const;
   void               MakeCustomMenuList();
   void              *New(ENewType defConstructor = kClassNew);
   void              *New(void *arena, ENewType defConstructor = kClassNew);
   void              *NewArray(Long_t nElements, ENewType defConstructor = kClassNew);
   void              *NewArray(Long_t nElements, void *arena, ENewType defConstructor = kClassNew);
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
   void               SetContextMenuTitle(const char *title);
   void               SetGlobalIsA(IsAGlobalFunc_t);
   void               SetDeclFile(const char *name, int line) { fDeclFileName = name; fDeclFileLine = line; }
   void               SetDelete(ROOT::DelFunc_t deleteFunc);
   void               SetDeleteArray(ROOT::DelArrFunc_t deleteArrayFunc);
   void               SetDestructor(ROOT::DesFunc_t destructorFunc);
   void               SetImplFileName(const char *implFileName) { fImplFileName = implFileName; }
   void               SetNew(ROOT::NewFunc_t newFunc);
   void               SetNewArray(ROOT::NewArrFunc_t newArrayFunc);
   TStreamerInfo     *SetStreamerInfo(Int_t version, const char *info="");
   void               SetUnloaded();
   Int_t              WriteBuffer(TBuffer &b, void *pointer, const char *info="");

   void               AdoptReferenceProxy(TVirtualRefProxy* proxy);
   void               AdoptStreamer(TClassStreamer *strm);
   void               AdoptMemberStreamer(const char *name, TMemberStreamer *strm);
   void               SetMemberStreamer(const char *name, MemberStreamerFunc_t strm);

#if 0
   // FIXME: Turn off for now, trouble when ptr is reallocated to
   //        some different type and we don't know.
#endif
   static std::multimap<void*, Version_t>& GetObjectVersionRepository() { return fgObjectVersionRepository; }
//#endif

   // Function to retrieve the TClass object and dictionary function
   static TClass        *GetClass(const char *name, Bool_t load = kTRUE);
   static TClass        *GetClass(const type_info &typeinfo, Bool_t load = kTRUE);
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
   void               Streamer(void *obj, TBuffer &b);

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

   template <typename T> TClass* GetClass(      T* /* dummy */)        { return GetROOT()->GetClass(typeid(T)); }
   template <typename T> TClass* GetClass(const T* /* dummy */)        { return GetROOT()->GetClass(typeid(T)); }

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
