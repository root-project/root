// @(#)root/meta:$Name:  $:$Id: TClass.h,v 1.17 2001/07/05 20:17:28 brun Exp $
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

class TBaseClass;
class TBrowser;
class TDataMember;
class TMethod;
class TRealData;
class TCint;
class TBuffer;
class TStreamerInfo;
class G__ClassInfo;


class TClass : public TDictionary {

friend class TCint;

private:
   TString           fName;            //name of class
   TObjArray        *fStreamerInfo;    //Array of TStreamerInfo
   TList            *fRealData;        //linked list for persistent members including base classes
   TList            *fBase;            //linked list for base classes
   TList            *fData;            //linked list for data members
   TList            *fMethod;          //linked list for methods
   TList            *fAllPubData;      //all public data members (including from base classes)
   TList            *fAllPubMethod;    //all public methods (including from base classes)
   const char       *fDeclFileName;    //name of class declaration file
   const char       *fImplFileName;    //name of class implementation file
   Short_t           fDeclFileLine;    //line of class declaration
   Short_t           fImplFileLine;    //line of class implementation
   UInt_t            fInstanceCount;   //number of instances of this class
   UInt_t            fOnHeap;          //number of instances on heap
   UInt_t            fCheckSum;        //checksum of data members and base classes
   Version_t         fClassVersion;    //Class version Identifier
   G__ClassInfo     *fClassInfo;       //pointer to CINT class info class

   TMethod          *GetClassMethod(Long_t faddr);

   static Bool_t     fgCallingNew;     //True when TClass:New is executing
   static Int_t      fgClassCount;     //provides unique id for a each class
                                       //stored in TObject::fUniqueID
   // Internal status bits
   enum { kLoading = BIT(14) };

public:
   // TClass status bits
   enum { kClassSaved = BIT(12) , kIgnoreTObjectStreamer = BIT(13),
          kUnloaded = BIT(15) };

   TClass();
   TClass(const char *name);
   TClass(const char *name, Version_t cversion,
          const char *dfil = 0, const char *ifil = 0,
          Int_t dl = 0, Int_t il = 0);
   virtual       ~TClass();
   void           AddInstance(Bool_t heap = kFALSE) { fInstanceCount++; if (heap) fOnHeap++; }
   virtual void   Browse(TBrowser *b);
   void           BuildRealData(void *pointer=0);
   Bool_t         CanIgnoreTObjectStreamer() { return TestBit(kIgnoreTObjectStreamer);}
   Int_t          Compare(const TObject *obj) const;
   void           Draw(Option_t *option="");
   void          *DynamicCast(const TClass *base, void *obj, Bool_t up = kTRUE);
   char          *EscapeChars(char * text) const;
   UInt_t         GetCheckSum() const;
   Version_t      GetClassVersion() const { return fClassVersion; }
   TDataMember   *GetDataMember(const char *datamember);
   const char    *GetDeclFileName() const { return fDeclFileName; }
   Short_t        GetDeclFileLine() const { return fDeclFileLine; }
   G__ClassInfo  *GetClassInfo() const { return fClassInfo; }
   TList         *GetListOfDataMembers();
   TList         *GetListOfBases();
   TList         *GetListOfMethods();
   TList         *GetListOfRealData() const { return fRealData; }
   TList         *GetListOfAllPublicMethods();
   TList         *GetListOfAllPublicDataMembers();
   const char    *GetName() const { return fName.Data(); }
   const char    *GetTitle() const;
   const char    *GetImplFileName() const { return fImplFileName; }
   Short_t        GetImplFileLine() const { return fImplFileLine; }
   TClass        *GetBaseClass(const char *classname);
   TClass        *GetBaseClass(const TClass *base);
   Int_t          GetBaseClassOffset(const TClass *base);
   TClass        *GetBaseDataMember(const char *datamember);
   UInt_t         GetInstanceCount() const { return fInstanceCount; }
   UInt_t         GetHeapInstanceCount() const { return fOnHeap; }
   void           GetMenuItems(TList *listitems);
   TMethod       *GetMethod(const char *method, const char *params);
   TMethod       *GetMethodWithPrototype(const char *method, const char *proto);
   TMethod       *GetMethodAny(const char *method);
   TMethod       *GetMethodAllAny(const char *method);
   Int_t          GetNdata();
   Int_t          GetNmethods();
   TObjArray     *GetStreamerInfos() const {return fStreamerInfo;}
   TStreamerInfo *GetStreamerInfo(Int_t version=0);
   ULong_t        Hash() const { return fName.Hash(); }
   void           IgnoreTObjectStreamer(Bool_t ignore=kTRUE);
   Bool_t         InheritsFrom(const char *cl) const;
   Bool_t         InheritsFrom(const TClass *cl) const;
   Bool_t         IsFolder() const {return kTRUE;}
   Bool_t         IsLoaded() const;
   void          *New(Bool_t defConstructor = kTRUE);
   void          *New(void *arena, Bool_t defConstructor = kTRUE);
   void           Destructor(void *obj, Bool_t dtorOnly = kFALSE);
   Int_t          ReadBuffer(TBuffer &b, void *pointer, Int_t version, UInt_t start, UInt_t count);
   Int_t          ReadBuffer(TBuffer &b, void *pointer);
   void           ResetInstanceCount() { fInstanceCount = fOnHeap = 0; }
   Int_t          Size() const;
   TStreamerInfo *SetStreamerInfo(Int_t version, const char *info="");
   void           SetUnloaded();
   Long_t         Property() const;
   void           SetStreamer(const char *name, Streamer_t p);
   Int_t          WriteBuffer(TBuffer &b, void *pointer, const char *info="");

   static Int_t   AutoBrowse(TObject *obj,TBrowser *browser);
   static Bool_t  IsCallingNew();
   static TClass *Load(TBuffer &b);
   void           Store(TBuffer &b) const;

   ClassDef(TClass,0)  //Dictionary containing class information
};

extern TClass *CreateClass(const char *cname, Version_t id,
                           const char *dfil, const char *ifil,
                           Int_t dl, Int_t il);

#endif
