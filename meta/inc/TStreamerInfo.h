// @(#)root/meta:$Name:  $:$Id: TStreamerInfo.h,v 1.47 2004/01/27 19:52:47 brun Exp $
// Author: Rene Brun   12/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStreamerInfo
#define ROOT_TStreamerInfo


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerInfo                                                        //
//                                                                      //
// Describe Streamer information for one class version                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TClass
#include "TClass.h"
#endif
#ifndef ROOT_TClonesArray
#include "TClonesArray.h"
#endif

class TStreamerElement;
class TStreamerBasicType;
class TVirtualCollectionProxy;

class TStreamerInfo : public TNamed {

private:
   
   class CompInfo {
   // Class used to cache information (see fComp)
   public: 
      TClass          *fClass; 
      TString          fClassName;
      TMemberStreamer *fStreamer;
      CompInfo() : fClass(0), fClassName(""), fStreamer(0) {};
      ~CompInfo() {};
      void Update(const TClass *oldcl, TClass *newcl) {
         if (fClass==oldcl) fClass=newcl;
         else if (fClass==0) fClass =gROOT->GetClass(fClassName);
      }
   };

   UInt_t            fCheckSum;       //checksum of original class
   Int_t             fClassVersion;   //Class version identifier
   Int_t             fNumber;         //!Unique identifier
   Int_t             fNdata;          //!number of optmized types
   Int_t             fSize;           //!size of the persistent class
   Int_t            *fType;           //![fNdata]
   Int_t            *fNewType;        //![fNdata]
   Int_t            *fOffset;         //![fNdata]
   Int_t            *fLength;         //![fNdata]
   ULong_t          *fElem;           //![fNdata]
   ULong_t          *fMethod;         //![fNdata]
   CompInfo         *fComp;           //![fNdata] additional info
   Bool_t            fOptimized;      //! true if has been optimized
   TClass           *fClass;          //!pointer to class
   TObjArray        *fElements;       //Array of TStreamerElements
   Version_t         fOldVersion;     //! Version of the TStreamerInfo object read from the file
   Bool_t            fIsBuilt;        //! true if the TStreamerInfo has been 'built'

   static  Int_t     fgCount;         //Number of TStreamerInfo instances
   static  Bool_t    fgCanDelete;     //True if ReadBuffer can delete object
   static  Bool_t    fgOptimize;      //True if optimization on
   static TStreamerElement *fgElement; //Pointer to current TStreamerElement
   void              BuildUserInfo(const char *info);
   static Double_t   GetValueAux(Int_t type, void *ladd, int k);
   static void       PrintValueAux(char *ladd, Int_t atype, TStreamerElement * aElement, Int_t aleng, Int_t *count);
//VP   Int_t             ReadBufferAux       (TBuffer &b, char **arr, Int_t first,Int_t narr,Int_t eoffset,Int_t mode);
//VP   Int_t             WriteBufferClonesAux      (TBuffer &b, char **arr, Int_t first,Int_t narr,Int_t eoffset,Int_t mode);
//VP   Int_t             ReadBufferClonesAux       (TBuffer &b, char **arr, Int_t first,Int_t narr,Int_t eoffset,Int_t mode);
public:

   //status bits
   enum { kCannotOptimize = BIT(12),kRecovered=BIT(13)};

   enum EReadWrite {
      kBase     =  0,  kOffsetL = 20,  kOffsetP = 40,  kCounter =  6,  kCharStar = 7,
      kChar     =  1,  kShort   =  2,  kInt     =  3,  kLong    =  4,  kFloat    = 5,
      kDouble   =  8,  kDouble32=  9,
      kUChar    = 11,  kUShort  = 12,  kUInt    = 13,  kULong   = 14,  kBits     = 15,
      kLong64   = 16,  kULong64 = 17,
      kObject   = 61,  kAny     = 62,  kObjectp = 63,  kObjectP = 64,  kTString  = 65,
      kTObject  = 66,  kTNamed  = 67,  kAnyp    = 68,  kAnyP    = 69,  kAnyPnoVT = 70,
      kSTLp     = 71,
      kSkip     = 100, kSkipL = 120, kSkipP   = 140,
      kConv     = 200, kConvL = 220, kConvP   = 240,
      kSTL      = 300, kSTLstring = 365,
      kStreamer = 500, kStreamLoop = 501,
      kMissing  = 99999
   };

//	Some comments about EReadWrite
//	kBase    : base class element
//	kOffsetL : fixed size array 
//	kOffsetP : pointer to object 
//	kCounter : counter for array size 
//	kCharStar: pointer to array of char 
//	kBits    : TObject::fBits in case of a referenced object
//	kObject  : Class  derived from TObject
//	kObjectp : Class* derived from TObject and with    comment field //->Class
//	kObjectP : Class* derived from TObject and with NO comment field //->Class
//	kAny     : Class  not derived from TObject
//	kAnyp    : Class* not derived from TObject with    comment field //->Class
//	kAnyP    : Class* not derived from TObject with NO comment field //->Class
// kAnyPnoVT: Class* not derived from TObject with NO comment field //->Class and Class has NO virtual table
// kSTLp    : Pointer to STL container.
//	kTString	: TString, special case
//	kTObject	: TObject, special case
//	kTNamed  : TNamed , special case



   TStreamerInfo();
   TStreamerInfo(TClass *cl, const char *info);
   virtual            ~TStreamerInfo();
   void                Build();
   void                BuildCheck();
   void                BuildEmulated(TFile *file);
   void                BuildOld();
   void                Compile();
   void                ComputeSize();
   void                ForceWriteInfo(TFile *file, Bool_t force=kFALSE);
   Int_t               GenerateHeaderFile(const char *dirname);
   TClass             *GetClass() const {return fClass;}
   UInt_t              GetCheckSum() const {return fCheckSum;}
   Int_t               GetClassVersion() const {return fClassVersion;}
   Int_t               GetDataMemberOffset(TDataMember *dm, TMemberStreamer *&streamer) const;
   TObjArray          *GetElements() const {return fElements;}
   ULong_t            *GetElems()   const {return fElem;}
   Int_t               GetNdata()   const {return fNdata;}
   Int_t               GetNumber()  const {return fNumber;}
   Int_t              *GetLengths() const {return fLength;}
   ULong_t            *GetMethods() const {return fMethod;}
   Int_t              *GetNewTypes() const {return fNewType;}
   Int_t               GetOffset(const char *) const;
   Int_t              *GetOffsets() const {return fOffset;}
   Int_t               GetSize()    const;
   Int_t               GetSizeElements()    const;
   TStreamerElement   *GetStreamerElement(const char*datamember, Int_t& offset) const;
   Int_t              *GetTypes()   const {return fType;}
   Double_t            GetValue(char *pointer, Int_t i, Int_t j, Int_t len) const;
   Double_t            GetValueClones(TClonesArray *clones, Int_t i, Int_t j, Int_t k, Int_t eoffset) const;
   Double_t            GetValueSTL(TVirtualCollectionProxy *cont, Int_t i, Int_t j, Int_t k, Int_t eoffset) const;
   Bool_t              IsBuilt() const { return fIsBuilt; }
   Bool_t              IsOptimized() const {return fOptimized;}
   Int_t               IsRecovered() const {return TestBit(kRecovered);}
   void                ls(Option_t *option="") const;
   Int_t               New(const char *p);
   void                PrintValue(const char *name, char *pointer, Int_t i, Int_t len, Int_t lenmax=1000) const;
   void                PrintValueClones(const char *name, TClonesArray *clones, Int_t i, Int_t eoffset, Int_t lenmax=1000) const;
   void                PrintValueSTL(const char *name, TVirtualCollectionProxy *cont, Int_t i, Int_t eoffset, Int_t lenmax=1000) const;
   Int_t               ReadBuffer(TBuffer &b, void *pointer, Int_t first,Int_t narr=1,Int_t eoffset=0,Int_t mode=0);
   Int_t               ReadBufferSkip(TBuffer &b, char **arr, Int_t i,Int_t kase, TStreamerElement *aElement, Int_t narr, Int_t eoffset);
   Int_t               ReadBufferConv(TBuffer &b, char **arr, Int_t i,Int_t kase, TStreamerElement *aElement, Int_t narr, Int_t eoffset);
   Int_t               ReadBufferClones(TBuffer &b, TClonesArray *clones, Int_t nc, Int_t first, Int_t eoffset);
   Int_t               ReadBufferSTL(TBuffer &b, TVirtualCollectionProxy *cont, Int_t nc, Int_t first, Int_t eoffset);
   void                SetCheckSum(UInt_t checksum) {fCheckSum = checksum;}
   void                SetClass(TClass *cl) {fClass = cl;}
   void                SetClassVersion(Int_t vers) {fClassVersion=vers;}
   void                TagFile(TFile *fFile);
   Int_t               WriteBuffer(TBuffer &b, char *pointer, Int_t first);
   Int_t               WriteBufferClones(TBuffer &b, TClonesArray *clones, Int_t nc, Int_t first, Int_t eoffset);
   Int_t               WriteBufferSTL   (TBuffer &b, TVirtualCollectionProxy *cont,   Int_t nc, Int_t first, Int_t eoffset);
   virtual void        Update(const TClass *oldClass, TClass *newClass);

   static TStreamerElement   *GetCurrentElement();
   static TStreamerBasicType *GetElementCounter(const char *countName, TClass *cl);
   static Bool_t       CanOptimize();
   static void         Optimize(Bool_t opt=kTRUE);
   static Bool_t       CanDelete();
   static void         SetCanDelete(Bool_t opt=kTRUE);

   Int_t             WriteBufferAux      (TBuffer &b, char **arr, Int_t first,Int_t narr,Int_t eoffset,Int_t mode);

   ClassDef(TStreamerInfo,3)  //Streamer information for one class version
};


#endif
