// @(#)root/meta:$Name:  $:$Id: TStreamerInfo.h,v 1.31 2001/11/22 15:05:21 brun Exp $
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

class TStreamerInfo : public TNamed {

private:

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
   Bool_t            fOptimized;      //! true if has been optimized
   TClass           *fClass;          //!pointer to class
   TObjArray        *fElements;       //Array of TStreamerElements

   static  Int_t     fgCount;         //Number of TStreamerInfo instances
   static  Bool_t    fgCanDelete;     //True if ReadBuffer can delete object
   static  Bool_t    fgOptimize;      //True if optimization on
   static TStreamerElement *fgElement; //Pointer to current TStreamerElement
   void              BuildUserInfo(const char *info);

public:

   //status bits
   enum { kCannotOptimize = BIT(12)};

   enum EReadWrite {
      kBase   =  0,  kOffsetL = 20,  kOffsetP = 40,  kCounter =  6,  kCharStar=7,
      kChar   =  1,  kShort   =  2,  kInt     =  3,  kLong    =  4,  kFloat =  5, kDouble  =  8,
      kUChar  = 11,  kUShort  = 12,  kUInt    = 13,  kULong   = 14,  kBits  = 15,
      kObject = 61,  kAny     = 62,  kObjectp = 63,  kObjectP = 64,
      kTString= 65,  kTObject = 66,  kTNamed  = 67,  kMissing = 99999,
      kSkip   =100,  kSkipL   =120,  kSkipP   =140,
      kConv   =200,  kConvL   =220,  kConvP   =240,  kStreamer=500,  kStreamLoop=501
   };

   TStreamerInfo();
   TStreamerInfo(TClass *cl, const char *info);
   virtual            ~TStreamerInfo();
   void                Build();
   void                BuildCheck();
   void                BuildFake(TFile *file);
   void                BuildOld();
   void                Compile();
   void                ComputeSize();
   void                ForceWriteInfo(TFile *file);
   Int_t               GenerateHeaderFile(const char *dirname);
   TClass             *GetClass() const {return fClass;}
   UInt_t              GetCheckSum() const {return fCheckSum;}
   Int_t               GetClassVersion() const {return fClassVersion;}
   Int_t               GetDataMemberOffset(TDataMember *dm, Streamer_t &streamer) const;
   TObjArray          *GetElements() const {return fElements;}
   ULong_t            *GetElems()   const {return fElem;}
   Int_t               GetNdata()   const {return fNdata;}
   Int_t               GetNumber()  const {return fNumber;}
   Int_t              *GetLengths() const {return fLength;}
   ULong_t            *GetMethods() const {return fMethod;}
   Int_t              *GetOffsets() const {return fOffset;}
   Int_t               GetSize()    const;
   TStreamerElement   *GetStreamerElement(const char*datamember, Int_t& offset) const;
   Int_t              *GetTypes()   const {return fType;}
   Double_t            GetValue(char *pointer, Int_t i, Int_t j, Int_t len) const;
   Double_t            GetValueClones(TClonesArray *clones, Int_t i, Int_t j, Int_t k, Int_t eoffset) const;
   Bool_t              IsOptimized() const {return fOptimized;}
   void                ls(Option_t *option="") const;
   Int_t               New(const char *p);
   void                PrintValue(const char *name, char *pointer, Int_t i, Int_t len) const;
   void                PrintValueClones(const char *name, TClonesArray *clones, Int_t i, Int_t eoffset) const;
   Int_t               ReadBuffer(TBuffer &b, char *pointer, Int_t first);
   Int_t               ReadBufferClones(TBuffer &b, TClonesArray *clones, Int_t nc, Int_t first, Int_t eoffset);
   void                SetClass(TClass *cl) {fClass = cl;}
   void                SetClassVersion(Int_t vers) {fClassVersion=vers;}
   void                TagFile(TFile *fFile);
   Int_t               WriteBuffer(TBuffer &b, char *pointer, Int_t first);
   Int_t               WriteBufferClones(TBuffer &b, TClonesArray *clones, Int_t nc, Int_t first, Int_t eoffset);

   static TStreamerElement   *GetCurrentElement();
   static TStreamerBasicType *GetElementCounter(const char *countName, TClass *cl, Int_t version);
   static Bool_t       CanOptimize();
   static void         Optimize(Bool_t opt=kTRUE);
   static Bool_t       CanDelete();
   static void         SetCanDelete(Bool_t opt=kTRUE);

   ClassDef(TStreamerInfo,2)  //Streamer information for one class version
};


#endif
