// @(#)root/meta:$Name:  $:$Id: TStreamerInfo.h,v 1.1 2000/11/21 21:10:30 brun Exp $
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

#ifndef ROOT_TNamed
#include "TNamed.h"
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
   Int_t            *fType;           //![fNdata]
   Int_t            *fNewType;        //![fNdata]
   Int_t            *fOffset;         //![fNdata]
   Int_t            *fLength;         //![fNdata]
   ULong_t          *fElem;           //![fNdata]
   ULong_t          *fMethod;         //![fNdata]
   TClass           *fClass;          //!pointer to class
   TObjArray        *fElements;       //Array of TStreamerElements
   
   static  Int_t     fgCount;         //Number of TStreamerInfo instances

   void              BuildUserInfo(const char *info);
   void              Compile();
           
public:

   TStreamerInfo();
   TStreamerInfo(TClass *cl, const char *info);
   virtual            ~TStreamerInfo();
   void                Build();
   void                BuildCheck();
   void                BuildOld();
   Int_t               GenerateHeaderFile(const char *dirname);
   TClass             *GetClass() {return fClass;}
   UInt_t              GetCheckSum() {return fCheckSum;}
   Int_t               GetClassVersion() {return fClassVersion;}
   Int_t               GetDataMemberOffset(TDataMember *dm, void *&streamer);
   TObjArray          *GetElements() {return fElements;}
   Int_t               GetNumber() {return fNumber;}
   void                ls(Option_t *option="");
   Int_t               ReadBuffer(TBuffer &b, char *pointer);
   Int_t               ReadBufferClones(TBuffer &b, TClonesArray *clones, Int_t nc);
   Int_t               WriteBuffer(TBuffer &b, char *pointer);
   Int_t               WriteBufferClones(TBuffer &b, TClonesArray *clones, Int_t nc);
   
   static TStreamerBasicType *GetElementCounter(const char *countName, TClass *cl, Int_t version);
   
   ClassDef(TStreamerInfo,1)  //Streamer information for one class version
};

   
#endif
