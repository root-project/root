// @(#)root/net:$Id$
// Author: Fons Rademakers   19/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMessage
#define ROOT_TMessage


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMessage                                                             //
//                                                                      //
// Message buffer class used for serializing objects and sending them   //
// over the network.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TBufferFile
#include "TBufferFile.h"
#endif
#ifndef ROOT_MessageTypes
#include "MessageTypes.h"
#endif
#ifndef ROOT_TBits
#include "TBits.h"
#endif

class TList;
class TVirtualStreamerInfo;

class TMessage : public TBufferFile {

friend class TAuthenticate;
friend class TSocket;
friend class TPSocket;
friend class TXSocket;

private:
   TList   *fInfos;       //Array of TStreamerInfo used in WriteObject
   TBits    fBitsPIDs;    //Array of bits to mark the TProcessIDs uids written to the message
   UInt_t   fWhat;        //Message type
   TClass  *fClass;       //If message is kMESS_OBJECT pointer to object's class
   Int_t    fCompress;    //Compression level and algorithm
   char    *fBufComp;     //Compressed buffer
   char    *fBufCompCur;  //Current position in compressed buffer
   char    *fCompPos;     //Position of fBufCur when message was compressed
   Bool_t   fEvolution;   //True if support for schema evolution required

   static Bool_t fgEvolution;  //True if global support for schema evolution required

   // TMessage objects cannot be copied or assigned
   TMessage(const TMessage &);           // not implemented
   void operator=(const TMessage &);     // not implemented

   // used by friend TSocket
   Bool_t TestBitNumber(UInt_t bitnumber) const { return fBitsPIDs.TestBitNumber(bitnumber); }

protected:
   TMessage(void *buf, Int_t bufsize);   // only called by T(P)Socket::Recv()
   void SetLength() const;               // only called by T(P)Socket::Send()

public:
   TMessage(UInt_t what = kMESS_ANY, Int_t bufsiz = TBuffer::kInitialSize);
   virtual ~TMessage();

   void     ForceWriteInfo(TVirtualStreamerInfo *info, Bool_t force);
   void     Forward();
   TClass  *GetClass() const { return fClass;}
   void     TagStreamerInfo(TVirtualStreamerInfo* info);
   void     Reset();
   void     Reset(UInt_t what) { SetWhat(what); Reset(); }
   UInt_t   What() const { return fWhat; }
   void     SetWhat(UInt_t what);

   void     EnableSchemaEvolution(Bool_t enable = kTRUE) { fEvolution = enable; }
   Bool_t   UsesSchemaEvolution() const { return fEvolution; }
   Int_t    GetCompressionAlgorithm() const;
   Int_t    GetCompressionLevel() const;
   Int_t    GetCompressionSettings() const;
   void     SetCompressionAlgorithm(Int_t algorithm=0);
   void     SetCompressionLevel(Int_t level=1);
   void     SetCompressionSettings(Int_t settings=1);
   Int_t    Compress();
   Int_t    Uncompress();
   char    *CompBuffer() const { return fBufComp; }
   Int_t    CompLength() const { return (Int_t)(fBufCompCur - fBufComp); }
   void     WriteObject(const TObject *obj);
   UShort_t WriteProcessID(TProcessID *pid);

   static void   EnableSchemaEvolutionForAll(Bool_t enable = kTRUE);
   static Bool_t UsesSchemaEvolutionForAll();

   ClassDef(TMessage,0)  // Message buffer class
};

//______________________________________________________________________________
inline Int_t TMessage::GetCompressionAlgorithm() const
{
   return (fCompress < 0) ? -1 : fCompress / 100;
}

//______________________________________________________________________________
inline Int_t TMessage::GetCompressionLevel() const
{
   return (fCompress < 0) ? -1 : fCompress % 100;
}

//______________________________________________________________________________
inline Int_t TMessage::GetCompressionSettings() const
{
   return (fCompress < 0) ? -1 : fCompress;
}

#endif
