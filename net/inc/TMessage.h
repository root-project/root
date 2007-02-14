// @(#)root/net:$Name:  $:$Id: TMessage.h,v 1.9 2005/12/09 15:12:19 rdm Exp $
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


class TMessage : public TBufferFile {

friend class TAuthenticate;
friend class TSocket;
friend class TPSocket;
friend class TXSocket;

private:
   UInt_t   fWhat;        //Message type
   TClass  *fClass;       //If message is kMESS_OBJECT pointer to object's class
   Int_t    fCompress;    //Compression level from 0 (not compressed) to 9 (max compression)
   char    *fBufComp;     //Compressed buffer
   char    *fBufCompCur;  //Current position in compressed buffer
   char    *fCompPos;     //Position of fBufCur when message was compressed

   // TMessage objects cannot be copied or assigned
   TMessage(const TMessage &);           // not implemented
   void operator=(const TMessage &);     // not implemented

protected:
   TMessage(void *buf, Int_t bufsize);   // only called by T(P)Socket::Recv()
   void SetLength() const;               // only called by T(P)Socket::Send()

public:
   TMessage(UInt_t what = kMESS_ANY);
   virtual ~TMessage();

   void     Forward();
   TClass  *GetClass() const { return fClass; }
   void     Reset();
   void     Reset(UInt_t what) { SetWhat(what); Reset(); }
   UInt_t   What() const { return fWhat; }
   void     SetWhat(UInt_t what);

   void     SetCompressionLevel(Int_t level = 1);
   Int_t    GetCompressionLevel() const { return fCompress; }
   Int_t    Compress();
   Int_t    Uncompress();
   char    *CompBuffer() const { return fBufComp; }
   Int_t    CompLength() const { return (Int_t)(fBufCompCur - fBufComp); }

   ClassDef(TMessage,0)  // Message buffer class
};

#endif
