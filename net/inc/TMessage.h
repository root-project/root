// @(#)root/net:$Name:  $:$Id: TMessage.h,v 1.2 2000/08/21 10:37:30 rdm Exp $
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

#ifndef ROOT_TBuffer
#include "TBuffer.h"
#endif
#ifndef ROOT_MessageTypes
#include "MessageTypes.h"
#endif


class TMessage : public TBuffer {

friend class TSocket;

private:
   UInt_t   fWhat;   //message type
   TClass  *fClass;  //if message is kMESS_OBJECT, pointer to object's class

   // TMessage objects cannot be copied or assigned
   TMessage(const TMessage &);           // not implemented
   void operator=(const TMessage &);     // not implemented

   TMessage(void *buf, Int_t bufsize);   // only called by TSocket::Recv()
   void SetLength() const;               // only called by TSocket::Send()

public:
   TMessage(UInt_t what = kMESS_ANY);
   virtual ~TMessage() { }

   void     Forward();
   TClass  *GetClass() const { return fClass; }
   void     Reset();
   void     Reset(UInt_t what) { SetWhat(what); Reset(); }
   UInt_t   What() const { return fWhat; }
   void     SetWhat(UInt_t what);

   void     WriteObject(const TObject *obj);

   ClassDef(TMessage,0)  // Message buffer class
};

#endif
