// @(#)root/net:$Name:  $:$Id: TPSocket.h,v 1.1 2001/01/26 16:55:07 rdm Exp $
// Author: Fons Rademakers   20/1/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPSocket
#define ROOT_TPSocket


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPSocket                                                             //
//                                                                      //
// This class implements parallel client sockets. A parallel socket is  //
// an endpoint for communication between two machines. It is parallel   //
// because several TSockets are open at the same time to the same       //
// destination. This especially speeds up communication over Big Fat    //
// Pipes (i.e. high bandwidth, high latency WAN connections).           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSocket
#include "TSocket.h"
#endif

class TMonitor;
class TMessage;


class TPSocket : public TNamed {

friend class TPServerSocket;

private:
   TSocket    *fSetupSocket;     // initial setup socket
   TSocket   **fSockets;         // array of parallel sockets
   TMonitor   *fWriteMonitor;    // monitor write on parallel sockets
   TMonitor   *fReadMonitor;     // monitor read from parallel sockets
   Int_t       fSize;            // number of parallel sockets
   Int_t      *fWriteBytesLeft;  // bytes left to write for specified socket
   Int_t      *fReadBytesLeft;   // bytes left to read for specified socket
   char      **fWritePtr;        // pointer to write buffer for specified socket
   char      **fReadPtr;         // pointer to read buffer for specified socket

   TPSocket(TSocket *pSockets[], Int_t size);
   TPSocket(const TPSocket &);        // not implemented
   void operator=(const TPSocket &);  // idem
   void Init(Int_t tcpwindowsize);
   Option_t *GetOption() const { return TObject::GetOption(); }

public:
   TPSocket(TInetAddress address, const char *service, Int_t size, Int_t tcpwindowsize = -1);
   TPSocket(TInetAddress address, Int_t port, Int_t size, Int_t tcpwindowsize = -1);
   TPSocket(const char *host, const char *service, Int_t size, Int_t tcpwindowsize = -1);
   TPSocket(const char *host, Int_t port, Int_t size, Int_t tcpwindowsize = -1);
   virtual ~TPSocket();

   virtual void Close(Option_t *opt="");

   virtual Int_t Send(const TMessage &mess);
   virtual Int_t SendRaw(const void *buffer, Int_t length);
   virtual Int_t Recv(TMessage *&mess);
   virtual Int_t RecvRaw(void *buffer, Int_t length);

   Bool_t                IsValid() const { return fSockets == 0 ? kFALSE : kTRUE; }
   Int_t                 GetErrorCode() const;
   virtual Int_t         SetOption(ESockOptions opt, Int_t val);
   virtual Int_t         GetOption(ESockOptions opt, Int_t &val);
   Int_t                 GetSize() const { return fSize; }

   ClassDef(TPSocket,0)  // Parallel client socket
};

#endif
