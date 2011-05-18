// @(#)root/net:$Id$
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


class TPSocket : public TSocket {

friend class TPServerSocket;

private:
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
   void Init(Int_t tcpwindowsize, TSocket *sock = 0);
   Option_t *GetOption() const { return TObject::GetOption(); }

public:
   TPSocket(TInetAddress address, const char *service, Int_t size,
            Int_t tcpwindowsize = -1);
   TPSocket(TInetAddress address, Int_t port, Int_t size,
            Int_t tcpwindowsize = -1);
   TPSocket(const char *host, const char *service, Int_t size,
            Int_t tcpwindowsize = -1);
   TPSocket(const char *host, Int_t port, Int_t size, Int_t tcpwindowsize = -1);
   TPSocket(const char *host, Int_t port, Int_t size, TSocket *sock);
   virtual ~TPSocket();

   void          Close(Option_t *opt="");
   Int_t         GetDescriptor() const;
   TInetAddress  GetLocalInetAddress();

   Int_t   Send(const TMessage &mess);
   Int_t   Send(Int_t kind) { return TSocket::Send(kind); }
   Int_t   Send(Int_t status, Int_t kind) { return TSocket::Send(status, kind); }
   Int_t   Send(const char *mess, Int_t kind = kMESS_STRING) { return TSocket::Send(mess, kind); }
   Int_t   SendRaw(const void *buffer, Int_t length, ESendRecvOptions opt);
   Int_t   Recv(TMessage *&mess);
   Int_t   Recv(Int_t &status, Int_t &kind) { return TSocket::Recv(status, kind); }
   Int_t   Recv(char *mess, Int_t max) { return TSocket::Recv(mess, max); }
   Int_t   Recv(char *mess, Int_t max, Int_t &kind) { return TSocket::Recv(mess, max, kind); }
   Int_t   RecvRaw(void *buffer, Int_t length, ESendRecvOptions opt);

   Bool_t  IsValid() const { return fSockets ? kTRUE : kFALSE; }
   Int_t   GetErrorCode() const;
   Int_t   SetOption(ESockOptions opt, Int_t val);
   Int_t   GetOption(ESockOptions opt, Int_t &val);
   Int_t   GetSize() const { return fSize; }

   ClassDef(TPSocket,0)  // Parallel client socket
};

#endif
