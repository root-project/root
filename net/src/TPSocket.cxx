// @(#)root/net:$Name:$:$Id:$
// Author: Fons Rademakers   22/1/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TPSocket.h"
#include "TServerSocket.h"
#include "TMonitor.h"
#include "TSystem.h"
#include "TMessage.h"
#include "Bytes.h"
#include "TROOT.h"
#include "TError.h"


ClassImp(TPSocket)

//______________________________________________________________________________
TPSocket::TPSocket(TInetAddress addr, const char *service, Int_t size,
                   Int_t tcpwindowsize) : TNamed(addr.GetHostName(), service)
{
   // Create a parallel socket. Connect to the named service at address addr.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   fSetupSocket = new TSocket(addr, service);
   fSize = size;
   Init(tcpwindowsize);
}

//______________________________________________________________________________
TPSocket::TPSocket(TInetAddress addr, Int_t port, Int_t size,
                   Int_t tcpwindowsize) : TNamed(addr.GetHostName(), "")
{
   // Create a parallel socket. Connect to the specified port # at address addr.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   fSetupSocket = new TSocket(addr, port);
   fSize = size;
   Init(tcpwindowsize);
}

//______________________________________________________________________________
TPSocket::TPSocket(const char *host, const char *service, Int_t size,
                   Int_t tcpwindowsize) : TNamed(host, service)
{
   // Create a parallel socket. Connect to named service on the remote host.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   fSetupSocket = new TSocket(host, service);
   fSize = size;
   Init(tcpwindowsize);
}

//______________________________________________________________________________
TPSocket::TPSocket(const char *host, Int_t port, Int_t size,
                   Int_t tcpwindowsize) : TNamed(host, "")
{
   // Create a parallel socket. Connect to specified port # on the remote host.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   fSetupSocket = new TSocket(host, port);
   fSize = size;
   Init(tcpwindowsize);
}

//______________________________________________________________________________
TPSocket::TPSocket(TSocket *pSockets[], Int_t size)
{
   // Create a parallel socket. This ctor is called by TPServerSocket.

   fSockets = pSockets;
   fSize    = size;

   // set socket options (no blocking and no delay)
   SetOption(kNoDelay, 1);
   SetOption(kNoBlock, 1);

   fSetupSocket    = 0;
   fWriteMonitor   = new TMonitor;
   fReadMonitor    = new TMonitor;
   fWriteBytesLeft = new Int_t[fSize];
   fReadBytesLeft  = new Int_t[fSize];
   fWritePtr       = new char*[fSize];
   fReadPtr        = new char*[fSize];

   for (int i = 0; i < fSize; i++) {
      fWriteMonitor->Add(fSockets[i], TMonitor::kWrite);
      fReadMonitor->Add(fSockets[i], TMonitor::kRead);
   }
   fWriteMonitor->DeActivateAll();
   fReadMonitor->DeActivateAll();

   SetName(fSockets[0]->GetName());
   gROOT->GetListOfSockets()->Add(this);
}

//______________________________________________________________________________
TPSocket::~TPSocket()
{
   // Cleanup the parallel socket.

   Close();

   delete fSetupSocket;
   delete fWriteMonitor;
   delete fReadMonitor;
   delete [] fWriteBytesLeft;
   delete [] fReadBytesLeft;
   delete [] fWritePtr;
   delete [] fReadPtr;
}

//______________________________________________________________________________
void TPSocket::Close(Option_t *option)
{
   // Close a parallel socket. If option is "force", calls shutdown(id,2) to
   // shut down the connection. This will close the connection also
   // for the parent of this process. Also called via the dtor (without
   // option "force", call explicitely Close("force") if this is desired).

   for (int i = 0; i < fSize; i++) {
      fSockets[i]->Close(option);
      delete fSockets[i];
   }
   delete [] fSockets;
   fSockets = 0;
}

//______________________________________________________________________________
void TPSocket::Init(Int_t tcpwindowsize)
{
   // Create a parallel socket to the specified host.

   fSockets        = 0;
   fWriteMonitor   = 0;
   fReadMonitor    = 0;
   fWriteBytesLeft = 0;
   fReadBytesLeft  = 0;
   fWritePtr       = 0;
   fReadPtr        = 0;

   if (!fSetupSocket->IsValid())
      return;

   // create server that will be used to accept the parallel sockets from
   // the remote host, use port=0 to scan for a free port
   TServerSocket ss(0, kFALSE, fSize, tcpwindowsize);

   // send the local port number of the just created server socket and the
   // number of desired parallel sockets
   fSetupSocket->Send(ss.GetLocalPort(), fSize);

   fSockets = new TSocket*[fSize];

   // establish fSize parallel socket connections between client and server
   for (int i = 0; i < fSize; i++) {
      fSockets[i] = ss.Accept();
      gROOT->GetListOfSockets()->Remove(fSockets[i]);
   }

   // set socket options (no blocking and no delay)
   SetOption(kNoDelay, 1);
   SetOption(kNoBlock, 1);

   fWriteMonitor   = new TMonitor;
   fReadMonitor    = new TMonitor;
   fWriteBytesLeft = new Int_t[fSize];
   fReadBytesLeft  = new Int_t[fSize];
   fWritePtr       = new char*[fSize];
   fReadPtr        = new char*[fSize];

   for (int i = 0; i < fSize; i++) {
      fWriteMonitor->Add(fSockets[i], TMonitor::kWrite);
      fReadMonitor->Add(fSockets[i], TMonitor::kRead);
   }
   fWriteMonitor->DeActivateAll();
   fReadMonitor->DeActivateAll();

   delete fSetupSocket;
   fSetupSocket = 0;

   gROOT->GetListOfSockets()->Add(this);
}

//______________________________________________________________________________
Int_t TPSocket::Send(const TMessage &mess)
{
   // Send a TMessage object. Returns the number of bytes in the TMessage
   // that were sent and -1 in case of error. A kMESS_ACK which has been
   // or'ed with a TMessage::What will be ignored.

   return 0;
}

//______________________________________________________________________________
Int_t TPSocket::SendRaw(const void *buffer, Int_t length)
{
   // Send a raw buffer of specified length. Returns the number of bytes
   // send and -1 in case of error.

   if (!fSockets) return -1;

   // if data buffer size < 4K use only one socket
   Int_t i, nsocks = fSize, len = length;
   if (len < 4096)
      nsocks = 1;

   ESendRecvOptions sendopt = kDontBlock;
   if (nsocks == 1)
      sendopt = kDefault;

   // setup pointer appropriately for transferring data equally on the
   // parallel sockets
   for (i = 0; i < nsocks; i++) {
      fWriteBytesLeft[i] = len/nsocks;
      fWritePtr[i] = (char *)buffer + (i*fWriteBytesLeft[i]);
      fWriteMonitor->Activate(fSockets[i]);
   }
   fWriteBytesLeft[nsocks-1] += len%nsocks;

   // send the data on the parallel sockets
   while (len > 0) {
      TSocket *s = fWriteMonitor->Select();
      for (int is = 0; is < nsocks; is++) {
         if (s == fSockets[is]) {
            if (fWriteBytesLeft[is] > 0) {
               Int_t nsent;
               if ((nsent = fSockets[is]->SendRaw(fWritePtr[is],
                                                  fWriteBytesLeft[is],
                                                  sendopt)) < 0) {
                  fWriteMonitor->DeActivateAll();
                  return -1;
               }
               fWriteBytesLeft[is] -= nsent;
               fWritePtr[is] += nsent;
               len -= nsent;
            }
         }
      }
   }
   fWriteMonitor->DeActivateAll();

   return length;
}

//______________________________________________________________________________
Int_t TPSocket::Recv(TMessage *&mess)
{
   // Receive a TMessage object. The user must delete the TMessage object.
   // Returns length of message in bytes (can be 0 if other side of connection
   // is closed) or -1 in case of error or -4 in case a non-blocking socket would
   // block (i.e. there is nothing to be read). In those case mess == 0.

   if (!fSockets) {
      mess = 0;
      return -1;
   }


   return 0;
}

//______________________________________________________________________________
Int_t TPSocket::RecvRaw(void *buffer, Int_t length)
{
   // Send a raw buffer of specified length. Returns the number of bytes
   // sent or -1 in case of error.

   if (!fSockets) return -1;

   // if data buffer size < 4K use only one socket
   Int_t i, nsocks = fSize, len = length;
   if (len < 4096)
      nsocks = 1;

   ESendRecvOptions recvopt = kDontBlock;
   if (nsocks == 1)
      recvopt = kDefault;

   // setup pointer appropriately for transferring data equally on the
   // parallel sockets
   for (i = 0; i < nsocks; i++) {
      fReadBytesLeft[i] = len/nsocks;
      fReadPtr[i] = (char *)buffer + (i*fReadBytesLeft[i]);
      fReadMonitor->Activate(fSockets[i]);
   }
   fReadBytesLeft[nsocks-1] += len%nsocks;

   // start receiving data on all sockets. Receive data as and when
   // they are available on a socket by by using select.
   // Exit the loop as soon as all data has been received.
   while (len > 0) {
      TSocket *s = fReadMonitor->Select();
      for (int is = 0; is < nsocks; is++) {
         if (s == fSockets[is]) {
            if (fReadBytesLeft[is] > 0) {
               Int_t nrecv;
               if ((nrecv = fSockets[is]->RecvRaw(fReadPtr[is],
                                                  fReadBytesLeft[is],
                                                  recvopt)) <= 0) {
                  fReadMonitor->DeActivateAll();
                  return -1;
               }
               fReadBytesLeft[is] -= nrecv;
               fReadPtr[is] += nrecv;
               len -= nrecv;
            }
         }
      }
   }
   fReadMonitor->DeActivateAll();

   return length;
}

//______________________________________________________________________________
Int_t TPSocket::SetOption(ESockOptions opt, Int_t val)
{
   // Set socket options.

   Int_t ret = 0;
   for (int i = 0; i < fSize; i++)
      ret = fSockets[i]->SetOption(opt, val);
   return ret;
}

//______________________________________________________________________________
Int_t TPSocket::GetOption(ESockOptions opt, Int_t &val)
{
   // Get socket options. Returns -1 in case of error.

   Int_t ret = 0;
   for (int i = 0; i < fSize; i++)
      ret = fSockets[i]->GetOption(opt, val);
   return ret;
}

//______________________________________________________________________________
Int_t TPSocket::GetErrorCode() const
{
   // Returns error code. Meaning depends on context where it is called.
   // If no error condition returns 0 else a value < 0.

   return fSockets[0] ? fSockets[0]->GetErrorCode() : 0;
}
