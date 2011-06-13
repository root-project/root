// @(#)root/net:$Id$
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
#include "TUrl.h"
#include "TServerSocket.h"
#include "TMonitor.h"
#include "TSystem.h"
#include "TMessage.h"
#include "Bytes.h"
#include "TROOT.h"
#include "TError.h"
#include "TVirtualMutex.h"

ClassImp(TPSocket)

//______________________________________________________________________________
TPSocket::TPSocket(TInetAddress addr, const char *service, Int_t size,
                   Int_t tcpwindowsize) : TSocket(addr, service)
{
   // Create a parallel socket. Connect to the named service at address addr.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   fSize = size;
   Init(tcpwindowsize);
}

//______________________________________________________________________________
TPSocket::TPSocket(TInetAddress addr, Int_t port, Int_t size,
                   Int_t tcpwindowsize) : TSocket(addr, port)
{
   // Create a parallel socket. Connect to the specified port # at address addr.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   fSize = size;
   Init(tcpwindowsize);
}

//______________________________________________________________________________
TPSocket::TPSocket(const char *host, const char *service, Int_t size,
                   Int_t tcpwindowsize) : TSocket(host, service)
{
   // Create a parallel socket. Connect to named service on the remote host.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   fSize = size;
   Init(tcpwindowsize);
}

//______________________________________________________________________________
TPSocket::TPSocket(const char *host, Int_t port, Int_t size,
                   Int_t tcpwindowsize)
                  : TSocket(host, port, (Int_t)(size > 1 ? -1 : tcpwindowsize))
{
   // Create a parallel socket. Connect to specified port # on the remote host.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   // To avoid uninitialization problems when Init is not called ...
   fSockets        = 0;
   fWriteMonitor   = 0;
   fReadMonitor    = 0;
   fWriteBytesLeft = 0;
   fReadBytesLeft  = 0;
   fWritePtr       = 0;
   fReadPtr        = 0;

   // set to the real value only at end (except for old servers)
   fSize           = 1;

   // to control the flow
   Bool_t valid = TSocket::IsValid();

   // check if we are called from CreateAuthSocket()
   Bool_t authreq = kFALSE;
   char *pauth = (char *)strstr(host, "?A");
   if (pauth) {
      authreq = kTRUE;
   }

   // perhaps we can use fServType here ... to be checked
   Bool_t rootdSrv = (strstr(host,"rootd")) ? kTRUE : kFALSE;

   // try authentication , if required
   if (authreq) {
      if (valid) {
         if (!Authenticate(TUrl(host).GetUser())) {
            if (rootdSrv && (fRemoteProtocol > 0 && fRemoteProtocol < 10)) {
               // We failed because we are talking to an old
               // server: we need to re-open the connection
               // and communicate the size first
               Int_t tcpw = (size > 1 ? -1 : tcpwindowsize);
               TSocket *ns = new TSocket(host, port, tcpw);
               if (ns->IsValid()) {
                  R__LOCKGUARD2(gROOTMutex);
                  gROOT->GetListOfSockets()->Remove(ns);
                  fSocket = ns->GetDescriptor();
                  fSize = size;
                  Init(tcpwindowsize);
               }
               if ((valid = IsValid())) {
                  if (!Authenticate(TUrl(host).GetUser())) {
                     TSocket::Close();
                     valid = kFALSE;
                  }
               }
            } else {
               TSocket::Close();
               valid = kFALSE;
            }
         }
      }
      // reset url to the original state
      *pauth = '\0';
      SetUrl(host);
   }

   // open the sockets ...
   if (!rootdSrv || fRemoteProtocol > 9) {
      if (valid) {
         fSize = size;
         Init(tcpwindowsize);
      }
   }
}

//______________________________________________________________________________
TPSocket::TPSocket(const char *host, Int_t port, Int_t size, TSocket *sock)
{
   // Create a parallel socket on a connection already opened via
   // TSocket sock.
   // This constructor is provided to optimize TNetFile opening when
   // instatiated via a call to TXNetFile.
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   // To avoid uninitialization problems when Init is not called ...
   fSockets        = 0;
   fWriteMonitor   = 0;
   fReadMonitor    = 0;
   fWriteBytesLeft = 0;
   fReadBytesLeft  = 0;
   fWritePtr       = 0;
   fReadPtr        = 0;

   // set to the real value only at end (except for old servers)
   fSize           = 1;

   // We need a opened connection
   if (!sock) return;

   // Now import existing socket info
   fSocket         = sock->GetDescriptor();
   fService        = sock->GetService();
   fAddress        = sock->GetInetAddress();
   fLocalAddress   = sock->GetLocalInetAddress();
   fBytesSent      = sock->GetBytesSent();
   fBytesRecv      = sock->GetBytesRecv();
   fCompress       = sock->GetCompressionSettings();
   fSecContext     = sock->GetSecContext();
   fRemoteProtocol = sock->GetRemoteProtocol();
   fServType       = (TSocket::EServiceType)sock->GetServType();
   fTcpWindowSize  = sock->GetTcpWindowSize();

   // to control the flow
   Bool_t valid = sock->IsValid();

   // check if we are called from CreateAuthSocket()
   Bool_t authreq = kFALSE;
   char *pauth = (char *)strstr(host, "?A");
   if (pauth) {
      authreq = kTRUE;
   }

   // perhaps we can use fServType here ... to be checked
   Bool_t rootdSrv = (strstr(host,"rootd")) ? kTRUE : kFALSE;

   // try authentication , if required
   if (authreq) {
      if (valid) {
         if (!Authenticate(TUrl(host).GetUser())) {
            if (rootdSrv && (fRemoteProtocol > 0 && fRemoteProtocol < 10)) {
               // We failed because we are talking to an old
               // server: we need to re-open the connection
               // and communicate the size first
               Int_t tcpw = (size > 1 ? -1 : fTcpWindowSize);
               TSocket *ns = new TSocket(host, port, tcpw);
               if (ns->IsValid()) {
                  R__LOCKGUARD2(gROOTMutex);
                  gROOT->GetListOfSockets()->Remove(ns);
                  fSocket = ns->GetDescriptor();
                  fSize = size;
                  Init(fTcpWindowSize);
               }
               if ((valid = IsValid())) {
                  if (!Authenticate(TUrl(host).GetUser())) {
                     TSocket::Close();
                     valid = kFALSE;
                  }
               }
            } else {
               TSocket::Close();
               valid = kFALSE;
            }
         }
      }
      // reset url to the original state
      *pauth = '\0';
      SetUrl(host);
   }

   // open the sockets ...
   if (!rootdSrv || fRemoteProtocol > 9) {
      if (valid) {
         fSize = size;
         Init(fTcpWindowSize, sock);
      }
   }

   // Add to the list if everything OK
   if (IsValid()) {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Add(this);
   }
}

//______________________________________________________________________________
TPSocket::TPSocket(TSocket *pSockets[], Int_t size)
{
   // Create a parallel socket. This ctor is called by TPServerSocket.

   fSockets = pSockets;
   fSize    = size;

   // set descriptor if simple socket (needed when created
   // by TPServerSocket)
   if (fSize <= 1)
      fSocket = fSockets[0]->GetDescriptor();

   // set socket options (no blocking and no delay)
   SetOption(kNoDelay, 1);
   if (fSize > 1)
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

   SetName(fSockets[0]->GetName());
   SetTitle(fSockets[0]->GetTitle());
   fAddress = fSockets[0]->GetInetAddress();

   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Add(this);
   }
}

//______________________________________________________________________________
TPSocket::~TPSocket()
{
   // Cleanup the parallel socket.

   Close();

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


   if (!IsValid()) {
      // if closing happens too early (e.g. timeout) the underlying
      // socket may still be open
      TSocket::Close(option);
      return;
   }

   if (fSize <= 1) {
      TSocket::Close(option);
   } else {
      for (int i = 0; i < fSize; i++) {
         fSockets[i]->Close(option);
         delete fSockets[i];
      }
   }
   delete [] fSockets;
   fSockets = 0;

   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Remove(this);
   }
}

//______________________________________________________________________________
void TPSocket::Init(Int_t tcpwindowsize, TSocket *sock)
{
   // Create a parallel socket to the specified host.

   fSockets        = 0;
   fWriteMonitor   = 0;
   fReadMonitor    = 0;
   fWriteBytesLeft = 0;
   fReadBytesLeft  = 0;
   fWritePtr       = 0;
   fReadPtr        = 0;

   if ((sock && !sock->IsValid()) || !TSocket::IsValid())
      return;

   Int_t i = 0;

   if (fSize <= 1) {
      // check if single mode
      fSize = 1;

      // set socket options (no delay)
      if (sock)
         sock->SetOption(kNoDelay, 1);
      else
         TSocket::SetOption(kNoDelay, 1);

      // if yes, communicate this to server
      // (size = 0 for backward compatibility)
      if (sock) {
         if (sock->Send((Int_t)0, (Int_t)0) < 0)
            Warning("Init", "%p: problems sending (0,0)", sock);
      } else {
         if (TSocket::Send((Int_t)0, (Int_t)0) < 0)
            Warning("Init", "problems sending (0,0)");
      }

      // needs to fill additional private members
      fSockets = new TSocket*[1];
      fSockets[0]= (TSocket *)this;

   } else {

      // create server that will be used to accept the parallel sockets from
      // the remote host, use port=0 to scan for a free port
      TServerSocket ss(0, kFALSE, fSize, tcpwindowsize);

      // send the local port number of the just created server socket and the
      // number of desired parallel sockets
      if (sock) {
         if (sock->Send(ss.GetLocalPort(), fSize) < 0)
            Warning("Init", "%p: problems sending size", sock);
      } else {
         if (TSocket::Send(ss.GetLocalPort(), fSize) < 0)
            Warning("Init", "problems sending size");
      }

      fSockets = new TSocket*[fSize];

      // establish fSize parallel socket connections between client and server
      for (i = 0; i < fSize; i++) {
         fSockets[i] = ss.Accept();
         R__LOCKGUARD2(gROOTMutex);
         gROOT->GetListOfSockets()->Remove(fSockets[i]);
      }

      // set socket options (no blocking and no delay)
      SetOption(kNoDelay, 1);
      SetOption(kNoBlock, 1);

      // close original socket
      if (sock)
         sock->Close();
      else
         gSystem->CloseConnection(fSocket, kFALSE);
      fSocket = -1;
   }

   fWriteMonitor   = new TMonitor;
   fReadMonitor    = new TMonitor;
   fWriteBytesLeft = new Int_t[fSize];
   fReadBytesLeft  = new Int_t[fSize];
   fWritePtr       = new char*[fSize];
   fReadPtr        = new char*[fSize];

   for (i = 0; i < fSize; i++) {
      fWriteMonitor->Add(fSockets[i], TMonitor::kWrite);
      fReadMonitor->Add(fSockets[i], TMonitor::kRead);
   }
   fWriteMonitor->DeActivateAll();
   fReadMonitor->DeActivateAll();
}

//______________________________________________________________________________
TInetAddress TPSocket::GetLocalInetAddress()
{
   // Return internet address of local host to which the socket is bound.
   // In case of error TInetAddress::IsValid() returns kFALSE.

   if (fSize<= 1)
      return TSocket::GetLocalInetAddress();

   if (IsValid()) {
      if (fLocalAddress.GetPort() == -1)
         fLocalAddress = gSystem->GetSockName(fSockets[0]->GetDescriptor());
      return fLocalAddress;
   }
   return TInetAddress();
}

//______________________________________________________________________________
Int_t TPSocket::GetDescriptor() const
{
   // Return socket descriptor

   if (fSize <= 1)
      return TSocket::GetDescriptor();

   return fSockets ? fSockets[0]->GetDescriptor() : -1;

}

//______________________________________________________________________________
Int_t TPSocket::Send(const TMessage &mess)
{
   // Send a TMessage object. Returns the number of bytes in the TMessage
   // that were sent and -1 in case of error. In case the TMessage::What
   // has been or'ed with kMESS_ACK, the call will only return after having
   // received an acknowledgement, making the sending process synchronous.
   // Returns -4 in case of kNoBlock and errno == EWOULDBLOCK.

   if (!fSockets || fSize <= 1)
      return TSocket::Send(mess);  // only the case when called via Init()

   if (!IsValid()) {
      return -1;
   }

   if (mess.IsReading()) {
      Error("Send", "cannot send a message used for reading");
      return -1;
   }

   // send streamer infos in case schema evolution is enabled in the TMessage
   SendStreamerInfos(mess);

   // send the process id's so TRefs work
   SendProcessIDs(mess);

   mess.SetLength();   //write length in first word of buffer

   if (GetCompressionLevel() > 0 && mess.GetCompressionLevel() == 0)
      const_cast<TMessage&>(mess).SetCompressionSettings(fCompress);

   if (mess.GetCompressionLevel() > 0)
      const_cast<TMessage&>(mess).Compress();

   char *mbuf = mess.Buffer();
   Int_t mlen = mess.Length();
   if (mess.CompBuffer()) {
      mbuf = mess.CompBuffer();
      mlen = mess.CompLength();
   }

   Int_t nsent, ulen = (Int_t) sizeof(UInt_t);
   // send length
   if ((nsent = SendRaw(mbuf, ulen, kDefault)) <= 0)
      return nsent;

   // send buffer (this might go in parallel)
   if ((nsent = SendRaw(mbuf+ulen, mlen-ulen, kDefault)) <= 0)
      return nsent;

   // if acknowledgement is desired, wait for it
   if (mess.What() & kMESS_ACK) {
      char buf[2];
      if (RecvRaw(buf, sizeof(buf), kDefault) < 0)
         return -1;
      if (strncmp(buf, "ok", 2)) {
         Error("Send", "bad acknowledgement");
         return -1;
      }
   }

   return nsent;  //length - length header
}

//______________________________________________________________________________
Int_t TPSocket::SendRaw(const void *buffer, Int_t length, ESendRecvOptions opt)
{
   // Send a raw buffer of specified length. Returns the number of bytes
   // send and -1 in case of error.

   if (fSize == 1)
      return TSocket::SendRaw(buffer,length,opt);

   if (!fSockets) return -1;

   // if data buffer size < 4K use only one socket
   Int_t i, nsocks = fSize, len = length;
   if (len < 4096)
      nsocks = 1;

   ESendRecvOptions sendopt = kDontBlock;
   if (nsocks == 1)
      sendopt = kDefault;

   if (opt != kDefault) {
      nsocks  = 1;
      sendopt = opt;
   }

   if (nsocks == 1)
      fSockets[0]->SetOption(kNoBlock, 0);
   else
      fSockets[0]->SetOption(kNoBlock, 1);

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
again:
               ResetBit(TSocket::kBrokenConn);
               if ((nsent = fSockets[is]->SendRaw(fWritePtr[is],
                                                  fWriteBytesLeft[is],
                                                  sendopt)) <= 0) {
                  if (nsent == -4) {
                     // got EAGAIN/EWOULDBLOCK error, keep trying...
                     goto again;
                  }
                  fWriteMonitor->DeActivateAll();
                  if (nsent == -5) {
                     // connection reset by peer or broken ...
                     SetBit(TSocket::kBrokenConn);
                     Close();
                  }
                  return -1;
               }
               if (opt == kDontBlock) {
                  fWriteMonitor->DeActivateAll();
                  return nsent;
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

   if (fSize <= 1)
      return TSocket::Recv(mess);

   if (!IsValid()) {
      mess = 0;
      return -1;
   }

oncemore:
   Int_t  n;
   UInt_t len;
   if ((n = RecvRaw(&len, sizeof(UInt_t), kDefault)) <= 0) {
      mess = 0;
      return n;
   }
   len = net2host(len);  //from network to host byte order

   char *buf = new char[len+sizeof(UInt_t)];
   if ((n = RecvRaw(buf+sizeof(UInt_t), len, kDefault)) <= 0) {
      delete [] buf;
      mess = 0;
      return n;
   }

   mess = new TMessage(buf, len+sizeof(UInt_t));

   // receive any streamer infos
   if (RecvStreamerInfos(mess))
      goto oncemore;

   // receive any process ids
   if (RecvProcessIDs(mess))
      goto oncemore;

   if (mess->What() & kMESS_ACK) {
      char ok[2] = { 'o', 'k' };
      if (SendRaw(ok, sizeof(ok), kDefault) < 0) {
         delete mess;
         mess = 0;
         return -1;
      }
      mess->SetWhat(mess->What() & ~kMESS_ACK);
   }

   return n;
}

//______________________________________________________________________________
Int_t TPSocket::RecvRaw(void *buffer, Int_t length, ESendRecvOptions opt)
{
   // Send a raw buffer of specified length. Returns the number of bytes
   // sent or -1 in case of error.

   if (fSize <= 1)
      return TSocket::RecvRaw(buffer,length,opt);

   if (!fSockets) return -1;

   // if data buffer size < 4K use only one socket
   Int_t i, nsocks = fSize, len = length;
   if (len < 4096)
      nsocks = 1;

   ESendRecvOptions recvopt = kDontBlock;
   if (nsocks == 1)
      recvopt = kDefault;

   if (opt != kDefault) {
      nsocks  = 1;
      recvopt = opt;
   }

   if (nsocks == 1)
      fSockets[0]->SetOption(kNoBlock, 0);
   else
      fSockets[0]->SetOption(kNoBlock, 1);

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
               ResetBit(TSocket::kBrokenConn);
               if ((nrecv = fSockets[is]->RecvRaw(fReadPtr[is],
                                                  fReadBytesLeft[is],
                                                  recvopt)) <= 0) {
                  fReadMonitor->DeActivateAll();
                  if (nrecv == -5) {
                     // connection reset by peer or broken ...
                     SetBit(TSocket::kBrokenConn);
                     Close();
                  }
                  return -1;
               }
               if (opt == kDontBlock) {
                  fReadMonitor->DeActivateAll();
                  return nrecv;
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

   if (fSize <= 1)
      return TSocket::SetOption(opt,val);

   Int_t ret = 0;
   for (int i = 0; i < fSize; i++)
      ret = fSockets[i]->SetOption(opt, val);
   return ret;
}

//______________________________________________________________________________
Int_t TPSocket::GetOption(ESockOptions opt, Int_t &val)
{
   // Get socket options. Returns -1 in case of error.

   if (fSize <= 1)
      return TSocket::GetOption(opt,val);

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

   if (fSize <= 1)
      return TSocket::GetErrorCode();

   return fSockets[0] ? fSockets[0]->GetErrorCode() : 0;
}
