// @(#)root/netx:$Name:  $:$Id: TXPhyConnection.cxx,v 1.5 2005/01/05 01:20:11 rdm Exp $
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXPhyConnection                                                      //
//                                                                      //
// Class handling physical connections to xrootd servers.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXPhyConnection.h"
#include "TError.h"
#include "TXDebug.h"
#include "TXMessage.h"
#include "TString.h"
#include "TEnv.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TThread.h"
#include "TApplication.h"
#include "Riostream.h"


ClassImp(TXPhyConnection);

//____________________________________________________________________________
TThread::VoidFunc_t SocketReaderThread(void * arg)
{
   // This thread is the base for the async capabilities of TXPhyConnection
   // It repeatedly keeps reading from the socket, while feeding the
   // MsqQ with a stream of TXMessages containing what's happening
   // at the socket level

   TXPhyConnection *thisObj;

   if (DebugLevel() >= kHIDEBUG)
      Info("SocketReaderThread", "Reader Thread starting");

   // It should be possible to cancel the thread as soon as the
   // cancellation request is received.
   TThread::SetCancelAsynchronous();
   TThread::SetCancelOn();

   if (DebugLevel() >= kHIDEBUG)
      Info("SocketReaderThread", "Reader Thread started");

   thisObj = (TXPhyConnection *)arg;

   // Set running state ...
   thisObj->ReaderStarted();

   while (!(thisObj->ReaderThreadKilled())) {
      thisObj->BuildXMessage(kDefault, kTRUE, kTRUE);
      if (!thisObj->ReaderThreadKilled())
         thisObj->CheckAutoTerm();
   }

   if (DebugLevel() >= kHIDEBUG)
      Info("SocketReaderThread","Reader Thread exiting");
   return 0;
}

//____________________________________________________________________________
TXPhyConnection::TXPhyConnection(TXAbsUnsolicitedMsgHandler *h)
{
   // Constructor

   // Initialization of lock mutex

   if (!(fRwMutex = new TMutex(kTRUE)))
      Error("TXPhyConnection",
            "can't create mutex for read/write: out of system resources");

   if (!(fMutex = new TMutex(kTRUE)))
      Error("TXPhyConnection",
            "can't create mutex for local locks: out of system resources");

   Touch();

   fServer = kUnknown;
   SetLogged(kNo);
   fRequestTimeout = gEnv->GetValue("XNet.RequestTimeout",
                                    DFLT_REQUESTTIMEOUT);
   UnsolicitedMsgHandler = h;

   fReaderthreadhandler = 0;
   fReaderthreadrunning = kFALSE;
   fReaderthreadkilled = kFALSE;

   fReaderCV = new TCondition();
}

//____________________________________________________________________________
TXPhyConnection::~TXPhyConnection()
{
   // Destructor

   Disconnect();

   SafeDelete(fRwMutex);
   SafeDelete(fReaderCV);
}

//____________________________________________________________________________
Bool_t TXPhyConnection::Connect(TString TcpAddress, Int_t TcpPort,
                                Int_t TcpWindowSize)
{
   // Connect to remote server
   R__LOCKGUARD(fMutex);

   if (DebugLevel() >= kHIDEBUG)
      Info("Connect", "Connecting to [%s:%d]", TcpAddress.Data(), TcpPort);

   fSocket = new TXSocket(TcpAddress, TcpPort, TcpWindowSize);

   if(!fSocket) {
      Error("Connect","Fatal ERROR *** Object creation with new failed !"
                      " Probable system resources exhausted.");
      gSystem->Abort();
   }

   fSocket->TryConnect();

   if (!fSocket->IsValid()) {
      Error("Connect",
            "can't open connection to xrootd/rootd on host [%s:%d]",
            TcpAddress.Data(), TcpPort);
      Disconnect();
      return kFALSE;
   }
   fSocket->SetOption(kNoDelay, 1);

   Touch();

   fTTLsec = DATA_TTL;

   if (DebugLevel() >= kHIDEBUG)
      Info("Connect", "Connected to host [%s:%d].",
	   TcpAddress.Data(), TcpPort);

   fRemoteAddress = TcpAddress;
   fRemotePort = TcpPort;
   fReaderthreadrunning = kFALSE;

   return kTRUE;
}

//____________________________________________________________________________
void TXPhyConnection::StartReader()
{
   // Start reader thread

   bool running;
   {
      R__LOCKGUARD(fMutex);
      running = fReaderthreadrunning;
   }

   // Parametric asynchronous stuff.
   // If we are going Sync, then nothing has to be done,
   // otherwise the reader thread must be started
   if ( !running &&
         gEnv->GetValue("XNet.GoAsynchronous", DFLT_GOASYNC) ) {

      if (DebugLevel() >= kHIDEBUG)
         Info("StartReader", "Starting reader thread...");

      // Now we launch  the reader thread
      fReaderthreadhandler =
         new TThread((TThread::VoidFunc_t) SocketReaderThread, this);

      if (!fReaderthreadhandler)
         Info("StartReader",
              "Can't create reader thread: out of system resources");
      else {
         // Start the thread
         fReaderthreadhandler->Run();
         // Make sure that it is really running
         do {
            {
               R__LOCKGUARD(fMutex);
               running = fReaderthreadrunning;
            }
            if (!running) {
               if (DebugLevel() >= kHIDEBUG)
                  Info("StartReader","Waiting a little bit ...");
               fReaderCV->TimedWaitRelative(100);
            }
         } while (!running);
      }
   }
}

//____________________________________________________________________________
void TXPhyConnection::ReaderStarted()
{
   // Called inside SocketReaderThread to flag the running status
   // of the thread

   R__LOCKGUARD(fMutex);
   fReaderthreadrunning = kTRUE;
   fReaderCV->Signal();
}

//____________________________________________________________________________
Bool_t TXPhyConnection::ReConnect(TString TcpAddress, Int_t TcpPort,
                                  Int_t TcpWindowSize)
{
   // Re-connection attempt

   Disconnect();
   return Connect(TcpAddress, TcpPort, TcpWindowSize);
}

//____________________________________________________________________________
TSocket *TXPhyConnection::SaveSocket()
{
   // Return copy of the TSocket part of the existing socket
   // Used to save an open connection to rootd daemons

   TSocket *opensock = 0;
   if (fSocket) {

      if (fReaderthreadrunning) {
         fReaderthreadkilled = kTRUE;
         fReaderthreadhandler->Kill();
      }

      // Extract TSocket part of TXSocket
      opensock = fSocket->ExtractSocket();

      // Signal deactivation
      fTTLsec = 0;
   }
   return opensock;
}

//____________________________________________________________________________
void TXPhyConnection::Disconnect()
{
   // Terminate connection

   if (fReaderthreadrunning) {
      fReaderthreadkilled = kTRUE;
      fReaderthreadhandler->Kill();
   }

   fSocket = 0;
}

//____________________________________________________________________________
void TXPhyConnection::Touch()
{
   // Set last-use-time to present time

   R__LOCKGUARD(fMutex);

   time_t t = time(0);
   if (DebugLevel() >= kDUMPDEBUG)
      Info("Touch", "Setting 'fLastUseTimestamp' to current time: %d",t);

   fLastUseTimestamp = t;
}

//____________________________________________________________________________
Int_t TXPhyConnection::ReadRaw(void *buf, Int_t len, ESendRecvOptions opt)
{
   // Receive 'len' bytes from the connected server and store them in 'buf'.
   // Return number of bytes received.

   Int_t res;

   Touch();

   if (IsValid()) {

      if (DebugLevel() >= kDUMPDEBUG)
         Info("ReadRaw", "Reading from socket: %d [%s:%d]",
               fSocket->GetDescriptor(), fRemoteAddress.Data(), fRemotePort);

      res = fSocket->RecvRaw(buf, len, opt);

      if ((res <= 0) && (res != TXSOCK_ERR_TIMEOUT) && fReaderthreadkilled &&
          (DebugLevel() >= kHIDEBUG)) {
         Info("ReadRaw", "Reader thread has been killed");
         return res;
      }

      if ((res <= 0) && (res != TXSOCK_ERR_TIMEOUT) &&
          (DebugLevel() >= kHIDEBUG) && (gSystem->GetErrno()) )
         Info("ReadRaw", "Read error [%s:%d]. Errno %d:'%s'.",
               fRemoteAddress.Data(), fRemotePort, gSystem->GetErrno(),
               gSystem->GetError() );

      // If a socket error comes, then we disconnect (and destroy the fSocket)
      // but we have not to disconnect in the case of a timeout
      if (((res <= 0) && (res != TXSOCK_ERR_TIMEOUT)) ||
          (!fSocket->IsValid())) {

         if (DebugLevel() >= kHIDEBUG)
            Info("ReadRaw",
                 "Socket reported a disconnection (server[%s:%d]). Closing.",
                 fRemoteAddress.Data(), fRemotePort);
         Disconnect();
      }

      Touch();

      return res;
   }
   else {
      // Socket already destroyed or disconnected
      if (DebugLevel() >= kDUMPDEBUG)
         Info("ReadRaw", "Socket is disconnected (server [%s:%d])",
              fRemoteAddress.Data(), fRemotePort);
      return TXSOCK_ERR;
   }
}

//____________________________________________________________________________
TXMessage *TXPhyConnection::ReadXMessage(Int_t streamid)
{
   // Gets a full loaded TXMessage from this phyconn.
   // May be a pure msg pick from a queue

   Touch();
   return fMsgQ.GetMsg(streamid, fRequestTimeout );

}

//____________________________________________________________________________
TXMessage *TXPhyConnection::BuildXMessage(ESendRecvOptions opt,
                                          Bool_t IgnoreTimeouts, Bool_t Enqueue)
{
   // Builds an TXMessage, and makes it read its header/data from the socket
   // Also put automatically the msg into the queue

   TXMessage *m;

   m = new TXMessage();
   if (!m) {
      Error("BuildXMessage", "Fatal ERROR *** Object creation with new failed !"
                             " Probable system resources exhausted.");
      gSystem->Abort();
   }
   m->ReadRaw(this, opt);

   if (fReaderthreadkilled) {
      if (m) delete m;
      return (TXMessage *)0;
   }

   if (m->IsAttn()) {

      // Here we insert the PhyConn-level support for unsolicited responses
      // Some of them will be propagated in some way to the upper levels
      //  TLogConn, TConnMgr, TXNetConn
      HandleUnsolicited(m);

      // The purpose of this message ends here
      delete m;
      m = 0;
   }
   else
      if (Enqueue) {
         // If we have to ignore the socket timeouts, then we have not to
         // feed the queue with them. In this case, the newly created TXMessage
         // has to be freed.
         bool waserror;
         if (IgnoreTimeouts) {

            if (m->GetStatusCode() != TXMessage::kXMSC_timeout) {
               waserror = m->IsError();
               fMsgQ.PutMsg(m);
               if (waserror)
                  for (int kk=0; kk < 10; kk++)
                      fMsgQ.PutMsg(0);
            } else {
               delete m;
               m = 0;
            }
         } else
            fMsgQ.PutMsg(m);
      }

   return m;
}

//____________________________________________________________________________
void TXPhyConnection::CheckAutoTerm()
{
   // Check if auto-termination is needed

   Bool_t doexit = kFALSE;
   {
      R__LOCKGUARD(fMutex);

      // Parametric asynchronous stuff
      // If we are going async, we might be willing to term ourself
      if (!IsValid() && gEnv->GetValue("XNet.GoAsynchronous", DFLT_GOASYNC)) {
         if (TThread::SelfId() == fReaderthreadhandler->GetId()) {
            // Notify termination, if requested
            if (DebugLevel() >= kHIDEBUG)
               Info("CheckAutoTerm", "self-Cancelling reader thread.");
            // Reset thread handlers (real termination will be done
            // at ::Exit() )
            fReaderthreadhandler = 0;
            fReaderthreadrunning = kFALSE;
            // Destroy the socket
            delete fSocket;
            fSocket = 0;
            // We are going to exit
            doexit = kTRUE;
         }
      }
   }

   // Now exit, if requested
   if (doexit) {
      UnlockChannel();
      TThread::Exit(0);
   }
}

//____________________________________________________________________________
void TXPhyConnection::HandleUnsolicited(TXMessage *m)
{
   // Local processing of unsolicited responses is done here

   Bool_t ProcessingToGo = kTRUE;
   struct ServerResponseBody_Attn *attnbody;

   Touch();

   // Local processing of the unsolicited TXMessage
   attnbody = (struct ServerResponseBody_Attn *)m->GetData();
   if (attnbody) {

      switch (attnbody->actnum) {
      case kXR_asyncms:
         // A message arrived from the server. Let's print it.
         Info("HandleUnsolicited",
              "Message from server at socket %d[%s:%d]: '%s'.",
              fSocket->GetDescriptor(), fRemoteAddress.Data(),
              fRemotePort, attnbody->parms);
         ProcessingToGo = kFALSE;
         break;
      }
   }

   // Now we propagate the message to the interested object, if any
   // It could be some sort of upper layer of the architecture
   if (ProcessingToGo)
      SendUnsolicitedMsg(this, m);
}

//____________________________________________________________________________
Int_t TXPhyConnection::WriteRaw(const void *buf, Int_t len, ESendRecvOptions opt)
{
   // Send 'len' bytes located at 'buf' to the connected server.
   // Return number of bytes sent.

   Int_t res;

   Touch();

   if (IsValid()) {

      if (DebugLevel() >= kDUMPDEBUG)
         Info("WriteRaw", "Writing to socket %d[%s:%d]",
              fSocket->GetDescriptor(), fRemoteAddress.Data(), fRemotePort);

      res = fSocket->SendRaw(buf, len, opt);

      if ((res <= 0)  && (res != TXSOCK_ERR_TIMEOUT) &&
          (DebugLevel() >= kHIDEBUG) && (gSystem->GetErrno()))
         Info("WriteRaw", "Write error [%s:%d]. Errno: %d:'%s'.",
              fRemoteAddress.Data(), fRemotePort, gSystem->GetErrno(),
              gSystem->GetError() );

      // If a socket error comes, then we disconnect (and destroy the fSocket)
      if ((res < 0) || (!fSocket->IsValid())) {
         if (DebugLevel() >= kHIDEBUG)
            Info("WriteRaw",
                 "Socket reported a disconnection (server[%s:%d]). Closing.",
                 fRemoteAddress.Data(), fRemotePort);
         Disconnect();
      }

      Touch();
      return( res );
   }
   else {
      // Socket already destroyed or disconnected
      if (DebugLevel() >= kDUMPDEBUG)
         Info("WriteRaw", "Socket is disconnected (server [%s:%d])",
              fRemoteAddress.Data(), fRemotePort);
      return TXSOCK_ERR;
   }
}

//____________________________________________________________________________
UInt_t TXPhyConnection::GetBytesSent()
{
   // Return number of bytes sent

   if (IsValid())
      return fSocket->GetBytesSent();
   else {
      // Socket already destroyed or disconnected
      if (DebugLevel() >= kDUMPDEBUG)
         Info("GetBytesSent",
              "Socket is disconnected (server [%s:%d])",
              fRemoteAddress.Data(), fRemotePort);
      return 0;
   }
}

//____________________________________________________________________________
UInt_t TXPhyConnection::GetBytesRecv()
{
   // Return number of bytes received

   if (IsValid())
      return fSocket->GetBytesRecv();
   else {
      // Socket already destroyed or disconnected
      if(DebugLevel() >= kDUMPDEBUG)
         Info("GetBytesRecv",
              "Socket is disconnected (server [%s:%d])",
              fRemoteAddress.Data(), fRemotePort);
      return 0;
   }
}

//____________________________________________________________________________
UInt_t TXPhyConnection::GetSocketBytesSent()
{
   // Return number of bytes sent

   if (IsValid())
      return fSocket->GetSocketBytesSent();
   else {
      // Socket already destroyed or disconnected
      if(DebugLevel() >= kDUMPDEBUG)
         Info("GetSocketBytesSent",
              "Socket is disconnected (server [%s:%s])",
              fRemoteAddress.Data(), fRemotePort);
      return 0;
   }
}

//____________________________________________________________________________
UInt_t TXPhyConnection::GetSocketBytesRecv()
{
   // Return number of bytes received

   if (IsValid())
      return fSocket->GetSocketBytesRecv();
   else {
      // Socket already destroyed or disconnected
      if(DebugLevel() >= kDUMPDEBUG)
         Info("GetSocketBytesRecv",
              "Socket is disconnected (server [%s:%s])",
              fRemoteAddress.Data(), fRemotePort);
      return 0;
   }
}

//____________________________________________________________________________
Bool_t TXPhyConnection::ExpiredTTL()
{
   // Check expiration time
   return( (time(0) - fLastUseTimestamp) > fTTLsec ? kTRUE : kFALSE);
}

//____________________________________________________________________________
void TXPhyConnection::LockChannel()
{
   // Lock
   fRwMutex->Lock();
}

//____________________________________________________________________________
void TXPhyConnection::UnlockChannel()
{
   // Unlock
   fRwMutex->UnLock();
}
