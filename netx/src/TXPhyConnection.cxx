// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXPhyConnection                                                      //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// Class handling physical connections to xrootd servers                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <time.h>
#include <string.h>
#include <stdlib.h>
#include "TXPhyConnection.h"
#include "TError.h"
#include "TXDebug.h"
#include "TXMessage.h"
#include "TString.h"
#include <sys/socket.h>
#include "TEnv.h"
#include "TSystem.h"


#include <TThread.h>
#include <TApplication.h>
#include <Riostream.h>

ClassImp(TXPhyConnection);

//____________________________________________________________________________
void *SocketReaderThread(void * arg)
{
   // This thread is the base for the async capabilities of TXPhyConnection
   // It repeatedly keeps reading from the socket, while feeding the
   // MsqQ with a stream of TXMessages containing what's happening
   // at the socket level

   TXPhyConnection *thisObj;

   TThread::Lock();
   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("SocketReaderThread", "Reader Thread starting.");

   TThread::UnLock();

   TThread::SetCancelOn();
   TThread::SetCancelAsynchronous();

   TThread::Lock();
   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("SocketReaderThread", "Reader Thread started.");
   TThread::UnLock();

   thisObj = (TXPhyConnection *)arg;

   while (1) {
      //TThread::Lock();
      thisObj->BuildXMessage(kDefault, kTRUE, kTRUE);
      //TThread::UnLock();
   }


   TThread::Exit();
   return 0;
}

//____________________________________________________________________________
TXPhyConnection::TXPhyConnection(TXAbsUnsolicitedMsgHandler *h)
{
   // Constructor


   // Initialization of lock mutex

   fRwMutex = new TMutex(kTRUE);

   if (!fRwMutex)
      Error("TXPhyConnection", 
            "can't create mutex for read/write: out of system resources");

   Touch();

   fServer = kUnknown;
   SetLogged(kNo);
   fRequestTimeout = gEnv->GetValue("XNet.RequestTimeout",
                                    DFLT_REQUESTTIMEOUT);
   UnsolicitedMsgHandler = h;

   fReaderthreadhandler = 0;
   fReaderthreadrunning = kFALSE;
}

//____________________________________________________________________________
TXPhyConnection::~TXPhyConnection()
{
   // Destructor

   Disconnect();

   SafeDelete(fRwMutex);
}

//____________________________________________________________________________
Bool_t TXPhyConnection::Connect(TString TcpAddress, Int_t TcpPort, 
                                Int_t TcpWindowSize)
{
   // Connect to remote server

   if (DebugLevel() >= TXDebug::kHIDEBUG)
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

   if (DebugLevel() >= TXDebug::kHIDEBUG)
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

   // Parametric asynchronous stuff.
   // If we are going Sync, then nothing has to be done,
   // otherwise the reader thread must be started
   if ( (!fReaderthreadrunning) && 
         gEnv->GetValue("XNet.GoAsynchronous", DFLT_GOASYNC) ) {

      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("StartReader", "Starting reader thread...");

      // Now we launch  the reader thread
      fReaderthreadhandler = new TThread((TThread::VoidFunc_t) SocketReaderThread,
					 this);

      if (!fReaderthreadhandler)
         Info("StartReader",
              "Can't create reader thread: out of system resources");
      else {
	 // We want this thread to terminate when requested so


	 fReaderthreadhandler->Run();

         fReaderthreadrunning = kTRUE;
      }
   }
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
void TXPhyConnection::Disconnect()
{



   // Parametric asynchronous stuff
   // If we are going async, we have to terminate the reader thread
   if (gEnv->GetValue("XNet.GoAsynchronous", DFLT_GOASYNC)) {



      if (fReaderthreadrunning) {
      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("Disconnect", "Terminating reader thread.");

	 fReaderthreadhandler->Kill();

      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("Disconnect", "Waiting for the reader thread termination...");

         fReaderthreadhandler->Join();

      }

      fReaderthreadrunning = kFALSE;
      SafeDelete(fReaderthreadhandler);
      fReaderthreadhandler = 0;
   }

   // Disconnect from remote server
   if (DebugLevel() >= TXDebug::kDUMPDEBUG)
      Info("Disconnect", "Deleting low level socket...");

   SafeDelete(fSocket);
   fSocket = 0;

}

//____________________________________________________________________________
void TXPhyConnection::Touch()
{
   // Set last-use-time to present time

   time_t t = time(0);
   if (DebugLevel() >= TXDebug::kDUMPDEBUG)
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

      if (DebugLevel() >= TXDebug::kDUMPDEBUG)
         Info("ReadRaw", "Reading from socket: %d[%s:%d]",
               fSocket->GetDescriptor(), fRemoteAddress.Data(), fRemotePort);

      res = fSocket->RecvRaw(buf, len, opt);

      if ((res <= 0) && (res != TXSOCK_ERR_TIMEOUT) &&
          (DebugLevel() >= TXDebug::kHIDEBUG) && (gSystem->GetErrno()) )
         Info("ReadRaw", "Read error [%s:%d]. Errno %d:'%s'.",
               fRemoteAddress.Data(), fRemotePort, gSystem->GetErrno(),
               gSystem->GetError() );

      // If a socket error comes, then we disconnect (and destroy the fSocket)
      // but we have not to disconnect in the case of a timeout
      if (((res <= 0) && (res != TXSOCK_ERR_TIMEOUT)) ||
          (!fSocket->IsValid())) {

         if (DebugLevel() >= TXDebug::kHIDEBUG)
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
      if (DebugLevel() >= TXDebug::kDUMPDEBUG)
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
         if ( !IgnoreTimeouts || !m->IsError() )
            fMsgQ.PutMsg(m);
         else {
            delete m;
            m = 0;
         }
      }
  
   return m;
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

      if (DebugLevel() >= TXDebug::kDUMPDEBUG)
         Info("WriteRaw", "Writing to socket %d[%s:%d]",
              fSocket->GetDescriptor(), fRemoteAddress.Data(), fRemotePort);
    
      res = fSocket->SendRaw(buf, len, opt);

      if ((res <= 0)  && (res != TXSOCK_ERR_TIMEOUT) &&
          (DebugLevel() >= TXDebug::kHIDEBUG) && (gSystem->GetErrno()))
         Info("WriteRaw", "Write error [%s:%d]. Errno: %d:'%s'.",
              fRemoteAddress.Data(), fRemotePort, gSystem->GetErrno(),
              gSystem->GetError() );

      // If a socket error comes, then we disconnect (and destroy the fSocket)
      if ((res < 0) || (!fSocket->IsValid())) {
         if (DebugLevel() >= TXDebug::kHIDEBUG)
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
      if (DebugLevel() >= TXDebug::kDUMPDEBUG)
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
      if (DebugLevel() >= TXDebug::kDUMPDEBUG)
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
      if(DebugLevel() >= TXDebug::kDUMPDEBUG)
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
      if(DebugLevel() >= TXDebug::kDUMPDEBUG)
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
      if(DebugLevel() >= TXDebug::kDUMPDEBUG)
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
