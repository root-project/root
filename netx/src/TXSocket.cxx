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
// TXSocket                                                             //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// Extension of TSocket to handle read/write and connection timeouts    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXSocket.h"
#include "TEnv.h"
#include "TError.h"
#include "TROOT.h"

#include "TException.h"

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include "TXDebug.h"
#include "TXMutexLocker.h"

ClassImp(TXSocket);

Int_t excode = 1;

extern "C" void *SocketConnecterThread(void *);


//_____________________________________________________________________________
TXSocket::TXSocket(TString TcpAddress, Int_t TcpPort, Int_t TcpWindowSize) 
         : TSocket()
{
   // Create a TXSocket object (that doesn't actually connect to any server.
   // The dedicated thread SocketConnecterThread will do).

   // Init of the separate connect parms
   fHost2contact.TcpAddress = TcpAddress;
   fHost2contact.TcpPort = TcpPort;
   fHost2contact.TcpWindowSize = TcpWindowSize;
   fConnectSem = 0;
   fRequestTimeout = gEnv->GetValue("XNet.RequestTimeout",
                                     DFLT_REQUESTTIMEOUT);
   fASYNC = gEnv->GetValue("XNet.GoAsynchronous", DFLT_GOASYNC);

   fMonMutex = new TMutex(kTRUE);

   if (!fMonMutex)
      Error("TXPhyConnection", 
            "can't create mutex for TMonitor protection: out of system resources");

   fWriteMonitor = 0;
   fReadMonitor  = 0;

   fReadMonitorActCnt = 0;
   fWriteMonitorActCnt = 0;
}

//_____________________________________________________________________________
TXSocket::~TXSocket()
{
   // Destructor 
   if (fWriteMonitor)
      fWriteMonitor->DeActivateAll();

   if (fReadMonitor)
      fReadMonitor->DeActivateAll();

   SafeDelete( fWriteMonitor );
   SafeDelete( fReadMonitor );

   SafeDelete( fMonMutex );

}




//_____________________________________________________________________________
void TXSocket::ReadMonitorActivate() {
   TXMutexLocker mtx(fMonMutex);

   if (!fReadMonitorActCnt)
      fReadMonitor->Activate(this);

   fReadMonitorActCnt++;

}

//_____________________________________________________________________________
void TXSocket::ReadMonitorDeactivate() {
   TXMutexLocker mtx(fMonMutex);

   if (fReadMonitorActCnt > 0) {
      fReadMonitorActCnt--;

  
      if (!fReadMonitorActCnt)
	 fReadMonitor->DeActivateAll();
   }
}

//_____________________________________________________________________________
void TXSocket::WriteMonitorActivate() {
   TXMutexLocker mtx(fMonMutex);

   if (!fWriteMonitorActCnt)
      fWriteMonitor->Activate(this);

   fWriteMonitorActCnt++;

}

//_____________________________________________________________________________
void TXSocket::WriteMonitorDeactivate() {
   TXMutexLocker mtx(fMonMutex);


   if (fWriteMonitorActCnt > 0) {
      fWriteMonitorActCnt--;

      if (!fWriteMonitorActCnt)
	 fWriteMonitor->DeActivateAll();

   }

}

//_____________________________________________________________________________
Int_t TXSocket::RecvRaw(void* buffer, Int_t length, ESendRecvOptions opt)
{
   // Override of TSocket::RecvRaw. Before calling TSocket::RecvRaw we poll for a
   // while on the socket descriptor waiting for a POLLIN event (data to read).

   time_t starttime;
   Int_t bytesread = 0, n;

   SetOption(kNoBlock, 0);
   if (!fASYNC)
      ReadMonitorActivate();

   // We cycle until we have all the data we are waiting for
   // Or until a timeout occurs
   starttime = time(0);
   while (bytesread < length) {

   TSocket *s = (TSocket *)-1;

      // We cycle on the poll, ignoring the possible interruptions
      do { 
         // If too much time has elapsed, then we return an error
         if ((time(0) - starttime) > fRequestTimeout) {

            if (!fASYNC)
	       ReadMonitorDeactivate();

            if (!fASYNC || (DebugLevel() >= TXDebug::kDUMPDEBUG))
               Error("RecvRaw","Request timed out %d seconds reading %d bytes"
                     " from socket %d (server[%s:%d])", 
                     fRequestTimeout, length, fSocket,
                     GetInetAddress().GetHostName(), GetPort());

	    return TXSOCK_ERR_TIMEOUT;
         }

         // Wait for a socket ready for receiving
         s = fReadMonitor->Select(1000);
      
      } while (s == (TSocket *)-1);
      
      if (!s->IsValid()) {
         if (!fASYNC)
	    ReadMonitorDeactivate();

         return TXSOCK_ERR;
      }
      
      n = TSocket::RecvRaw((char *)buffer + bytesread, length - bytesread, opt);
      if (!n) {
         if (!fASYNC)
            ReadMonitorDeactivate();
         return (0);
      }

      bytesread += n;

   } // while

   if (!fASYNC)
      ReadMonitorDeactivate();
   return bytesread;
}

//_____________________________________________________________________________
Int_t TXSocket::SendRaw(const void* buffer, Int_t length, ESendRecvOptions opt)
{
   // Override of TSocket::SendRaw. Before calling TSocket::SendRaw we poll
   // for a while on the socket descriptor waiting for a POLLOUT event 
   // (writes will not hang)

   time_t starttime;
   Int_t byteswritten = 0, n;

   if (!TSocket::IsValid())
      return TXSOCK_ERR;

   SetOption(kNoBlock, 0);
   WriteMonitorActivate();

   // We cycle until we have all the data we are waiting for
   // Or until a timeout occurs
   starttime = time(0);
   while (byteswritten < length) {

      TSocket *s = (TSocket *)-1;

      do {
         // If too much time has elapsed, then we return an error
         if ( (time(0) - starttime) > fRequestTimeout ) {
            Error("SendRaw","Request timed out %d seconds writing %d bytes"
                  " from socket %d (server[%s:%d])", fRequestTimeout, length,
                  fSocket, GetInetAddress().GetHostName(), GetPort());

	    //WriteMonitorDeactivate();

	    return TXSOCK_ERR_TIMEOUT;
         }
      
         // Wait for a socket ready for sending
         s = fWriteMonitor->Select(1000);

      } while (s == (TSocket *)-1);

      if (!s->IsValid()) {
	 //WriteMonitorDeactivate();

         return TXSOCK_ERR;
      }

      n = TSocket::SendRaw((char *)buffer + byteswritten,
                                   length - byteswritten, opt);
      if (!n) {
	 //WriteMonitorDeactivate();
         return (0);
      }
    
      byteswritten += n;

   } // while

   WriteMonitorDeactivate();

return byteswritten;
}

//_____________________________________________________________________________
extern "C" void *SocketConnecterThread(void * arg)
{
   // This thread tries to connect the given (and calling) TXSocket to
   // its destination host. In the meanwhile, the caller can simply check
   // whether the socket descriptor becomes good after the condition variable
   // says something, or the timeout has elapsed
   int oldcanctype;

   TXSocket *thisObj;

   thisObj = (TXSocket *)arg;

   // We want this thread to terminate when requested so
   pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldcanctype);

   thisObj->Create(thisObj->fHost2contact.TcpAddress,
                   thisObj->fHost2contact.TcpPort,
                   thisObj->fHost2contact.TcpWindowSize);
  
   pthread_testcancel();

   // Something happened since we are here. So we signal the
   // condition variable.
   thisObj->fConnectSem->Post();

   pthread_exit(0);
   return 0;
}

//_____________________________________________________________________________
void TXSocket::TryConnect()
{
   // Connection attempt

   this->Create(this->fHost2contact.TcpAddress,
                this->fHost2contact.TcpPort,
                this->fHost2contact.TcpWindowSize);

   // Now, we know the result of the connect process from the
   // socket descriptor. If and when needed.

}

//_____________________________________________________________________________
void TXSocket::CatchTimeOut()
{
   // Called in connection with a timer timeout 

   ::Error("TXSocket::CatchTimeOut", 
           "Timeout elapsed after %d seconds for connection to server", 
           gEnv->GetValue("XNet.ConnectTimeout", DFLT_CONNECTTIMEOUT));
   return;
}

//_____________________________________________________________________________
void TXSocket::Create(TString host, Int_t port, Int_t tcpwindowsize)
{
   // Create a connection

   Assert(gROOT);
   Assert(gSystem);

   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("Create","Setting fService to %s", gSystem->GetServiceByPort(port));
   fService = gSystem->GetServiceByPort(port);

   fAddress = gSystem->GetHostByName(host.Data());

   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("Create","Setting fAddress.fPort to %d", port);

   fAddress.fPort = port;
   SetName(fAddress.GetHostName());
   SetTitle(fService);

   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("Create","Calling TUnixSystem::OpenConnection with params"
           " %s:%d tcpwin=%d", host.Data(), port, tcpwindowsize);

   // set an alarm that will send a SIGALARM after
   // XNet.ConnectTimeout
   // in order to stop the syscalls and retry
   TTimer alarm(0, kFALSE);
   alarm.SetInterruptSyscalls();

   // The static method CatchTimeOut will be called at timeout
   alarm.Connect("Timeout()", "TXSocket", 0, "CatchTimeOut()");

   // TTimer::Starts wants millisec
   Int_t to = gEnv->GetValue("XNet.ConnectTimeout",DFLT_CONNECTTIMEOUT);
   alarm.Start(to*1000, kTRUE);

   // Now connect
   fSocket = gSystem->OpenConnection(host.Data(), port, tcpwindowsize);

   if (fSocket == -1) {
      fAddress.fPort = -1;
      if(DebugLevel() >= TXDebug::kHIDEBUG)
         Info("Create","Connection failed. Setting fSocket to -1");
   }
   else {

      fWriteMonitor   = new TMonitor;
      fReadMonitor    = new TMonitor;
      fWriteMonitor->Add(this, TMonitor::kWrite);
      fReadMonitor->Add(this, TMonitor::kRead);

      fWriteMonitor->DeActivateAll();
      fReadMonitor->DeActivateAll();

      fReadMonitorActCnt = 0;
      fWriteMonitorActCnt = 0;

      gROOT->GetListOfSockets()->Add(this);
      if (fASYNC)
         ReadMonitorActivate();
   }
}
