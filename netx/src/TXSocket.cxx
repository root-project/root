// @(#)root/netx:$Name:  $:$Id: TXSocket.cxx,v 1.4 2004/09/08 10:21:40 brun Exp $
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
// TXSocket                                                             //
//                                                                      //
// Extension of TSocket to handle read/write and connection timeouts.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXSocket.h"
#include "TEnv.h"
#include "TError.h"
#include "TException.h"
#include "TXDebug.h"

ClassImp(TXSocket);

//_____________________________________________________________________________
TXSocket::TXSocket(TString TcpAddress, Int_t TcpPort, Int_t TcpWindowSize)
         : TSocket()
{
   // Create a TXSocket object (that doesn't actually connect to any server.

   // Init of the separate connect parms
   fHost2contact.TcpAddress = TcpAddress;
   fHost2contact.TcpPort = TcpPort;
   fHost2contact.TcpWindowSize = TcpWindowSize;
   fRequestTimeout = gEnv->GetValue("XNet.RequestTimeout",
                                     DFLT_REQUESTTIMEOUT);
   fASYNC = gEnv->GetValue("XNet.GoAsynchronous", DFLT_GOASYNC);

}

//_____________________________________________________________________________
TXSocket::~TXSocket()
{
   // Destructor
}

//_____________________________________________________________________________
Int_t TXSocket::RecvRaw(void* buffer, Int_t length, ESendRecvOptions opt)
{
   // Override of TSocket::RecvRaw. Before calling TSocket::RecvRaw we poll for a
   // while on the socket descriptor waiting for a POLLIN event (data to read).

   time_t starttime;
   Int_t bytesread = 0, n;

   SetOption(kNoBlock, 0);

   // We cycle until we have all the data we are waiting for
   // Or until a timeout occurs
   starttime = time(0);
   while (bytesread < length) {

      Int_t ReadyToRecv = 0;
      // We cycle on the poll, ignoring the possible interruptions
      do {
         // If too much time has elapsed, then we return an error
         if ((time(0) - starttime) > fRequestTimeout) {
            if (!fASYNC || (DebugLevel() >= kDUMPDEBUG))
               Error("RecvRaw","Request timed out %d seconds reading %d bytes"
                     " from socket %d (server[%s:%d])",
                     fRequestTimeout, length, fSocket,
                     GetInetAddress().GetHostName(), GetPort());

	    return TXSOCK_ERR_TIMEOUT;
         }

         // Wait for a socket ready for receiving
         ReadyToRecv = TSocket::Select(TSocket::kRead,1000);

      } while (!ReadyToRecv);

      if (ReadyToRecv < 0)
         return TXSOCK_ERR;

      n = TSocket::RecvRaw((char *)buffer + bytesread, length - bytesread, opt);
      if (!n)
         return (0);

      bytesread += n;

   } // while

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

   // We cycle until we have all the data we are waiting for
   // Or until a timeout occurs
   starttime = time(0);
   while (byteswritten < length) {

      Int_t ReadyToWrite = 0;

      do {
         // If too much time has elapsed, then we return an error
         if ( (time(0) - starttime) > fRequestTimeout ) {
            Error("SendRaw","Request timed out %d seconds writing %d bytes"
                  " from socket %d (server[%s:%d])", fRequestTimeout, length,
                  fSocket, GetInetAddress().GetHostName(), GetPort());

	    return TXSOCK_ERR_TIMEOUT;
         }

         // Wait for a socket ready for sending
         ReadyToWrite = TSocket::Select(TSocket::kWrite,1000);

      } while (!ReadyToWrite);

      if (ReadyToWrite < 0)
         return TXSOCK_ERR;

      n = TSocket::SendRaw((char *)buffer + byteswritten,
                                   length - byteswritten, opt);
      if (!n)
         return (0);

      byteswritten += n;

   } // while

   return byteswritten;
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

   Assert(gSystem);

   if (DebugLevel() >= kHIDEBUG)
      Info("Create","Setting fService to %s", gSystem->GetServiceByPort(port));
   fService = gSystem->GetServiceByPort(port);

   fAddress = gSystem->GetHostByName(host.Data());

   if (DebugLevel() >= kHIDEBUG)
      Info("Create","Setting fAddress.fPort to %d", port);

   fAddress.fPort = port;
   SetName(fAddress.GetHostName());
   SetTitle(fService);

   fUrl = host;
   fTcpWindowSize = tcpwindowsize;
   fServType = TSocket::kROOTD;

   if (DebugLevel() >= kHIDEBUG)
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
      if(DebugLevel() >= kHIDEBUG)
         Info("Create","Connection failed. Setting fSocket to -1");
   } else {
      // Add to the list
      gROOT->GetListOfSockets()->Add(this);
   }
}

//____________________________________________________________________________
TSocket *TXSocket::ExtractSocket()
{
   // Return copy of the underlying TSocket part and set the descriptor
   // to -1 (so that the connection is not closed when the TXSocket is
   // deleted).
   // Used to save an open connection to rootd daemons

   TSocket *sock = 0;
   if (IsValid()) {
      sock = new TSocket((const TSocket &)(*this));
      SetDescriptor(-1);
   }
   return sock;
}
