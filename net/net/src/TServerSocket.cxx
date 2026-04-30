// @(#)root/net:$Id$
// Author: Fons Rademakers   18/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\file TServerSocket.cxx
\class TServerSocket
\brief This class implements server sockets.
\note This class deals with sockets: the user is entirely responsible for the security of their usage, for example, but
not limited to, the management of the connections to said sockets.

This class implements server sockets. A server socket waits for
requests to come in over the network. It performs some operation
based on that request and then possibly returns a full duplex socket
to the requester. The actual work is done via the TSystem class
(either TUnixSystem or TWinNTSystem).

**/


#include "TServerSocket.h"
#include "TSocket.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TError.h"
#include <string>
#include "TVirtualMutex.h"

TVirtualMutex *gSrvAuthenticateMutex = 0;

////////////////////////////////////////////////////////////////////////////////
/// Create a server socket object for a named service. Set reuse to true
/// to force reuse of the server socket (i.e. do not wait for the time
/// out to pass). Using backlog one can set the desirable queue length
/// for pending connections.
/// Use tcpwindowsize to specify the size of the receive buffer, it has
/// to be specified here to make sure the window scale option is set (for
/// tcpwindowsize > 65KB and for platforms supporting window scaling).
/// The socketBindOption parameter allows to specify how the socket will be
/// bound. See the documentation of ESocketBindOption for the details.
/// Use IsValid() to check the validity of the
/// server socket. In case server socket is not valid use GetErrorCode()
/// to obtain the specific error value. These values are:
///  0 = no error (socket is valid)
/// -1 = low level socket() call failed
/// -2 = low level bind() call failed
/// -3 = low level listen() call failed
/// Every valid server socket is added to the TROOT sockets list which
/// will make sure that any open sockets are properly closed on
/// program termination.

TServerSocket::TServerSocket(const char *service, Bool_t reuse, Int_t backlog,
                             Int_t tcpwindowsize)
{
   R__ASSERT(gROOT);
   R__ASSERT(gSystem);

   SetName("ServerSocket");

   fSecContext = 0;
   fSecContexts = new TList;

   // If this is a local path, try announcing a UNIX socket service
   ResetBit(TSocket::kIsUnix);
   if (service && (!gSystem->AccessPathName(service) ||
#ifndef WIN32
      service[0] == '/')) {
#else
      service[0] == '/' || (service[1] == ':' && service[2] == '/'))) {
#endif
      SetBit(TSocket::kIsUnix);
      fService = "unix:";
      fService += service;
      fSocket = gSystem->AnnounceUnixService(service, backlog);
      if (fSocket >= 0) {
         R__LOCKGUARD(gROOTMutex);
         gROOT->GetListOfSockets()->Add(this);
      }
   } else {
      // TCP / UDP socket
      fService = service;
      int port = gSystem->GetServiceByName(service);
      if (port != -1) {
         fSocket = gSystem->AnnounceTcpService(port, reuse, backlog, tcpwindowsize, ESocketBindOption::kInaddrLoopback);
         if (fSocket >= 0) {
            R__LOCKGUARD(gROOTMutex);
            gROOT->GetListOfSockets()->Add(this);
         }
      } else {
         fSocket = -1;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create a server socket object on a specified port. Set reuse to true
/// to force reuse of the server socket (i.e. do not wait for the time
/// out to pass). Using backlog one can set the desirable queue length
/// for pending connections. If port is 0 a port scan will be done to
/// find a free port. This option is mutual exlusive with the reuse option.
/// Use tcpwindowsize to specify the size of the receive buffer, it has
/// to be specified here to make sure the window scale option is set (for
/// tcpwindowsize > 65KB and for platforms supporting window scaling).
/// The socketBindOption parameter allows to specify how the socket will be
/// bound. See the documentation of ESocketBindOption for the details.
/// Use IsValid() to check the validity of the
/// server socket. In case server socket is not valid use GetErrorCode()
/// to obtain the specific error value. These values are:
///  0 = no error (socket is valid)
/// -1 = low level socket() call failed
/// -2 = low level bind() call failed
/// -3 = low level listen() call failed
/// Every valid server socket is added to the TROOT sockets list which
/// will make sure that any open sockets are properly closed on
/// program termination.

TServerSocket::TServerSocket(Int_t port, Bool_t reuse, Int_t backlog, Int_t tcpwindowsize,
                             ESocketBindOption socketBindOption)
{
   R__ASSERT(gROOT);
   R__ASSERT(gSystem);

   SetName("ServerSocket");

   fSecContext = 0;
   fSecContexts = new TList;
   fService = gSystem->GetServiceByPort(port);
   SetTitle(fService);

   fSocket = gSystem->AnnounceTcpService(port, reuse, backlog, tcpwindowsize, socketBindOption);
   if (fSocket >= 0) {
      R__LOCKGUARD(gROOTMutex);
      gROOT->GetListOfSockets()->Add(this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor: cleanup authentication stuff (if any) and close

TServerSocket::~TServerSocket()
{
   R__LOCKGUARD2(gSrvAuthenticateMutex);
   if (fSecContexts) {
      // Remove the list
      fSecContexts->Delete();
      SafeDelete(fSecContexts);
      fSecContexts = 0;
   }

   Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Accept a connection on a server socket. Returns a full-duplex
/// communication TSocket object. If no pending connections are
/// present on the queue and nonblocking mode has not been enabled
/// with SetOption(kNoBlock,1) the call blocks until a connection is
/// present. The returned socket must be deleted by the user. The socket
/// is also added to the TROOT sockets list which will make sure that
/// any open sockets are properly closed on program termination.
/// In case of error 0 is returned and in case non-blocking I/O is
/// enabled and no connections are available -1 is returned.
/// Note: opt used to pass authentication options but is currently unused.

TSocket *TServerSocket::Accept(UChar_t /* opt */)
{
   if (fSocket == -1) { return 0; }

   TSocket *socket = new TSocket;

   Int_t soc = gSystem->AcceptConnection(fSocket);
   if (soc == -1) { delete socket; return 0; }
   if (soc == -2) { delete socket; return (TSocket*) -1; }

   socket->fSocket  = soc;
   socket->fSecContext = 0;
   socket->fService = fService;
   if (!TestBit(TSocket::kIsUnix))
      socket->fAddress = gSystem->GetPeerName(socket->fSocket);
   if (socket->fSocket >= 0) {
      R__LOCKGUARD(gROOTMutex);
      gROOT->GetListOfSockets()->Add(socket);
   }

   return socket;
}

////////////////////////////////////////////////////////////////////////////////
/// Return internet address of host to which the server socket is bound,
/// i.e. the local host. In case of error TInetAddress::IsValid() returns
/// kFALSE.

TInetAddress TServerSocket::GetLocalInetAddress()
{
   if (fSocket != -1) {
      if (fAddress.GetPort() == -1)
         fAddress = gSystem->GetSockName(fSocket);
      return fAddress;
   }
   return TInetAddress();
}

////////////////////////////////////////////////////////////////////////////////
/// Get port # to which server socket is bound. In case of error returns -1.

Int_t TServerSocket::GetLocalPort()
{
   if (fSocket != -1) {
      if (fAddress.GetPort() == -1)
         fAddress = GetLocalInetAddress();
      return fAddress.GetPort();
   }
   return -1;
}
