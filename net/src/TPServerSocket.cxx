// @(#)root/net:$Name:$:$Id:$
// Author: Fons Rademakers   19/1/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPServerSocket                                                       //
//                                                                      //
// This class implements parallel server sockets. A parallel server     //
// socket waits for requests to come in over the network. It performs   //
// some operation based on that request and then possibly returns a     //
// full duplex parallel socket to the requester. The actual work is     //
// done via the TSystem class (either TUnixSystem, TWin32System or      //
// TMacSystem).                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPServerSocket.h"
#include "TPSocket.h"
#include "TROOT.h"


ClassImp(TPServerSocket)

//______________________________________________________________________________
TPServerSocket::TPServerSocket(Int_t port, Bool_t reuse, Int_t backlog,
                               Int_t tcpwindowsize) : TNamed("PServerSocket", "")
{
   // Create a parallel server socket object on a specified port. Set reuse
   // to true to force reuse of the server socket (i.e. do not wait for the
   // time out to pass). Using backlog one can set the desirable queue length
   // for pending connections.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Use IsValid() to check the validity of the
   // server socket. In case server socket is not valid use GetErrorCode()
   // to obtain the specific error value. These values are:
   //  0 = no error (socket is valid)
   // -1 = low level socket() call failed
   // -2 = low level bind() call failed
   // -3 = low level listen() call failed
   // Every valid server socket is added to the TROOT sockets list which
   // will make sure that any open sockets are properly closed on
   // program termination.

   fSetupServer   = new TServerSocket(port, reuse, backlog, tcpwindowsize);
   fTcpWindowSize = tcpwindowsize;
   if (fSetupServer->IsValid()) {
      gROOT->GetListOfSockets()->Remove(fSetupServer);
      gROOT->GetListOfSockets()->Add(this);
   }
}

//______________________________________________________________________________
TPServerSocket::TPServerSocket(const char *service, Bool_t reuse, Int_t backlog,
                               Int_t tcpwindowsize) : TNamed("PServerSocket", "")
{
   // Create a parallel server socket object for a named service. Set reuse
   // to true to force reuse of the server socket (i.e. do not wait for the
   // time out to pass). Using backlog one can set the desirable queue length
   // for pending connections.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Use IsValid() to check the validity of the
   // server socket. In case server socket is not valid use GetErrorCode()
   // to obtain the specific error value. These values are:
   //  0 = no error (socket is valid)
   // -1 = low level socket() call failed
   // -2 = low level bind() call failed
   // -3 = low level listen() call failed
   // Every valid server socket is added to the TROOT sockets list which
   // will make sure that any open sockets are properly closed on
   // program termination.

   fSetupServer   = new TServerSocket(service, reuse, backlog, tcpwindowsize);
   fTcpWindowSize = tcpwindowsize;
   if (fSetupServer->IsValid()) {
      gROOT->GetListOfSockets()->Remove(fSetupServer);
      gROOT->GetListOfSockets()->Add(this);
   }
}

//______________________________________________________________________________
TPServerSocket::~TPServerSocket()
{
   // Delete parallel server socket.

   Close();
   delete fSetupServer;
}

//______________________________________________________________________________
TPSocket *TPServerSocket::Accept()
{
   // Accept a connection on a parallel server socket. Returns a full-duplex
   // parallel communication TPSocket object. If no pending connections are
   // present on the queue and nonblocking mode has not been enabled
   // with SetOption(kNoBlock,1) the call blocks until a connection is
   // present. The returned socket must be deleted by the user. The socket
   // is also added to the TROOT sockets list which will make sure that
   // any open sockets are properly closed on program termination.
   // In case of error 0 is returned and in case non-blocking I/O is
   // enabled and no connections are available -1 is returned.

   TSocket  *setupSocket;
   TSocket  **pSockets;
   TPSocket *newPSocket;

   Int_t size, port;

   // wait for the incoming connections to the server and accept them
   setupSocket = fSetupServer->Accept();

   // receive the port number and number of parallel sockets from the
   // client and establish 'n' connections
   setupSocket->Recv(port, size);

   pSockets = new TSocket*[size];

   for (int i = 0; i < size; i++) {
      pSockets[i] = new TSocket(setupSocket->GetInetAddress(), port, fTcpWindowSize);
      gROOT->GetListOfSockets()->Remove(pSockets[i]);
   }

   // create TPSocket object with all the accepted sockets
   newPSocket = new TPSocket(pSockets, size);

   // clean up
   delete setupSocket;

   // return the TPSocket object
   return newPSocket;
}

//______________________________________________________________________________
void TPServerSocket::Close(Option_t *option)
{
   // Close the socket. If option is "force", calls shutdown(id,2) to
   // shut down the connection. This will close the connection also
   // for the parent of this process. Also called via the dtor (without
   // option "force", call explicitely Close("force") if this is desired).

   if (fSetupServer->IsValid())
      gROOT->GetListOfSockets()->Remove(this);

   fSetupServer->Close(option);
}

//______________________________________________________________________________
TInetAddress TPServerSocket::GetLocalInetAddress()
{
   // Return internet address of host to which the server socket is bound,
   // i.e. the local host. In case of error TInetAddress::IsValid() returns
   // kFALSE.

   return fSetupServer->GetLocalInetAddress();
}

//______________________________________________________________________________
Int_t TPServerSocket::GetLocalPort()
{
   // Get port # to which server socket is bound. In case of error returns -1.

   return fSetupServer->GetLocalPort();
}

//______________________________________________________________________________
Int_t TPServerSocket::SetOption(ESockOptions opt, Int_t val)
{
   // Set socket options.

   return fSetupServer->SetOption(opt, val);
}

//______________________________________________________________________________
Int_t TPServerSocket::GetOption(ESockOptions opt, Int_t &val)
{
   // Get socket options. Returns -1 in case of error.

   return fSetupServer->GetOption(opt, val);
}

//______________________________________________________________________________
Int_t TPServerSocket::GetErrorCode() const
{
   // Returns error code. Meaning depends on context where it is called.
   // If no error condition returns 0 else a value < 0.
   // For example see TPServerSocket ctor.

   return fSetupServer->GetErrorCode();
}
