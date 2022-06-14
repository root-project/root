// @(#)root/net:$Id$
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
// done via the TSystem class (either TUnixSystem or TWinNTSystem).     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPServerSocket.h"
#include "TPSocket.h"
#include "TROOT.h"
#include "TVirtualMutex.h"

ClassImp(TPServerSocket);

////////////////////////////////////////////////////////////////////////////////
/// Create a parallel server socket object on a specified port. Set reuse
/// to true to force reuse of the server socket (i.e. do not wait for the
/// time out to pass). Using backlog one can set the desirable queue length
/// for pending connections.
/// Use tcpwindowsize to specify the size of the receive buffer, it has
/// to be specified here to make sure the window scale option is set (for
/// tcpwindowsize > 65KB and for platforms supporting window scaling).
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

TPServerSocket::TPServerSocket(Int_t port, Bool_t reuse, Int_t backlog,
                               Int_t tcpwindowsize) :
   TServerSocket(port, reuse, backlog, tcpwindowsize)
{
   fTcpWindowSize = tcpwindowsize;
   SetName("PServerSocket");
}

////////////////////////////////////////////////////////////////////////////////
/// Create a parallel server socket object for a named service. Set reuse
/// to true to force reuse of the server socket (i.e. do not wait for the
/// time out to pass). Using backlog one can set the desirable queue length
/// for pending connections.
/// Use tcpwindowsize to specify the size of the receive buffer, it has
/// to be specified here to make sure the window scale option is set (for
/// tcpwindowsize > 65KB and for platforms supporting window scaling).
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

TPServerSocket::TPServerSocket(const char *service, Bool_t reuse, Int_t backlog,
                               Int_t tcpwindowsize) :
   TServerSocket(service, reuse, backlog, tcpwindowsize)
{
   fTcpWindowSize = tcpwindowsize;
   SetName("PServerSocket");
}

////////////////////////////////////////////////////////////////////////////////
/// Accept a connection on a parallel server socket. Returns a full-duplex
/// parallel communication TPSocket object. If no pending connections are
/// present on the queue and nonblocking mode has not been enabled
/// with SetOption(kNoBlock,1) the call blocks until a connection is
/// present. The returned socket must be deleted by the user. The socket
/// is also added to the TROOT sockets list which will make sure that
/// any open sockets are properly closed on program termination.
/// In case of error 0 is returned and in case non-blocking I/O is
/// enabled and no connections are available -1 is returned.

TSocket *TPServerSocket::Accept(UChar_t Opt)
{
   TSocket  *setupSocket = 0;
   TSocket  **pSockets;
   TPSocket *newPSocket = 0;

   Int_t size, port;

   // wait for the incoming connections to the server and accept them
   setupSocket = TServerSocket::Accept(Opt);

   if (setupSocket == 0) return 0;

   // receive the port number and number of parallel sockets from the
   // client and establish 'n' connections
   if (setupSocket->Recv(port, size) < 0) {
      Error("Accept", "error receiving port number and number of sockets");
      return 0;
   }

   // Check if client is running in single mode
   if (size == 0) {
      pSockets = new TSocket*[1];

      pSockets[0] = setupSocket;

      // create TPSocket object with the original socket
      newPSocket = new TPSocket(pSockets, 1);

   } else {
      pSockets = new TSocket*[size];

      for (int i = 0; i < size; i++) {
         pSockets[i] = new TSocket(setupSocket->GetInetAddress(),
                                           port, fTcpWindowSize);
         R__LOCKGUARD(gROOTMutex);
         gROOT->GetListOfSockets()->Remove(pSockets[i]);
      }

      // create TPSocket object with all the accepted sockets
      newPSocket = new TPSocket(pSockets, size);

   }

   // Transmit authentication information, if any
   if (setupSocket->IsAuthenticated())
      newPSocket->SetSecContext(setupSocket->GetSecContext());

   // clean up, if needed
   if (size > 0)
      delete setupSocket;

   // return the TSocket object
   return newPSocket;
}

