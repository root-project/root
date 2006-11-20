// @(#)root/net:$Name:  $:$Id: TSocket.cxx,v 1.39 2006/05/16 06:28:23 brun Exp $
// Author: Fons Rademakers   18/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSocket                                                              //
//                                                                      //
// This class implements client sockets. A socket is an endpoint for    //
// communication between two machines.                                  //
// The actual work is done via the TSystem class (either TUnixSystem    //
// or TWinNTSystem).                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Bytes.h"
#include "NetErrors.h"
#include "TEnv.h"
#include "TError.h"
#include "TMessage.h"
#include "TPSocket.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TString.h"
#include "TSystem.h"
#include "TUrl.h"
#include "TVirtualAuth.h"
#include "TVirtualMutex.h"

ULong64_t TSocket::fgBytesSent = 0;
ULong64_t TSocket::fgBytesRecv = 0;

//
// Client "protocol changes"
//
// This was in TNetFile and TAuthenticate before, but after the introduction
// of TSocket::CreateAuthSocket the common place for all the clients is TSocket,
// so this seems to be the right place for a version number
//
// 7: added support for ReOpen(), kROOTD_BYE and kROOTD_PROTOCOL2
// 8: added support for update being a create (open stat = 2 and not 1)
// 9: added new authentication features (see README.AUTH)
// 10: added support for authenticated socket via TSocket::CreateAuthSocket(...)
// 11: modified SSH protocol + support for server 'no authentication' mode
// 12: add random tags to avoid reply attacks (password+token)
// 13: authentication re-organization; cleanup in PROOF
// 14: support for SSH authentication via SSH tunnel
// 15: cope with fixes in TUrl::GetFile
// 16: add env setup message exchange
//
Int_t TSocket::fgClientProtocol = 16;  // increase when client protocol changes

TVirtualMutex *gSocketAuthMutex = 0;

ClassImp(TSocket)

//______________________________________________________________________________
TSocket::TSocket(TInetAddress addr, const char *service, Int_t tcpwindowsize)
         : TNamed(addr.GetHostName(), service)
{
   // Create a socket. Connect to the named service at address addr.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   R__ASSERT(gROOT);
   R__ASSERT(gSystem);

   fService = service;
   fSecContext = 0;
   fRemoteProtocol= -1;
   fServType = kSOCKD;
   if (fService.Contains("root"))
      fServType = kROOTD;
   if (fService.Contains("proof"))
      fServType = kPROOFD;
   fAddress = addr;
   fAddress.fPort = gSystem->GetServiceByName(service);
   fBytesSent = 0;
   fBytesRecv = 0;
   fCompress  = 0;
   fTcpWindowSize = tcpwindowsize;

   if (fAddress.GetPort() != -1) {
      fSocket = gSystem->OpenConnection(addr.GetHostName(), fAddress.GetPort(),
                                        tcpwindowsize);

      if (fSocket != -1) {
         R__LOCKGUARD2(gROOTMutex);
         gROOT->GetListOfSockets()->Add(this);
      }
   } else
      fSocket = -1;

}

//______________________________________________________________________________
TSocket::TSocket(TInetAddress addr, Int_t port, Int_t tcpwindowsize)
         : TNamed(addr.GetHostName(), "")
{
   // Create a socket. Connect to the specified port # at address addr.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   R__ASSERT(gROOT);
   R__ASSERT(gSystem);

   fService = gSystem->GetServiceByPort(port);
   fSecContext = 0;
   fRemoteProtocol= -1;
   fServType = kSOCKD;
   if (fService.Contains("root"))
      fServType = kROOTD;
   if (fService.Contains("proof"))
      fServType = kPROOFD;
   fAddress = addr;
   fAddress.fPort = port;
   SetTitle(fService);
   fBytesSent = 0;
   fBytesRecv = 0;
   fCompress  = 0;
   fTcpWindowSize = tcpwindowsize;

   fSocket = gSystem->OpenConnection(addr.GetHostName(), fAddress.GetPort(),
                                     tcpwindowsize);
   if (fSocket == -1)
      fAddress.fPort = -1;
   else {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Add(this);
   }
}

//______________________________________________________________________________
TSocket::TSocket(const char *host, const char *service, Int_t tcpwindowsize)
         : TNamed(host, service)
{
   // Create a socket. Connect to named service on the remote host.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   R__ASSERT(gROOT);
   R__ASSERT(gSystem);

   fService = service;
   fSecContext = 0;
   fRemoteProtocol= -1;
   fServType = kSOCKD;
   if (fService.Contains("root"))
      fServType = kROOTD;
   if (fService.Contains("proof"))
      fServType = kPROOFD;
   fAddress = gSystem->GetHostByName(host);
   fAddress.fPort = gSystem->GetServiceByName(service);
   SetName(fAddress.GetHostName());
   fBytesSent = 0;
   fBytesRecv = 0;
   fCompress  = 0;
   fTcpWindowSize = tcpwindowsize;

   if (fAddress.GetPort() != -1) {
      fSocket = gSystem->OpenConnection(host, fAddress.GetPort(), tcpwindowsize);
      if (fSocket != -1) {
         R__LOCKGUARD2(gROOTMutex);
         gROOT->GetListOfSockets()->Add(this);
      }
   } else
      fSocket = -1;
}

//______________________________________________________________________________
TSocket::TSocket(const char *url, Int_t port, Int_t tcpwindowsize)
         : TNamed(TUrl(url).GetHost(), "")
{
   // Create a socket; see CreateAuthSocket for the form of url.
   // Connect to the specified port # on the remote host.
   // If user is specified in url, try authentication as user.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   R__ASSERT(gROOT);
   R__ASSERT(gSystem);

   fUrl = TString(url);
   TString host(TUrl(fUrl).GetHost());

   fService = gSystem->GetServiceByPort(port);
   fSecContext = 0;
   fRemoteProtocol= -1;
   fServType = kSOCKD;
   if (fUrl.Contains("root"))
      fServType = kROOTD;
   if (fUrl.Contains("proof"))
      fServType = kPROOFD;
   fAddress = gSystem->GetHostByName(host);
   fAddress.fPort = port;
   SetName(fAddress.GetHostName());
   SetTitle(fService);
   fBytesSent = 0;
   fBytesRecv = 0;
   fCompress  = 0;
   fTcpWindowSize = tcpwindowsize;

   fSocket = gSystem->OpenConnection(host, fAddress.GetPort(), tcpwindowsize);
   if (fSocket == -1) {
      fAddress.fPort = -1;
   } else {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Add(this);
   }
}

//______________________________________________________________________________
TSocket::TSocket(Int_t desc) : TNamed("", "")
{
   // Create a socket. The socket will use descriptor desc.

   R__ASSERT(gROOT);
   R__ASSERT(gSystem);

   fSecContext = 0;
   fRemoteProtocol= 0;
   fService = (char *)kSOCKD;
   fBytesSent = 0;
   fBytesRecv = 0;
   fCompress  = 0;

   if (desc >= 0) {
      fSocket  = desc;
      fAddress = gSystem->GetPeerName(fSocket);
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Add(this);
   } else
      fSocket = -1;
}

//______________________________________________________________________________
TSocket::TSocket(const TSocket &s) : TNamed(s)
{
   // TSocket copy ctor.

   fSocket         = s.fSocket;
   fService        = s.fService;
   fAddress        = s.fAddress;
   fLocalAddress   = s.fLocalAddress;
   fBytesSent      = s.fBytesSent;
   fBytesRecv      = s.fBytesRecv;
   fCompress       = s.fCompress;
   fSecContext     = s.fSecContext;
   fRemoteProtocol = s.fRemoteProtocol;
   fServType       = s.fServType;

   if (fSocket != -1) {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Add(this);
   }
}

//______________________________________________________________________________
void TSocket::Close(Option_t *option)
{
   // Close the socket. If option is "force", calls shutdown(id,2) to
   // shut down the connection. This will close the connection also
   // for the parent of this process. Also called via the dtor (without
   // option "force", call explicitely Close("force") if this is desired).

   Bool_t force = option ? (!strcmp(option, "force") ? kTRUE : kFALSE) : kFALSE;

   if (fSocket != -1) {
      gSystem->CloseConnection(fSocket, force);
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfSockets()->Remove(this);
   }
   fSocket = -1;
}

//______________________________________________________________________________
TInetAddress TSocket::GetLocalInetAddress()
{
   // Return internet address of local host to which the socket is bound.
   // In case of error TInetAddress::IsValid() returns kFALSE.

   if (IsValid()) {
      if (fLocalAddress.GetPort() == -1)
         fLocalAddress = gSystem->GetSockName(fSocket);
      return fLocalAddress;
   }
   return TInetAddress();
}

//______________________________________________________________________________
Int_t TSocket::GetLocalPort()
{
   // Return the local port # to which the socket is bound.
   // In case of error return -1.

   if (IsValid()) {
      if (fLocalAddress.GetPort() == -1)
         GetLocalInetAddress();
      return fLocalAddress.GetPort();
   }
   return -1;
}

//______________________________________________________________________________
Int_t TSocket::Select(Int_t interest, Long_t timeout)
{
   // Waits for this socket to change status. If interest=kRead,
   // the socket will be watched to see if characters become available for
   // reading; if interest=kWrite the socket will be watched to
   // see if a write will not block.
   // The argument 'timeout' specifies a maximum time to wait in millisec.
   // Default no timeout.
   // Returns 1 if a change of status of interest has been detected within
   // timeout; 0 in case of timeout; < 0 if an error occured.

   Int_t rc = 1;

   // Associate a TFileHandler to this socket
   TFileHandler fh(fSocket, interest);

   // Wait for an event now
   rc = gSystem->Select(&fh, timeout);

   return rc;
}

//______________________________________________________________________________
Int_t TSocket::Send(Int_t kind)
{
   // Send a single message opcode. Use kind (opcode) to set the
   // TMessage "what" field. Returns the number of bytes that were sent
   // (always sizeof(Int_t)) and -1 in case of error. In case the kind has
   // been or'ed with kMESS_ACK, the call will only return after having
   // received an acknowledgement, making the sending process synchronous.

   TMessage mess(kind);

   Int_t nsent;
   if ((nsent = Send(mess)) < 0)
      return -1;

   return nsent;
}

//______________________________________________________________________________
Int_t TSocket::Send(Int_t status, Int_t kind)
{
   // Send a status and a single message opcode. Use kind (opcode) to set the
   // TMessage "what" field. Returns the number of bytes that were sent
   // (always 2*sizeof(Int_t)) and -1 in case of error. In case the kind has
   // been or'ed with kMESS_ACK, the call will only return after having
   // received an acknowledgement, making the sending process synchronous.

   TMessage mess(kind);
   mess << status;

   Int_t nsent;
   if ((nsent = Send(mess)) < 0)
      return -1;

   return nsent;
}

//______________________________________________________________________________
Int_t TSocket::Send(const char *str, Int_t kind)
{
   // Send a character string buffer. Use kind to set the TMessage "what" field.
   // Returns the number of bytes in the string str that were sent and -1 in
   // case of error. In case the kind has been or'ed with kMESS_ACK, the call
   // will only return after having received an acknowledgement, making the
   // sending process synchronous.

   TMessage mess(kind);
   if (str) mess.WriteString(str);

   Int_t nsent;
   if ((nsent = Send(mess)) < 0)
      return -1;

   return nsent - sizeof(Int_t);    // - TMessage::What()
}

//______________________________________________________________________________
Int_t TSocket::Send(const TMessage &mess)
{
   // Send a TMessage object. Returns the number of bytes in the TMessage
   // that were sent and -1 in case of error. In case the TMessage::What
   // has been or'ed with kMESS_ACK, the call will only return after having
   // received an acknowledgement, making the sending process synchronous.
   // Returns -4 in case of kNoBlock and errno == EWOULDBLOCK.
   // Returns -5 if pipe broken or reset by peer (EPIPE || ECONNRESET).

   TSystem::ResetErrno();

   if (fSocket == -1) return -1;

   if (mess.IsReading()) {
      Error("Send", "cannot send a message used for reading");
      return -1;
   }

   mess.SetLength();   //write length in first word of buffer

   if (fCompress > 0 && mess.GetCompressionLevel() == 0)
      const_cast<TMessage&>(mess).SetCompressionLevel(fCompress);

   if (mess.GetCompressionLevel() > 0)
      const_cast<TMessage&>(mess).Compress();

   char *mbuf = mess.Buffer();
   Int_t mlen = mess.Length();
   if (mess.CompBuffer()) {
      mbuf = mess.CompBuffer();
      mlen = mess.CompLength();
   }

   Int_t nsent;
   if ((nsent = gSystem->SendRaw(fSocket, mbuf, mlen, 0)) <= 0) {
      if (nsent == -5) {
         // Connection reset by peer or broken
         Close();
      }
      return nsent;
   }

   fBytesSent  += nsent;
   fgBytesSent += nsent;

   // If acknowledgement is desired, wait for it
   if (mess.What() & kMESS_ACK) {
      TSystem::ResetErrno();
      char buf[2];
      Int_t n = 0;
      if ((n = gSystem->RecvRaw(fSocket, buf, sizeof(buf), 0)) < 0) {
         if (n == -5) {
            // Connection reset by peer or broken
            Close();
         } else
            n = -1;
         return n;
      }
      if (strncmp(buf, "ok", 2)) {
         Error("Send", "bad acknowledgement");
         return -1;
      }
      fBytesRecv  += 2;
      fgBytesRecv += 2;
   }

   return nsent - sizeof(UInt_t);  //length - length header
}

//______________________________________________________________________________
Int_t TSocket::SendObject(const TObject *obj, Int_t kind)
{
   // Send an object. Returns the number of bytes sent and -1 in case of error.
   // In case the "kind" has been or'ed with kMESS_ACK, the call will only
   // return after having received an acknowledgement, making the sending
   // synchronous.

   TMessage mess(kind);
   mess.WriteObject(obj);

   Int_t nsent;
   if ((nsent = Send(mess)) < 0)
      return -1;

   return nsent;
}

//______________________________________________________________________________
Int_t TSocket::SendRaw(const void *buffer, Int_t length, ESendRecvOptions opt)
{
   // Send a raw buffer of specified length. Using option kOob one can send
   // OOB data. Returns the number of bytes sent or -1 in case of error.
   // Returns -4 in case of kNoBlock and errno == EWOULDBLOCK.
   // Returns -5 if pipe broken or reset by peer (EPIPE || ECONNRESET).

   TSystem::ResetErrno();

   if (fSocket == -1) return -1;

   Int_t nsent;

   if ((nsent = gSystem->SendRaw(fSocket, buffer, length, (int) opt)) <= 0) {
      if (nsent == -5) {
         // Connection reset or broken: close
         Close();
      }
      return nsent;
   }

   fBytesSent  += nsent;
   fgBytesSent += nsent;

   return nsent;
}

//______________________________________________________________________________
Int_t TSocket::Recv(char *str, Int_t max)
{
   // Receive a character string message of maximum max length. The expected
   // message must be of type kMESS_STRING. Returns length of received string
   // (can be 0 if otherside of connection is closed) or -1 in case of error
   // or -4 in case a non-blocking socket would block (i.e. there is nothing
   // to be read).

   Int_t n, kind;

   if ((n = Recv(str, max, kind)) <= 0) {
      if (n == -5)
         n = -1;
      return n;
   }

   if (kind != kMESS_STRING) {
      Error("Recv", "got message of wrong kind (expected %d, got %d)",
            kMESS_STRING, kind);
      return -1;
   }

   return n;
}

//______________________________________________________________________________
Int_t TSocket::Recv(char *str, Int_t max, Int_t &kind)
{
   // Receive a character string message of maximum max length. Returns in
   // kind the message type. Returns length of received string+4 (can be 0 if
   // other side of connection is closed) or -1 in case of error or -4 in
   // case a non-blocking socket would block (i.e. there is nothing to be read).

   Int_t     n;
   TMessage *mess;

   if ((n = Recv(mess)) <= 0) {
      if (n == -5)
         n = -1;
      return n;
   }

   kind = mess->What();
   if (str) {
      if (mess->BufferSize() > (Int_t)sizeof(Int_t)) // if mess contains more than kind
         mess->ReadString(str, max);
      else
         str[0] = 0;
   }
   delete mess;

   return n;   // number of bytes read (len of str + sizeof(kind)
}

//______________________________________________________________________________
Int_t TSocket::Recv(Int_t &status, Int_t &kind)
{
   // Receives a status and a message type. Returns length of received
   // integers, 2*sizeof(Int_t) (can be 0 if other side of connection
   // is closed) or -1 in case of error or -4 in case a non-blocking
   // socket would block (i.e. there is nothing to be read).

   Int_t     n;
   TMessage *mess;

   if ((n = Recv(mess)) <= 0) {
      if (n == -5)
         n = -1;
      return n;
   }

   kind = mess->What();
   (*mess) >> status;

   delete mess;

   return n;   // number of bytes read (2 * sizeof(Int_t)
}

//______________________________________________________________________________
Int_t TSocket::Recv(TMessage *&mess)
{
   // Receive a TMessage object. The user must delete the TMessage object.
   // Returns length of message in bytes (can be 0 if other side of connection
   // is closed) or -1 in case of error or -4 in case a non-blocking socket
   // would block (i.e. there is nothing to be read) or -5 if pipe broken
   // or reset by peer (EPIPE || ECONNRESET). In those case mess == 0.

   TSystem::ResetErrno();

   if (fSocket == -1) {
      mess = 0;
      return -1;
   }

   Int_t  n;
   UInt_t len;
   if ((n = gSystem->RecvRaw(fSocket, &len, sizeof(UInt_t), 0)) <= 0) {
      if (n == -5) {
         // Connection reset or broken
         Close();
      }
      mess = 0;
      return n;
   }
   len = net2host(len);  //from network to host byte order

   char *buf = new char[len+sizeof(UInt_t)];
   if ((n = gSystem->RecvRaw(fSocket, buf+sizeof(UInt_t), len, 0)) <= 0) {
      if (n == -5) {
         // Connection reset or broken
         Close();
      }
      delete [] buf;
      mess = 0;
      return n;
   }

   fBytesRecv  += n + sizeof(UInt_t);
   fgBytesRecv += n + sizeof(UInt_t);

   mess = new TMessage(buf, len+sizeof(UInt_t));

   if (mess->What() & kMESS_ACK) {
      char ok[2] = { 'o', 'k' };
      Int_t n = 0;
      if ((n = gSystem->SendRaw(fSocket, ok, sizeof(ok), 0)) < 0) {
         if (n == -5) {
            // Connection reset or broken
            Close();
         }
         delete mess;
         mess = 0;
         return n;
      }
      mess->SetWhat(mess->What() & ~kMESS_ACK);

      fBytesSent  += 2;
      fgBytesSent += 2;
   }

   return n;
}

//______________________________________________________________________________
Int_t TSocket::RecvRaw(void *buffer, Int_t length, ESendRecvOptions opt)
{
   // Receive a raw buffer of specified length bytes. Using option kPeek
   // one can peek at incoming data. Returns -1 in case of error. In case
   // of opt == kOob: -2 means EWOULDBLOCK and -3 EINVAL. In case of non-blocking
   // mode (kNoBlock) -4 means EWOULDBLOCK. Returns -5 if pipe broken or
   // reset by peer (EPIPE || ECONNRESET).

   TSystem::ResetErrno();

   if (fSocket == -1) return -1;

   Int_t n;

   if ((n = gSystem->RecvRaw(fSocket, buffer, length, (int) opt)) <= 0) {
      if (n == -5) {
         // Connection reset or broken
         Close();
      }
      return n;
   }

   fBytesRecv  += n;
   fgBytesRecv += n;

   return n;
}

//______________________________________________________________________________
Int_t TSocket::SetOption(ESockOptions opt, Int_t val)
{
   // Set socket options.

   if (fSocket == -1) return -1;

   return gSystem->SetSockOpt(fSocket, opt, val);
}

//______________________________________________________________________________
Int_t TSocket::GetOption(ESockOptions opt, Int_t &val)
{
   // Get socket options. Returns -1 in case of error.

   if (fSocket == -1) return -1;

   return gSystem->GetSockOpt(fSocket, opt, &val);
}

//______________________________________________________________________________
Int_t TSocket::GetErrorCode() const
{
   // Returns error code. Meaning depends on context where it is called.
   // If no error condition returns 0 else a value < 0.
   // For example see TServerSocket ctor.

   if (!IsValid())
      return fSocket;

   return 0;
}

//______________________________________________________________________________
void TSocket::SetCompressionLevel(Int_t level)
{
   // Set the message compression level. Can be between 0 and 9 with 0
   // being no compression and 9 maximum compression. In general the default
   // level of 1 is the best compromise between achieved compression and
   // cpu time. Compression will only happen when the message is > 256 bytes.

   if (level < 0) level = 0;
   if (level > 9) level = 9;

   fCompress = level;
}

//______________________________________________________________________________
Bool_t TSocket::Authenticate(const char *user)
{
   // Authenticated the socket with specified user.

   Bool_t rc = kFALSE;

   // Parse protocol name, for PROOF, send message with server role
   TString sproto = TUrl(fUrl).GetProtocol();
   if (sproto.Contains("sockd")) {
      fServType = kSOCKD;
   } else if (sproto.Contains("rootd")) {
      fServType = kROOTD;
   } else if (sproto.Contains("proofd")) {
      fServType = kPROOFD;
      // Parse options
      TString opt(TUrl(fUrl).GetOptions());
      //First letter in Opt describes type of proofserv to invoke
      if (!strncasecmp(opt, "S", 1)) {
         Send("slave");
      } else if (!strncasecmp(opt, "M", 1)) {
         Send("master");
      } else {
         Warning("Authenticate",
                 "called by TSlave: unknown option '%c' %s",
                 opt[0], " - assuming Slave");
         Send("slave");
      }
   }
   if (gDebug > 2)
      Info("Authenticate","Local protocol: %s",sproto.Data());

   // Get server protocol level
   Int_t kind = kROOTD_PROTOCOL;
   // Warning: for backward compatibility reasons here we have to
   // send exactly 4 bytes: for fgClientClientProtocol > 99
   // the space in the format must be dropped
   if (fRemoteProtocol == -1) {
      Send(Form(" %d", fgClientProtocol), kROOTD_PROTOCOL);
      Recv(fRemoteProtocol, kind);
      //
      // If we are talking to an old rootd server we get a fatal
      // error here and we need to reopen the connection,
      // communicating first the size of the parallel socket
      if (kind == kROOTD_ERR) {
         fRemoteProtocol = 9;
         return kFALSE;
      }
   }

   // Find out whether authentication is required
   Bool_t runauth = kTRUE;
   if (fRemoteProtocol > 1000) {
      // Authentication not required by the remote server
      runauth = kFALSE;
      fRemoteProtocol %= 1000;
   }

   // If authentication is required, we need to find out which library
   // has to be loaded (preparation for near future, 9/7/05)
   TString host = GetInetAddress().GetHostName();
   if (runauth) {

      // Default (future)
      TString alib = "Xrd";
      if (fRemoteProtocol < 100) {
         // Standard Authentication lib
         alib = "Root";
      }

      // Load the plugin
      TPluginHandler *h =
         gROOT->GetPluginManager()->FindHandler("TVirtualAuth", alib);
      if (!h || h->LoadPlugin() != 0) {
         Error("Authenticate",
               "could not load properly %s authentication plugin", alib.Data());
         return rc;
      }

      // Get an instance of the interface class
      TVirtualAuth *auth = (TVirtualAuth *)(h->ExecPlugin(0));
      if (!auth) {
         Error("Authenticate", "could not instantiate the interface class");
         return rc;
      }
      if (gDebug > 1)
         Info("Authenticate", "class for '%s' authentication loaded", alib.Data());

      Option_t *opts = (gROOT->IsProofServ()) ? "P" : "";
      if (!(auth->Authenticate(this, host, user, opts))) {
         Error("Authenticate",
               "authentication attempt failed for %s@%s", user, host.Data());
      } else {
         rc = kTRUE;
      }
   } else {

      // Communicate who we are and our target user
      UserGroup_t *u = gSystem->GetUserInfo();
      if (u) {
         Send(Form("%s %s", u->fUser.Data(), user), kROOTD_USER);
         delete u;
      } else
         Send(Form("-1 %s", user), kROOTD_USER);

      rc = kFALSE;

      // Receive confirmation that everything went well
      Int_t kind, stat;
      if (Recv(stat, kind) > 0) {

         if (kind == kROOTD_ERR) {
            if (gDebug > 0)
               TSocket::NetError("TSocket::Authenticate", stat);
         } else if (kind == kROOTD_AUTH) {

            // Authentication was not required: create inactive
            // security context for consistency
            fSecContext = new TSecContext(user, host, 0, -4, 0, 0);
            if (gDebug > 3)
               Info("Authenticate", "no authentication required remotely");

            // Set return flag;
            rc = 1;
         } else {
            if (gDebug > 0)
               Info("Authenticate", "expected message type %d, received %d",
                    kROOTD_AUTH, kind);
         }
      } else {
         if (gDebug > 0)
            Info("Authenticate", "error receiving message");
      }

   }

   return rc;
}

//______________________________________________________________________________
TSocket *TSocket::CreateAuthSocket(const char *url, Int_t size,
                                   Int_t tcpwindowsize, TSocket *opensock)
{
   // Creates a socket or a parallel socket and authenticates to the
   // remote server.
   //
   // url: [[proto][p][auth]://][user@]host[:port][/service][?options]
   //
   // where  proto = "sockd", "rootd", "proofd"
   //                indicates the type of remote server;
   //                if missing "sockd" is assumed ("sockd" indicates
   //                any remote server session using TServerSocket)
   //          [p] = for parallel sockets (forced internally for
   //                rootd; ignored for proofd)
   //       [auth] = "up", "s", "k", "g", "h", "ug" to force UsrPwd,
   //                SRP, Krb5, Globus, SSH or UidGid authentication
   //       [port] = is the remote port number
   //    [service] = service name used to determine the port
   //                (for backward compatibility, specification of
   //                 port as priority)
   //     options  = "m" or "s", when proto=proofd indicates whether
   //                we are master or slave (used internally by
   //                TSlave)
   //
   // An already opened connection can be used by passing its socket
   // in opensock.
   //
   // Example:
   //
   //   TSocket::CreateAuthSocket("rootds://qwerty@machine.fq.dn:5051")
   //
   //   creates an authenticated socket to a rootd server running
   //   on remote machine machine.fq.dn on port 5051; "parallel" sockets
   //   are forced internally because rootd expects
   //   parallel sockets; however a simple socket will be created
   //   in this case because the size is 0 (the default);
   //   authentication will attempt protocol SRP first.
   //
   //   TSocket::CreateAuthSocket("pk://qwerty@machine.fq.dn:5052",3)
   //
   //   creates an authenticated parallel socket of size 3 to a sockd
   //   server running on remote machine machine.fq.dn on port 5052;
   //   authentication will attempt protocol Kerberos first.
   //
   // NB: may hang if the remote server is not of the correct type;
   //     at present TSocket has no way to find out the type of the
   //     remote server automatically
   //
   // Returns pointer to an authenticated socket or 0 if creation or
   // authentication is unsuccessful.

   R__LOCKGUARD2(gSocketAuthMutex);

   // Url to be passed to choosen constructor
   TString eurl(url);

   // Parse protocol, if any
   Bool_t parallel = kFALSE;
   TString proto(TUrl(url).GetProtocol());
   TString protosave = proto;

   // Get rid of authentication suffix
   TString asfx = "";
   if (proto.EndsWith("up") || proto.EndsWith("ug")) {
      asfx = proto;
      asfx.Remove(0,proto.Length()-2);
      proto.Resize(proto.Length()-2);
   } else if (proto.EndsWith("s") || proto.EndsWith("k") ||
              proto.EndsWith("g") || proto.EndsWith("h")) {
      asfx = proto;
      asfx.Remove(0,proto.Length()-1);
      proto.Resize(proto.Length()-1);
   }

   // Find out if parallel (ignore if proofd, force if rootd)
   if (((proto.EndsWith("p") || size > 1) &&
               !proto.BeginsWith("proof")) ||
         proto.BeginsWith("root") ) {
      parallel = kTRUE;
      if (proto.EndsWith("p"))
         proto.Resize(proto.Length()-1);
   }

   // Force "sockd" if the rest is not recognized
   if (!proto.BeginsWith("sock") && !proto.BeginsWith("proof") &&
       !proto.BeginsWith("root"))
      proto = "sockd";

   // Substitute this for original proto in eurl
   protosave += "://";
   proto += asfx;
   proto += "://";
   eurl.ReplaceAll(protosave,proto);

   // Create the socket now

   TSocket *sock = 0;
   if (!parallel) {

      // Simple socket
      if (opensock && opensock->IsValid())
         sock = opensock;
      else
         sock = new TSocket(eurl, TUrl(url).GetPort(), tcpwindowsize);

      // Authenticate now
      if (sock && sock->IsValid()) {
         if (!sock->Authenticate(TUrl(url).GetUser())) {
            sock->Close();
            delete sock;
            sock = 0;
         }
      }

   } else {

      // Tell TPSocket that we want authentication, which has to
      // be done using the original socket before creation of set
      // of parallel sockets
      if (eurl.Contains("?"))
         eurl.Resize(eurl.Index("?"));
      eurl += "?A";

      // Parallel socket
      if (opensock && opensock->IsValid())
         sock = new TPSocket(eurl, TUrl(url).GetPort(), size, opensock);
      else
         sock = new TPSocket(eurl, TUrl(url).GetPort(), size, tcpwindowsize);

      // Cleanup if failure ...
      if (sock && !sock->IsAuthenticated()) {
         // Nothing to do except setting sock to NULL
         if (sock->IsValid())
            // And except when the sock is valid; this typically
            // happens when talking to a old server, because the
            // the parallel socket system is open before authentication
            delete sock;
         sock = 0;
      }
   }

   return sock;
}

//______________________________________________________________________________
TSocket *TSocket::CreateAuthSocket(const char *user, const char *url,
                                   Int_t port, Int_t size, Int_t tcpwindowsize,
                                   TSocket *opensock)
{
   // Creates a socket or a parallel socket and authenticates to the
   // remote server specified in 'url' on remote 'port' as 'user'.
   //
   // url: [[proto][p][auth]://]host[/?options]
   //
   // where  proto = "sockd", "rootd", "proofd"
   //                indicates the type of remote server
   //                if missing "sockd" is assumed ("sockd" indicates
   //                any remote server session using TServerSocket)
   //          [p] = for parallel sockets (forced internally for
   //                rootd)
   //       [auth] = "up", "s", "k", "g", "h", "ug" to force UsrPwd,
   //                SRP, Krb5, Globus, SSH or UidGid authentication
   //    [options] = "m" or "s", when proto=proofd indicates whether
   //                we are master or slave (used internally by TSlave)
   //
   // An already opened connection can be used by passing its socket
   // in opensock.
   //
   // Example:
   //
   //   TSocket::CreateAuthSocket("qwerty","rootdps://machine.fq.dn",5051)
   //
   //   creates an authenticated socket to a rootd server running
   //   on remote machine machine.fq.dn on port 5051; "parallel"
   //   sockets are forced internally because rootd expects
   //   parallel sockets; however a simple socket will be created
   //   in this case because the size is 0 (the default);
   //   authentication will attempt protocol SRP first.
   //
   //   TSocket::CreateAuthSocket("qwerty","pk://machine.fq.dn:5052",3)
   //
   //   creates an authenticated parallel socket of size 3 to a sockd
   //   server running on remote machine machine.fq.dn on port 5052;
   //   authentication will attempt protocol Kerberos first.
   //
   // NB: may hang if the remote server is not of the correct type;
   //     at present TSocket has no way to find out the type of the
   //     remote server automatically
   //
   // Returns pointer to an authenticated socket or 0 if creation or
   // authentication is unsuccessful.

   R__LOCKGUARD2(gSocketAuthMutex);

   // Extended url to be passed to base call
   TString eurl;

   // Add protocol, if any
   if (TString(TUrl(url).GetProtocol()).Length() > 0) {
      eurl += TString(TUrl(url).GetProtocol());
      eurl += TString("://");
   }
   // Add user, if any
   if (!user || strlen(user) > 0) {
      eurl += TString(user);
      eurl += TString("@");
   }
   // Add host
   eurl += TString(TUrl(url).GetHost());
   // Add port
   eurl += TString(":");
   eurl += (port > 0 ? port : 0);
   // Add options, if any
   if (TString(TUrl(url).GetOptions()).Length() > 0) {
      eurl += TString("/?");
      eurl += TString(TUrl(url).GetOptions());
   }

   // Create the socket and return it
   return TSocket::CreateAuthSocket(eurl,size,tcpwindowsize,opensock);
}

//______________________________________________________________________________
Int_t TSocket::GetClientProtocol()
{
   // Static method returning supported client protocol.

   return fgClientProtocol;
}

//______________________________________________________________________________
void TSocket::NetError(const char *where, Int_t err)
{
   // Print error string depending on error code.

   // Make sure it is in range
   err = (err < kErrError) ? ((err > -1) ? err : 0) : kErrError;

   if (gDebug > 0)
      ::Error(where, "%s", gRootdErrStr[err]);
}
