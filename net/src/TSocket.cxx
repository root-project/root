// @(#)root/net:$Name:  $:$Id: TSocket.cxx,v 1.19 2004/05/18 11:56:38 rdm Exp $
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
// The actual work is done via the TSystem class (either TUnixSystem,   //
// TWin32System or TMacSystem).                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAuthenticate.h"
#include "THostAuth.h"
#include "TUrl.h"
#include "TPSocket.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TMessage.h"
#include "Bytes.h"
#include "TROOT.h"
#include "TError.h"

UInt_t TSocket::fgBytesSent = 0;
UInt_t TSocket::fgBytesRecv = 0;


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

   Assert(gROOT);
   Assert(gSystem);

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

   if (fAddress.GetPort() != -1) {
      fSocket = gSystem->OpenConnection(addr.GetHostName(), fAddress.GetPort(),
                                        tcpwindowsize);
      if (fSocket != -1) gROOT->GetListOfSockets()->Add(this);
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

   Assert(gROOT);
   Assert(gSystem);

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

   fSocket = gSystem->OpenConnection(addr.GetHostName(), fAddress.GetPort(),
                                     tcpwindowsize);
   if (fSocket == -1)
      fAddress.fPort = -1;
   else
      gROOT->GetListOfSockets()->Add(this);
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

   Assert(gROOT);
   Assert(gSystem);

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

   if (fAddress.GetPort() != -1) {
      fSocket = gSystem->OpenConnection(host, fAddress.GetPort(), tcpwindowsize);
      if (fSocket != -1)
         gROOT->GetListOfSockets()->Add(this);

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

   Assert(gROOT);
   Assert(gSystem);

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

   fSocket = gSystem->OpenConnection(host, fAddress.GetPort(), tcpwindowsize);
   if (fSocket == -1) {
      fAddress.fPort = -1;
   } else {
      gROOT->GetListOfSockets()->Add(this);
   }
}

//______________________________________________________________________________
TSocket::TSocket(Int_t desc) : TNamed("", "")
{
   // Create a socket. The socket will use descriptor desc.

   Assert(gROOT);
   Assert(gSystem);

   fSecContext = 0;
   fRemoteProtocol= 0;
   fService = (char *)kSOCKD;
   fBytesSent = 0;
   fBytesRecv = 0;
   fCompress  = 0;

   if (desc >= 0) {
      fSocket  = desc;
      fAddress = gSystem->GetPeerName(fSocket);
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

   if (fSocket != -1) gROOT->GetListOfSockets()->Add(this);
}

//______________________________________________________________________________
void TSocket::Close(Option_t *option)
{
   // Close the socket. If option is "force", calls shutdown(id,2) to
   // shut down the connection. This will close the connection also
   // for the parent of this process. Also called via the dtor (without
   // option "force", call explicitely Close("force") if this is desired).

   Bool_t force = option ? (!strcmp(option, "force") ? kTRUE : kFALSE) : kFALSE;

   // deactivate used sec context if talking to proofd daemon running
   // an old protocol (sec context disactivated remotely)
   if (fSecContext && fSecContext->IsActive()) {
      TIter last(fSecContext->GetSecContextCleanup(),kIterBackward);
      TSecContextCleanup *nscc = 0;
      while ((nscc = (TSecContextCleanup *)last())) {
         if (nscc->GetType() == TSocket::kPROOFD &&
             nscc->GetProtocol() < 9) {
            fSecContext->DeActivate("");
            break;
         }
      }
   }

   if (fSocket != -1) {
      gSystem->CloseConnection(fSocket, force);
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
   Bool_t runAuth = kTRUE;

   // Parse protocol name, for PROOF, send message with server role
   Bool_t Master = kFALSE;
   TString SProtocol = TUrl(fUrl).GetProtocol();
   if (SProtocol == "") {
      SProtocol = "root";
   } else if (SProtocol.Contains("rootd")) {
      SProtocol.ReplaceAll("d",1,"",0);
      fServType = kROOTD;
   } else if (SProtocol.Contains("proofd")) {
      SProtocol.ReplaceAll("d",1,"",0);
      TString Opt(TUrl(fUrl).GetOptions());

      if (!strncasecmp(Opt, "M", 1)) {
         Send("slave");
         Master = kTRUE;
      } else if (!strncasecmp(Opt, "C", 1)) {
         Send("master");
      } else {
         Warning("Authenticate",
                 "called by TSlave: unknown option '%s' %s",
                 Opt.Data()," - assuming Master");
         Send("slave");
         Master = kTRUE;
      }
      fServType = kPROOFD;
   }
   if (gDebug > 2)
      Info("Authenticate","Local protocol: %s",SProtocol.Data());

   // Get server protocol level
   Int_t kind;
   // Warning: for backward compatibility reasons here we have to
   // send exactly 4 bytes: for fgClientClientProtocol > 99
   // the space in the format must be dropped
   Send(Form(" %d", TAuthenticate::GetClientProtocol()), kROOTD_PROTOCOL);
   Recv(fRemoteProtocol, kind);

   // If we are talking to an old rootd server we get a fatal
   // error here and we need to reopen the connection,
   // communicating first the size of the parallel socket
   if (kind == kROOTD_ERR) {
      fRemoteProtocol = 9;
      return kFALSE;
   }

   if (fRemoteProtocol > 1000) {
      // Authentication not required by the remote server
      runAuth = kFALSE;
      fRemoteProtocol %= 1000;
   }

   if (fServType == kROOTD) {
      if (fRemoteProtocol > 6 && fRemoteProtocol < 10) {
         // Middle aged versions expect client protocol now
         Send(Form("%d", TAuthenticate::GetClientProtocol()), kROOTD_PROTOCOL2);
         Recv(fRemoteProtocol, kind);
      }
   }

   // Remote Host
   TString Host = GetInetAddress().GetHostName();

   if (runAuth) {

      // Init authentication
      TAuthenticate *auth = new TAuthenticate(this,Host,
                     Form("%s:%d",SProtocol.Data(),fRemoteProtocol), user);

      // If PROOF client and trasmission of the SRP password is
      // requested make sure that ReUse is switched on to get and
      // send also the Public Key
      // Masters do this automatically upon reception of valid info
      // (see TSlave.cxx)
      if (!Master && fServType == kPROOFD) {
         if (gEnv->GetValue("Proofd.SendSRPPwd",0)) {
            Int_t kSRP = TAuthenticate::kSRP;
            TString SRPDets(auth->GetHostAuth()->GetDetails(kSRP));
            Int_t pos = SRPDets.Index("ru:0");
            if (pos > -1) {
               SRPDets.ReplaceAll("ru:0",4,"ru:1",4);
               auth->GetHostAuth()->SetDetails(kSRP,SRPDets);
            } else {
               TSubString ss = SRPDets.SubString("ru:no",TString::kIgnoreCase);
               if (!ss.IsNull()) {
                  SRPDets.ReplaceAll(ss.Data(),5,"ru:1",4);
                  auth->GetHostAuth()->SetDetails(kSRP,SRPDets);
               }
            }
         }
      }

      // Attempt authentication
      if (!auth->Authenticate()) {
         // Close the socket if unsuccessful
         Error("Authenticate",
               "authentication failed for %s@%s",auth->GetUser(),Host.Data());
         // This is to terminate properly remote proofd in case of failure
         if (fServType == kPROOFD)
            Send(Form("%d %s", gSystem->GetPid(), Host.Data()), kROOTD_CLEANUP);
      } else {
         // Set return flag;
         rc = kTRUE;
         // Search pointer to relevant TSecContext
         SetSecContext(auth->GetSecContext());
      }

      delete auth;

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
               TAuthenticate::AuthError("Authenticate(TSocket)", stat);
         } else if (kind == kROOTD_AUTH) {

            // Authentication was not required: create inactive
            // security context for consistency
            TSecContext *ctx = new TSecContext(user, Host,0, -4, 0, 0);
            if (gDebug > 3)
               Info("Authenticate", "no authentication required remotely");

            // Save pointer
            SetSecContext(ctx);

            // Set return flag;
            rc = kTRUE;
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
TSocket *TSocket::CreateAuthSocket(const char *url,
                                   Int_t size, Int_t tcpwindowsize)
{
   // Creates a socket or a parallel socket and authenticates to the
   // remote server.
   //
   // url: [proto[p][auth]://][user@]host[:port][/service][?options]
   //
   // where  proto = "sockd", "rootd", "proofd"
   //                indicates the type of remote server
   //                ("sockd" not operational yet)
   //          [p] = for parallel sockets (forced internally for
   //                rootd)
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
   // Example:
   //
   //   TSocket::CreateAuthSocket("rootds://qwerty@machine.fq.dn:5051")
   //
   //   creates an authenticated socket to a rootd server running
   //   on remote machine machine.fq.dn on port 5051; "parallel" sockets
   //   are forced internally because rootd expects
   //   parallel sockets; however a simple socket will be created
   //   in this case because the size is 0 (the default).
   //
   // Returns pointer to an authenticated socket or 0 if creation or
   // authentication is unsuccessful.

   // Url to be passed to choosen constructor
   TString eurl(url);

   // Check if parallel
   Bool_t parallel = kFALSE;
   TString proto(TUrl(url).GetProtocol());
   if (proto.Contains("sockd")) {
      if (proto.Index("dp",1) > 1 || size > 1)
         parallel = kTRUE;
      eurl.ReplaceAll("dp",2,"d",1);
   }
   if (proto.Contains("rootd")) {
      parallel = kTRUE;
      eurl.ReplaceAll("dp",2,"d",1);
   }

   // Create the socket now

   TSocket *sock = 0;
   if (!parallel) {

      // Simple socket
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
                                   Int_t port, Int_t size, Int_t tcpwindowsize)
{
   // Creates a socket or a parallel socket and authenticates to the
   // remote server specified in 'url' on remote 'port' as 'user'.
   //
   // url: [proto[p][auth]://]host[/?options]
   //
   // where  proto = "sockd", "rootd", "proofd"
   //                indicates the type of remote server
   //                ("sockd" not operational yet)
   //          [p] = for parallel sockets (forced internally for
   //                rootd)
   //       [auth] = "up", "s", "k", "g", "h", "ug" to force UsrPwd,
   //                SRP, Krb5, Globus, SSH or UidGid authentication
   //    [options] = "m" or "s", when proto=proofd indicates whether
   //                we are master or slave (used internally by TSlave)
   //
   // Example:
   //
   //   TSocket::CreateAuthSocket("qwerty","rootdps://machine.fq.dn",5051)
   //
   //   creates an authenticated socket to a rootd server running
   //   on remote machine machine.fq.dn on port 5051, authenticating
   //   as 'qwerty'; "parallel" sockets are forced internally because
   //   rootd expects parallel sockets; however a simple socket will
   //   be created in this case because the size is 0 (the default).
   //
   // Returns pointer to an authenticated socket or 0 if creation or
   // authentication is unsuccessful.

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
   return TSocket::CreateAuthSocket(eurl,size,tcpwindowsize);
}

//______________________________________________________________________________
Int_t TSocket::SecureSend(const char *in, Int_t keyType)
{
   // If authenticated and related SecContext is active
   // secure-sends "in" to host using RSA "keyType" stored in TAuthenticate.
   // Returns # bytes send or -1 in case of error.

   if (IsAuthenticated() && fSecContext->IsActive())
      return TAuthenticate::SecureSend(this, keyType, in);
   return -1;
}

//______________________________________________________________________________
Int_t TSocket::SendHostAuth()
{
   // Sends the list of the relevant THostAuth objects to the master or
   // to the active slaves, typically data servers external to the proof
   // cluster. The list is of THostAuth to be sent is specified by
   // TAuthenticate::fgProofAuthInfo after directives found in the
   // .rootauthrc family files ('proofserv' key)
   // Returns -1 if a problem sending THostAuth has occured, -2 in case
   // of problems closing the transmission.

   Int_t retval = 0, ns = 0;

   TIter next(TAuthenticate::GetProofAuthInfo());
   THostAuth *ha;
   while ((ha = (THostAuth *)next())) {
      TString Buf;
      ha->AsString(Buf);
      if((ns = Send(Buf, kPROOF_SENDHOSTAUTH)) < 1) {
         retval = -1;
         break;
      }
      if (gDebug > 2)
         Info("SendHostAuth","sent %d bytes (%s)",ns,Buf.Data());
   }

   // End of transmission ...
   if ((ns = Send("END", kPROOF_SENDHOSTAUTH)) < 1)
      retval = -2;
   if (gDebug > 2)
      Info("SendHostAuth","sent %d bytes for closing",ns);

   return retval;
}

//______________________________________________________________________________
Int_t TSocket::RecvHostAuth(Option_t *opt, const char *proofconf)
{
   // Receive from client/master directives for authentications, create
   // related THostAuth and add them to the TAuthenticate::ProofAuthInfo
   // list. Opt = "M" or "m" if Master, "S" or "s" if Proof slave.
   // The 'proofconf' file is read only if Master

   // Check if Master
   Bool_t Master = !strncasecmp(opt,"M",1) ? kTRUE : kFALSE;

   // First read directives from <rootauthrc>, <proofconf> and alike files
   if (Master)
      TAuthenticate::ReadRootAuthrc(proofconf);
   else
      TAuthenticate::ReadRootAuthrc();

   // Receive buffer
   Int_t kind;
   char buf[kMAXSECBUF];
   Int_t nr = Recv(buf, kMAXSECBUF, kind);
   if (nr < 0 || kind != kPROOF_SENDHOSTAUTH) {
      Error("RecvHostAuth", "received: kind: %d (%d bytes)", kind, nr);
      return -1;
   }
   if (gDebug > 2)
      Info("RecvHostAuth","received %d bytes (%s)",nr,buf);

   while (strcmp(buf, "END")) {
      // Clean buffer
      Int_t nc = (nr < kMAXSECBUF)? nr : kMAXSECBUF ;
      buf[nc] = '\0';

      // Create THostAuth
      THostAuth *ha = new THostAuth((const char *)&buf);

      // Check if there is already one compatible
      Int_t kExact = 0;
      THostAuth *haex = 0;
      Bool_t FromProofAI = kFALSE;
      if (Master) {
         // Look first in the proof list
         haex = TAuthenticate::GetHostAuth(ha->GetHost(),ha->GetUser(),"P",&kExact);
         // If nothing found, look also in the standard list
         if (!haex) {
            haex =
                TAuthenticate::GetHostAuth(ha->GetHost(),ha->GetUser(),"R",&kExact);
         } else
            FromProofAI = kTRUE;
      } else {
         // For slaves look first in the standard list only
         haex = TAuthenticate::GetHostAuth(ha->GetHost(),ha->GetUser(),"R",&kExact);
      }

      if (haex) {
         // If yes, action depends on whether it matches exactly or not
         if (kExact == 1) {
            // Update info in AuthInfo if Slave or in ProofAuthInfo
            // if Master and the entry was already in ProofAuthInfo
            if (!Master || FromProofAI) {
               // update this existing one with the information found in
               // in the new one, if needed
               haex->Update(ha);
               // Delete temporary THostAuth
               SafeDelete(ha);
            } else
               // Master, entry not already in ProofAuthInfo,
               // Add it to the list
               TAuthenticate::GetProofAuthInfo()->Add(ha);
         } else {
            // update this new one with the information found in
            // in the existing one (if needed) and ...
            Int_t i = 0;
            for (; i < haex->NumMethods(); i++) {
               Int_t met = haex->GetMethod(i);
               if (!ha->HasMethod(met))
                  ha->AddMethod(met,haex->GetDetails(met));
            }
            if (Master)
               // ... add the new one to the list
               TAuthenticate::GetProofAuthInfo()->Add(ha);
            else
               // We add this one to the standard list
               TAuthenticate::GetAuthInfo()->Add(ha);
         }
      } else {
         if (Master)
            // We add this one to the list for forwarding
            TAuthenticate::GetProofAuthInfo()->Add(ha);
         else
            // We add this one to the standard list
            TAuthenticate::GetAuthInfo()->Add(ha);
      }


      // Get the next one
      nr = Recv(buf, kMAXSECBUF, kind);
      if (nr < 0 || kind != kPROOF_SENDHOSTAUTH) {
         Info("RecvHostAuth","Error: received: kind: %d (%d bytes)", kind, nr);
         return -1;
      }
      if (gDebug > 2)
         Info("RecvHostAuth","received %d bytes (%s)",nr,buf);
   }

   return 0;
}

//______________________________________________________________________________
Int_t TSocket::SecureRecv(TString &str, Int_t key)
{
   // Receive encoded string and decode it with 'key' type

   char *buf = 0;
   Int_t rc = TAuthenticate::SecureRecv(this, key, &buf);
   str = TString(buf);
   if (buf) delete[] buf;

   return rc;
}
