// @(#)root/net:$Name:  $:$Id: TSocket.cxx,v 1.4 2000/08/21 14:48:37 rdm Exp $
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

#include "TSocket.h"
#include "TSystem.h"
#include "TMessage.h"
#include "Bytes.h"
#include "TROOT.h"
#include "TError.h"

UInt_t TSocket::fgBytesSent = 0;
UInt_t TSocket::fgBytesRecv = 0;

ClassImp(TSocket)

//______________________________________________________________________________
TSocket::TSocket(TInetAddress addr, const char *service)
         : TNamed(addr.GetHostName(), service)
{
   // Create a socket. Connect to the named service at address addr.
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   Assert(gROOT);
   Assert(gSystem);

   fService = service;
   fAddress = addr;
   fAddress.fPort = gSystem->GetServiceByName(service);
   fBytesSent = 0;
   fBytesRecv = 0;

   if (fAddress.GetPort() != -1) {
      fSocket = gSystem->OpenConnection(addr.GetHostName(), fAddress.GetPort());
      if (fSocket != -1) gROOT->GetListOfSockets()->Add(this);
   } else
      fSocket = -1;
}

//______________________________________________________________________________
TSocket::TSocket(TInetAddress addr, Int_t port)
         : TNamed(addr.GetHostName(), "")
{
   // Create a socket. Connect to the specified port # at address addr.
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   Assert(gROOT);
   Assert(gSystem);

   fService = gSystem->GetServiceByPort(port);
   fAddress = addr;
   fAddress.fPort = port;
   SetTitle(fService);
   fBytesSent = 0;
   fBytesRecv = 0;

   fSocket = gSystem->OpenConnection(addr.GetHostName(), fAddress.GetPort());
   if (fSocket == -1)
      fAddress.fPort = -1;
   else
      gROOT->GetListOfSockets()->Add(this);
}

//______________________________________________________________________________
TSocket::TSocket(const char *host, const char *service)
         : TNamed(host, service)
{
   // Create a socket. Connect to named service on the remote host.
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   Assert(gROOT);
   Assert(gSystem);

   fService = service;
   fAddress = gSystem->GetHostByName(host);
   fAddress.fPort = gSystem->GetServiceByName(service);
   SetName(fAddress.GetHostName());
   fBytesSent = 0;
   fBytesRecv = 0;

   if (fAddress.GetPort() != -1) {
      fSocket = gSystem->OpenConnection(host, fAddress.GetPort());
      if (fSocket != -1) gROOT->GetListOfSockets()->Add(this);
   } else
      fSocket = -1;
}

//______________________________________________________________________________
TSocket::TSocket(const char *host, Int_t port)
         : TNamed(host, "")
{
   // Create a socket. Connect to specified port # on the remote host.
   // Returns when connection has been accepted by remote side. Use IsValid()
   // to check the validity of the socket. Every socket is added to the TROOT
   // sockets list which will make sure that any open sockets are properly
   // closed on program termination.

   Assert(gROOT);
   Assert(gSystem);

   fService = gSystem->GetServiceByPort(port);
   fAddress = gSystem->GetHostByName(host);
   fAddress.fPort = port;
   SetName(fAddress.GetHostName());
   SetTitle(fService);
   fBytesSent = 0;
   fBytesRecv = 0;

   fSocket = gSystem->OpenConnection(host, fAddress.GetPort());
   if (fSocket == -1)
      fAddress.fPort = -1;
   else
      gROOT->GetListOfSockets()->Add(this);
}

//______________________________________________________________________________
TSocket::TSocket(Int_t desc) : TNamed("", "")
{
   // Create a socket. The socket will use descriptor desc.

   Assert(gROOT);
   Assert(gSystem);

   fBytesSent = 0;
   fBytesRecv = 0;

   if (desc >= 0) {
      fSocket  = desc;
      fAddress = gSystem->GetPeerName(fSocket);
      gROOT->GetListOfSockets()->Add(this);
   } else
      fSocket = -1;
}

//______________________________________________________________________________
TSocket::TSocket(const TSocket &s)
{
   // TSocket copy ctor.

   fSocket       = s.fSocket;
   fService      = s.fService;
   fAddress      = s.fAddress;
   fLocalAddress = s.fLocalAddress;
   fBytesSent    = s.fBytesSent;
   fBytesRecv    = s.fBytesRecv;

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

   if (fSocket != -1) {
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

   if (fSocket != -1) {
      if (fLocalAddress.GetPort() == -1)
         fLocalAddress = GetLocalInetAddress();
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
   return Send(mess);
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
   return Send(mess);
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

   if (fSocket == -1) return -1;
   if (mess.IsReading()) {
      Error("Send", "cannot send a message used for reading");
      return -1;
   }

   Int_t nsent;
   mess.SetLength();   //write length in first word of buffer
   if ((nsent = gSystem->SendRaw(fSocket, mess.Buffer(), mess.Length(), 0)) < 0)
      return -1;

   fBytesSent  += nsent;
   fgBytesSent += nsent;

   // If acknowledgement is desired, wait for it
   if (mess.What() & kMESS_ACK) {
      char buf[2];
      if (gSystem->RecvRaw(fSocket, buf, sizeof(buf), 0) < 0)
         return -1;
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
   return Send(mess);
}

//______________________________________________________________________________
Int_t TSocket::SendRaw(const void *buffer, Int_t length, ESendRecvOptions opt)
{
   // Send a raw buffer of specified length. Using option kOob one can send
   // OOB data. Returns the number of bytes sent or -1 in case of error.

   if (fSocket == -1) return -1;

   Int_t nsent;

   if ((nsent = gSystem->SendRaw(fSocket, buffer, length, (int) opt)) < 0)
      return -1;

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

   if ((n = Recv(str, max, kind)) <= 0)
      return n;

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

   if ((n = Recv(mess)) <= 0)
      return n;

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

   if ((n = Recv(mess)) <= 0)
      return n;

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
   // is closed) or -1 in case of error or -4 in case a non-blocking socket would
   // block (i.e. there is nothing to be read). In those case mess == 0.

   TSystem::ResetErrno();

   if (fSocket == -1) {
      mess = 0;
      return -1;
   }

   Int_t  n;
   UInt_t len;
   if ((n = gSystem->RecvRaw(fSocket, &len, sizeof(UInt_t), 0)) <= 0) {
      mess = 0;
      return n;
   }
   len = net2host(len);  //from network to host byte order

   char *buf = new char[len+sizeof(UInt_t)];
   if ((n = gSystem->RecvRaw(fSocket, buf+sizeof(UInt_t), len, 0)) <= 0) {
      delete [] buf;
      mess = 0;
      return n;
   }

   fBytesRecv  += n + sizeof(UInt_t);
   fgBytesRecv += n + sizeof(UInt_t);

   mess = new TMessage(buf, len+sizeof(UInt_t));

   if (mess->What() & kMESS_ACK) {
      char ok[2] = { 'o', 'k' };
      if (gSystem->SendRaw(fSocket, ok, sizeof(ok), 0) < 0) {
         delete mess;
         mess = 0;
         return -1;
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
   // mode (kNoBlock) -4 means EWOULDBLOCK.

   TSystem::ResetErrno();

   if (fSocket == -1) return -1;

   Int_t n;

   if ((n = gSystem->RecvRaw(fSocket, buffer, length, (int) opt)) <= 0)
      return n;

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
