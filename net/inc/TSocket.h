// @(#)root/net:$Name:  $:$Id: TSocket.h,v 1.2 2000/11/27 10:48:19 rdm Exp $
// Author: Fons Rademakers   18/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSocket
#define ROOT_TSocket


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

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TInetAddress
#include "TInetAddress.h"
#endif
#ifndef ROOT_MessageTypes
#include "MessageTypes.h"
#endif

enum ESockOptions {
   kSendBuffer,        // size of send buffer
   kRecvBuffer,        // size of receive buffer
   kOobInline,         // OOB message inline
   kKeepAlive,         // keep socket alive
   kReuseAddr,         // allow reuse of local portion of address 5-tuple
   kNoDelay,           // send without delay
   kNoBlock,           // non-blocking I/O
   kProcessGroup,      // socket process group (used for SIGURG and SIGIO)
   kAtMark,            // are we at out-of-band mark (read only)
   kBytesToRead        // get number of bytes to read, FIONREAD (read only)
};

enum ESendRecvOptions {
   kDefault,           // default option (= 0)
   kOob,               // send or receive out-of-band data
   kPeek               // peek at incoming message (receive only)
};


class TMessage;

class TSocket : public TNamed {

friend class TServerSocket;
friend class TProofServ;   // to be able to call SetDescriptor()

protected:
   Int_t         fSocket;         // socket descriptor
   TString       fService;        // name of service (matches remote port #)
   TInetAddress  fAddress;        // remote internet address and port #

   TSocket() { fSocket = -1; fBytesSent = fBytesRecv = 0; }

private:
   TInetAddress  fLocalAddress;   // local internet address and port #
   UInt_t        fBytesSent;      // total bytes sent using this socket
   UInt_t        fBytesRecv;      // total bytes received over this socket

   static UInt_t fgBytesSent;     // total bytes sent by all socket objects
   static UInt_t fgBytesRecv;     // total bytes received by all socket objects

   void operator=(const TSocket &);  // not implemented
   void SetDescriptor(Int_t desc) { fSocket = desc; }
   Option_t *GetOption() const { return TObject::GetOption(); }

public:
   TSocket(TInetAddress address, const char *service, Int_t recvbuf = -1);
   TSocket(TInetAddress address, Int_t port, Int_t recvbuf = -1);
   TSocket(const char *host, const char *service, Int_t recvbuf = -1);
   TSocket(const char *host, Int_t port, Int_t recvbuf = -1);
   TSocket(Int_t descriptor);
   TSocket(const TSocket &s);
   virtual ~TSocket() { Close(); }

   virtual void          Close(Option_t *opt="");
   Int_t                 GetDescriptor() const { return fSocket; }
   TInetAddress          GetInetAddress() const { return fAddress; }
   virtual TInetAddress  GetLocalInetAddress();
   Int_t                 GetPort() const { return fAddress.GetPort(); }
   virtual Int_t         GetLocalPort();
   UInt_t                GetBytesSent() const { return fBytesSent; }
   UInt_t                GetBytesRecv() const { return fBytesRecv; }
   virtual Int_t         Send(const TMessage &mess);
   virtual Int_t         Send(Int_t kind);
   virtual Int_t         Send(Int_t status, Int_t kind);
   virtual Int_t         Send(const char *mess, Int_t kind = kMESS_STRING);
   virtual Int_t         SendObject(const TObject *obj, Int_t kind = kMESS_OBJECT);
   virtual Int_t         SendRaw(const void *buffer, Int_t length, ESendRecvOptions opt = kDefault);
   virtual Int_t         Recv(TMessage *&mess);
   virtual Int_t         Recv(Int_t &status, Int_t &kind);
   virtual Int_t         Recv(char *mess, Int_t max);
   virtual Int_t         Recv(char *mess, Int_t max, Int_t &kind);
   virtual Int_t         RecvRaw(void *buffer, Int_t length, ESendRecvOptions opt = kDefault);
   Bool_t                IsValid() const { return fSocket < 0 ? kFALSE : kTRUE; }
   Int_t                 GetErrorCode() const;
   virtual Int_t         SetOption(ESockOptions opt, Int_t val);
   virtual Int_t         GetOption(ESockOptions opt, Int_t &val);

   static  UInt_t        GetSocketBytesSent() { return fgBytesSent; }
   static  UInt_t        GetSocketBytesRecv() { return fgBytesRecv; }

   ClassDef(TSocket,1)  //This class implements client sockets
};

#endif
