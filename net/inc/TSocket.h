// @(#)root/net:$Name:  $:$Id: TSocket.h,v 1.13 2004/05/27 09:03:05 rdm Exp $
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
#ifndef ROOT_TSecContext
#include "TSecContext.h"
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
   kPeek,              // peek at incoming message (receive only)
   kDontBlock          // send/recv as much data as possible without blocking
};


class TMessage;
class THostAuth;
class TSecContext;


class TSocket : public TNamed {

friend class TProofServ;   // to be able to call SetDescriptor(), RecvHostAuth()
friend class TServerSocket;
friend class TSlave;       // to be able to call SendHostAuth()

public:
   enum EServiceType { kSOCKD, kROOTD, kPROOFD };

protected:
   TInetAddress  fAddress;        // remote internet address and port #
   UInt_t        fBytesRecv;      // total bytes received over this socket
   UInt_t        fBytesSent;      // total bytes sent using this socket
   TInetAddress  fLocalAddress;   // local internet address and port #
   Int_t         fRemoteProtocol; // protocol of remote daemon
   TSecContext  *fSecContext;     // after a successful Authenticate call points to related security context
   TString       fService;        // name of service (matches remote port #)
   EServiceType  fServType;       // remote service type
   Int_t         fSocket;         // socket descriptor
   Int_t         fCompress;       // compression level from 0 (not compressed) to 9 (max compression)
   TString       fUrl;            // needs this for special authentication options

   static UInt_t fgBytesRecv;     // total bytes received by all socket objects
   static UInt_t fgBytesSent;     // total bytes sent by all socket objects

   TSocket() { fSocket = -1; fBytesSent = fBytesRecv = 0; fCompress = 0; fSecContext = 0; }
   Bool_t       Authenticate(const char *user);

private:
   void         operator=(const TSocket &);  // not implemented
   Option_t    *GetOption() const { return TObject::GetOption(); }
   Int_t        RecvHostAuth(Option_t *opt, const char *proofconf = 0);
   Int_t        SecureRecv(TString &out, Int_t dec, Int_t key = 1);
   Int_t        SecureSend(const char *in, Int_t enc, Int_t keyType = 1);
   Int_t        SendHostAuth();
   void         SetDescriptor(Int_t desc) { fSocket = desc; }

public:
   TSocket(TInetAddress address, const char *service, Int_t tcpwindowsize = -1);
   TSocket(TInetAddress address, Int_t port, Int_t tcpwindowsize = -1);
   TSocket(const char *host, const char *service, Int_t tcpwindowsize = -1);
   TSocket(const char *host, Int_t port, Int_t tcpwindowsize = -1);
   TSocket(Int_t descriptor);
   TSocket(const TSocket &s);
   virtual ~TSocket() { Close(); }

   virtual void          Close(Option_t *opt="");
   virtual Int_t         GetDescriptor() const { return fSocket; }
   TInetAddress          GetInetAddress() const { return fAddress; }
   virtual TInetAddress  GetLocalInetAddress();
   Int_t                 GetPort() const { return fAddress.GetPort(); }
   const char           *GetService() const { return fService; }
   Int_t                 GetServType() const { return (Int_t)fServType; }
   virtual Int_t         GetLocalPort();
   UInt_t                GetBytesSent() const { return fBytesSent; }
   UInt_t                GetBytesRecv() const { return fBytesRecv; }
   Int_t                 GetCompressionLevel() const { return fCompress; }
   Int_t                 GetErrorCode() const;
   virtual Int_t         GetOption(ESockOptions opt, Int_t &val);
   Int_t                 GetRemoteProtocol() const { return fRemoteProtocol; }
   TSecContext          *GetSecContext() const { return fSecContext; }
   const char           *GetUrl() const { return fUrl; }
   virtual Bool_t        IsAuthenticated() const { return fSecContext ? kTRUE : kFALSE; }
   virtual Bool_t        IsValid() const { return fSocket < 0 ? kFALSE : kTRUE; }
   virtual Int_t         Recv(TMessage *&mess);
   virtual Int_t         Recv(Int_t &status, Int_t &kind);
   virtual Int_t         Recv(char *mess, Int_t max);
   virtual Int_t         Recv(char *mess, Int_t max, Int_t &kind);
   virtual Int_t         RecvRaw(void *buffer, Int_t length, ESendRecvOptions opt = kDefault);
   virtual Int_t         Send(const TMessage &mess);
   virtual Int_t         Send(Int_t kind);
   virtual Int_t         Send(Int_t status, Int_t kind);
   virtual Int_t         Send(const char *mess, Int_t kind = kMESS_STRING);
   virtual Int_t         SendObject(const TObject *obj, Int_t kind = kMESS_OBJECT);
   virtual Int_t         SendRaw(const void *buffer, Int_t length,
                                 ESendRecvOptions opt = kDefault);
   void                  SetCompressionLevel(Int_t level = 1);
   virtual Int_t         SetOption(ESockOptions opt, Int_t val);
   void                  SetRemoteProtocol(Int_t rproto) { fRemoteProtocol = rproto; }
   void                  SetSecContext(TSecContext *ctx) { fSecContext = ctx; }
   void                  SetUrl(const char *url) { fUrl = url; }

   static UInt_t         GetSocketBytesSent() { return fgBytesSent; }
   static UInt_t         GetSocketBytesRecv() { return fgBytesRecv; }

   static TSocket       *CreateAuthSocket(const char *user, const char *host, Int_t port,
                                          Int_t size = 0, Int_t tcpwindowsize = -1);
   static TSocket       *CreateAuthSocket(const char *url,
                                          Int_t size = 0, Int_t tcpwindowsize = -1);

   ClassDef(TSocket,1)  //This class implements client sockets
};

#endif
