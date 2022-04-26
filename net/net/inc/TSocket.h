// @(#)root/net:$Id$
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
// or TWinNTSystem).                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSystem.h"
#include "Compression.h"
#include "TNamed.h"
#include "TBits.h"
#include "TInetAddress.h"
#include "MessageTypes.h"
#include "TVirtualAuth.h"
#include "TSecContext.h"
#include "TTimeStamp.h"
#include "TVirtualMutex.h"

class TMessage;
class THostAuth;

class TSocket : public TNamed {

friend class TServerSocket;
friend class TProofServ;   // to be able to call SetDescriptor(), RecvHostAuth()
friend class TSlave;       // to be able to call SendHostAuth()

public:
   enum EStatusBits { kIsUnix = BIT(16),    // set if unix socket
                      kBrokenConn = BIT(17) // set if conn reset by peer or broken
                    };
   enum EInterest { kRead = 1, kWrite = 2 };
   enum EServiceType { kSOCKD, kROOTD, kPROOFD };

protected:
   enum ESocketErrors {
     kInvalid = -1,
     kInvalidStillInList = -2
   };
   TInetAddress  fAddress;        // remote internet address and port #
   UInt_t        fBytesRecv;      // total bytes received over this socket
   UInt_t        fBytesSent;      // total bytes sent using this socket
   Int_t         fCompress;       // Compression level and algorithm
   TInetAddress  fLocalAddress;   // local internet address and port #
   Int_t         fRemoteProtocol; // protocol of remote daemon
   TSecContext  *fSecContext;     // after a successful Authenticate call
                                  // points to related security context
   TString       fService;        // name of service (matches remote port #)
   EServiceType  fServType;       // remote service type
   Int_t         fSocket;         // socket descriptor
   Int_t         fTcpWindowSize;  // TCP window size (default 65535);
   TString       fUrl;            // needs this for special authentication options
   TBits         fBitsInfo;       // bits array to mark TStreamerInfo classes already sent
   TList        *fUUIDs;          // list of TProcessIDs already sent through the socket

   TVirtualMutex *fLastUsageMtx;   // Protect last usage setting / reading
   TTimeStamp    fLastUsage;      // Time stamp of last usage

   static ULong64_t fgBytesRecv;  // total bytes received by all socket objects
   static ULong64_t fgBytesSent;  // total bytes sent by all socket objects

   static Int_t  fgClientProtocol; // client "protocol" version

   TSocket() : fAddress(), fBytesRecv(0), fBytesSent(0), fCompress(ROOT::RCompressionSetting::EAlgorithm::kUseGlobal),
               fLocalAddress(), fRemoteProtocol(), fSecContext(0), fService(),
               fServType(kSOCKD), fSocket(-1), fTcpWindowSize(0), fUrl(),
               fBitsInfo(), fUUIDs(0), fLastUsageMtx(0), fLastUsage() { }

   Bool_t       Authenticate(const char *user);
   void         SetDescriptor(Int_t desc) { fSocket = desc; }
   void         SendStreamerInfos(const TMessage &mess);
   Bool_t       RecvStreamerInfos(TMessage *mess);
   void         SendProcessIDs(const TMessage &mess);
   Bool_t       RecvProcessIDs(TMessage *mess);
   void         MarkBrokenConnection();

private:
   TSocket&      operator=(const TSocket &) = delete;
   Option_t     *GetOption() const override { return TObject::GetOption(); }

public:
   TSocket(TInetAddress address, const char *service, Int_t tcpwindowsize = -1);
   TSocket(TInetAddress address, Int_t port, Int_t tcpwindowsize = -1);
   TSocket(const char *host, const char *service, Int_t tcpwindowsize = -1);
   TSocket(const char *host, Int_t port, Int_t tcpwindowsize = -1);
   TSocket(const char *sockpath);
   TSocket(Int_t descriptor);
   TSocket(Int_t descriptor, const char *sockpath);
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
   Int_t                 GetCompressionAlgorithm() const;
   Int_t                 GetCompressionLevel() const;
   Int_t                 GetCompressionSettings() const;
   Int_t                 GetErrorCode() const;
   virtual Int_t         GetOption(ESockOptions opt, Int_t &val);
   Int_t                 GetRemoteProtocol() const { return fRemoteProtocol; }
   TSecContext          *GetSecContext() const { return fSecContext; }
   Int_t                 GetTcpWindowSize() const { return fTcpWindowSize; }
   TTimeStamp            GetLastUsage() { R__LOCKGUARD2(fLastUsageMtx); return fLastUsage; }
   const char           *GetUrl() const { return fUrl.Data(); }
   virtual Bool_t        IsAuthenticated() const { return fSecContext ? kTRUE : kFALSE; }
   virtual Bool_t        IsValid() const { return fSocket < 0 ? kFALSE : kTRUE; }
   virtual Int_t         Recv(TMessage *&mess);
   virtual Int_t         Recv(Int_t &status, Int_t &kind);
   virtual Int_t         Recv(char *mess, Int_t max);
   virtual Int_t         Recv(char *mess, Int_t max, Int_t &kind);
   virtual Int_t         RecvRaw(void *buffer, Int_t length, ESendRecvOptions opt = kDefault);
   virtual Int_t         Reconnect() { return -1; }
   virtual Int_t         Select(Int_t interest = kRead, Long_t timeout = -1);
   virtual Int_t         Send(const TMessage &mess);
   virtual Int_t         Send(Int_t kind);
   virtual Int_t         Send(Int_t status, Int_t kind);
   virtual Int_t         Send(const char *mess, Int_t kind = kMESS_STRING);
   virtual Int_t         SendObject(const TObject *obj, Int_t kind = kMESS_OBJECT);
   virtual Int_t         SendRaw(const void *buffer, Int_t length,
                                 ESendRecvOptions opt = kDefault);
   void                  SetCompressionAlgorithm(Int_t algorithm = ROOT::RCompressionSetting::EAlgorithm::kUseGlobal);
   void                  SetCompressionLevel(Int_t level = ROOT::RCompressionSetting::ELevel::kUseMin);
   void                  SetCompressionSettings(Int_t settings = ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault);
   virtual Int_t         SetOption(ESockOptions opt, Int_t val);
   void                  SetRemoteProtocol(Int_t rproto) { fRemoteProtocol = rproto; }
   void                  SetSecContext(TSecContext *ctx) { fSecContext = ctx; }
   void                  SetService(const char *service) { fService = service; }
   void                  SetServType(Int_t st) { fServType = (EServiceType)st; }
   void                  SetUrl(const char *url) { fUrl = url; }

   void                  Touch() { R__LOCKGUARD2(fLastUsageMtx); fLastUsage.Set(); }

   static Int_t          GetClientProtocol();

   static ULong64_t      GetSocketBytesSent();
   static ULong64_t      GetSocketBytesRecv();

   static TSocket       *CreateAuthSocket(const char *user, const char *host,
                                          Int_t port, Int_t size = 0,
                                          Int_t tcpwindowsize = -1, TSocket *s = 0, Int_t *err = 0);
   static TSocket       *CreateAuthSocket(const char *url, Int_t size = 0,
                                          Int_t tcpwindowsize = -1, TSocket *s = 0, Int_t *err = 0);
   static void           NetError(const char *where, Int_t error);

   ClassDefOverride(TSocket,0)  //This class implements client sockets
};

//______________________________________________________________________________
inline Int_t TSocket::GetCompressionAlgorithm() const
{
   return (fCompress < 0) ? -1 : fCompress / 100;
}

//______________________________________________________________________________
inline Int_t TSocket::GetCompressionLevel() const
{
   return (fCompress < 0) ? -1 : fCompress % 100;
}

//______________________________________________________________________________
inline Int_t TSocket::GetCompressionSettings() const
{
   return (fCompress < 0) ? -1 : fCompress;
}

#endif
