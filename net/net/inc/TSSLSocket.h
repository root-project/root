// @(#)root/net:$Id: TSSLSocket.h
// Author: Alejandro Alvarez 16/09/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSSLSocket
#define ROOT_TSSLSocket

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSSLSocket                                                           //
//                                                                      //
// A TSocket wrapped in by SSL.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSocket
#include "TSocket.h"
#endif

typedef struct ssl_st     SSL;
typedef struct ssl_ctx_st SSL_CTX;

class TSSLSocket : public TSocket {
protected:
   TSSLSocket() : TSocket() {}

private:
   // CA, client cert/key... are class properties
   static char fgSSLCAFile[];
   static char fgSSLCAPath[];
   static char fgSSLUCert[];
   static char fgSSLUKey[];

   // Object properties
   SSL_CTX *fSSLCtx;
   SSL     *fSSL;

   void WrapWithSSL();

public:
   TSSLSocket(TInetAddress addr, const char *service, Int_t tcpwindowsize = -1);
   TSSLSocket(TInetAddress addr, Int_t port, Int_t tcpwindowsize = -1);
   TSSLSocket(const char *host, const char *service, Int_t tcpwindowsize = -1);
   TSSLSocket(const char *url, Int_t port, Int_t tcpwindowsize = -1);
   TSSLSocket(const char *sockpath);
   TSSLSocket(Int_t desc);
   TSSLSocket(Int_t desc, const char *sockpath);
   TSSLSocket(const TSSLSocket &s);
   virtual ~TSSLSocket();

   void  Close(Option_t *option="");

   // Set up the SSL environment for the next instantiation
   static void SetUpSSL(const char *cafile, const char *capath,
                        const char *ucert,  const char *ukey);

   // The rest of the Send and Recv calls rely ultimately on these,
   // so it is enough to overload them
   Int_t Recv(TMessage *&mess);
   Int_t RecvRaw(void *buffer, Int_t length, ESendRecvOptions opt = kDefault);
   Int_t Send(const TMessage &mess);
   Int_t SendRaw(const void *buffer, Int_t length,
                 ESendRecvOptions opt = kDefault);

   // Issue with hidden method :(
   Int_t Send(Int_t kind)                                  { return TSocket::Send(kind); }
   Int_t Send(Int_t status, Int_t kind)                    { return TSocket::Send(status, kind); }
   Int_t Send(const char *mess, Int_t kind = kMESS_STRING) { return TSocket::Send(mess, kind); }
   Int_t Recv(Int_t &status, Int_t &kind)                  { return TSocket::Recv(status, kind); }
   Int_t Recv(char *mess, Int_t max)                       { return TSocket::Recv(mess, max); }
   Int_t Recv(char *mess, Int_t max, Int_t &kind)          { return TSocket::Recv(mess, max, kind); }

   ClassDef(TSSLSocket,0)  // SSL wrapped socket
};

#endif
