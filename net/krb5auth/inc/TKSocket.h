// @(#)root/krb5auth:$Id$
// Author: Maarten Ballintijn   27/10/2003

#ifndef ROOT_TKSocket
#define ROOT_TKSocket

#if !defined(__CINT__)

#ifndef ROOT_Krb5Auth
#include "Krb5Auth.h"
#endif

#else

typedef void* krb5_principal;
typedef void* krb5_auth_context;
typedef void* krb5_context;
typedef void* krb5_ccache;
typedef void* krb5_principal;

#endif

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TSocket;


class TKSocket : public TObject {

private:
   TSocket                *fSocket;       //underlying socket
   krb5_principal          fServer;       //server principal
   krb5_auth_context       fAuthContext;  //per connection kerberos authentication context

   static krb5_context     fgContext;     //shared kerberos context
   static krb5_ccache      fgCCDef;       //shared default credential cache
   static krb5_principal   fgClient;      //client principal

   TKSocket(TSocket *s = 0);

public:
   ~TKSocket();

   enum EEncoding { kNone = 0, kSafe = 1, kPriv = 2 };

   struct Desc_t {
      Short_t  fLength;
      Short_t  fType;
   };

   Int_t BlockRead(char *&buf, EEncoding &type);
   Int_t BlockWrite(const char *buf, Int_t len, EEncoding type);

   static TKSocket *Connect(const char *server, Int_t port);

   ClassDef(TKSocket,0) // General kerberized socket
};

#endif
