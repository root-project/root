// @(#)root/net:$Name:  $:$Id: TSecContext.h,v 1.1 2003/08/29 10:38:19 rdm Exp $
// Author: G. Ganis   31/03/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSecContext
#define ROOT_TSecContext


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSecContext                                                         //
//                                                                      //
// Contains details about successful authentications                    //
// Used by THostAuth                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TDatime
#include "TDatime.h"
#endif
#ifndef ROOT_TAuthenticate
#include "TAuthenticate.h"
#endif

// Jan 1, 1995, 00:00:00 in sec from EPOCH (Jan 1, 1970)
const Int_t kROOTTZERO = 788914800;

// Small class with information for final cleanup
class TSecContextCleanup;
class TPwdCtx;

class TSecContext : public TObject {

private:
   void        *fContext;             // Krb5, Globus: ptr to specific sec context
   TList       *fCleanup;             // Points to list with info for remote cleanup
   TString      fDetails;             // Auth process details (user, principal, ... )
   TDatime      fExpDate;             // Expiring date (one sec precision)
   TString      fHost;                // Remote host name
   Int_t        fMethod;              // Authentication method used
   Int_t        fOffSet;              // offset in remote host auth tab file (in bytes)
   Int_t        fRSAKey;              // Type of RSA key used
   TString      fToken;               // Token identifying this authentication
   TString      fUser;                // Remote login username

public:

   TSecContext(const char *url, Int_t meth, Int_t offset,
               const char *details, const char *token, 
               TDatime expdate = kROOTTZERO, void *ctx = 0, Int_t key = 1);
   TSecContext(const char *user, const char *host, Int_t meth, Int_t offset, 
               const char *details, const char *token, 
               TDatime expdate = kROOTTZERO, void *ctx = 0, Int_t key = 1);
   virtual    ~TSecContext();

   void        AddForCleanup(Int_t port, Int_t proto, Int_t type);
   const char *AsString() const;
   void        Cleanup();

   void        DeActivate(Option_t *opt = "CR");
   void       *GetContext() const { return fContext; }
   const char *GetDetails() const { return fDetails; }
   TDatime     GetExpDate() const { return fExpDate; }
   const char *GetHost()    const { return fHost; }
   Int_t       GetMethod()  const { return fMethod; }
   const char *GetMethodName() const 
                   { return TAuthenticate::GetAuthMethod(fMethod); }
   Int_t       GetOffSet()  const { return fOffSet; }
   Int_t       GetRSAKey()  const { return fRSAKey; }
   TList      *GetSecContextCleanup() const { return fCleanup; }
   const char *GetToken()   const { return fToken; }
   const char *GetUser()    const { return fUser; }

   Bool_t      IsA(const char *methodname) const;
   Bool_t      IsActive()   const;

   virtual void Print(Option_t *option = "F") const;

   void        SetExpDate(TDatime expdate)  { fExpDate= expdate; }
   void        SetOffSet(Int_t offset)      { fOffSet = offset; }
   void        SetUser(const char *user)    { fUser   = user; }

   ClassDef(TSecContext,0)  // Class providing host specific authentication information
};

//
// TSecContextCleanup
//
// When the context is destroyed the remote authentication table 
// should be updated; also, for globus, remote shared memory segments
// should be destroyed; for this we need to open a socket to a remote
// service; we keep track here of port and type of socket needed by 
// the remote service used in connection with this security context.
// The last used is the first in the list.
// This info is used in TAuthenticate::CleanupSecContext to trasmit 
// the actual cleanup request
//
class TSecContextCleanup : public TObject {

private:
   Int_t   fPort;
   Int_t   fServerProtocol;
   Int_t   fServerType;     // 0 = sockd, 1 = rootd, 2 = proofd

public:
   TSecContextCleanup(Int_t port, Int_t proto, Int_t type) : 
               fPort(port), fServerProtocol(proto), fServerType(type) { };
   virtual ~TSecContextCleanup() { };

   Int_t   GetPort() const { return fPort; }
   Int_t   GetProtocol() const { return fServerProtocol; }
   Int_t   GetType() const { return fServerType; }

   ClassDef(TSecContextCleanup,0)  
};

//
// TPwdCtx
//
// To store associated passwd for UsrPwd and SRP methods
//
class TPwdCtx {

private:
   TString fPasswd;
   Bool_t  fPwHash;

public:
   TPwdCtx(const char *pwd, Bool_t pwh): fPasswd(pwd), fPwHash(pwh) {};
   virtual ~TPwdCtx() {};

   const char *GetPasswd() const { return fPasswd; }
   Bool_t      IsPwHash() const { return fPwHash; }

};



#endif
