// @(#)root/net:$Id$
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

// Jan 1, 1995, 00:00:00 in sec from EPOCH (Jan 1, 1970)
const TDatime kROOTTZERO = 788914800;

// Small class with information for final cleanup
class TSecContextCleanup;
class TPwdCtx;

class TSecContext : public TObject {

friend class TRootSecContext;

private:
   void        *fContext;             // Krb5, Globus: ptr to specific sec context
   TList       *fCleanup;             // Points to list with info for remote cleanup
   TDatime      fExpDate;             // Expiring date (one sec precision)
   TString      fHost;                // Remote host name
   TString      fID;                  // String identifying uniquely this context
   Int_t        fMethod;              // Authentication method used
   TString      fMethodName;          // Authentication method name
   Int_t        fOffSet;              // offset in remote host auth tab file (in bytes)
   TString      fToken;               // Token identifying this authentication
   TString      fUser;                // Remote login username

   virtual Bool_t  CleanupSecContext(Bool_t all);
   void         Cleanup();

protected:
   TSecContext(const TSecContext&);
   TSecContext& operator=(const TSecContext&);

public:

   TSecContext(const char *url, Int_t meth, Int_t offset,
               const char *id, const char *token,
               TDatime expdate = kROOTTZERO, void *ctx = 0);
   TSecContext(const char *user, const char *host, Int_t meth, Int_t offset,
               const char *id, const char *token,
               TDatime expdate = kROOTTZERO, void *ctx = 0);
   virtual    ~TSecContext();

   void        AddForCleanup(Int_t port, Int_t proto, Int_t type);
   virtual const char *AsString(TString &out);

   virtual void DeActivate(Option_t *opt = "CR");
   void       *GetContext() const { return fContext; }
   TDatime     GetExpDate() const { return fExpDate; }
   const char *GetHost()    const { return fHost; }
   const char *GetID() const { return fID; }
   Int_t       GetMethod()  const { return fMethod; }
   const char *GetMethodName() const { return fMethodName; }
   Int_t       GetOffSet()  const { return fOffSet; }
   TList      *GetSecContextCleanup() const { return fCleanup; }
   const char *GetToken()   const { return fToken; }
   const char *GetUser()    const { return fUser; }

   Bool_t      IsA(const char *methodname);
   Bool_t      IsActive()   const;

   virtual void Print(Option_t *option = "F") const;

   void        SetExpDate(TDatime expdate)  { fExpDate= expdate; }
   void        SetID(const char *id)        { fID= id; }
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

   ClassDef(TSecContextCleanup,0) //Update the remote authentication table
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
