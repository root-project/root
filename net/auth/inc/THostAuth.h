// @(#)root/auth:$Id$
// Author: G. Ganis   19/03/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THostAuth
#define ROOT_THostAuth


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THostAuth                                                            //
//                                                                      //
// Contains details about host-specific authentication methods and the  //
// result of their application                                          //
// Used by TAuthenticate                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"
#include "TList.h"
#include "TRootSecContext.h"
#include "AuthConst.h"

#include "TSecContext.h" // for kROOTTZERO.

class THostAuth : public TObject {

private:
   TString      fHost;             // Host
   Char_t       fServer;           // Server (kSOCKD,kROOTD,kPROOFD)
   TString      fUser;             // Username
   Int_t        fNumMethods;       // Number of AuthMethods
   Int_t        fMethods[kMAXSEC]; // AuthMethods
   TString      fDetails[kMAXSEC]; // AuthDetails
   Int_t        fSuccess[kMAXSEC]; // Statistics of successful attempts / per method
   Int_t        fFailure[kMAXSEC]; // Statistics of failed attempts / per method
   Bool_t       fActive;           // Flag used in cleaning/reset

   TList       *fSecContexts;  // List of TSecContexts related to this THostAuth

   void         Create(const char *host, const char *user, Int_t nmeth = 0,
                       Int_t *authmeth = nullptr, char **details = nullptr);
public:

   THostAuth();
   THostAuth(const char *host, const char *user,
             Int_t nmeth = 0, Int_t *authmeth = nullptr, char **details = nullptr);
   THostAuth(const char *host, Int_t server, const char *user,
             Int_t nmeth = 0, Int_t *authmeth = nullptr, char **details = nullptr);
   THostAuth(const char *host, const char *user, Int_t authmeth,
             const char *details);
   THostAuth(const char *host, Int_t server, const char *user, Int_t authmeth,
             const char *details);
   THostAuth(const char *asstring);
   THostAuth(THostAuth &ha);

   virtual ~THostAuth();

   void     AsString(TString &out) const;

   Int_t    NumMethods() const { return fNumMethods; }
   Int_t    GetMethod(Int_t idx) const { return fMethods[idx]; }
   Bool_t   HasMethod(Int_t level, Int_t *pos = nullptr);
   void     AddMethod(Int_t level, const char *details = nullptr);
   void     RemoveMethod(Int_t level);
   void     ReOrder(Int_t nmet, Int_t *fmet);
   void     Update(THostAuth *ha);
   void     SetFirst(Int_t level);
   void     AddFirst(Int_t level, const char *details = nullptr);
   void     SetLast(Int_t level);
   void     CountFailure(Int_t level);
   void     CountSuccess(Int_t level);
   Int_t    GetFailure(Int_t idx) const { return fFailure[idx]; }
   Int_t    GetSuccess(Int_t idx) const { return fSuccess[idx]; }
   Bool_t   IsActive() const { return fActive; }
   void     DeActivate() { fActive = kFALSE; }
   void     Activate() { fActive = kTRUE; }
   void     Reset();

   const char *GetDetails(Int_t level);
   const char *GetDetailsByIdx(Int_t idx) const { return fDetails[idx]; }
   void        SetDetails(Int_t level, const char *details);

   const char *GetHost() const { return fHost; }
   Int_t    GetServer() const { return (Int_t)fServer; }
   const char *GetUser() const { return fUser; }

   void     SetHost(const char *host) { fHost = host; }
   void     SetServer(Int_t server) { fServer = (Char_t)server; }
   void     SetUser(const char *user) { fUser = user; }

   TList   *Established() const { return fSecContexts; }
   void     SetEstablished(TList *nl) { fSecContexts = nl; }

   void  Print(Option_t *option = "") const override;
   void     PrintEstablished() const;

   TRootSecContext *CreateSecContext(const char *user, const char *host, Int_t meth,
                                     Int_t offset, const char *details,
                                     const char *token, TDatime expdate = kROOTTZERO,
                                     void *ctx = nullptr, Int_t key = -1);

   ClassDefOverride(THostAuth,1)  // Class providing host specific authentication information
};

#endif
