// @(#)root/net:$Name:  $:$Id: TInetAddress.h,v 1.4 2001/10/01 09:46:32 rdm Exp $
// Author: G. Ganis   31/03/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAuthDetails
#define ROOT_TAuthDetails


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAuthDetails                                                         //
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
#ifndef ROOT_TAuthenticate
#include "TAuthenticate.h"
#endif

class TAuthDetails : public TObject {

public:
   enum EService { kROOTD=1, kPROOFD };

private:
   TString      fHost;                // Remote host and service (in form  host:port:service)
   TString      fRealHost;            // Remote host name
   Int_t        fPort;                // Remote Port number
   EService     fService;             // Remote Service flag
   Int_t        fMethod;              // Authentication method used
   Int_t        fRemoteOffSet;        // offset in remote host auth tab file (in bytes)
   TString      fRemoteLogin;         // Remote login name (either auth user or one of the available anonymous)
   TString      fDetails;             // Details for the auth process (user, principal, ... )
   Bool_t       fReUse;               // Determines if established authentication context should be reused
   TString      fToken;               // Token identifying this authentication
   Int_t        fRSAKey;              // Type of RSA key used

public:

   TAuthDetails(const char *host, Int_t meth, Int_t offset, Bool_t reuse,
                const char *details, const char *token, Int_t key,
                const char *login = "");
   virtual ~TAuthDetails() { }

   const char *GetHost()    const;
   Int_t       GetPort()    const;
   Int_t       GetService() const;
   Int_t       GetMethod()  const { return fMethod; }
   const char *GetDetails() const { return fDetails; }
   Bool_t      GetReUse()   const { return fReUse; }
   const char *GetLogin()   const { return fRemoteLogin; }
   Int_t       GetOffSet()  const { return fRemoteOffSet; }
   const char *GetToken()   const { return fToken; }
   Int_t       GetRSAKey()  const { return fRSAKey; }

   void     SetOffSet(Int_t offset)      { fRemoteOffSet = offset; }
   void     SetReUse(Bool_t reuse)       { fReUse        = reuse; }
   void     SetLogin(const char *login)  { fRemoteLogin  = login; }

   void     Print(Option_t *option = "") const;

   ClassDef(TAuthDetails,0)  // Class providing host specific authentication information
};

#endif
