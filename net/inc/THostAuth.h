// @(#)root/net:$Name:  $:$Id: THostAuth.h,v 1.1 2003/08/29 10:38:19 rdm Exp $
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

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif


class THostAuth : public TObject {

private:
   TString      fHost;         // Host
   TString      fUser;         // Username
   Int_t        fNumMethods;   // Number of AuthMethods
   Int_t       *fMethods;      // AuthMethods
   TString     *fDetails;      // AuthDetails

   TList       *fEstablished;  // List of (TAuthDetails) established authentications

public:

   THostAuth();
   THostAuth(const char *host, const char *user, Int_t nmeth, Int_t *authmeth,
             char **details);
   THostAuth(const char *host, const char *user, Int_t authmeth,
             const char *details);
   virtual ~THostAuth();

   Int_t    NumMethods() const { return fNumMethods; }
   Int_t    GetMethods(Int_t meth) const { return fMethods[meth]; }
   void     AddMethod(Int_t level, const char *details);
   void     RemoveMethod(Int_t level);
   void     ReOrder(Int_t nmet, Int_t *fmet);
   void     SetFirst(Int_t method);
   void     SetFirst(Int_t level, const char *details);

   const char *GetDetails(Int_t level);
   void        SetDetails(Int_t level, const char *details);

   const char *GetHost() const { return fHost; }
   const char *GetUser() const { return fUser; }

   void     SetHost(const char *host) { fHost = host; }
   void     SetUser(const char *user) { fUser = user; }

   TList   *Established() const { return fEstablished; }

   void     Print(Option_t *option = "") const;
   void     Print(const char *proc);
   void     PrintEstablished();

   ClassDef(THostAuth,0)  // Class providing host specific authentication information
};

#endif
