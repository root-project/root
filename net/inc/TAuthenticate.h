// @(#)root/net:$Name:  $:$Id: TAuthenticate.h,v 1.4 2000/12/19 14:32:44 rdm Exp $
// Author: Fons Rademakers   26/11/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAuthenticate
#define ROOT_TAuthenticate


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAuthenticate                                                        //
//                                                                      //
// An authentication module for ROOT based network services, like rootd //
// and proofd.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TSocket;

typedef Int_t (*SecureAuth_t)(TSocket *sock, const char *user,
                              const char *passwd, const char *remote);
typedef Int_t (*Krb5Auth_t)(TSocket *sock, TString &user);


class TAuthenticate : public TObject {

public:
   enum ESecurity { kClear, kSRP, kKrb5 }; // type of authentication

private:
   TString   fUser;      // user to be authenticated
   TString   fPasswd;    // user's password
   TString   fProtocol;  // remote service (rootd, proofd)
   TString   fRemote;    // remote host to which we want to connect
   TSocket  *fSocket;    // connection to remote daemon
   ESecurity fSecurity;  // logon security level

   static TString       fgUser;
   static TString       fgPasswd;
   static SecureAuth_t  fgSecAuthHook;
   static Krb5Auth_t    fgKrb5AuthHook;

public:
   TAuthenticate(TSocket *sock, const char *remote, const char *proto,
                 Int_t security);
   virtual ~TAuthenticate() { }

   Bool_t      Authenticate();
   Bool_t      CheckNetrc(TString &user, TString &passwd);
   const char *GetUser() const { return fUser.Data(); }
   const char *GetPasswd() const { return fPasswd.Data(); }

   static const char *GetGlobalUser();
   static const char *GetGlobalPasswd();
   static void        SetGlobalUser(const char *user);
   static void        SetGlobalPasswd(const char *passwd);
   static char       *PromptUser(const char *remote);
   static char       *PromptPasswd(const char *prompt = "Password: ");
   static void        SetSecureAuthHook(SecureAuth_t func);
   static void        SetKrb5AuthHook(Krb5Auth_t func);
   static void        AuthError(const char *where, Int_t error);

   ClassDef(TAuthenticate,0)  // Class providing remote authentication service
};

#endif
