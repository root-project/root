// @(#)root/net:$Name:$:$Id:$
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


class TAuthenticate : public TObject {

private:
   TString   fUser;      // user to be authentiated
   TString   fProtocol;  // remote service (root, roots, proof, proofs)
   TString   fRemote;    // remote host to which we want to connect
   TSocket  *fSocket;    // connection to remote daemon

   static const char   *fgUser;
   static const char   *fgPasswd;
   static SecureAuth_t  fgSecAuthHook;

   Bool_t CheckNetrc(char *&user, char *&passwd);

public:
   TAuthenticate(TSocket *sock, const char *proto, const char *remote);
   virtual ~TAuthenticate() { }

   Bool_t Authenticate(TString &user);

   static void  SetUser(const char *user);
   static void  SetPasswd(const char *passwd);
   static void  SetSecureAuthHook(SecureAuth_t func);
   static char *GetUser(const char *remote);
   static char *GetPasswd(const char *prompt = "Password: ");
   static void  AuthError(const char *where, Int_t error);

   ClassDef(TAuthenticate,0)  // Class providing remote authentication service
};

#endif
