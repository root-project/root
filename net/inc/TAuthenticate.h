// @(#)root/net:$Name:  $:$Id: TAuthenticate.h,v 1.5 2002/03/20 18:47:30 rdm Exp $
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
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_THostAuth
#include "THostAuth.h"
#endif
#ifndef ROOT_rsafun
#include "rsafun.h"
#endif


// Number of security levels and masks
// (should be the same as in rpdutils/inc/rpdp.h)
const Int_t       kMAXSEC         = 6;
const Int_t       kMAXSECBUF      = 2048;
const Int_t       kAUTH_REUSE_MSK = 0x1;
const Int_t       kAUTH_CRYPT_MSK = 0x2;

class TSocket;
class TAuthenticate;

typedef Int_t (*SecureAuth_t)(TAuthenticate *auth, const char *user, const char *passwd,
                              const char *remote, TString &det, Int_t version);
typedef Int_t (*Krb5Auth_t)(TAuthenticate *auth, TString &user, TString &det, Int_t version);
typedef Int_t (*GlobusAuth_t)(TAuthenticate *auth, TString &user, TString &det);


class TAuthenticate : public TObject {

public:
   enum ESecurity { kClear, kSRP, kKrb5, kGlobus, kSSH, kRfio }; // type of authentication

private:
   TString    fUser;      // user to be authenticated
   TString    fPasswd;    // user's password
   TString    fProtocol;  // remote service (rootd, proofd)
   TString    fRemote;    // remote host to which we want to connect
   TSocket   *fSocket;    // connection to remote daemon
   ESecurity  fSecurity;  // actual logon security level
   TString    fDetails;   // logon details (method dependent ...)
   THostAuth *fHostAuth;  // pointer to relevant authentication info
   Int_t      fVersion;   // 0,1,2, ... accordingly to remote daemon version
   Int_t      fRSAKey;    // Type of RSA key used

   static TString        fgUser;
   static TString        fgPasswd;
   static SecureAuth_t   fgSecAuthHook;
   static Krb5Auth_t     fgKrb5AuthHook;
   static GlobusAuth_t   fgGlobusAuthHook;

   static TList          fgAuthInfo;

   static Int_t          fgRSAInit;
   static rsa_KEY        fgRSAPriKey;
   static rsa_KEY        fgRSAPubKey;
   static rsa_KEY_export fgRSAPubExport;

   void           SetEnvironment();
   Bool_t         GetUserPasswd(TString &user, TString &passwd);
   Int_t          ClearAuth(TString &user, TString &passwd);
   Int_t          RfioAuth(TString &user);
   Int_t          SshAuth(TString &user);

   char          *GetRandString(Int_t Opt,Int_t Len);
   Int_t          ProveId();

   static Int_t   CheckRootAuthrc(const char *Host, char ***user,
                                  Int_t **nmeth, Int_t **authmeth, char ***det);
   static Bool_t  CheckHost(const char *Host, const char *host);
   static Bool_t  CheckHostWild(const char *Host, const char *host);
   static void    FileExpand(char *fin, FILE *ftmp);
   static void    DecodeDetails(char *details, char *pt, char *ru, char **us);
   static void    DecodeDetailsGlobus(char *details, char *pt, char *ru,
                                      char **cd, char **cf, char **kf, char **ad);

public:
   TAuthenticate(TSocket *sock, const char *remote, const char *proto,
                 const char *user = "");
   virtual ~TAuthenticate() { }

   Bool_t             Authenticate();
   Bool_t             CheckNetrc(TString &user, TString &passwd);
   const char        *GetUser() const { return fUser; }
   const char        *GetPasswd() const { return fPasswd; }
   const char        *GetProtocol() const { return fProtocol; }
   const char        *GetSshUser() const;
   void               SetUser(const char *user) { fUser = user; }
   void               SetHostAuth(const char *host, const char *user);
   void               SetSecurity(Int_t fSec) { fSecurity = (ESecurity)fSec; }
   ESecurity          GetSecurity() const { return fSecurity; }
   const char        *GetDetails() const { return fDetails; }
   THostAuth         *GetHostAuth() const { return fHostAuth; }
   TSocket           *GetSocket() const { return fSocket; }
   void               SetVersion(Int_t fVer) { fVersion = fVer; }
   Int_t              GetVersion() const { return fVersion; }

   void               SetRSAKey(Int_t fKey) { fRSAKey = fKey; }
   Int_t              GetRSAKey() const { return fRSAKey; }
   Int_t              GenRSAKeys();

   static Int_t       GetAuthMeth(const char *Host, const char *protocol, char ***user,
                                  Int_t **nmeth, Int_t **authmeth, char ***det);
   static THostAuth  *GetHostAuth(const char *host, const char *user="");
   static void        RemoveHostAuth(THostAuth *ha);
   static void        ReadAuthRc(const char *host, const char *user="");
   static void        PrintHostAuth();

   static const char *GetGlobalUser();
   static const char *GetGlobalPasswd();
   static void        SetGlobalUser(const char *user);
   static void        SetGlobalPasswd(const char *passwd);
   static char       *PromptUser(const char *remote);
   static char       *PromptPasswd(const char *prompt = "Password: ");

   static void        SetSecureAuthHook(SecureAuth_t func);
   static void        SetKrb5AuthHook(Krb5Auth_t func);
   static void        SetGlobusAuthHook(GlobusAuth_t func);
   static GlobusAuth_t GetGlobusAuthHook();
   static const char *GetRSAPubExport();
   static Int_t       GetRSAInit();
   static void        SetRSAInit();

   static void        AuthError(const char *where, Int_t error);
   static Int_t       SecureSend(TSocket *Socket, Int_t KeyType, char *In);
   static Int_t       SecureRecv(TSocket *Socket, Int_t KeyType, char **Out);

   static Int_t       AuthExists(TAuthenticate *auth, Int_t method, TString &details,
                                 const char *Options, Int_t *Message, Int_t *Rflag);
   static void        SaveAuthDetails(TAuthenticate *auth, Int_t method, Int_t offset,
                                      Int_t reuse, TString &details, const char *rlogin,
                                      Int_t keytype, const char *token);
   static void        SetOffSet(THostAuth *hostauth, Int_t method, TString &details, Int_t offset);
   static Int_t       GetOffSet(TAuthenticate *auth, Int_t method, TString &details, char **token);
   static char       *GetDefaultDetails(Int_t method, Int_t opt, const char *user);
   static char       *GetRemoteLogin(THostAuth *hostauth, Int_t method, const char *details);

   static TList      *GetAuthInfo();

   ClassDef(TAuthenticate,0)  // Class providing remote authentication service
};

#endif
