// @(#)root/net:$Name:  $:$Id: TAuthenticate.h,v 1.16 2003/11/20 23:00:46 rdm Exp $
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
#ifndef ROOT_TDatime
#include "TDatime.h"
#endif
#ifndef ROOT_rsafun
#include "rsafun.h"
#endif
#ifndef ROOT_AuthConst
#include "AuthConst.h"
#endif

class TAuthenticate;
class THostAuth;
class TSocket;
class TSecContext;

typedef Int_t (*CheckSecCtx_t)(const char *subj, TSecContext *ctx);
typedef Int_t (*GlobusAuth_t)(TAuthenticate *auth, TString &user, TString &det);
typedef Int_t (*Krb5Auth_t)(TAuthenticate *auth, TString &user, TString &det, Int_t version);
typedef Int_t (*SecureAuth_t)(TAuthenticate *auth, const char *user, const char *passwd,
                              const char *remote, TString &det, Int_t version);

class TAuthenticate : public TObject {

friend class TSecContext;
friend class TSocket;

public:
   enum ESecurity { kClear, kSRP, kKrb5, kGlobus, kSSH, kRfio }; // type of authentication

private:
   TString      fDetails;     // logon details (method dependent ...)
   THostAuth   *fHostAuth;    // pointer to relevant authentication info
   TString      fPasswd;      // user's password
   TString      fProtocol;    // remote service (rootd, proofd)
   Bool_t       fPwHash;      // kTRUE if fPasswd is a passwd hash
   TString      fRemote;      // remote host to which we want to connect
   Int_t        fRSAKey;      // Type of RSA key used
   TSecContext *fSecContext;  // pointer to relevant sec context
   ESecurity    fSecurity;    // actual logon security level
   TSocket     *fSocket;      // connection to remote daemon
   Bool_t       fSRPPwd;      // kTRUE if fPasswd is a SRP passwd
   Int_t        fVersion;     // 0,1,2, ... accordingly to remote daemon version
   TString      fUser;        // user to be authenticated

   Bool_t       Authenticate();
   Bool_t       CheckNetrc(TString &user, TString &passwd);
   Bool_t       CheckNetrc(TString &user, TString &passwd,
                           Bool_t &pwhash, Bool_t &srppwd);
   Int_t        GenRSAKeys();
   Bool_t       GetPwHash() const { return fPwHash; }
   Int_t        GetRSAKey() const { return fRSAKey; }
   TSecContext *GetSecContext() const { return fSecContext; }
   ESecurity    GetSecurity() const { return fSecurity; }
   Bool_t       GetSRPPwd() const { return fSRPPwd; }
   const char  *GetSshUser(TString user) const;
   Int_t        GetVersion() const { return fVersion; }
   Int_t        ClearAuth(TString &user, TString &passwd, Bool_t &pwhash);
   Bool_t       GetUserPasswd(TString &user, TString &passwd,
                              Bool_t &pwhash, Bool_t &srppwd);
   char        *GetRandString(Int_t Opt,Int_t Len);
   Int_t        RfioAuth(TString &user);
   void         SetEnvironment();
   Int_t        SshAuth(TString &user);

   static TList         *fgAuthInfo;
   static TString        fgAuthMeth[kMAXSEC];
   static Bool_t         fgAuthReUse;    // kTRUE is ReUse required
   static Int_t          fgClientProtocol;  // client protocol level
   static TString        fgDefaultUser;  // Default user information
   static TDatime        fgExpDate;      // Expiring date for new security contexts
   static GlobusAuth_t   fgGlobusAuthHook;
   static Krb5Auth_t     fgKrb5AuthHook;
   static TDatime        fgLastAuthrc;    // Time of last reading of fgRootAuthrc
   static TString        fgOptAuthrc;    // Last option used to parse fgRootAuthrc
   static TString        fgPasswd;
   static Bool_t         fgPromptUser;   // kTRUE if user prompt required
   static TList         *fgProofAuthInfo;  // Specific lists of THostAuth fro proof
   static Bool_t         fgPwHash;      // kTRUE if fgPasswd is a passwd hash
   static TString        fgRootAuthrc;    // Path to last rootauthrc-like file read
   static Int_t          fgRSAInit;
   static rsa_KEY        fgRSAPriKey;
   static rsa_KEY        fgRSAPubKey;
   static rsa_KEY_export fgRSAPubExport;
   static SecureAuth_t   fgSecAuthHook;
   static Bool_t         fgSRPPwd;      // kTRUE if fgPasswd is a SRP passwd
   static TString        fgUser;
   static Bool_t         fgUsrPwdCrypt;  // kTRUE if encryption for UsrPwd is required

   static Bool_t         CheckHost(const char *Host, const char *host);
   static Bool_t         CleanupSecContext(TSecContext *ctx, Bool_t all);
   static void           FileExpand(const char *fin, FILE *ftmp);
   static void           RemoveSecContext(TSecContext *ctx);

public:
   TAuthenticate(TSocket *sock, const char *remote, const char *proto,
                 const char *user = "");
   virtual ~TAuthenticate() { }

   Int_t              AuthExists(TString User, Int_t method, const char *Options,
                          Int_t *Message, Int_t *Rflag, CheckSecCtx_t funcheck);
   THostAuth         *GetHostAuth() const { return fHostAuth; }
   const char        *GetProtocol() const { return fProtocol; }
   const char        *GetRemoteHost() const { return fRemote; }
   TSocket           *GetSocket() const { return fSocket; }
   const char        *GetUser() const { return fUser; }
   void               SetSecContext(TSecContext *ctx) { fSecContext = ctx; }

   static void        AuthError(const char *where, Int_t error);
   static Bool_t      CheckProofAuth(Int_t cSec, TString &det);
   static void        CleanupSecContextAll();
   static void        DecodeRSAPublic(const char *rsapubexport, rsa_NUMBER &n, rsa_NUMBER &d);
   static TList      *GetAuthInfo();
   static const char *GetAuthMethod(Int_t idx);
   static Int_t       GetAuthMethodIdx(const char *meth);
   static Bool_t      GetAuthReUse();
   static Int_t       GetClientProtocol();
   static char       *GetDefaultDetails(Int_t method, Int_t opt, const char *user);
   static const char *GetDefaultUser();
   static TDatime     GetGlobalExpDate();
   static Bool_t      GetGlobalPwHash();
   static Bool_t      GetGlobalSRPPwd();
   static const char *GetGlobalUser();
   static GlobusAuth_t GetGlobusAuthHook();
   static THostAuth  *GetHostAuth(const char *host, const char *user="", 
                                  Option_t *opt = "R", Int_t *Exact = 0);
   static Bool_t      GetPromptUser();
   static TList      *GetProofAuthInfo();
   static Int_t       GetRSAInit();
   static const char *GetRSAPubExport();
   static THostAuth  *HasHostAuth(const char *host, const char *user, Option_t *opt = "R");
   static void        MergeHostAuthList(TList *Std, TList *New, Option_t *Opt = "");
   static char       *PromptPasswd(const char *prompt = "Password: ");
   static char       *PromptUser(const char *remote);
   static void        ReadProofConf(const char *proofconf);
   static Int_t       ReadRootAuthrc(const char *proofconf = 0);
   static void        RemoveHostAuth(THostAuth *ha, Option_t *opt = "");
   static Int_t       SecureRecv(TSocket *Socket, Int_t KeyType, char **Out);
   static Int_t       SecureSend(TSocket *Socket, Int_t KeyType, const char *In);
   static void        SendRSAPublicKey(TSocket *Socket);
   static void        SetAuthReUse(Bool_t authreuse);
   static void        SetDefaultUser(const char *defaultuser);
   static void        SetGlobalExpDate(TDatime expdate);
   static void        SetGlobalPasswd(const char *passwd);
   static void        SetGlobalPwHash(Bool_t pwhash);
   static void        SetGlobalSRPPwd(Bool_t srppwd);
   static void        SetGlobalUser(const char *user);
   static void        SetGlobusAuthHook(GlobusAuth_t func);
   static void        SetKrb5AuthHook(Krb5Auth_t func);
   static void        SetPromptUser(Bool_t promptuser);
   static void        SetRSAInit();
   static void        SetRSAPublic(const char *rsapubexport);
   static void        SetSecureAuthHook(SecureAuth_t func);
   static void        Show(Option_t *opt="S");

   ClassDef(TAuthenticate,0)  // Class providing remote authentication service
};

#endif
