// @(#)root/net:$Name:  $:$Id: TAuthenticate.cxx,v 1.59 2004/07/01 18:49:31 rdm Exp $
// Author: Fons Rademakers   26/11/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAuthenticate                                                        //
//                                                                      //
// An authentication module for ROOT based network services, like rootd //
// and proofd.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "config.h"

#include "TAuthenticate.h"
#include "TApplication.h"
#include "THostAuth.h"
#include "TSecContext.h"
#include "TPluginManager.h"
#include "TNetFile.h"
#include "TPSocket.h"
#include "TSystem.h"
#include "TError.h"
#include "Getline.h"
#include "TROOT.h"
#include "TEnv.h"
#include "TList.h"
#include "NetErrors.h"
#include "TRegexp.h"
#include "snprintf.h"

#ifndef R__LYNXOS
#include <sys/stat.h>
#endif
#include <errno.h>
#include <sys/types.h>
#include <time.h>
#if !defined(R__WIN32) && !defined(R__MACOSX) && !defined(R__FBSD)
#include <crypt.h>
#endif
#ifdef WIN32
#  include <io.h>
#endif /* WIN32 */
#if defined(R__FBSD)
#  include <unistd.h>
#endif

#if defined(R__ALPHA) || defined(R__SGI) || defined(R__MACOSX)
extern "C" char *crypt(const char *, const char *);
#endif

#ifdef R__GLBS
#   include <sys/ipc.h>
#   include <sys/shm.h>
#endif

#ifdef R__SSL
// SSL specific headers
#   include <openssl/bio.h>
#   include <openssl/err.h>
#   include <openssl/pem.h>
#   include <openssl/rand.h>
#   include <openssl/rsa.h>
#endif

// Statics initialization
TList         *TAuthenticate::fgAuthInfo = 0;
TString        TAuthenticate::fgAuthMeth[] = { "UsrPwd", "SRP", "Krb5",
                                               "Globus", "SSH", "UidGid" };
Bool_t         TAuthenticate::fgAuthReUse;
TString        TAuthenticate::fgDefaultUser;
TDatime        TAuthenticate::fgExpDate;
GlobusAuth_t   TAuthenticate::fgGlobusAuthHook;
Krb5Auth_t     TAuthenticate::fgKrb5AuthHook;
TString        TAuthenticate::fgKrb5Principal;
TDatime        TAuthenticate::fgLastAuthrc;    // Time of last reading of fgRootAuthrc
TString        TAuthenticate::fgPasswd;
Bool_t         TAuthenticate::fgPromptUser;
TList         *TAuthenticate::fgProofAuthInfo = 0;
Bool_t         TAuthenticate::fgPwHash;
Bool_t         TAuthenticate::fgReadHomeAuthrc = kTRUE; // on/off search for $HOME/.rootauthrc
TString        TAuthenticate::fgRootAuthrc;    // Path to last rootauthrc-like file read
Int_t          TAuthenticate::fgRSAKey  = -1;  // Default RSA key type to be used
Int_t          TAuthenticate::fgRSAInit = 0;
rsa_KEY        TAuthenticate::fgRSAPriKey;
rsa_KEY_export TAuthenticate::fgRSAPubExport[2] = {{0,0},{0,0}};
rsa_KEY        TAuthenticate::fgRSAPubKey;
#ifdef R__SSL
BF_KEY         TAuthenticate::fgBFKey;
#endif
SecureAuth_t   TAuthenticate::fgSecAuthHook;
Bool_t         TAuthenticate::fgSRPPwd;
TString        TAuthenticate::fgUser;
Bool_t         TAuthenticate::fgUsrPwdCrypt;

// Protocol changes (this was in TNetFile before)
// 6 -> 7: added support for ReOpen(), kROOTD_BYE and kROOTD_PROTOCOL2
// 7 -> 8: added support for update being a create (open stat = 2 and not 1)
// 8 -> 9: added new authentication features (see README.AUTH)
// 9 -> 10: added support for authenticated socket via TSocket::CreateAuthSocket(...)
// 10 -> 11: modified SSH protocol + support for server 'no authentication' mode
Int_t TAuthenticate::fgClientProtocol = 11;  // increase when client protocol changes

// Standar version of Sec Context match checking
Int_t StdCheckSecCtx(const char *, TSecContext *);

// ID of the main thread as unique identifier
static Int_t gProcessID = -1;

ClassImp(TAuthenticate)

//______________________________________________________________________________
TAuthenticate::TAuthenticate(TSocket *sock, const char *remote,
                             const char *proto, const char *user)
{
   // Create authentication object.

   // Use the ID of the starting thread as unique identifier
   if (gProcessID < 0)
      gProcessID = gSystem->GetPid();

   fSocket   = sock;
   fRemote   = remote;
   fHostAuth = 0;
   fVersion  = 5;                // The latest, by default
   fSecContext = 0;

   if (gDebug > 2)
      Info("TAuthenticate", "Enter: local host: %s, user is: %s (proto: %s)",
           gSystem->HostName(), user, proto);

   // Set protocol string.
   // Check if version should be different ...
   char *pdd;
   Int_t servtype = TSocket::kSOCKD;
   if (proto && strlen(proto) > 0) {
      char *sproto = StrDup(proto);
      if ((pdd = strstr(sproto, ":")) != 0) {
         int rproto = atoi(pdd + 1);
         *pdd = '\0';
         if (strstr(sproto, "root") != 0) {
            if (rproto < 12 ) {
               fVersion = 4;
               if (rproto < 11 ) {
                  fVersion = 3;
                  if (rproto < 9 ) {
                     fVersion = 2;
                     if (rproto < 8) {
                        fVersion = 1;
                        if (rproto < 6)
                           fVersion = 0;
                     }
                  }
               }
            }
            servtype = TSocket::kROOTD;
         }
         if (strstr(sproto, "proof") != 0) {
            if (rproto < 11) {
               fVersion = 4;
               if (rproto < 10) {
                  fVersion = 3;
                  if (rproto < 8) {
                     fVersion = 2;
                     if (rproto < 7)
                        fVersion = 1;
                  }
               }
            }
            servtype = TSocket::kPROOFD;
         }
         if (gDebug > 3)
            Info("TAuthenticate",
                 "service: %s (remote protocol: %d): fVersion: %d", sproto,
                 rproto, fVersion);
      }
      fProtocol = sproto;
      delete [] sproto;
   }

   // Check or get user name
   fUser = "";
   TString CheckUser;
   if (user && strlen(user) > 0) {
      fUser = user;
      CheckUser = user;
   } else {
      UserGroup_t *u = gSystem->GetUserInfo();
      if (u)
         CheckUser = u->fUser;
      delete u;
   }
   fPasswd = "";
   fPwHash = kFALSE;
   fSRPPwd = kFALSE;

   // Type of RSA key
   if (fgRSAKey < 0) {
      fgRSAKey  = 0;                // Default key
#if R__SSL
      // Another choice possible: check user preferences
      if (gEnv->GetValue("RSA.KeyType",0) == 1)
         fgRSAKey = 1;
#endif
   }
   // This is the key actually used: we propose the default
   // to the server, and behave according to its reply
   fRSAKey = fgRSAKey;
   if (gDebug > 3)
      Info("TAuthenticate","RSA key: default type %d", fgRSAKey);

   // RSA key generation (one per session)
   if (!fgRSAInit) {
      GenRSAKeys();
      fgRSAInit = 1;
   }

   // Check and save the host FQDN ...
   TString fqdn;
   TInetAddress addr = gSystem->GetHostByName(fRemote);
   if (addr.IsValid()) {
      fqdn = addr.GetHostName();
      if (fqdn == "UnNamedHost")
         fqdn = addr.GetHostAddress();
   }
   TString fqdnsrv(Form("%s:%d",fqdn.Data(),servtype));

   // Read directives from files; re-read if files have changed
   TAuthenticate::ReadRootAuthrc();

   if (gDebug > 3) {
      Info("TAuthenticate",
           "number of HostAuth Instantiations in memory: %d",
           GetAuthInfo()->GetSize());
      TAuthenticate::Show("H");
      TAuthenticate::Show("P");
   }

   // Check the list of auth info for already loaded info about this host
   fHostAuth = GetHostAuth(fqdnsrv, CheckUser);

   // If for whatever (and unlikely) reason nothing has been found
   // we look for the old envs defaulting to method 0 (UsrPwd)
   // if they are missing or meaningless
   if (!fHostAuth) {

      TString Tmp;
      if (fProtocol.Contains("proof")) {
         Tmp = TString(gEnv->GetValue("Proofd.Authentication", "0"));
      } else if (fProtocol.Contains("root")) {
         Tmp = TString(gEnv->GetValue("Rootd.Authentication", "0"));
      }
      char am[kMAXSEC][10];
      Int_t nw = sscanf(Tmp.Data(), "%s %s %s %s %s %s",
                        am[0], am[1], am[2], am[3], am[4], am[5]);

      Int_t i = 0, nm = 0, me[kMAXSEC];
      for( ; i < nw; i++) {
         Int_t met = -1;
         if (strlen(am[i]) > 1) {
            met = GetAuthMethodIdx(am[i]);
         } else {
            met = atoi(am[i]);
         }
         if (met > -1 && met < kMAXSEC) {
            me[nm++] = met;
         }
      }

      // Create THostAuth
      if (nm)
         fHostAuth = new THostAuth(fRemote,fUser,nm,me,0);
      else
         fHostAuth = new THostAuth(fRemote,fUser,0,(const char *)0);
   }

   // If a specific method has been requested via the protocol
   // set it as first
   Int_t sec = -1;
   TString tmp = fProtocol;
   tmp.ReplaceAll("root",4,"",0);
   tmp.ReplaceAll("proof",5,"",0);
   if (!strncmp(tmp.Data(),"up",2))
      sec = 0;
   else if (!strncmp(tmp.Data(),"s",1))
      sec = 1;
   else if (!strncmp(tmp.Data(),"k",1))
      sec = 2;
   else if (!strncmp(tmp.Data(),"g",1))
      sec = 3;
   else if (!strncmp(tmp.Data(),"h",1))
      sec = 4;
   else if (!strncmp(tmp.Data(),"ug",2))
      sec = 5;
   if (sec > -1 && sec < kMAXSEC) {
      if (fHostAuth->HasMethod(sec)) {
         fHostAuth->SetFirst(sec);
      } else {
         char *dtmp = GetDefaultDetails(sec, 1, CheckUser);
         TString det(dtmp);
         fHostAuth->AddFirst(sec, det);
         if (dtmp)
            delete[] dtmp;
      }
   }

   // This is what we have in memory
   if (gDebug > 3) {
      TIter next(fHostAuth->Established());
      TSecContext *ctx;
      while ((ctx = (TSecContext *) next()))
         ctx->Print("0");
   }
}

//______________________________________________________________________________
Bool_t TAuthenticate::Authenticate()
{
   // Authenticate to remote rootd or proofd server. Return kTRUE if
   // authentication succeeded.

   Int_t RemMeth = 0, rMth[kMAXSEC], tMth[kMAXSEC] = {0};
   Int_t meth = 0;
   char NoSupport[80] = { 0 };
   char TriedMeth[80] = { 0 };

   TString user, passwd;
   Bool_t pwhash;

   Int_t ntry = 0;
   if (gDebug > 2)
      Info("Authenticate", "enter: fUser: %s", fUser.Data());
   NoSupport[0] = 0;

 negotia:
   tMth[meth] = 1;
   if (gDebug > 2) {
      ntry++;
      Info("Authenticate", "try #: %d", ntry);
   }

   user = "";
   passwd = "";
   pwhash = kFALSE;

   // Security level from the list (if not in cleanup mode ...)
   fSecurity = (ESecurity) fHostAuth->GetMethod(meth);
   fDetails = fHostAuth->GetDetails((Int_t) fSecurity);
   if (gDebug > 2)
      Info("Authenticate",
           "trying authentication: method:%d, default details:%s",
           fSecurity, fDetails.Data());

   // Keep track of tried methods in a list
   if (strlen(TriedMeth) > 0)
      sprintf(TriedMeth, "%s %s", TriedMeth, fgAuthMeth[fSecurity].Data());
   else
      sprintf(TriedMeth, "%s", fgAuthMeth[fSecurity].Data());

   // Set environments
   SetEnvironment();

   // This is for dynamic loads ...
#ifdef ROOTLIBDIR
   TString RootDir = TString(ROOTLIBDIR);
#else
   TString RootDir = TString(gRootDir) + "/lib";
#endif

   // Auth calls depend of fSec
   Int_t st = -1;
   if (fSecurity == kClear) {

      Bool_t rc = kFALSE;

      // UsrPwd Authentication
      user = fgDefaultUser;
      if (user != "")
         CheckNetrc(user, passwd, pwhash, kFALSE);
      if (passwd == "") {
         if (fgPromptUser)
            user = PromptUser(fRemote);
         rc = GetUserPasswd(user, passwd, pwhash, kFALSE);
      }
      fUser = user;
      fPasswd = passwd;

      if (!rc) {

         if (fUser != "root")
            st = ClearAuth(user, passwd, pwhash);
      } else {
         Error("Authenticate",
               "unable to get user name for UsrPwd authentication");
      }

   } else if (fSecurity == kSRP) {

      Bool_t rc = kFALSE;

      // SRP Authentication
      user = fgDefaultUser;
      if (user != "")
         CheckNetrc(user, passwd, pwhash, kTRUE);
      if (passwd == "") {
         if (fgPromptUser)
            user = PromptUser(fRemote);
         rc = GetUserPasswd(user, passwd, pwhash, kTRUE);
      }
      fUser = user;
      fPasswd = passwd;

      if (!fgSecAuthHook) {

         char *p;
         TString lib = RootDir + "/libSRPAuth";
         if ((p = gSystem->DynamicPathName(lib, kTRUE))) {
            delete[]p;
            gSystem->Load(lib);
         }
      }
      if (!rc && fgSecAuthHook) {

         st = (*fgSecAuthHook) (this, user, passwd, fRemote, fDetails,
                                fVersion);
      } else {
         if (!fgSecAuthHook)
            Error("Authenticate",
                  "no support for SRP authentication available");
         if (rc)
            Error("Authenticate",
                  "unable to get user name for SRP authentication");
      }
      // Fill present user info ...
      if (st == 1) {
         fPwHash = kFALSE;
         fSRPPwd = kTRUE;
      }

   } else if (fSecurity == kKrb5) {

      if (fVersion > 0) {

         // Kerberos 5 Authentication
         if (!fgKrb5AuthHook) {
            char *p;
            TString lib = RootDir + "/libKrb5Auth";
            if ((p = gSystem->DynamicPathName(lib, kTRUE))) {
               delete[]p;
               gSystem->Load(lib);
            }
         }
         if (fgKrb5AuthHook) {
            fUser = fgDefaultUser;
            st = (*fgKrb5AuthHook) (this, fUser, fDetails, fVersion);
         } else {
            Error("Authenticate",
                  "support for kerberos5 auth locally unavailable");
         }
      } else {
         if (gDebug > 0)
            Info("Authenticate",
                 "remote daemon does not support Kerberos authentication");
         if (strlen(NoSupport) > 0)
            sprintf(NoSupport, "%s/Krb5", NoSupport);
         else
            sprintf(NoSupport, "Krb5");
      }

   } else if (fSecurity == kGlobus) {
      if (fVersion > 1) {

         // Globus Authentication
         if (!fgGlobusAuthHook) {
            char *p;
            TString lib = RootDir + "/libGlobusAuth";
            if ((p = gSystem->DynamicPathName(lib, kTRUE))) {
               delete[]p;
               gSystem->Load(lib);
            }
         }
         if (fgGlobusAuthHook) {
            st = (*fgGlobusAuthHook) (this, fUser, fDetails);
         } else {
            Error("Authenticate",
                  "no support for Globus authentication available");
         }
      } else {
         if (gDebug > 0)
            Info("Authenticate",
                 "remote daemon does not support Globus authentication");
         if (strlen(NoSupport) > 0)
            sprintf(NoSupport, "%s/Globus", NoSupport);
         else
            sprintf(NoSupport, "Globus");
      }


   } else if (fSecurity == kSSH) {

      if (fVersion > 1) {

         // SSH Authentication
         st = SshAuth(fUser);

      } else {
         if (gDebug > 0)
            Info("Authenticate",
                 "remote daemon does not support SSH authentication");
         if (strlen(NoSupport) > 0)
            sprintf(NoSupport, "%s/SSH", NoSupport);
         else
            sprintf(NoSupport, "SSH");
      }

   } else if (fSecurity == kRfio) {

      if (fVersion > 1) {

         // UidGid Authentication
         st = RfioAuth(fUser);

      } else {
         if (gDebug > 0)
            Info("Authenticate",
                 "remote daemon does not support UidGid authentication");
         if (strlen(NoSupport) > 0)
            sprintf(NoSupport, "%s/UidGid", NoSupport);
         else
            sprintf(NoSupport, "UidGid");
      }
   }
   // Analyse the result now ...
   Int_t kind, stat;
   if (st == 1) {
      fHostAuth->CountSuccess((Int_t)fSecurity);
      if (gDebug > 2)
         fSecContext->Print();
      if (fSecContext->IsActive())
         fSecContext->AddForCleanup(fSocket->GetPort(),
             fSocket->GetRemoteProtocol(),fSocket->GetServType());
      return kTRUE;
   } else {
      fHostAuth->CountFailure((Int_t)fSecurity);
      if (fVersion > 2) {
         if (st == -2) {
            // Remote host does not accepts connections from local host
            return kFALSE;
         }
      }
      Int_t nmet = fHostAuth->NumMethods();
      Int_t remloc = nmet - meth - 1;
      if (gDebug > 2)
         Info("Authenticate",
              "got st=%d: still %d methods locally available",
              st, remloc);
      if (st == -1) {
         if (gDebug > 2)
            Info("Authenticate",
                 "method not even started: insufficient or wrong info: %s",
                 "try with next method, if any");
         if (meth < nmet - 1) {
            meth++;
            goto negotia;
         } else if (strlen(NoSupport) > 0)
            Info("Authenticate",
                 "attempted methods %s are not supported by remote server version",
                 NoSupport);
         Info("Authenticate",
              "failure: list of attempted methods: %s", TriedMeth);
         return kFALSE;
      } else {
         if (fVersion < 2) {
            if (gDebug > 2)
               Info("Authenticate",
                    "negotiation not supported remotely: try next method, if any");
            if (meth < nmet - 1) {
               meth++;
               goto negotia;
            } else if (strlen(NoSupport) > 0)
               Info("Authenticate",
                    "attempted methods %s are not supported by remote server version",
                    NoSupport);
            Info("Authenticate",
                 "failure: list of attempted methods: %s", TriedMeth);
            return kFALSE;
         }
         // Attempt negotiation ...
         if (fSocket->Recv(stat, kind) < 0)
            return kFALSE;
         if (gDebug > 2)
            Info("Authenticate",
                 "after failed attempt: kind= %d, stat= %d", kind, stat);
         if (kind == kROOTD_ERR) {
            if (gDebug > 0)
               AuthError("Authenticate", stat);
            Info("Authenticate",
                 "failure: list of attempted methods: %s", TriedMeth);
            return kFALSE;
         } else if (kind == kROOTD_NEGOTIA) {
            if (stat > 0) {
               int len = 3 * stat;
               char *answer = new char[len];
               int nrec = fSocket->Recv(answer, len, kind);  // returns user
               if (nrec < 0)
                  return kFALSE;
               if (kind != kMESS_STRING)
                  Warning("Authenticate",
                          "strings with accepted methods not received (%d:%d)",
                          kind, nrec);
               RemMeth =
                   sscanf(answer, "%d %d %d %d %d %d", &rMth[0], &rMth[1],
                          &rMth[2], &rMth[3], &rMth[4], &rMth[5]);
               if (gDebug > 0 && remloc > 0)
                  Info("Authenticate",
                       "remotely allowed methods not yet tried: %s",
                       answer);
            } else if (stat == 0) {
               Info("Authenticate",
                    "no more methods accepted remotely to be tried");
               if (strlen(NoSupport) > 0)
                  Info("Authenticate",
                       "attempted methods %s are not supported"
                       " by remote server version",NoSupport);
               Info("Authenticate",
                    "failure: list of attempted methods: %s", TriedMeth);
               return kFALSE;
            }
            // If no more local methods, exit
            if (remloc < 1) {
               if (strlen(NoSupport) > 0)
                  Info("Authenticate",
                       "attempted methods %s are not supported"
                       " by remote server version",NoSupport);
               Info("Authenticate",
                    "failure: list of attempted methods: %s", TriedMeth);
               return kFALSE;
            }
            // Look if a non tried method matches
            int i, j;
            char lav[40] = { 0 };
            for (i = 0; i < RemMeth; i++) {
               for (j = 0; j < nmet; j++) {
                  if (fHostAuth->GetMethod(j) == rMth[i] && tMth[j] == 0) {
                     meth = j;
                     goto negotia;
                  }
                  if (i == 0)
                     sprintf(lav, "%s %d", lav, fHostAuth->GetMethod(j));
               }
            }
            if (gDebug > 0)
               Warning("Authenticate",
                       "no match with those locally available: %s",
                       lav);
            if (strlen(NoSupport) > 0)
               Info("Authenticate",
                    "attempted methods %s are not supported by"
                    " remote server version",NoSupport);
            Info("Authenticate",
                 "failure: list of attempted methods: %s", TriedMeth);
            return kFALSE;
         } else                 // unknown message code at this stage
         if (strlen(NoSupport) > 0)
            Info("Authenticate",
                 "attempted methods %s are not supported by remote server version",
                 NoSupport);
            Info("Authenticate",
                 "failure: list of attempted methods: %s", TriedMeth);
         return kFALSE;
      }
   }
}

//______________________________________________________________________________
void TAuthenticate::SetEnvironment()
{
   // Set default authentication environment. The values are inferred
   // from fSecurity and fDetails.

   if (gDebug > 2)
      Info("SetEnvironment",
           "setting environment: fSecurity:%d, fDetails:%s", fSecurity,
           fDetails.Data());

   // Defaults
   fgDefaultUser = fgUser;
   if (fSecurity == kKrb5)
      fgAuthReUse = kFALSE;
   else
      fgAuthReUse = kTRUE;
   fgPromptUser = kFALSE;

   // Decode fDetails, is non empty ...
   if (fDetails != "") {
      char UsDef[kMAXPATHLEN] = { 0 };
      Int_t lDet = strlen(fDetails.Data()) + 2;
      char Pt[5] = { 0 }, Ru[5] = { 0 };
      Int_t hh = 0, mm = 0;
      char *Us = 0, *Cd = 0, *Cf = 0, *Kf = 0, *Ad = 0, *Cp = 0, *Pp = 0;
      const char *ptr;

      TString UsrPromptDef = TString(GetAuthMethod(fSecurity)) + ".LoginPrompt";
      if ((ptr = strstr(fDetails, "pt:")) != 0) {
         sscanf(ptr + 3, "%s %s", Pt, UsDef);
      } else {
         if (!strncasecmp(gEnv->GetValue(UsrPromptDef,""),"no",2) ||
             !strncmp(gEnv->GetValue(UsrPromptDef,""),"0",1))
            strcpy(Pt,"0");
         else
            strcpy(Pt,"1");
      }
      TString UsrReUseDef = TString(GetAuthMethod(fSecurity)) + ".ReUse";
      if ((ptr = strstr(fDetails, "ru:")) != 0) {
         sscanf(ptr + 3, "%s %s", Ru, UsDef);
      } else {
         if (!strncasecmp(gEnv->GetValue(UsrReUseDef,""),"no",2) ||
             !strncmp(gEnv->GetValue(UsrReUseDef,""),"0",1))
            strcpy(Ru,"0");
         else
            strcpy(Ru,"1");
      }
      TString UsrValidDef = TString(GetAuthMethod(fSecurity)) + ".Valid";
      TString Hours(gEnv->GetValue(UsrValidDef,"24:00"));
      Int_t pd = 0;
      if ((pd = Hours.Index(":")) > -1) {
         TString Minutes = Hours;
         Hours.Resize(pd);
         Minutes.Replace(0,pd+1,"");
         hh = atoi(Hours.Data());
         mm = atoi(Minutes.Data());
      } else {
         hh = atoi(Hours.Data());
         mm = 0;
      }

      // Now action depends on method ...
      if (fSecurity == kGlobus) {
         Cd = new char[lDet];
         Cf = new char[lDet];
         Kf = new char[lDet];
         Ad = new char[lDet];
         Cd[0] = '\0';
         Cf[0] = '\0';
         Kf[0] = '\0';
         Ad[0] = '\0';
         if ((ptr = strstr(fDetails, "cd:")) != 0)
            sscanf(ptr, "%s %s", Cd, UsDef);
         if ((ptr = strstr(fDetails, "cf:")) != 0)
            sscanf(ptr, "%s %s", Cf, UsDef);
         if ((ptr = strstr(fDetails, "kf:")) != 0)
            sscanf(ptr, "%s %s", Kf, UsDef);
         if ((ptr = strstr(fDetails, "ad:")) != 0)
            sscanf(ptr, "%s %s", Ad, UsDef);
         if (gDebug > 2) {
            Info("SetEnvironment",
                 "details:%s, Pt:%s, Ru:%s, Cd:%s, Cf:%s, Kf:%s, Ad:%s",
                 fDetails.Data(), Pt, Ru, Cd, Cf, Kf, Ad);
         }
      } else if (fSecurity == kClear) {
         Us = new char[lDet];
         Us[0] = '\0';
         Cp = new char[lDet];
         Cp[0] = '\0';
         if ((ptr = strstr(fDetails, "us:")) != 0)
            sscanf(ptr + 3, "%s %s", Us, UsDef);
         if ((ptr = strstr(fDetails, "cp:")) != 0)
            sscanf(ptr + 3, "%s %s", Cp, UsDef);
         if (gDebug > 2)
            Info("SetEnvironment", "details:%s, Pt:%s, Ru:%s, Us:%s Cp:%s",
                 fDetails.Data(), Pt, Ru, Us, Cp);
      } else if (fSecurity == kKrb5) {
         Us = new char[lDet];
         Us[0] = '\0';
         Pp = new char[lDet];
         Pp[0] = '\0';
         if ((ptr = strstr(fDetails, "us:")) != 0)
            sscanf(ptr + 3, "%s %s", Us, UsDef);
         if ((ptr = strstr(fDetails, "pp:")) != 0)
            sscanf(ptr + 3, "%s %s", Pp, UsDef);
         if (gDebug > 2)
            Info("SetEnvironment", "details:%s, Pt:%s, Ru:%s, Us:%s Pp:%s",
                 fDetails.Data(), Pt, Ru, Us, Pp);
      } else {
         Us = new char[lDet];
         Us[0] = '\0';
         if ((ptr = strstr(fDetails, "us:")) != 0)
            sscanf(ptr + 3, "%s %s", Us, UsDef);
         if (gDebug > 2)
            Info("SetEnvironment", "details:%s, Pt:%s, Ru:%s, Us:%s",
                 fDetails.Data(), Pt, Ru, Us);
      }

      // Set Prompt flag
      if (!strncasecmp(Pt, "yes",3) || !strncmp(Pt, "1", 1))
         fgPromptUser = kTRUE;

      // Set ReUse flag
      if (fSecurity == kKrb5) {
         fgAuthReUse = kFALSE;
         if (!strncasecmp(Ru, "yes",3) || !strncmp(Ru, "1",1))
            fgAuthReUse = kTRUE;
      } else {
         fgAuthReUse = kTRUE;
         if (!strncasecmp(Ru, "no",2) || !strncmp(Ru, "0",1))
            fgAuthReUse = kFALSE;
      }

      // Set Expiring date
      fgExpDate = TDatime();
      fgExpDate.Set(fgExpDate.Convert() + hh*3600 + mm*60);

      // UnSet Crypt flag for UsrPwd, if requested
      if (fSecurity == kClear) {
         fgUsrPwdCrypt = kTRUE;
         if (!strncmp(Cp, "no", 2) || !strncmp(Cp, "0", 1))
            fgUsrPwdCrypt = kFALSE;
      }
      // Build UserDefaults
      UsDef[0] = '\0';
      if (fSecurity == kGlobus) {
         if (Cd != 0) {
            strcat(UsDef," ");
            strcat(UsDef,Cd);
            delete[] Cd;
            Cd = 0;
         }
         if (Cf != 0) {
            strcat(UsDef," ");
            strcat(UsDef,Cf);
            delete[] Cf;
            Cf = 0;
         }
         if (Kf != 0) {
            strcat(UsDef," ");
            strcat(UsDef,Kf);
            delete[] Kf;
            Kf = 0;
         }
         if (Ad != 0) {
            strcat(UsDef," ");
            strcat(UsDef,Ad);
            delete[] Ad;
            Ad = 0;
         }
      } else {
         if (fSecurity == kKrb5) {
            // Collect info about principal, if any
            if (Pp && strlen(Pp) > 0) {
               fgKrb5Principal = TString(Pp);
               delete[] Pp;
               Pp = 0;
            } else {
               // Allow specification via 'us:' key
               if (Us && strlen(Us) > 0 && strstr(Us,"@")) {
                  fgKrb5Principal = TString(Us);
                  delete[] Us;
                  Us = 0;
               }
            }
            // command line user specification (fUser) gets highest priority
            if (fUser.Length()) {
               sprintf(UsDef, "%s", fUser.Data());
            } else {
               if (Us && strlen(Us) > 0 && !strstr(Us,"@")) {
                  sprintf(UsDef, "%s", Us);
                  delete[] Us;
                  Us = 0;
               }
            }
            if (Us != 0) {
               delete[] Us;
               Us = 0;
            }
            if (Pp != 0) {
               delete[] Pp;
               Pp = 0;
            }
         } else {
            // give highest priority to command-line specification
            if (fUser == "") {
               if (Us != 0) {
                  sprintf(UsDef, "%s", Us);
                  delete[] Us;
                  Us = 0;
               }
            } else
               sprintf(UsDef, "%s", fUser.Data());
         }
      }
      if (strlen(UsDef) > 0) {
         fgDefaultUser = UsDef;
      } else {
         if (fgUser != "") {
            fgDefaultUser = fgUser;
         } else {
            UserGroup_t *u = gSystem->GetUserInfo();
            if (u)
               fgDefaultUser = u->fUser;
            delete u;
         }
      }
      if (fgDefaultUser == "anonymous" || fgDefaultUser == "rootd" ||
          // when set by user don't prompt for it anymore
          fgUser != "" || fUser != "")
         fgPromptUser = kFALSE;

      if (gDebug > 2)
         Info("SetEnvironment", "UsDef:%s", fgDefaultUser.Data());

      if (Us) delete[] Us;
      if (Cd) delete[] Cd;
      if (Cf) delete[] Cf;
      if (Kf) delete[] Kf;
      if (Ad) delete[] Ad;
      if (Cp) delete[] Cp;
      if (Pp) delete[] Pp;
   }
}

//______________________________________________________________________________
Bool_t TAuthenticate::GetUserPasswd(TString &user, TString &passwd,
                                    Bool_t &pwhash, Bool_t srppwd)
{
   // Try to get user name and passwd from several sources.

   if (gDebug > 3)
      Info("GetUserPasswd", "Enter: User: '%s' Hash:%d SRP:%d",
            user.Data(),(Int_t)pwhash,(Int_t)srppwd);

   // Get user and passwd set via static functions SetUser and SetPasswd.
   if (user == "") {
      if (fgUser != "")
         user = fgUser;
      if (passwd == "" && fgPasswd != "" && srppwd == fgSRPPwd) {
         passwd = fgPasswd;
         pwhash = fgPwHash;
      }
   } else {
      if (fgUser != "" && user == fgUser) {
         if (passwd == "" && fgPasswd != "" && srppwd == fgSRPPwd) {
             passwd = fgPasswd;
             pwhash = fgPwHash;
         }
      }
   }
   if (gDebug > 3)
      Info("GetUserPasswd", "In memory: User: '%s' Hash:%d",
            user.Data(),(Int_t)pwhash);

   // Check system info for user if still not defined
   if (user == "") {
      UserGroup_t *u = gSystem->GetUserInfo();
      if (u)
         user = u->fUser;
      delete u;
      if (gDebug > 3)
         Info("GetUserPasswd", "In memory: User: '%s' Hash:%d",
            user.Data(),(Int_t)pwhash);
   }

   // Check ~/.rootnetrc and ~/.netrc files if user was not set via
   // the static SetUser() method.
   if (user == "" || passwd == "") {
      if (gDebug > 3)
         Info("GetUserPasswd", "Checking .netrc family ...");
      CheckNetrc(user, passwd, pwhash, srppwd);
   }
   if (gDebug > 3)
      Info("GetUserPasswd", "From .netrc family: User: '%s' Hash:%d",
            user.Data(),(Int_t)pwhash);

   // If user also not set via  ~/.rootnetrc or ~/.netrc ask user.
   if (user == "") {
      user = PromptUser(fRemote);
      if (user == "") {
         Error("GetUserPasswd", "user name not set");
         return 1;
      }
   }

   return 0;
}

//______________________________________________________________________________
Bool_t TAuthenticate::CheckNetrc(TString &user, TString &passwd)
{
   // Try to get user name and passwd from the ~/.rootnetrc or
   // ~/.netrc files. For more info see the version with 4 arguments.
   // This version is maintained for backward compatability reasons.

   Bool_t hash, srppwd;

   // Set srppwd flag
   srppwd = (fSecurity == kSRP) ? kTRUE : kFALSE;

   return CheckNetrc(user, passwd, hash, srppwd);
}

//______________________________________________________________________________
Bool_t TAuthenticate::CheckNetrc(TString &user, TString &passwd,
                                 Bool_t &pwhash, Bool_t srppwd)
{
   // Try to get user name and passwd from the ~/.rootnetrc or
   // ~/.netrc files. First ~/.rootnetrc is tried, after that ~/.netrc.
   // These files will only be used when their access masks are 0600.
   // Returns kTRUE if user and passwd were found for the machine
   // specified in the URL. If kFALSE, user and passwd are "".
   // If srppwd == kTRUE then a SRP ('secure') pwd is searched for in
   // the files.
   // The boolean pwhash is set to kTRUE if the returned passwd is to
   // be understood as password hash, i.e. if the 'password-hash' keyword
   // is found in the 'machine' lines; not implemented for 'secure'
   // and the .netrc file.
   // The format of these files are:
   //
   // # this is a comment line
   // machine <machine fqdn> login <user> password <passwd>
   // machine <machine fqdn> login <user> password-hash <passwd>
   //
   // and in addition ~/.rootnetrc also supports:
   //
   // secure <machine fqdn> login <user> password <passwd>
   //
   // for the secure protocols. All lines must start in the first column.

   Bool_t result = kFALSE;
   Bool_t first = kTRUE;
   TString remote = fRemote;

   passwd = "";
   pwhash = kFALSE;

   char *net =
       gSystem->ConcatFileName(gSystem->HomeDirectory(), ".rootnetrc");

   // Determine FQDN of the host ...
   TInetAddress addr = gSystem->GetHostByName(fRemote);
   if (addr.IsValid()) {
      remote = addr.GetHostName();
      if (remote == "UnNamedHost")
         remote = addr.GetHostAddress();
   }

 again:
#ifdef WIN32
   // Since Win32 does not have proper protections use file always
   FILE * fd1;
   if ((fd1 = fopen(net, "r"))) {
      fclose(fd1);
      if (1) {
#else
   // Only use file when its access rights are 0600
   struct stat buf;
   if (stat(net, &buf) == 0) {
      if (S_ISREG(buf.st_mode) && !S_ISDIR(buf.st_mode) &&
          (buf.st_mode & 0777) == (S_IRUSR | S_IWUSR)) {
#endif
         FILE *fd = fopen(net, "r");
         char line[256];
         while (fgets(line, sizeof(line), fd) != 0) {
            if (line[0] == '#')
               continue;
            char word[6][64];
            int nword = sscanf(line, "%s %s %s %s %s %s", word[0], word[1],
                               word[2], word[3], word[4], word[5]);
            if (nword != 6)
               continue;
            if (srppwd && strcmp(word[0], "secure"))
               continue;
            if (!srppwd && strcmp(word[0], "machine"))
               continue;
            if (strcmp(word[2], "login"))
               continue;
            if (srppwd && strcmp(word[4], "password"))
               continue;
            if (!srppwd &&
               strcmp(word[4], "password") && strcmp(word[4], "password-hash"))
               continue;

            // Determine FQDN of the host name found in the file ...
            TString host_tmp = word[1];
            TInetAddress addr = gSystem->GetHostByName(word[1]);
            if (addr.IsValid()) {
               host_tmp = addr.GetHostName();
               if (host_tmp == "UnNamedHost")
                  host_tmp = addr.GetHostAddress();
            }

            if (host_tmp == remote) {
               if (user == "") {
                  user = word[3];
                  passwd = word[5];
                  if (!strcmp(word[4], "password-hash"))
                     pwhash = kTRUE;
                  result = kTRUE;
                  break;
               } else {
                  if (!strcmp(word[3], user.Data())) {
                     passwd = word[5];
                     if (!strcmp(word[4], "password-hash"))
                        pwhash = kTRUE;
                     result = kTRUE;
                     break;
                  }
               }
            }
         }
         fclose(fd);
      } else
         Warning("CheckNetrc",
                 "file %s exists but has not 0600 permission", net);
   }
   delete[]net;

   if (first && !srppwd && !result) {
      net = gSystem->ConcatFileName(gSystem->HomeDirectory(), ".netrc");
      first = kFALSE;
      goto again;
   }

   return result;
}

//______________________________________________________________________________
const char *TAuthenticate::GetGlobalUser()
{
   // Static method returning the global user.

   return fgUser;
}

//______________________________________________________________________________
Bool_t TAuthenticate::GetGlobalPwHash()
{
   // Static method returning the global password hash flag.

   return fgPwHash;
}

//______________________________________________________________________________
Bool_t TAuthenticate::GetGlobalSRPPwd()
{
   // Static method returning the global SRP password flag.

   return fgSRPPwd;
}

//______________________________________________________________________________
TDatime TAuthenticate::GetGlobalExpDate()
{
   // Static method returning default expiring date for new validity contexts

   return fgExpDate;
}

//______________________________________________________________________________
const char *TAuthenticate::GetDefaultUser()
{
   // Static method returning the default user information.

   return fgDefaultUser;
}

//______________________________________________________________________________
const char *TAuthenticate::GetKrb5Principal()
{
   // Static method returning the principal to be used to init Krb5 tickets.

   return fgKrb5Principal;
}

//______________________________________________________________________________
Bool_t TAuthenticate::GetAuthReUse()
{
   // Static method returning the authentication reuse settings.

   return fgAuthReUse;
}

//______________________________________________________________________________
Bool_t TAuthenticate::GetPromptUser()
{
   // Static method returning the prompt user settings.

   return fgPromptUser;
}

//______________________________________________________________________________
const char *TAuthenticate::GetAuthMethod(Int_t idx)
{
   // Static method returning the method corresponding to idx.

   if (idx < 0 || idx > kMAXSEC-1) {
      ::Error("Authenticate::GetAuthMethod", "idx out of bounds (%d)", idx);
      idx = 0;
   }
   return fgAuthMeth[idx];
}

//______________________________________________________________________________
Int_t TAuthenticate::GetAuthMethodIdx(const char *meth)
{
   // Static method returning the method index (which can be used to find
   // the method in GetAuthMethod()). Returns -1 in case meth is not found.

   if (meth && meth[0]) {
      for (Int_t i = 0; i < kMAXSEC; i++) {
         if (!fgAuthMeth[i].CompareTo(meth, TString::kIgnoreCase))
            return i;
      }
   }

   return -1;
}

//______________________________________________________________________________
char *TAuthenticate::PromptUser(const char *remote)
{
   // Static method to prompt for the user name to be used for authentication
   // to rootd or proofd. User is asked to type user name.
   // Returns user name (which must be deleted by caller) or 0.
   // If non-interactive run (eg ProofServ) returns default user.

   const char *user;
   if (fgDefaultUser != "")
      user = fgDefaultUser;
   else
      user = gSystem->Getenv("USER");
#ifdef R__WIN32
   if (!user)
      user = gSystem->Getenv("USERNAME");
#endif
   if (isatty(0) == 0 || isatty(1) == 0) {
      ::Warning("TAuthenticate::PromptUser",
                "not tty: cannot prompt for user, returning default");
      if (strlen(user))
         return StrDup(user);
      else
         return StrDup("None");
   }

   char *usr = Getline(Form("Name (%s:%s): ", remote, user));
   if (usr[0]) {
      usr[strlen(usr) - 1] = 0; // get rid of \n
      if (strlen(usr))
         return StrDup(usr);
      else
         return StrDup(user);
   }
   return 0;
}

//______________________________________________________________________________
char *TAuthenticate::PromptPasswd(const char *prompt)
{
   // Static method to prompt for the user's passwd to be used for
   // authentication to rootd or proofd. Uses non-echoing command line
   // to get passwd. Returns passwd (which must de deleted by caller) or 0.
   // If non-interactive run (eg ProofServ) returns -1

   if (isatty(0) == 0 || isatty(1) == 0) {
      ::Warning("TAuthenticate::PromptPasswd",
                "not tty: cannot prompt for passwd, returning -1");
      static char noint[4] = {"-1"};
      return StrDup(noint);
   }

   Gl_config("noecho", 1);
   char *pw = Getline((char *) prompt);
   Gl_config("noecho", 0);
   if (pw[0]) {
      pw[strlen(pw) - 1] = 0;   // get rid of \n
      return StrDup(pw);
   }
   return 0;
}

//______________________________________________________________________________
GlobusAuth_t TAuthenticate::GetGlobusAuthHook()
{
   // Static method returning the globus authorization hook.

   return fgGlobusAuthHook;
}

//______________________________________________________________________________
const char *TAuthenticate::GetRSAPubExport(Int_t key)
{
   // Static method returning the RSA public keys.

   key = (key >= 0 && key <= 1) ? key : 0;
   return fgRSAPubExport[key].keys;
}

//______________________________________________________________________________
Int_t TAuthenticate::GetRSAInit()
{
   // Static method returning the RSA initialization flag.

   return fgRSAInit;
}

//______________________________________________________________________________
void TAuthenticate::SetDefaultRSAKeyType(Int_t key)
{
   // Static method setting the default type of RSA key.

   if (key >= 0 && key <= 1)
      fgRSAKey = key;
}

//______________________________________________________________________________
void TAuthenticate::SetRSAInit()
{
   // Static method setting RSA initialization flag.

   fgRSAInit = 1;
}

//______________________________________________________________________________
TList *TAuthenticate::GetAuthInfo()
{
   // Static method returning the list with authentication details.

   if (!fgAuthInfo)
      fgAuthInfo = new TList;
   return fgAuthInfo;
}

//______________________________________________________________________________
TList *TAuthenticate::GetProofAuthInfo()
{
   // Static method returning the list with authentication directives
   // to be sent to proof.

   if (!fgProofAuthInfo)
      fgProofAuthInfo = new TList;
   return fgProofAuthInfo;
}

//______________________________________________________________________________
void TAuthenticate::AuthError(const char *where, Int_t err)
{
   // Print error string depending on error code.

   ::Error(Form("TAuthenticate::%s", where), gRootdErrStr[err]);
}

//______________________________________________________________________________
void TAuthenticate::SetGlobalUser(const char *user)
{
   // Set global user name to be used for authentication to rootd or proofd.

   if (fgUser != "")
      fgUser = "";

   if (user && user[0])
      fgUser = user;
}

//______________________________________________________________________________
void TAuthenticate::SetGlobalPasswd(const char *passwd)
{
   // Set global passwd to be used for authentication to rootd or proofd.

   if (fgPasswd != "")
      fgPasswd = "";

   if (passwd && passwd[0])
      fgPasswd = passwd;
}

//______________________________________________________________________________
void TAuthenticate::SetGlobalPwHash(Bool_t pwhash)
{
   // Set global passwd hash flag to be used for authentication to rootd or proofd.

   fgPwHash = pwhash;
}

//______________________________________________________________________________
void TAuthenticate::SetGlobalSRPPwd(Bool_t srppwd)
{
   // Set global SRP passwd flag to be used for authentication to rootd or proofd.

   fgSRPPwd = srppwd;
}

//______________________________________________________________________________
void TAuthenticate::SetReadHomeAuthrc(Bool_t readhomeauthrc)
{
   // Set flag controlling the reading of $HOME/.rootauthrc.
   // In PROOF the administrator may want to switch off private settings.
   // Always true, may only be set false via option to proofd.

   fgReadHomeAuthrc = readhomeauthrc;
}

//______________________________________________________________________________
void TAuthenticate::SetGlobalExpDate(TDatime expdate)
{
   // Set default expiring date for new validity contexts

   fgExpDate = expdate;
}

//______________________________________________________________________________
void TAuthenticate::SetDefaultUser(const char *defaultuser)
{
   // Set default user name.

   if (fgDefaultUser != "")
      fgDefaultUser = "";

   if (defaultuser && defaultuser[0])
      fgDefaultUser = defaultuser;
}

//______________________________________________________________________________
void TAuthenticate::SetAuthReUse(Bool_t authreuse)
{
   // Set global AuthReUse flag

   fgAuthReUse = authreuse;
}

//______________________________________________________________________________
void TAuthenticate::SetPromptUser(Bool_t promptuser)
{
   // Set global PromptUser flag

   fgPromptUser = promptuser;
}

//______________________________________________________________________________
void TAuthenticate::SetSecureAuthHook(SecureAuth_t func)
{
   // Set secure authorization function. Automatically called when libSRPAuth
   // is loaded.

   fgSecAuthHook = func;
}

//______________________________________________________________________________
void TAuthenticate::SetKrb5AuthHook(Krb5Auth_t func)
{
   // Set kerberos5 authorization function. Automatically called when
   // libKrb5Auth is loaded.

   fgKrb5AuthHook = func;
}

//______________________________________________________________________________
void TAuthenticate::SetGlobusAuthHook(GlobusAuth_t func)
{
   // Set Globus authorization function. Automatically called when
   // libGlobusAuth is loaded.

   fgGlobusAuthHook = func;
}

//______________________________________________________________________________
Int_t TAuthenticate::SshError(const char *errorfile)
{
   // SSH error parsing: returns
   //     0  :  no error or fatal
   //     1  :  should retry (eg 'connection closed by remote host')

   Int_t error = 0;

   if (!gSystem->AccessPathName(errorfile, kReadPermission)) {
      FILE *ferr = fopen(errorfile,"r");
      if (ferr) {
         // Get list of errors for which one should retry
         char *serr = StrDup(gEnv->GetValue("SSH.ErrorRetry", ""));
         // Prepare for parsing getting rid of '"'s
         Int_t lerr = strlen(serr);
         char *pc = (char *)memchr(serr,'"',lerr);
         while (pc) {
           *pc = '\0';
            pc = (char *)memchr(pc+1,'"',strlen(pc+1));
         }
         // Now read the file
         char line[kMAXPATHLEN];
         while (fgets(line,sizeof(line),ferr)) {
            // Get rid of trailing '\n'
            if (line[strlen(line)-1] == '\n')
               line[strlen(line)-1] = '\0';
            if (gDebug > 2)
               Info("SshError","read line: %s",line);
            pc = serr;
            while (pc < serr + lerr) {
               if (pc[0] == '\0' || pc[0] == ' ')
                  pc++;
               else {
                  if (gDebug > 2)
                     Info("SshError","checking error: '%s'",pc);
                  if (strstr(line,pc))
                     error = 1;
                  pc += strlen(pc);
               }
            }
         }
         // Close file
         fclose(ferr);
         // Free allocated memory
         if (serr) delete[] serr;
      }
   }
   return error;
}

//______________________________________________________________________________
Int_t TAuthenticate::SshAuth(TString &User)
{
   // SSH client authentication code.

   Int_t sshproto = 1;
   if (fVersion < 4)
      sshproto = 0;

   // Find out whihc command we should be using
   char cmdref[2][5] = {"ssh", "scp"};
   char scmd[5] = "";
   char *gSshExe = 0;
   Bool_t notfound = kTRUE;

   while (notfound && sshproto > -1) {

      strcpy(scmd,cmdref[sshproto]);

      // Check First if a 'scmd' executable exists ...
      gSshExe = gSystem->Which(gSystem->Getenv("PATH"),
                               scmd, kExecutePermission);
      if (!gSshExe) {
         if (gDebug > 2)
            Info("SshAuth", "%s not found in $PATH", scmd);

         // Still allow for client definition of the ssh location ...
         if (strcmp(gEnv->GetValue("SSH.ExecDir", "-1"), "-1")) {
            if (gDebug > 2)
               Info("SshAuth", "searching user defined path ...");
            if (gSshExe) delete[] gSshExe;
            gSshExe = StrDup(Form("%s/%s",
                            (char *)gEnv->GetValue("SSH.ExecDir", ""), scmd));
            if (gSystem->AccessPathName(gSshExe, kExecutePermission)) {
               if (gDebug > 2)
                  Info("SshAuth", "%s not executable", gSshExe);
            } else
               notfound = kFALSE;
         }
      } else
         notfound = kFALSE;
      if (notfound) sshproto--;
   }

   // Check if the command was found
   if (notfound) {
      if (gSshExe) delete[] gSshExe;
      return -1;
   }
   if (gDebug > 2)
      Info("SshAuth", "%s is %s (sshproto: %d)", scmd, gSshExe, sshproto);

   // SSH-like authentication code.
   // Returns 0 in case authentication failed
   //         1 in case of success
   //        -1 in case of the remote node does not seem to support
   //           SSH-like Authentication
   //        -2 in case of the remote node does not seem to allow
   //           connections from this node

   char SecName[kMAXPATHLEN] = { 0 };

   // Determine user name ...
   User = GetSshUser(User);

   // Check ReUse
   Int_t ReUse = (int)fgAuthReUse;
   fDetails = TString(Form("pt:%d ru:%d us:",(int)fgPromptUser,(int)fgAuthReUse))
            + User;

   // Create Options string
   int Opt = ReUse * kAUTH_REUSE_MSK + fRSAKey * kAUTH_RSATY_MSK;
   TString Options(Form("%d none %ld %s %d", Opt,
                       (Long_t)User.Length(),User.Data(),sshproto));

   // Check established authentications
   Int_t kind = kROOTD_SSH;
   Int_t retval = ReUse;
   Int_t rc = 0;
   if ((rc = AuthExists(User, (Int_t) TAuthenticate::kSSH, Options,
                   &kind, &retval, &StdCheckSecCtx)) == 1) {
      // A valid authentication exists: we are done ...
      return 1;
   }
   if (rc == -2) {
      return rc;
   }
   if (retval == kErrNotAllowed && kind == kROOTD_ERR) {
      return 0;
   }
   // Check return flags
   if (kind != kROOTD_SSH)
      return 0;                 // something went wrong
   if (retval == 0)
      return 0;                 // no remote support for SSH
   if (retval == -2)
      return 0;                 // user unkmown to remote host

   // Wait for the server to communicate remote pid and location
   // of command to execute
   char cmdinfo[kMAXPATHLEN] = { 0 };
   Int_t reclen = (retval+1 > kMAXPATHLEN) ? kMAXPATHLEN : retval+1 ;
   if (fSocket->Recv(cmdinfo, reclen, kind) < 0)
      return 0;
   if (kind != kROOTD_SSH)
      return 0;                 // something went wrong
   if (gDebug > 3)
      Info("SshAuth", "received from server command info: %s", cmdinfo);

   int rport = -1;
   char *pd = 0;
   if ((pd = strstr(cmdinfo, ":")) != 0) {
      Int_t clen = (Int_t) (pd - cmdinfo) - 1;
      cmdinfo[clen] = '\0';
      char ss[2][20] = {{0},{0}};
      Int_t ns = sscanf(pd-1,"%s %s",ss[0],ss[1]);
      Int_t i = 0;
      for( ; i < ns; i++ ) {
         if (strlen(ss[i])) {
            if (!strncmp(ss[i],"p:",2)) {
               char *pp = (char *)(ss[i])+2;
               rport = atoi(pp);
            }
#ifdef R__SSL
            if (!strncmp(ss[i],"k:",2)) {
               char *pk = (char *)(ss[i])+2;
               if (atoi(pk) == 1)
                  fRSAKey = 1;
            }
#endif
         }
      }
   }

   // If we are a non-interactive session we cannot reply
   TString noPrompt = "";
   if (isatty(0) == 0 || isatty(1) == 0) {
     noPrompt  = TString("-o 'PasswordAuthentication no' ");
     noPrompt += TString("-o 'StrictHostKeyChecking no' ");
     if (gDebug > 3)
        Info("SshAuth", "using noprompt options: %s", noPrompt.Data());
   }

   // Send authentication request to remote sshd
   // Create command
   int ssh_rc = 1;
   Int_t ntry = gEnv->GetValue("SSH.MaxRetry",100);
   TString fileErr = "";
   if (sshproto == 0) {
      // Prepare local file first in the home directory
      TString fileErr = "rootsshtmp_";
      FILE *floc = gSystem->TempFileName(fileErr,gSystem->HomeDirectory());
      if (floc == 0) {
         // Try the temp directory
         fileErr = "rootsshtmp_";
         floc = gSystem->TempFileName(fileErr);
      }
      fileErr.Append(".error");
      TString sshcmd(Form("%s -x -l %s %s",
                          gSshExe, User.Data(), noPrompt.Data()));
      if (rport != -1)
         sshcmd += TString(Form(" -p %d",rport));
      sshcmd += TString(Form(" %s %s",fRemote.Data(), cmdinfo));
      sshcmd += TString(Form(" 1> /dev/null 2> %s",fileErr.Data()));

      // Execute command
      Int_t again = 1;
      while (ssh_rc && again && ntry--) {
         ssh_rc = gSystem->Exec(sshcmd);
         if (ssh_rc) {
            again = SshError(fileErr);
            if (gDebug > 3)
               Info("SshAuth", "%d: sleeping: rc: %d, again:%d, ntry: %d",
                         gProcessID, ssh_rc, again, ntry);
            if (again)
               gSystem->Sleep(1);
         }
      }
   } else {
      // Prepare local file first in the home directory
      TString fileLoc = "rootsshtmp_";
      FILE *floc = gSystem->TempFileName(fileLoc,gSystem->HomeDirectory());
      if (floc == 0) {
         // Try the temp directory
         fileLoc = "rootsshtmp_";
         floc = gSystem->TempFileName(fileLoc);
      }

      if (floc != 0) {
         // Close file and change permissions before filling it
         fclose(floc);
         if (chmod(fileLoc, 0600) == -1) {
            Info("SshAuth", "fchmod error: %d", errno);
            ssh_rc = 2;
         } else {
            floc = fopen(fileLoc, "w");
            if (ReUse == 1) {
               // Send our public key
               if (fVersion > 4) {
                  fprintf(floc,"k: %d\n",fRSAKey+1);
                  fwrite(fgRSAPubExport[fRSAKey].keys,1,
                         fgRSAPubExport[fRSAKey].len,floc);
               } else {
                  fprintf(floc,"k: %s\n",fgRSAPubExport[0].keys);
               }
            } else
               // Just a notification
               fprintf(floc,"k: -1\n");
            fclose(floc);
            ssh_rc = 0;
         }
         if (!ssh_rc) {
            fileErr = TString(fileLoc).Append(".error");
            TString sshcmd(Form("%s -p %s", gSshExe, noPrompt.Data()));
            if (rport != -1)
               sshcmd += TString(Form(" -P %d",rport));
            sshcmd += TString(Form(" %s",fileLoc.Data()));
            sshcmd += TString(Form(" %s@%s:%s 1> /dev/null",
                                   User.Data(),fRemote.Data(),cmdinfo));
            sshcmd += TString(Form(" 2> %s",fileErr.Data()));
            // Execute command
            ssh_rc = 1;
            Int_t again = 1;
            while (ssh_rc && again && ntry--) {
               ssh_rc = gSystem->Exec(sshcmd);
               if (ssh_rc) {
                  again = SshError(fileErr);
                  if (gDebug > 3)
                     Info("SshAuth", "%d: sleeping: rc: %d, again:%d, ntry: %d",
                               gProcessID, ssh_rc, again, ntry);
                  if (again)
                     // Wait 1 sec before retry
                     gSystem->Sleep(1000);
               }
            }
         }
      } else {
         // Problems creating temporary file: return ...
         ssh_rc = 1;
      }
      // Remove the file after use ...
      if (!gSystem->AccessPathName(fileLoc,kFileExists)) {
         gSystem->Unlink(fileLoc);
      }
   }
   // Remove the file after use ...
   if (!gSystem->AccessPathName(fileErr,kFileExists)) {
      gSystem->Unlink(fileErr);
   }
   if (gDebug > 3)
      Info("SshAuth", "%d: system return code: %d (%d)",
                      gProcessID, ssh_rc, ntry+1);

   if (ssh_rc && sshproto == 0) {

      Int_t srvtyp = fSocket->GetServType();
      Int_t rproto = fSocket->GetRemoteProtocol();
      Int_t level = 2;
      if ((srvtyp == TSocket::kROOTD && rproto < 10) ||
          (srvtyp == TSocket::kPROOFD && rproto < 9))
         level = 1;
      if ((srvtyp == TSocket::kROOTD && rproto < 8) ||
          (srvtyp == TSocket::kPROOFD && rproto < 7))
         level = 0;
      if (level) {
         Int_t port = fSocket->GetPort();
         TSocket *newsock = 0;
         TString url(Form("sockd://%s",fRemote.Data()));
         if (srvtyp == TSocket::kROOTD) {
            // Parallel socket requested by 'rootd'
            url.ReplaceAll("sockd",5,"rootd",5);
            newsock = new TPSocket(url.Data(),port,1,-1);
         } else {
            if (srvtyp == TSocket::kPROOFD)
               url.ReplaceAll("sockd",5,"proofd",6);
            newsock = new TSocket(fRemote.Data(),port,-1);
            if (srvtyp == TSocket::kPROOFD)
               newsock->Send("failure notification");
         }
         // prepare info to send
         char cd1[1024], pipe[1024], dum[1024];
         Int_t id3;
         sscanf(cmdinfo, "%s %d %s %s", cd1, &id3, pipe, dum);
         snprintf(SecName, kMAXPATHLEN, "%d -1 0 %s %d %s %d",
                  -gProcessID, pipe,
                 (int)strlen(User), User.Data(), fgClientProtocol);
         newsock->Send(SecName, kROOTD_SSH);
         if (level > 1) {
            // Improved diagnostics
            // Receive diagnostics message
            if (newsock->Recv(retval, kind) >= 0) {
               char *Buf = new char[retval+1];
               if (newsock->Recv(Buf, retval+1, kind) >= 0) {
                  if (strncmp(Buf,"OK",2)) {
                     Info("SshAuth", "from remote host %s:", fRemote.Data());
                     Info("SshAuth", ">> nothing listening on port %s %s",Buf,
                                     "(supposed to be associated to sshd)");
                     Info("SshAuth", ">> contact the daemon administrator at %s",
                                     fRemote.Data());
                  } else {
                     if (gDebug > 0) {
                        Info("SshAuth", "from remote host %s:", fRemote.Data());
                        Info("SshAuth", ">> something listening on the port"
                                     " supposed to be associated to sshd.");
                        Info("SshAuth", ">> You have probably mistyped your"
                                        " password. Or you tried to hack the"
                                        " system.");
                        Info("SshAuth", ">> If the problem persists you may"
                                        " consider contacting the daemon");
                        Info("SshAuth", ">> administrator at %s.",fRemote.Data());
                     }
                  }
               }
               if (Buf)
                  delete[] Buf;
            }
         }
         SafeDelete(newsock);
         // Receive error message
         if (fSocket->Recv(retval, kind) >= 0) {  // for consistency
            if (kind == kROOTD_ERR) {
               if (gDebug > 0)
                  AuthError("SshAuth", retval);
            }
         }
      }
      return 0;
   } else if (ssh_rc && sshproto > 0) {
      // Communicate failure
      if (fSocket->Send("0", kROOTD_SSH) < 0)
         Info("SshAuth", "error communicating failure");
      return 0;
   }

   // Communicate success
   if (sshproto > 0) {
      if (fSocket->Send("1", kROOTD_SSH) < 0)
         Info("SshAuth", "error communicating success");
   }

   Int_t nrec = 0;
   // Receive key request info and type of key (if ok, error otherwise)
   if ((nrec = fSocket->Recv(retval, kind)) < 0)  // returns user
      return 0;
   if (gDebug > 3)
      Info("SshAuth", "got message %d, flag: %d", kind, retval);

   // Check if an error occured
   if (kind == kROOTD_ERR) {
      if (gDebug > 0)
         AuthError("SshAuth", retval);
      return 0;
   }

   if (ReUse == 1 && sshproto == 0) {

      // Save type of key
      if (kind != kROOTD_RSAKEY  || retval < 1 || retval > 2)
         Warning("SshAuth",
                 "problems recvn RSA key flag: got message %d, flag: %d",
                 kind, retval);

      fRSAKey = retval - 1;

      // Send the key securely
      if (SendRSAPublicKey(fSocket,fRSAKey) < 0)
         return 0;

      // Receive username used for login
      if ((nrec = fSocket->Recv(retval, kind)) < 0)  // returns user
         return 0;
      if (gDebug > 3)
         Info("SshAuth", "got message %d, flag: %d", kind, retval);
   }

   if (kind != kROOTD_SSH || retval < 1) {
      Warning("SshAuth",
              "problems recvn (user,offset) length (%d:%d bytes:%d)", kind,
              retval, nrec);
      return 0;
   }

   char answer[256];
   reclen = (retval+1 > 256) ? 256 : retval+1;
   if ((nrec = fSocket->Recv(answer, reclen, kind)) < 0)  // returns user
      return 0;
   if (kind != kMESS_STRING)
      Warning("SshAuth", "username and offset not received (%d:%d)", kind,
               nrec);

   // Parse answer
   char lUser[128];
   int OffSet = -1;
   sscanf(answer, "%s %d", lUser, &OffSet);
   if (gDebug > 3)
      Info("SshAuth", "received from server: user: %s, offset: %d", lUser,
           OffSet);

   // Receive Token
   char *Token = 0;
   if (ReUse == 1 && OffSet > -1) {
      if (SecureRecv(fSocket, 1, fRSAKey, &Token) == -1) {
         Warning("SshAuth", "problems secure-receiving token -"
                            " may result in corrupted token");
         return 0;
      }
      if (gDebug > 3)
         Info("SshAuth", "received from server: token: '%s' ", Token);
   } else {
      Token = StrDup("");
   }

   // Create SecContext object
   fSecContext = fHostAuth->CreateSecContext((const char *)lUser, fRemote,
                 (Int_t)kSSH, OffSet, fDetails,
                 (const char *)Token, fgExpDate, 0, fRSAKey);

   // Release allocated memory ...
   if (Token) delete[] Token;

   // Get and Analyse the reply
   if (fSocket->Recv(retval, kind) < 0)
      return 0;
   if (gDebug > 3)
      Info("SshAuth", "received from server: kind: %d, retval: %d", kind,
           retval);

   if (kind != kROOTD_AUTH) {
      return 0;
   } else {
      return retval;
   }
}

//______________________________________________________________________________
const char *TAuthenticate::GetSshUser(TString User) const
{
   // Method returning the User to be used for the ssh login.
   // Looks first at SSH.Login and finally at env USER.
   // If SSH.LoginPrompt is set to 'yes' it prompts for the 'login name'

   static TString user = "";

   if (User == "") {
      if (fgPromptUser) {
         user = PromptUser(fRemote);
      } else {
         user = fgDefaultUser;
         if (user == "")
            user = PromptUser(fRemote);
      }
   } else {
      user = User;
   }

   return user;
}

//______________________________________________________________________________
Bool_t TAuthenticate::CheckHost(const char *Host, const char *host)
{
   // Check if 'Host' matches 'host':
   // this means either equal or "containing" it, even with wild cards *
   // in the first field (in the case 'host' is a name, ie not IP address)
   // Returns kTRUE if the two matches.

   Bool_t retval = kTRUE;

   // Both strings should have been defined
   if (!Host || !host)
      return kFALSE;

   // 'host' == '*' indicates any 'Host' ...
   if (!strcmp(host,"*"))
      return kTRUE;

   // If 'host' contains at a letter or an hyphen it is assumed to be
   // a host name. Otherwise a name.
   // Check also for wild cards
   Bool_t name = kFALSE;
   TRegexp rename("[+a-zA-Z]");
   Int_t len;
   if (rename.Index(host,&len) != -1 || strstr(host,"-"))
      name = kTRUE;

   // Check also for wild cards
   Bool_t wild = kFALSE;
   if (strstr(host,"*"))
      wild = kTRUE;

   // Now build the regular expression for final checking
   TRegexp rehost(host,wild);

   // Host to check
   TString theHost(Host);
   if (!name) {
      TInetAddress addr = gSystem->GetHostByName(Host);
      theHost = addr.GetHostAddress();
      if (gDebug > 2)
         ::Info("TAuthenticate::CheckHost", "checking host IP: %s", theHost.Data());
   }

   // Check 'Host' against 'rehost'
   Ssiz_t pos = rehost.Index(theHost,&len);
   if (pos == -1)
      retval = kFALSE;

   // If IP and no wilds, it should match either
   // the beginning or the end of the string
   if (!wild) {
      if (pos > 0 && pos != (Ssiz_t)(theHost.Length()-strlen(host)))
         retval = kFALSE;
   }

   return retval;
}

//______________________________________________________________________________
Int_t TAuthenticate::RfioAuth(TString &User)
{
   // UidGid client authentication code.
   // Returns 0 in case authentication failed
   //         1 in case of success
   //        <0 in case of system error

   if (gDebug > 2)
      Info("RfioAuth", "enter ... User %s", User.Data());

   // Get user info ... ...
   UserGroup_t *pw = gSystem->GetUserInfo(gSystem->GetEffectiveUid());
   if (pw) {

      // These are the details to be saved in case of success ...
      User = pw->fUser;
      fDetails = TString("pt:0 ru:0 us:") + User;

      // Check that we are not root ...
      if (pw->fUid != 0) {

         UserGroup_t *grp = gSystem->GetGroupInfo(gSystem->GetEffectiveGid());

         // Get effective user & group ID associated with the current process...
         Int_t uid = pw->fUid;
         Int_t gid = grp ? grp->fGid : pw->fGid;

         delete grp;

         // Send request ....
         TString sstr = TString(Form("%d %d", uid, gid));
         if (gDebug > 3)
            Info("RfioAuth", "sending ... %s", sstr.Data());
         Int_t ns = 0;
         if ((ns = fSocket->Send(sstr.Data(), kROOTD_RFIO)) < 0)
            return 0;
         if (gDebug > 3)
            Info("RfioAuth", "sent ... %d bytes (expected > %d)", ns,
                 sstr.Length());

         // Get answer
         Int_t stat, kind;
         if (fSocket->Recv(stat, kind) < 0)
            return 0;
         if (gDebug > 3)
            Info("RfioAuth", "after kROOTD_RFIO: kind= %d, stat= %d", kind,
                 stat);

         // Query result ...
         if (kind == kROOTD_AUTH && stat >= 1) {
            // Create inactive SecContext object for use in TSocket
            fSecContext =
               fHostAuth->CreateSecContext((const char *)pw->fUser,
                                  fRemote, kRfio, -stat, fDetails, 0);
            delete pw;
            return 1;
         } else {
            TString Server = "sockd";
            if (fProtocol.Contains("root"))
               Server = "rootd";
            if (fProtocol.Contains("proof"))
               Server = "proofd";

            // Authentication failed
            if (stat == kErrConnectionRefused) {
               if (gDebug > 0)
                  Error("RfioAuth",
                     "%s@%s does not accept connections from %s%s",
                     Server.Data(),fRemote.Data(),
                     fUser.Data(),gSystem->HostName());
               delete pw;
               return -2;
            } else if (stat == kErrNotAllowed) {
               if (gDebug > 0)
                  Error("RfioAuth",
                     "%s@%s does not accept %s authentication from %s@%s",
                     Server.Data(),fRemote.Data(),
                     TAuthenticate::fgAuthMeth[5].Data(),
                     fUser.Data(),gSystem->HostName());
            } else {
              if (gDebug > 0)
                 AuthError("RfioAuth", stat);
            }
            delete pw;
            return 0;
         }
      } else {
         Warning("RfioAuth", "UidGid login as \"root\" not allowed");
         return -1;
      }
   }
   delete pw;
   return -1;
}

//______________________________________________________________________________
Int_t TAuthenticate::ClearAuth(TString &User, TString &Passwd, Bool_t &PwHash)
{
   // UsrPwd client authentication code.
   // Returns 0 in case authentication failed
   //         1 in case of success

   if (gDebug > 2)
      Info("ClearAuth", "enter: User: %s (passwd hashed?: %d)",
                        User.Data(),(Int_t)PwHash);

   Int_t ReUse    = fgAuthReUse;
   Int_t Prompt   = fgPromptUser;
   Int_t Crypt    = fgUsrPwdCrypt;
   Int_t NeedSalt = 1;
   if (PwHash)
     NeedSalt = 0;
   fDetails = TString(Form("pt:%d ru:%d cp:%d us:",
                           fgPromptUser, fgAuthReUse, fgUsrPwdCrypt)) + User;
   if (gDebug > 2)
      Info("ClearAuth", "ru:%d pt:%d cp:%d ns:%d rk:%d",
           fgAuthReUse,fgPromptUser,fgUsrPwdCrypt,NeedSalt,fgRSAKey);
#ifdef R__WIN32
   Crypt = 0;
#endif
   Int_t stat, kind;

   if (fVersion > 1) {

      //
      // New protocol
      //
      Int_t Anon = 0;
      TString Salt = "";
      TString PasHash = "";

      // Get effective user (fro remote checks in $HOME/.rhosts)
      UserGroup_t *pw = gSystem->GetUserInfo(gSystem->GetEffectiveUid());
      TString EffUser;
      if (pw) {
         EffUser = TString(pw->fUser);
         delete pw;
      } else
         EffUser = User;

      // Create Options string
      int Opt = (ReUse * kAUTH_REUSE_MSK) + (Crypt * kAUTH_CRYPT_MSK) +
                (NeedSalt * kAUTH_SSALT_MSK) + (fRSAKey * kAUTH_RSATY_MSK);
      TString Options(Form("%d %ld %s %ld %s", Opt,
                          (Long_t)User.Length(), User.Data(),
                          (Long_t)EffUser.Length(), EffUser.Data()));

      // Check established authentications
      kind = kROOTD_USER;
      stat = ReUse;
      Int_t rc = 0;
      if ((rc = AuthExists(User, (Int_t) TAuthenticate::kClear, Options,
                           &kind, &stat, &StdCheckSecCtx)) == 1) {
         // A valid authentication exists: we are done ...
         return 1;
      }
      if (rc == -2) {
         return rc;
      }
      if (stat == kErrNotAllowed && kind == kROOTD_ERR) {
         return 0;
      }

      if (kind == kROOTD_AUTH && stat == -1) {
         if (gDebug > 3)
            Info("ClearAuth", "anonymous user");
         Anon  = 1;
         Crypt = 0;
         ReUse = 0;
         NeedSalt = 0;
      }

      if (Anon == 0 && Crypt == 1) {

         // Check that we got the right thing ..
         if (kind != kROOTD_RSAKEY || stat < 1 || stat > 2 ) {
            // Check for errors
            if (kind == kROOTD_ERR) {
               AuthError("ClearAuth", stat);
            } else {
               Warning("ClearAuth",
                       "problems recvn RSA key flag: got message %d, flag: %d",
                       kind, stat);
            }
            return 0;
         }
         if (gDebug > 3)
            Info("ClearAuth", "get key request ...");

         // Save type of key
         fRSAKey = stat - 1;

         // Send the key securely
         if (SendRSAPublicKey(fSocket,fRSAKey) < 0)
            return 0;

         int Slen = 0;
         if (NeedSalt) {
            // Receive password salt
            char *TmpSalt = 0;
            if ((Slen = SecureRecv(fSocket, 1, fRSAKey, &TmpSalt)) == -1) {
               Warning("ClearAuth", "problems secure-receiving salt -"
                       " may result in corrupted salt");
               Warning("ClearAuth", "switch off reuse for this session");
               NeedSalt = 0;
               return 0;
            }
            if (Slen) {
               Salt = TString(TmpSalt);
               delete[] TmpSalt;
            }
            if (gDebug > 2)
               Info("ClearAuth", "got salt: '%s' (len: %d)", Salt.Data(), Slen);
         } else {
            if (gDebug > 2)
               Info("ClearAuth", "Salt not required");
            if (fSocket->Recv(stat, kind) < 0)
               return 0;
            if (kind != kMESS_ANY || stat != 0) {
               Warning("ClearAuth","Potential problems: got msg type:"
                       " %d value: %d (expecting: %d 0)",
                       kind,stat,(Int_t)kMESS_ANY);
            }
         }
         // We may not have got a salt (if the server may not access it
         // or if it needs the full password, like for AFS ...)
         if (!Slen)
            NeedSalt = 0;
      }
      // Now get the password either from prompt or from memory, if saved already
      if (Anon == 1) {

         if (fgPasswd.Contains("@")) {
            // Anonymous like login with user chosen passwd ...
            Passwd = fgPasswd;
         } else {
           // Anonymous like login with automatic passwd generation ...
           TString LocalUser;
           UserGroup_t *pw = gSystem->GetUserInfo();
           if (pw)
              LocalUser = StrDup(pw->fUser);
           delete pw;
           static TString LocalFQDN;
           if (LocalFQDN == "") {
              TInetAddress addr = gSystem->GetHostByName(gSystem->HostName());
              if (addr.IsValid()) {
                 LocalFQDN = addr.GetHostName();
                 if (LocalFQDN == "UnNamedHost")
                    LocalFQDN = addr.GetHostAddress();
              }
           }
           Passwd = Form("%s@%s", LocalUser.Data(), LocalFQDN.Data());
           if (gDebug > 2)
              Info("ClearAuth",
                   "automatically generated anonymous passwd: %s",
                   Passwd.Data());
         }

      } else {

         if (Prompt == 1 || PasHash.Length() == 0) {

            if (Passwd == "") {
               char *pwd = PromptPasswd(Form("%s@%s password: ",
                                        User.Data(),fRemote.Data()));
               Passwd = TString(pwd);
               delete[] pwd;
               if (Passwd == "") {
                  Error("ClearAuth", "password not set");
                  fSocket->Send("-1", kROOTD_PASS);  // Needs this for consistency
                  return 0;
               }
            }
            if (NeedSalt && !PwHash) {
#ifndef R__WIN32
               PasHash = TString(crypt(Passwd, Salt));
               if (!PasHash.BeginsWith(Salt)) {
                  // not the right version of the crypt function:
                  // do not send hash
                  PasHash = Passwd;
               }
#else
               PasHash = Passwd;
#endif
            } else {
               PasHash = Passwd;
            }
         }

      }

      // Store password for later use
      fgUser = fUser;
      fgPwHash = kFALSE;
      fPwHash = kFALSE;
      fgPasswd = Passwd;
      fPasswd = Passwd;
      fSRPPwd = kFALSE;
      fgSRPPwd = kFALSE;

      // Send it to server
      if (Anon == 0 && Crypt == 1) {

         // Needs to send this for consistency
         if (fSocket->Send("\0", kROOTD_PASS) < 0)
            return 0;
         if (SecureSend(fSocket, 1, fRSAKey, PasHash.Data()) == -1) {
            Warning("ClearAuth", "problems secure-sending pass hash"
                    " - may result in authentication failure");
            return 0;
         }
      } else {

         // Standard technique: invert passwd
         if (Passwd != "") {
            for (int i = 0; i < Passwd.Length(); i++) {
               char inv = ~Passwd(i);
               Passwd.Replace(i, 1, inv);
            }
         }
         if (fSocket->Send(Passwd.Data(), kROOTD_PASS) < 0)
            return 0;
      }

      Int_t nrec = 0;
      // Receive username used for login
      if ((nrec = fSocket->Recv(stat, kind)) < 0 )  // returns user
         return 0;
      if (gDebug > 3)
         Info("ClearAuth", "after kROOTD_PASS: kind= %d, stat= %d", kind,
              stat);

      // Check for errors
      if (kind == kROOTD_ERR) {
         if (gDebug > 0)
            AuthError("ClearAuth", stat);
         fgPasswd = "";
         return 0;
      }

      if (kind != kROOTD_PASS || stat < 1)
         Warning("ClearAuth",
                 "problems recvn (user,offset) length (%d:%d bytes:%d)",
                 kind, stat, nrec);

      // Get user and offset
      char answer[256];
      int reclen = (stat+1 > 256) ? 256 : stat+1;
      if ((nrec = fSocket->Recv(answer, reclen, kind)) < 0)
         return 0;
      if (kind != kMESS_STRING)
         Warning("ClearAuth",
                 "username and offset not received (%d:%d)", kind,
                 nrec);

      // Parse answer
      Int_t OffSet = -1;
      char lUser[128];
      sscanf(answer, "%s %d", lUser, &OffSet);
      if (gDebug > 3)
         Info("ClearAuth",
              "received from server: user: %s, offset: %d (%s)", lUser,
              OffSet, answer);

      // Return username
      User = lUser;

      char *Token = 0;
      if (ReUse == 1 && OffSet > -1) {
         // Receive Token
         if (Crypt == 1) {
            if (SecureRecv(fSocket, 1, fRSAKey, &Token) == -1) {
               Warning("ClearAuth",
                       "problems secure-receiving token -"
                       " may result in corrupted token");
               return 0;
            }
         } else {
            Int_t Tlen = 9;
            Token = new char[Tlen];
            if (fSocket->Recv(Token, Tlen, kind) < 0) {
               delete[] Token;
               return 0;
            }
            if (kind != kMESS_STRING)
               Warning("ClearAuth", "token not received (%d:%d)", kind,
                       nrec);
            // Invert Token
            for (int i = 0; i < (int) strlen(Token); i++) {
               Token[i] = ~Token[i];
            }

         }
         if (gDebug > 3)
            Info("ClearAuth", "received from server: token: '%s' ",
                 Token);
      }
      TPwdCtx *pwdctx = new TPwdCtx(fPasswd,fPwHash);
      // Create SecContext object
      fSecContext = fHostAuth->CreateSecContext((const char *)lUser, fRemote,
                    kClear, OffSet, fDetails, (const char *)Token,
                    fgExpDate, (void *)pwdctx, fRSAKey);

      // Release allocated memory ...
      if (Token) delete[] Token;

      // This from remote login
      if (fSocket->Recv(stat, kind) < 0)
         return 0;


      if (kind == kROOTD_AUTH && stat >= 1) {
         return 1;
      } else {
         fgPasswd = "";
         if (kind == kROOTD_ERR)
            if (gDebug > 0)
               AuthError("ClearAuth", stat);
         return 0;
      }

   } else {

      // Old Protocol

      // Send username
      if (fSocket->Send(User.Data(), kROOTD_USER) < 0)
         return 0;

      // Get replay from server
      if (fSocket->Recv(stat, kind) < 0)
         return 0;

      // This check should guarantee backward compatibility with a private
      // version of rootd used by CDF
      if (kind == kROOTD_AUTH && stat == 1) {
         fSecContext =
            fHostAuth->CreateSecContext(User,fRemote,kClear,-1,fDetails,0);
         return 1;
      }

      if (kind == kROOTD_ERR) {
         TString Server = "sockd";
         if (fProtocol.Contains("root"))
            Server = "rootd";
         if (fProtocol.Contains("proof"))
            Server = "proofd";
         if (stat == kErrConnectionRefused) {
            if (gDebug > 0)
               Error("ClearAuth",
                  "%s@%s does not accept connections from %s@%s",
                  Server.Data(),fRemote.Data(),
                  fUser.Data(),gSystem->HostName());
            return -2;
         } else if (stat == kErrNotAllowed) {
            if (gDebug > 0)
               Error("ClearAuth",
                  "%s@%s does not accept %s authentication from %s@%s",
                  Server.Data(),fRemote.Data(),
                  TAuthenticate::fgAuthMeth[0].Data(),
                  fUser.Data(),gSystem->HostName());
         } else {
           if (gDebug > 0)
              AuthError("ClearAuth", stat);
         }
         return 0;
      }
      // Prepare Passwd to send
    badpass1:
      if (Passwd == "") {
         Passwd = PromptPasswd(
                  Form("%s@%s password: ",User.Data(),fRemote.Data()));
         if (Passwd == "")
            Error("ClearAuth", "password not set");
      }
      if (fUser == "anonymous" || fUser == "rootd") {
         if (!Passwd.Contains("@")) {
            Warning("ClearAuth",
                    "please use passwd of form: user@host.do.main");
            Passwd = "";
            goto badpass1;
         }
      }

      fgPasswd = Passwd;
      fPasswd = Passwd;

      // Invert passwd
      if (Passwd != "") {
         for (int i = 0; i < Passwd.Length(); i++) {
            char inv = ~Passwd(i);
            Passwd.Replace(i, 1, inv);
         }
      }
      // Send it over the net
      if (fSocket->Send(Passwd, kROOTD_PASS) < 0)
         return 0;

      // Get result of attempt
      if (fSocket->Recv(stat, kind) < 0)  // returns user
         return 0;
      if (gDebug > 3)
         Info("ClearAuth", "after kROOTD_PASS: kind= %d, stat= %d", kind,
              stat);

      if (kind == kROOTD_AUTH && stat == 1) {
         fSecContext =
            fHostAuth->CreateSecContext(User,fRemote,kClear,-1,fDetails,0);
         return 1;
      } else {
         if (kind == kROOTD_ERR)
            if (gDebug > 0)
               AuthError("ClearAuth", stat);
         return 0;
      }
   }
}

//______________________________________________________________________________
THostAuth *TAuthenticate::GetHostAuth(const char *host, const char *user,
                                      Option_t *Opt, Int_t *Exact)
{
   // Sets fUser=user and search fgAuthInfo for the entry pertaining to
   // (host,user), setting fHostAuth accordingly.
   // If Opt = "P" use fgProofAuthInfo list instead
   // If no entry is found fHostAuth is not changed

   if (Exact)
      *Exact = 0;
   if (gDebug > 2)
      ::Info("TAuthenticate::GetHostAuth", "enter ... %s ... %s", host, user);

   // Strip off the servertype, if any
   Int_t SrvTyp = -1;
   TString Host = host;
   if (Host.Contains(":")) {
      char *ps = (char *)strstr(host,":");
      if (ps)
         SrvTyp = atoi(ps+1);
      Host.Remove(Host.Index(":"));
   }
   TString HostFQDN = Host;
   if (strncmp(host,"default",7) && !HostFQDN.Contains("*")) {
     TInetAddress addr = gSystem->GetHostByName(HostFQDN);
     if (addr.IsValid()) {
        HostFQDN = addr.GetHostName();
        if (HostFQDN == "UnNamedHost")
           HostFQDN = addr.GetHostAddress();
     }
   }
   TString User = user;
   if (!User.Length())
      User = "*";
   THostAuth *rHA = 0;

   // Check list of auth info for already loaded info about this host
   TIter *next = new TIter(GetAuthInfo());
   if (!strncasecmp(Opt,"P",1)) {
      SafeDelete(next);
      next = new TIter(GetProofAuthInfo());
   }

   THostAuth *ai;
   Bool_t NotFound = kTRUE;
   Bool_t ServerOK = kTRUE;
   while ((ai = (THostAuth *) (*next)())) {
      if (gDebug > 3)
         ai->Print("Authenticate::GetHostAuth");

      // Server
      if (!(ServerOK = (ai->GetServer() == -1) ||
                       (ai->GetServer() == SrvTyp)))
         continue;

      // Use default entry if existing and nothing more specific is found
      if (!strcmp(ai->GetHost(),"default") && ServerOK && NotFound)
         rHA = ai;

      // Check
      if (CheckHost(HostFQDN,ai->GetHost()) &&
          CheckHost(User,ai->GetUser())     && ServerOK) {
         rHA = ai;
         NotFound = kFALSE;
      }

      if (HostFQDN == ai->GetHost() &&
          User == ai->GetUser()     && SrvTyp == ai->GetServer() ) {
         rHA = ai;
         if (Exact)
            *Exact = 1;
         break;
      }
   }
   SafeDelete(next);
   return rHA;
}

//______________________________________________________________________________
THostAuth *TAuthenticate::HasHostAuth(const char *host, const char *user,
                                      Option_t *Opt)
{
   // Checks if a THostAuth with exact match for {host,user} exists
   // in the fgAuthInfo list
   // If Opt = "P" use ProofAuthInfo list instead
   // Returns pointer to it or 0

   if (gDebug > 2)
      ::Info("TAuthenticate::HasHostAuth", "enter ... %s ... %s", host, user);

   // Strip off the servertype, if any
   Int_t SrvTyp = -1;
   TString hostFQDN = host;
   if (hostFQDN.Contains(":")) {
      char *ps = (char *)strstr(host,":");
      if (ps)
         SrvTyp = atoi(ps+1);
      hostFQDN.Remove(hostFQDN.Index(":"));
   }
   if (strncmp(host,"default",7) && !hostFQDN.Contains("*")) {
     TInetAddress addr = gSystem->GetHostByName(hostFQDN);
     if (addr.IsValid()) {
        hostFQDN = addr.GetHostName();
        if (hostFQDN == "UnNamedHost")
           hostFQDN = addr.GetHostAddress();
     }
   }

   TIter *next = new TIter(GetAuthInfo());
   if (!strncasecmp(Opt,"P",1)) {
      SafeDelete(next);
      next = new TIter(GetProofAuthInfo());
   }
   THostAuth *ai;
   while ((ai = (THostAuth *) (*next)())) {

      if (hostFQDN == ai->GetHost() &&
          !strcmp(user, ai->GetUser()) && SrvTyp == ai->GetServer()) {
         return ai;
      }
   }
   SafeDelete(next);
   return 0;
}

//______________________________________________________________________________
void TAuthenticate::FileExpand(const char *fexp, FILE *ftmp)
{
   // Expands include directives found in fexp files
   // The expanded, temporary file, is pointed to by 'ftmp'
   // and should be already open. To be called recursively.

   FILE *fin;
   char line[kMAXPATHLEN];
   char cinc[20], fileinc[kMAXPATHLEN];

   if (gDebug > 2)
     ::Info("TAuthenticate::FileExpand", "enter ... '%s' ... 0x%lx", fexp, (Long_t)ftmp);

   fin = fopen(fexp, "r");
   if (fin == 0)
      return;

   while (fgets(line, sizeof(line), fin) != 0) {
      // Skip comment lines
      if (line[0] == '#')
         continue;
      if (line[strlen(line) - 1] == '\n')
         line[strlen(line) - 1] = '\0';
      if (gDebug > 2)
         ::Info("TAuthenticate::FileExpand", "read line ... '%s'", line);
      int nw = sscanf(line, "%s %s", cinc, fileinc);
      if (nw < 1)
         continue;              // Not enough info in this line
      if (strcmp(cinc, "include") != 0) {
         // copy line in temporary file
         fprintf(ftmp, "%s\n", line);
      } else {

         // Drop quotes or double quotes, if any
         TString Line(line);
         Line.ReplaceAll("\"",1,"",0);
         Line.ReplaceAll("'",1,"",0);
         sscanf(Line.Data(), "%s %s", cinc, fileinc);

         // support environment directories ...
         if (fileinc[0] == '$') {
            TString FileInc(fileinc);
            TString EnvDir(fileinc);
            if (EnvDir.Contains("/")) {
               EnvDir.Remove(EnvDir.Index("/"));
               EnvDir.Remove(0,1);
               if (gSystem->Getenv(EnvDir.Data())) {
                  FileInc.Remove(0,1);
                  FileInc.ReplaceAll(EnvDir.Data(),gSystem->Getenv(EnvDir.Data()));
                  fileinc[0] = '\0';
                  strcpy(fileinc,FileInc.Data());
              }
            }
         }

         // open (expand) file in temporary file ...
         if (fileinc[0] == '~') {
            // needs to expand
            int flen =
                strlen(fileinc) + strlen(gSystem->Getenv("HOME")) + 10;
            char *ffull = new char[flen];
            sprintf(ffull, "%s/%s", gSystem->Getenv("HOME"), fileinc + 1);
            strcpy(fileinc, ffull);
         }
         // Check if file exist and can be read ... ignore if not ...
         if (!gSystem->AccessPathName(fileinc, kReadPermission)) {
            FileExpand(fileinc, ftmp);
         } else {
            ::Warning("TAuthenticate::FileExpand",
                      "file specified by 'include' cannot be open or read (%s)",
                      fileinc);
         }
      }
   }
   fclose(fin);
}

//______________________________________________________________________________
char *TAuthenticate::GetDefaultDetails(int sec, int opt, const char *usr)
{
   // Determine default authentication details for method 'sec' and user 'usr'.
   // Checks .rootrc family files. Returned string must be deleted by the user.

   char temp[kMAXPATHLEN] = { 0 };
   const char copt[2][5] = { "no", "yes" };

   if (gDebug > 2)
      ::Info("TAuthenticate::GetDefaultDetails",
             "enter ... %d ...pt:%d ... '%s'", sec, opt, usr);

   if (opt < 0 || opt > 1)
      opt = 1;

   // UsrPwd
   if (sec == TAuthenticate::kClear) {
      if (strlen(usr) == 0 || !strncmp(usr,"*",1))
         usr = gEnv->GetValue("UsrPwd.Login", "");
      sprintf(temp, "pt:%s ru:%s cp:%s us:%s",
              gEnv->GetValue("UsrPwd.LoginPrompt", copt[opt]),
              gEnv->GetValue("UsrPwd.ReUse", "1"),
              gEnv->GetValue("UsrPwd.Crypt", "1"), usr);

      // SRP
   } else if (sec == TAuthenticate::kSRP) {
      if (strlen(usr) == 0 || !strncmp(usr,"*",1))
         usr = gEnv->GetValue("SRP.Login", "");
      sprintf(temp, "pt:%s ru:%s us:%s",
              gEnv->GetValue("SRP.LoginPrompt", copt[opt]),
              gEnv->GetValue("SRP.ReUse", "0"), usr);

      // Kerberos
   } else if (sec == TAuthenticate::kKrb5) {
      if (strlen(usr) == 0 || !strncmp(usr,"*",1))
         usr = gEnv->GetValue("Krb5.Login", "");
      sprintf(temp, "pt:%s ru:%s us:%s",
              gEnv->GetValue("Krb5.LoginPrompt", copt[opt]),
              gEnv->GetValue("Krb5.ReUse", "0"), usr);

      // Globus
   } else if (sec == TAuthenticate::kGlobus) {
      sprintf(temp, "pt:%s ru:%s %s",
              gEnv->GetValue("Globus.LoginPrompt", copt[opt]),
              gEnv->GetValue("Globus.ReUse", "1"),
              gEnv->GetValue("Globus.Login", ""));

      // SSH
   } else if (sec == TAuthenticate::kSSH) {
      if (strlen(usr) == 0 || !strncmp(usr,"*",1))
         usr = gEnv->GetValue("SSH.Login", "");
      sprintf(temp, "pt:%s ru:%s us:%s",
              gEnv->GetValue("SSH.LoginPrompt", copt[opt]),
              gEnv->GetValue("SSH.ReUse", "1"), usr);

      // Uid/Gid
   } else if (sec == TAuthenticate::kRfio) {
      if (strlen(usr) == 0 || !strncmp(usr,"*",1))
         usr = gEnv->GetValue("UidGid.Login", "");
      sprintf(temp, "pt:%s us:%s",
              gEnv->GetValue("UidGid.LoginPrompt", copt[opt]), usr);
   }
   if (gDebug > 2)
      ::Info("TAuthenticate::GetDefaultDetails", "returning ... %s", temp);

   return StrDup(temp);
}

//______________________________________________________________________________
void TAuthenticate::RemoveHostAuth(THostAuth * ha, Option_t *Opt)
{
   // Remove THostAuth instance from the list

   if (!strncasecmp(Opt,"P",1))
      GetProofAuthInfo()->Remove(ha);
   else
      GetAuthInfo()->Remove(ha);
   // ... destroy it
   delete ha;
}

//______________________________________________________________________________
void TAuthenticate::Show(Option_t *opt)
{
   // Print info about the authentication sector.
   // If 'opt' contains 's' or 'S' prints information about established TSecContext,
   // else prints information about THostAuth (if 'opt' is 'p' or 'P', prints
   // Proof related information)

   TString Opt(opt);

   if (Opt.Contains("s",TString::kIgnoreCase)) {

      // Print established security contexts
      TIter next(gROOT->GetListOfSecContexts());
      TSecContext *sc = 0;
      while ((sc = (TSecContext *)next()))
         sc->Print();

   } else {

      ::Info("::Print",
         " +--------------------------- BEGIN --------------------------------+");
      ::Info("::Print",
         " +                                                                  +");
      if (Opt.Contains("p",TString::kIgnoreCase)) {
         ::Info("::Print",
         " + List fgProofAuthInfo has %4d members                            +",
           GetProofAuthInfo()->GetSize());
         ::Info("::Print",
         " +                                                                  +");
         ::Info("::Print",
         " +------------------------------------------------------------------+");
         TIter next(GetProofAuthInfo());
         THostAuth *ai;
         while ((ai = (THostAuth *) next())) {
            ai->Print();
         }
      } else {
         ::Info("::Print",
         " + List fgAuthInfo has %4d members                                 +",
          GetAuthInfo()->GetSize());
         ::Info("::Print",
         " +                                                                  +");
         ::Info("::Print",
         " +------------------------------------------------------------------+");
         TIter next(GetAuthInfo());
         THostAuth *ai;
         while ((ai = (THostAuth *) next())) {
            ai->Print();
            ai->PrintEstablished();
         }
      }
      ::Info("::Print",
         " +---------------------------- END ---------------------------------+");
   }
}

//______________________________________________________________________________
Int_t TAuthenticate::AuthExists(TString User, Int_t Method, const char *Options,
                                Int_t *Message, Int_t *Rflag,
                                CheckSecCtx_t CheckSecCtx)
{
   // Check if we have a valid established sec context in memory
   // Retrieves relevant info and negotiates with server.
   // Options = "Opt,strlen(User),User.Data()"
   // Message = kROOTD_USER, ...

   // Welcome message, if requested ...
   if (gDebug > 2)
      Info("AuthExists","%d: enter: msg: %d options: '%s'",
              Method,*Message, Options);

   // Look for an existing security context matching this request
   Bool_t NotHA = kFALSE;

   // First in the local list
   TIter next(fHostAuth->Established());
   TSecContext *SecCtx;
   while ((SecCtx = (TSecContext *)next())) {
      if (SecCtx->GetMethod() == Method) {
         if (fRemote == SecCtx->GetHost()) {
            if (CheckSecCtx &&
              (*CheckSecCtx)(User,SecCtx) == 1)
               break;
         }
      }
   }

   // If nothing found, try the all list
   if (!SecCtx) {
      next = TIter(gROOT->GetListOfSecContexts());
      while ((SecCtx = (TSecContext *)next())) {
         if (SecCtx->GetMethod() == Method) {
            if (fRemote == SecCtx->GetHost()) {
               if (CheckSecCtx &&
                  (*CheckSecCtx)(User,SecCtx) == 1) {
                  NotHA = kTRUE;
                  break;
               }
            }
         }
      }
   }

   // If we have been given a valid sec context retrieve some info
   Int_t OffSet = -1;
   TString Token;
   if (SecCtx) {
      OffSet = SecCtx->GetOffSet();
      Token = SecCtx->GetToken();
      if (gDebug > 2)
         Info("AuthExists",
              "found valid TSecContext: OffSet: %d Token: '%s'",
              OffSet, Token.Data());
   }

   // Prepare string to be sent to the server
   TString sstr(Form("%d %d %s", gProcessID, OffSet, Options));

   // Send Message
   if (fSocket->Send(sstr, *Message) < 0)
      return -2;;

   Int_t ReUse = *Rflag;
   if (ReUse == 1 && OffSet > -1) {

      // Receive result of checking offset
      // But only for recent servers
      Int_t rproto = fSocket->GetRemoteProtocol();
      Bool_t oldsrv = ((fProtocol.BeginsWith("root") && rproto == 9) ||
                       (fProtocol.BeginsWith("proof") && rproto == 8));
      Int_t stat = 1, kind;
      if (!oldsrv) {
         if (fSocket->Recv(stat, kind) < 0)
            return -2;
         if (kind != kROOTD_AUTH)
            Warning("AuthExists","protocol error: expecting %d got %d"
                    " (value: %d)",kROOTD_AUTH,kind,stat);
      }

      if (stat == 1) {
         if (gDebug > 2)
            Info("AuthExists","OffSet OK");

         Int_t RSAKey = SecCtx->GetRSAKey();
         if (gDebug > 2)
            Info("AuthExists", "key type: %d", RSAKey);

         if (RSAKey > -1) {
            // Send Token encrypted
            if (SecureSend(fSocket, 1, RSAKey, Token) == -1) {
               Warning("AuthExists", "problems secure-sending Token %s",
                       "- may trigger problems in proofing Id ");
               return -2;
            }
         } else {
            // Send inverted
            for (int i = 0; i < Token.Length(); i++) {
               char inv = ~Token(i);
               Token.Replace(i, 1, inv);
            }
            if (fSocket->Send(Token, kMESS_STRING) < 0)
               return -2;
         }
      } else {
         if (gDebug > 0)
            Info("AuthExists","OffSet not OK - rerun authentication");
         // If the sec context was not valid, deactivate it ...
         if (SecCtx)
            SecCtx->DeActivate("");
      }
   }

   Int_t stat, kind;
   if (fSocket->Recv(stat, kind) < 0)
      return -2;
   if (gDebug > 3)
      Info("AuthExists","%d: after msg %d: kind= %d, stat= %d",
                        Method,*Message, kind, stat);

   // Return flags
   *Message = kind;
   *Rflag = stat;

   if (kind == kROOTD_ERR) {
      TString Server = "sockd";
      if (fSocket->GetServType() == TSocket::kROOTD)
         Server = "rootd";
      if (fSocket->GetServType() == TSocket::kPROOFD)
         Server = "proofd";
      if (stat == kErrConnectionRefused) {
         Error("AuthExists","%s@%s does not accept connections from %s@%s",
               Server.Data(),fRemote.Data(),fUser.Data(),gSystem->HostName());
         return -2;
      } else if (stat == kErrNotAllowed) {
         if (gDebug > 0)
            Info("AuthExists",
                 "%s@%s does not accept %s authentication from %s@%s",
                 Server.Data(),fRemote.Data(), fgAuthMeth[Method].Data(),
                 fUser.Data(),gSystem->HostName());
      } else {
        if (gDebug > 0)
           AuthError("AuthExists", stat);
      }
      // If the sec context was not valid, deactivate it ...
      if (SecCtx)
         SecCtx->DeActivate("");
      return 0;
   }

   if (kind == kROOTD_AUTH && stat >= 1) {
      if (!SecCtx)
         SecCtx =
            fHostAuth->CreateSecContext(fUser,fRemote,Method,-stat,fDetails,0);
      if (gDebug > 3) {
         if (stat == 1)
            Info("AuthExists", "valid authentication exists");
         if (stat == 2)
            Info("AuthExists", "valid authentication exists: offset changed");
         if (stat == 3)
            Info("AuthExists", "remote access authorized by /etc/hosts.equiv");
         if (stat == 4)
            Info("AuthExists", "no authentication required remotely");
      }

      if (stat == 2) {
         int newOffSet;
         // Receive new offset ...
         if (fSocket->Recv(newOffSet, kind) < 0)
            return -2;
         // ... and save it
         SecCtx->SetOffSet(newOffSet);
      }

      fSecContext = SecCtx;
      // Add it to local list for later use (if not already there)
      if (NotHA)
         fHostAuth->Established()->Add(SecCtx);
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
void TAuthenticate::InitRandom()
{
   // Initialize random machine using seed from /dev/urandom
   // (or current time if /dev/urandom not available).

   static Bool_t notinit = kTRUE;

   if (notinit) {
      const char *randdev = "/dev/urandom";
      Int_t fd;
      UInt_t seed;
      if ((fd = open(randdev, O_RDONLY)) != -1) {
         if (gDebug > 2)
            ::Info("InitRandom", "taking seed from %s", randdev);
         read(fd, &seed, sizeof(seed));
         close(fd);
      } else {
         if (gDebug > 2)
            ::Info("InitRandom", "%s not available: using time()", randdev);
         seed = time(0);   //better use times() + win32 equivalent
      }
      srand(seed);
      notinit = kFALSE;
   }
}

//______________________________________________________________________________
Int_t TAuthenticate::GenRSAKeys()
{
   // Generate a valid pair of private/public RSA keys to protect for
   // authentication token exchange

   if (gDebug > 2)
      Info("GenRSAKeys", "enter");

   if (fgRSAInit == 1) {
      if (gDebug > 2)
         Info("GenRSAKeys", "Keys prviously generated - return");
   }

   // This is for dynamic loads ...
#ifdef ROOTLIBDIR
   TString lib = TString(ROOTLIBDIR) + "/libRsa";
#else
   TString lib = TString(gRootDir) + "/lib/libRsa";
#endif

   // This is the local RSA implementation
   if (!rsa_fun::fg_rsa_genprim) {
      char *p;
      if ((p = gSystem->DynamicPathName(lib, kTRUE))) {
         delete[]p;
         gSystem->Load(lib);
      }
   }

   // Init random machine
   TAuthenticate::InitRandom();

#ifdef R__SSL
   if (fgRSAKey == 1) {
      // Generate also the SSL key
      if (gDebug > 2)
         Info("GenRSAKeys","SSL: Generate Blowfish key");

      // Number of bits for key
      Int_t nbits = gEnv->GetValue("SSL.BFBits",256);

      // Minimum is 128
      nbits = (nbits >= 128) ? nbits : 128;

      // Max to limit size of buffers to 15912 (internal limitation)
      nbits = (nbits <= 15912) ? nbits : 15912;

      // Closer Number of chars
      Int_t klen = nbits / 8 ;

      // Init random engine
      char *rbuf = GetRandString(0,klen);
      RAND_seed(rbuf,strlen(rbuf));

      // This is what we export
      fgRSAPubExport[1].len = klen;
      fgRSAPubExport[1].keys = rbuf;
      if (gDebug > 2)
         Info("GenRSAKeys","SSL: BF key length: %d", fgRSAPubExport[1].len);

      // Now set the key locally in BF form
      BF_set_key(&fgBFKey, klen, (const unsigned char *)rbuf);
   }
#endif

   // Sometimes some bunch is not decrypted correctly
   // That's why we make retries to make sure that encryption/decryption
   // works as expected
   Bool_t NotOk = 1;
   rsa_NUMBER p1, p2, rsa_n, rsa_e, rsa_d;
   Int_t l_n = 0, l_e = 0, l_d = 0;
   char buf_n[rsa_STRLEN], buf_e[rsa_STRLEN], buf_d[rsa_STRLEN];
#if R__RSADEB
   char buf[rsa_STRLEN];
#endif

   Int_t NAttempts = 0;
   Int_t thePrimeLen = 20;
   Int_t thePrimeExp = 40;   // Prime probability = 1-0.5^thePrimeExp
   while (NotOk && NAttempts < kMAXRSATRIES) {

      NAttempts++;
      if (gDebug > 2 && NAttempts > 1) {
         Info("GenRSAKeys", "retry no. %d",NAttempts);
         srand(rand());
      }

      // Valid pair of primes
      p1 = rsa_fun::fg_rsa_genprim(thePrimeLen, thePrimeExp);
      p2 = rsa_fun::fg_rsa_genprim(thePrimeLen+1, thePrimeExp);

      // Retry if equal
      Int_t NPrimes = 0;
      while (rsa_fun::fg_rsa_cmp(&p1, &p2) == 0 && NPrimes < kMAXRSATRIES) {
         NPrimes++;
         if (gDebug > 2)
            Info("GenRSAKeys", "equal primes: regenerate (%d times)",NPrimes);
         srand(rand());
         p1 = rsa_fun::fg_rsa_genprim(thePrimeLen, thePrimeExp);
         p2 = rsa_fun::fg_rsa_genprim(thePrimeLen+1, thePrimeExp);
      }
#if R__RSADEB
      if (gDebug > 3) {
         rsa_fun::fg_rsa_num_sput(&p1, buf, rsa_STRLEN);
         Info("GenRSAKeys", "local: p1: '%s' ", buf);
         rsa_fun::fg_rsa_num_sput(&p2, buf, rsa_STRLEN);
         Info("GenRSAKeys", "local: p2: '%s' ", buf);
      }
#endif
      // Generate keys
      if (rsa_fun::fg_rsa_genrsa(p1, p2, &rsa_n, &rsa_e, &rsa_d)) {
         if (gDebug > 2 && NAttempts > 1)
            Info("GenRSAKeys"," genrsa: unable to generate keys (%d)",
                 NAttempts);
         continue;
      }

      // Get equivalent strings and determine their lengths
      rsa_fun::fg_rsa_num_sput(&rsa_n, buf_n, rsa_STRLEN);
      l_n = strlen(buf_n);
      rsa_fun::fg_rsa_num_sput(&rsa_e, buf_e, rsa_STRLEN);
      l_e = strlen(buf_e);
      rsa_fun::fg_rsa_num_sput(&rsa_d, buf_d, rsa_STRLEN);
      l_d = strlen(buf_d);

#if R__RSADEB
      if (gDebug > 3) {
         Info("GenRSAKeys", "local: n: '%s' length: %d", buf_n, l_n);
         Info("GenRSAKeys", "local: e: '%s' length: %d", buf_e, l_e);
         Info("GenRSAKeys", "local: d: '%s' length: %d", buf_d, l_d);
      }
#endif
      if (rsa_fun::fg_rsa_cmp(&rsa_n, &rsa_e) <= 0)
         continue;
      if (rsa_fun::fg_rsa_cmp(&rsa_n, &rsa_d) <= 0)
         continue;

      // Now we try the keys
      char Test[2 * rsa_STRLEN] = "ThisIsTheStringTest01203456-+/";
      Int_t lTes = 31;
      char *Tdum = GetRandString(0, lTes - 1);
      strncpy(Test, Tdum, lTes);
      delete[]Tdum;
      char buf[2 * rsa_STRLEN];
      if (gDebug > 3)
         Info("GenRSAKeys", "local: test string: '%s' ", Test);

      // Private/Public
      strncpy(buf, Test, lTes);
      buf[lTes] = 0;

      // Try encryption with private key
      int lout = rsa_fun::fg_rsa_encode(buf, lTes, rsa_n, rsa_e);
      if (gDebug > 3)
         Info("GenRSAKeys",
              "local: length of crypted string: %d bytes", lout);

      // Try decryption with public key
      rsa_fun::fg_rsa_decode(buf, lout, rsa_n, rsa_d);
      buf[lTes] = 0;
      if (gDebug > 3)
         Info("GenRSAKeys", "local: after private/public : '%s' ", buf);

      if (strncmp(Test, buf, lTes))
         continue;

      // Public/Private
      strncpy(buf, Test, lTes);
      buf[lTes] = 0;

      // Try encryption with public key
      lout = rsa_fun::fg_rsa_encode(buf, lTes, rsa_n, rsa_d);
      if (gDebug > 3)
         Info("GenRSAKeys", "local: length of crypted string: %d bytes ",
              lout);

      // Try decryption with private key
      rsa_fun::fg_rsa_decode(buf, lout, rsa_n, rsa_e);
      buf[lTes] = 0;
      if (gDebug > 3)
         Info("GenRSAKeys", "local: after public/private : '%s' ", buf);

      if (strncmp(Test, buf, lTes))
         continue;

      NotOk = 0;
   }

   // Save Private key
   rsa_fun::fg_rsa_assign(&fgRSAPriKey.n, &rsa_n);
   rsa_fun::fg_rsa_assign(&fgRSAPriKey.e, &rsa_e);

   // Save Public key
   rsa_fun::fg_rsa_assign(&fgRSAPubKey.n, &rsa_n);
   rsa_fun::fg_rsa_assign(&fgRSAPubKey.e, &rsa_d);

#if R__RSADEB
   if (gDebug > 2) {
      // Determine their lengths
      Info("GenRSAKeys", "local: generated keys are:");
      Info("GenRSAKeys", "local: n: '%s' length: %d", buf_n, l_n);
      Info("GenRSAKeys", "local: e: '%s' length: %d", buf_e, l_e);
      Info("GenRSAKeys", "local: d: '%s' length: %d", buf_d, l_d);
   }
#endif
   // Export form
   if (fgRSAPubExport[0].keys) {
      delete[] fgRSAPubExport[0].keys;
      fgRSAPubExport[0].len = 0;
   }
   fgRSAPubExport[0].len = l_n + l_d + 4;
   fgRSAPubExport[0].keys = new char[fgRSAPubExport[0].len];

   fgRSAPubExport[0].keys[0] = '#';
   memcpy(fgRSAPubExport[0].keys + 1, buf_n, l_n);
   fgRSAPubExport[0].keys[l_n + 1] = '#';
   memcpy(fgRSAPubExport[0].keys + l_n + 2, buf_d, l_d);
   fgRSAPubExport[0].keys[l_n + l_d + 2] = '#';
   fgRSAPubExport[0].keys[l_n + l_d + 3] = 0;
#if R__RSADEB
   if (gDebug > 2)
      Info("GenRSAKeys", "local: export pub: '%s'", fgRSAPubExport[0].keys);
#else
   if (gDebug > 2)
      Info("GenRSAKeys", "local: export pub length: %d bytes", fgRSAPubExport[0].len);
#endif

   // Set availability flag
   fgRSAInit = 1;

   return 0;
}

//______________________________________________________________________________
char *TAuthenticate::GetRandString(Int_t Opt, Int_t Len)
{
   // Allocates and fills a 0 terminated buffer of length Len+1 with
   // Len random characters.
   // Returns pointer to the buffer (to be deleted by the caller)
   // Opt = 0      any non dangerous char
   //       1      letters and numbers  (upper and lower case)
   //       2      hex characters       (upper and lower case)

   int iimx[4][4] = { {0x0, 0xffffff08, 0xafffffff, 0x2ffffffe}, // Opt = 0
                      {0x0, 0x3ff0000, 0x7fffffe, 0x7fffffe},    // Opt = 1
                      {0x0, 0x3ff0000, 0x7e, 0x7e},              // Opt = 2
                      {0x0, 0x3ffc000, 0x7fffffe, 0x7fffffe}     // Opt = 3
                    };

   const char *cOpt[4] = { "Any", "LetNum", "Hex", "Crypt" };

   //  Default option 0
   if (Opt < 0 || Opt > 2) {
      Opt = 0;
      if (gDebug > 2)
         Info("GetRandString", "unknown option: %d : assume 0", Opt);
   }
   if (gDebug > 2)
      Info("GetRandString", "enter ... Len: %d %s", Len, cOpt[Opt]);

   // Allocate buffer
   char *Buf = new char[Len + 1];

   // Init random machine (if needed)
   TAuthenticate::InitRandom();

   // randomize
   Int_t k = 0;
   Int_t i, j, l, m, frnd;
   while (k < Len) {
      frnd = rand();
      for (m = 7; m < 32; m += 7) {
         i = 0x7F & (frnd >> m);
         j = i / 32;
         l = i - j * 32;
         if ((iimx[Opt][j] & (1 << l))) {
            Buf[k] = i;
            k++;
         }
         if (k == Len)
            break;
      }
   }

   // null terminated
   Buf[Len] = 0;
   if (gDebug > 3)
      Info("GetRandString", "got '%s' ", Buf);

   return Buf;
}

//______________________________________________________________________________
Int_t TAuthenticate::SecureSend(TSocket *sock, Int_t enc,
                                Int_t key, const char *str)
{
   // Encode null terminated str using the session private key indcated by enc
   // and sends it over the network
   // Returns number of bytes sent, or -1 in case of error.
   // enc = 1 for private encoding, enc = 2 for public encoding

   char buftmp[kMAXSECBUF];
   char buflen[20];

   if (gDebug > 2)
      ::Info("TAuthenticate::SecureSend", "local: enter ... (enc: %d)", enc);

   Int_t slen = strlen(str) + 1;
   Int_t ttmp = 0;
   Int_t nsen = -1;

   if (key == 0) {
      strncpy(buftmp, str, slen);
      buftmp[slen] = 0;
      if (enc == 1)
         ttmp = rsa_fun::fg_rsa_encode(buftmp, slen, fgRSAPriKey.n,
                                                     fgRSAPriKey.e);
      else if (enc == 2)
         ttmp = rsa_fun::fg_rsa_encode(buftmp, slen, fgRSAPubKey.n,
                                                     fgRSAPubKey.e);
      else
         return nsen;
   } else if (key == 1) {
#ifdef R__SSL
      ttmp = strlen(str);
      if ((ttmp % 8) > 0)           // It should be a multiple of 8!
         ttmp = ((ttmp + 8)/8) * 8;
      unsigned char iv[8];
      memset((void *)&iv[0],0,8);
      BF_cbc_encrypt((const unsigned char *)str, (unsigned char *)buftmp,
                     strlen(str), &fgBFKey, iv, BF_ENCRYPT);
#else
      if (gDebug > 0)
         ::Info("TAuthenticate::SecureSend","not compiled with SSL support:"
                " you should not have got here!");
#endif
   } else {
      if (gDebug > 0)
         ::Info("TAuthenticate::SecureSend","unknown key type (%d)",key);
      return nsen;
   }

   snprintf(buflen,20,"%d",ttmp);
   if (sock->Send(buflen, kROOTD_ENCRYPT) < 0)
      return -1;
   nsen = sock->SendRaw(buftmp, ttmp);
   if (gDebug > 3)
      ::Info("TAuthenticate::SecureSend",
             "local: sent %d bytes (expected: %d)", nsen,ttmp);


   return nsen;
}

//______________________________________________________________________________
Int_t TAuthenticate::SecureRecv(TSocket *sock, Int_t dec, Int_t key, char **str)
{
   // Receive str from sock and decode it using key indicated by key type
   // Return number of received bytes or -1 in case of error.
   // dec = 1 for private decoding, dec = 2 for public decoding

   char buftmp[kMAXSECBUF];
   char buflen[20];

   Int_t nrec = -1;
   // We must get a pointer ...
   if (!str)
      return nrec;

   Int_t kind;
   if (sock->Recv(buflen, 20, kind) < 0)
      return -1;
   Int_t len = atoi(buflen);
   if (gDebug > 3)
      ::Info("TAuthenticate::SecureRecv", "got len '%s' %d (msg kind: %d)",
             buflen, len, kind);
   if (len == 0) {
      return len;
   }
   if (!strncmp(buflen, "-1", 2))
      return nrec;

   // Receive buffer
   if ((nrec = sock->RecvRaw(buftmp, len)) < 0)
      return nrec;
   if (key == 0) {
      if (dec == 1)
         rsa_fun::fg_rsa_decode(buftmp, len, fgRSAPriKey.n, fgRSAPriKey.e);
      else if (dec == 2)
         rsa_fun::fg_rsa_decode(buftmp, len, fgRSAPubKey.n, fgRSAPubKey.e);
      else
         return -1;

      // Prepare output
      *str = new char[strlen(buftmp) + 1];
      strcpy(*str, buftmp);

   } else if (key == 1) {
#ifdef R__SSL
      unsigned char iv[8];
      memset((void *)&iv[0],0,8);
      *str = new char[nrec + 1];
      BF_cbc_encrypt((const unsigned char *)buftmp, (unsigned char *)(*str),
                      nrec, &fgBFKey, iv, BF_DECRYPT);
      (*str)[nrec] = '\0';
#else
      if (gDebug > 0)
         ::Info("TAuthenticate::SecureRecv","not compiled with SSL support:"
                " you should not have got here!");
#endif
   } else {
      if (gDebug > 0)
         ::Info("TAuthenticate::SecureRecv","unknown key type (%d)",key);
      return -1;
   }

   return nrec;
}

//______________________________________________________________________________
Int_t TAuthenticate::DecodeRSAPublic(const char *RSAPubExport, rsa_NUMBER &RSA_n,
                                     rsa_NUMBER &RSA_d, void **RSASSL)
{
   // Store RSA public keys from export string RSAPubExport.

   if (!RSAPubExport)
      return -1;

   if (gDebug > 2)
      ::Info("TAuthenticate::DecodeRSAPublic",
             "enter: string length: %d bytes", strlen(RSAPubExport));

   char str[kMAXPATHLEN] = { 0 };
   Int_t klen = strlen(RSAPubExport);
   if (klen > kMAXPATHLEN - 1) {
      ::Info("TAuthenticate::DecodeRSAPublic",
             "key too long (%d): truncate to %d",klen,kMAXPATHLEN);
      klen = kMAXPATHLEN - 1;
   }
   memcpy(str, RSAPubExport, klen);
   str[klen] ='\0';

   Int_t keytype = -1;

   if (klen > 0) {

      // Skip spaces at beginning, if any
      int k = 0;
      while (str[k] == 32) k++;

      if (str[k] == '#') {

         keytype = 0;

         // The format is #<hex_n>#<hex_d>#
         char *pd1 = strstr(str, "#");
         char *pd2 = strstr(pd1 + 1, "#");
         char *pd3 = strstr(pd2 + 1, "#");
         if (pd1 && pd2 && pd3) {
            // Get <hex_n> ...
            int l1 = (int) (pd2 - pd1 - 1);
            char *RSA_n_exp = new char[l1 + 1];
            strncpy(RSA_n_exp, pd1 + 1, l1);
            RSA_n_exp[l1] = 0;
            if (gDebug > 2)
               ::Info("TAuthenticate::DecodeRSAPublic",
                      "got %d bytes for RSA_n_exp", strlen(RSA_n_exp));
            // Now <hex_d>
            int l2 = (int) (pd3 - pd2 - 1);
            char *RSA_d_exp = new char[l2 + 1];
            strncpy(RSA_d_exp, pd2 + 1, l2);
            RSA_d_exp[l2] = 0;
            if (gDebug > 2)
               ::Info("TAuthenticate::DecodeRSAPublic",
                      "got %d bytes for RSA_d_exp", strlen(RSA_d_exp));

            rsa_fun::fg_rsa_num_sget(&RSA_n, RSA_n_exp);
            rsa_fun::fg_rsa_num_sget(&RSA_d, RSA_d_exp);

            if (RSA_n_exp)
               if (RSA_n_exp) delete[] RSA_n_exp;
            if (RSA_d_exp)
               if (RSA_d_exp) delete[] RSA_d_exp;

         } else
            ::Info("TAuthenticate::DecodeRSAPublic","bad format for input string");
#ifdef R__SSL
      } else {
         // try SSL
         keytype = 1;

         RSA *RSAtmp;

         // Bio for exporting the pub key
         BIO *bpub = BIO_new(BIO_s_mem());

         // Write key from kbuf to BIO
         BIO_write(bpub,(void *)str,strlen(str));

         // Read pub key from BIO
         if (!(RSAtmp = PEM_read_bio_RSAPublicKey(bpub, 0, 0, 0))) {
            if (gDebug > 0)
              ::Info("TAuthenticate::DecodeRSAPublic",
                     "unable to read pub key from bio");
         } else
            *RSASSL = (void *)RSAtmp;
      }
#else
      } else {
         if (gDebug > 0)
            ::Info("TAuthenticate::DecodeRSAPublic","not compiled with SSL support:"
                      " you should not have got here!");
      }
#endif
   }

   return keytype;
}

//______________________________________________________________________________
Int_t TAuthenticate::SetRSAPublic(const char *RSAPubExport, Int_t klen)
{
   // Store RSA public keys from export string RSAPubExport.
   // Returns type of stored key, or -1 is not recognized

   if (gDebug > 2)
      ::Info("TAuthenticate::SetRSAPublic",
             "enter: string length %d bytes", strlen(RSAPubExport));

   Int_t RSAKey = -1;
   if (!RSAPubExport)
      return RSAKey;

   if (klen > 0) {

      // Skip spaces at beginning, if any
      int k = 0;
      while (RSAPubExport[k] == 32) k++;

      if (RSAPubExport[k] == '#') {

         // Decode input string
         rsa_NUMBER RSA_n, RSA_d;
         RSAKey = TAuthenticate::DecodeRSAPublic(RSAPubExport,RSA_n,RSA_d);

         // Save Public key
         rsa_fun::fg_rsa_assign(&fgRSAPubKey.n, &RSA_n);
         rsa_fun::fg_rsa_assign(&fgRSAPubKey.e, &RSA_d);

      } else {
         RSAKey = 1;
#ifdef R__SSL
         // Now set the key locally in BF form
         BF_set_key(&fgBFKey, klen, (const unsigned char *)RSAPubExport);
#else
         if (gDebug > 0)
            ::Info("TAuthenticate::SetRSAPublic",
                   "not compiled with SSL support:"
                   " you should not have got here!");
#endif
      }
   }

   return RSAKey;
}

//______________________________________________________________________________
Int_t TAuthenticate::SendRSAPublicKey(TSocket *socket, Int_t key)
{
   // Receives Server RSA Public key
   // Sends local RSA public key encoded

   // Receive server public key
   char serverPubKey[kMAXSECBUF];
   int kind, nr = 0;
   if ((nr = socket->Recv(serverPubKey, kMAXSECBUF, kind)) < 0)
      return nr;
   if (gDebug > 3)
      ::Info("TAuthenticate::SendRSAPublicKey",
             "received key from server %d bytes", strlen(serverPubKey));

   // Decode it
   rsa_NUMBER RSA_n, RSA_d;
#ifdef R__SSL
   RSA *RSASSLServer = 0;
   if (TAuthenticate::DecodeRSAPublic(serverPubKey,RSA_n,RSA_d,
                                      (void **)&RSASSLServer) != key) {
      if (RSASSLServer)
         RSA_free(RSASSLServer);
      return -1;
   }
#else
   if (TAuthenticate::DecodeRSAPublic(serverPubKey,RSA_n,RSA_d) != key)
      return -1;
#endif

   // Send local public key, encodes
   char buftmp[kMAXSECBUF];
   char buflen[20];
   Int_t slen = fgRSAPubExport[key].len;
   Int_t ttmp = 0;
   if (key == 0) {
      strncpy(buftmp,fgRSAPubExport[key].keys,slen);
      buftmp[slen] = 0;
      ttmp = rsa_fun::fg_rsa_encode(buftmp, slen, RSA_n, RSA_d);
      sprintf(buflen, "%d", ttmp);
   } else if (key == 1) {
#ifdef R__SSL
      Int_t lcmax = RSA_size(RSASSLServer) - 11;
      Int_t kk = 0;
      Int_t ke = 0;
      Int_t ns = slen;
      while (ns > 0) {
         Int_t lc = (ns > lcmax) ? lcmax : ns ;
         if ((ttmp = RSA_public_encrypt(lc,
                    (unsigned char *)&fgRSAPubExport[key].keys[kk],
                    (unsigned char *)&buftmp[ke],
                     RSASSLServer,RSA_PKCS1_PADDING)) < 0) {
            char cerr[120];
            ERR_error_string(ERR_get_error(), cerr);
            ::Info("TAuthenticate::SendRSAPublicKey","SSL: error: '%s' ",cerr);
         }
         kk += lc;
         ke += ttmp;
         ns -= lc;
      }
      ttmp = ke;
      sprintf(buflen, "%d", ttmp);
#else
      if (gDebug > 0)
         ::Info("TAuthenticate::SendRSAPublicKey","not compiled with SSL support:"
                " you should not have got here!");
      return -1;
#endif
   } else {
      if (gDebug > 0)
         ::Info("TAuthenticate::SendRSAPublicKey","unknown key type (%d)",key);
#ifdef R__SSL
      if (RSASSLServer)
         RSA_free(RSASSLServer);
#endif
      return -1;
   }

   // Send length first
   if ((nr = socket->Send(buflen, kROOTD_ENCRYPT)) < 0)
      return nr;
   // Send Key. second ...
   Int_t nsen = socket->SendRaw(buftmp, ttmp);
   if (gDebug > 3)
         ::Info("TAuthenticate::SendRSAPublicKey",
                "local: sent %d bytes (expected: %d)", nsen,ttmp);
#ifdef R__SSL
   if (RSASSLServer)
      RSA_free(RSASSLServer);
#endif
   return nsen;
}

//______________________________________________________________________________
Int_t TAuthenticate::GetClientProtocol()
{
   // Static method returning supported client protocol.

   return fgClientProtocol;
}

//______________________________________________________________________________
void TAuthenticate::CleanupSecContextAll(Option_t *opt)
{
   // Ask remote client to cleanup all active security context
   // Static method called in TROOT for final cleanup

   TIter next(gROOT->GetListOfSecContexts());
   TSecContext *nsc ;
   while ((nsc = (TSecContext *)next())) {
      if (nsc->IsActive()) {
         TAuthenticate::CleanupSecContext(nsc,kTRUE);
         nsc->DeActivate("");
         // All have been remotely Deactivated
         TIter nxtl(gROOT->GetListOfSecContexts());
         TSecContext *nscl;
         while ((nscl = (TSecContext *)nxtl())) {
            if (nscl != nsc && !strcmp(nscl->GetHost(),nsc->GetHost())) {
               // Need to set ofs=-1 to avoid sending another
               // cleanup request
               nscl->DeActivate("");
            }
         }
      }
   }

   // Clear the list
   gROOT->GetListOfSecContexts()->Clear();

   if (opt && !strncasecmp(opt,"k",1)) {
      // We are quitting, so cleanup memory also memory
      Int_t i = 2;
      while (i--) {
         if (fgRSAPubExport[i].keys)
            delete[] fgRSAPubExport[i].keys;
         fgRSAPubExport[i].len = 0;
      }
   }
}
//______________________________________________________________________________
Bool_t TAuthenticate::CleanupSecContext(TSecContext *ctx, Bool_t all)
{
   // Ask remote client to cleanup security context 'ctx'
   // If 'all', all sec context with the same host as ctx
   // are cleaned.
   // Static method called by ~TSecContext

   Bool_t cleaned = kFALSE;

   // Nothing to do if inactive ...
   if (!ctx->IsActive())
      return kTRUE;

   // Contact remote services that used this context,
   // starting from the last ...
   TIter last(ctx->GetSecContextCleanup(),kIterBackward);
   TSecContextCleanup *nscc = 0;
   while ((nscc = (TSecContextCleanup *)last()) && !cleaned) {

      // First check if remote daemon supports cleaning
      Int_t srvtyp = nscc->GetType();
      Int_t rproto = nscc->GetProtocol();
      Int_t level = 2;
      if ((srvtyp == TSocket::kROOTD && rproto < 10) ||
          (srvtyp == TSocket::kPROOFD && rproto < 9))
         level = 1;
      if ((srvtyp == TSocket::kROOTD && rproto < 8) ||
          (srvtyp == TSocket::kPROOFD && rproto < 7))
         level = 0;
      if (level) {
         TString rHost(ctx->GetHost());
         Int_t port = nscc->GetPort();

         TSocket *news = new TSocket(rHost.Data(),port,-1);

         if (news && news->IsValid()) {
            if (srvtyp == TSocket::kPROOFD) {
               news->SetOption(kNoDelay, 1);
               news->Send("cleaning request");
            } else
               news->SetOption(kNoDelay, 0);

            // Backward compatibility: send socket size
            if (srvtyp == TSocket::kROOTD && level == 1)
               news->Send((Int_t)0, (Int_t)0);

            if (all || level == 1) {
               news->Send(Form("%d",gProcessID),
                               kROOTD_CLEANUP);
               cleaned = kTRUE;
            } else {
               news->Send(Form("%d %d %d %s",gProcessID,ctx->GetMethod(),
                               ctx->GetOffSet(),ctx->GetUser()),kROOTD_CLEANUP);
               if (TAuthenticate::SecureSend(news, 1, ctx->GetRSAKey(),
                  (char *)ctx->GetToken()) == -1) {
                  ::Info("CleanupSecContext", "problems securesending token");
               } else {
                  cleaned = kTRUE;
               }
            }
            if (cleaned && gDebug > 2) {
               char srvname[3][10] = {"sockd", "rootd", "proofd"};
               ::Info("CleanupSecContext",
                    "remote %s notified for cleanup (%s,%d)",
                    srvname[srvtyp],rHost.Data(),port);
            }
         }
         SafeDelete(news);
      }
   }

   if (!cleaned)
      if (gDebug > 2)
         ::Info("CleanupSecContext",
                "unable to open valid socket for cleanup for %s",
                 ctx->GetHost());

   return cleaned;

}

//______________________________________________________________________________
Int_t TAuthenticate::ReadRootAuthrc(const char *proofconf)
{
   // Read authentication directives from $ROOTAUTHRC, $HOME/.rootauthrc or
   // <Root_etc_dir>/system.rootauthrc and create related THostAuth objects.
   // Files are read only if they changed since last reading
   // If 'proofconf' is defined, check also file proofconf for directives

   // rootauthrc family
   char *authrc = 0;
   if (gSystem->Getenv("ROOTAUTHRC") != 0) {
      authrc = StrDup(gSystem->Getenv("ROOTAUTHRC"));
   } else {
      if (fgReadHomeAuthrc)
         authrc = gSystem->ConcatFileName(gSystem->HomeDirectory(), ".rootauthrc");
   }
   if (authrc && gDebug > 2)
      ::Info("TAuthenticate::ReadRootAuthrc", "Checking file: %s", authrc);
   if (!authrc || gSystem->AccessPathName(authrc, kReadPermission)) {
      if (authrc && gDebug > 1)
         ::Info("TAuthenticate::ReadRootAuthrc",
                "file %s cannot be read (errno: %d)", authrc, errno);
      delete [] authrc;
#ifdef ROOTETCDIR
      authrc = gSystem->ConcatFileName(ROOTETCDIR,"system.rootauthrc");
#else
      char etc[1024];
#ifdef WIN32
      sprintf(etc, "%s\\etc", gRootDir);
#else
      sprintf(etc, "%s/etc", gRootDir);
#endif
      authrc = gSystem->ConcatFileName(etc,"system.rootauthrc");
#endif
      if (gDebug > 2)
         ::Info("TAuthenticate::ReadRootAuthrc", "Checking system file:%s",authrc);
      if (gSystem->AccessPathName(authrc, kReadPermission)) {
         if (gDebug > 1)
            ::Info("TAuthenticate::ReadRootAuthrc",
                   "file %s cannot be read (errno: %d)", authrc, errno);
         delete [] authrc;
         return 0;
      }
   }

   // Check if file has changed since last read
   TString tRootAuthrc = authrc;
   if (tRootAuthrc == fgRootAuthrc) {
      struct stat si;
      stat(tRootAuthrc, &si);
      if ((UInt_t)si.st_mtime < fgLastAuthrc.Convert()) {
         if (gDebug > 1)
            ::Info("TAuthenticate::ReadRootAuthrc",
                   "file %s already read", authrc);
         delete [] authrc;
         return 0;
      }
   }

   // Save filename in static variable
   fgRootAuthrc = tRootAuthrc;
   fgLastAuthrc = TDatime();

   // THostAuth lists
   TList *AuthInfo = TAuthenticate::GetAuthInfo();
   TList *ProofAuthInfo = TAuthenticate::GetProofAuthInfo();

   // Expand File into temporary file name and open it
   int expand = 1;
   TString filetmp = "rootauthrc";
   FILE *ftmp = gSystem->TempFileName(filetmp);
   if (gDebug > 2)
      ::Info("TAuthenticate::ReadRootAuthrc", "got tmp file: %s open at 0x%lx",
              filetmp.Data(), (Long_t)ftmp);
   if (ftmp == 0)
      expand = 0;  // Problems opening temporary file: ignore 'include's ...

   FILE *fd = 0;
   // If the temporary file is open, copy everything to the new file ...
   if (expand == 1) {
      TAuthenticate::FileExpand(authrc, ftmp);
      fd = ftmp;
      rewind(fd);
   } else {
      // Open file
      fd = fopen(authrc, "r");
      if (fd == 0) {
         if (gDebug > 2)
            ::Info("TAuthenticate::ReadRootAuthrc",
                   "file %s cannot be open (errno: %d)", authrc, errno);
         delete [] authrc;
         return 0;
      }
   }

   // Now scan file for meaningful directives
   TList TmpAuthInfo;
   char line[kMAXPATHLEN];
   Bool_t cont = kFALSE;
   TString ProofServ;
   while (fgets(line, sizeof(line), fd) != 0) {

      // Skip comment lines
      if (line[0] == '#')
         continue;

      // Get rid of end of line '\n', if there ...
      if (line[strlen(line) - 1] == '\n')
         line[strlen(line) - 1] = '\0';

      // Skip empty lines
      if (strlen(line) == 0)
         continue;

      // Now scan
      char *tmp = new char[strlen(line)+1];
      strcpy(tmp,line);
      char *nxt = strtok(tmp," ");

      if (!strcmp(nxt, "proofserv") || cont) {

         // Building the list of data servers for proof (analyzed at the end)
         char *ph = 0;
         if (cont)
            ph = nxt;
         else
            ph = strtok(0," ");
         while (ph) {
            if (*ph != 92) {
               ProofServ += TString((const char *)ph);
               ProofServ += TString(" ");
               cont = kFALSE;
            } else {
               cont = kTRUE;
            }
            ph = strtok(0," ");
         }

      } else {

         TString hostsrv = nxt;
         TString host   = hostsrv;
         TString server = "";
         if (hostsrv.Contains(":")) {
            server = hostsrv;
            host.Remove(host.Index(":"));
            server.Remove(0,server.Index(":")+1);
         }
         Int_t srvtyp = -1;
         if (server.Length()) {
            if (server == "0" || server.BeginsWith("sock"))
               srvtyp = TSocket::kSOCKD;
            else if (server == "1" || server.BeginsWith("root"))
               srvtyp = TSocket::kROOTD;
            else if (server == "2" || server.BeginsWith("proof"))
               srvtyp = TSocket::kPROOFD;
         }

         // Line with host info directives
         TString user = "*";

         char *nxt = strtok(0," ");
         if (!strncmp(nxt,"user",4)) {
            nxt = strtok(0," ");
            if (strncmp(nxt,"list",4) && strncmp(nxt,"method",6)) {
               user = TString(nxt);
               nxt = strtok(0," ");
            }
         }

         // Get related THostAuth, if exists in the tmp list,
         TIter next(&TmpAuthInfo);
         THostAuth *ha;
         while ((ha = (THostAuth *)next())) {
            if (host == ha->GetHost() && user == ha->GetUser() &&
                srvtyp == ha->GetServer())
               break;
         }
         if (!ha) {
            // Create a new one
            ha = new THostAuth(host,srvtyp,user);
            TmpAuthInfo.Add(ha);
         }

         if (!strncmp(nxt,"list",4)) {
            // list of methods for {host,usr}
            Int_t nm = 0, me[kMAXSEC] = {0};
            char *mth = strtok(0," ");
            while (mth) {
               Int_t met = -1;
               if (strlen(mth) > 1) {
                  // Method passed as string: translate it to number
                  met = GetAuthMethodIdx(mth);
                  if (met == -1 && gDebug > 2)
                     ::Info("TAuthenticate::ReadRootAuthrc",
                            "unrecognized method (%s): ", mth);
               } else {
                  met = atoi(mth);
               }
               if (met > -1 && met < kMAXSEC)
                  me[nm++] = met;
               mth = strtok(0," ");
            }
            if (nm)
               ha->ReOrder(nm,me);

         } else if (!strncmp(nxt,"method",6)) {

            // details for {host,usr,method}
            char *mth = strtok(0," ");
            Int_t met = -1;
            if (strlen(mth) > 1) {
               // Method passed as string: translate it to number
               met = GetAuthMethodIdx(mth);
               if (met == -1 && gDebug > 2)
                  ::Info("TAuthenticate::ReadRootAuthrc",
                         "unrecognized method (%s): ", mth);
            } else {
               met = atoi(mth);
            }
            if (met > -1 && met < kMAXSEC) {
               const char *det = 0;
               nxt = strtok(0," ");
               if (nxt) {
                  det = (const char *)strstr(line,nxt);
               }
               if (ha->HasMethod(met))
                  ha->SetDetails(met,det);
               else
                  ha->AddMethod(met,det);
            }
         }
      }
      if (tmp) delete[] tmp;
   }
   // Close file and remove it if temporary
   fclose(fd);
   if (expand == 1)
      gSystem->Unlink(filetmp);
   // Cleanup allocated memory
   if (authrc) delete[] authrc;

   // Update AuthInfo with new info found
   TAuthenticate::MergeHostAuthList(AuthInfo,&TmpAuthInfo);

   // Print those left, if requested ...
   if (gDebug > 2)
      TAuthenticate::Show();

   // Now create the list of THostAuth to be sent over to
   // the Master/Slaves, if requested ...
   TList TmpProofAuthInfo;
   if (ProofServ.Length() > 0) {
      char *tmp = new char[ProofServ.Length()+1];
      strcpy(tmp,ProofServ.Data());
      char *nxt = strtok(tmp," ");
      while (nxt) {
         TString Tmp((const char *)nxt);
         Int_t pdd = -1;
         // host
         TString host;
         if ((pdd = Tmp.Index(":")) == -1) {
            host = Tmp;
         } else {
            host = Tmp;
            host.Resize(pdd);
            if (!host.Length())
               host = "*";
            Tmp.Remove(0,pdd+1);
         }
         // user
         TString user;
         if ((pdd = Tmp.Index(":")) == -1) {
            user = Tmp;
         } else {
            user = Tmp;
            user.Resize(pdd);
            if (!user.Length())
               user = "*";
            Tmp.Remove(0,pdd+1);
         }
         // method(s)
         TString meth;
         Int_t nm = 0, me[kMAXSEC] = {0}, met = -1;
         while (Tmp.Length() > 0) {
            meth = Tmp;
            if ((pdd = Tmp.Index(":")) > -1)
               meth.Resize(pdd);
            if (meth.Length() > 1) {
               // Method passed as string: translate it to number
               met = GetAuthMethodIdx(meth.Data());
               if (met == -1 && gDebug > 2)
                  ::Info("TAuthenticate::ReadRootAuthrc",
                         "unrecognized method (%s): ",meth.Data());
            } else if (meth.Length() == 1) {
               met = atoi(meth.Data());
               if (met > -1 && met < kMAXSEC)
                  me[nm++] = met;
            }
            if (pdd > -1)
               Tmp.Remove(0,pdd+1);
            else
               Tmp.Resize(0);
         }

         // Get related THostAuth, if exists, or create a new one
         THostAuth *ha = 0;
         THostAuth *hatmp = TAuthenticate::GetHostAuth(host,user);
         if (!hatmp) {
            ha = new THostAuth(host,user,nm,me,0);
         } else {
            // Create an empty THostAuth
            ha = new THostAuth(host,user);
            // Update with hatmp info
            ha->Update(hatmp);
            // ReOrder following new directives
            ha->ReOrder(nm,me);
         }
         // Add to the tmp list
         TmpProofAuthInfo.Add(ha);
         // Go to next
         nxt = strtok(0," ");
      }
      if (tmp) delete[] tmp;
   }

   // Update ProofAuthInfo with new info found
   TAuthenticate::MergeHostAuthList(ProofAuthInfo,&TmpProofAuthInfo,"P");
   // Print those, if requested ...
   if (gDebug > 2)
      TAuthenticate::Show("P");

   // If Proof Master scan also <proof.conf> alike files
   if (proofconf)
      TAuthenticate::ReadProofConf(proofconf);

   return AuthInfo->GetSize();
}


//______________________________________________________________________________
void TAuthenticate::ReadProofConf(const char *conffile)
{
   // Collect information needed for authentication to slaves from
   // $HOME/.proof.conf or <Root_Dir>/proof/etc/proof.conf
   // Update or create THostAuth objects accordingly
   // Add them to the ProofAuthInfo list.

   if (gDebug > 2)
      ::Info("ReadProofConf", "Enter ... (%s)", conffile);

   // Get pointer to lists with authentication info
   TList *AuthInfo = GetAuthInfo();

   // Check authentication methods applicability
   Int_t i = 0;
   Bool_t AuthAvailable[kMAXSEC] = {0};
   TString AuthDet[kMAXSEC];
   for (; i < kMAXSEC; i++){
      AuthAvailable[i] = kFALSE;
      if (i == 0 && fgUser != "" && fgPasswd != "") {
         AuthAvailable[i] = kTRUE;
         AuthDet[i] = TString(Form("pt:0 ru:1 us:%s", fgUser.Data()));
      } else {
         AuthAvailable[i] = CheckProofAuth(i,AuthDet[i]);
      }
      if (gDebug > 2)
         ::Info("ReadProofConf","meth:%d avail:%d det:%s",
                i,AuthAvailable[i],AuthDet[i].Data());
   }

   // Check configuration file
   Bool_t HaveConf = kTRUE;
   char fconf[256];
   sprintf(fconf, "%s/.%s", gSystem->Getenv("HOME"), conffile);
   if (gDebug > 2)
      ::Info("ReadProofConf", "checking PROOF config file %s", fconf);
   if (gSystem->AccessPathName(fconf, kReadPermission)) {
      TApplication *lApp = gROOT->GetApplication();
      sprintf(fconf, "%s/proof/etc/%s",lApp->Argv(2), conffile);
      if (gDebug > 2)
         ::Info("ReadProofConf", "checking PROOF config file %s", fconf);
      if (gSystem->AccessPathName(fconf, kReadPermission)) {
         if (gDebug > 1)
            ::Info("ReadProofConf", "no PROOF config file found");
         HaveConf = kFALSE;
      }
   } else {
      if (gDebug > 2)
         ::Info("ReadProofConf", "using PROOF config file: %s", fconf);
   }

   // Scan config file for authentication directives
   if (HaveConf) {

      FILE *pconf;
      if ((pconf = fopen(fconf, "r"))) {

         // read the config file
         char line[256];
         while (fgets(line, sizeof(line), pconf)) {

            // Skip comment lines
            if (line[0] == '#')
               continue;

            // Skip lines not containing slave info
            if (!strstr(line,"slave"))
               continue;

            // Get rid of end of line '\n', if there ...
            if (line[strlen(line) - 1] == '\n')
               line[strlen(line) - 1] = '\0';

            // Now scan
            char *tmp = new char[strlen(line)+1];
            strcpy(tmp,line);
            char *nxt = strtok(tmp," ");

            // First should "slave"
            if (strncmp(nxt,"slave",5))
               continue;

            // Save slave host name
            TString SlaveHost((const char *)strtok(0," "));

            Int_t nm = 0, me[kMAXSEC] = {0};
            TString det[kMAXSEC];
            char *mth = strtok(0," ");
            while (mth) {

               // {port,perf,image} entries all have a '='
               if (strncmp(mth,"=",1)) {

                  Int_t met = -1;
                  if (strlen(mth) > 1) {
                     // Method passed as string: translate it to number
                     met = GetAuthMethodIdx(mth);
                     if (met == -1 && gDebug > 2)
                        ::Info("ReadProofConf",
                               "unrecognized method (%s): ", mth);
                  } else {
                     met = atoi(mth);
                  }
                  if (met > -1 && met < kMAXSEC) {
                     if (AuthAvailable[met]) {
                        det[nm] = AuthDet[met];
                        me[nm++] = met;
                     }
                  }
               }
               // Get next
               mth = strtok(0," ");
            }
            if (mth) delete[] mth;

            // Check if a HostAuth object for this (host,user) pair already exists
            TString SlaveSrv(Form("%s:%d",SlaveHost.Data(),TSocket::kPROOFD));
            THostAuth *ha = TAuthenticate::GetHostAuth(SlaveSrv,fgUser);

            if (!ha || !strcmp(ha->GetHost(),"default")) {
               if (ha) {
                  // Got a default entry: create a new one from a copy
                  THostAuth *han = new THostAuth(*ha);
                  ha = han;
                  ha->SetHost(SlaveHost);
                  ha->SetServer(TSocket::kPROOFD);
                  ha->SetUser(fgUser);
                  // Reset list of established sec context;
                  TList *nl = new TList;
                  ha->SetEstablished(nl);
               } else
                  // Create new one and add it to the list
                  ha = new THostAuth(SlaveHost,TSocket::kPROOFD,fgUser);

               // Add UidGid if not already there
               Int_t kLocalRfio = TAuthenticate::kRfio;
               if (!ha->HasMethod(kLocalRfio))
                  ha->AddMethod(kLocalRfio,AuthDet[kLocalRfio]);

               // Add this ThostAuth to lists
               AuthInfo->Add(ha);
            }

            // Reorder accordingly to new directives
            Int_t i = nm;
            for(; i > 0; i--)
               ha->AddFirst(me[i-1],det[i-1]);

            if (tmp) delete[] tmp;

         } // fgets

      } // fopen

      // close file
      fclose(pconf);
   }

}

//______________________________________________________________________________
Bool_t TAuthenticate::CheckProofAuth(Int_t cSec, TString &Out)
{
   // Check if the authentication method can be attempted for the client.

   Bool_t rc = kFALSE;
   const char sshid[3][20] = { "/.ssh/identity", "/.ssh/id_dsa", "/.ssh/id_rsa" };
   const char netrc[2][20] = { "/.netrc", "/.rootnetrc" };
   TString Details, User;

   // Get user logon name
   UserGroup_t *pw = gSystem->GetUserInfo();
   if (pw) {
      User = TString(pw->fUser);
      delete pw;
   } else {
      ::Info("CheckProofAuth",
             "not properly logged on (getpwuid unable to find relevant info)!");
      Out = "";
      return rc;
   }

   // UsrPwd
   if (cSec == (Int_t) TAuthenticate::kClear) {
      Int_t i = 0;
      for (; i < 2; i++) {
         TString infofile = TString(gSystem->HomeDirectory())+TString(netrc[i]);
         if (!gSystem->AccessPathName(infofile, kReadPermission))
            rc = kTRUE;
      }
      if (rc)
         Out = Form("pt:0 ru:1 us:%s",User.Data());
   }

   // SRP
   if (cSec == (Int_t) TAuthenticate::kSRP) {
#ifdef R__SRP
      Out = Form("pt:0 ru:1 us:%s",User.Data());
      rc = kTRUE;
#endif
   }

   // Kerberos
   if (cSec == (Int_t) TAuthenticate::kKrb5) {
#ifdef R__KRB5
      Out = Form("pt:0 ru:0 us:%s",User.Data());
      rc = kTRUE;
#endif
   }

   // Globus
   if (cSec == (Int_t) TAuthenticate::kGlobus) {
#ifdef R__GLBS
      TApplication *lApp = gROOT->GetApplication();
      if (lApp != 0 && lApp->Argc() > 9) {
         if (gROOT->IsProofServ()) {
            // Delegated Credentials
            Int_t ShmId = atoi(lApp->Argv(9));
            if (ShmId != -1) {
               struct shmid_ds shm_ds;
               if (shmctl(ShmId, IPC_STAT, &shm_ds) == 0)
                  rc = kTRUE;
            }
            if (rc) {
               // Build details .. CA dir
               TString Adir(gSystem->Getenv("X509_CERT_DIR"));
               // Usr Cert
               TString Ucer(gSystem->Getenv("X509_USER_CERT"));
               // Usr Key
               TString Ukey(gSystem->Getenv("X509_USER_KEY"));
               // Usr Dir
               TString Cdir = Ucer;
               Cdir.Resize(Cdir.Last('/')+1);
               // Create Output
               Out = Form("pt=0 ru:1 cd:%s cf:%s kf:%s ad:%s",
                          Cdir.Data(),Ucer.Data(),Ukey.Data(),Adir.Data());
            }
         }
      }
#endif
   }

   // SSH
   if (cSec == (Int_t) TAuthenticate::kSSH) {
      Int_t i = 0;
      for (; i < 3; i++) {
         TString infofile = TString(gSystem->HomeDirectory())+TString(sshid[i]);
         if (!gSystem->AccessPathName(infofile,kReadPermission))
            rc = kTRUE;
      }
      if (rc)
         Out = Form("pt:0 ru:1 us:%s",User.Data());
   }

   // Rfio
   if (cSec == (Int_t) TAuthenticate::kRfio) {
      Out = Form("pt:0 ru:0 us:%s",User.Data());
      rc = kTRUE;
   }

   if (gDebug > 3) {
      if (strlen(Out) > 0)
         ::Info("CheckProofAuth",
                "meth: %d ... is available: details: %s", cSec, Out.Data());
      else
         ::Info("CheckProofAuth",
                "meth: %d ... is NOT available", cSec);
   }

   // return
   return rc;
}


//______________________________________________________________________________
Int_t StdCheckSecCtx(const char *User, TSecContext *Ctx)
{
   // Standard version of CheckSecCtx to be passed to TAuthenticate::AuthExists
   // Check if User is matches the one in Ctx
   // Returns: 1 if ok, 0 if not
   // Deactivates Ctx is not valid

   Int_t rc = 0;

   if (Ctx->IsActive()) {
      if (!strcmp(User,Ctx->GetUser()))
         rc = 1;
   }
   return rc;
}

//______________________________________________________________________________
void TAuthenticate::MergeHostAuthList(TList *Std, TList *New, Option_t *Opt)
{
   // Tool for updating fgAuthInfo or fgProofAuthInfo
   // 'New' contains list of last input information through (re)reading
   // of a rootauthrc-alike file. 'New' info has priority.
   // 'Std' is cleaned from inactive members.
   // 'New' members used to update existing members in 'Std' are
   // removed from 'New', do that they do not leak
   // Opt = "P" for ProofAuthInfo.

   // Remove inactive from the 'Std'
   TIter nxstd(Std);
   THostAuth *ha;
   while ((ha = (THostAuth *) nxstd())) {
      if (!ha->IsActive()) {
         Std->Remove(ha);
         SafeDelete(ha);
      }
   }

   // Merge 'New' info in 'Std'
   TIter nxnew(New);
   THostAuth *hanew;
   while ((hanew = (THostAuth *)nxnew())) {
      if (hanew->NumMethods()) {
         TString HostSrv(Form("%s:%d",hanew->GetHost(),hanew->GetServer()));
         THostAuth *hastd =
            TAuthenticate::HasHostAuth(HostSrv,hanew->GetUser(),Opt);
         if (hastd) {
            // Update with new info
            hastd->Update(hanew);
            // Flag for removal
            hanew->DeActivate();
         } else {
            // Add new ThostAuth to Std
            Std->Add(hanew);
         }
      } else
         // Flag for removal empty objects
         hanew->DeActivate();
   }

   // Cleanup memory before quitting
   nxnew.Reset();
   while ((hanew = (THostAuth *)nxnew())) {
      if (!hanew->IsActive()) {
         New->Remove(hanew);
         SafeDelete(hanew);
      }
   }

}

//______________________________________________________________________________
void TAuthenticate::RemoveSecContext(TSecContext *ctx)
{
   // Tool for removing SecContext ctx from THostAuth listed in
   // fgAuthInfo or fgProofAuthInfo

   THostAuth *ha = 0;

   // AuthInfo first
   TIter nxai(GetAuthInfo());
   while ((ha = (THostAuth *)nxai())) {
      TIter next(ha->Established());
      TSecContext *lctx = 0;
      while ((lctx = (TSecContext *) next())) {
         if (lctx == ctx) {
            ha->Established()->Remove(ctx);
            break;
         }
      }
   }

   // ProofAuthInfo second
   TIter nxpa(GetProofAuthInfo());
   while ((ha = (THostAuth *)nxpa())) {
      TIter next(ha->Established());
      TSecContext *lctx = 0;
      while ((lctx = (TSecContext *) next())) {
         if (lctx == ctx) {
            ha->Established()->Remove(ctx);
            break;
         }
      }
   }

}
