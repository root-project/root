// @(#)root/net:$Name:  $:$Id: TAuthenticate.cxx,v 1.30 2003/11/18 19:28:25 rdm Exp $
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
#include "THostAuth.h"
#include "TAuthDetails.h"
#include "TNetFile.h"
#include "TSocket.h"
#include "TSystem.h"
#include "TError.h"
#include "Getline.h"
#include "TROOT.h"
#include "TEnv.h"
#include "TList.h"
#include "NetErrors.h"
#include "TRegexp.h"

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


Int_t TAuthenticate::fgRSAInit = 0;
rsa_KEY TAuthenticate::fgRSAPriKey;
rsa_KEY TAuthenticate::fgRSAPubKey;
rsa_KEY_export TAuthenticate::fgRSAPubExport = { 0, 0 };

TString TAuthenticate::fgUser;
TString TAuthenticate::fgPasswd;
Bool_t TAuthenticate::fgPwHash;
Bool_t TAuthenticate::fgSRPPwd;
SecureAuth_t TAuthenticate::fgSecAuthHook;
Krb5Auth_t TAuthenticate::fgKrb5AuthHook;
GlobusAuth_t TAuthenticate::fgGlobusAuthHook;
TString TAuthenticate::fgDefaultUser;
Bool_t TAuthenticate::fgAuthReUse;
Bool_t TAuthenticate::fgPromptUser;
Bool_t TAuthenticate::fgUsrPwdCrypt;

TList *TAuthenticate::fgAuthInfo = 0;

TString TAuthenticate::fgAuthMeth[] = { "UsrPwd", "SRP", "Krb5", "Globus", "SSH", "UidGid" };

ClassImp(TAuthenticate)

//______________________________________________________________________________
TAuthenticate::TAuthenticate(TSocket *sock, const char *remote,
                             const char *proto, const char *user)
{
   // Create authentication object.

   fSocket   = sock;
   fRemote   = remote;
   fHostAuth = 0;
   fVersion  = 3;                // The latest, by default
   fRSAKey   = 0;

   if (gDebug > 2)
      Info("TAuthenticate", "Enter: local host: %s, user is: %s (proto: %s)",
           gSystem->HostName(), user, proto);

   // Set protocol string.
   // Check if version should be different ...
   char *pdd;
   if (proto && strlen(proto) > 0) {
      char *sproto = StrDup(proto);
      if ((pdd = strstr(sproto, ":")) != 0) {
         int rproto = atoi(pdd + 1);
         *pdd = '\0';
         if (strstr(sproto, "root") != 0) {
            if (rproto < 9 ) {
               fVersion = 2;
               if (rproto < 8) {
                  fVersion = 1;
                  if (rproto < 6)
                     fVersion = 0;
               }
            }
         }
         if (strstr(sproto, "proof") != 0) {
            if (rproto < 8) {
               fVersion = 2;
               if (rproto < 7)
                  fVersion = 1;
            }
         }
         if (gDebug > 3)
            Info("TAuthenticate",
                 "service: %s (remote protocol: %d): fVersion: %d", sproto,
                 rproto, fVersion);
      }
      fProtocol = sproto;
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
   if (gDebug > 3)
      Info("TAuthenticate",
           "number of HostAuth Instantiations in memory: %d",
           GetAuthInfo()->GetSize());

   // Check list of auth info for already loaded info about this host
   fHostAuth = GetHostAuth(fqdn, CheckUser);

   // If we did not find a good THostAuth instantiation, create one
   if (fHostAuth == 0) {
      // Determine applicable auth methods from client choices
      Int_t *nmeth;
      Int_t *security[kMAXSEC];
      char **details[kMAXSEC];
      char **usr = new char *[1];
      usr[0] = StrDup(fUser);
      GetAuthMeth(fqdn, fProtocol, &usr, &nmeth, security, details);

      // Translate to Input for THostAuth
      int i, nm = nmeth[0], am[kMAXSEC];
      char *det[kMAXSEC];
      for (i = 0; i < kMAXSEC; i++) {
         if (i < nm) {
            am[i] = security[i][0];
            det[i] = StrDup(details[i][0]);
         } else {
            am[i] = -1;
            det[i] = 0;
         }
      }
      if (gDebug > 3) {
         Info("TAuthenticate", "got %d methods", nmeth[0]);
         for (i = 0; i < nmeth[0]; i++) {
            Info("TAuthenticate", "got (%d,0) security:%d details:%s", i,
                 security[i][0], details[i][0]);
         }
      }
      // Create THostAuth object
      fHostAuth = new THostAuth(fqdn, fUser, nm, am, det);
      // ... and add it to the list
      GetAuthInfo()->Add(fHostAuth);
      if (gDebug > 3)
         fHostAuth->Print();
      for (i = 0; i < nmeth[0]; i++) {   // what if nu > 0? (rdm)
         if (security[i]) delete[] security[i];
         if (details[i][0]) delete[] details[i][0];
         if (det[i]) delete[] det[i];
      }
      if (nmeth) delete [] nmeth;
      if (usr[0]) delete[] usr[0];
   }

   // If a secific method has been requested via the protocol
   // set it as first
   Int_t Sec = -1;
   if (fProtocol.Contains("roots") || fProtocol.Contains("proofs")) {
      Sec = TAuthenticate::kSRP;
   } else if (fProtocol.Contains("rootk") || fProtocol.Contains("proofk")) {
      Sec = TAuthenticate::kKrb5;
   }
   if (Sec > -1 && Sec < kMAXSEC) {
      if (fHostAuth->HasMethod(Sec)) {
         fHostAuth->SetFirst(Sec);
      } else {
         TString Det(GetDefaultDetails(Sec, 1, CheckUser));
         fHostAuth->SetFirst(Sec, Det);
      }
   }

   // This is what we have in memory
   if (gDebug > 3) {
      TIter next(fHostAuth->Established());
      TAuthDetails *ad;
      while ((ad = (TAuthDetails *) next()))
         ad->Print("0");
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
   fSecurity = (ESecurity) fHostAuth->GetMethods(meth);
   fDetails = fHostAuth->GetDetails((Int_t) fSecurity);
   if (gDebug > 2)
      Info("Authenticate",
           "trying authentication: method:%d, default details:%s",
           fSecurity, fDetails.Data());

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

      // UsrPwd Authentication
      user = fgDefaultUser;
      if (user != "")
         CheckNetrc(user, passwd, pwhash, (Bool_t &)kFALSE);
      if (passwd == "") {
         if (fgPromptUser)
            user = PromptUser(fRemote);
         if (GetUserPasswd(user, passwd, pwhash, (Bool_t &)kFALSE))
            return kFALSE;
      }
      fUser = user;
      fPasswd = passwd;

      if (fUser != "root")
         st = ClearAuth(user, passwd, pwhash);

   } else if (fSecurity == kSRP) {

      // SRP Authentication
      user = fgDefaultUser;
      if (user != "")
         CheckNetrc(user, passwd, pwhash, (Bool_t &)kTRUE);
      if (passwd == "") {
         if (fgPromptUser)
            user = PromptUser(fRemote);
         if (GetUserPasswd(user, passwd, pwhash, (Bool_t &)kTRUE))
            return kFALSE;
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
      if (fgSecAuthHook) {
         st = (*fgSecAuthHook) (this, user, passwd, fRemote, fDetails,
                                fVersion);
      } else {
         Error("Authenticate",
               "no support for SRP authentication available");
         return kFALSE;
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
            return kFALSE;
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
            return kFALSE;
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
      return kTRUE;
   } else {
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
         if (meth < nmet) {
            meth++;
            goto negotia;
         } else if (strlen(NoSupport) > 0)
            Info("Authenticate",
                 "attempted methods %s are not supported by remote server version",
                 NoSupport);
         return kFALSE;
      } else {
         if (fVersion < 2) {
            if (gDebug > 2)
               Info("Authenticate",
                    "negotiation not supported remotely: try next method, if any");
            if (meth < nmet) {
               meth++;
               goto negotia;
            } else if (strlen(NoSupport) > 0)
               Info("Authenticate",
                    "attempted methods %s are not supported by remote server version",
                    NoSupport);
            return kFALSE;
         }
         // Attempt negotiation ...
         fSocket->Recv(stat, kind);
         if (gDebug > 2)
            Info("Authenticate",
                 "after failed attempt: kind= %d, stat= %d", kind, stat);
         if (kind == kROOTD_ERR) {
            if (gDebug > 0)
               AuthError("Authenticate", stat);
            return kFALSE;
         } else if (kind == kROOTD_NEGOTIA) {
            if (stat > 0) {
               int len = 3 * stat;
               char *answer = new char[len];
               int nrec = fSocket->Recv(answer, len, kind);  // returns user
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
               if (strlen(NoSupport) > 0)
                  Info("Authenticate",
                       "attempted methods %s are not supported by remote server version",
                       NoSupport);
               return kFALSE;
            }
            // If no more local methods, exit
            if (remloc < 1) {
               if (strlen(NoSupport) > 0)
                  Info("Authenticate",
                       "attempted methods %s are not supported by remote server version",
                       NoSupport);
               return kFALSE;
            }
            // Look if a non tried method matches
            int i, j;
            char lav[40] = { 0 };
            for (i = 0; i < RemMeth; i++) {
               for (j = 0; j < nmet; j++) {
                  if (fHostAuth->GetMethods(j) == rMth[i] && tMth[j] == 0) {
                     meth = j;
                     goto negotia;
                  }
                  if (i == 0)
                     sprintf(lav, "%s %d", lav, fHostAuth->GetMethods(j));
               }
            }
            if (gDebug > 0)
               Warning("Authenticate",
                       "do not match with those locally available: %s",
                       lav);
            if (strlen(NoSupport) > 0)
               Info("Authenticate",
                    "attempted methods %s are not supported by remote server version",
                    NoSupport);
            return kFALSE;
         } else                 // unknown message code at this stage
         if (strlen(NoSupport) > 0)
            Info("Authenticate",
                 "attempted methods %s are not supported by remote server version",
                 NoSupport);
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
      int lDet = strlen(fDetails.Data()) + 2;
      char Pt[5] = { 0 }, Ru[5] = { 0 };
      char *Us = 0, *Cd = 0, *Cf = 0, *Kf = 0, *Ad = 0, *Cp = 0;
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

      // UnSet Crypt flag for UsrPwd, if requested
      if (fSecurity == kClear) {
         fgUsrPwdCrypt = kTRUE;
         if (!strncmp(Cp, "no", 2) || !strncmp(Cp, "0", 1))
            fgUsrPwdCrypt = kFALSE;
      }
      // Build UserDefaults
      if (fSecurity == kGlobus) {
         UsDef[0] = '\0';
         if (Cd != 0) {
            strcat(UsDef," ");
            strcat(UsDef,Cd);
            delete[] Cd;
         }
         if (Cf != 0) {
            strcat(UsDef," ");
            strcat(UsDef,Cf);
            delete[] Cf;
         }
         if (Kf != 0) {
            strcat(UsDef," ");
            strcat(UsDef,Kf);
            delete[] Kf;
         }
         if (Ad != 0) {
            strcat(UsDef," ");
            strcat(UsDef,Ad);
            delete[] Ad;
         }
      } else {
         if (fUser == "") {
            if (Us != 0) {
               sprintf(UsDef, "%s", Us);
               delete[] Us;
            }
         } else {
            if (fSecurity == kKrb5) {
               if (Us != 0) {
                  char *pat = strstr(Us, "@");
                  if (pat != 0)
                     sprintf(UsDef, "%s%s", fUser.Data(), pat);
                  else
                     sprintf(UsDef, "%s", fUser.Data());
               } else {
                  sprintf(UsDef, "%s", fUser.Data());
               }
            } else {
               sprintf(UsDef, "%s", fUser.Data());
            }
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
          fgUser != "")  // when set by user don't prompt for it anymore
         fgPromptUser = kFALSE;

      if (gDebug > 2)
         Info("SetEnvironment", "UsDef:%s", fgDefaultUser.Data());
   }
}

//______________________________________________________________________________
Bool_t TAuthenticate::GetUserPasswd(TString &user, TString &passwd,
                                    Bool_t &pwhash, Bool_t &srppwd)
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
                                 Bool_t &pwhash, Bool_t &srppwd)
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
const char *TAuthenticate::GetGlobalPasswd()
{
   // Static method returning the global global password.

   return fgPasswd;
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
const char *TAuthenticate::GetDefaultUser()
{
   // Static method returning the default user information.

   return fgDefaultUser;
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
      return StrDup("-1");
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
const char *TAuthenticate::GetRSAPubExport()
{
   // Static method returning the RSA public keys.

   return fgRSAPubExport.keys;
}

//______________________________________________________________________________
Int_t TAuthenticate::GetRSAInit()
{
   // Static method returning the RSA initialization flag.

   return fgRSAInit;
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
void TAuthenticate::AuthError(const char *where, Int_t err)
{
   // Print error string depending on error code.

   ::Error(where, gRootdErrStr[err]);
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
Int_t TAuthenticate::SshAuth(TString & User)
{
   // SSH client authentication code.

   // Check First if a 'ssh' executable exists ...
   char *gSshExe =
       gSystem->Which(gSystem->Getenv("PATH"), "ssh", kExecutePermission);

   if (!gSshExe) {
      if (gDebug > 2)
         Info("SshAuth", "ssh not found in $PATH");
      return -1;
   }

   if (gDebug > 2)
      Info("SshAuth", "ssh is %s", gSshExe);

   // Still allow for client definition of the ssh location ...
   if (strcmp(gEnv->GetValue("SSH.ExecDir", "-1"), "-1")) {
      if (gSshExe) delete[] gSshExe;
      gSshExe =
          StrDup(Form
                 ("%s/ssh", (char *) gEnv->GetValue("SSH.ExecDir", "")));
      if (gSystem->AccessPathName(gSshExe, kExecutePermission)) {
         Info("SshAuth", "%s not executable", gSshExe);
         if (gSshExe) delete[] gSshExe;
         return -1;
      }
   }
   // SSH-like authentication code.
   // Returns 0 in case authentication failed
   //         1 in case of success
   //        -1 in case of the remote node does not seem to support SSH-like Authentication
   //        -2 in case of the remote node does not seem to allow connections from this node

   char SecName[kMAXPATHLEN] = { 0 };

   // Determine user name ...
   User = GetSshUser();

   // Check ReUse
   Int_t ReUse = (int)fgAuthReUse;
   fDetails = TString(Form("pt:%d ru:%d us:",(int)fgPromptUser,(int)fgAuthReUse))
            + User;

   int retval, kind;

   // Create Options string
   char *Options = new char[strlen(User.Data()) + 40];
   int Opt = ReUse * kAUTH_REUSE_MSK;
   sprintf(Options, "%d none %ld %s", Opt, (Long_t)strlen(User.Data()),
           User.Data());

   // Check established authentications
   kind = kROOTD_SSH;
   retval = ReUse;
   Int_t rc = 0;
   if ((rc =
        AuthExists(this, (Int_t) TAuthenticate::kSSH, fDetails, Options,
                   &kind, &retval)) == 1) {
      // A valid authentication exists: we are done ...
      if (Options) delete[] Options;
      return 1;
   }
   if (Options) delete[] Options;
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

   // Wait for the server to communicate remote pid and location of command to execute
   char *CmdInfo = new char[retval + 1];
   fSocket->Recv(CmdInfo, retval + 1, kind);
   if (kind != kROOTD_SSH)
      return 0;                 // something went wrong
   if (gDebug > 3)
      Info("SshAuth", "received from server command info: %s", CmdInfo);

   int rport = -1;
   char *pp = 0;
   if ((pp = strstr(CmdInfo, "port")) != 0) {
      int clen = (int) (pp - CmdInfo);
      rport = atoi(pp + 5);
      CmdInfo[clen] = '\0';
      if (gDebug > 3)
         Info("SshAuth", "using port: %d, command info: %s", rport,
              CmdInfo);
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
   char sshcmd[kMAXPATHLEN] = { 0 };
   if (rport == -1) {
      // Remote server did not specify a specific port ... use our default, whatever it is ...
      sprintf(sshcmd, "%s -x -l %s %s %s %s", gSshExe, User.Data(),
              noPrompt.Data(), fRemote.Data(), CmdInfo);
   } else {
      // Remote server did specify a specific port ...
      sprintf(sshcmd, "%s -x -l %s -p %d %s %s %s", gSshExe, User.Data(),
              rport, noPrompt.Data(), fRemote.Data(), CmdInfo);
   }

   // Execute command
   int ssh_rc = gSystem->Exec(sshcmd);
   if (gDebug > 3)
      Info("SshAuth", "system return code: %d", ssh_rc);

   if (ssh_rc) {
      // This is bad ... needs to notify rootd by other means ...
      TSocket *newsock =
          new TSocket(fRemote.Data(), fSocket->GetInetAddress().GetPort(),
                      -1);
      newsock->SetOption(kNoDelay, 1);  // Set some socket options
      newsock->Send((Int_t) 0, (Int_t) 0);  // Tell rootd we do not want parallel connection
      char cd1[1024], pipe[1024];
      int id1, id2, id3;
      sscanf(CmdInfo, "%s %d %s %d %d", cd1, &id3, pipe, &id1, &id2);
      sprintf(SecName, "%d -1 0 %s %ld %s None", -gSystem->GetPid(), pipe,
              (Long_t)strlen(User), User.Data());
      newsock->Send(SecName, kROOTD_SSH);
      // Receive error message
      fSocket->Recv(retval, kind);  // for consistency
      if (kind == kROOTD_ERR) {
         if (gDebug > 0)
            AuthError("SshAuth", retval);
      }
      SafeDelete(newsock);

      return 0;
   }

   // Receive key request info and type of key (if ok, error otherwise)
   int nrec = fSocket->Recv(retval, kind);  // returns user
   if (gDebug > 3)
      Info("SshAuth", "got message %d, flag: %d", kind, retval);

   // Check if an error occured
   if (kind == kROOTD_ERR) {
      if (gDebug > 0)
         AuthError("SshAuth", retval);
      return 0;
   }

   if (ReUse == 1) {

      // Save type of key
      if (kind != kROOTD_RSAKEY)
         Warning("SshAuth",
                 "problems recvn RSA key flag: got message %d, flag: %d",
                 kind, fRSAKey);

      fRSAKey = 1;

      // Send the key securely
      SendRSAPublicKey(fSocket);

      // Receive username used for login
      nrec = fSocket->Recv(retval, kind);  // returns user
      if (gDebug > 3)
         Info("SshAuth", "got message %d, flag: %d", kind, retval);
   }

   if (kind != kROOTD_SSH || retval < 1)
      Warning("SshAuth",
              "problems recvn (user,offset) length (%d:%d bytes:%d)", kind,
              retval, nrec);

   char *answer = new char[retval + 1];
   nrec = fSocket->Recv(answer, retval + 1, kind);  // returns user
   if (kind != kMESS_STRING)
      Warning("SshAuth", "username and offset not received (%d:%d)", kind,
              nrec);

   // Parse answer
   char *lUser = new char[retval];
   int OffSet = -1;
   sscanf(answer, "%s %d", lUser, &OffSet);
   if (gDebug > 3)
      Info("SshAuth", "received from server: user: %s, offset: %d", lUser,
           OffSet);

   // Receive Token
   char *Token = 0;
   if (ReUse == 1 && OffSet > -1) {
      if (SecureRecv(fSocket, 1, &Token) == -1) {
         Warning("SshAuth",
                 "problems secure-receiving token - may result in corrupted token");
      }
      if (gDebug > 3)
         Info("SshAuth", "received from server: token: '%s' ", Token);
   } else {
      Token = StrDup("");
   }

   // Create and save AuthDetails object
   SaveAuthDetails(this, (Int_t) TAuthenticate::kSSH, OffSet, ReUse,
                   fDetails, lUser, fRSAKey, Token);

   // Release allocated memory ...
   if (answer) delete[] answer;
   if (lUser) delete[] lUser;
   if (Token) delete[] Token;

   // Get and Analyse the reply
   fSocket->Recv(retval, kind);
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
const char *TAuthenticate::GetSshUser() const
{
   // Method returning the User to be used for the ssh login.
   // Looks first at SSH.Login and finally at env USER.
   // If SSH.LoginPrompt is set to 'yes' it prompts for the 'login name'

   static TString user = "";

   if (fgPromptUser) {
      user = PromptUser(fRemote);
   } else {
      user = fgDefaultUser;
      if (user == "")
         user = PromptUser(fRemote);
   }

   return user;
}

//______________________________________________________________________________
Int_t TAuthenticate::GetAuthMeth(const char *Host, const char *Proto,
                                 char ***User, Int_t **NumMeth,
                                 Int_t **AuthMeth, char ***Details)
{
   // This method looks for the available methods (as chosen by the user)
   // for authentication vis-a-vis of host 'Host' and depending on protocol
   // Proto (either root - rootd, roots, rootk - or proof - proofd, proofs,
   // proofk - families).
   // Information is looked for in ~/.rootauthrc and in the .rootrc family
   // of files via Rootd.Authentication and Proofd.Authentication variables.
   // Return number of methods, their codes in AuthMeth and strings with
   // auth details in Details (login name, principals, etc ...).
   // Space for AuthMeth and Details must be allocated outside
   // Default method is SSH.

   Int_t i;

   if (gDebug > 2)
      ::Info("GetAuthMeth", "enter: h:%s p:%s u:%s (0x%lx 0x%lx) ",
              Host, Proto, *User[0], (Long_t) (*User), (Long_t) (*User[0]));

   if (*User[0] == 0)
      *User[0] = StrDup("");

   // Check then .rootauthrc (if there)
   char temp[kMAXPATHLEN];
   Int_t *am[kMAXSEC], *nh, nu = 0, j = 0;
   char **det[kMAXSEC];
   if ((nu = CheckRootAuthrc(Host, User, &nh, am, det)) > 0) {
      if (gDebug > 3)
         ::Info("GetAuthMeth", "found %d users - nh: %d 0x%lx %s ", nu,
                nh[0], (Long_t) (*User[0]), *User[0]);
      *NumMeth = new int[nu];
      for (i = 0; i < kMAXSEC; i++) {
         AuthMeth[i] = new int[nu];
         Details[i] = new char *[nu];
         for (j = 0; j < nu; j++) {
            (*NumMeth)[j] = nh[j];
            Details[i][j] = 0;
            AuthMeth[i][j] = -1;
            if (i < nh[j]) {
               AuthMeth[i][j] = am[i][j];
               if (det[i][j] == 0 || strlen(det[i][j]) == 0) {
                  strcpy(temp, "");
                  // Set user
                  char *usr = StrDup((*User)[j]);
                  // Check env variables for more details, if any ...
                  Details[i][j] = GetDefaultDetails(am[i][j], 1, usr);
                  if (usr) delete[] usr;
               } else {
                  Details[i][j] = det[i][j];
               }
            }
         }
      }
      if (gDebug > 3) {
         ::Info("GetAuthMeth", "found %d users", nu);
         for (j = 0; j < nu; j++) {
            ::Info("GetAuthMeth", "returning %d ", nh[j]);
            for (i = 0; i < nh[j]; i++) {
               ::Info("GetAuthMeth", "method[%d]: %d - details[%d]: %s ",
                      i, AuthMeth[i][j], i, Details[i][j]);
            }
         }
      }
      return nu;
   }
   // Check globals in .rootrc family of files ...
   char auth[40] = { 0 };
   if (strstr(Proto, "proof") != 0) {
      sprintf(auth, "%s", gEnv->GetValue("Proofd.Authentication", "4"));
   } else if (strstr(Proto, "root") != 0) {
      sprintf(auth, "%s", gEnv->GetValue("Rootd.Authentication", "4"));
   }
   if (gDebug > 3)
      ::Info("GetAuthMeth",
             "Found method(s) '%s' from globals in .rootrc (User[0]=%s)",
              auth,(*User)[0]);
   nh = new int;
   nh[0] = 0;
   for (i = 0; i < kMAXSEC; i++) {
      am[i] = new int[1];
   }
   if (strlen(auth) > 0) {
      nh[0] =
          sscanf(auth, "%d %d %d %d %d %d", &am[0][0], &am[1][0],
                 &am[2][0], &am[3][0], &am[4][0], &am[5][0]);
   }
   if (nh[0] > 0) {
      nu = 1;
      *NumMeth = new int[1];
      *NumMeth[0] = nh[0];
      for (i = 0; i < kMAXSEC; i++) {
         AuthMeth[i] = new int[1];
         Details[i] = new char *[nh[0]];
         if (i < nh[0]) {
            AuthMeth[i][0] = am[i][0];
            strcpy(temp, "");
            Details[i][0] = GetDefaultDetails(am[i][0], 1, (*User)[0]);
         } else {
            AuthMeth[i][0] = -1;
            Details[i][0] = 0;
         }
      }
   }
   if (gDebug > 2) {
      ::Info("GetAuthMeth", "for user '%s' returning %d methods",
             (*User)[0], nh[0]);
      for (i = 0; i < nh[0]; i++) {
         ::Info("GetAuthMeth", "   method[%d]: %d - details[%d]: %s ", i,
                AuthMeth[i][0], i, Details[i][0]);
      }
   }
   return nu;
}

//______________________________________________________________________________
Int_t TAuthenticate::CheckRootAuthrc(const char *Host, char ***user,
                                     Int_t ** nh, Int_t ** am, char ***det)
{
   // Try to get info about authetication policies for Host

   int Nuser = 0, CheckUser = 0;
   int nmeth = 0, found = 0;
   Bool_t retval = kFALSE;
   char *net, *UserRq = 0;

   if (gSystem->Getenv("ROOTAUTHRC") != 0) {
      net = (char *) gSystem->Getenv("ROOTAUTHRC");
   } else {
      net = gSystem->ConcatFileName(gSystem->HomeDirectory(), ".rootauthrc");
   }
   if (gDebug > 2)
      ::Info("CheckRootAuthrc", "enter: host:%s user:%s file:%s", Host,
             (*user)[0], net);

   // Check if file can be read ...
   if (gSystem->AccessPathName(net, kReadPermission)) {
      if (gDebug > 1)
         ::Info("TAuthenticate::CheckRootAuthrc",
                "file %s cannot be read (errno: %d)", net, errno);
      return 0;
   }
   // Variables for scan ...
   char line[kMAXPATHLEN], rest[kMAXPATHLEN];

   // Generate temporary file name and open it
   int expand = 1;
   TString filetmp = "rootauthrc";
   FILE *ftmp = gSystem->TempFileName(filetmp);

   if (gDebug > 2)
      ::Info("TAuthenticate::CheckRootAuthrc", "got tmp file: %s open at 0x%lx",
              filetmp.Data(), (Long_t)ftmp);
   if (ftmp == 0)
      expand = 0;  // Problems opening temporary file: ignore 'include' directives ...

   FILE *fd = 0;
   // If the temporary file is open, copy everything to the new file ...
   if (expand == 1) {

      TAuthenticate::FileExpand(net, ftmp);
      fd = ftmp;
      rewind(fd);

   } else {
      // Open file
      fd = fopen(net, "r");
      if (fd == 0) {
         if (gDebug > 2)
            ::Info("TAuthenticate::CheckRootAuthrc",
                   "file %s cannot be opened (errno: %d)",
                    net, errno);
         return 0;
      }
   }
   // If empty user you need first to count how many entries are to be read
   char host[kMAXPATHLEN], info[kMAXPATHLEN];
   if ((*user)[0] == 0 || strlen((*user)[0]) == 0) {
      char usrtmp[256];
      unsigned int tlen = kMAXPATHLEN;
      char *Temp = new char[tlen];
      Temp[0] = '\0';
      while (fgets(line, sizeof(line), fd) != 0) {
         // Skip comment lines
         if (line[0] == '#')
            continue;
         if (!strncmp(line,"proofserv",9))
            continue;
         char *pstr = 0;
         char *pdef = strstr(line, "default");
         sscanf(line,"%s %s",host,info);
         if (CheckHost(Host, host) || (pdef != 0 && pdef == line)) {

            unsigned int hlen = strlen(Host);
            if (strstr(Temp, Host) == 0) {
               if ((strlen(Temp) + hlen + 2) > tlen) {
                  char *NewTemp = StrDup(Temp);
                  if (Temp) delete[] Temp;
                  tlen += kMAXPATHLEN;
                  Temp = new char[tlen];
                  strcpy(Temp, NewTemp);
                  if (NewTemp) delete[] NewTemp;
               }
               sprintf(Temp, "%s %s", Temp, Host);
               Nuser++;
            } else {
               pstr = strstr(line, "user");
               if (pstr != 0) {
                  sscanf(pstr + 4, "%s %s", usrtmp, rest);
                  if (strstr(Temp, usrtmp) == 0) {
                     if ((strlen(Temp) + strlen(usrtmp) + 2) > tlen) {
                        char *NewTemp = StrDup(Temp);
                        if (Temp) delete[] Temp;
                        tlen += kMAXPATHLEN;
                        Temp = new char[tlen];
                        strcpy(Temp, NewTemp);
                        if (NewTemp) delete[] NewTemp;
                     }
                     sprintf(Temp, "%s %s", Temp, usrtmp);
                     Nuser++;
                  }
               }
            }
         }
      }
      if (Temp) delete[] Temp;
      if (gDebug > 3)
         ::Info("CheckRootAuthrc",
                "found %d different entries for host %s", Nuser, Host);

      if (Nuser) {
         if ((*user)[0]) delete[] (*user)[0];
         (*user) = new char *[Nuser];
         *nh = new int[Nuser];
         int i;
         for (i = 0; i < kMAXSEC; i++) {
            am[i] = new int[Nuser];
            det[i] = new char *[Nuser];
         }
         UserRq = StrDup("-1");
      } else {
         UserRq = StrDup("none");
         if ((*user)[0]) delete[] (*user)[0];
         (*user)[0] = StrDup("-1");
      }
   } else {
      CheckUser = 1;
      UserRq = StrDup((*user)[0]);
      *nh = new int[1];
      int i;
      for (i = 0; i < kMAXSEC; i++) {
         am[i] = new int[1];
         det[i] = new char *[1];
      }
   }

   // If nothing found return ...
   if (!strncmp(UserRq,"none",4)) {
      fclose(fd);
      if (net) delete[] net;
      if (UserRq) delete[] UserRq;
      if (expand == 1)
         gSystem->Unlink(filetmp);
      return 0;
   } else
     // rewind the file and start scanning it ...
      rewind(fd);

   // Scan it ...
   char opt[20];
   int mth[6] = { 0 }, meth;
   int ju = 0, nu = 0;
   while (fgets(line, sizeof(line), fd) != 0) {
      int i;
      // Skip comment lines
      if (line[0] == '#')
         continue;

      // Get rid of end of line '\n', if there ...
      if (line[strlen(line) - 1] == '\n')
         line[strlen(line) - 1] = '\0';

      // scan line
      int nw = sscanf(line, "%s %s %s", host, opt, rest);

      // no useful info provided for this host
      if (nw < 2)
         continue;

      // Notify
      if (gDebug > 3)
         ::Info("CheckRootAuthrc", "found line ... %s ", line);

      // The list of data servers for proof is analyzed elsewhere (TProof ...)
      if (!strcmp(host, "proofserv"))
         continue;

      if (strcmp(host, "default")) {
         if (!CheckHost(Host, host))
            continue;
      } else {
         // This is a default entry: ignore it if a host-specific entry was already
         // found, analyse it otherwise ...
         if (found == 1)
            continue;
      }

      // Make sure that 'rest' contains all the rest ...
      strcpy(rest, strstr(line, opt) + strlen(opt) + 1);

      // Is there a user specified?
      if (!strcmp(opt, "user")) {
         // Yes: check if already entered ...
         char *usr = new char[strlen(rest) + 5];
         sscanf(rest, "%s %s", usr, rest);
         if (gDebug > 3)
            ::Info("CheckRootAuthrc",
                   "found 'user': %s requested: \"%s\" (%d,%d) (rest:%s)",
                   usr, UserRq, CheckUser, nu, rest);

         if (CheckUser == 1) {
            if (strcmp(UserRq, usr))
               continue;
         }
         int newu = 1;
         if (nu > 0) {
            int j;
            for (j = 0; j < nu; j++) {
               if (!strcmp((*user)[j], usr)) {
                  ju = j;
                  newu = 0;
               }
            }
         }
         if (newu == 1) {
            if (CheckUser == 0) {
               ju = nu++;
            } else {
               ju = 0;
               nu = 1;
               if ((*user)[ju]) delete[] (*user)[ju];
            }
            (*user)[ju] = StrDup(usr);
            (*nh)[ju] = 0;
            for (i = 0; i < kMAXSEC; i++) {
               am[i][ju] = -1;
               det[i][ju] = 0;
            }
         }
         if (usr) delete[] usr;
         // Now reposition opt and rest ...
         sscanf(rest, "%s %s", opt, rest);
         strcpy(rest, strstr(line, opt) + strlen(opt) + 1);
      } else {
         // No: record an anonymous entry ...
         int newu = 1;
         if (nu > 0) {
            int j;
            for (j = 0; j < nu; j++) {
               if (!strcmp((*user)[j], "-1")) {
                  ju = j;
                  newu = 0;
               }
            }
         }
         if (newu == 1) {
            if (CheckUser == 0) {
               ju = nu++;
            } else {
               ju = 0;
               nu = 1;
               if ((*user)[ju]) delete[] (*user)[ju];
            }
            (*user)[ju] = StrDup("-1");
            (*nh)[ju] = 0;
            for (i = 0; i < kMAXSEC; i++) {
               am[i][ju] = -1;
               det[i][ju] = 0;
            }
         }
      }

      // get rid of opt keyword
      strcpy(rest, strstr(line, opt) + strlen(opt) + 1);
      if (!strcmp(opt, "list")) {
         if (gDebug > 3)
            ::Info("CheckRootAuthrc", "found 'list': rest:%s", rest);
         char cmth[kMAXSEC][20];
         int nw =
             sscanf(rest, "%s %s %s %s %s %s", cmth[0], cmth[1], cmth[2],
                    cmth[3], cmth[4], cmth[5]);
         nmeth = 0;
         for (i = 0; i < nw; i++) {
            int met = -1;
            if (strlen(cmth[i]) > 1) {
               // Method passed as string: translate it to number
               met = GetAuthMethodIdx(cmth[i]);
               if (met == -1 && gDebug > 2)
                  ::Info("CheckRootAuthrc",
                         "unrecognized method (%s): ", cmth[i]);
            } else {
               met = atoi(cmth[i]);
            }
            if (met >= 0 && met < kMAXSEC) {
               mth[nmeth] = met;
               nmeth++;
            } else if (gDebug > 2)
               ::Info("CheckRootAuthrc", "unrecognized method (%d): ",
                      met);
         }
         int *tmp_am = 0;
         char **tmp_det = 0;
         if ((*nh)[ju] > 0) {
            tmp_am = new int[(*nh)[ju]];
            tmp_det = new char *[(*nh)[ju]];
            for (i = 0; i < (*nh)[ju]; i++) {
               tmp_am[i] = am[i][ju];
               tmp_det[i] = StrDup(det[i][i]);
            }
         }
         int k, j = nmeth;
         for (i = 0; i < kMAXSEC; i++) {
            if (i < nmeth) {
               am[i][ju] = mth[i];
               for (k = 0; k < (*nh)[ju]; k++) {
                  if (tmp_am[k] == mth[i]) {
                     det[i][ju] = StrDup(tmp_det[k]);
                     if (tmp_det[k]) delete[] tmp_det[k];
                     tmp_det[k] = 0;
                  }
               }
            } else {
               k = 0;
               while (k < (*nh)[ju] && am[i][ju] == -1) {
                  if (tmp_det[k] != 0) {
                     j++;
                     am[i][ju] = mth[k];
                     det[i][ju] = StrDup(tmp_det[k]);
                     delete[] tmp_det[k];
                     tmp_det[k] = 0;
                  }
                  k++;
               }
            }
         }
         if (tmp_am) delete[] tmp_am;
         if (tmp_det) delete[] tmp_det;
         (*nh)[ju] = nmeth;
      } else {

         if (!strcmp(opt, "method")) {
            if (gDebug > 3)
               ::Info("CheckRootAuthrc", "found 'method': rest:%s", rest);

            //         nw= sscanf(rest,"%d %s",&meth,info);
            char cmeth[20];
            nw = sscanf(rest, "%s %s", cmeth, info);
            meth = -1;
            if (strlen(cmeth) > 1) {
               // Method passed as string: translate it to number
               meth = GetAuthMethodIdx(cmeth);
               if (meth == -1 && gDebug > 2)
                  ::Info("CheckRootAuthrc",
                         "unrecognized method (%s): ", cmeth);
            } else {
               meth = atoi(cmeth);
            }
            if (meth < 0 || meth > (kMAXSEC - 1)) {
               if (gDebug > 2)
                  ::Info("CheckRootAuthrc", "unrecognized method (%d): ",
                         meth);
               continue;
            }

            if (nw > 1) {
               char *pinfo = strstr(rest, info);
               strncpy(info, pinfo, strlen(rest));
            } else
               info[0] = '\0';

            if (found == 0) {
               for (i = 0; i < kMAXSEC; i++) {
                  am[i][ju] = -1;
               }
            }
            int j = -1;
            for (i = 0; i < (*nh)[ju]; i++) {
               if (am[i][ju] == meth) {
                  j = i;
                  if (strlen(info) > 0)
                     det[i][ju] = StrDup(info);
               }
            }
            if (j == -1) {
               if ((*nh)[ju] < kMAXSEC) {
                  am[(*nh)[ju]][ju] = meth;
                  if (strlen(info) > 0)
                     det[(*nh)[ju]][ju] = StrDup(info);
                  ((*nh)[ju])++;
               } else {
                  char *ostr = new char[4 * kMAXSEC];
                  ostr[4 * kMAXSEC] = '\0';
                  for (i = 0; i < kMAXSEC; i++) {
                     sprintf(ostr, "%s %d", ostr, am[i][ju]);
                  }
                  if (gDebug > 0)
                     ::Warning("CheckRootAuthrc",
                               "output am badly filled: %s", ostr);
                  if (ostr) delete[] ostr;
               }
            }
            if (gDebug > 3)
               ::Info("CheckRootAuthrc",
                      "method with info : %s has been recorded (ju:%d, nh:%d)",
                      info, ju, (*nh)[ju]);
         } else {               // Unknown option
            ::Warning("CheckRootAuthrc", "found unknown option: %s", opt);
            continue;
         }
      }
      // Found new entry matching: superseed previous result
      found = 1;
      retval = kTRUE;
   }

   if (CheckUser == 1 && nu == 1) {
      if (!strcmp((*user)[0], "-1")) {
         if ((*user)[0]) delete[] (*user)[0];
         (*user)[0] = StrDup(UserRq);
      }
   }

   if (gDebug > 3) {
      rewind(fd);
      ::Info("CheckRootAuthrc",
             "+----------- temporary file: %s -------------------------+",
             filetmp.Data());
      ::Info("CheckRootAuthrc", "+");
      while (fgets(line, sizeof(line), fd) != 0) {
         if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1] = '\0';
         ::Info("CheckRootAuthrc", "+ %s", line);
      }
      ::Info("CheckRootAuthrc", "+");
      ::Info("CheckRootAuthrc",
             "+---------------------------------------------------------------------+");
      ::Info("CheckRootAuthrc",
             "+----------- Valid info fetched for this call ------------------------+");
      ::Info("CheckRootAuthrc", "+");
      ::Info("CheckRootAuthrc", "+   Host: %s - Number of users found: %d",
             Host, nu);
      for (ju = 0; ju < nu; ju++) {
         ::Info("CheckRootAuthrc", "+");
         ::Info("CheckRootAuthrc",
                "+      Dumping user: %s ( %d methods found)", (*user)[ju],
                (*nh)[ju]);
         int i;
         for (i = 0; i < (*nh)[ju]; i++) {
            ::Info("CheckRootAuthrc", "+        %d: method: %d  det:'%s'",
                   i, am[i][ju], det[i][ju]);
         }
      }
      ::Info("CheckRootAuthrc", "+");
      ::Info("CheckRootAuthrc",
             "+---------------------------------------------------------------------+");
   }

   fclose(fd);
   if (net) delete[] net;
   if (UserRq) delete[] UserRq;
   if (expand == 1)
      gSystem->Unlink(filetmp);

   return nu;
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
         ::Info("CheckHost", "checking host IP: %s", theHost.Data());
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
Int_t TAuthenticate::RfioAuth(TString & User)
{
   // UidGid client authentication code.
   // Returns 0 in case authentication failed
   //         1 in case of success
   //        <0 in case of system error

   if (gDebug > 2)
      Info("RfioAuth", "enter ... User %s", User.Data());

   // Get user info ... ...
   UserGroup_t *pw = gSystem->GetUserInfo();
   if (pw) {

      // These are the details to be saved in case of success ...
      User = pw->fUser;
      fDetails = TString("pt:0 ru:0 us:") + User;

      // Check that we are not root ...
      if (strcmp(pw->fUser, "root")) {

         // Get group ID associated with the current process ...
         Int_t uid = pw->fUid;
         Int_t gid = pw->fGid;

         // Send request ....
         char *sstr = new char[40];
         sprintf(sstr, "%d %d", uid, gid);
         if (gDebug > 3)
            Info("RfioAuth", "sending ... %s", sstr);
         int ns = fSocket->Send(sstr, kROOTD_RFIO);
         if (gDebug > 3)
            Info("RfioAuth", "sent ... %d bytes (expected > %d)", ns,
                 strlen(sstr));

         // Get answer
         Int_t stat, kind;
         fSocket->Recv(stat, kind);
         if (gDebug > 3)
            Info("RfioAuth", "after kROOTD_RFIO: kind= %d, stat= %d", kind,
                 stat);

         // Query result ...
         if (kind == kROOTD_AUTH && stat >= 1) {
            // Authentication OK ...
            SaveAuthDetails(this, (Int_t) TAuthenticate::kRfio, -1, 0,
                            fDetails, pw->fUser, 0, 0);
            return 1;
         } else {
            TString Server = "sockd";
            if (fProtocol.Contains("root"))
               Server = "rootd";
            if (fProtocol.Contains("proof"))
               Server = "proofd";

            // Authentication failed
            if (stat == kErrConnectionRefused) {
               Error("RfioAuth",
                     "%s@%s does not accept connections from %s%s",
                     Server.Data(),fRemote.Data(),
                     fUser.Data(),gSystem->HostName());
               return -2;
            } else if (stat == kErrNotAllowed) {
               Error("RfioAuth",
                     "%s@%s does not accept %s authentication from %s@%s",
                     Server.Data(),fRemote.Data(),
                     TAuthenticate::fgAuthMeth[5].Data(),
                     fUser.Data(),gSystem->HostName());
            } else {
              if (gDebug > 0)
                 AuthError("RfioAuth", stat);
            }
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
Int_t TAuthenticate::ClearAuth(TString & User, TString & Passwd, Bool_t & PwHash)
{
   // UsrPwd client authentication code.
   // Returns 0 in case authentication failed
   //         1 in case of success

   if (gDebug > 2)
      Info("ClearAuth", "enter: User: %s (passwd hashed?: %d)", User.Data(),(Int_t)PwHash);

   Int_t ReUse    = fgAuthReUse;
   Int_t Prompt   = fgPromptUser;
   Int_t Crypt    = fgUsrPwdCrypt;
   Int_t NeedSalt = 1;
   if (PwHash)
     NeedSalt = 0;
   fDetails = TString(Form("pt:%d ru:%d cp:%d us:",
                           fgPromptUser, fgAuthReUse, fgUsrPwdCrypt)) + User;
   if (gDebug > 2)
      Info("ClearAuth", "ru:%d pt:%d cp:%d ns:%d",
           fgAuthReUse,fgPromptUser,fgUsrPwdCrypt,NeedSalt);
#ifdef R__WIN32
   Crypt = 0;
#endif


   Int_t stat, kind;

   if (fVersion > 1) {

      // New protocol

      Int_t Anon = 0;
      Int_t OffSet = -1;
      char *Salt = 0;
      char *PasHash = 0;

      // Create Options string
      char *Options = new char[strlen(User.Data()) + 40];
      int Opt = (ReUse * kAUTH_REUSE_MSK) + (Crypt * kAUTH_CRYPT_MSK) +
                (NeedSalt * kAUTH_SSALT_MSK);
      sprintf(Options, "%d %ld %s", Opt, (Long_t)strlen(User), User.Data());

      // Check established authentications
      kind = kROOTD_USER;
      stat = ReUse;
      Int_t rc = 0;
      if ((rc =
           AuthExists(this, (Int_t) TAuthenticate::kClear, fDetails,
                      Options, &kind, &stat)) == 1) {
         // A valid authentication exists: we are done ...
         if (Options) delete[] Options;
         if (gDebug > 3)
            Info("ClearAuth", "valid authentication exists: return 1");
         return 1;
      }
      if (Options) delete[] Options;
      if (rc == -2) {
         return rc;
      }
      if (stat == kErrNotAllowed && kind == kROOTD_ERR) {
         return 0;
      }

      if (kind == kROOTD_AUTH && stat == -1) {
         if (gDebug > 3)
            Info("ClearAuth", "anonymous user", kind, stat);
         Anon  = 1;
         Crypt = 0;
         ReUse = 0;
      }

      if (OffSet == -1 && Anon == 0 && Crypt == 1) {

         // Check that we got the right thing ..
         if (kind != kROOTD_RSAKEY) {
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
         fRSAKey = 1;

         // Send the key securely
         SendRSAPublicKey(fSocket);

         if (NeedSalt) {
            // Receive password salt
            if (SecureRecv(fSocket, 1, &Salt) == -1) {
               Warning("ClearAuth",
                       "problems secure-receiving salt - may result in corrupted salt");
               Warning("ClearAuth", "switch off reuse for this session");
               Crypt = 0;
            }
            if (gDebug > 2)
               Info("ClearAuth", "got salt: '%s'", Salt);
         } else {
            if (gDebug > 2)
               Info("ClearAuth", "Salt not required");
            fSocket->Recv(stat, kind);
            if (kind != kMESS_ANY || stat != 0) {
               Warning("ClearAuth",
                  "Potential problems: got msg type: %d value: %d (expecting: %d 0)",
                   kind,stat,(Int_t)kMESS_ANY);
            }
         }
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

         if (Prompt == 1 || PasHash == 0) {

            if (Passwd == "") {
               Passwd =
                 PromptPasswd(Form("%s@%s password: ",User.Data(),fRemote.Data()));
               if (Passwd == "") {
                  Error("ClearAuth", "password not set");
                  if (PasHash) delete[] PasHash;
                  if (Salt) delete[] Salt;
                  fSocket->Send("-1", kROOTD_PASS);  // Needs this for consistency
                  return 0;
               }
            }
            if (Crypt == 1) {
               // Get hash (only if not already hashed ...)
               //if (strncmp(Passwd,Salt,strlen(Salt))) {
               if (!PwHash) {
#ifndef R__WIN32
                  PasHash = StrDup(crypt(Passwd, Salt));
#endif
               } else {
                  PasHash = StrDup(Passwd);
               }
            }
         }

      }

      // Send it to server
      if (Anon == 0 && Crypt == 1) {

         // Store for later use
         fgPasswd = PasHash;
         fPasswd = PasHash;
         fgPwHash = kTRUE;
         fPwHash = kTRUE;
         fSRPPwd = kFALSE;
         fgSRPPwd = kFALSE;

         fSocket->Send("\0", kROOTD_PASS);  // Needs this for consistency
         if (SecureSend(fSocket, 1, PasHash) == -1) {
            Warning("ClearAuth",
                    "problems secure-sending pass hash - may result in authentication failure");
         }
      } else {

         // Store for later use
         fgPasswd = Passwd;
         fPasswd = Passwd;
         fgPwHash = kFALSE;
         fPwHash = kFALSE;
         fSRPPwd = kFALSE;
         fgSRPPwd = kFALSE;

         // Standard technique: invert passwd
         if (Passwd != "") {
            for (int i = 0; i < Passwd.Length(); i++) {
               char inv = ~Passwd(i);
               Passwd.Replace(i, 1, inv);
            }
         }
         fSocket->Send(Passwd.Data(), kROOTD_PASS);
      }

      // Receive username used for login
      int nrec = fSocket->Recv(stat, kind);  // returns user
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
      if (OffSet == -1) {

         if (kind != kROOTD_PASS || stat < 1)
            Warning("ClearAuth",
                    "problems recvn (user,offset) length (%d:%d bytes:%d)",
                    kind, stat, nrec);

         // Get user and offset
         char *answer = new char[stat + 1];
         nrec = fSocket->Recv(answer, stat + 1, kind);
         if (kind != kMESS_STRING)
            Warning("ClearAuth",
                    "username and offset not received (%d:%d)", kind,
                    nrec);

         // Parse answer
         char *lUser = new char[stat];
         sscanf(answer, "%s %d", lUser, &OffSet);
         if (gDebug > 3)
            Info("ClearAuth",
                 "received from server: user: %s, offset: %d (%s)", lUser,
                 OffSet, answer);

         // Return username
         User = lUser;

         char *Token = 0;
         if (ReUse == 1) {
            // Receive Token
            if (Crypt == 1) {
               if (SecureRecv(fSocket, 1, &Token) == -1) {
                  Warning("ClearAuth",
                          "problems secure-receiving token - may result in corrupted token");
               }
            } else {
               Int_t Tlen = 9;
               Token = new char[Tlen];
               fSocket->Recv(Token, Tlen, kind);
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
         // Create and save AuthDetails object
         SaveAuthDetails(this, (Int_t) TAuthenticate::kClear, OffSet,
                         ReUse, fDetails, lUser, fRSAKey, Token);

         // This from remote login
         fSocket->Recv(stat, kind);

         if (answer) delete[] answer;
         if (lUser) delete[] lUser;
      }
      // Release allocated memory ...
      if (Salt) delete[] Salt;
      if (PasHash) delete[] PasHash;


      if (kind == kROOTD_AUTH && stat >= 1) {
         if (stat == 2) {
            int newOffSet;
            // Receive new offset ...
            fSocket->Recv(newOffSet, kind);
            // ... and save it
            SetOffSet(fHostAuth, (Int_t) TAuthenticate::kClear, fDetails,
                      newOffSet);
         }
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
      fSocket->Send(User.Data(), kROOTD_USER);

      // Get replay from server
      fSocket->Recv(stat, kind);
      if (kind == kROOTD_ERR) {
         TString Server = "sockd";
         if (fProtocol.Contains("root"))
            Server = "rootd";
         if (fProtocol.Contains("proof"))
            Server = "proofd";
         if (stat == kErrConnectionRefused) {
            Error("ClearAuth",
                  "%s@%s does not accept connections from %s@%s",
                  Server.Data(),fRemote.Data(),
                  fUser.Data(),gSystem->HostName());
            return -2;
         } else if (stat == kErrNotAllowed) {
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
      fSocket->Send(Passwd, kROOTD_PASS);

      // Get result of attempt
      fSocket->Recv(stat, kind);  // returns user
      if (gDebug > 3)
         Info("ClearAuth", "after kROOTD_PASS: kind= %d, stat= %d", kind,
              stat);

      if (kind == kROOTD_AUTH && stat == 1) {
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
Int_t TAuthenticate::GetOffSet(TAuthenticate * Auth, Int_t Method,
                               TString & Details, char **Token)
{
   // Check if already authenticated for Method with Details
   // Return OffSet in the affirmative case or -1.

   Int_t OffSet = -1;

   if (gDebug > 2)
      ::Info("GetOffSet", "analyzing: Method:%d, Details:%s", Method,
             Details.Data());

   THostAuth *HostAuth = Auth->GetHostAuth();
   int Nw = 0, i = 0;
   char Pt[5] = { 0 }, Ru[5] = {
   0}, *Wd[4] = {
   0};

   if (Method == TAuthenticate::kGlobus) {
      DecodeDetailsGlobus((char *) Details.Data(), Pt, Ru, &Wd[0], &Wd[1],
                          &Wd[2], &Wd[3]);
      Nw = 0;
      for (i = 0; i < 4; i++) {
         if (Wd[i] != 0 && strlen(Wd[i]) > 0)
            Nw++;
      }
   } else {
      DecodeDetails((char *) Details.Data(), Pt, Ru, &Wd[0]);
      Nw = 0;
      if (Wd[0] != 0 && strlen(Wd[0]) > 0)
         Nw++;
   }
   if (gDebug > 3)
      ::Info("GetOffSet", "found Nw: %d, Wd: %s %s %s %s", Nw, Wd[0],
             Wd[1], Wd[2], Wd[3]);
   if (Nw == 0) {
      if (gDebug > 3)
         ::Info("GetOffSet", "nothing to compare: return");
      return OffSet;
   }
   // Check we already authenticated
   TIter next(HostAuth->Established());
   TAuthDetails *ai;
   while ((ai = (TAuthDetails *) next())) {
      if (gDebug > 3)
         ::Info("GetOffSet", "found entry: met:%d det:%s off:%d",
                ai->GetMethod(), ai->GetDetails(), ai->GetOffSet());
      if (ai->GetMethod() == Method) {
         int match = 1;
         for (i = 0; i < Nw; i++) {
            if (strstr(ai->GetDetails(), Wd[i]) == 0)
               match = 0;
         }
         if (match == 1) {
            OffSet = ai->GetOffSet();
            //strcpy(*Token,ai->GetToken());
            if (OffSet > -1) {
               *Token = StrDup(ai->GetToken());
               Auth->SetRSAKey(ai->GetRSAKey());
            }
         }
      }
   }
   if (gDebug > 2)
      ::Info("GetOffSet", "returning: %d", OffSet);
   for (i = 0; i < Nw; i++) {
      if (Wd[i] != 0)
         delete[] Wd[i];
   }
   if (*Token == 0) {
      Auth->SetRSAKey(0);
   }
   return OffSet;
}

//______________________________________________________________________________
void TAuthenticate::SetOffSet(THostAuth * HostAuth, Int_t Method,
                              TString & Details, Int_t OffSet)
{
   // Save new offset

   TIter next(HostAuth->Established());
   TAuthDetails *ai;
   while ((ai = (TAuthDetails *) next())) {
      if (ai->GetMethod() == Method) {
         if (strstr(ai->GetDetails(), Details) != 0)
            ai->SetOffSet(OffSet);
      }
   }
}

//______________________________________________________________________________
char *TAuthenticate::GetRemoteLogin(THostAuth * HostAuth, Int_t Method,
                                    const char *Details)
{
   // Check if already authenticated for Method with Details
   // Return remote user login name in the affirmative case or 0
   // The string should be freed by the caller with 'delete'.

   char *rlogin = 0;

   int Nw = 0, i = 0;
   char Pt[5] = { 0 }, Ru[5] = {
   0}, *Wd[4] = {
   0};

   if (Method == TAuthenticate::kGlobus) {
      DecodeDetailsGlobus((char *) Details, Pt, Ru, &Wd[0], &Wd[1], &Wd[2],
                          &Wd[3]);
      Nw = 0;
      for (i = 0; i < 4; i++) {
         if (Wd[i] != 0 && strlen(Wd[i]) > 0)
            Nw++;
      }
   } else {
      DecodeDetails((char *) Details, Pt, Ru, &Wd[0]);
      Nw = 0;
      if (Wd[0] != 0 && strlen(Wd[0]) > 0)
         Nw++;
   }
   if (gDebug > 2)
      ::Info("GetRemoteLogin", "details:%s", Details);

   // Check we already authenticated
   TIter next(HostAuth->Established());
   TAuthDetails *ai;
   while ((ai = (TAuthDetails *) next())) {
      if (ai->GetMethod() == Method) {
         int match = 1;
         for (i = 0; i < Nw; i++) {
            if (strstr(ai->GetDetails(), Wd[i]) == 0)
               match = 0;
         }
         if (match == 1) {
            rlogin = StrDup(ai->GetLogin());
         }
      }
   }
   if (gDebug > 2)
      ::Info("GetRemoteLogin", "returning: %s", rlogin);

   for (i = 0; i < Nw; i++) {
      if (Wd[i] != 0)
         delete[] Wd[i];
   }
   return rlogin;
}

//______________________________________________________________________________
void TAuthenticate::SaveAuthDetails(TAuthenticate * Auth, Int_t Method,
                                    Int_t OffSet, Int_t ReUse,
                                    TString & Details, const char *rlogin,
                                    Int_t key, const char *token)
{
   THostAuth *HostAuth = Auth->GetHostAuth();
   TSocket *Socket = Auth->GetSocket();
   const char *Protocol = Auth->GetProtocol();

   const char *remote = HostAuth->GetHost();
   int port = Socket->GetPort();
   int service = 1;
   if (strstr(Protocol, "proof") != 0)
      service = 2;

   // If no meaningful offset is passed, then check if a
   // similar entry already exist; if so, we do not need
   // to create duplicates ...
   if (OffSet == -1) {

      TIter next(HostAuth->Established());
      TAuthDetails *ai;
      while ((ai = (TAuthDetails *) next())) {
         if (ai->GetMethod() == Method) {
            if (strstr(ai->GetDetails(), Details) != 0) {
               if (ai->GetOffSet() == -1)
                  return;
            }
         }
      }
   }

   // Create AuthDetails object
   TAuthDetails *fAuthDetails =
       new TAuthDetails(Form("%s:%d:%d", remote, port, service),
                        Method, OffSet, ReUse, (char *) Details.Data(),
                        token, key, rlogin);
   // Add it to the list
   HostAuth->Established()->Add(fAuthDetails);

   return;
}

//______________________________________________________________________________
void TAuthenticate::DecodeDetails(char *details, char *Pt, char *Ru,
                                  char **Us)
{
   // Parse details looking for user info

   if (gDebug > 2)
      ::Info("DecodeDetails", "analyzing ... %s", details);

   *Us = 0;

   if (Pt == 0 || Ru == 0) {
      ::Error("DecodeDetails",
              "memory for Pt and Ru must be allocated elsewhere (Pt:0x%lx, Ru:0x%lx)",
              (Long_t) Pt, (Long_t) Ru);
      return;
   }

   if (strlen(details) > 0) {
      int lDet = strlen(details) + 2;
      char *ptr, *Temp = new char[lDet];
      if ((ptr = strstr(details, "pt:")) != 0)
         sscanf(ptr + 3, "%s %s", Pt, Temp);
      if ((ptr = strstr(details, "ru:")) != 0)
         sscanf(ptr + 3, "%s %s", Ru, Temp);
      if ((ptr = strstr(details, "us:")) != 0) {
         *Us = new char[lDet];
         *Us[0] = '\0';
         sscanf(ptr + 3, "%s %s", *Us, Temp);
      }
      if (gDebug > 3)
         ::Info("DecodeDetails", "Pt:%s, Ru:%s, Us:%s", Pt, Ru, *Us);
      if (Temp) delete[] Temp;
   }
}

//______________________________________________________________________________
void TAuthenticate::DecodeDetailsGlobus(char *details, char *Pt, char *Ru,
                                        char **Cd, char **Cf, char **Kf,
                                        char **Ad)
{
   // Parse details looking for globus authentication info

   if (gDebug > 2)
      ::Info("DecodeDetailsGlobus", "analyzing ... %s", details);

   *Cd = 0, *Cf = 0, *Kf = 0, *Ad = 0;

   if (Pt == 0 || Ru == 0) {
      ::Error("DecodeDetailsGlobus",
              "memory for Pt and Ru must be allocated elsewhere (Pt:0x%lx, Ru:0x%lx)",
              (Long_t) Pt, (Long_t) Ru);
      return;
   }

   if (strlen(details) > 0) {
      int lDet = strlen(details) + 2;
      char *ptr, *Temp = new char[lDet];
      if ((ptr = strstr(details, "pt:")) != 0)
         sscanf(ptr + 3, "%s %s", Pt, Temp);
      if ((ptr = strstr(details, "ru:")) != 0)
         sscanf(ptr + 3, "%s %s", Ru, Temp);
      if ((ptr = strstr(details, "cd:")) != 0) {
         *Cd = new char[lDet];
         *Cd[0] = '\0';
         sscanf(ptr, "%s %s", *Cd, Temp);
      }
      if ((ptr = strstr(details, "cf:")) != 0) {
         *Cf = new char[lDet];
         *Cf[0] = '\0';
         sscanf(ptr, "%s %s", *Cf, Temp);
      }
      if ((ptr = strstr(details, "kf:")) != 0) {
         *Kf = new char[lDet];
         *Kf[0] = '\0';
         sscanf(ptr, "%s %s", *Kf, Temp);
      }
      if ((ptr = strstr(details, "ad:")) != 0) {
         *Ad = new char[lDet];
         *Ad[0] = '\0';
         sscanf(ptr, "%s %s", *Ad, Temp);
      }

      if (gDebug > 3)
         ::Info("DecodeDetailsGlobus", "Pt:%s, Ru:%s, %s, %s, %s, %s", Pt,
                Ru, *Cd, *Cf, *Kf, *Ad);

      if (Temp) delete[] Temp;
   }
}

//______________________________________________________________________________
void TAuthenticate::SetHostAuth(const char *host, const char *user)
{
   // Sets fUser=user and search fgAuthInfo for the entry pertaining to
   // (host,user), setting fHostAuth accordingly.
   // If no entry is found fHostAuth is not changed

   if (gDebug > 2)
      Info("SetHostAuth", "enter ... %s ... %s", host, user);

   // Set fUser
   SetUser(user);

   // Check list of auth info for already loaded info about this host
   TIter next(GetAuthInfo());
   THostAuth *ai;
   while ((ai = (THostAuth *) next())) {
      if (gDebug > 3)
         ai->Print("Authenticate:SetHostAuth");
      if (!strcmp(host, ai->GetHost()) && !strcmp(user, ai->GetUser())) {
         fHostAuth = ai;
         break;
      }
   }
}

//______________________________________________________________________________
THostAuth *TAuthenticate::GetHostAuth(const char *host, const char *user)
{
   // Sets fUser=user and search fgAuthInfo for the entry pertaining to
   // (host,user), setting fHostAuth accordingly.
   // If no entry is found fHostAuth is not changed

   if (gDebug > 2)
      ::Info("GetHostAuth", "enter ... %s ... %s", host, user);
   int ulen = strlen(user);
   THostAuth *rHA = 0;

   // Check and save the host FQDN ...
   TString lHost = host;
   TInetAddress addr = gSystem->GetHostByName(lHost);
   if (addr.IsValid()) {
      lHost = addr.GetHostName();
      if (lHost == "UnNamedHost")
         lHost = addr.GetHostAddress();
   }

   // Check list of auth info for already loaded info about this host
   TIter next(GetAuthInfo());
   THostAuth *ai;
   Bool_t NotFound = kTRUE;
   while ((ai = (THostAuth *) next())) {
      if (gDebug > 3)
         ai->Print("Authenticate:GetHostAuth");

      // Use default entry if existing and nothing more specific is found
      if (!strcmp(ai->GetHost(),"default") && NotFound)
         rHA = ai;

      // Check
      if (ulen > 0) {
         if (CheckHost(lHost,ai->GetHost()) && !strcmp(user, ai->GetUser())) {
            rHA = ai;
            NotFound = kFALSE;
         }
      } else {
         if (CheckHost(lHost,ai->GetHost())) {
            rHA = ai;
            NotFound = kFALSE;
         }
      }

      if (ulen > 0) {
         if (lHost == ai->GetHost() && !strcmp(user, ai->GetUser())) {
            rHA = ai;
            break;
         }
      } else {
         if (lHost == ai->GetHost()) {
            rHA = ai;
            break;
         }
      }
   }
   return rHA;
}

//______________________________________________________________________________
void TAuthenticate::FileExpand(const char *fexp, FILE * ftmp)
{
   // Expands include directives found in fexp files
   // The expanded, temporary file, is pointed to by 'ftmp'
   // and should be already open. To be called recursively.

   FILE *fin;
   char line[kMAXPATHLEN];
   char cinc[20], fileinc[kMAXPATHLEN];

   if (gDebug > 2)
     ::Info("FileExpand", "enter ... '%s' ... 0x%lx", fexp, (Long_t)ftmp);

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
         ::Info("FileExpand", "read line ... '%s'", line);
      int nw = sscanf(line, "%s %s", cinc, fileinc);
      if (nw < 2)
         continue;              // Not enough info in this line
      if (strcmp(cinc, "include") != 0) {
         // copy line in temporary file
         fprintf(ftmp, "%s\n", line);
      } else {
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
            ::Warning("FileExpand",
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
      ::Info("GetDefaultDetails", "enter ... %d ...pt:%d ... '%s'", sec,
             opt, usr);

   if (opt < 0 || opt > 1)
      opt = 1;

   // UsrPwd
   if (sec == TAuthenticate::kClear) {
      if (strlen(usr) == 0 || !strcmp(usr,"-1"))
         usr = gEnv->GetValue("UsrPwd.Login", "");
      sprintf(temp, "pt:%s ru:%s cp:%s us:%s",
              gEnv->GetValue("UsrPwd.LoginPrompt", copt[opt]),
              gEnv->GetValue("UsrPwd.ReUse", "1"),
              gEnv->GetValue("UsrPwd.Crypt", "1"), usr);

      // SRP
   } else if (sec == TAuthenticate::kSRP) {
      if (strlen(usr) == 0 || !strcmp(usr,"-1"))
         usr = gEnv->GetValue("SRP.Login", "");
      sprintf(temp, "pt:%s ru:%s us:%s",
              gEnv->GetValue("SRP.LoginPrompt", copt[opt]),
              gEnv->GetValue("SRP.ReUse", "0"), usr);

      // Kerberos
   } else if (sec == TAuthenticate::kKrb5) {
      if (strlen(usr) == 0 || !strcmp(usr,"-1"))
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
      if (strlen(usr) == 0 || !strcmp(usr,"-1"))
         usr = gEnv->GetValue("SSH.Login", "");
      sprintf(temp, "pt:%s ru:%s us:%s",
              gEnv->GetValue("SSH.LoginPrompt", copt[opt]),
              gEnv->GetValue("SSH.ReUse", "1"), usr);

      // Uid/Gid
   } else if (sec == TAuthenticate::kRfio) {
      if (strlen(usr) == 0 || !strcmp(usr,"-1"))
         usr = gEnv->GetValue("UidGid.Login", "");
      sprintf(temp, "pt:%s us:%s",
              gEnv->GetValue("UidGid.LoginPrompt", copt[opt]), usr);
   }
   if (gDebug > 2)
      ::Info("GetDefaultDetails", "returning ... %s", temp);

   return StrDup(temp);
}

//______________________________________________________________________________
void TAuthenticate::RemoveHostAuth(THostAuth * ha)
{
   // Remove THostAuth instance from the list

   // Remove from the list ...
   GetAuthInfo()->Remove(ha);

   // ... destroy it
   delete ha;
}

//______________________________________________________________________________
void TAuthenticate::ReadAuthRc(const char *host, const char *user)
{
   // Read methods for a given host (and user) from .rootauthrc

   if (gDebug > 2)
      ::Info("ReadAuthRc", "enter: host: '%s', user: '%s'", host, user);

   // Check and save the host FQDN ...
   TString fqdn;
   TInetAddress addr = gSystem->GetHostByName(host);
   if (addr.IsValid()) {
      fqdn = addr.GetHostName();
      if (fqdn == "UnNamedHost")
         fqdn = addr.GetHostAddress();
   }
   if (gDebug > 3)
      ::Info("ReadAuthRc",
             "number of HostAuth Instantiations in memory: %d",
             GetAuthInfo()->GetSize());

   // Determine applicable auth methods from client choices
   Int_t *nmeth, *security[kMAXSEC];
   char **details[kMAXSEC];
   char **usr = new char *[1];
   if (strlen(user) > 0)
      usr[0] = StrDup(user);
   else
      usr[0] = StrDup("");

   int nu =
       GetAuthMeth(fqdn.Data(), "rootd", &usr, &nmeth, security, details);
   if (gDebug > 3)
      ::Info("ReadAuthRc", "found %d users", nu);

   int ju = 0, i, j;
   for (ju = 0; ju < nu; ju++) {

      // Translate to Input for THostAuth
      int nm = nmeth[ju], am[kMAXSEC];
      char *det[kMAXSEC];
      for (i = 0; i < kMAXSEC; i++) {
         if (i < nm) {
            am[i] = security[i][ju];
            det[i] = StrDup(details[i][ju]);
         } else {
            am[i] = -1;
            det[i] = 0;
         }
      }
      if (gDebug > 3) {
         ::Info("ReadAuthRc", "got %d methods (ju: %d)", nm, ju);
         for (i = 0; i < nm; i++) {
            ::Info("ReadAuthRc", "got (%d,0) security:%d details:%s", i,
                   am[i], det[i]);
         }
      }
      // Check list of auth info for already loaded info about this (host,user)
      THostAuth *hostAuth = 0;
      TIter next(GetAuthInfo());
      THostAuth *ai;
      while ((ai = (THostAuth *) next())) {
         if (gDebug > 3)
            ai->Print();
         if (fqdn == ai->GetHost() && !strcmp(usr[ju], ai->GetUser())) {
            hostAuth = ai;
            break;
         }
      }

      if (hostAuth == 0) {
         // Create THostAuth object
         hostAuth = new THostAuth(fqdn, usr[ju], nm, am, det);
         // ... and add it to the list
         GetAuthInfo()->Add(hostAuth);
      } else {
         // Modify existing entry ...
         Bool_t *RemoveMethod = new Bool_t[hostAuth->NumMethods()];
         for (i = 0; i < hostAuth->NumMethods(); i++)
            RemoveMethod[i] = kTRUE;
         for (i = 0; i < nmeth[0]; i++) {
            int j, jm = -1;
            for (j = 0; j < hostAuth->NumMethods(); j++) {
               if (am[i] == hostAuth->GetMethods(j)) {
                  hostAuth->SetDetails(am[i], det[i]);
                  jm = j;
                  RemoveMethod[j] = kFALSE;
               }
            }
            if (jm == -1)
               hostAuth->AddMethod(am[i], det[i]);
         }
         for (i = 0; i < hostAuth->NumMethods(); i++)
            if (RemoveMethod[i])
               hostAuth->RemoveMethod(hostAuth->GetMethods(i));
      }
      if (gDebug > 3)
         hostAuth->Print();
      for (i = 0; i < nm; i++) {
         if (det[i] != 0)
            delete[] det[i];
      }
   }
   for (i = 0; i < kMAXSEC; i++) {
      if (security[i] != 0)
         delete[] security[i];
      if (details[i] != 0) {
         for (j = 0; j < nu; j++) {
            if (details[i][j] != 0)
               delete[] details[i][j];
         }
      }
   }
   for (i = 0; i < nu; i++) {
      if (usr[i]) delete[] usr[i];
   }
   if (nmeth) delete[] nmeth;
   if (usr) delete[] usr;
}

//______________________________________________________________________________
void TAuthenticate::PrintHostAuth()
{
   // Print info abour existing HostAuth instantiations

   TIter next(GetAuthInfo());
   THostAuth *ai;
   while ((ai = (THostAuth *) next())) {
      ai->Print();
      ai->PrintEstablished();
   }
}

//______________________________________________________________________________
Int_t TAuthenticate::AuthExists(TAuthenticate *Auth, Int_t Sec,
                                TString &Details, const char *Options,
                                Int_t *Message, Int_t *Rflag)
{
   // Check if we have a valid established sec context in memory
   // Retrieves relevant info and negotiates with server.
   // Options = "Opt,strlen(User),User.Data()"
   // Message = kROOTD_USER, ...

   THostAuth *HostAuth = Auth->GetHostAuth();
   TSocket *Socket = Auth->GetSocket();

   if (gDebug > 2)
      ::Info("AuthExists", "%d: enter: msg: %d options: '%s'", Sec,
             *Message, Options);


   // Check we already authenticated
   Int_t OffSet = -1;
   char *Token = 0;
   //   char   *eTkn = 0;
   Int_t ReUse = *Rflag;
   //   Int_t  Len   = 0;
   if (ReUse == 1) {

      // Get OffSet and token, if any ...
      OffSet = GetOffSet(Auth, Sec, Details, &Token);
      if (gDebug > 3)
         ::Info("AuthExists", "%d: offset in memory is: %d ('%s')", Sec,
                OffSet, Token);
   }
   // Prepare string to be sent to the server
   char *sstr = new char[strlen(Options) + 20];
   sprintf(sstr, "%d %d %s", gSystem->GetPid(), OffSet, Options);

   // Send Message
   Socket->Send(sstr, *Message);

   if (ReUse == 1 && OffSet > -1) {

      Int_t RSAKey = Auth->GetRSAKey();
      if (gDebug > 2)
         ::Info("AuthExists", "key type: %d", RSAKey);

      if (RSAKey > 0) {
         // Send Token encrypted
         if (SecureSend(Socket, 1, Token) == -1) {
            ::Warning("AuthExists",
                      "problems secure-sending Token - may trigger problems in proofing Id ");
         }
      } else {
         // Send Token inverted
         for (int i = 0; i < (int) strlen(Token); i++) {
            Token[i] = ~Token[i];
         }
         Socket->Send(Token, kMESS_STRING);
      }

   }
   // Release allocated memory
   if (Token) delete[] Token;
   if (sstr) delete[] sstr;

   Int_t stat, kind;
   Socket->Recv(stat, kind);
   if (gDebug > 3)
      ::Info("AuthExists", "%d: after msg %d: kind= %d, stat= %d", Sec,
             *Message, kind, stat);

   // Return flags
   *Message = kind;
   *Rflag = stat;

   if (kind == kROOTD_ERR) {
      TString Server = "sockd";
      if (strstr(Auth->GetProtocol(),"root"))
         Server = "rootd";
      if (strstr(Auth->GetProtocol(),"proof"))
         Server = "proofd";
      if (stat == kErrConnectionRefused) {
         ::Error("AuthExists",
                 "%s@%s does not accept connections from %s@%s",
                 Server.Data(),Auth->GetRemoteHost(),
                 Auth->GetUser(),gSystem->HostName());
         return -2;
      } else if (stat == kErrNotAllowed) {
         if (gDebug > 0)
            ::Info("AuthExists",
                   "%s@%s does not accept %s authentication from %s@%s",
                   Server.Data(),Auth->GetRemoteHost(),
                   TAuthenticate::fgAuthMeth[Sec].Data(),
                   Auth->GetUser(),gSystem->HostName());
      } else {
        if (gDebug > 0)
           AuthError("AuthExists", stat);
      }
      return 0;
   }

   if (kind == kROOTD_AUTH && stat >= 1) {
      if (stat == 2) {
         int newOffSet;
         // Receive new offset ...
         Socket->Recv(newOffSet, kind);
         // ... and save it
         SetOffSet(HostAuth, Sec, Details, newOffSet);
      }
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TAuthenticate::GenRSAKeys()
{
   // Generate a valid pair of private/public RSA keys to protect for authentication
   // token exchange

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
   Int_t seed = 1;
   if (!gSystem->AccessPathName("/dev/random", kReadPermission)) {
      if (gDebug > 2)
         Info("GenRSAKeys", "taking seed from /dev/random");
      char brnd[4];
      FILE *frnd = fopen("/dev/random","r");
      fread(brnd,1,4,frnd);
      seed = *((int *)brnd);
      fclose(frnd);
   } else {
      if (gDebug > 2)
         Info("GenRSAKeys", "/dev/random not available: using time()");
      seed = time(0);
   }
   srand(seed);

   // Sometimes some bunch is not decrypted correctly
   // That's why we make retries to make sure that encryption/decryption works as expected
   Bool_t NotOk = 1;
   rsa_NUMBER p1, p2, rsa_n, rsa_e, rsa_d;
   Int_t l_n = 0, l_e = 0, l_d = 0;
   char buf_n[rsa_STRLEN], buf_e[rsa_STRLEN], buf_d[rsa_STRLEN];
#if 0
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
#if 0
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

#if 0
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

#if 0
   if (gDebug > 2) {
      // Determine their lengths
      Info("GenRSAKeys", "local: generated keys are:");
      Info("GenRSAKeys", "local: n: '%s' length: %d", buf_n, l_n);
      Info("GenRSAKeys", "local: e: '%s' length: %d", buf_e, l_e);
      Info("GenRSAKeys", "local: d: '%s' length: %d", buf_d, l_d);
   }
#endif
   // Export form
   fgRSAPubExport.len = l_n + l_d + 4;
   fgRSAPubExport.keys = new char[fgRSAPubExport.len];

   fgRSAPubExport.keys[0] = '#';
   memcpy(fgRSAPubExport.keys + 1, buf_n, l_n);
   fgRSAPubExport.keys[l_n + 1] = '#';
   memcpy(fgRSAPubExport.keys + l_n + 2, buf_d, l_d);
   fgRSAPubExport.keys[l_n + l_d + 2] = '#';
   fgRSAPubExport.keys[l_n + l_d + 3] = 0;
#if 0
   if (gDebug > 2)
      Info("GenRSAKeys", "local: export pub: '%s'", fgRSAPubExport.keys);
#else
   if (gDebug > 2)
      Info("GenRSAKeys", "local: export pub length: %d bytes", fgRSAPubExport.len);
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

   // Get current time as seed for rand().
   time_t curtime;
   time(&curtime);
   int seed = (int) curtime;

   // feed seed
   if (seed)
      srand(seed);

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
Int_t TAuthenticate::SecureSend(TSocket *Socket, Int_t Key, const char *Str)
{
   // Encode null terminated Str using the session private key indcated by Key
   // and sends it over the network
   // Returns number of bytes sent, or -1 in case of error.
   // Key = 1 for private encoding, Key = 2 for public encoding

   char BufTmp[kMAXSECBUF];
   char BufLen[20];

   if (gDebug > 2)
      ::Info("SecureSend", "local: enter ... (key: %d)", Key);

   Int_t sLen = strlen(Str) + 1;
   Int_t Ttmp = 0;
   Int_t Nsen = -1;

   if (Key == 1) {
      strncpy(BufTmp, Str, sLen);
      BufTmp[sLen] = 0;
      Ttmp =
          rsa_fun::fg_rsa_encode(BufTmp, sLen, fgRSAPriKey.n,
                                 fgRSAPriKey.e);
      sprintf(BufLen, "%d", Ttmp);
      Socket->Send(BufLen, kROOTD_ENCRYPT);
      Nsen = Socket->SendRaw(BufTmp, Ttmp);
      if (gDebug > 3)
         ::Info("SecureSend",
                "local: sent %d bytes (expected: %d)", Nsen,Ttmp);
   } else if (Key == 2) {
      strncpy(BufTmp, Str, sLen);
      BufTmp[sLen] = 0;
      Ttmp =
          rsa_fun::fg_rsa_encode(BufTmp, sLen, fgRSAPubKey.n,
                                 fgRSAPubKey.e);
      sprintf(BufLen, "%d", Ttmp);
      Socket->Send(BufLen, kROOTD_ENCRYPT);
      Nsen = Socket->SendRaw(BufTmp, Ttmp);
      if (gDebug > 3)
         ::Info("SecureSend",
                "local: sent %d bytes (expected: %d)", Nsen,Ttmp);
   } else {
      ::Info("SecureSend", "unknown key option (%d) - return", Key);
   }
   return Nsen;
}

//______________________________________________________________________________
Int_t TAuthenticate::SecureRecv(TSocket *Socket, Int_t Key, char **Str)
{
   // Receive Len bytes from Socket and decode them in Str using key indicated by Key type
   // Return number of received bytes or -1 in case of error.
   // Key = 1 for private decoding, Key = 2 for public decoding

   char BufTmp[kMAXSECBUF];
   char BufLen[20];

   Int_t Nrec = -1;
   // We must get a pointer ...
   if (!Str)
      return Nrec;

   Int_t kind;
   Socket->Recv(BufLen, 20, kind);
   Int_t Len = atoi(BufLen);
   if (gDebug > 3)
      ::Info("SecureRecv", "got len '%s' %d (msg kind: %d)", BufLen, Len,
             kind);
   if (!strncmp(BufLen, "-1", 2))
      return Nrec;

   // Now proceed
   if (Key == 1) {
      Nrec = Socket->RecvRaw(BufTmp, Len);
      rsa_fun::fg_rsa_decode(BufTmp, Len, fgRSAPriKey.n, fgRSAPriKey.e);
      if (gDebug > 3)
         ::Info("SecureRecv", "local: decoded string is %d bytes long ", strlen(BufTmp));
   } else if (Key == 2) {
      Nrec = Socket->RecvRaw(BufTmp, Len);
      rsa_fun::fg_rsa_decode(BufTmp, Len, fgRSAPubKey.n, fgRSAPubKey.e);
      if (gDebug > 3)
         ::Info("SecureRecv", "local: decoded string is %d bytes long ", strlen(BufTmp));
   } else {
      ::Info("SecureRecv", "unknown key option (%d) - return", Key);
   }

   *Str = new char[strlen(BufTmp) + 1];
   strcpy(*Str, BufTmp);

   return Nrec;
}

//______________________________________________________________________________
void TAuthenticate::DecodeRSAPublic(const char *RSAPubExport, rsa_NUMBER &RSA_n, rsa_NUMBER &RSA_d)
{
   // Store RSA public keys from export string RSAPubExport.

   if (!RSAPubExport)
      return;

   if (gDebug > 2)
      ::Info("DecodeRSAPublic","enter: string length: %d bytes", strlen(RSAPubExport));

   char Str[kMAXPATHLEN] = { 0 };
   strcpy(Str, RSAPubExport);

   if (strlen(Str) > 0) {
      // The format is #<hex_n>#<hex_d>#
      char *pd1 = strstr(Str, "#");
      char *pd2 = strstr(pd1 + 1, "#");
      char *pd3 = strstr(pd2 + 1, "#");
      if (pd1 && pd2 && pd3) {
         // Get <hex_n> ...
         int l1 = (int) (pd2 - pd1 - 1);
         char *RSA_n_exp = new char[l1 + 1];
         strncpy(RSA_n_exp, pd1 + 1, l1);
         RSA_n_exp[l1] = 0;
         if (gDebug > 2)
            ::Info("DecodeRSAPublic","got %d bytes for RSA_n_exp", strlen(RSA_n_exp));
         // Now <hex_d>
         int l2 = (int) (pd3 - pd2 - 1);
         char *RSA_d_exp = new char[l2 + 1];
         strncpy(RSA_d_exp, pd2 + 1, l2);
         RSA_d_exp[l2] = 0;
         if (gDebug > 2)
            ::Info("DecodeRSAPublic","got %d bytes for RSA_d_exp", strlen(RSA_d_exp));

         rsa_fun::fg_rsa_num_sget(&RSA_n, RSA_n_exp);
         rsa_fun::fg_rsa_num_sget(&RSA_d, RSA_d_exp);

         if (RSA_n_exp)
            if (RSA_n_exp) delete[] RSA_n_exp;
         if (RSA_d_exp)
            if (RSA_d_exp) delete[] RSA_d_exp;

      } else
         ::Info("DecodeRSAPublic","bad format for input string");
   }
}

//______________________________________________________________________________
void TAuthenticate::SetRSAPublic(const char *RSAPubExport)
{
   // Store RSA public keys from export string RSAPubExport.

   if (gDebug > 2)
      ::Info("SetRSAPublic","enter: string length %d bytes", strlen(RSAPubExport));

   if (!RSAPubExport)
      return;

   // Decode input string
   rsa_NUMBER RSA_n, RSA_d;
   TAuthenticate::DecodeRSAPublic(RSAPubExport,RSA_n,RSA_d);

   // Save Public key
   rsa_fun::fg_rsa_assign(&fgRSAPubKey.n, &RSA_n);
   rsa_fun::fg_rsa_assign(&fgRSAPubKey.e, &RSA_d);
}

//______________________________________________________________________________
void TAuthenticate::SendRSAPublicKey(TSocket *Socket)
{
   // Receives Server RSA Public key
   // Sends local RSA public key encodded

   // Receive server public key
   char ServerPubKey[kMAXSECBUF];
   int kind;
   Socket->Recv(ServerPubKey, kMAXSECBUF, kind);
   if (gDebug > 3)
      ::Info("SendRSAPublicKey", "received key from server %d bytes",
            strlen(ServerPubKey));

   // Decode it
   rsa_NUMBER RSA_n, RSA_d;
   TAuthenticate::DecodeRSAPublic(ServerPubKey,RSA_n,RSA_d);

   // Send local public key, encodes
   char BufTmp[kMAXSECBUF];
   Int_t sLen = fgRSAPubExport.len;
   strncpy(BufTmp,fgRSAPubExport.keys,sLen);
   BufTmp[sLen] = 0;
   Int_t Ttmp =
        rsa_fun::fg_rsa_encode(BufTmp, sLen, RSA_n, RSA_d);

   // Send length first
   char BufLen[20];
   sprintf(BufLen, "%d", Ttmp);
   Socket->Send(BufLen, kROOTD_ENCRYPT);
   // Send Key. second ...
   Int_t Nsen = Socket->SendRaw(BufTmp, Ttmp);
   if (gDebug > 3)
         ::Info("SendRSAPublicKey",
                "local: sent %d bytes (expected: %d)", Nsen,Ttmp);
}
