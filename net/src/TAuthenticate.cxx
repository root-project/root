// @(#)root/net:$Name:  $:$Id: TAuthenticate.cxx,v 1.12 2003/09/02 15:10:17 rdm Exp $
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
#include "TApplication.h"
#include "TEnv.h"
#include "TList.h"
#include "NetErrors.h"

#ifndef R__LYNXOS
#include <sys/stat.h>
#endif
#include <errno.h>
#include <sys/types.h>
#include <time.h>
#ifndef R__WIN32
#include <crypt.h>
#endif

#if defined(__osf__) || defined(__sgi)
extern "C" char *crypt(const char *, const char *);
#endif


Int_t TAuthenticate::fgRSAInit = 0;
rsa_KEY TAuthenticate::fgRSAPriKey;
rsa_KEY TAuthenticate::fgRSAPubKey;
rsa_KEY_export TAuthenticate::fgRSAPubExport = { 0, 0 };

TString TAuthenticate::fgUser;
TString TAuthenticate::fgPasswd;
SecureAuth_t TAuthenticate::fgSecAuthHook;
Krb5Auth_t TAuthenticate::fgKrb5AuthHook;
GlobusAuth_t TAuthenticate::fgGlobusAuthHook;

TList *TAuthenticate::fgAuthInfo = 0;


// For fast name-to-number translation for authentication methods
static const char kMethods[] = "usrpwd srp    krb5   globus ssh    uidgid";

ClassImp(TAuthenticate)

//______________________________________________________________________________
TAuthenticate::TAuthenticate(TSocket *sock, const char *remote,
                             const char *proto, const char *user)
{
   // Create authentication object.

   fSocket   = sock;
   fRemote   = remote;
   fHostAuth = 0;
   fVersion  = 2;                // The latest, by default
   fRSAKey   = 0;


   if (gDebug > 2)
      Info("TAuthenticate", "Enter: local host:%s: user is: %s (proto:%s)",
           gSystem->Getenv("HOST"), user, proto);

   // Set protocol string.
   // Check if version should be different ...
   char *pdd;
   if (proto && strlen(proto) > 0) {
      char *sproto = StrDup(proto);
      if ((pdd = strstr(sproto, ":")) != 0) {
         int rproto = atoi(pdd + 1);
         int lproto = (int) (pdd - sproto);
         sproto[lproto] = '\0';
         if (strstr(sproto, "root") != 0) {
            if (rproto < 8) {
               fVersion = 1;
               if (rproto < 6)
                  fVersion = 0;
            }
         }
         if (strstr(sproto, "proof") != 0) {
            if (rproto < 7)
               fVersion = 1;
         }
         if (gDebug > 3)
            Info("TAuthenticate",
                 "service: %s (remote protocol: %d): fVersion: %d", sproto,
                 rproto, fVersion);

         fProtocol = sproto;
      }
      SafeDelete(sproto);
   }

   // Check or get user name
   if (user && strlen(user) > 0) {
      fUser = user;
   } else {
      UserGroup_t *u = gSystem->GetUserInfo();
      if (u)
         fUser = u->fUser;
      delete u;
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
   TIter next(GetAuthInfo());
   THostAuth *ai;
   while ((ai = (THostAuth *) next())) {
      if (gDebug > 4) {
         ai->Print();
         ai->PrintEstablished();
      }
      if (fqdn == ai->GetHost() && fUser == ai->GetUser()) {
         fHostAuth = ai;
         break;
      }
   }

   // If we did not find a good THostAuth instantiation, create one
   if (fHostAuth == 0) {
      // Determine applicable auth methods from client choices
      Int_t *nmeth;                //not cleaned up? (rdm)
      Int_t *security[kMAXSEC];
      char **details[kMAXSEC];
      char **usr = new char *[1];  //not cleaned up? (rdm)
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
      if (gDebug > 4) {
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
      if (gDebug > 4)
         fHostAuth->Print();
      for (i = 0; i < nmeth[0]; i++) {   // what if nu > 0? (rdm)
         SafeDelete(security[i]);
         SafeDelete(details[i][0]);
         SafeDelete(det[i]);
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

   Int_t RemMeth = 0, rMth[kMAXSEC];
   Int_t meth = 0;
   char NoSupport[80] = { 0 };

   TString user, passwd;

   Int_t ntry = 0;
   if (gDebug > 2)
      Info("Authenticate", "enter: fUser: %s", fUser.Data());
   NoSupport[0] = 0;

 negotia:

   if (gDebug > 2) {
      ntry++;
      Info("Authenticate", "try #: %d", ntry);
   }

   user = "";
   passwd = "";

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

      // Clear Authentication
      if (!strcmp(gSystem->Getenv("PROMPTUSER"), "0")
          || !strcmp(gSystem->Getenv("PROMPTUSER"), "no")) {
         user = gSystem->Getenv("DEFAULTUSER");
      }
      if (GetUserPasswd(user, passwd))
         return kFALSE;
      if (fUser != "root")
         st = ClearAuth(user, passwd);

   } else if (fSecurity == kSRP) {

      // SRP Authentication
      if (!strcmp(gSystem->Getenv("PROMPTUSER"), "0")) {
         user = gSystem->Getenv("DEFAULTUSER");
      }
      if (GetUserPasswd(user, passwd))
         return kFALSE;
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

         // Rfio Authentication
         st = RfioAuth(fUser);

      } else {
         if (gDebug > 0)
            Info("Authenticate",
                 "remote daemon does not support Rfio authentication");
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
      if (gDebug > 2)
         Info("Authenticate",
              "got st=%d: still %d methods locally available", st,
              nmet - meth - 1);
      if ((nmet - meth - 1) < 1) {
         if (strlen(NoSupport) > 0)
            Info("Authenticate",
                 "attempted methods %s are not supported by remote server version",
                 NoSupport);
         return kFALSE;
      }
      if (st == -1) {
         if (gDebug > 2)
            Info("Authenticate",
                 "method not even started: insufficient or wrong info: try with next method, if any");
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
               if (gDebug > 0)
                  Info("Authenticate",
                       "remotely allowed methods still to be tried: %s",
                       answer);
            } else if (stat == 0) {
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
               for (j = meth + 1; j < nmet; j++) {
                  if (fHostAuth->GetMethods(j) == rMth[i]) {
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
   // Set environment variables relevant for the authentication process
   // PROMPTUSER, AUTHREUSE and DEFAULTUSER.
   // The values are inferred from fSecurity and fDetails.

   if (gDebug > 2)
      Info("SetEnvironment",
           "setting environment: fSecurity:%d, fDetails:%s", fSecurity,
           fDetails.Data());

   // Defaults
   gSystem->Setenv("PROMPTUSER", "0");
   gSystem->Setenv("AUTHREUSE", "0");
   gSystem->Setenv("DEFAULTUSER", "");

   // Decode fDetails, is non empty ...
   if (fDetails != "") {
      char UsDef[kMAXPATHLEN] = { 0 };
      int lDet = strlen(fDetails.Data()) + 2;
      char Pt[5] = { 0 }, Ru[5] = { 0 };
      char *Us = 0, *Cd = 0, *Cf = 0, *Kf = 0, *Ad = 0, *Cp = 0;
      const char *ptr;
      if ((ptr = strstr(fDetails, "pt:")) != 0)
         sscanf(ptr + 3, "%s %s", Pt, UsDef);
      if ((ptr = strstr(fDetails, "ru:")) != 0)
         sscanf(ptr + 3, "%s %s", Ru, UsDef);

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
      gSystem->Setenv("PROMPTUSER", "0");
      if (!strncmp(Pt, "yes", 3) || !strncmp(Pt, "1", 1))
         gSystem->Setenv("PROMPTUSER", "1");

      // Set ReUse flag
      if (fSecurity != kRfio) {
         gSystem->Setenv("AUTHREUSE", "1");
         if (!strcmp(Ru, "no") || !strcmp(Ru, "0"))
            gSystem->Setenv("AUTHREUSE", "0");
      } else {
         gSystem->Setenv("AUTHREUSE", "0");
         if (!strcmp(Ru, "yes") || !strcmp(Ru, "1"))
            gSystem->Setenv("AUTHREUSE", "1");
      }

      if (fSecurity == kClear) {
         // Set Crypt flag
         gSystem->Setenv("CLEARCRYPT", "1");
         if (!strncmp(Cp, "no", 2) || !strncmp(Cp, "0", 1))
            gSystem->Setenv("CLEARCRYPT", "0");
      }
      // Build UserDefaults
      if (fSecurity == kGlobus) {
         if (Cd != 0) {
            sprintf(UsDef, "%s %s", UsDef, Cd);
            SafeDelete(Cd);
         }
         if (Cf != 0) {
            sprintf(UsDef, "%s %s", UsDef, Cf);
            SafeDelete(Cf);
         }
         if (Kf != 0) {
            sprintf(UsDef, "%s %s", UsDef, Kf);
            SafeDelete(Kf);
         }
         if (Ad != 0) {
            sprintf(UsDef, "%s %s", UsDef, Ad);
            SafeDelete(Ad);
         }
      } else {
         if (fUser == "") {
            if (Us != 0) {
               sprintf(UsDef, "%s", Us);
               SafeDelete(Us);
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
      if (strlen(UsDef) > 0)
         gSystem->Setenv("DEFAULTUSER", (const char *) UsDef);

      if (gDebug > 2)
         Info("SetEnvironment", "UsDef:%s", UsDef);
   }
}

//______________________________________________________________________________
Bool_t TAuthenticate::GetUserPasswd(TString & user, TString & passwd)
{
   // Try to get user name and passwd from several sources

   // Get user and passwd set via static functions SetUser and SetPasswd.
//    if ( user  =="" && fgUser   != "" )
//       user = fgUser;
//    if ( passwd=="" && fgPasswd != "" )
//       passwd = fgPasswd;


   if (user == "") {
      if (fgUser != "")
         user = fgUser;
      if (passwd == "" && fgPasswd != "")
         passwd = fgPasswd;
   } else {
      if (fgUser != "" && user == fgUser) {
         if (passwd == "" && fgPasswd != "")
            passwd = fgPasswd;
      }
   }


   // Check ~/.rootnetrc and ~/.netrc files if user was not set via
   // the static SetUser() method.
   if (user == "" || passwd == "")
      CheckNetrc(user, passwd);

   // If user also not set via  ~/.rootnetrc or ~/.netrc ask user.
   if (user == "") {
      user = PromptUser(fRemote);
      if (user == "") {
         Error("GetUserPasswd", "user name not set");
         return 1;
      }
   }
   // Fill present user info ...
   fUser = user;
   fPasswd = passwd;

   // For potential later use in Proof
   fgUser = fUser;
   fgPasswd = fPasswd;

   return 0;
}

//______________________________________________________________________________
Bool_t TAuthenticate::CheckNetrc(TString & user, TString & passwd)
{
   // Try to get user name and passwd from the ~/.rootnetrc or
   // ~/.netrc files. First ~/.rootnetrc is tried, after that ~/.netrc.
   // These files will only be used when their access masks are 0600.
   // Returns kTRUE if user and passwd were found for the machine
   // specified in the URL. If kFALSE, user and passwd are "".
   // The format of these files are:
   //
   // # this is a comment line
   // machine <machine fqdn> login <user> password <passwd>
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
            if (fSecurity == kSRP && strcmp(word[0], "secure"))
               continue;
            if (fSecurity == kClear && strcmp(word[0], "machine"))
               continue;
            if (strcmp(word[2], "login"))
               continue;
            if (strcmp(word[4], "password"))
               continue;

            if (!strcmp(word[1], remote)) {
               if (user == "") {
                  user = word[3];
                  passwd = word[5];
                  result = kTRUE;
                  break;
               } else {
                  if (!strcmp(word[3], user.Data())) {
                     passwd = word[5];
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

   if (first && fSecurity == kClear && !result) {
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
char *TAuthenticate::PromptUser(const char *remote)
{
   // Static method to prompt for the user name to be used for authentication
   // to rootd or proofd. User is asked to type user name.
   // Returns user name (which must be deleted by caller) or 0.
   // If non-interactive run (eg ProofServ) returns default user.

   TApplication *lApp = gROOT->GetApplication();

   const char *user;
   if (gSystem->Getenv("DEFAULTUSER") != 0
       && strlen(gSystem->Getenv("DEFAULTUSER")) > 0)
      user = gSystem->Getenv("DEFAULTUSER");
   else
      user = gSystem->Getenv("USER");
#ifdef R__WIN32
   if (!user)
      user = gSystem->Getenv("USERNAME");
#endif
   if (strstr(lApp->Argv()[1], "proof") != 0) {
      ::Warning("PromptUser",
                "proofserv: cannot prompt for user: returning default");
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

   TApplication *lApp = gROOT->GetApplication();
   if (strstr(lApp->Argv()[1], "proof") != 0) {
      ::Warning("PromptPasswd",
                "proofserv: cannot prompt for passwd: returning -1");
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
   // Use ssh to authenticate.

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
   if (strcmp(gEnv->GetValue("Ssh.ExecDir", "-1"), "-1")) {
      SafeDelete(gSshExe);
      gSshExe =
          StrDup(Form
                 ("%s/ssh", (char *) gEnv->GetValue("Ssh.ExecDir", "")));
      if (gSystem->AccessPathName(gSshExe, kExecutePermission)) {
         Info("SshAuth", "%s not executable", gSshExe);
         SafeDelete(gSshExe);
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
   Int_t ReUse = 1, Prompt = 0;
   char PromptReUse[20];
   if (gSystem->Getenv("AUTHREUSE") != 0 &&
       !strcmp(gSystem->Getenv("AUTHREUSE"), "0"))
      ReUse = 0;
   if (gSystem->Getenv("PROMPTUSER") != 0
       && !strcmp(gSystem->Getenv("PROMPTUSER"), "1"))
      Prompt = 1;
   sprintf(PromptReUse, "pt:%d ru:%d us:", Prompt, ReUse);
   fDetails = (const char *) PromptReUse + User;

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
      SafeDelete(Options);
      return 1;
   }
   if (rc == -2) {
      SafeDelete(Options);
      return rc;
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
   // Send authentication request to remote sshd
   // Create command
   char sshcmd[kMAXPATHLEN] = { 0 };
   if (rport == -1) {
      // Remote server did not specify a specific port ... use our default, whatever it is ...
      sprintf(sshcmd, "%s -x -l %s %s %s", gSshExe, User.Data(),
              fRemote.Data(), CmdInfo);
   } else {
      // Remote server did specify a specific port ...
      sprintf(sshcmd, "%s -x -l %s -p %d %s %s", gSshExe, User.Data(),
              rport, fRemote.Data(), CmdInfo);
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
      int id1, id2;
      sscanf(CmdInfo, "%s %s %d %d", cd1, pipe, &id1, &id2);
      sprintf(SecName, "%d -1 0 %s %ld %s None", -gSystem->GetPid(), pipe,
              (Long_t)strlen(User), User.Data());
      newsock->Send(SecName, kROOTD_SSH);
      delete newsock;
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
      if (retval == kErrConnectionRefused)
         return -2;
      return 0;
   }

   if (ReUse == 1) {

      // Save type of key
      if (kind != kROOTD_RSAKEY)
         Warning("SshAuth",
                 "problems recvn RSA key flag: got message %d, flag: %d",
                 kind, fRSAKey);

      fRSAKey = 1;

      // RSA key generation (one per session)
      if (!fgRSAInit) {
         GenRSAKeys();
         fgRSAInit = 1;
      }
      // Send key
      if (gDebug > 3)
         Info("SshAuth", "sending Local Key:\n '%s'", fgRSAPubExport.keys);
      fSocket->Send(fgRSAPubExport.keys, kROOTD_RSAKEY);

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
      if (SecureRecv(fSocket, fRSAKey, &Token) == -1) {
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
   SafeDelete(answer);
   SafeDelete(lUser);
   SafeDelete(Token);

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
   // Looks first at Ssh.Login and finally at env USER.
   // If Ssh.LoginPrompt is set to 'yes' it prompts for the 'login name'

   static TString user = "";

   if (gSystem->Getenv("PROMPTUSER") != 0
       && !strcmp(gSystem->Getenv("PROMPTUSER"), "1")) {
      user = PromptUser(fRemote);
   } else {
      user = gSystem->Getenv("DEFAULTUSER");
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

   int i;

   if (gDebug > 2)
      ::Info("GetAuthMeth", "enter: %s %s %s 0x%lx 0x%lx ", Host, Proto,
             *User[0], (long) (*User), (long) (*User[0]));

   if (*User[0] == 0)
      *User[0] = StrDup("");

   // If 'host' is ourselves, then use rfio (to setup things correctly)
   // Check and save the host FQDN ...
   static TString LocalFQDN;
   if (LocalFQDN == "") {
      TInetAddress addr = gSystem->GetHostByName(gSystem->HostName());
      if (addr.IsValid()) {
         LocalFQDN = addr.GetHostName();
         if (LocalFQDN == "UnNamedHost")
            LocalFQDN = addr.GetHostAddress();
      }
   }

   if (LocalFQDN == Host) {
      if (gDebug > 3)
         ::Info("GetAuthMeth", "remote host is the local one (%s)",
                LocalFQDN.Data());
      *NumMeth = new int[1];
      *NumMeth[0] = 1;
      AuthMeth[0] = new int[1];
      AuthMeth[0][0] = 5;
      Details[0] = new char *[1];
      Details[0][0] = StrDup(Form("pt:0 ru:0 us:%s", *User[0]));
      return 1;
   }
   // If specific protocol was specified then it has absolute priority ...
   if (!strcmp(Proto, "roots") || !strcmp(Proto, "proofs")) {
      *NumMeth = new int[1];
      *NumMeth[0] = 1;
      AuthMeth[0] = new int[1];
      AuthMeth[0][0] = 1;
      Details[0] = new char *[1];
      Details[0][0] = StrDup(Form("pt:%s ru:%s us:%s",
                                  gEnv->GetValue("SRP.LoginPrompt", "no"),
                                  gEnv->GetValue("SRP.ReUse", "0"),
                                  gEnv->GetValue("SRP.Login", *User[0])));
      return 1;
   } else if (!strcmp(Proto, "rootk") || !strcmp(Proto, "proofk")) {
      *NumMeth = new int[1];
      *NumMeth[0] = 1;
      AuthMeth[0] = new int[1];
      AuthMeth[0][0] = 2;
      Details[0] = new char *[1];
      Details[0][0] = StrDup(Form("pt:%s ru:%s us:%s",
                                  gEnv->GetValue("Krb5.LoginPrompt", "no"),
                                  gEnv->GetValue("Krb5.ReUse", "0"),
                                  gEnv->GetValue("Krb5.Login", *User[0])));
      return 1;
   }
   // Check then .rootauthrc (if there)
   char temp[kMAXPATHLEN];
   Int_t *am[kMAXSEC], *nh, nu = 0, j = 0;
   char **det[kMAXSEC];
   if ((nu = CheckRootAuthrc(Host, User, &nh, am, det)) > 0) {
      if (gDebug > 3)
         ::Info("GetAuthMeth", "found %d users - nh: %d 0x%lx %s ", nu,
                nh[0], (long) (*User[0]), *User[0]);
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
                  Details[i][j] = GetDefaultDetails(am[i][j], 0, usr);
                  SafeDelete(usr);
               } else {
                  Details[i][j] = det[i][j];
               }
            }
         }
      }
      if (gDebug > 4) {
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
   nh = new int;
   nh[0] = 0;
   am[0] = new int[kMAXSEC];
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
      return 0;
   }
   // Variables for scan ...
   char line[kMAXPATHLEN], rest[kMAXPATHLEN];

   // Generate temporary file name and open it
   int expand = 1;
   FILE *ftmp = 0;
   char filetmp[L_tmpnam];
   tmpnam(filetmp);
   ftmp = fopen(filetmp, "w+");
   if (ftmp == 0)
      expand = 0;  // Problems opening temporary file: ignore 'include' directives ...

   // Open file
   FILE *fd = fopen(net, "r");

   // If the temporary file is open, copy everything to the new file ...
   if (expand == 1) {
      TAuthenticate::FileExpand(net, ftmp);
      fclose(fd);
      fd = ftmp;
      rewind(fd);
   }
   // If empty user you need first to count how many entries are to be read
   if ((*user)[0] == 0 || strlen((*user)[0]) == 0) {
      char usrtmp[256];
      unsigned int tlen = kMAXPATHLEN;
      char *Temp = new char[tlen];
      Temp[0] = '\0';
      while (fgets(line, sizeof(line), fd) != 0) {
         // Skip comment lines
         if (line[0] == '#')
            continue;
         char *pstr = strstr(line, Host);
         char *pdef = strstr(line, "default");
         if ((pstr != 0 && pstr == line) || (pdef != 0 && pdef == line)) {
            unsigned int hlen = strlen(Host);
            if (strstr(Temp, Host) == 0) {
               if ((strlen(Temp) + hlen + 2) > tlen) {
                  char *NewTemp = StrDup(Temp);
                  SafeDelete(Temp);
                  tlen += kMAXPATHLEN;
                  Temp = new char[tlen];
                  strcpy(Temp, NewTemp);
                  SafeDelete(NewTemp);
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
                        SafeDelete(Temp);
                        tlen += kMAXPATHLEN;
                        Temp = new char[tlen];
                        strcpy(Temp, NewTemp);
                        SafeDelete(NewTemp);
                     }
                     sprintf(Temp, "%s %s", Temp, usrtmp);
                     Nuser++;
                  }
               }
            }
         }
      }
      if (Temp != 0)
         SafeDelete(Temp);
      if (gDebug > 3)
         ::Info("CheckRootAuthrc",
                "found %d different entries for host %s", Nuser, Host);

      if ((*user)[0] != 0)
         SafeDelete((*user)[0]);
      (*user) = new char *[Nuser];
      *nh = new int[Nuser];
      int i;
      for (i = 0; i < kMAXSEC; i++) {
         am[i] = new int[Nuser];
         det[i] = new char *[Nuser];
      }
      UserRq = StrDup("-1");
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
   rewind(fd);

   // Scan it ...
   char host[kMAXPATHLEN], info[kMAXPATHLEN];
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
      if (gDebug > 4)
         ::Info("CheckRootAuthrc", "found line ... %s ", line);

      // The list of data servers for proof is analyzed elsewhere (TProof ...)
      if (!strcmp(host, "proofdserv"))
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
               if ((*user)[ju] != 0)
                  SafeDelete((*user)[ju]);
            }
            (*user)[ju] = StrDup(usr);
            (*nh)[ju] = 0;
            for (i = 0; i < kMAXSEC; i++) {
               am[i][ju] = -1;
               det[i][ju] = 0;
            }
         }
         if (usr != 0)
            SafeDelete(usr);
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
               if ((*user)[ju] != 0)
                  SafeDelete((*user)[ju]);
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

         //       nmeth= sscanf(rest,"%d %d %d %d %d %d",&mth[0],&mth[1],&mth[2],&mth[3],&mth[4],&mth[5]);
         char cmth[kMAXSEC][20];
         int nw =
             sscanf(rest, "%s %s %s %s %s %s", cmth[0], cmth[1], cmth[2],
                    cmth[3], cmth[4], cmth[5]);
         nmeth = 0;
         for (i = 0; i < nw; i++) {
            int met = -1;
            if (strlen(cmth[i]) > 1) {
               // Method passed as string: translate it to number
               const char *pmet = strstr(kMethods, cmth[i]);
               if (pmet != 0) {
                  met = ((int) (pmet - kMethods)) / 7;
               } else {
                  if (gDebug > 2)
                     ::Info("CheckRootAuthrc",
                            "unrecognized method (%s): ", cmth[i]);
                  met = -1;
               }
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
                     SafeDelete(tmp_det[k]);
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
                     SafeDelete(tmp_det[k]);
                     tmp_det[k] = 0;
                  }
                  k++;
               }
            }
         }
         if (tmp_am != 0)
            SafeDelete(tmp_am);
         if (tmp_det != 0)
            SafeDelete(tmp_det);
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
               const char *pmet = strstr(kMethods, cmeth);
               if (pmet != 0) {
                  meth = ((int) (pmet - kMethods)) / 7;
               } else {
                  if (gDebug > 2)
                     ::Info("CheckRootAuthrc",
                            "unrecognized method (%s): ", cmeth);
                  meth = -1;
               }
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
                  SafeDelete(ostr);
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
         SafeDelete((*user)[0]);
         (*user)[0] = StrDup(UserRq);
      }
   }

   if (gDebug > 4) {
      rewind(fd);
      ::Info("CheckRootAuthrc",
             "+----------- temporary file: %s -------------------------+",
             filetmp);
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
   SafeDelete(net);
   SafeDelete(UserRq);
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

   // Get IP of the host in form of a string
   TInetAddress addr = gSystem->GetHostByName(Host);
   char *IP = StrDup(addr.GetHostAddress());
   if (gDebug > 2)
      ::Info("CheckHost", "host: %s --> IP: %s", Host, IP);

   // now check validity of 'host' format
   // Try first to understand whether it is an address or a name ...
   int i, name = 0, namew = 0, nd = 0, nn = 0, nnmx = 0,
       nnmi = strlen(host);
   for (i = 0; i < (int) strlen(host); i++) {
      if (host[i] == '.') {
         nd++;
         if (nn > nnmx)
            nnmx = nn;
         if (nn < nnmi)
            nnmi = nn;
         nn = 0;
         continue;
      }
      int j = (int) host[i];
      if (j < 48 || j > 57)
         name = 1;
      if (host[i] == '*') {
         namew = 1;
         if (nd > 0) {
            retval = kFALSE;
            goto exit;
         }
      }
      nn++;
   }

   // Act accordingly ...
   if (name == 0) {
      if (nd < 4) {
         if (strlen(host) < 16) {
            if (nnmx < 4) {
               if (nd == 3 || host[strlen(host) - 1] == '.') {
                  char *sp = strstr(IP, host);
                  if (sp == 0 || sp != IP) {
                     retval = kFALSE;
                     goto exit;
                  }
               }
            }
         }
      }
   } else {
      if (namew == 0) {
         if (nd > 0) {
            if (nd > 1 || nnmi > 0) {
               const char *sp = strstr(Host, host);
               if (sp == 0 || sp != Host) {
                  retval = kFALSE;
                  goto exit;
               }
            }
         }
      } else {
         if (!CheckHostWild(Host, host)) {
            retval = kFALSE;
            goto exit;
         }
      }
   }

   if (gDebug > 2)
      ::Info("CheckHost", "info for host found in table ");

 exit:
   SafeDelete(IP);
   return retval;
}

//______________________________________________________________________________
Bool_t TAuthenticate::CheckHostWild(const char *Host, const char *host)
{
   // Checks if 'host' is compatible with 'Host' taking into account
   // wild cards in the machine name (first field of FQDN) ...
   // Returns 0 if successful, 1 otherwise ...

   Bool_t rc = kTRUE;
   char *fH, *sH, *dum, *sp, *k;
   int i, j, lmax;

   if (gDebug > 2)
      ::Info("CheckHostWild", "enter: H: '%s' h: '%s'", Host, host);

   // Max length for dinamic allocation
   lmax = strlen(Host) > strlen(host) ? strlen(Host) : strlen(host);

   // allocate
   fH = new char[lmax];
   sH = new char[lmax];
   dum = new char[lmax];

   // Determine 'Host' first field (the name) ...
   for (i = 0; i < (int) strlen(Host); i++) {
      if (Host[i] == '.')
         break;
   }
   strncpy(fH, Host, i);
   fH[i] = '\0';
   // ... and also the second one (the domain)
   strcpy(sH, Host + i);
   if (gDebug > 3)
      ::Info("CheckHostWild", "fH:%s sH:%s", fH, sH);

   // Now check the first field ...
   j = 0;
   k = fH;
   for (i = 0; i < (int) strlen(host); i++) {
      if (host[i] == '.')
         break;
      if (host[i] == '*') {
         if (i > 0) {
            // this is the part of name before the '*' ....
            strncpy(dum, host + j, i - j);
            dum[i - j] = '\0';
            if (gDebug > 3)
               ::Info("CheckHostWild", "k:%s dum:%s", k, dum);
            sp = strstr(k, dum);
            if (sp == 0) {
               rc = kFALSE;
               goto exit;
            }
            j = i + 1;
            k = sp + strlen(dum) + 1;
         } else
            j++;
      }
   }
   // Now check the domain name (if the name matches ...)
   if (rc) {
      strcpy(dum, host + i);
      if (gDebug > 3)
         ::Info("CheckHostWild", "sH:%s dum:%s", sH, dum);
      sp = strstr(sH, dum);
      if (sp == 0) {
         rc = kFALSE;
         goto exit;
      }
   }

 exit:
   // Release allocated memory ...
   SafeDelete(fH);
   SafeDelete(sH);
   SafeDelete(dum);
   return rc;
}

//______________________________________________________________________________
Int_t TAuthenticate::RfioAuth(TString & User)
{
   // Rfio-like authentication code.
   // Returns 0 in case authentication failed
   //         1 in case of success
   //        <0 in case of system error

   if (gDebug > 2)
      Info("RfioAuth", "enter ... User %s", User.Data());

   User = gSystem->Getenv("USER");
   fDetails = TString("pt:0 ru:0 us:") + User;

   // Now check that we are not root ...
   UserGroup_t *pw = gSystem->GetUserInfo();
   if (pw) {
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
            // Authentication failed
            if (kind == kROOTD_ERR) {
               if (gDebug > 0)
                  AuthError("Authenticate RFIO", stat);
               if (stat == kErrConnectionRefused)
                  return -2;
            }
            return 0;
         }
      } else {
         Warning("RfioAuth", "RFIO login as \"root\" not allowed");
         return -1;
      }
   }
   delete pw;
   return -1;
}

//______________________________________________________________________________
Int_t TAuthenticate::ClearAuth(TString & User, TString & Passwd)
{
   // Clear-like authentication code.
   // Returns 0 in case authentication failed
   //         1 in case of success

   if (gDebug > 2)
      Info("ClearAuth", "enter: User: %s", User.Data());

   // Check ReUse
   Int_t ReUse = 1, Prompt = 0, Crypt = 1;
   char PromptReUse[40];
   if (gSystem->Getenv("AUTHREUSE") != 0 &&
       !strcmp(gSystem->Getenv("AUTHREUSE"), "0"))
      ReUse = 0;
   if (gSystem->Getenv("PROMPTUSER") != 0 &&
       !strcmp(gSystem->Getenv("PROMPTUSER"), "1"))
      Prompt = 1;
   if (gSystem->Getenv("CLEARCRYPT") != 0 &&
       !strcmp(gSystem->Getenv("CLEARCRYPT"), "0"))
      Crypt = 0;
#ifdef R__WIN32
   Crypt = 0;
#endif
   sprintf(PromptReUse, "pt:%d ru:%d cp:%d us:", Prompt, ReUse, Crypt);
   fDetails = (TString) ((const char *) PromptReUse + User);

   Int_t stat, kind;

   if (fVersion > 1) {

      // New protocol

      Int_t Anon = 0;
      Int_t OffSet = -1;
      char *Salt = 0;
      char *PasHash = 0;

      // Create Options string
      char *Options = new char[strlen(User.Data()) + 40];
      int Opt = (ReUse * kAUTH_REUSE_MSK) + (Crypt * kAUTH_CRYPT_MSK);
      sprintf(Options, "%d %ld %s", Opt, (Long_t)strlen(User), User.Data());

      // Check established authentications
      kind = kROOTD_USER;
      stat = ReUse;
      Int_t rc = 0;
      if ((rc =
           AuthExists(this, (Int_t) TAuthenticate::kClear, fDetails,
                      Options, &kind, &stat)) == 1) {
         // A valid authentication exists: we are done ...
         SafeDelete(Options);
         if (gDebug > 3)
            Info("ClearAuth", "valid authentication exists: return 1");
         return 1;
      }
      if (rc == -2) {
         SafeDelete(Options);
         return rc;
      }

      if (kind == kROOTD_AUTH && stat == -1) {
         if (gDebug > 3)
            Info("ClearAuth", "anounymous user", kind, stat);
         Anon = 1;
      }

      if (OffSet == -1 && Anon == 0 && Crypt == 1) {

         // Check that we got the right thing ..
         if (kind != kROOTD_RSAKEY) {
            Warning("ClearAuth",
                    "problems recvn RSA key flag: got message %d, flag: %d",
                    kind, stat);
            return 0;
         }
         if (gDebug > 3)
            Info("ClearAuth", "get key request ...");

         // Save type of key
         fRSAKey = 1;

         // RSA key generation (one per session)
         if (!fgRSAInit) {
            GenRSAKeys();
            fgRSAInit = 1;
         }
         // Send key to server
         if (gDebug > 3)
            Info("ClearAuth", "sending Local Key:\n '%s'",
                 fgRSAPubExport.keys);
         fSocket->Send(fgRSAPubExport.keys, kROOTD_RSAKEY);

         // Receive password salt
         if (SecureRecv(fSocket, fRSAKey, &Salt) == -1) {
            Warning("ClearAuth",
                    "problems secure-receiving salt - may result in corrupted salt");
            Warning("ClearAuth", "switch off reuse for this session");
            Crypt = 0;
         }
         if (gDebug > 2)
            Info("ClearAuth", "got salt: '%s'", Salt);
      }
      // Now get the password either from prompt or from memory, if saved already
      if (Anon == 1 && Prompt == 0) {

         // Anonymous like login with automatic passwd generation ...
         char *LocalUser = StrDup(gSystem->Getenv("USER"));
         if (LocalUser == 0 || strlen(LocalUser) <= 0) {
            UserGroup_t *pw = gSystem->GetUserInfo();
            if (pw)
               LocalUser = StrDup(pw->fUser);
            delete pw;
         }
         if (strlen(gSystem->Getenv("HOST")) > 0) {
            Passwd = Form("%s@%s", LocalUser, gSystem->Getenv("HOST"));
         } else {
            Passwd = Form("%s@localhost", LocalUser);
         }
         if (gDebug > 2)
            Info("ClearAuth",
                 "automatically generated anonymous passwd: %s",
                 Passwd.Data());
         SafeDelete(LocalUser);

      } else {

         if (Prompt == 1 || PasHash == 0) {
          badpass:
            if (Passwd == "") {
               Passwd = PromptPasswd();
               if (Passwd == "") {
                  Error("ClearAuth", "password not set");
                  SafeDelete(PasHash);
                  SafeDelete(Salt);
                  return 0;
               }
            }
            if (Crypt == 1) {
               // Get hash
#ifndef R__WIN32
               PasHash = StrDup(crypt(Passwd, Salt));
#endif
            }
            if (Anon == 1) {
               if (!Passwd.Contains("@")) {
                  Warning("ClearAuth",
                          "please use passwd of form: user@host.do.main");
                  Passwd = "";
                  goto badpass;
               }
            }
         }

      }
      fgPasswd = Passwd;
      fPasswd = Passwd;

      // Send it to server
      if (Anon == 0 && Crypt == 1) {
         fSocket->Send("\0", kROOTD_PASS);  // Needs this for consistency
         if (SecureSend(fSocket, fRSAKey, PasHash) == -1) {
            Warning("ClearAuth",
                    "problems secure-sending pass hash - may result in authentication failure");
         }
      } else {
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
               if (SecureRecv(fSocket, fRSAKey, &Token) == -1) {
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

         SafeDelete(answer);
         SafeDelete(lUser);
      }
      // Release allocated memory ...
      SafeDelete(Salt);
      SafeDelete(PasHash);


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
         if (gDebug > 0)
            AuthError("ClearAuth", stat);
         if (stat == kErrConnectionRefused)
            return -2;
         return 0;
      }
      // Prepare Passwd to send
    badpass1:
      if (Passwd == "") {
         Passwd = PromptPasswd();
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
         SafeDelete(Wd[i]);
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
         SafeDelete(Wd[i]);
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
              (long) Pt, (long) Ru);
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
      SafeDelete(Temp);
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
              (long) Pt, (long) Ru);
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

      SafeDelete(Temp);
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

   // Check list of auth info for already loaded info about this host
   TIter next(GetAuthInfo());
   THostAuth *ai;
   while ((ai = (THostAuth *) next())) {
      if (gDebug > 3)
         ai->Print("Authenticate:GetHostAuth");

      if (ulen > 0) {
         if (!strcmp(host, ai->GetHost()) && !strcmp(user, ai->GetUser())) {
            rHA = ai;
            break;
         }
      } else {
         if (!strcmp(host, ai->GetHost())) {
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

   fin = fopen(fexp, "r");
   if (fin == 0)
      return;

   while (fgets(line, sizeof(line), fin) != 0) {
      // Skip comment lines
      if (line[0] == '#')
         continue;
      if (line[strlen(line) - 1] == '\n')
         line[strlen(line) - 1] = '\0';
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
   // Determine default authentication details for method 'sec' and user 'usr'
   // checks .rootrc family files. Returned string must be deleted by the user.

   char temp[kMAXPATHLEN] = { 0 };
   const char copt[2][5] = { "no", "yes" };

   if (gDebug > 2)
      ::Info("GetDefaultDetails", "enter ... %d ...pt:%d ... '%s'", sec,
             opt, usr);

   if (opt < 0 || opt > 1)
      opt = 1;

   // UsrPwdClear
   if (sec == TAuthenticate::kClear) {
      if (strlen(usr) == 0)
         usr = gEnv->GetValue("UsrPwd.Login", "");
      sprintf(temp, "pt:%s ru:%s us:%s cp:%s",
              gEnv->GetValue("UsrPwd.LoginPrompt", copt[opt]),
              gEnv->GetValue("UsrPwd.ReUse", "1"), usr,
              gEnv->GetValue("UsrPwd.Crypt", "1"));

      // SRP
   } else if (sec == TAuthenticate::kSRP) {
      if (strlen(usr) == 0)
         usr = gEnv->GetValue("SRP.Login", "");
      sprintf(temp, "pt:%s ru:%s us:%s",
              gEnv->GetValue("SRP.LoginPrompt", copt[opt]),
              gEnv->GetValue("SRP.ReUse", "0"), usr);

      // Kerberos
   } else if (sec == TAuthenticate::kKrb5) {
      if (strlen(usr) == 0)
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
      if (strlen(usr) == 0)
         usr = gEnv->GetValue("Ssh.Login", "");
      sprintf(temp, "pt:%s ru:%s us:%s",
              gEnv->GetValue("Ssh.LoginPrompt", copt[opt]),
              gEnv->GetValue("Ssh.ReUse", "1"), usr);

      // Uid/Gid
   } else if (sec == TAuthenticate::kRfio) {
      if (strlen(usr) == 0)
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
   usr[0] = StrDup(user);
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
         for (i = 0; i < nmeth[0]; i++) {
            int j, jm = -1;
            for (j = 0; j < hostAuth->NumMethods(); j++) {
               if (am[i] == hostAuth->GetMethods(j)) {
                  hostAuth->SetDetails(am[i], det[i]);
                  jm = j;
               }
            }
            if (jm == -1)
               hostAuth->AddMethod(am[i], det[i]);
         }
      }
      if (gDebug > 3)
         hostAuth->Print();
      for (i = 0; i < nm; i++) {
         if (det[i] != 0)
            SafeDelete(det[i]);
      }
   }
   for (i = 0; i < kMAXSEC; i++) {
      if (security[i] != 0)
         SafeDelete(security[i]);
      if (details[i] != 0) {
         for (j = 0; j < nu; j++) {
            if (details[i][j] != 0)
               SafeDelete(details[i][j]);
         }
      }
   }
   for (i = 0; i < nu; i++) {
      SafeDelete(usr[i]);
   }
   SafeDelete(nmeth);
   SafeDelete(usr);
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
         if (SecureSend(Socket, RSAKey, Token) == -1) {
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

      // Answer to server question to prove your id
      //     Auth->ProveId();
   }
   // Release allocated memory
   SafeDelete(Token);
   SafeDelete(sstr);

   Int_t stat, kind;
   Socket->Recv(stat, kind);
   if (gDebug > 3)
      ::Info("AuthExists", "%d: after msg %d: kind= %d, stat= %d", Sec,
             *Message, kind, stat);

   // Return flags
   *Message = kind;
   *Rflag = stat;

   if (kind == kROOTD_ERR) {
      if (gDebug > 0)
         AuthError("CheckAuth", stat);
      if (stat == kErrConnectionRefused)
         return -2;
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
   // Sometimes some bunch is not decrypted correctly
   // That's why we make retries to make sure that encryption/decryption works as expected
   Bool_t NotOk = 1;
   rsa_NUMBER p1, p2, rsa_n, rsa_e, rsa_d;
   int l_n, l_e, l_d;
   char buf[rsa_STRLEN];
   char buf_n[rsa_STRLEN], buf_e[rsa_STRLEN], buf_d[rsa_STRLEN];

   while (NotOk) {

      // Valid pair of primes
      p1 = rsa_fun::fg_rsa_genprim(10, 20);
      p2 = rsa_fun::fg_rsa_genprim(11, 20);

      // Retry if equal
      while (rsa_fun::fg_rsa_cmp(&p1, &p2) == 0)
         p2 = rsa_fun::fg_rsa_genprim(11, 20);

      if (gDebug > 3) {
         rsa_fun::fg_rsa_num_sput(&p1, buf, rsa_STRLEN);
         Info("GenRSAKeys", "local: p1: '%s' ", buf);
         rsa_fun::fg_rsa_num_sput(&p2, buf, rsa_STRLEN);
         Info("GenRSAKeys", "local: p2: '%s' ", buf);
      }
      // Generate keys
      rsa_fun::fg_rsa_genrsa(p1, p2, &rsa_n, &rsa_e, &rsa_d);

      // Determine their lengths
      rsa_fun::fg_rsa_num_sput(&rsa_n, buf_n, rsa_STRLEN);
      l_n = strlen(buf_n);
      rsa_fun::fg_rsa_num_sput(&rsa_e, buf_e, rsa_STRLEN);
      l_e = strlen(buf_e);
      rsa_fun::fg_rsa_num_sput(&rsa_d, buf_d, rsa_STRLEN);
      l_d = strlen(buf_d);

      if (gDebug > 3) {
         Info("GenRSAKeys", "local: n: '%s' length: %d", buf_n, l_n);
         Info("GenRSAKeys", "local: e: '%s' length: %d", buf_e, l_e);
         Info("GenRSAKeys", "local: d: '%s' length: %d (%d)", buf_d, l_d);
      }
//      if (l_n != 22 && l_e != 22 && l_d != 22) continue;
//      if (l_n == 23 || l_e == 23 || l_d == 23) continue;
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
              "local: length of crypted string: %d bytes '%s'", lout, buf);

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

   if (gDebug > 2) {
      // Determine their lengths
      Info("GenRSAKeys", "local: generated keys are:");
      Info("GenRSAKeys", "local: n: '%s' length: %d", buf_n, l_n);
      Info("GenRSAKeys", "local: e: '%s' length: %d", buf_e, l_e);
      Info("GenRSAKeys", "local: d: '%s' length: %d", buf_d, l_d);
   }
   // Export form
   fgRSAPubExport.len = l_n + l_d + 4;
   fgRSAPubExport.keys = new char[fgRSAPubExport.len];

   fgRSAPubExport.keys[0] = '#';
   memcpy(fgRSAPubExport.keys + 1, buf_n, l_n);
   fgRSAPubExport.keys[l_n + 1] = '#';
   memcpy(fgRSAPubExport.keys + l_n + 2, buf_d, l_d);
   fgRSAPubExport.keys[l_n + l_d + 2] = '#';
   fgRSAPubExport.keys[l_n + l_d + 3] = 0;
   if (gDebug > 2)
      Info("GenRSAKeys", "local: export pub: '%s'", fgRSAPubExport.keys);

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
Int_t TAuthenticate::SecureSend(TSocket *Socket, Int_t Key, char *Str)
{
   // Encode null terminated Str using the session private key indcated by Key
   // and sends it over the network
   // Returns number of bytes sent.or -1 in case of error.

   char BufTmp[kMAXSECBUF];
   char BufLen[20];

   if (gDebug > 2)
      ::Info("SecureSend", "local: enter ... '%s' (%d)", Str, Key);

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
      if (gDebug > 4)
         ::Info("SecureSend",
                "local: sent %d bytes (expected: %d) (buffer '%s')", Nsen,
                Ttmp, BufTmp);
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

   char BufTmp[kMAXSECBUF];
   char BufLen[20];

   Int_t Nrec = -1;
   // We must get a pointer ...
   if (!Str)
      return Nrec;

   Int_t kind;
   Socket->Recv(BufLen, 20, kind);
   Int_t Len = atoi(BufLen);
   if (gDebug > 4)
      ::Info("SecureRecv", "got len '%s' %d (msg kind: %d)", BufLen, Len,
             kind);
   if (!strncmp(BufLen, "-1", 2))
      return Nrec;

   // Now proceed
   if (Key == 1) {
      Nrec = Socket->RecvRaw(BufTmp, Len);
      rsa_fun::fg_rsa_decode(BufTmp, Len, fgRSAPriKey.n, fgRSAPriKey.e);
      if (gDebug > 4)
         ::Info("SecureRecv", "local: decoded string: '%s' ", BufTmp);
   } else {
      ::Info("SecureRecv", "unknown key option (%d) - return", Key);
   }

   *Str = new char[strlen(BufTmp) + 1];
   strcpy(*Str, BufTmp);

   return Nrec;
}

//______________________________________________________________________________
Int_t TAuthenticate::ProveId()
{
   // Answer to server requests to prove ID.

#if 0
   //crypt not available on windows (rdm)
   char *Question = 0;
   // Send Answer encrypted
//  if (SecureRecv(fSocket,fRSAKey,&Question) == -1){
//    Warning("ProveId","problems secure-receiving Question - may result in corrupted Question");
//  }
//  if (gDebug > 3) Info("ProveId","got question '%s'",Question);

   Question = GetRandString(2, 6);

   // Prepare answer
   char Answer[kMAXPATHLEN];
   sprintf(Answer, "%s$%s", Question, crypt(Question, Question));
   if (gDebug > 3)
      Info("ProveId", "our Answer '%s' ", Answer);

   // Send Answer encrypted
   if (SecureSend(fSocket, fRSAKey, Answer) == -1) {
      Warning("ProveId",
              "problems secure-sending Answer - may trigger problems in proofing Id ");
   }
   SafeDelete(Question);
#endif
   return 0;
}
