// @(#)root/net:$Name:  $:$Id: TAuthenticate.cxx,v 1.7 2001/01/25 14:06:04 rdm Exp $
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

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#ifndef R__LYNXOS
#include <sys/stat.h>
#endif
#include <errno.h>

#include "TAuthenticate.h"
#include "TNetFile.h"
#include "TSocket.h"
#include "TSystem.h"
#include "TError.h"
#include "Getline.h"


TString      TAuthenticate::fgUser;
TString      TAuthenticate::fgPasswd;
SecureAuth_t TAuthenticate::fgSecAuthHook;


ClassImp(TAuthenticate)

//______________________________________________________________________________
TAuthenticate::TAuthenticate(TSocket *sock, const char *remote,
                             const char *proto, Int_t security)
{
   // Create authentication object.

   fSocket   = sock;
   fRemote   = remote;
   fProtocol = proto;
   fSecurity = (ESecurity) security;
}

//______________________________________________________________________________
Bool_t TAuthenticate::Authenticate()
{
   // Authenticate to remote rootd or proofd server. Return kTRUE if
   // authentication succeeded.

   Bool_t result = kFALSE;

   TString user;
   TString passwd;

   // Get user and passwd set via static functions SetUser and SetPasswd.
   if (fgUser != "")
      user = fgUser;
   if (fgPasswd != "")
      passwd = fgPasswd;

   // Check ~/.rootnetrc and ~/.netrc files if user was not set via
   // the static SetUser() method.
   if (user == "")
      CheckNetrc(user, passwd);

   // If user also not set via  ~/.rootnetrc or ~/.netrc ask user.
   if (user == "") {
      user = PromptUser(fRemote);
      if (user == "")
         Error("Authenticate", "user name not set");
   }

   fUser   = user;
   fPasswd = passwd;

   // if not anonymous login try to use secure authentication
   if (fSecurity == kSRP && fUser != "anonymous" && fUser != "rootd") {
      if (!fgSecAuthHook) {
         char *p;
#ifdef ROOTLIBDIR
         TString lib = TString(ROOTLIBDIR) + "/libSRPAuth";
#else
         TString lib = TString(gRootDir) + "/lib/libSRPAuth";
#endif
         if ((p = gSystem->DynamicPathName(lib, kTRUE))) {
            delete [] p;
            gSystem->Load(lib);
         }
      }
      if (fgSecAuthHook) {
         Int_t st = (*fgSecAuthHook)(fSocket, user, passwd, fRemote);
         if (st == 0)
            return kFALSE;
         if (st == 1)
            return kTRUE;
         if (st == 2)
            Warning("Authenticate", "remote %s does not support secure authentication",
                    fProtocol.Data());
      } else {
         Error("Authenticate", "no support for secure authentication available");
         return kFALSE;
      }
   }

   fSocket->Send(user, kROOTD_USER);

   Int_t stat, kind;

   fSocket->Recv(stat, kind);

   if (kind == kROOTD_ERR) {
      AuthError("Authenticate", stat);
      return result;
   }
   if (kind == kROOTD_AUTH && stat == 1) {
      result = kTRUE;
      return result;
   }

badpass:
   if (passwd == "") {
      passwd = PromptPasswd();
      if (passwd == "")
         Error("Authenticate", "password not set");
   }

   if (fUser == "anonymous" || fUser == "rootd") {
      if (!passwd.Contains("@")) {
         Warning("Authenticate", "please use passwd of form: user@host.do.main");
         passwd = "";
         goto badpass;
      }
   }

   fPasswd = passwd;

   if (passwd != "") {
      for (int i = 0; i < passwd.Length(); i++) {
         char inv = ~passwd(i);
         passwd.Replace(i, 1, inv);
      }
   }

   fSocket->Send(passwd, kROOTD_PASS);

   fSocket->Recv(stat, kind);
   if (kind == kROOTD_ERR)
      AuthError("Authenticate", stat);
   if (kind == kROOTD_AUTH && stat == 1)
      result = kTRUE;

   return result;
}

//______________________________________________________________________________
Bool_t TAuthenticate::CheckNetrc(TString &user, TString &passwd)
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

   Bool_t  result = kFALSE;
   Bool_t  first  = kTRUE;
   TString remote = fRemote;

   user = passwd = "";

   char *net = gSystem->ConcatFileName(gSystem->HomeDirectory(), ".rootnetrc");

again:
#ifdef WIN32
   // Since Win32 does not have proper protections use file always
   FILE *fd1;
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
         if (first) {
            TInetAddress addr = gSystem->GetHostByName(fRemote);
            if (addr.IsValid()) {
               remote = addr.GetHostName();
               if (remote == "UnNamedHost")
                  remote = addr.GetHostAddress();
            }
         }
         FILE *fd = fopen(net, "r");
         char line[256];
         while (fgets(line, sizeof(line), fd) != 0) {
            if (line[0] == '#') continue;
            char word[6][64];
            int nword = sscanf(line, "%s %s %s %s %s %s", word[0], word[1],
                               word[2], word[3], word[4], word[5]);
            if (nword != 6) continue;
            if (fSecurity == kSRP    && strcmp(word[0], "secure"))  continue;
            if (fSecurity == kNormal && strcmp(word[0], "machine")) continue;
            if (strcmp(word[2], "login"))    continue;
            if (strcmp(word[4], "password")) continue;

            if (!strcmp(word[1], remote)) {
               user   = word[3];
               passwd = word[5];
               result = kTRUE;
               break;
            }
         }
         fclose(fd);
      } else
         Warning("CheckNetrc", "file %s exists but has not 0600 permission", net);
   }
   delete [] net;

   if (first && fSecurity == kNormal && !result) {
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

   const char *user = gSystem->Getenv("USER");
#ifdef R__WIN32
   if (!user)
      user = gSystem->Getenv("USERNAME");
#endif
   char *usr = Getline(Form("Name (%s:%s): ", remote, user));
   if (usr[0]) {
      usr[strlen(usr)-1] = 0;   // get rid of \n
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

   Gl_config("noecho", 1);
   char *pw = Getline((char*)prompt);
   Gl_config("noecho", 0);
   if (pw[0]) {
      pw[strlen(pw)-1] = 0;   // get rid of \n
      return StrDup(pw);
   }
   return 0;
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
