// @(#)root/net:$Name:  $:$Id: TAuthenticate.cxx,v 1.1 2000/11/27 10:35:05 rdm Exp $
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

R__EXTERN const char *kRootdErrStr[];

char *TAuthenticate::fgUser;
char *TAuthenticate::fgPasswd;
SecureAuth_t TAuthenticate::fgSecAuthHook;


ClassImp(TAuthenticate)

//______________________________________________________________________________
TAuthenticate::TAuthenticate(TSocket *sock, const char *proto,
                             const char *remote)
{
   // Create authentication object.

   fSocket   = sock;
   fProtocol = proto;
   fRemote   = remote;
}

//______________________________________________________________________________
Bool_t TAuthenticate::Authenticate(TString &usr)
{
   // Authenticate to remote rootd server. Return kTRUE if authentication
   // succeeded.

   Bool_t result = kFALSE;

   char *user   = 0;
   char *passwd = 0;

   // Get user and passwd set via static functions SetUser and SetPasswd.
   if (fgUser)
      user = StrDup(fgUser);
   if (fgPasswd)
      passwd = StrDup(fgPasswd);

   // Check ~/.netrc file if user was not set via the static SetUser() method.
   if (!user)
      CheckNetrc(user, passwd);

   // If user also not set via ~/.netrc ask user.
   if (!user) {
      user = GetUser(fRemote);
      if (!user)
         Error("Authenticate", "user name not set");
   }

   fUser = user;
   usr   = user;

   // if not anonymous login try to use secure authentication
   if ((fProtocol == "roots" || fProtocol == "proofs") &&
       fUser != "anonymous" && fUser != "rootd") {
      if (!fgSecAuthHook) {
         char *p;
         char *lib = Form("%s/lib/libSRPAuth", gRootDir);
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
                    fProtocol.BeginsWith("root") ? "rootd" : "proofd");
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
      goto out;
   }
   if (kind == kROOTD_AUTH && stat == 1) {
      result = kTRUE;
      goto out;
   }

badpass:
   if (!passwd) {
      passwd = GetPasswd();
      if (!passwd)
         Error("Authenticate", "password not set");
   }

   if (fUser == "anonymous" || fUser == "rootd") {
      if (!strchr(passwd, '@')) {
         Warning("Authenticate", "please use passwd of form: user@host.do.main");
         delete [] passwd;
         passwd = 0;
         goto badpass;
      }
   }

   if (passwd) {
      int n = strlen(passwd);
      for (int i = 0; i < n; i++)
         passwd[i] = ~passwd[i];
   }

   fSocket->Send(passwd, kROOTD_PASS);

   fSocket->Recv(stat, kind);
   if (kind == kROOTD_ERR)
      AuthError("Authenticate", stat);
   if (kind == kROOTD_AUTH && stat == 1)
      result = kTRUE;

out:
   delete [] user;
   delete [] passwd;

   return result;
}

//______________________________________________________________________________
Bool_t TAuthenticate::CheckNetrc(char *&user, char *&passwd)
{
   // Try to get user name and passwd from the ~/.netrc file.
   // This file will only be used when its access mask is 0600.
   // Returns kTRUE if user and passwd were found for the machine
   // specified in the URL. User and passwd must be deleted by
   // the caller. If kFALSE, user and passwd are 0.

   Bool_t result = kFALSE;
   user = passwd = 0;

   char *net = gSystem->ConcatFileName(gSystem->HomeDirectory(), ".netrc");

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
         FILE *fd = fopen(net, "r");
         char line[256];
         while (fgets(line, sizeof(line), fd) != 0) {
            if (line[0] == '#') continue;
            char word[6][64];
            int nword = sscanf(line, "%s %s %s %s %s %s", word[0], word[1],
                               word[2], word[3], word[4], word[5]);
            if (nword != 6) continue;
            if (strcmp(word[0], "machine"))  continue;
            if (strcmp(word[2], "login"))    continue;
            if (strcmp(word[4], "password")) continue;

            if (!strcmp(word[1], fRemote)) {
               user   = StrDup(word[3]);
               passwd = StrDup(word[5]);
               result = kTRUE;
               break;
            }
         }
         fclose(fd);
      }
   }
   delete [] net;

   return result;
}

//______________________________________________________________________________
char *TAuthenticate::GetUser(const char *remote)
{
   // Static method to get user name to be used for authentication to rootd
   // or proofd. User is asked to type user name.
   // Returns user name (which must be deleted by caller) or 0.

   char *usr = Getline(Form("Name (%s:%s): ", remote, gSystem->Getenv("USER")));
   if (usr[0]) {
      usr[strlen(usr)-1] = 0;   // get rid of \n
      if (strlen(usr))
         return StrDup(usr);
      else
         return StrDup(gSystem->Getenv("USER"));
   }
   return 0;
}

//______________________________________________________________________________
char *TAuthenticate::GetPasswd(const char *prompt)
{
   // Static method to get passwd to be used for authentication to rootd
   // or proofd. Uses non-echoing command line to get passwd.
   // Returns passwd (which must de deleted by caller) or 0.

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

   ::Error(where, kRootdErrStr[err]);
}

//______________________________________________________________________________
void TAuthenticate::SetUser(const char *user)
{
   // Set user name to be used for authentication to rootd.

   if (fgUser)
      delete [] fgUser;

   if (!user || !user[0])
      fgUser = 0;
   else
      fgUser = StrDup(user);
}

//______________________________________________________________________________
void TAuthenticate::SetPasswd(const char *passwd)
{
   // Set passwd to be used for authentication to rootd.

   if (fgPasswd)
      delete [] fgPasswd;

   if (!passwd || !passwd[0])
      fgPasswd = 0;
   else
      fgPasswd = StrDup(passwd);
}

//______________________________________________________________________________
void TAuthenticate::SetSecureAuthHook(SecureAuth_t func)
{
   // Set secure authorization function. Automatically called when libSRPAuth
   // is loaded.

   fgSecAuthHook = func;
}
