// @(#)root/auth:$Id$
// Author: G. Ganis, Nov 2006

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef WIN32
#   include <unistd.h>
#else
#   define ssize_t int
#   include <io.h>
#   include <sys/types.h>
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAFS                                                                 //
//                                                                      //
// Utility class to acquire and handle an AFS tokens.                   //
// Interface to libTAFS.so.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "AFSAuth.h"
#include "TAFS.h"
#include "TError.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TString.h"
#include "TSystem.h"
#include "Varargs.h"
#include "Getline.h"

Bool_t          TAFS::fgUsePwdDialog = kTRUE;
TPluginHandler *TAFS::fgPasswdDialog = (TPluginHandler *)(-1);

ClassImp(TAFS)

// Hook to the constructor. This is needed to avoid using the plugin manager
// which may create problems in multi-threaded environments.
extern "C" {
   TAFS *GetTAFS(const char *f, const char *u, Int_t lf) {
      // Create and instance and return it only if valid
      TAFS *afs = new TAFS(f, u, lf);
      if (afs->Verify())
         return afs;
      delete afs;
      return 0;
   }
}

//________________________________________________________________________
TAFS::TAFS(const char *fpw, const char *user, int life)
{
   // Constructor: get AFS token for usr using credentials from file 'fpw'.
   // If 'usr' is undefined the current user is used.
   // If 'fpw' is undefined the caller is prompt for a password.

   // Used to test validity
   fToken = 0;

   // Determine the user
   TString usr = (user && strlen(user) > 0) ? user : "";
   if (usr.IsNull()) {
      UserGroup_t *u = gSystem->GetUserInfo();
      if (u) {
         usr = (const char *) u->fUser;
         delete u;
      } else {
         Info("TAFS","user undefined");
         return;
      }
   }

   // Find credentials
   char *pw = 0;
   Int_t pwlen = 0;
   if (fpw) {
      // Reading credentials from file
      struct stat st;
      if (!stat(fpw, &st)) {
         pwlen = st.st_size;
         // Open the file for reading
         Int_t fd = open(fpw, O_RDONLY);
         if (fd > 0) {
            pw = new char[pwlen];
            if (read(fd, pw, pwlen) != pwlen) {
               delete [] pw;
               pw = 0;
               pwlen = 0;
            }
         }
      }
      // Notify failure
      if (!pw) {
         Info("TAFS","could not read credentials from %s", fpw);
      }
   }

   // Prompt for credentials if not yet found
   if (!pw) {

      TString prompt = Form("AFS password for %s@%s", usr.Data(), AFSLocalCell());

      // Init the dialog box, if needed
      if (fgUsePwdDialog) {
         if (fgPasswdDialog == (TPluginHandler *)(-1)) {
            if (!gROOT->IsBatch()) {
               if ((fgPasswdDialog =
                    gROOT->GetPluginManager()->FindHandler("TGPasswdDialog")))
                  if (fgPasswdDialog->LoadPlugin() == -1) {
                     fgPasswdDialog = 0;
                     Warning("TAFS",
                             "could not load plugin for the password dialog box");
                  }
            } else
               fgPasswdDialog = 0;
         }
      } else {
         fgPasswdDialog = 0;
      }

      // Get the password now
      char buf[128];
      pw = buf;
      if (fgPasswdDialog) {
         // Use graphic dialog
         fgPasswdDialog->ExecPlugin(3, prompt.Data(), buf, 128);
         // Wait until the user is done
         while (gROOT->IsInterrupted())
            gSystem->DispatchOneEvent(kFALSE);
      } else {
         if (isatty(0) != 0 && isatty(1) != 0) {
            Gl_config("noecho", 1);
            pw = Getline((char *) prompt.Data());
            Gl_config("noecho", 0);
         } else {
            Warning("TAFS", "not tty: cannot prompt for passwd: failure");
            pw[0] = 0;
         }
      }

      // Final checks
      if (pw[0]) {
         if (pw[strlen(pw)-1] == '\n')
            pw[strlen(pw) - 1] = 0;   // get rid of \n
      }
   }

   // Now get the token
   char *emsg;
   if (!(fToken = GetAFSToken(usr, pw, pwlen, life, &emsg))) {
      Info("TAFS", "token acquisition failed: %s", emsg);
      return;
   }

   // Success
   return;
}

//________________________________________________________________________
TAFS::~TAFS()
{
   // Destructor

   if (fToken)
      DeleteAFSToken(fToken);
}

//________________________________________________________________________
Int_t TAFS::Verify()
{
   // Return seconds to expiration (negative means expired)

   return (fToken ? VerifyAFSToken(fToken) : -1);
}

//________________________________________________________________________
void TAFS::SetUsePwdDialog(Bool_t on)
{
   // Switch on/off usage of password dialog box

   fgUsePwdDialog = on;
}
