// @(#)root/proofd:$Name:  $:$Id: XrdROOT.cxx,v 1.1 2007/06/12 13:51:04 ganis Exp $
// Author: Gerardo Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdROOT                                                              //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Class describing a ROOT version                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdProofdPlatform.h"

#include "XrdROOT.h"
#include "XrdProofdProtocol.h"
#include "XrdSys/XrdSysPriv.hh"

// Tracing
#include "XrdProofdTrace.h"
static const char *gTraceID = " ";
extern XrdOucTrace *XrdProofdTrace;
#define TRACEID gTraceID

//__________________________________________________________________________
XrdROOT::XrdROOT(const char *dir, const char *tag)
{
   // Constructor: validates 'dir', gets the version and defines the tag.

   fStatus = -1;
   fSrvProtVers = -1;

   // 'dir' must make sense
   if (!dir || strlen(dir) <= 0)
      return;
   if (tag && strlen(tag) > 0) {
      fExport = tag;
      fExport += " "; fExport += dir;
   } else
      fExport += dir;

   // The path should exist and be statable
   struct stat st;
   if (stat(dir, &st) == -1) {
      XPDERR("XrdROOT: unable to stat path "<<dir);
      return;
   }
   // ... and be a directory
   if (!S_ISDIR(st.st_mode)) {
      XPDERR("XrdROOT: path "<<dir<<" is not a directory");
      return;
   }
   fDir = dir;

   // Get the version
   XrdOucString version;
   if (GetROOTVersion(dir, version) == -1) {
      XPDERR("XrdROOT: unable to extract ROOT version from path "<<dir);
      return;
   }

   // Default tag is the version
   fTag = (!tag || strlen(tag) <= 0) ? version : tag;

   // The application to be run
   fPrgmSrv = dir;
   fPrgmSrv += "/bin/proofserv";

   // Export string
   fExport = fTag;
   fExport += " "; fExport += version;
   fExport += " "; fExport += dir;

   // First step OK
   fStatus = 0;
}

//__________________________________________________________________________
bool XrdROOT::Validate()
{
   // Validates 'dir' (temporarly stored in fExport) and makes sure the
   // associated 'proofserv' can be started

   if (IsInvalid()) {
      // Cannot be validated
      XPDERR("XrdROOT::Validate: invalid instance - cannot be validated");
      return 0;
   }

   // Validate it, retrieving at the same time the PROOF protocol run by it
   if (ValidatePrgmSrv() == -1) {
      XPDERR("XrdROOT:Validate: unable to validate "<< fPrgmSrv);
      return 0;
   }

   // Finalize export string
   fExport += " "; fExport += (int)fSrvProtVers;

   // The instance has been validated
   fStatus = 1;
   return 1;
}

//__________________________________________________________________________
int XrdROOT::GetROOTVersion(const char *dir, XrdOucString &version)
{
   // Get ROOT version associated with 'dir'.

   int rc = -1;

   XrdOucString versfile = dir;
   versfile += "/include/RVersion.h";

   // Open file
   FILE *fv = fopen(versfile.c_str(), "r");
   if (!fv) {
      XPDERR("XrdROOT::GetROOTVersion: unable to open "<<versfile);
      return rc;
   }

   // Read the file
   char line[1024];
   while (fgets(line, sizeof(line), fv)) {
      char *pv = (char *) strstr(line, "ROOT_RELEASE");
      if (pv) {
         if (line[strlen(line)-1] == '\n')
            line[strlen(line)-1] = 0;
         pv += strlen("ROOT_RELEASE") + 1;
         version = pv;
         version.replace("\"","");
         rc = 0;
         break;
      }
   }

   // Close the file
   fclose(fv);

   // Done
   return rc;
}

//__________________________________________________________________________
int XrdROOT::ValidatePrgmSrv()
{
   // Start a trial server application to test forking and get the version
   // of the protocol run by the PROOF server.
   // Return 0 if everything goes well, -1 in cse of any error.

   XPDPRT("XrdROOT::ValidatePrgmSrv: forking test and protocol retrieval");

   // Make sure the application path has been defined
   if (fPrgmSrv.length() <= 0) {
      XPDERR("XrdROOT::ValidatePrgmSrv: "
            " path to PROOF server application undefined - exit");
      return -1;
   }

   // Pipe to communicate the protocol number
   int fp[2];
   if (pipe(fp) != 0) {
      XPDERR("XrdROOT::ValidatePrgmSrv: unable to generate pipe for"
            " PROOT protocol number communication");
      return -1;
   }

   // Fork a test agent process to handle this session
   TRACE(FORK,"XrdROOT::ValidatePrgmSrv: forking external proofsrv");
   int pid = -1;
   if (!(pid = XrdProofdProtocol::fgSched->Fork("proofsrv"))) {

      char *argvv[5] = {0};

      // start server
      argvv[0] = (char *)fPrgmSrv.c_str();
      argvv[1] = (char *)"proofserv";
      argvv[2] = (char *)"xpd";
      argvv[3] = (char *)"test";
      argvv[4] = 0;

      // Set basic environment for proofserv
      if (XrdProofdProtocol::SetProofServEnv(this) != 0) {
         TRACE(XERR, "XrdROOT::ValidatePrgmSrv:"
                       " SetProofServEnv did not return OK - EXIT");
         exit(1);
      }

      // Set Open socket
      char *ev = new char[25];
      sprintf(ev, "ROOTOPENSOCK=%d", fp[1]);
      putenv(ev);

      // Prepare for execution: we need to acquire the identity of
      // a normal user
      if (!getuid()) {
         XrdProofUI ui;
         if (XrdProofdAux::GetUserInfo(geteuid(), ui) != 0) {
            TRACE(XERR, "XrdROOT::ValidatePrgmSrv:"
                          " could not get info for user-id: "<<geteuid());
            exit(1);
         }

         // acquire permanently target user privileges
         if (XrdSysPriv::ChangePerm((uid_t)ui.fUid, (gid_t)ui.fGid) != 0) {
            TRACE(XERR, "XrdROOT::ValidatePrgmSrv: can't acquire "<<
                          ui.fUser <<" identity");
            exit(1);
         }

      }

      // Run the program
      execv(fPrgmSrv.c_str(), argvv);

      // We should not be here!!!
      TRACE(XERR, "XrdROOT::ValidatePrgmSrv:"
                    " returned from execv: bad, bad sign !!!");
      exit(1);
   }

   // parent process
   if (pid < 0) {
      XPDERR("XrdROOT::ValidatePrgmSrv: forking failed - exit");
      close(fp[0]);
      close(fp[1]);
      return -1;
   }

   // now we wait for the callback to be (successfully) established
   TRACE(FORK, "XrdROOT::ValidatePrgmSrv:"
               " test server launched: wait for protocol ");

   // Read protocol
   int proto = -1;
   struct pollfd fds_r;
   fds_r.fd = fp[0];
   fds_r.events = POLLIN;
   int pollRet = 0;
   // We wait for 60 secs max (30 x 2000 millisecs): this is enough to
   // cover possible delays due to heavy load
   int ntry = 30;
   while (pollRet == 0 && ntry--) {
      while ((pollRet = poll(&fds_r, 1, 2000)) < 0 &&
             (errno == EINTR)) { }
      if (pollRet == 0)
         TRACE(DBG,"XrdROOT::ValidatePrgmSrv: "
                   "receiving PROOF server protocol number: waiting 2 s ...");
   }
   if (pollRet > 0) {
      if (read(fp[0], &proto, sizeof(proto)) != sizeof(proto)) {
         XPDERR("ValidatePrgmSrv: "
               " XrdROOT::problems receiving PROOF server protocol number");
         return -1;
      }
   } else {
      if (pollRet == 0) {
         XPDERR("XrdROOT::ValidatePrgmSrv: "
               " timed-out receiving PROOF server protocol number");
      } else {
         XPDERR("XrdROOT::ValidatePrgmSrv: "
               " failed to receive PROOF server protocol number");
      }
      return -1;
   }

   // Record protocol
   fSrvProtVers = (kXR_int16) ntohl(proto);

   // Cleanup
   close(fp[0]);
   close(fp[1]);

   // We are done
   return 0;
}
