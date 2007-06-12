// @(#)root/proofd:$Name:  $:$Id:  $
// Author: G. Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdClient                                                      //
//                                                                      //
// Author: G. Ganis, CERN, 2007                                         //
//                                                                      //
// Auxiliary class describing a PROOF client.                           //
// Used by XrdProofdProtocol.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "XrdNet/XrdNet.hh"

#include "XrdProofdAux.h"
#include "XrdProofdClient.h"
#include "XrdProofdProtocol.h"
#include "XrdProofGroup.h"

#include "XrdProofdTrace.h"
static const char *gTraceID = " ";
extern XrdOucTrace *XrdProofdTrace;
#define TRACEID gTraceID

//__________________________________________________________________________
XrdProofdClient::XrdProofdClient(const char *cid,
                                 short int clientvers, XrdProofUI ui)
{
   // Constructor

   fClientID = (cid) ? strdup(cid) : 0;
   fClientVers = clientvers;
   fProofServs.reserve(10);
   fClients.reserve(10);
   fUI = ui;
   fUNIXSock = 0;
   fUNIXSockPath = 0;
   fUNIXSockSaved = 0;
   fROOT = 0;
   fGroup = 0;
   fWorkerProofServ = 0;
   fMasterProofServ = 0;
   fIsValid = 0;
}

//__________________________________________________________________________
XrdProofdClient::~XrdProofdClient()
{
   // Destructor

   SafeFree(fClientID);

   // Unix socket
   SafeDel(fUNIXSock);
   SafeDelArray(fUNIXSockPath);
}

//__________________________________________________________________________
bool XrdProofdClient::Match(const char *id, const char *grp)
{
   // return TRUE if this instance matches 'id' (and 'grp', if defined) 

   bool rc = (id && !strcmp(id, fClientID)) ? 1 : 0;
   if (rc && grp && strlen(grp) > 0)
      rc = (fGroup && !strcmp(grp, fGroup->Name())) ? 1 : 0;

   return rc;
}

//__________________________________________________________________________
int XrdProofdClient::GetClientID(XrdProofdProtocol *p)
{
   // Get next free client ID. If none is found, increase the vector size
   // and get the first new one

   XrdOucMutexHelper mh(fMutex);

   int ic = 0;
   // Search for free places in the existing vector
   for (ic = 0; ic < (int)fClients.size() ; ic++) {
      if (!fClients[ic]) {
         fClients[ic] = p;
         return ic;
      }
   }

   // We need to resize (double it)
   if (ic >= (int)fClients.capacity())
      fClients.reserve(2*fClients.capacity());

   // Fill in new element
   fClients.push_back(p);

   TRACE(DBG, "XrdProofdClient::GetClientID: size: "<<fClients.size());

   // We are done
   return ic;
}

//__________________________________________________________________________
int XrdProofdClient::CreateUNIXSock(XrdOucError *edest, char *tmpdir)
{
   // Create UNIX socket for internal connections

   TRACE(ACT, "CreateUNIXSock: enter");

   // Make sure we do not have already a socket
   if (fUNIXSock && fUNIXSockPath) {
       TRACE(DBG,"CreateUNIXSock: UNIX socket exists already! (" <<
             fUNIXSockPath<<")");
       return 0;
   }

   // Make sure we do not have inconsistencies
   if (fUNIXSock || fUNIXSockPath) {
       TRACE(XERR,"CreateUNIXSock: inconsistent values: corruption? (sock: " <<
                 fUNIXSock<<", path: "<< fUNIXSockPath);
       return -1;
   }

   // Inputs must make sense
   if (!edest || !tmpdir) {
       TRACE(XERR,"CreateUNIXSock: invalid inputs: edest: " <<
                 (int *)edest <<", tmpdir: "<< (int *)tmpdir);
       return -1;
   }

   // Create socket
   fUNIXSock = new XrdNet(edest);

   // Create path
   fUNIXSockPath = new char[strlen(tmpdir)+strlen("/xpdsock_XXXXXX")+2];
   sprintf(fUNIXSockPath,"%s/xpdsock_XXXXXX", tmpdir);
   int fd = mkstemp(fUNIXSockPath);
   if (fd > -1) {
      close(fd);
      if (fUNIXSock->Bind(fUNIXSockPath)) {
         TRACE(XERR,"CreateUNIXSock: warning:"
                   " problems binding to UNIX socket; path: " <<fUNIXSockPath);
         return -1;
      } else
         TRACE(DBG, "CreateUNIXSock: path for UNIX for socket is " <<fUNIXSockPath);
   } else {
      TRACE(XERR,"CreateUNIXSock: unable to generate unique"
            " path for UNIX socket; tried path " << fUNIXSockPath);
      return -1;
   }

   // We are done
   return 0;
}

//__________________________________________________________________________
void XrdProofdClient::SaveUNIXPath()
{
   // Save UNIX path in <SandBox>/.unixpath

   TRACE(ACT,"SaveUNIXPath: enter: saved? "<<fUNIXSockSaved);

   // Make sure we do not have already a socket
   if (fUNIXSockSaved) {
      TRACE(DBG,"SaveUNIXPath: UNIX path saved already");
      return;
   }

   // Make sure we do not have already a socket
   if (!fUNIXSockPath) {
       TRACE(XERR,"SaveUNIXPath: UNIX path undefined!");
       return;
   }

   // File name
   XrdOucString fn = fUI.fWorkDir;
   fn += "/.unixpath";

   // Open the file for appending
   FILE *fup = fopen(fn.c_str(), "a+");
   if (!fup) {
      TRACE(XERR, "SaveUNIXPath: cannot open file "<<fn<<
            " for appending (errno: "<<errno<<")");
      return;
   }

   // Lock the file
   lseek(fileno(fup), 0, SEEK_SET);
   if (lockf(fileno(fup), F_LOCK, 0) == -1) {
      TRACE(XERR, "SaveUNIXPath: cannot lock file "<<fn<<
            " (errno: "<<errno<<")");
      fclose(fup);
      return;
   }

   // Read content, if any
   char ln[1024], path[1024];
   int pid = -1;
   std::list<XrdOucString *> actln;
   while (fgets(ln, sizeof(ln), fup)) {
      // Get rid of '\n'
      if (ln[strlen(ln)-1] == '\n')
         ln[strlen(ln)-1] = '\0';
      // Skip empty or comment lines
      if (strlen(ln) <= 0 || ln[0] == '#')
         continue;
      // Get PID and path
      sscanf(ln, "%d %s", &pid, path);
      // Verify if still running
      int vrc = -1;
      if ((vrc = XrdProofdProtocol::VerifyProcessByID(pid, "xrootd")) != 0) {
         // Still there
         actln.push_back(new XrdOucString(ln));
      } else if (vrc == 0) {
         // Not running: remove the socket path
         TRACE(DBG, "SaveUNIXPath: unlinking socket path "<< path);
         if (unlink(path) != 0 && errno != ENOENT) {
            TRACE(XERR, "SaveUNIXPath: problems unlinking socket path "<< path<<
                    " (errno: "<<errno<<")");
         }
      }
   }

   // Truncate the file
   if (ftruncate(fileno(fup), 0) == -1) {
      TRACE(XERR, "SaveUNIXPath: cannot truncate file "<<fn<<
                 " (errno: "<<errno<<")");
      lseek(fileno(fup), 0, SEEK_SET);
      lockf(fileno(fup), F_ULOCK, 0);
      fclose(fup);
      return;
   }

   // If active sockets still exist, write out new composition
   if (actln.size() > 0) {
      std::list<XrdOucString *>::iterator i;
      for (i = actln.begin(); i != actln.end(); ++i) {
         fprintf(fup, "%s\n", (*i)->c_str());
         delete (*i);
      }
   }

   // Append the path and our process ID
   lseek(fileno(fup), 0, SEEK_END);
   fprintf(fup, "%d %s\n", getppid(), fUNIXSockPath);

   // Unlock the file
   lseek(fileno(fup), 0, SEEK_SET);
   if (lockf(fileno(fup), F_ULOCK, 0) == -1)
      TRACE(XERR, "SaveUNIXPath: cannot unlock file "<<fn<<
                 " (errno: "<<errno<<")");

   // Close the file
   fclose(fup);

   // Path saved
   fUNIXSockSaved = 1;
}
