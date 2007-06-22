// @(#)root/proofd:$Name:  $:$Id: XrdProofdClient.cxx,v 1.3 2007/06/21 11:31:40 ganis Exp $
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
#include "XrdProofdPlatform.h"
#include "XrdProofdProtocol.h"
#include "XrdProofGroup.h"

#include "XrdProofdTrace.h"
static const char *gTraceID = " ";
extern XrdOucTrace *XrdProofdTrace;
#define TRACEID gTraceID

int XrdProofdClient::fgMaxOldLogs = XPC_DEFMAXOLDLOGS;

//__________________________________________________________________________
bool XpdSessionTagComp(XrdOucString *&lhs, XrdOucString *&rhs)
{
   // Compare times from session tag strings

   if (!lhs || !rhs)
      return 1;

   // Left hand side
   XrdOucString ll(*lhs);
   ll.erase(ll.rfind('-'));
   ll.erase(0, ll.rfind('-')+1);
   int tl = strtol(ll.c_str(), 0, 10);

   // Right hand side
   XrdOucString rr(*rhs);
   rr.erase(rr.rfind('-'));
   rr.erase(0, rr.rfind('-')+1);
   int tr = strtol(rr.c_str(), 0, 10);

   // Done
   return ((tl < tr) ? 0 : 1);
}

#if defined(__sun)
//__________________________________________________________________________
static void Sort(std::list<XrdOucString *> *lst)
{
   // Sort ascendingly the list.
   // Function used on Solaris where std::list::sort() does not support an
   // alternative comparison algorithm.

   // Check argument
   if (!lst)
      return;

   // If empty or just one element, nothing to do
   if (lst->size() < 2)
      return;

   // Fill a temp array with the current status
   XrdOucString **ta = new XrdOucString *[lst->size()];
   std::list<XrdOucString *>::iterator i;
   int n = 0;
   for (i = lst->begin(); i != lst->end(); ++i)
      ta[n++] = *i;

   // Now start the loops
   XrdOucString *tmp = 0;
   bool notyet = 1;
   int jold = 0;
   while (notyet) {
      int j = jold;
      while (j < n - 1) {
         if (XpdSessionTagComp(ta[j], ta[j+1]))
            break;
         j++;
      }
      if (j >= n - 1) {
         notyet = 0;
      } else {
         jold = j + 1;
         XPDSWAP(ta[j], ta[j+1], tmp);
         int k = j;
         while (k > 0) {
            if (!XpdSessionTagComp(ta[k], ta[k-1])) {
               XPDSWAP(ta[k], ta[k-1], tmp);
            } else {
               break;
            }
            k--;
         }
      }
   }

   // Empty the original list
   lst->clear();

   // Fill it again
   while (n--)
      lst->push_back(ta[n]);

   // Clean up
   delete[] ta;
}
#endif

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
void XrdProofdClient::CountSession(int n, bool worker)
{
   // Count session of type srvtype

   if (worker)
      fWorkerProofServ += n;
   else
      fMasterProofServ += n;
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
//______________________________________________________________________________
int XrdProofdClient::GuessTag(XrdOucString &tag, int ridx)
{
   // Guess session tag completing 'tag' (typically "-<pid>") by scanning the
   // active session file or the session dir.
   // In case of success, tag is filled with the full tag and 0 is returned.
   // In case of failure, -1 is returned.

   TRACE(ACT, "GuessTag: enter: tag: "<<tag);

   bool found = 0;
   bool last = (tag == "last") ? 1 : 0;

   if (!last && tag.length() > 0) {
      // Scan the sessions file
      XrdOucString fn = Workdir();
      fn += "/.sessions";

      // Open the file for reading
      FILE *fact = fopen(fn.c_str(), "a+");
      if (fact) {
         // Lock the file
         if (lockf(fileno(fact), F_LOCK, 0) == 0) {
            // Read content, if already existing
            char ln[1024];
            while (fgets(ln, sizeof(ln), fact)) {
               // Get rid of '\n'
               if (ln[strlen(ln)-1] == '\n')
                  ln[strlen(ln)-1] = '\0';
               // Skip empty or comment lines
               if (strlen(ln) <= 0 || ln[0] == '#')
                  continue;
               // Count if not the one we want to remove
               if (!strstr(ln, tag.c_str())) {
                  tag = ln;
                  found = 1;
                  break;
               }
            }
            // Unlock the file
            lseek(fileno(fact), 0, SEEK_SET);
            if (lockf(fileno(fact), F_ULOCK, 0) == -1)
               TRACE(DBG, "GuessTag: cannot unlock file "<<fn<<" ; fact: "<<fact<<
                          ", fd: "<< fileno(fact) << " (errno: "<<errno<<")");

         } else {
            TRACE(DBG, "GuessTag: cannot lock file: "<<fn<<" ; fact: "<<fact<<
                       ", fd: "<< fileno(fact) << " (errno: "<<errno<<")");
         }
         // Close the file
         fclose(fact);

      } else {
         TRACE(DBG, "GuessTag: cannot open file "<<fn<<
                    " for reading (errno: "<<errno<<")");
      }
   }

   if (!found) {

      // Search the tag in the dirs
      std::list<XrdOucString *> staglst;
      int rc = GetSessionDirs(3, &staglst, &tag);
      if (rc < 0) {
         TRACE(XERR, "GuessTag: cannot scan dir "<<Workdir());
         return -1;
      }
      found = (rc == 1) ? 1 : 0;

      if (!found) {
         // Take last one, if required
         if (last) {
            tag = staglst.front()->c_str();
            found = 1;
         } else {
            if (ridx < 0) {
               int itag = ridx;
               // Reiterate back
               std::list<XrdOucString *>::iterator i;
               for (i = staglst.end(); i != staglst.begin(); --i) {
                  if (itag == 0) {
                     tag = (*i)->c_str();
                     found = 1;
                     break;
                  }
                  itag++;
               }
            }
         }
      }
      // Cleanup
      staglst.clear();
      // Correct the tag
      if (found) {
         tag.replace("session-", "");
      } else {
         TRACE(DBG, "GuessTag: tag "<<tag<<" not found in dir");
      }
   }

   // We are done
   return ((found) ? 0 : -1);
}

//______________________________________________________________________________
int XrdProofdClient::AddNewSession(const char *tag)
{
   // Record entry for new proofserv session tagged 'tag' in the active
   // sessions file (<SandBox>/.sessions). The file is created if needed.
   // Return 0 on success, -1 on error. 


   // Check inputs
   if (!tag) {
      XPDPRT("XrdProofdProtocol::AddNewSession: invalid input");
      return -1;
   }
   TRACE(ACT, "AddNewSession: enter: tag:"<<tag);

   // File name
   XrdOucString fn = Workdir();
   fn += "/.sessions";

   // Open the file for appending
   FILE *fact = fopen(fn.c_str(), "a+");
   if (!fact) {
      TRACE(XERR, "AddNewSession: cannot open file "<<fn<<
                 " for appending (errno: "<<errno<<")");
      return -1;
   }

   // Lock the file
   lseek(fileno(fact), 0, SEEK_SET);
   if (lockf(fileno(fact), F_LOCK, 0) == -1) {
      TRACE(XERR, "AddNewSession: cannot lock file "<<fn<<
                 " (errno: "<<errno<<")");
      fclose(fact);
      return -1;
   }

   bool writeout = 1;

   // Check if already there
   std::list<XrdOucString *> actln;
   char ln[1024];
   while (fgets(ln, sizeof(ln), fact)) {
      // Get rid of '\n'
      if (ln[strlen(ln)-1] == '\n')
         ln[strlen(ln)-1] = '\0';
      // Skip empty or comment lines
      if (strlen(ln) <= 0 || ln[0] == '#')
         continue;
      // Count if not the one we want to remove
      if (strstr(ln, tag))
         writeout = 0;
   }

   // Append the session unique tag
   if (writeout) {
      lseek(fileno(fact), 0, SEEK_END);
      fprintf(fact, "%s\n", tag);
   }

   // Unlock the file
   lseek(fileno(fact), 0, SEEK_SET);
   if (lockf(fileno(fact), F_ULOCK, 0) == -1)
      TRACE(XERR, "AddNewSession: cannot unlock file "<<fn<<
                 " (errno: "<<errno<<")");

   // Close the file
   fclose(fact);

   // We are done
   return 0;
}

//______________________________________________________________________________
int XrdProofdClient::MvOldSession(const char *tag)
{
   // Move record for tag from the active sessions file to the old 
   // sessions file (<SandBox>/.sessions). The active file is removed if
   // empty after the operation. The old sessions file is created if needed.
   // If the static fgMaxOldLogs > 0, logs for a fgMaxOldLogs number of sessions
   // are kept in the sandbox; working dirs for sessions in excess are removed.
   // By default logs for the last 10 sessions are kept; the limit can be changed
   // via the static method XrdProofdClient::SetMaxOldLogs.
   // Return 0 on success, -1 on error.

   char ln[1024];

   // Check inputs
   if (!tag) {
      TRACE(XERR, "MvOldSession: invalid input");
      return -1;
   }
   TRACE(ACT, "MvOldSession: enter: tag:"<<tag<<", maxold:"<<fgMaxOldLogs);

   // Update of the active file
   XrdOucString fna = Workdir();
   fna += "/.sessions";

   // Open the file
   FILE *fact = fopen(fna.c_str(), "a+");
   if (!fact) {
      TRACE(XERR, "MvOldSession: cannot open file "<<fna<<
                 " (errno: "<<errno<<")");
      return -1;
   }

   // Lock the file
   if (lockf(fileno(fact), F_LOCK, 0) == -1) {
      TRACE(XERR, "MvOldSession: cannot lock file "<<fna<<
                 " (errno: "<<errno<<")");
      fclose(fact);
      return -1;
   }

   // Read content, if already existing
   std::list<XrdOucString *> actln;
   while (fgets(ln, sizeof(ln), fact)) {
      // Get rid of '\n'
      if (ln[strlen(ln)-1] == '\n')
         ln[strlen(ln)-1] = '\0';
      // Skip empty or comment lines
      if (strlen(ln) <= 0 || ln[0] == '#')
         continue;
      // Count if not the one we want to remove
      if (!strstr(ln, tag))
         actln.push_back(new XrdOucString(ln));
   }

   // Truncate the file
   if (ftruncate(fileno(fact), 0) == -1) {
      TRACE(XERR, "MvOldSession: cannot truncate file "<<fna<<
                 " (errno: "<<errno<<")");
      lseek(fileno(fact), 0, SEEK_SET);
      lockf(fileno(fact), F_ULOCK, 0);
      fclose(fact);
      return -1;
   }

   // If active sessions still exist, write out new composition
   bool unlk = 1;
   if (actln.size() > 0) {
      unlk = 0;
      std::list<XrdOucString *>::iterator i;
      for (i = actln.begin(); i != actln.end(); ++i) {
         fprintf(fact, "%s\n", (*i)->c_str());
         delete (*i);
      }
   }

   // Unlock the file
   lseek(fileno(fact), 0, SEEK_SET);
   if (lockf(fileno(fact), F_ULOCK, 0) == -1)
      TRACE(XERR, "MvOldSession: cannot unlock file "<<fna<<
                  " (errno: "<<errno<<")");

   // Close the file
   fclose(fact);

   // Unlink the file if empty
   if (unlk)
      if (unlink(fna.c_str()) == -1) 
         TRACE(XERR, "MvOldSession: cannot unlink file "<<fna<<
                    " (errno: "<<errno<<")");

   // Flag the session as closed
   XrdOucString fterm = Workdir();
   fterm += (strstr(tag,"session-")) ? "/" : "/session-";
   fterm += tag;
   fterm += "/.terminated";
   // Create the file
   FILE *ft = fopen(fterm.c_str(), "w");
   if (!ft) {
      TRACE(XERR, "MvOldSession: cannot open file "<<fterm<<
                 " (errno: "<<errno<<")");
      return -1;
   }
   fclose(ft);

   // If a limit on the number of sessions dirs is set, apply it
   if (fgMaxOldLogs > 0) {

      // Get list of terminated session working dirs
      std::list<XrdOucString *> staglst;
      if (GetSessionDirs(2, &staglst) != 0) {
         TRACE(XERR, "MvOldSession: cannot get list of dirs ");
         return -1;
      }
      TRACE(DBG, "MvOldSession: number of working dirs: "<<staglst.size());

      std::list<XrdOucString *>::iterator i;
      for (i = staglst.begin(); i != staglst.end(); ++i) {
         TRACE(HDBG, "MvOldSession: found "<<(*i)->c_str());
      }

      // Remove the oldest, if needed
      while ((int)staglst.size() > fgMaxOldLogs) {
         XrdOucString *s = staglst.back();
         if (s) {
            TRACE(HDBG, "MvOldSession: removing "<<s->c_str());
            // Remove associated workdir
            XrdOucString rmcmd = "/bin/rm -rf ";
            rmcmd += Workdir();
            rmcmd += '/';
            rmcmd += s->c_str();
            system(rmcmd.c_str());
            // Delete the string
            delete s;
         }
         // Remove the last element
         staglst.pop_back();
      }

      // Clear the list
      staglst.clear();
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdClient::GetSessionDirs(int opt, std::list<XrdOucString *> *sdirs,
                                    XrdOucString *tag)
{
   // Scan the sandbox for sessions working dirs and return their
   // sorted (according to creation time, first is the newest) list
   // in 'sdirs'.
   // The option 'opt' may have 3 values:
   //    0        all working dirs are kept
   //    1        active sessions only
   //    2        terminated sessions only
   //    3        search entry containing 'tag' and fill tag with
   //             the full entry name; if defined, sdirs is filled
   // Returns -1 otherwise in case of failure.
   // In case of success returns 0 for opt < 3, 1 if found or 0 if not
   // found for opt == 3.

   // If unknown take all
   opt = (opt >= 0 && opt <= 3) ? opt : 0;

   // Check inputs
   if ((opt < 3 && !sdirs) || (opt == 3 && !tag)) {
      TRACE(XERR, "GetSessionDirs: invalid inputs");
      return -1;
   }

   TRACE(ACT, "GetSessionDirs: enter: opt: "<<opt<<", dir: "<<Workdir());

   // Open dir
   DIR *dir = opendir(Workdir());
   if (!dir) {
      TRACE(XERR, "GetSessionDirs: cannot open dir "<<Workdir()<<
            " (errno: "<<errno<<")");
      return -1;
   }

   // Scan the directory, and save the paths for terminated sessions
   // into a list
   bool found = 0;
   struct dirent *ent = 0;
   while ((ent = (struct dirent *)readdir(dir))) {
      if (!strncmp(ent->d_name, "session-", 8)) {
         bool keep = 1;
         if (opt == 3 && tag->length() > 0) {
            if (strstr(ent->d_name, tag->c_str())) {
               *tag = ent->d_name;
               found = 1;
            }
         } else {
            if (opt > 0) {
               XrdOucString fterm(Workdir());
               fterm += '/';
               fterm += ent->d_name;
               fterm += "/.terminated";
               int rc = access(fterm.c_str(), F_OK);
               if ((opt == 1 && rc == 0) || (opt == 2 && rc != 0))
                  keep = 0;
            }
         }
         TRACE(HDBG, "GetSessionDirs: found entry "<<ent->d_name<<", keep: "<<keep);
         if (sdirs && keep)
            sdirs->push_back(new XrdOucString(ent->d_name));
      }
   }

   // Close the directory
   closedir(dir);

   // Sort the list
   if (sdirs)
#if !defined(__sun)
      sdirs->sort(&XpdSessionTagComp);
#else
      Sort(sdirs);
#endif

   // Done
   return ((opt == 3 && found) ? 1 : 0);
}
