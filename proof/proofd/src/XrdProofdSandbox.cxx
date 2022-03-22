// @(#)root/proofd:$Id$
// Author: G. Ganis  Jan 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdSandbox                                                     //
//                                                                      //
// Authors: G. Ganis, CERN, 2008                                        //
//                                                                      //
// Create and manage a Unix sandbox.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "XrdProofdPlatform.h"

#include "XrdProofdSandbox.h"
#include "XrdSys/XrdSysPriv.hh"

// Tracing utilities
#include "XrdProofdTrace.h"

// Modified via config directives by the manager
int          XrdProofdSandbox::fgMaxOldSessions = 10;
XrdOucString XrdProofdSandbox::fgWorkdir = "";
XrdProofUI   XrdProofdSandbox::fgUI;

////////////////////////////////////////////////////////////////////////////////
/// Assert existence on the sandbox for the user defined by 'ui'.
/// The sandbox is created under fgWorkdir or $HOME/proof; the boolean
/// 'full' controls the set of directories to be asserted: the sub-set
/// {cache, packages, .creds} is always asserted; if full is true also
/// the sub-dirs {queries, datasets} are asserted.
/// If 'changeown' is true the sandbox ownership is set to 'ui'; this
/// requires su-privileges.
/// The constructor also builds the list of sessions directories in the
/// sandbox; directories corresponding to terminated sessions are
/// removed if the total number of session directories is larger than
/// fgMaxOldSessions .

XrdProofdSandbox::XrdProofdSandbox(XrdProofUI ui, bool full, bool changeown)
                : fChangeOwn(changeown), fUI(ui)
{
   XPDLOC(CMGR, "XrdProofdSandbox")

   fValid = 0;

   // The first time fill the info about the owner of this process
   if (fgUI.fUid < 0)
      XrdProofdAux::GetUserInfo(getuid(), fgUI);

   // Default working directory location for the effective user
   XrdOucString defdir = fgUI.fHomeDir;
   if (!defdir.endswith('/')) defdir += "/";
   defdir += ".proof/";
   XrdOucString initus = ui.fUser[0];
   int iph = STR_NPOS;
   if (fgWorkdir.length() > 0) {
      // The user directory path will be <workdir>/<user>
      fDir = fgWorkdir;
      if (fDir.find("<user>") == STR_NPOS) {
         if (!fDir.endswith('/')) fDir += "/";
         fDir += "<user>";
      }
      // Replace supported place-holders
      fDir.replace("<workdir>", defdir);
      // Index of first place-holder
      iph = fDir.find("<effuser>");
      int iu = fDir.find("<u>");
      int ius = fDir.find("<user>");
      if (iu != STR_NPOS)
         if ((iph != STR_NPOS && iu < iph) || (iph == STR_NPOS)) iph = iu;
      if (ius != STR_NPOS)
         if ((iph != STR_NPOS && ius < iph) || (iph == STR_NPOS)) iph = ius;
      // Replace supported place-holders
      fDir.replace("<effuser>", fgUI.fUser);
      fDir.replace("<u>", initus);
      fDir.replace("<user>", ui.fUser);
   } else {
      if (changeown || ui.fUser == fgUI.fUser) {
         // Default: $HOME/proof
         fDir = ui.fHomeDir;
         if (!fDir.endswith('/'))
            fDir += "/";
         fDir += ".proof";
      } else {
         // ~daemon_owner/.proof/<user>
         fDir = fgUI.fHomeDir;
         if (!fDir.endswith('/'))
            fDir += "/";
         fDir += ".proof/";
         fDir += ui.fUser;
      }
   }
   TRACE(REQ, "work dir = " << fDir);

   // Make sure the directory exists
   if (iph > -1) {
      // Recursively ...
      XrdOucString path, sb;
      path.assign(fDir, 0, iph - 1);
      int from = iph;
      while ((from = fDir.tokenize(sb, from, '/')) != -1) {
         path += sb;
         if (XrdProofdAux::AssertDir(path.c_str(), ui, changeown) == -1) {
            fErrMsg += "unable to create work dir: ";
            fErrMsg += path;
            TRACE(XERR, fErrMsg);
            return;
         }
         path += "/";
      }
   } else {
      if (XrdProofdAux::AssertDir(fDir.c_str(), ui, changeown) == -1) {
         fErrMsg += "unable to create work dir: ";
         fErrMsg += fDir;
         TRACE(XERR, fErrMsg);
         return;
      }
   }

   // Dirs to be asserted
   const char *basicdirs[4] = { "/cache", "/packages", "/.creds", "/queries" };
   int i = 0;
   int n = (full) ? 4 : 3;
   for (i = 0; i < n; i++) {
      XrdOucString dir = fDir;
      dir += basicdirs[i];
      if (XrdProofdAux::AssertDir(dir.c_str(), ui, changeown) == -1) {
         fErrMsg += "unable to create dir: ";
         fErrMsg += dir;
         TRACE(XERR, fErrMsg);
         return;
      }
   }

   // Set validity
   fValid = 1;

   // Trim old terminated sessions
   TrimSessionDirs();
}

////////////////////////////////////////////////////////////////////////////////
/// Compare times from session tag strings

bool XpdSessionTagComp(XrdOucString *&lhs, XrdOucString *&rhs)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Sort ascendingly the list.
/// Function used on Solaris where std::list::sort() does not support an
/// alternative comparison algorithm.

static void Sort(std::list<XrdOucString *> *lst)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Scan the sandbox for sessions working dirs and return their
/// sorted (according to creation time, first is the newest) list
/// in 'sdirs'.
/// The option 'opt' may have 3 values:
///    0        all working dirs are kept
///    1        active sessions only
///    2        terminated sessions only
///    3        search entry containing 'tag' and fill tag with
///             the full entry name; if defined, sdirs is filled
/// Returns -1 otherwise in case of failure.
/// In case of success returns 0 for opt < 3, 1 if found or 0 if not
/// found for opt == 3.

int XrdProofdSandbox::GetSessionDirs(int opt, std::list<XrdOucString *> *sdirs,
                                     XrdOucString *tag)
{
   XPDLOC(CMGR, "Sandbox::GetSessionDirs")

   // If unknown take all
   opt = (opt >= 0 && opt <= 3) ? opt : 0;

   // Check inputs
   if ((opt < 3 && !sdirs) || (opt == 3 && !tag)) {
      TRACE(XERR, "invalid inputs");
      return -1;
   }

   TRACE(DBG, "opt: "<<opt<<", dir: "<<fDir);

   // Open dir
   DIR *dir = opendir(fDir.c_str());
   if (!dir) {
      TRACE(XERR, "cannot open dir "<<fDir<< " (errno: "<<errno<<")");
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
               XrdOucString fterm(fDir.c_str());
               fterm += '/';
               fterm += ent->d_name;
               fterm += "/.terminated";
               int rc = access(fterm.c_str(), F_OK);
               if ((opt == 1 && rc == 0) || (opt == 2 && rc != 0))
                  keep = 0;
            }
         }
         TRACE(HDBG, "found entry "<<ent->d_name<<", keep: "<<keep);
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

////////////////////////////////////////////////////////////////////////////////
/// Record entry for new proofserv session tagged 'tag' in the active
/// sessions file (`<SandBox>/.sessions`). The file is created if needed.
/// Return 0 on success, -1 on error.

int XrdProofdSandbox::AddSession(const char *tag)
{
   XPDLOC(CMGR, "Sandbox::AddSession")

   // Check inputs
   if (!tag) {
      XPDPRT("invalid input");
      return -1;
   }
   TRACE(DBG, "tag:"<<tag);

   XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
   if (XpdBadPGuard(pGuard, fUI.fUid) && fChangeOwn) {
      TRACE(XERR, "could not get privileges");
      return -1;
   }

   // File name
   XrdOucString fn = fDir;
   fn += "/.sessions";

   // Open the file for appending
   FILE *fact = fopen(fn.c_str(), "a+");
   if (!fact) {
      TRACE(XERR, "cannot open file "<<fn<<" for appending (errno: "<<errno<<")");
      return -1;
   }

   // Lock the file
   lseek(fileno(fact), 0, SEEK_SET);
   if (lockf(fileno(fact), F_LOCK, 0) == -1) {
      TRACE(XERR, "cannot lock file "<<fn<<" (errno: "<<errno<<")");
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
      TRACE(XERR, "cannot unlock file "<<fn<<" (errno: "<<errno<<")");

   // Close the file
   fclose(fact);

   // We are done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Guess session tag completing 'tag' (typically "-<pid>") by scanning the
/// active session file or the session dir.
/// In case of success, tag is filled with the full tag and 0 is returned.
/// In case of failure, -1 is returned.

int XrdProofdSandbox::GuessTag(XrdOucString &tag, int ridx)
{
   XPDLOC(CMGR, "Sandbox::GuessTag")

   TRACE(DBG, "tag: "<<tag);

   bool found = 0;
   bool last = (tag == "last") ? 1 : 0;

   if (!last && tag.length() > 0) {
      // Scan the sessions file
      XrdOucString fn = fDir;
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
               TRACE(DBG, "cannot unlock file "<<fn<<" ; fact: "<<fact<<
                          ", fd: "<< fileno(fact) << " (errno: "<<errno<<")");

         } else {
            TRACE(DBG, "cannot lock file: "<<fn<<" ; fact: "<<fact<<
                       ", fd: "<< fileno(fact) << " (errno: "<<errno<<")");
         }
         // Close the file
         fclose(fact);

      } else {
         TRACE(DBG, "cannot open file "<<fn<<
                    " for reading (errno: "<<errno<<")");
      }
   }

   if (!found) {

      // Search the tag in the dirs
      std::list<XrdOucString *> staglst;
      staglst.clear();
      int rc = GetSessionDirs(3, &staglst, &tag);
      if (rc < 0) {
         TRACE(XERR, "cannot scan dir "<<fDir);
         return -1;
      }
      found = (rc == 1) ? 1 : 0;

      if (!found && !staglst.empty()) {
         // Take last one, if required
         if (last) {
            tag = staglst.front()->c_str();
            found = 1;
         } else {
            if (ridx < 0) {
               int itag = ridx;
               // Reiterate back
               std::list<XrdOucString *>::iterator i;
               for (i = staglst.begin(); i != staglst.end(); ++i) {
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
         TRACE(DBG, "tag "<<tag<<" not found in dir");
      }
   }

   // We are done
   return ((found) ? 0 : -1);
}

////////////////////////////////////////////////////////////////////////////////
/// Move record for tag from the active sessions file to the old
/// sessions file (`<SandBox>/.sessions`). The active file is removed if
/// empty after the operation. The old sessions file is created if needed.
/// Return 0 on success, -1 on error.

int XrdProofdSandbox::RemoveSession(const char *tag)
{
   XPDLOC(CMGR, "Sandbox::RemoveSession")

   char ln[1024];

   // Check inputs
   if (!tag) {
      TRACE(XERR, "invalid input");
      return -1;
   }
   TRACE(DBG, "tag:"<<tag);

   XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
   if (XpdBadPGuard(pGuard, fUI.fUid) && fChangeOwn) {
      TRACE(XERR, "could not get privileges");
      return -1;
   }

   // Update of the active file
   XrdOucString fna = fDir;
   fna += "/.sessions";

   // Open the file
   FILE *fact = fopen(fna.c_str(), "a+");
   if (!fact) {
      TRACE(XERR, "cannot open file "<<fna<<" (errno: "<<errno<<")");
      return -1;
   }

   // Lock the file
   if (lockf(fileno(fact), F_LOCK, 0) == -1) {
      TRACE(XERR, "cannot lock file "<<fna<<" (errno: "<<errno<<")");
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
      TRACE(XERR, "cannot truncate file "<<fna<<" (errno: "<<errno<<")");
      lseek(fileno(fact), 0, SEEK_SET);
      if (lockf(fileno(fact), F_ULOCK, 0) != 0)
         TRACE(XERR, "cannot lockf file "<<fna<<" (errno: "<<errno<<")");
      fclose(fact);
      return -1;
   }

   // If active sessions still exist, write out new composition
   bool unlk = 1;
   if (!actln.empty()) {
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
      TRACE(DBG, "cannot unlock file "<<fna<<" (errno: "<<errno<<")");

   // Close the file
   fclose(fact);

   // Unlink the file if empty
   if (unlk)
      if (unlink(fna.c_str()) == -1)
         TRACE(DBG, "cannot unlink file "<<fna<<" (errno: "<<errno<<")");

   // Flag the session as closed
   XrdOucString fterm = fDir;
   fterm += (strstr(tag,"session-")) ? "/" : "/session-";
   fterm += tag;
   fterm += "/.terminated";
   // Create the file
   FILE *ft = fopen(fterm.c_str(), "w");
   if (!ft) {
      TRACE(XERR, "cannot open file "<<fterm<<" (errno: "<<errno<<")");
      return -1;
   }
   fclose(ft);

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// If the static fgMaxOldLogs > 0, logs for a fgMaxOldLogs number of sessions
/// are kept in the sandbox; working dirs for sessions in excess are removed.
/// By default logs for the last 10 sessions are kept; the limit can be changed
/// via the static method XrdProofdClient::SetMaxOldLogs.
/// Return 0 on success, -1 on error.

int XrdProofdSandbox::TrimSessionDirs()
{
   XPDLOC(CMGR, "Sandbox::TrimSessionDirs")

   TRACE(DBG, "maxold:"<<fgMaxOldSessions);

   // To avoid dead locks we must close the file and do the mv actions after
   XrdOucString tobemv, fnact = fDir;
   fnact += "/.sessions";
   FILE *f = fopen(fnact.c_str(), "r");
   if (f) {
      char ln[1024];
      while (fgets(ln, sizeof(ln), f)) {
         if (ln[strlen(ln)-1] == '\n')
            ln[strlen(ln)-1] = 0;
         char *p = strrchr(ln, '-');
         if (p) {
            int pid = strtol(p+1, 0, 10);
            if (!XrdProofdAux::VerifyProcessByID(pid)) {
               tobemv += ln;
               tobemv += '|';
            }
         }
      }
      fclose(f);
   }

   XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
   if (XpdBadPGuard(pGuard, fUI.fUid) && fChangeOwn) {
      TRACE(XERR, "could not get privileges to trim directories");
      return -1;
   }

   // Mv inactive sessions, if needed
   if (tobemv.length() > 0) {
      char del = '|';
      XrdOucString tag;
      int from = 0;
      while ((from = tobemv.tokenize(tag, from, del)) != -1) {
         if (RemoveSession(tag.c_str()) == -1)
            TRACE(XERR, "problems tagging session as old in sandbox");
      }
   }

   // If a limit on the number of sessions dirs is set, apply it
   if (fgMaxOldSessions > 0) {

      // Get list of terminated session working dirs
      std::list<XrdOucString *> staglst;
      staglst.clear();
      if (GetSessionDirs(2, &staglst) != 0) {
         TRACE(XERR, "cannot get list of dirs ");
         return -1;
      }
      TRACE(DBG, "number of working dirs: "<<staglst.size());

      if (TRACING(HDBG)) {
         std::list<XrdOucString *>::iterator i;
         for (i = staglst.begin(); i != staglst.end(); ++i) {
            TRACE(HDBG, "found "<<(*i)->c_str());
         }
      }

      // Remove the oldest, if needed
      while ((int)staglst.size() > fgMaxOldSessions) {
         XrdOucString *s = staglst.back();
         if (s) {
            TRACE(HDBG, "removing "<<s->c_str());
            // Remove associated workdir
            XrdOucString rmcmd = "/bin/rm -rf ";
            rmcmd += fDir;
            rmcmd += '/';
            rmcmd += s->c_str();
            if (system(rmcmd.c_str()) == -1)
               TRACE(XERR, "cannot invoke system("<<rmcmd<<") (errno: "<<errno<<")");
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


