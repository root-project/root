// @(#)root/proofd:$Id$
// Author: G. Ganis June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdManager                                                     //
//                                                                      //
// Author: G. Ganis, CERN, 2007                                         //
//                                                                      //
// Class mapping manager functionality.                                 //
// On masters it keeps info about the available worker nodes and allows //
// communication with them.                                             //
// On workers it handles the communication with the master.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "XrdProofdPlatform.h"

#include "XrdProofdManager.h"

#include "XrdVersion.hh"
#include "Xrd/XrdProtocol.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPriv.hh"

#include "XpdSysPlugin.h"
#include "XpdSysTimer.h"
#include "XpdSysDNS.h"

#include "XrdProofdAdmin.h"
#include "XrdProofdClient.h"
#include "XrdProofdClientMgr.h"
#include "XrdProofdConfig.h"
#include "XrdProofdNetMgr.h"
#include "XrdProofdPriorityMgr.h"
#include "XrdProofdProofServMgr.h"
#include "XrdProofdProtocol.h"
#include "XrdProofGroup.h"
#include "XrdProofSched.h"
#include "XrdProofdProofServ.h"
#include "XrdProofWorker.h"
#include "XrdROOT.h"
#include "rpdconn.h"

// Tracing utilities
#include "XrdProofdTrace.h"

#include <grp.h>
#include <unistd.h>

// Auxilliary sructure used internally to extract list of allowed/denied user names
// when running in access control mode
typedef struct {
   XrdOucString allowed;
   XrdOucString denied;
} xpd_acm_lists_t;

// Protocol loader; arguments: const char *pname, char *parms,  XrdProtocol_Config *pi
typedef XrdProtocol *(*XrdProtocolLoader_t)(const char *, char *, XrdProtocol_Config *);

#ifdef __sun
/*-
 * Copyright (c) 1991, 1993
 * The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * This product includes software developed by the University of
 * California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#if 0
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)getgrouplist.c   8.2 (Berkeley) 12/8/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/getgrouplist.c,v 1.14 2005/05/03 16:20:03 delphij Exp $");
#endif

/*
 * get credential
 */
#include <sys/types.h>

#include <string.h>

int
getgrouplist(const char *uname, gid_t agroup, gid_t *groups, int *grpcnt)
{
   const struct group *grp;
   int i, maxgroups, ngroups, ret;

   ret = 0;
   ngroups = 0;
   maxgroups = *grpcnt;
   /*
    * When installing primary group, duplicate it;
    * the first element of groups is the effective gid
    * and will be overwritten when a setgid file is executed.
    */
   groups ? groups[ngroups++] = agroup : ngroups++;
   if (maxgroups > 1)
      groups ? groups[ngroups++] = agroup : ngroups++;
   /*
    * Scan the group file to find additional groups.
    */
   setgrent();
   while ((grp = getgrent()) != NULL) {
      if (groups) {
         for (i = 0; i < ngroups; i++) {
            if (grp->gr_gid == groups[i])
               goto skip;
         }
      }
      for (i = 0; grp->gr_mem[i]; i++) {
         if (!strcmp(grp->gr_mem[i], uname)) {
            if (ngroups >= maxgroups) {
               ret = -1;
               break;
            }
            groups ? groups[ngroups++] = grp->gr_gid : ngroups++;
            break;
         }
      }
skip:
      ;
   }
   endgrent();
   *grpcnt = ngroups;
   return (ret);
}
#endif

//--------------------------------------------------------------------------
//
// XrdProofdManagerCron
//
// Function run in separate thread doing regular checks
//
////////////////////////////////////////////////////////////////////////////////
/// This is an endless loop to periodically check the system

void *XrdProofdManagerCron(void *p)
{
   XPDLOC(PMGR, "ManagerCron")

   XrdProofdManager *mgr = (XrdProofdManager *)p;
   if (!(mgr)) {
      TRACE(REQ, "undefined manager: cannot start");
      return (void *)0;
   }

   TRACE(REQ, "started with frequency " << mgr->CronFrequency() << " sec");

   // Get Midnight time
   int now = time(0);
   int mid = XrdSysTimer::Midnight(now);
   while (mid < now) {
      mid += 86400;
   }
   TRACE(REQ, "midnight in  " << (mid - now) << " secs");

   while (1) {
      // Do something here
      TRACE(REQ, "running periodical checks");
      // Check the log file ownership
      mgr->CheckLogFileOwnership();
      // Wait a while
      int tw = mgr->CronFrequency();
      now = time(0);
      if ((mid - now) <= tw) {
         tw = mid - now + 2; // Always run a check just after midnight
         mid += 86400;
      }

      // Check if reconfiguration of some services is required (triggered by a change
      // of the configuration file)
      if (mgr->SessionMgr()) mgr->SessionMgr()->Config(1);
      if (mgr->GroupsMgr()) mgr->GroupsMgr()->Config(mgr->GroupsMgr()->GetCfgFile());

      XrdSysTimer::Wait(tw * 1000);
   }

   // Should never come here
   return (void *)0;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor

XrdProofdManager::XrdProofdManager(char *parms, XrdProtocol_Config *pi, XrdSysError *edest)
                 : XrdProofdConfig(pi->ConfigFN, edest)
{

   fParms = parms; // only used for construction: not to be trusted later on
   fPi = pi;       // only used for construction: not to be trusted later on

   fSrvType = kXPD_AnyServer;
   fEffectiveUser = "";
   fHost = "";
   fPort = XPD_DEF_PORT;
   fImage = "";        // image name for these servers
   fSockPathDir = "";
   fStageReqRepo = "";
   fTMPdir = "/tmp";
   fWorkDir = "";
   fMUWorkDir = "";
   fSuperMst = 0;
   fRemotePLite = 0;
   fNamespace = "/proofpool";
   fMastersAllowed.clear();
   fOperationMode = kXPD_OpModeOpen;
   fMultiUser = 0;
   fChangeOwn = 0;
   fCronFrequency = 30;

   // Data dir
   fDataDir = "";        // Default <workdir>/<user>/data
   fDataDirOpts = "";    // Default: no action
   fDataDirUrlOpts = ""; // Default: none

   // Tools to enable xrootd file serving
   fXrootdLibPath = "<>";
   fXrootdPlugin = 0;

   // Proof admin path
   fAdminPath = pi->AdmPath;
   fAdminPath += "/.xproofd.";

   // Lib paths for proofserv
   fBareLibPath = "";
   fRemoveROOTLibPaths = 0;
   fLibPathsToRemove.Purge();

   // Services
   fSched = pi->Sched;
   fAdmin = 0;
   fClientMgr = 0;
   fGroupsMgr = 0;
   fNetMgr = 0;
   fPriorityMgr = 0;
   fProofSched = 0;
   fSessionMgr = 0;

   // Configuration directives
   RegisterDirectives();

   // Admin request handler
   fAdmin = new XrdProofdAdmin(this, pi, edest);

   // Client manager
   fClientMgr = new XrdProofdClientMgr(this, pi, edest);

   // Network manager
   fNetMgr = new XrdProofdNetMgr(this, pi, edest);

   // Priority manager
   fPriorityMgr = new XrdProofdPriorityMgr(this, pi, edest);

   // ROOT versions manager
   fROOTMgr = new XrdROOTMgr(this, pi, edest);

   // Session manager
   fSessionMgr = new XrdProofdProofServMgr(this, pi, edest);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

XrdProofdManager::~XrdProofdManager()
{
   // Destroy the configuration handler
   SafeDelete(fAdmin);
   SafeDelete(fClientMgr);
   SafeDelete(fNetMgr);
   SafeDelete(fPriorityMgr);
   SafeDelete(fProofSched);
   SafeDelete(fROOTMgr);
   SafeDelete(fSessionMgr);
   SafeDelete(fXrootdPlugin);
}

////////////////////////////////////////////////////////////////////////////////
/// Load the Xrootd protocol, if required

XrdProtocol *XrdProofdManager::LoadXrootd(char *parms, XrdProtocol_Config *pi, XrdSysError *edest)
{
   XPDLOC(ALL, "Manager::LoadXrootd")

   XrdProtocol *xrp = 0;

   // Create the plug-in instance
   fXrootdPlugin = new XrdSysPlugin((edest ? edest : (XrdSysError *)0), fXrootdLibPath.c_str());
   if (!fXrootdPlugin) {
      TRACE(XERR, "could not create plugin instance for "<<fXrootdLibPath.c_str());
      return xrp;
   }

   // Get the function
   XrdProtocolLoader_t ep = (XrdProtocolLoader_t) fXrootdPlugin->getPlugin("XrdgetProtocol");
   if (!ep) {
      TRACE(XERR, "could not find 'XrdgetProtocol()' in "<<fXrootdLibPath.c_str());
      return xrp;
   }

   // Get the server object
   if (!(xrp = (*ep)("xrootd", parms, pi))) {
      TRACE(XERR, "Unable to create xrootd protocol service object via " << fXrootdLibPath.c_str());
      SafeDelete(fXrootdPlugin);
   } else {
      // Notify
      TRACE(ALL, "xrootd protocol service created");
   }

   return xrp;
}

////////////////////////////////////////////////////////////////////////////////
/// Make sure that the log file belongs to the original effective user

void XrdProofdManager::CheckLogFileOwnership()
{
   XPDLOC(ALL, "Manager::CheckLogFileOwnership")

   // Nothing to do if not priviledged
   if (getuid()) return;

   struct stat st;
   if (fstat(STDERR_FILENO, &st) != 0) {
      if (errno != ENOENT) {
         TRACE(XERR, "could not stat log file; errno: " << errno);
         return;
      }
   }

   TRACE(HDBG, "uid: " << st.st_uid << ", gid: " << st.st_gid);

   // Get original effective user identity
   struct passwd *epwd = getpwuid(XrdProofdProtocol::EUidAtStartup());
   if (!epwd) {
      TRACE(XERR, "could not get effective user identity; errno: " << errno);
      return;
   }

   // Set ownership of the log file to the effective user
   if (st.st_uid != epwd->pw_uid || st.st_gid != epwd->pw_gid) {
      if (fchown(STDERR_FILENO, epwd->pw_uid, epwd->pw_gid) != 0) {
         TRACE(XERR, "could not set stderr ownership; errno: " << errno);
         return;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Check if master 'm' is allowed to connect to this host

bool XrdProofdManager::CheckMaster(const char *m)
{
   bool rc = 1;

   if (fMastersAllowed.size() > 0) {
      rc = 0;
      XrdOucString wm(m);
      std::list<XrdOucString *>::iterator i;
      for (i = fMastersAllowed.begin(); i != fMastersAllowed.end(); ++i) {
         if (wm.matches((*i)->c_str())) {
            rc = 1;
            break;
         }
      }
   }

   // We are done
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the user is allowed to use the system
/// Return 0 if OK, -1 if not.

int XrdProofdManager::CheckUser(const char *usr, const char *grp,
                                XrdProofUI &ui, XrdOucString &e, bool &su)
{
   su = 0;
   // User must be defined
   if (!usr || strlen(usr) <= 0) {
      e = "CheckUser: 'usr' string is undefined ";
      return -1;
   }

   // No 'root' logins
   if (strlen(usr) == 4 && !strcmp(usr, "root")) {
      e = "CheckUser: 'root' logins not accepted ";
      return -1;
   }

   // Group must be defined
   if (!grp || strlen(grp) <= 0) {
      e = "CheckUser: 'grp' string is undefined ";
      return -1;
   }

   XrdSysMutexHelper mtxh(&fMutex);

   // Here we check if the user is known locally.
   // If not, we fail for now.
   // In the future we may try to get a temporary account
   if (fChangeOwn) {
      if (XrdProofdAux::GetUserInfo(usr, ui) != 0) {
         e = "CheckUser: unknown ClientID: ";
         e += usr;
         return -1;
      }
   } else {
      // We assign the ui of the effective user
      if (XrdProofdAux::GetUserInfo(geteuid(), ui) != 0) {
         e = "CheckUser: problems getting user info for id: ";
         e += (int)geteuid();
         return -1;
      }
   }

   // Check if super user
   if (fSuperUsers.length() > 0) {
      XrdOucString tkn;
      int from = 0;
      while ((from = fSuperUsers.tokenize(tkn, from, ',')) != -1) {
         if (tkn == usr) {
            su = 1;
            break;
         }
      }
   }

   // If we are in controlled mode we have to check if the user (and possibly
   // its group) are in the authorized lists; otherwise we fail.
   // Privileged users are always allowed to connect.
   if (fOperationMode == kXPD_OpModeControlled) {

      // Policy: check first the general directive for groups; a user of a specific group
      // (both UNIX or PROOF groups) can be rejected by prefixing a '-'.
      // The group check fails if active (the allowedgroups directive has entries) and at
      // least of the two groups (UNIX or PROOF) are explicitly denied.
      // The result of the group check is superseeded by any explicit speicification in the
      // allowedusers, either positive or negative.
      // UNIX group includes secondary groups; this allows to enable/disable only members of a
      // specific subgroup
      //
      // Examples:
      //   Consider user 'katy' with UNIX group 'alfa' and PROOF group 'student',
      //   users 'jack' and 'john' with UNIX group 'alfa' and PROOF group 'postdoc'.
      //
      //   1.    xpd.allowedgroups alfa
      //         Users 'katy', 'jack' and 'john' are allowed because part of UNIX group 'alfa' (no 'allowedusers' directive)
      //   2.    xpd.allowedgroups student
      //         User 'katy' is allowed because part of PROOF group 'student';
      //         users 'jack' and 'john' are denied because not part of PROOF group 'student' (no 'allowedusers' directive)
      //   3.    xpd.allowedgroups alfa,-student
      //         User 'katy' is denied because part of PROOF group 'student' which is explicitly denied;
      //         users 'jack' and 'john' are allowed becasue part of UNIX group 'alfa' (no 'allowedusers' directive)
      //   4.    xpd.allowedgroups alfa,-student
      //         xpd.allowedusers katy,-jack
      //         User 'katy' is allowed because explicitly allowed by the 'allowedusers' directive;
      //         user 'jack' is denied because explicitly denied by the 'allowedusers' directive;
      //         user 'john' is allowed because part of 'alfa' and not explicitly denied by the 'allowedusers' directive
      //         (the allowedgroups directive is in this case ignored for users 'katy' and 'jack').

      bool grpok = 1;
      // Check unix groups (secondaries included)
      if (fAllowedGroups.Num() > 0) {
         // Reset the flag
         grpok = 0;
         int ugrpok = 0, pgrpok = 0;
         // Check UNIX groups info
         int ngrps = 10, neg, ig = 0;
#if defined(__APPLE__)
         int grps[10];
#else
         gid_t grps[10];
#endif
         XrdOucString g;
         if ((neg = getgrouplist(usr, ui.fGid, grps, &ngrps)) < 0) neg = 10;
         if (neg > 0) {
            for (ig = 0; ig < neg; ig++) {
               g.form("%d", (int) grps[ig]);
               int *st = fAllowedGroups.Find(g.c_str());
               if (st) {
                  if (*st == 1) {
                     ugrpok = 1;
                  } else {
                     e = "Controlled access (UNIX group): user '";
                     e += usr;
                     e = "', UNIX group '";
                     e += g;
                     e += "' denied to connect";
                     ugrpok = -1;
                     break;
                  }
               }
            }
         }
         // Check PROOF group info
         int *st = fAllowedGroups.Find(grp);
         if (st) {
            if (*st == 1) {
               pgrpok = 1;
            } else {
               if (e.length() <= 0)
                  e = "Controlled access";
               e += " (PROOF group): user '";
               e += usr;
               e += "', PROOF group '";
               e += grp;
               e += "' denied to connect";
               pgrpok = -1;
            }
         }
         // At least one must be explicitly allowed with the other not explicitly denied
         grpok = ((ugrpok == 1 && pgrpok >= 0) || (ugrpok >= 0 && pgrpok == 1)) ? 1 : 0;
      }
      // Check username
      int usrok = 0;
      if (fAllowedUsers.Num() > 0) {
         // If we do not have a group specification we need to explicitly allow the user
         if (fAllowedGroups.Num() <= 0) usrok = -1;
         // Look into the hash
         int *st = fAllowedUsers.Find(usr);
         if (st) {
            if (*st == 1) {
               usrok = 1;
            } else {
               e = "Controlled access: user '";
               e += usr;
               e += "', PROOF group '";
               e += grp;
               e += "' not allowed to connect";
               usrok = -1;
            }
         }
      }
      // Super users are always allowed
      if (su) {
         usrok = 1;
         e = "";
      }
      // We fail if either the user is explicitly denied or it is not explicitly allowed
      // and the group is denied
      if (usrok == -1 || (!grpok && usrok != 1)) return -1;
   }

   // OK
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Load PROOF scheduler

XrdProofSched *XrdProofdManager::LoadScheduler()
{
   XPDLOC(ALL, "Manager::LoadScheduler")

   XrdProofSched *sched = 0;
   XrdOucString name, lib, m;

   const char *cfn = CfgFile();

   // Locate first the relevant directives in the config file
   if (cfn && strlen(cfn) > 0) {
      XrdOucEnv myEnv;
      XrdOucStream cfg(fEDest, getenv("XRDINSTANCE"), &myEnv);
      // Open and attach the config file
      int cfgFD;
      if ((cfgFD = open(cfn, O_RDONLY, 0)) >= 0) {
         cfg.Attach(cfgFD);
         // Process items
         char *val = 0, *var = 0;
         while ((var = cfg.GetMyFirstWord())) {
            if (!(strcmp("xpd.sched", var))) {
               // Get the name
               val = cfg.GetWord();
               if (val && val[0]) {
                  name = val;
                  // Get the lib
                  val = cfg.GetWord();
                  if (val && val[0])
                     lib = val;
                  // We are done
                  break;
               }
            }
         }
         close(cfgFD);
      } else {
         XPDFORM(m, "failure opening config file; errno: %d", errno);
         TRACE(XERR, m);
      }
   }

   // If undefined or default init a default instance
   if (name == "default" || !(name.length() > 0 && lib.length() > 0)) {
      if ((name.length() <= 0 && lib.length() > 0) ||
          (name.length() > 0 && lib.length() <= 0)) {
         XPDFORM(m, "missing or incomplete info (name: %s, lib: %s)", name.c_str(), lib.c_str());
         TRACE(DBG, m);
      }
      TRACE(DBG, "instantiating default scheduler");
      sched = new XrdProofSched("default", this, fGroupsMgr, cfn, fEDest);
   } else {
      // Load the required plugin
      if (lib.beginswith("~") || lib.beginswith("$"))
         XrdProofdAux::Expand(lib);
      XrdSysPlugin *h = new XrdSysPlugin(fEDest, lib.c_str());
      if (!h)
         return (XrdProofSched *)0;
      // Get the scheduler object creator
      XrdProofSchedLoader_t ep = (XrdProofSchedLoader_t) h->getPlugin("XrdgetProofSched", 1);
      if (!ep) {
         delete h;
         return (XrdProofSched *)0;
      }
      // Get the scheduler object
      if (!(sched = (*ep)(cfn, this, fGroupsMgr, cfn, fEDest))) {
         TRACE(XERR, "unable to create scheduler object from " << lib);
         delete h;
         return (XrdProofSched *)0;
      }
      delete h;
   }
   // Check result
   if (!(sched->IsValid())) {
      TRACE(XERR, " unable to instantiate the " << sched->Name() << " scheduler using " << (cfn ? cfn : "<nul>"));
      delete sched;
      return (XrdProofSched *)0;
   }
   // Notify
   TRACE(ALL, "scheduler loaded: type: " << sched->Name());

   // All done
   return sched;
}

////////////////////////////////////////////////////////////////////////////////
/// Get a list of workers from the available resource broker

int XrdProofdManager::GetWorkers(XrdOucString &lw, XrdProofdProofServ *xps,
                                 const char *query)
{
   XPDLOC(ALL, "Manager::GetWorkers")

   int rc = 0;
   TRACE(REQ, "enter");

   // We need the scheduler at this point
   if (!fProofSched) {
      TRACE(XERR, "scheduler undefined");
      return -1;
   }

   // Query the scheduler for the list of workers
   std::list<XrdProofWorker *> wrks, uwrks;
   if ((rc = fProofSched->GetWorkers(xps, &wrks, query)) < 0) {
      TRACE(XERR, "error getting list of workers from the scheduler");
      return -1;
   }
   std::list<XrdProofWorker *>::iterator iw, iaw;
   // If we got a new list we save it into the session object
   if (rc == 0) {

      TRACE(DBG, "list size: " << wrks.size());

      XrdOucString ord;
      int ii = -1;
      // If in remote PLite mode, we need to isolate the number of workers
      // per unique node
      if (fRemotePLite) {
         for (iw = wrks.begin(); iw != wrks.end() ; ++iw) {
            XrdProofWorker *w = *iw;
            // Do we have it already in the unique list?
            bool isnew = 1;
            for (iaw = uwrks.begin(); iaw != uwrks.end() ; ++iaw) {
               XrdProofWorker *uw = *iaw;
               if (w->fHost == uw->fHost && w->fPort == uw->fPort) {
                  uw->fNwrks += 1;
                  isnew = 0;
                  break;
               }
            }
            if (isnew) {
               // Count (fActive is increased inside here)
               if (ii == -1) {
                  ord = "master";
               } else {
                  XPDFORM(ord, "%d", ii);
               }
               ii++;
               XrdProofWorker *uw = new XrdProofWorker(*w);
               uw->fType = 'S';
               uw->fOrd = ord;
               uwrks.push_back(uw);
               // Setup connection with the proofserv using the original
               xps->AddWorker(ord.c_str(), w);
               w->AddProofServ(xps);
            }
         }
         for (iw = uwrks.begin(); iw != uwrks.end() ; ++iw) {
            XrdProofWorker *w = *iw;
            // Master at the beginning
            if (w->fType == 'M') {
               if (lw.length() > 0) lw.insert('&',0);
               lw.insert(w->Export(), 0);
            } else {
               // Add separator if not the first
               if (lw.length() > 0) lw += '&';
               // Add export version of the info
               lw += w->Export(0);
            }
         }

      } else {

         // The full list
         for (iw = wrks.begin(); iw != wrks.end() ; ++iw) {
            XrdProofWorker *w = *iw;
            // Count (fActive is increased inside here)
            if (ii == -1)
               ord = "master";
            else
               XPDFORM(ord, "%d", ii);
            ii++;
            xps->AddWorker(ord.c_str(), w);
            // Add proofserv and increase the counter
            w->AddProofServ(xps);
         }
      }
   }

   int proto = (xps->ROOT()) ? xps->ROOT()->SrvProtVers() : -1;
   if (rc != 2 || (proto < 21 && rc == 0)) {
      // Get the list in exported format
      if (lw.length() <= 0) xps->ExportWorkers(lw);
      TRACE(DBG, "from ExportWorkers: " << lw);
   } else if (proto >= 21) {
      // Signal enqueing
      lw = XPD_GW_QueryEnqueued;
   }

   if (TRACING(REQ)) fNetMgr->Dump();

   // Clear the temp list
   if (!uwrks.empty()) {
      iw = uwrks.begin();
      while (iw != uwrks.end()) {
         XrdProofWorker *w = *iw;
         iw = uwrks.erase(iw);
         delete w;
      }
   }

   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Add the key value in the string passed via the void argument

static int FillKeyValues(const char *k, int *d, void *s)
{
   xpd_acm_lists_t *ls = (xpd_acm_lists_t *)s;

   if (ls) {
      XrdOucString &ss = (*d == 1) ? ls->allowed : ls->denied;
      // Add the key
      if (k) {
         XrdOucString sk;
         sk += k;
         if (!sk.isdigit()) {
            // If not empty add a separation ','
            if (ss.length() > 0) ss += ",";
            ss += sk;
         }
      }
   } else {
      // Not enough info: stop
      return 1;
   }

   // Check next
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Add the key value in the string passed via the void argument

static int RemoveInvalidUsers(const char *k, int *, void *s)
{
   XrdOucString *ls = (XrdOucString *)s;

   XrdProofUI ui;
   if (XrdProofdAux::GetUserInfo(k, ui) != 0) {
      // Username is unknown to the system: remove it to the list
      if (ls) {
         // If not empty add a separation ','
         if (ls->length() > 0) *ls += ",";
         // Add the key
         if (k) *ls += k;
      }
      // Negative return removes from the table
      return -1;
   }

   // Check next
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Run configuration and parse the entered config directives.
/// Return 0 on success, -1 on error

int XrdProofdManager::Config(bool rcf)
{
   XPDLOC(ALL, "Manager::Config")

   XrdSysMutexHelper mtxh(fMutex);

   // Run first the configurator
   if (XrdProofdConfig::Config(rcf) != 0) {
      XPDERR("problems parsing file ");
      return -1;
   }

   XrdOucString msg;
   msg = (rcf) ? "re-configuring" : "configuring";
   TRACE(ALL, msg);

   // Change/DonotChange ownership when logging clients
   fChangeOwn = (fMultiUser && getuid()) ? 0 : 1;

   // Notify port
   XPDFORM(msg, "listening on port %d", fPort);
   TRACE(ALL, msg);

   XrdProofUI ui;
   uid_t effuid = XrdProofdProtocol::EUidAtStartup();
   if (!rcf) {
      // Save Effective user
      if (XrdProofdAux::GetUserInfo(effuid, ui) == 0) {
         fEffectiveUser = ui.fUser;
      } else {
         XPDFORM(msg, "could not resolve effective uid %d (errno: %d)", effuid, errno);
         XPDERR(msg);
         return -1;
      }

      // Local FQDN
      char *host = XrdSysDNS::getHostName();
      fHost = host ? host : "";
      SafeFree(host);

      // Notify temporary directory
      TRACE(ALL, "using temp dir: " << fTMPdir);

      // Notify role
      const char *roles[] = { "any", "worker", "submaster", "master" };
      TRACE(ALL, "role set to: " << roles[fSrvType+1]);

      // Admin path
      fAdminPath += fPort;
      if (XrdProofdAux::AssertDir(fAdminPath.c_str(), ui, fChangeOwn) != 0) {
         XPDERR("unable to assert the admin path: " << fAdminPath);
         return -1;
      }
      TRACE(ALL, "admin path set to: " << fAdminPath);

      // Path for Unix sockets
      if (fSockPathDir.length() <= 0) {
         // Use default under the admin path
         XPDFORM(fSockPathDir, "%s/socks", fAdminPath.c_str());
      }
      if (XrdProofdAux::AssertDir(fSockPathDir.c_str(), ui, fChangeOwn) != 0) {
         XPDERR("unable to assert the admin path: " << fSockPathDir);
         return -1;
      }
      if (XrdProofdAux::ChangeMod(fSockPathDir.c_str(), 0777) != 0) {
         XPDERR("unable to set mode 0777 on: " << fSockPathDir);
         return -1;
      }
      TRACE(ALL, "unix sockets under: " << fSockPathDir);

      // Create / Update the process ID file under the admin path
      XrdOucString pidfile(fAdminPath);
      pidfile += "/xrootd.pid";
      FILE *fpid = fopen(pidfile.c_str(), "w");
      if (!fpid) {
         XPDFORM(msg, "unable to open pid file: %s; errno: %d", pidfile.c_str(), errno);
         XPDERR(msg);
         return -1;
      }
      fprintf(fpid, "%d", getpid());
      fclose(fpid);
   } else {
      if (XrdProofdAux::GetUserInfo(effuid, ui) == 0) {
         XPDFORM(msg, "could not resolve effective uid %d (errno: %d)", effuid, errno);
         XPDERR(msg);
      }
   }

   // Work directory, if specified
   XrdOucString wdir;
   if (fWorkDir.length() > 0) {
      // Make sure it exists
      if (XrdProofdAux::AssertDir(fWorkDir.c_str(), ui, fChangeOwn) != 0) {
         XPDERR("unable to assert working dir: " << fWorkDir);
         return -1;
      }
      if (fMUWorkDir.length() > 0) {
         fMUWorkDir.replace("<workdir>", fWorkDir);
         int iph = fMUWorkDir.find("<");
         if (iph != STR_NPOS) {
            wdir.assign(fMUWorkDir, 0, iph - 2);
            if (XrdProofdAux::AssertDir(wdir.c_str(), ui, fChangeOwn) != 0) {
               XPDERR("unable to assert working dir: " << wdir);
               return -1;
            }
            wdir = "";
         }
      }
   }
   wdir = (fMultiUser && fMUWorkDir.length() > 0) ? fMUWorkDir : fWorkDir;
   if (wdir.length() > 0) {
      TRACE(ALL, "working directories under: " << wdir);
      // Communicate it to the sandbox service
      XrdProofdSandbox::SetWorkdir(wdir.c_str());
   }

   // Data directory, if specified
   if (fDataDir.length() > 0) {
      if (fDataDir.endswith('/')) fDataDir.erasefromend(1);
      if (fDataDirOpts.length() > 0) {
         // Make sure it exists
         if (XrdProofdAux::AssertDir(fDataDir.c_str(), ui, fChangeOwn) != 0) {
            XPDERR("unable to assert data dir: " << fDataDir << " (opts: "<<fDataDirOpts<<")");
            return -1;
         }
         // Get the right privileges now
         XrdSysPrivGuard pGuard((uid_t)ui.fUid, (gid_t)ui.fGid);
         if (XpdBadPGuard(pGuard, ui.fUid)) {
            TRACE(XERR, "could not get privileges to set/change ownership of " << fDataDir);
            return -1;
         }
         if (chmod(fDataDir.c_str(), 0777) != 0) {
            XPDERR("problems setting permissions 0777 data dir: " << fDataDir);
            return -1;
         }
      }
      TRACE(ALL, "data directories under: " << fDataDir);
   }

   // Notify allow rules
   if (fSrvType == kXPD_Worker) {
      if (fMastersAllowed.size() > 0) {
         std::list<XrdOucString *>::iterator i;
         for (i = fMastersAllowed.begin(); i != fMastersAllowed.end(); ++i)
            TRACE(ALL, "masters allowed to connect: " << (*i)->c_str());
      } else {
         TRACE(ALL, "masters allowed to connect: any");
      }
   }

   // Pool and namespace
   if (fPoolURL.length() <= 0) {
      // Default pool entry point is this host
      fPoolURL = "root://";
      fPoolURL += fHost;
   }
   TRACE(ALL, "PROOF pool: " << fPoolURL);
   TRACE(ALL, "PROOF pool namespace: " << fNamespace);

   // Initialize resource broker (if not worker)
   if (fSrvType != kXPD_Worker) {

      // Scheduler instance
      if (!(fProofSched = LoadScheduler())) {
         XPDERR("scheduler initialization failed");
         return 0;
      }
      const char *st[] = { "disabled", "enabled" };
      TRACE(ALL, "user config files are " << st[fNetMgr->WorkerUsrCfg()]);
   }

   // If using the PLite optimization notify it
   if (fRemotePLite)
      TRACE(ALL, "multi-process on nodes handled with proof-lite");

   // Validate dataset sources (if not worker)
   fDataSetExp = "";
   if (fSrvType != kXPD_Worker && fDataSetSrcs.size() > 0) {
      // If first local, add it in front
      std::list<XrdProofdDSInfo *>::iterator ii = fDataSetSrcs.begin();
      bool goodsrc = 0;
      for (ii = fDataSetSrcs.begin(); ii != fDataSetSrcs.end();) {
         TRACE(ALL, ">> Defined dataset: " << (*ii)->ToString());
         if ((*ii)->fType == "file") {
            if (!(goodsrc = ValidateLocalDataSetSrc((*ii)->fUrl, (*ii)->fLocal))) {
               XPDERR("source " << (*ii)->fUrl << " could not be validated");
               ii = fDataSetSrcs.erase(ii);
            } else {
               // Check next
               ++ii;
            }
         } else {
            // Validate only "file" datasets
            TRACE(ALL, "Skipping validation (no \"file\" type dataset source)");
            ++ii;
         }
      }
      if (fDataSetSrcs.size() > 0) {
         TRACE(ALL, fDataSetSrcs.size() << " dataset sources defined");
         for (ii = fDataSetSrcs.begin(); ii != fDataSetSrcs.end(); ++ii) {
            TRACE(ALL, ">> Valid dataset: " << (*ii)->ToString());
            if ((*ii)->fLocal && (*ii)->fRW) {
               if (fDataSetExp.length() > 0) fDataSetExp += ",";
               fDataSetExp += ((*ii)->fUrl).c_str();
            }
         }
      } else {
         TRACE(ALL, "no dataset sources defined");
      }
   } else {
      TRACE(ALL, "no dataset sources defined");
   }

   // Superusers: add the effective user at startup
   XrdProofUI sui;
   if (XrdProofdAux::GetUserInfo(XrdProofdProtocol::EUidAtStartup(), sui) == 0) {
      if (fSuperUsers.find(sui.fUser.c_str()) == STR_NPOS) {
         if (fSuperUsers.length() > 0) fSuperUsers += ",";
         fSuperUsers += sui.fUser;
      }
   } else {
      XPDFORM(msg, "could not resolve effective uid %d (errno: %d)",
              XrdProofdProtocol::EUidAtStartup(), errno);
      XPDERR(msg);
   }
   XPDFORM(msg, "list of superusers: %s", fSuperUsers.c_str());
   TRACE(ALL, msg);

   // Notify controlled mode, if such
   if (fOperationMode == kXPD_OpModeControlled) {
      // Add superusers to the hash list of allowed users
      int from = 0;
      XrdOucString usr;
      while ((from = fSuperUsers.tokenize(usr, from, ',')) != STR_NPOS) {
         fAllowedUsers.Add(usr.c_str(), new int(1));
      }
      // If not in multiuser mode make sure that the users in the allowed list
      // are known to the system
      if (!fMultiUser) {
         XrdOucString ius;
         fAllowedUsers.Apply(RemoveInvalidUsers, (void *)&ius);
         if (ius.length()) {
            XPDFORM(msg, "running in controlled access mode: users removed because"
                         " unknown to the system: %s", ius.c_str());
            TRACE(ALL, msg);
         }
      }
      // Extract now the list of allowed users
      xpd_acm_lists_t uls;
      fAllowedUsers.Apply(FillKeyValues, (void *)&uls);
      if (uls.allowed.length()) {
         XPDFORM(msg, "running in controlled access mode: users allowed: %s", uls.allowed.c_str());
         TRACE(ALL, msg);
      }
      if (uls.denied.length()) {
         XPDFORM(msg, "running in controlled access mode: users denied: %s", uls.denied.c_str());
         TRACE(ALL, msg);
      }
      // Extract now the list of allowed groups
      xpd_acm_lists_t gls;
      fAllowedGroups.Apply(FillKeyValues, (void *)&gls);
      if (gls.allowed.length()) {
         XPDFORM(msg, "running in controlled access mode: UNIX groups allowed: %s", gls.allowed.c_str());
         TRACE(ALL, msg);
      }
      if (gls.denied.length()) {
         XPDFORM(msg, "running in controlled access mode: UNIX groups denied: %s", gls.denied.c_str());
         TRACE(ALL, msg);
      }
   }

   // Bare lib path
   if (getenv(XPD_LIBPATH)) {
      XrdOucString ctrim;
      if (fRemoveROOTLibPaths || fLibPathsToRemove.Num() > 0) {
         // Try to remove existing ROOT dirs in the path
         XrdOucString paths = getenv(XPD_LIBPATH);
         XrdOucString ldir;
         int from = 0;
         while ((from = paths.tokenize(ldir, from, ':')) != STR_NPOS) {
            bool remove = 0;
            if (ldir.length() > 0) {
               if (fLibPathsToRemove.Num() > 0 && fLibPathsToRemove.Find(ldir.c_str())) {
                  remove = 1;
               } else if (fRemoveROOTLibPaths) {
                  // Check this dir
                  DIR *dir = opendir(ldir.c_str());
                  if (dir) {
                     // Scan the directory
                     struct dirent *ent = 0;
                     while ((ent = (struct dirent *)readdir(dir))) {
                        if (!strncmp(ent->d_name, "libCore", 7)) {
                           remove = 1;
                           break;
                        }
                     }
                     // Close the directory
                     closedir(dir);
                  }
               }
            }
            if (!remove) {
               if (fBareLibPath.length() > 0)
                  fBareLibPath += ":";
               fBareLibPath += ldir;
            }
         }
         ctrim = " (lib paths filter applied)";
      } else {
         // Full path
         ctrim = " (full ";
         ctrim += XPD_LIBPATH;
         ctrim += ")";
         fBareLibPath = getenv(XPD_LIBPATH);
      }
      TRACE(ALL, "bare lib path for proofserv" << ctrim <<": " << fBareLibPath);
   }

   // Groups
   if (!fGroupsMgr)
      // Create default group, if none explicitly requested
      fGroupsMgr = new XrdProofGroupMgr;

   if (fGroupsMgr)
      fGroupsMgr->Print(0);

   // Config the admin handler
   if (fAdmin && fAdmin->Config(rcf) != 0) {
      XPDERR("problems configuring the admin handler");
      return -1;
   }

   // Config the network manager
   if (fNetMgr && fNetMgr->Config(rcf) != 0) {
      XPDERR("problems configuring the network manager");
      return -1;
   }

   // Config the priority manager
   if (fPriorityMgr && fPriorityMgr->Config(rcf) != 0) {
      XPDERR("problems configuring the priority manager");
      return -1;
   }

   // Config the ROOT versions manager
   if (fROOTMgr) {
      fROOTMgr->SetLogDir(fAdminPath.c_str());
      if (fROOTMgr && fROOTMgr->Config(rcf) != 0) {
         XPDERR("problems configuring the ROOT versions manager");
         return -1;
      }
   }

   // Config the client manager
   if (fClientMgr && fClientMgr->Config(rcf) != 0) {
      XPDERR("problems configuring the client manager");
      return -1;
   }

   // Config the session manager
   if (fSessionMgr && fSessionMgr->Config(rcf) != 0) {
      XPDERR("problems configuring the session manager");
      return -1;
   }

   // Config the scheduler
   if (fProofSched && fProofSched->Config(rcf) != 0) {
      XPDERR("problems configuring the scheduler");
      return -1;
   }

   // Xrootd protocol service
   if (!(fXrootd = LoadXrootd(fParms, fPi, fEDest))) {
      TRACE(ALL, "file serving (protocol: 'root://') not available");
   }

   if (!rcf) {
      // Start cron thread
      pthread_t tid;
      if (XrdSysThread::Run(&tid, XrdProofdManagerCron,
                            (void *)this, 0, "ProofdManager cron thread") != 0) {
         XPDERR("could not start cron thread");
         return 0;
      }
      TRACE(ALL, "manager cron thread started");
   }

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Validate local dataset src at URL (check the URL and make the relevant
/// directories).
/// Return 1 if OK, 0 if any problem arises

bool XrdProofdManager::ValidateLocalDataSetSrc(XrdOucString &url, bool &local)
{
   XPDLOC(ALL, "Manager::ValidateLocalDataSetSrc")

   TRACE(ALL, "validating '" << url << "' ...");
   local = 0;
   bool goodsrc = 1;
   if (url.length() > 0) {
      // Check if local source
      if (url.beginswith("file:")) url.replace("file:", "");
      if (url.beginswith("/")) {
         local = 1;
         goodsrc = 0;
         // Make sure the directory exists and has mode 0755
         XrdProofUI ui;
         XrdProofdAux::GetUserInfo(XrdProofdProtocol::EUidAtStartup(), ui);
         if (XrdProofdAux::AssertDir(url.c_str(), ui, ChangeOwn()) == 0) {
            goodsrc = 1;
            if (XrdProofdAux::ChangeMod(url.c_str(), 0777) != 0) {
               TRACE(XERR, "Problems setting permissions 0777 on path '" << url << "'");
            }
         } else {
            TRACE(XERR, "Cannot assert path '" << url << "' - ignoring");
         }
         if (goodsrc) {
            // Assert the file with dataset summaries
            XrdOucString fnpath(url.c_str());
            fnpath += "/dataset.list";
            if (access(fnpath.c_str(), F_OK) != 0) {
               FILE *flst = fopen(fnpath.c_str(), "w");
               if (!flst) {
                  TRACE(XERR, "Cannot open file '" << fnpath << "' for the dataset list; errno: " << errno);
                  goodsrc = 0;
               } else {
                  if (fclose(flst) != 0)
                     TRACE(XERR, "Problems closing file '" << fnpath << "'; errno: " << errno);
                  if (XrdProofdAux::ChangeOwn(fnpath.c_str(), ui) != 0) {
                     TRACE(XERR, "Problems asserting ownership of " << fnpath);
                  }
               }
            }
            // Make sure that everybody can modify the file for updates
            if (goodsrc && XrdProofdAux::ChangeMod(fnpath.c_str(), 0666) != 0) {
               TRACE(XERR, "Problems setting permissions to 0666 on file '" << fnpath << "'; errno: " << errno);
               goodsrc = 0;
            }
            // Assert the file with lock file path
            if (goodsrc) {
               fnpath.replace("/dataset.list", "/lock.location");
               FILE *flck = fopen(fnpath.c_str(), "a");
               if (!flck) {
                  TRACE(XERR, "Cannot open file '" << fnpath << "' with the lock file path; errno: " << errno);
               } else {
                  errno = 0;
                  off_t ofs = lseek(fileno(flck), 0, SEEK_CUR);
                  if (ofs == 0) {
                     // New file: write the default lock file path
                     XrdOucString fnlock(url);
                     fnlock.replace("/", "%");
                     fnlock.replace(":", "%");
                     fnlock.insert("/tmp/", 0);
                     fprintf(flck, "%s\n", fnlock.c_str());
                     if (fclose(flck) != 0)
                        TRACE(XERR, "Problems closing file '" << fnpath << "'; errno: " << errno);
                     flck = 0;
                     if (XrdProofdAux::ChangeOwn(fnpath.c_str(), ui) != 0) {
                        TRACE(XERR, "Problems asserting ownership of " << fnpath);
                     }
                  } else if (ofs == (off_t)(-1)) {
                     TRACE(XERR, "Problems getting current position on file '" << fnpath << "'; errno: " << errno);
                  }
                  if (flck && fclose(flck) != 0)
                     TRACE(XERR, "Problems closing file '" << fnpath << "'; errno: " << errno);
               }
            }
            // Make sure that everybody can modify the file for updates
            if (goodsrc && XrdProofdAux::ChangeMod(fnpath.c_str(), 0644) != 0) {
               TRACE(XERR, "Problems setting permissions to 0644 on file '" << fnpath << "'; errno: " << errno);
            }
         }
      }
   }
   else {
      TRACE(ALL, "New dataset with no URL!");
   }
   // Done
   return goodsrc;
}

////////////////////////////////////////////////////////////////////////////////
/// Register directives for configuration

void XrdProofdManager::RegisterDirectives()
{
   // Register special config directives
   Register("trace", new XrdProofdDirective("trace", this, &DoDirectiveClass));
   Register("groupfile", new XrdProofdDirective("groupfile", this, &DoDirectiveClass));
   Register("multiuser", new XrdProofdDirective("multiuser", this, &DoDirectiveClass));
   Register("maxoldlogs", new XrdProofdDirective("maxoldlogs", this, &DoDirectiveClass));
   Register("allow", new XrdProofdDirective("allow", this, &DoDirectiveClass));
   Register("allowedgroups", new XrdProofdDirective("allowedgroups", this, &DoDirectiveClass));
   Register("allowedusers", new XrdProofdDirective("allowedusers", this, &DoDirectiveClass));
   Register("role", new XrdProofdDirective("role", this, &DoDirectiveClass));
   Register("cron", new XrdProofdDirective("cron", this, &DoDirectiveClass));
   Register("port", new XrdProofdDirective("port", this, &DoDirectiveClass));
   Register("datadir", new XrdProofdDirective("datadir", this, &DoDirectiveClass));
   Register("datasetsrc", new XrdProofdDirective("datasetsrc", this, &DoDirectiveClass));
   Register("xrd.protocol", new XrdProofdDirective("xrd.protocol", this, &DoDirectiveClass));
   Register("filterlibpaths", new XrdProofdDirective("filterlibpaths", this, &DoDirectiveClass));
   Register("xrootd", new XrdProofdDirective("xrootd", this, &DoDirectiveClass));
   // Register config directives for strings
   Register("tmp", new XrdProofdDirective("tmp", (void *)&fTMPdir, &DoDirectiveString));
   Register("poolurl", new XrdProofdDirective("poolurl", (void *)&fPoolURL, &DoDirectiveString));
   Register("namespace", new XrdProofdDirective("namespace", (void *)&fNamespace, &DoDirectiveString));
   Register("superusers", new XrdProofdDirective("superusers", (void *)&fSuperUsers, &DoDirectiveString));
   Register("image", new XrdProofdDirective("image", (void *)&fImage, &DoDirectiveString));
   Register("workdir", new XrdProofdDirective("workdir", (void *)&fWorkDir, &DoDirectiveString));
   Register("sockpathdir", new XrdProofdDirective("sockpathdir", (void *)&fSockPathDir, &DoDirectiveString));
   Register("remoteplite", new XrdProofdDirective("remoteplite", (void *)&fRemotePLite, &DoDirectiveInt));
   Register("stagereqrepo", new XrdProofdDirective("stagereqrepo", (void *)&fStageReqRepo, &DoDirectiveString));
}

////////////////////////////////////////////////////////////////////////////////
/// Resolve special keywords in 's' for client 'pcl'. Recognized keywords
///     `<workdir>`          root for working dirs
///     `<host>`             local host name
///     `<port>`             daemon port
///     `<homedir>`          user home dir
///     `<user>`             user name
///     `<group>`            user group
///     `<uid>`              user ID
///     `<gid>`              user group ID
///     `<effuser>`          effective user name (for multiuser or user mapping modes)
/// Return the number of keywords resolved.

int XrdProofdManager::ResolveKeywords(XrdOucString &s, XrdProofdClient *pcl)
{
   XPDLOC(ALL, "Manager::ResolveKeywords")

   int nk = 0;

   TRACE(HDBG, "enter: " << s << " - WorkDir(): " << WorkDir());

   // Parse <workdir>
   if (s.replace("<workdir>", WorkDir()))
      nk++;

   TRACE(HDBG, "after <workdir>: " << s);

   // Parse <host>
   if (s.replace("<host>", Host()))
      nk++;

   TRACE(HDBG, "after <host>: " << s);

   // Parse <port>
   if (s.find("<port>") != STR_NPOS) {
      XrdOucString sport;
      sport += Port();
      if (s.replace("<port>", sport.c_str()))
         nk++;
   }

   // Parse <effuser> of the process
   if (s.find("<effuser>") != STR_NPOS) {
      XrdProofUI eui;
      if (XrdProofdAux::GetUserInfo(geteuid(), eui) == 0) {
         if (s.replace("<effuser>", eui.fUser.c_str()))
            nk++;
      }
   }

   // Parse <user>
   if (pcl)
      if (s.replace("<user>", pcl->User()))
         nk++;

   // Parse <group>
   if (pcl)
      if (s.replace("<group>", pcl->Group()))
         nk++;

   // Parse <homedir>
   if (pcl)
      if (s.replace("<homedir>", pcl->UI().fHomeDir.c_str()))
         nk++;

   // Parse <uid>
   if (pcl && (s.find("<uid>") != STR_NPOS)) {
      XrdOucString suid;
      suid += pcl->UI().fUid;
      if (s.replace("<uid>", suid.c_str()))
         nk++;
   }

   // Parse <gid>
   if (pcl && (s.find("<gid>") != STR_NPOS)) {
      XrdOucString sgid;
      sgid += pcl->UI().fGid;
      if (s.replace("<gid>", sgid.c_str()))
         nk++;
   }

   TRACE(HDBG, "exit: " << s);

   // We are done
   return nk;
}

//
// Special directive processors

////////////////////////////////////////////////////////////////////////////////
/// Update the priorities of the active sessions.

int XrdProofdManager::DoDirective(XrdProofdDirective *d,
                                  char *val, XrdOucStream *cfg, bool rcf)
{
   XPDLOC(ALL, "Manager::DoDirective")

   if (!d)
      // undefined inputs
      return -1;

   if (d->fName == "trace") {
      return DoDirectiveTrace(val, cfg, rcf);
   } else if (d->fName == "groupfile") {
      return DoDirectiveGroupfile(val, cfg, rcf);
   } else if (d->fName == "maxoldlogs") {
      return DoDirectiveMaxOldLogs(val, cfg, rcf);
   } else if (d->fName == "allow") {
      return DoDirectiveAllow(val, cfg, rcf);
   } else if (d->fName == "allowedgroups") {
      return DoDirectiveAllowedGroups(val, cfg, rcf);
   } else if (d->fName == "allowedusers") {
      return DoDirectiveAllowedUsers(val, cfg, rcf);
   } else if (d->fName == "role") {
      return DoDirectiveRole(val, cfg, rcf);
   } else if (d->fName == "multiuser") {
      return DoDirectiveMultiUser(val, cfg, rcf);
   } else if (d->fName == "port") {
      return DoDirectivePort(val, cfg, rcf);
   } else if (d->fName == "datadir") {
      return DoDirectiveDataDir(val, cfg, rcf);
   } else if (d->fName == "datasetsrc") {
      return DoDirectiveDataSetSrc(val, cfg, rcf);
   } else if (d->fName == "rootd") {
      return DoDirectiveRootd(val, cfg, rcf);
   } else if (d->fName == "rootdallow") {
      return DoDirectiveRootdAllow(val, cfg, rcf);
   } else if (d->fName == "xrd.protocol") {
      return DoDirectivePort(val, cfg, rcf);
   } else if (d->fName == "filterlibpaths") {
      return DoDirectiveFilterLibPaths(val, cfg, rcf);
   } else if (d->fName == "xrootd") {
      return DoDirectiveXrootd(val, cfg, rcf);
   }
   TRACE(XERR, "unknown directive: " << d->fName);
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Scan the config file for tracing settings

int XrdProofdManager::DoDirectiveTrace(char *val, XrdOucStream *cfg, bool)
{
   XPDLOC(ALL, "Manager::DoDirectiveTrace")

   if (!val || !cfg)
      // undefined inputs
      return -1;

   // Specifies tracing options. This works by levels and domains.
   //
   // Valid keyword levels are:
   //   err            trace errors                        [on]
   //   req            trace protocol requests             [on]*
   //   dbg            trace details about actions         [off]
   //   hdbg           trace more details about actions    [off]
   // Special forms of 'dbg' (always on if 'dbg' is required) are:
   //   login          trace details about login requests  [on]*
   //   fork           trace proofserv forks               [on]*
   //   mem            trace mem buffer manager            [off]
   //
   // Valid keyword domains are:
   //   rsp            server replies                      [off]
   //   aux            aux functions                       [on]
   //   cmgr           client manager                      [on]
   //   smgr           session manager                     [on]
   //   nmgr           network manager                     [on]
   //   pmgr           priority manager                    [on]
   //   gmgr           group manager                       [on]
   //   sched          details about scheduling            [on]
   //
   // Global switches:
   //   all or dump    full tracing of everything
   //
   // Defaults are shown in brackets; '*' shows the default when the '-d'
   // option is passed on the command line. Each option may be
   // optionally prefixed by a minus sign to turn off the setting.
   // Order matters: 'all' in last position enables everything; in first
   // position is corrected by subsequent settings
   //
   while (val && val[0]) {
      bool on = 1;
      if (val[0] == '-') {
         on = 0;
         val++;
      }
      if (!strcmp(val, "err")) {
         TRACESET(XERR, on);
      } else if (!strcmp(val, "req")) {
         TRACESET(REQ, on);
      } else if (!strcmp(val, "dbg")) {
         TRACESET(DBG, on);
         TRACESET(LOGIN, on);
         TRACESET(FORK, on);
         TRACESET(MEM, on);
      } else if (!strcmp(val, "login")) {
         TRACESET(LOGIN, on);
      } else if (!strcmp(val, "fork")) {
         TRACESET(FORK, on);
      } else if (!strcmp(val, "mem")) {
         TRACESET(MEM, on);
      } else if (!strcmp(val, "hdbg")) {
         TRACESET(HDBG, on);
         TRACESET(DBG, on);
         TRACESET(LOGIN, on);
         TRACESET(FORK, on);
         TRACESET(MEM, on);
      } else if (!strcmp(val, "rsp")) {
         TRACESET(RSP, on);
      } else if (!strcmp(val, "aux")) {
         TRACESET(AUX, on);
      } else if (!strcmp(val, "cmgr")) {
         TRACESET(CMGR, on);
      } else if (!strcmp(val, "smgr")) {
         TRACESET(SMGR, on);
      } else if (!strcmp(val, "nmgr")) {
         TRACESET(NMGR, on);
      } else if (!strcmp(val, "pmgr")) {
         TRACESET(PMGR, on);
      } else if (!strcmp(val, "gmgr")) {
         TRACESET(GMGR, on);
      } else if (!strcmp(val, "sched")) {
         TRACESET(SCHED, on);
      } else if (!strcmp(val, "all") || !strcmp(val, "dump")) {
         // Everything
         TRACE(ALL, "Setting trace: " << on);
         XrdProofdTrace->What = (on) ? TRACE_ALL : 0;
      }

      // Next
      val = cfg->GetWord();
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'groupfile' directive

int XrdProofdManager::DoDirectiveGroupfile(char *val, XrdOucStream *cfg, bool rcf)
{
   XPDLOC(ALL, "Manager::DoDirectiveGroupfile")

   if (!val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, Host()) == 0)
         return 0;

   // Defines file with the group info
   if (rcf) {
      SafeDelete(fGroupsMgr);
   } else if (fGroupsMgr) {
      TRACE(XERR, "groups manager already initialized: ignoring ");
      return -1;
   }
   fGroupsMgr = new XrdProofGroupMgr;
   fGroupsMgr->Config(val);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'maxoldlogs' directive

int XrdProofdManager::DoDirectiveMaxOldLogs(char *val, XrdOucStream *cfg, bool)
{
   if (!val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, Host()) == 0)
         return 0;

   // Max number of sessions per user
   int maxoldlogs = strtol(val, 0, 10);
   XrdProofdSandbox::SetMaxOldSessions(maxoldlogs);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'allow' directive

int XrdProofdManager::DoDirectiveAllow(char *val, XrdOucStream *cfg, bool)
{
   if (!val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, Host()) == 0)
         return 0;

   // Masters allowed to connect
   fMastersAllowed.push_back(new XrdOucString(val));
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'allowedgroups' directive

int XrdProofdManager::DoDirectiveAllowedGroups(char *val, XrdOucStream *cfg, bool)
{
   if (!val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, Host()) == 0)
         return 0;

   // We are in controlled mode
   fOperationMode = kXPD_OpModeControlled;

   // Input list (comma separated) of UNIX groups allowed to connect
   XrdOucString s = val;
   int from = 0;
   XrdOucString grp, gid;
   XrdProofGI gi;
   while ((from = s.tokenize(grp, from, ',')) != STR_NPOS) {
      int st = 1;
      if (grp.beginswith('-')) {
         st = 0;
         grp.erasefromstart(1);
      }
      // Unix or Proof group ?
      if (XrdProofdAux::GetGroupInfo(grp.c_str(), gi) == 0) {
         // Unix: add name and id
         gid.form("%d", (int) gi.fGid);
         fAllowedGroups.Add(gid.c_str(), new int(st));
      }
      // Add it to the list
      fAllowedGroups.Add(grp.c_str(), new int(st));
   }

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'allowedusers' directive

int XrdProofdManager::DoDirectiveAllowedUsers(char *val, XrdOucStream *cfg, bool)
{
   if (!val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, Host()) == 0)
         return 0;

   // We are in controlled mode
   fOperationMode = kXPD_OpModeControlled;

   // Input list (comma separated) of users allowed to connect
   XrdOucString s = val;
   int from = 0;
   XrdOucString usr;
   XrdProofUI ui;
   while ((from = s.tokenize(usr, from, ',')) != STR_NPOS) {
      int st = 1;
      if (usr.beginswith('-')) {
         st = 0;
         usr.erasefromstart(1);
      }
      // Add to the list; we will check later on the existence of the
      // user in the password file, depending on the 'multiuser' settings
      fAllowedUsers.Add(usr.c_str(), new int(st));
   }

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'role' directive

int XrdProofdManager::DoDirectiveRole(char *val, XrdOucStream *cfg, bool)
{
   if (!val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, Host()) == 0)
         return 0;

   // Role this server
   XrdOucString tval(val);
   if (tval == "supermaster") {
      fSrvType = kXPD_TopMaster;
      fSuperMst = 1;
   } else if (tval == "master") {
      fSrvType = kXPD_TopMaster;
   } else if (tval == "submaster") {
      fSrvType = kXPD_Master;
   } else if (tval == "worker") {
      fSrvType = kXPD_Worker;
   } else if (tval == "any") {
      fSrvType = kXPD_AnyServer;
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'xrd.protocol' directive to find the port

int XrdProofdManager::DoDirectivePort(char *val, XrdOucStream *, bool)
{
   if (!val)
      // undefined inputs
      return -1;

   XrdOucString port(val);
   if (port.beginswith("xproofd:")) {
      port.replace("xproofd:", "");
   }
   if (port.length() > 0 && port.isdigit()) {
      fPort = strtol(port.c_str(), 0, 10);
   }
   fPort = (fPort < 0) ? XPD_DEF_PORT : fPort;

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'multiuser' directive

int XrdProofdManager::DoDirectiveMultiUser(char *val, XrdOucStream *cfg, bool)
{
   XPDLOC(ALL, "Manager::DoDirectiveMultiUser")

   if (!val)
      // undefined inputs
      return -1;

   // Multi-user option
   int mu = strtol(val, 0, 10);
   fMultiUser = (mu == 1) ? 1 : fMultiUser;

   // Check if we need to change the working dir template
   val = cfg->GetWord();
   if (val) fMUWorkDir = val;

   TRACE(DBG, "fMultiUser: "<< fMultiUser << " work dir template: " << fMUWorkDir);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'datasetsrc' directive

int XrdProofdManager::DoDirectiveDataSetSrc(char *val, XrdOucStream *cfg, bool)
{
   if (!val)
      // undefined inputs
      return -1;

   // URL for this source
   XrdOucString type(val), url, opts, obscure;
   bool rw = 0, local = 0, goodsrc = 1;
   char *nxt = 0;
   while ((nxt = cfg->GetWord())) {
      if (!strcmp(nxt, "rw=1") || !strcmp(nxt, "rw:1")) {
         rw = 1;
      } else if (!strncmp(nxt, "url:", 4)) {
         url = nxt + 4;
         XrdClientUrlInfo u(url);
         if (u.Proto == "" && u.HostWPort == "") local = 1;
      } else if (!strncmp(nxt, "opt:", 4)) {
         opts = nxt + 4;
      } else {
         obscure += nxt;
         obscure += " ";
      }
   }

   // Add to the list
   if (goodsrc) {
      // If first local, add it in front
      std::list<XrdProofdDSInfo *>::iterator ii = fDataSetSrcs.begin();
      bool haslocal = 0;
      for (ii = fDataSetSrcs.begin(); ii != fDataSetSrcs.end(); ++ii) {
         if ((*ii)->fLocal) {
            haslocal = 1;
            break;
         }
      }
      // Default options
      if (opts.length() <= 0) {
         opts = rw ? "Ar:Av:" : "-Ar:-Av:";
      }
      XrdProofdDSInfo *dsi = new XrdProofdDSInfo(type.c_str(), url.c_str(),
         local, rw, opts.c_str(), obscure.c_str());
      if (haslocal || !local) {
         fDataSetSrcs.push_back(dsi);
      } else {
         fDataSetSrcs.push_front(dsi);
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'datadir' directive

int XrdProofdManager::DoDirectiveDataDir(char *val, XrdOucStream *cfg, bool)
{
   if (!val)
      // undefined inputs
      return -1;

   // Data directory and write permissions
   fDataDir = val;
   fDataDirOpts = "";
   fDataDirUrlOpts = "";
   XrdOucString opts;
   char *nxt = 0;
   while ((nxt = cfg->GetWord()) && (opts.length() == 0)) {
      opts = nxt;
   }
   if (opts.length() > 0) fDataDirOpts = opts;
   // Check if URL type options have been spcified in the main url
   int iq = STR_NPOS;
   if ((iq = fDataDir.rfind('?')) != STR_NPOS) {
      fDataDirUrlOpts.assign(fDataDir, iq + 1);
      fDataDir.erase(iq);
   }

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'xrootd' directive
///  xpd.xrootd [path/]libXrdXrootd.so

int XrdProofdManager::DoDirectiveXrootd(char *val, XrdOucStream *, bool)
{
   XPDLOC(ALL, "Manager::DoDirectiveXrootd")

   if (!val)
      // undefined inputs
      return -1;
   TRACE(ALL, "val: "<< val);
   // Check version (v3 does not have the plugin, loading v4 may lead to problems)
   if (XrdMajorVNUM(XrdVNUMBER) < 4) {
      TRACE(ALL, "WARNING: built against an XRootD version without libXrdXrootd.so :");
      TRACE(ALL, "WARNING:    loading external " << val << " may lead to incompatibilities");
   }

   fXrootdLibPath = val;

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'rootd' directive
/// xpd.rootd deny|allow [rootsys:`<tag>`] [path:abs-path/] [mode:ro|rw]
///            [auth:none|full] [other_rootd_args]

int XrdProofdManager::DoDirectiveRootd(char *, XrdOucStream *, bool)
{
   XPDLOC(ALL, "Manager::DoDirectiveRootd")

   TRACE(ALL, "unsupported!!! ");

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'rootdallow' directive
///  xpd.rootdallow host1,host2 host3
/// Host names may contain the wild card '*'

int XrdProofdManager::DoDirectiveRootdAllow(char *, XrdOucStream *, bool)
{
   XPDLOC(ALL, "Manager::DoDirectiveRootdAllow")

   TRACE(ALL, "unsupported!!! ");

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'filterlibpaths' directive
///  xpd.filterlibpaths 1|0 [path1,path2 path3 path4 ...]

int XrdProofdManager::DoDirectiveFilterLibPaths(char *val, XrdOucStream *cfg, bool)
{
   XPDLOC(ALL, "Manager::DoDirectiveRemoveLibPaths")

   if (!val)
      // undefined inputs
      return -1;

   // Rebuild arguments list
   fLibPathsToRemove.Purge();

   TRACE(ALL, "val: "<< val);

   // Whether to remove ROOT lib paths before adding the effective one
   fRemoveROOTLibPaths = (!strcmp(val, "1") || !strcmp(val, "yes")) ? 1 : 0;
   if (fRemoveROOTLibPaths)
      TRACE(ALL, "Filtering out ROOT lib paths from "<<XPD_LIBPATH);

   // Parse the rest, if any
   char *nxt = 0;
   while ((nxt = cfg->GetWord())) {
      XrdOucString pps(nxt), p;
      int from = 0;
      while ((from = pps.tokenize(p, from, ',')) != -1) {
         if (p.length() > 0) {
            fLibPathsToRemove.Add(p.c_str(), 0, 0, Hash_data_is_key);
            TRACE(ALL, "Filtering out from "<<XPD_LIBPATH<<" lib path '"<<p<<"'");
         }
      }
   }

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process manager request

int XrdProofdManager::Process(XrdProofdProtocol *p)
{
   XPDLOC(ALL, "Manager::Process")

   int rc = 0;
   XPD_SETRESP(p, "Process");

   TRACEP(p, REQ, "req id: " << p->Request()->header.requestid << " (" <<
          XrdProofdAux::ProofRequestTypes(p->Request()->header.requestid) << ")");

   // If the user is not yet logged in, restrict what the user can do
   if (!p->Status() || !(p->Status() & XPD_LOGGEDIN)) {
      switch (p->Request()->header.requestid) {
         case kXP_auth:
            return fClientMgr->Auth(p);
         case kXP_login:
            return fClientMgr->Login(p);
         default:
            TRACEP(p, XERR, "invalid request: " << p->Request()->header.requestid);
            response->Send(kXR_InvalidRequest, "Invalid request; user not logged in");
            return p->Link()->setEtext("protocol sequence error 1");
      }
   }

   // Once logged-in, the user can request the real actions
   XrdOucString emsg;
   switch (p->Request()->header.requestid) {
      case kXP_admin: {
         int type = ntohl(p->Request()->proof.int1);
         return fAdmin->Process(p, type);
      }
      case kXP_readbuf:
         return fNetMgr->ReadBuffer(p);
      case kXP_create:
      case kXP_destroy:
      case kXP_attach:
      case kXP_detach:
         return fSessionMgr->Process(p);
      default:
         emsg += "Invalid request: ";
         emsg += p->Request()->header.requestid;
         break;
   }

   // Notify invalid request
   response->Send(kXR_InvalidRequest, emsg.c_str());

   // Done
   return 0;
}
