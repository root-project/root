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

#ifdef OLDXRDOUC
#  include "XrdOuc/XrdOucPlugin.hh"
#  include "XrdOuc/XrdOucTimer.hh"
#else
#  include "XrdSys/XrdSysPlugin.hh"
#  include "XrdSys/XrdSysTimer.hh"
#endif
#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPriv.hh"

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

// Auxilliary sructure used internally to extract list of allowed/denied user names
// when running in access control mode
typedef struct {
   XrdOucString allowed;
   XrdOucString denied;
} xpd_acm_lists_t;

//--------------------------------------------------------------------------
//
// XrdProofdManagerCron
//
// Function run in separate thread doing regular checks
//
//--------------------------------------------------------------------------
void *XrdProofdManagerCron(void *p)
{
   // This is an endless loop to periodically check the system
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

//__________________________________________________________________________
XrdProofdManager::XrdProofdManager(XrdProtocol_Config *pi, XrdSysError *edest)
                 : XrdProofdConfig(pi->ConfigFN, edest)
{
   // Constructor

   fSrvType = kXPD_AnyServer;
   fEffectiveUser = "";
   fHost = "";
   fPort = XPD_DEF_PORT;
   fImage = "";        // image name for these servers
   fSockPathDir = "";
   fTMPdir = "/tmp";
   fWorkDir = "";
   fSuperMst = 0;
   fNamespace = "/proofpool";
   fMastersAllowed.clear();
   fOperationMode = kXPD_OpModeOpen;
   fMultiUser = 0;
   fChangeOwn = 0;
   fCronFrequency = 30;

   // Data dir
   fDataDir = "";        // Default <workdir>/<user>/data
   fDataDirOpts = "";    // Default: no action

   // Rootd file serving enabled by default in readonly mode
   fRootdExe = "<>";
   // Add mandatory arguments
   fRootdArgs.push_back(XrdOucString("-i"));
   fRootdArgs.push_back(XrdOucString("-nologin"));
   fRootdArgs.push_back(XrdOucString("-r"));            // Readonly
   fRootdArgs.push_back(XrdOucString("-noauth"));       // No auth
   // Build the argument list
   fRootdArgsPtrs = new const char *[fRootdArgs.size() + 2];
   fRootdArgsPtrs[0] = fRootdExe.c_str();
   int i = 1;
   std::list<XrdOucString>::iterator ia = fRootdArgs.begin();
   while (ia != fRootdArgs.end()) {
      fRootdArgsPtrs[i] = (*ia).c_str();
      i++; ia++;
   }
   fRootdArgsPtrs[fRootdArgs.size() + 1] = 0;
   // Started with 'system' (not 'fork')
   fRootdFork = 0;
      
   // Proof admin path
   fAdminPath = pi->AdmPath;
   fAdminPath += "/.xproofd.";

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

//__________________________________________________________________________
XrdProofdManager::~XrdProofdManager()
{
   // Destructor

   // Destroy the configuration handler
   SafeDelete(fAdmin);
   SafeDelete(fClientMgr);
   SafeDelete(fNetMgr);
   SafeDelete(fPriorityMgr);
   SafeDelete(fProofSched);
   SafeDelete(fROOTMgr);
   SafeDelete(fSessionMgr);
   SafeDelArray(fRootdArgsPtrs);
}

//__________________________________________________________________________
void XrdProofdManager::CheckLogFileOwnership()
{
   // Make sure that the log file belongs to the original effective user
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

//______________________________________________________________________________
bool XrdProofdManager::CheckMaster(const char *m)
{
   // Check if master 'm' is allowed to connect to this host
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

//_____________________________________________________________________________
int XrdProofdManager::CheckUser(const char *usr, const char *grp,
                                XrdProofUI &ui, XrdOucString &e, bool &su)
{
   // Check if the user is allowed to use the system
   // Return 0 if OK, -1 if not.

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

      // Policy: check first the general switch for groups; a user of a specific group can be
      // rejected by prefixing a '-'.
      // If a user is explicitely allowed we give the green light even if her/its group is
      // disallowed. If fAllowedUsers is empty, we just apply the group rules.
      //
      // Example:
      //
      // xpd.allowedgroups z2
      // xpd.allowedusers -jgrosseo,ganis
      //
      // accepts connections from all group 'z2' except user 'jgrosseo' and from user 'ganis'
      // even if not belonging to group 'z2'.

      bool grpok = 1;
      // Check unix group
      if (fAllowedGroups.Num() > 0) {
         // Reset the flag
         grpok = 0;
         bool ugrpok = 0, pgrpok = 0;
         // Check UNIX group info
         XrdProofGI gi;
         if (XrdProofdAux::GetGroupInfo(ui.fGid, gi) == 0) {
            int *st = fAllowedGroups.Find(gi.fGroup.c_str());
            if (st) {
               if (*st == 1) {
                  ugrpok = 1;
               } else {
                  e = "Controlled access: user '";
                  e += usr;
                  e = "', UNIX group '";
                  e += gi.fGroup;
                  e += "' denied to connect";
               }
            } else {
               ugrpok = 1;
            }
         }
         // Check PROOF group info
         int *st = fAllowedGroups.Find(grp);
         if (st) {
            if (*st == 1) {
               pgrpok = 1;
            } else {
               if (e.length() <= 0)
                  e = "Controlled access: ";
               e += "; user '";
               e += usr;
               e += "', PROOF group '";
               e += grp;
               e += "' denied to connect";
            }
         } else {
            pgrpok = 1;
         }
         // Both must be true
         grpok = (ugrpok && pgrpok) ? 1 : 0;
      }
      // Check username
      bool usrok = grpok;
      if (fAllowedUsers.Num() > 0) {
         // Reset the flag
         usrok = 0;
         // Look into the hash
         int *st = fAllowedUsers.Find(usr);
         if (st) {
            if (*st == 1) {
               usrok = 1;
            } else {
               e = "Controlled access: user '";
               e += usr;
               e += "' is not allowed to connect";
               usrok = 0;
            }
         }
      }
      // Super users are always allowed
      if (!usrok && su) {
         usrok = 1;
         e = "";
      }
      // Return now if disallowed
      if (!usrok) return -1;
   }

   // OK
   return 0;
}

//_____________________________________________________________________________
XrdProofSched *XrdProofdManager::LoadScheduler()
{
   // Load PROOF scheduler
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
         return (XrdProofSched *)0;
      }
   }
   // Check result
   if (!(sched->IsValid())) {
      TRACE(XERR, " unable to instantiate the " << sched->Name() << " scheduler using " << cfn);
      delete sched;
      return (XrdProofSched *)0;
   }
   // Notify
   TRACE(ALL, "scheduler loaded: type: " << sched->Name());

   // All done
   return sched;
}

//__________________________________________________________________________
int XrdProofdManager::GetWorkers(XrdOucString &lw, XrdProofdProofServ *xps,
                                 const char *query)
{
   // Get a list of workers from the available resource broker
   XPDLOC(ALL, "Manager::GetWorkers")

   int rc = 0;
   TRACE(REQ, "enter");

   // We need the scheduler at this point
   if (!fProofSched) {
      TRACE(XERR, "scheduler undefined");
      return -1;
   }

   // Query the scheduler for the list of workers
   std::list<XrdProofWorker *> wrks;
   if ((rc = fProofSched->GetWorkers(xps, &wrks, query)) < 0) {
      TRACE(XERR, "error getting list of workers from the scheduler");
      return -1;
   }
   // If we got a new list we save it into the session object
   if (rc == 0) {

      TRACE(DBG, "list size: " << wrks.size());

      // The full list
      XrdOucString ord;
      int ii = -1;
      std::list<XrdProofWorker *>::iterator iw;
      for (iw = wrks.begin(); iw != wrks.end() ; iw++) {
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

   int proto = (xps->ROOT()) ? xps->ROOT()->SrvProtVers() : -1;
   if (rc != 2 || (proto < 21 && rc == 0)) {
      // Get the list in exported format
      xps->ExportWorkers(lw);
      TRACE(DBG, "from ExportWorkers: " << lw);
   } else if (proto >= 21) {
      // Signal enqueing
      lw = XPD_GW_QueryEnqueued;
   }

   if (TRACING(REQ)) fNetMgr->Dump();

   return rc;
}

//______________________________________________________________________________
static int FillKeyValues(const char *k, int *d, void *s)
{
   // Add the key value in the string passed via the void argument

   xpd_acm_lists_t *ls = (xpd_acm_lists_t *)s;

   if (ls) {
      XrdOucString &ss = (*d == 1) ? ls->allowed : ls->denied;
      // If not empty add a separation ','
      if (ss.length() > 0) ss += ",";
      // Add the key
      if (k) ss += k;
   } else {
      // Not enough info: stop
      return 1;
   }

   // Check next
   return 0;
}

//______________________________________________________________________________
static int RemoveInvalidUsers(const char *k, int *, void *s)
{
   // Add the key value in the string passed via the void argument

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

//______________________________________________________________________________
int XrdProofdManager::Config(bool rcf)
{
   // Run configuration and parse the entered config directives.
   // Return 0 on success, -1 on error
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
      char *host = XrdNetDNS::getHostName();
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
   if (fWorkDir.length() > 0) {
      // Make sure it exists
      if (XrdProofdAux::AssertDir(fWorkDir.c_str(), ui, fChangeOwn) != 0) {
         XPDERR("unable to assert working dir: " << fWorkDir);
         return -1;
      }
      TRACE(ALL, "working directories under: " << fWorkDir);
      // Communicate it to the sandbox service
      XrdProofdSandbox::SetWorkdir(fWorkDir.c_str());
   }

   // Data directory, if specified
   if (fDataDir.length() > 0) {
      // Make sure it exists
      if (XrdProofdAux::AssertDir(fDataDir.c_str(), ui, fChangeOwn) != 0) {
         XPDERR("unable to assert data dir: " << fDataDir);
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

   // Validate dataset sources (if not worker)
   fDataSetExp = "";
   if (fSrvType != kXPD_Worker && fDataSetSrcs.size() > 0) {
      // If first local, add it in front
      std::list<XrdProofdDSInfo *>::iterator ii = fDataSetSrcs.begin();
      bool goodsrc = 0;
      for (ii = fDataSetSrcs.begin(); ii != fDataSetSrcs.end();) {
         if (!(goodsrc = ValidateLocalDataSetSrc((*ii)->fUrl, (*ii)->fLocal))) {
            XPDERR("source " << (*ii)->fUrl << " could not be validated");
            ii = fDataSetSrcs.erase(ii);
         } else {
            // Check next
            ii++;
         }
      }
      if (fDataSetSrcs.size() > 0) {
         TRACE(ALL, fDataSetSrcs.size() << " dataset sources defined");
         for (ii = fDataSetSrcs.begin(); ii != fDataSetSrcs.end(); ii++) {
            TRACE(ALL, " url:" << (*ii)->fUrl << ", local:" << (*ii)->fLocal << ", rw:" << (*ii)->fRW);
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
      // Try to remove existing ROOT dirs in the path
      XrdOucString paths = getenv(XPD_LIBPATH);
      XrdOucString ldir;
      int from = 0;
      while ((from = paths.tokenize(ldir, from, ':')) != STR_NPOS) {
         bool isROOT = 0;
         if (ldir.length() > 0) {
            // Check this dir
            DIR *dir = opendir(ldir.c_str());
            if (dir) {
               // Scan the directory
               struct dirent *ent = 0;
               while ((ent = (struct dirent *)readdir(dir))) {
                  if (!strncmp(ent->d_name, "libCore", 7)) {
                     isROOT = 1;
                     break;
                  }
               }
               // Close the directory
               closedir(dir);
            }
            if (!isROOT) {
               if (fBareLibPath.length() > 0)
                  fBareLibPath += ":";
               fBareLibPath += ldir;
            }
         }
      }
      TRACE(ALL, "bare lib path for proofserv: " << fBareLibPath);
   }

   // Groups
   if (!fGroupsMgr)
      // Create default group, if none explicitely requested
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

   // File server
   if (fRootdExe.length() > 0) {
      // Absolute or relative?
      if (!fRootdExe.beginswith("/")) {
         if (fROOTMgr) {
            XrdOucString rtag;
            if (fRootdExe.beginswith("<") && fRootdExe.endswith(">")) {
               if (fRootdExe.length() > 2) rtag.assign(fRootdExe, 1, fRootdExe.length() - 2);
               fRootdExe = "rootd";
               fRootdArgsPtrs[0] = fRootdExe.c_str();
            }
            XrdROOT *roo = 0;
            if (rtag.length() <= 0 || !(roo = fROOTMgr->GetVersion(rtag.c_str())))
               roo = fROOTMgr->DefaultVersion();
            if (roo && strlen(roo->BinDir()) > 0) {
               XrdOucString bindir(roo->BinDir());
               if (!bindir.endswith("/")) bindir += "/";
               fRootdExe.insert(bindir, 0);
               fRootdArgsPtrs[0] = fRootdExe.c_str();
            }
         }
      }
      // Create unix socket where to accepts callbacks from rootd launchers
      XrdOucString sockpath;
      XPDFORM(sockpath, "%s/xpd.%d.%d.rootd", fSockPathDir.c_str(), fPort, getpid());
      fRootdUnixSrv = new rpdunixsrv(sockpath.c_str());
      if (!fRootdUnixSrv || (fRootdUnixSrv && !fRootdUnixSrv->isvalid(0))) {
         XPDERR("could not start unix server connection on path "<<
                sockpath<<" - errno: "<<(int)errno);
         fRootdExe = "";
         return -1;
      }
      TRACE(ALL, "unix socket path for rootd call backs: "<<sockpath);
      // Check if access is controlled
      if (fRootdAllow.size() > 0) {
         XrdOucString hhs;
         std::list<XrdOucString>::iterator ia = fRootdAllow.begin();
         while (ia != fRootdAllow.end()) {
            if (hhs.length() > 0) hhs += ",";
            hhs += (*ia).c_str();
            ia++;
         }
         TRACE(ALL, "serving files with: '" << fRootdExe <<"' (protocol: 'rootd://') to ALLOWED hosts");
         TRACE(ALL, "rootd-allowed hosts: "<< hhs);
      } else {
         TRACE(ALL, "serving files with: '" << fRootdExe <<"' (protocol: 'rootd://') to ALL hosts");
      }
      
   } else {
      TRACE(ALL, "file serving (protocol: 'rootd://') explicitly disabled");
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

//______________________________________________________________________________
bool XrdProofdManager::ValidateLocalDataSetSrc(XrdOucString &url, bool &local)
{
   // Validate local dataset src at URL (check the URL and make the relevant
   // directories).
   // Return 1 if OK, 0 if any problem arises
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
               if (access(fnpath.c_str(), F_OK) != 0) {
                  FILE *flck = fopen(fnpath.c_str(), "w");
                  if (!flck) {
                     TRACE(XERR, "Cannot open file '" << fnpath << "' with the lock file path; errno: " << errno);
                  } else {
                     // Write the default lock file path
                     XrdOucString fnlock(url);
                     fnlock.replace("/", "%");
                     fnlock.replace(":", "%");
                     fnlock.insert("/tmp/", 0);
                     fprintf(flck, "%s\n", fnlock.c_str());
                     if (fclose(flck) != 0)
                        TRACE(XERR, "Problems closing file '" << fnpath << "'; errno: " << errno);
                     if (XrdProofdAux::ChangeOwn(fnpath.c_str(), ui) != 0) {
                        TRACE(XERR, "Problems asserting ownership of " << fnpath);
                     }
                  }
               }
            }
            // Make sure that everybody can modify the file for updates
            if (goodsrc && XrdProofdAux::ChangeMod(fnpath.c_str(), 0644) != 0) {
               TRACE(XERR, "Problems setting permissions to 0644 on file '" << fnpath << "'; errno: " << errno);
            }
         }
      }
   }
   // Done
   return goodsrc;
}

//______________________________________________________________________________
void XrdProofdManager::RegisterDirectives()
{
   // Register directives for configuration

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
   Register("rootd", new XrdProofdDirective("rootd", this, &DoDirectiveClass));
   Register("rootdallow", new XrdProofdDirective("rootdallow", this, &DoDirectiveClass));
   Register("xrd.protocol", new XrdProofdDirective("xrd.protocol", this, &DoDirectiveClass));
   // Register config directives for strings
   Register("tmp", new XrdProofdDirective("tmp", (void *)&fTMPdir, &DoDirectiveString));
   Register("poolurl", new XrdProofdDirective("poolurl", (void *)&fPoolURL, &DoDirectiveString));
   Register("namespace", new XrdProofdDirective("namespace", (void *)&fNamespace, &DoDirectiveString));
   Register("superusers", new XrdProofdDirective("superusers", (void *)&fSuperUsers, &DoDirectiveString));
   Register("image", new XrdProofdDirective("image", (void *)&fImage, &DoDirectiveString));
   Register("workdir", new XrdProofdDirective("workdir", (void *)&fWorkDir, &DoDirectiveString));
   Register("sockpathdir", new XrdProofdDirective("sockpathdir", (void *)&fSockPathDir, &DoDirectiveString));
}

//______________________________________________________________________________
int XrdProofdManager::ResolveKeywords(XrdOucString &s, XrdProofdClient *pcl)
{
   // Resolve special keywords in 's' for client 'pcl'. Recognized keywords
   //     <workdir>          root for working dirs
   //     <host>             local host name
   //     <homedir>          user home dir
   //     <user>             user name
   //     <group>            user group
   //     <uid>              user ID
   //     <gid>              user group ID
   // Return the number of keywords resolved.
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

//______________________________________________________________________________
int XrdProofdManager::DoDirective(XrdProofdDirective *d,
                                  char *val, XrdOucStream *cfg, bool rcf)
{
   // Update the priorities of the active sessions.
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
   }
   TRACE(XERR, "unknown directive: " << d->fName);
   return -1;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveTrace(char *val, XrdOucStream *cfg, bool)
{
   // Scan the config file for tracing settings
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

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveGroupfile(char *val, XrdOucStream *cfg, bool rcf)
{
   // Process 'groupfile' directive
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

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveMaxOldLogs(char *val, XrdOucStream *cfg, bool)
{
   // Process 'maxoldlogs' directive

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

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveAllow(char *val, XrdOucStream *cfg, bool)
{
   // Process 'allow' directive

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

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveAllowedGroups(char *val, XrdOucStream *cfg, bool)
{
   // Process 'allowedgroups' directive

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
   XrdOucString grp;
   XrdProofGI gi;
   while ((from = s.tokenize(grp, from, ',')) != STR_NPOS) {
      int st = 1;
      if (grp.beginswith('-')) {
         st = 0;
         grp.erasefromstart(1);
      }
      // Add it to the list (no check for the group file: we support also
      // PROOF groups)
      fAllowedGroups.Add(grp.c_str(), new int(st));
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveAllowedUsers(char *val, XrdOucStream *cfg, bool)
{
   // Process 'allowedusers' directive

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

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveRole(char *val, XrdOucStream *cfg, bool)
{
   // Process 'role' directive
#if defined(BUILD_BONJOUR)
   XPDLOC(ALL, "Manager::DoDirectiveRole")
#endif

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

#if defined(BUILD_BONJOUR)
   // Check the compatibility of the roles and give a warning to the user.
   if (!XrdProofdNetMgr::CheckBonjourRoleCoherence(SrvType(), fNetMgr->GetBonjourRequestedServiceType())) {
      TRACE(XERR, "Warning: xpd.role directive and xpd.bonjour service selection are not compatible");
   }
#endif

   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectivePort(char *val, XrdOucStream *, bool)
{
   // Process 'xrd.protocol' directive to find the port

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

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveMultiUser(char *val, XrdOucStream *cfg, bool)
{
   // Process 'multiuser' directive

   if (!val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, Host()) == 0)
         return 0;

   // Multi-user option
   int mu = strtol(val, 0, 10);
   fMultiUser = (mu == 1) ? 1 : fMultiUser;
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveDataSetSrc(char *val, XrdOucStream *cfg, bool)
{
   // Process 'datasetsrc' directive

   if (!val)
      // undefined inputs
      return -1;

   // URL for this source
   XrdOucString type(val), url, opts;
   bool rw = 0, local = 0, goodsrc = 1;
   char *nxt = 0;
   while ((nxt = cfg->GetWord())) {
      if (!strcmp(nxt, "rw=1") || !strcmp(nxt, "rw:1")) {
         rw = 1;
      } else if (!strncmp(nxt, "url:", 4)) {
         url = nxt + 4;
      } else if (!strncmp(nxt, "opt:", 4)) {
         opts = nxt + 4;
      }
   }

   // Add to the list
   if (goodsrc) {
      // If first local, add it in front
      std::list<XrdProofdDSInfo *>::iterator ii = fDataSetSrcs.begin();
      bool haslocal = 0;
      for (ii = fDataSetSrcs.begin(); ii != fDataSetSrcs.end(); ii++) {
         if ((*ii)->fLocal) {
            haslocal = 1;
            break;
         }
      }
      // Default options
      if (opts.length() <= 0) {
         opts = rw ? "Ar:Av:" : "-Ar:-Av:";
      }
      if (haslocal || !local) {
         fDataSetSrcs.push_back(new XrdProofdDSInfo(type.c_str(), url.c_str(), local, rw, opts.c_str()));
      } else {
         fDataSetSrcs.push_front(new XrdProofdDSInfo(type.c_str(), url.c_str(), local, rw, opts.c_str()));
      }
   }
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveDataDir(char *val, XrdOucStream *cfg, bool)
{
   // Process 'datadir' directive

   if (!val)
      // undefined inputs
      return -1;

   // Data directory and write permissions
   fDataDir = val;
   XrdOucString opts;
   char *nxt = 0;
   while ((nxt = cfg->GetWord()) && (opts.length() == 0)) {
      opts = nxt;
   }
   if (opts.length() > 0) fDataDirOpts = opts;

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveRootd(char *val, XrdOucStream *cfg, bool)
{
   // Process 'rootd' directive
   //  xpd.rootd deny|allow [rootsys:<tag>] [path:abs-path/] [mode:ro|rw] 
   //            [auth:none|full] [other_rootd_args]
   XPDLOC(ALL, "Manager::DoDirectiveRootd")

   if (!val)
      // undefined inputs
      return -1;

   // Rebuild arguments list
   fRootdArgs.clear();
   SafeDelArray(fRootdArgsPtrs);

   TRACE(ALL, "val: "<< val);

   // Parse directive
   XrdOucString mode("ro"), auth("none"), fork("0");
   bool denied = 0;
   char *nxt = val;
   do {
      if (!strcmp(nxt, "deny") || !strcmp(nxt, "disable") || !strcmp(nxt, "off")) {
         denied = 1;
         fRootdExe = "";
      } else if (!strcmp(nxt, "allow") || !strcmp(nxt, "enable") || !strcmp(nxt, "on")) {
         denied = 0;
         fRootdExe = "<>";
      } else if (!strncmp(nxt, "mode:", 5)) {
         mode = nxt + 5;
      } else if (!strncmp(nxt, "auth:", 5)) {
         auth = nxt + 5;
      } else if (!strncmp(nxt, "fork:", 5)) {
         fork = nxt + 5;
      } else {
         // Assume rootd argument
         fRootdArgs.push_back(XrdOucString(nxt));
      }
   } while ((nxt = cfg->GetWord()));

   if (!denied) {
      // If no exec given assume 'rootd' in the default path
      if (fRootdExe.length() <= 0) fRootdExe = "<>";
      // Add mandatory arguments
      fRootdArgs.push_back(XrdOucString("-i"));
      fRootdArgs.push_back(XrdOucString("-nologin"));
      if (mode == "ro") fRootdArgs.push_back(XrdOucString("-r"));
      if (auth == "none") fRootdArgs.push_back(XrdOucString("-noauth"));
      fRootdFork = (fork == "1" || fork == "yes") ? 1 : 0;
   } else {
      // Nothing else to do, if denied
      return 0;
   }
      
   // Build the argument list
   fRootdArgsPtrs = new const char *[fRootdArgs.size() + 2];
   fRootdArgsPtrs[0] = fRootdExe.c_str();
   int i = 1;
   std::list<XrdOucString>::iterator ia = fRootdArgs.begin();
   while (ia != fRootdArgs.end()) {
      fRootdArgsPtrs[i] = (*ia).c_str();
      i++; ia++;
   }
   fRootdArgsPtrs[fRootdArgs.size() + 1] = 0;

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveRootdAllow(char *val, XrdOucStream *cfg, bool)
{
   // Process 'rootdallow' directive
   //  xpd.rootdallow host1,host2 host3
   // Host names may contain the wild card '*'
   XPDLOC(ALL, "Manager::DoDirectiveRootdAllow")

   if (!val)
      // undefined inputs
      return -1;

   TRACE(ALL, "val: "<< val);

   // Parse directive
   XrdOucString hosts, h;
   char *nxt = val;
   do {
      hosts = nxt;
      int from = 0;
      while ((from = hosts.tokenize(h, from, ',')) != -1) {
         if (h.length() > 0) fRootdAllow.push_back(h);
      }
   } while ((nxt = cfg->GetWord()));

   // Done
   return 0;
}

//______________________________________________________________________________
bool XrdProofdManager::IsRootdAllowed(const char *host)
{
   // Check if 'host' is allowed to access files via rootd
   XPDLOC(ALL, "Manager::IsRootdAllowed")

   // Check if access is controlled
   if (fRootdAllow.size() <= 0) return 1;

   // Need an host name
   if (!host || strlen(host) <= 0) return 0;

   TRACE(DBG, "checking host: "<< host);
   
   XrdOucString h(host);
   std::list<XrdOucString>::iterator ia = fRootdAllow.begin();
   while (ia != fRootdAllow.end()) {
      if (h.matches((*ia).c_str(), '*') > 0) return 1;
      ia++;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::Process(XrdProofdProtocol *p)
{
   // Process manager request
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
