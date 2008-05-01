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
// Class mapping manager fonctionality.                                 //
// On masters it keeps info about the available worker nodes and allows //
// communication with them.                                             //
// On workers it handles the communication with the master.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef OLDXRDOUC
#  include "XrdOuc/XrdOucError.hh"
#  include "XrdOuc/XrdOucLogger.hh"
#  include "XrdOuc/XrdOucPlugin.hh"
#  define XPD_LOG_01 OUC_LOG_01
#else
#  include "XrdSys/XrdSysError.hh"
#  include "XrdSys/XrdSysLogger.hh"
#  include "XrdSys/XrdSysPlugin.hh"
#  define XPD_LOG_01 SYS_LOG_01
#endif

#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XProofProtocol.h"
#include "XrdProofConn.h"
#include "XrdProofGroup.h"
#include "XrdProofdClient.h"
#include "XrdProofdManager.h"
#include "XrdProofdPlatform.h"
#include "XrdProofdProtocol.h"
#include "XrdProofdResponse.h"
#include "XrdProofSched.h"
#include "XrdProofServProxy.h"
#include "XrdProofWorker.h"
#include "XrdROOT.h"

// Tracing utilities
#include "XrdProofdTrace.h"
static const char *gTraceID = " ";
extern XrdOucTrace *XrdProofdTrace;
#define TRACEID gTraceID

// Security handle
typedef XrdSecService *(*XrdSecServLoader_t)(XrdSysLogger *, const char *cfn);

//______________________________________________________________________________
int DoMgrDirective(XrdProofdDirective *d, char *val, XrdOucStream *cfg, bool rcf)
{
   // Generic directive processor

   if (!d || !(d->fVal))
      // undefined inputs
      return -1;

   return ((XrdProofdManager *)d->fVal)->ProcessDirective(d, val, cfg, rcf);
}

//__________________________________________________________________________
XrdProofdManager::XrdProofdManager()
{
   // Constructor

   fSrvType  = kXPD_AnyServer;
   fResourceType = kRTStatic;
   fEffectiveUser = "";
   fHost = "";
   fPort = XPD_DEF_PORT;
   fImage = "";        // image name for these servers
   fWorkDir = "";
   fPROOFcfg.fName = "";
   fPROOFcfg.fMtime = 0;
   fWorkers.clear();
   fNodes.clear();
   fNumLocalWrks = XrdProofdAux::GetNumCPUs();
   fEDest = 0;
   fSuperMst = 0;
   fRequestTO = 30;
   fCIA = 0;
   fROOT.clear();
   fNamespace = "/proofpool";
   fMastersAllowed.clear();
   fPriorities.clear();
   fWorkerUsrCfg = 0;
   fShutdownOpt = 1;
   fShutdownDelay = 0;
   fCron = 1;
   fCronFrequency = 60;
   fOperationMode = kXPD_OpModeOpen;
   fMultiUser = (!getuid()) ? 1 : 0;
   fChangeOwn = 0;
   fGroupsMgr = 0;
   fProofSched = 0;
   fOverallInflate = 1;
   fSchedOpt = kXPD_sched_off;
   fPriorityMax = 20;
   fPriorityMin = 1;
   fProofdClients.clear();
   fTerminatedProcess.clear();

   // Register (re-)config directives 
   RegisterConfigDirectives();
   RegisterReConfigDirectives();
}

//__________________________________________________________________________
XrdProofdManager::~XrdProofdManager()
{
   // Destructor

   // Cleanup the worker list and create default info
   std::list<XrdProofWorker *>::iterator w = fWorkers.begin();
   while (w != fWorkers.end()) {
      delete *w;
      w = fWorkers.erase(w);
   }
   // The nodes list points to the same object, no cleanup is needed
}

//__________________________________________________________________________
void XrdProofdManager::RegisterConfigDirectives()
{
   // Register directives for configuration

   // Register special config directives
   fConfigDirectives.Add("resource", new XrdProofdDirective("resource", this, &DoMgrDirective));
   fConfigDirectives.Add("groupfile", new XrdProofdDirective("groupfile", this, &DoMgrDirective));
   fConfigDirectives.Add("multiuser", new XrdProofdDirective("multiuser", this, &DoMgrDirective));
   fConfigDirectives.Add("priority", new XrdProofdDirective("priority", this, &DoMgrDirective));
   fConfigDirectives.Add("rootsys", new XrdProofdDirective("rootsys", this, &DoMgrDirective));
   fConfigDirectives.Add("shutdown", new XrdProofdDirective("shutdown", this, &DoMgrDirective));
   fConfigDirectives.Add("adminreqto", new XrdProofdDirective("adminreqto", this, &DoMgrDirective));
   fConfigDirectives.Add("maxoldlogs", new XrdProofdDirective("maxoldlogs", this, &DoMgrDirective));
   fConfigDirectives.Add("allow", new XrdProofdDirective("allow", this, &DoMgrDirective));
   fConfigDirectives.Add("allowedusers", new XrdProofdDirective("allowedusers", this, &DoMgrDirective));
   fConfigDirectives.Add("schedopt", new XrdProofdDirective("schedopt", this, &DoMgrDirective));
   fConfigDirectives.Add("role", new XrdProofdDirective("role", this, &DoMgrDirective));
   fConfigDirectives.Add("cron", new XrdProofdDirective("cron", this, &DoMgrDirective));
   fConfigDirectives.Add("xrd.protocol", new XrdProofdDirective("xrd.protocol", this, &DoMgrDirective));
   fConfigDirectives.Add("xrootd.seclib", new XrdProofdDirective("xrootd.seclib", this, &DoMgrDirective));
   // Register config directives for strings
   fConfigDirectives.Add("seclib", new XrdProofdDirective("seclib", (void *)&fSecLib, &DoDirectiveString));
   fConfigDirectives.Add("tmp", new XrdProofdDirective("tmp", (void *)&fTMPdir, &DoDirectiveString));
   fConfigDirectives.Add("poolurl", new XrdProofdDirective("poolurl", (void *)&fPoolURL, &DoDirectiveString));
   fConfigDirectives.Add("namespace", new XrdProofdDirective("namespace", (void *)&fNamespace, &DoDirectiveString));
   fConfigDirectives.Add("superusers", new XrdProofdDirective("superusers", (void *)&fSuperUsers, &DoDirectiveString));
   fConfigDirectives.Add("image", new XrdProofdDirective("image", (void *)&fImage, &DoDirectiveString));
   fConfigDirectives.Add("workdir", new XrdProofdDirective("workdir", (void *)&fWorkDir, &DoDirectiveString));
   fConfigDirectives.Add("proofplugin", new XrdProofdDirective("proofplugin", (void *)&fProofPlugin, &DoDirectiveString));
   // Register config directives for ints
   fConfigDirectives.Add("localwrks", new XrdProofdDirective("localwrks", (void *)&fNumLocalWrks, &DoDirectiveInt));
}

//__________________________________________________________________________
void XrdProofdManager::RegisterReConfigDirectives()
{
   // Register directives that can re-configure their values

   // Register special config directives
   fReConfigDirectives.Add("groupfile", new XrdProofdDirective("groupfile", this, &DoMgrDirective));
   fReConfigDirectives.Add("multiuser", new XrdProofdDirective("multiuser", this, &DoMgrDirective));
   fReConfigDirectives.Add("priority", new XrdProofdDirective("priority", this, &DoMgrDirective));
   fReConfigDirectives.Add("rootsys", new XrdProofdDirective("rootsys", this, &DoMgrDirective));
   fReConfigDirectives.Add("shutdown", new XrdProofdDirective("shutdown", this, &DoMgrDirective));
   fReConfigDirectives.Add("adminreqto", new XrdProofdDirective("adminreqto", this, &DoMgrDirective));
   fReConfigDirectives.Add("maxoldlogs", new XrdProofdDirective("maxoldlogs", this, &DoMgrDirective));
   fReConfigDirectives.Add("allow", new XrdProofdDirective("allow", this, &DoMgrDirective));
   fReConfigDirectives.Add("allowedusers", new XrdProofdDirective("allowedusers", this, &DoMgrDirective));
   fReConfigDirectives.Add("schedopt", new XrdProofdDirective("schedopt", this, &DoMgrDirective));
   fReConfigDirectives.Add("cron", new XrdProofdDirective("cron", this, &DoMgrDirective));
   // Register config directives for strings
   fReConfigDirectives.Add("tmp", new XrdProofdDirective("tmp", (void *)&fTMPdir, &DoDirectiveString));
   fReConfigDirectives.Add("poolurl", new XrdProofdDirective("poolurl", (void *)&fPoolURL, &DoDirectiveString));
   fReConfigDirectives.Add("namespace", new XrdProofdDirective("namespace", (void *)&fNamespace, &DoDirectiveString));
   fReConfigDirectives.Add("superusers",
                           new XrdProofdDirective("superusers", (void *)&fSuperUsers, &DoDirectiveString));
   fReConfigDirectives.Add("image", new XrdProofdDirective("image", (void *)&fImage, &DoDirectiveString));
   fReConfigDirectives.Add("workdir", new XrdProofdDirective("workdir", (void *)&fWorkDir, &DoDirectiveString));
   fReConfigDirectives.Add("proofplugin",
                           new XrdProofdDirective("proofplugin", (void *)&fProofPlugin, &DoDirectiveString));
   // Register config directives for ints
   fReConfigDirectives.Add("localwrks",
                           new XrdProofdDirective("localwrks", (void *)&fNumLocalWrks, &DoDirectiveInt));
}

//__________________________________________________________________________
int XrdProofdManager::Config(const char *fn, bool rcf, XrdSysError *e)
{
   // Config or re-config this instance using the information in file 'fn'
   // return 0 on success, -1 on error

   XrdOucString mp;

   // Error handler
   fEDest = (e) ? e : fEDest;

   if (fCfgFile.fName.length() <= 0 && (!fn || strlen(fn) <= 0)) {
      // Done
      if (fEDest)
         fEDest->Say(0, "ProofdManager: Config: no config file!");
      return -1;
   }

   // We need an error handler
   if (!fEDest) {
      TRACE(XERR, "ProofdManager: Config: error handler undefined!");
      return -1;
   }

   // Did the file changed ?
   if (fn) {
      if (fCfgFile.fName.length() <= 0 ||
         (fCfgFile.fName.length() > 0 && fCfgFile.fName != fn)) {
         fCfgFile.fName = fn;
         XrdProofdAux::Expand(fCfgFile.fName);
         fCfgFile.fMtime = 0;
      }
   }

   // Get the modification time
   struct stat st;
   if (stat(fCfgFile.fName.c_str(), &st) != 0)
      return -1;
   TRACE(DBG, "ProofdManager: Config: file: " << fCfgFile.fName);
   TRACE(DBG, "ProofdManager: Config: time of last modification: " << st.st_mtime);

   // File should be loaded only once
   if (st.st_mtime <= fCfgFile.fMtime)
      return 0;

   // Save the modification time
   fCfgFile.fMtime = st.st_mtime;

   // This part must be modified in atomic way
   XrdSysMutexHelper mhp(fMutex);

   // Effective user
   XrdProofUI ui;
   if (XrdProofdAux::GetUserInfo(geteuid(), ui) == 0) {
      fEffectiveUser = ui.fUser;
   } else {
      mp = "ProofdManager: Config: could not resolve effective user (getpwuid, errno: ";
      mp += errno;
      mp += ")";
      fEDest->Say(0, mp.c_str());
      return -1;
   }

   // Local FQDN
   char *host = XrdNetDNS::getHostName();
   fHost = host ? host : "";
   SafeFree(host);

   // Communicate the host name to the config directives, so that the (deprecated)
   // old style 'if' condition can be handled
   fConfigDirectives.Apply(SetHostInDirectives, (void *)fHost.c_str());
   fReConfigDirectives.Apply(SetHostInDirectives, (void *)fHost.c_str());

   // Default pool entry point is this host
   fPoolURL = "root://";
   fPoolURL += fHost;

   // Default tmp directory
   fTMPdir = "/tmp";

   XrdOucStream cfg(fEDest, getenv("XRDINSTANCE"));

   // Open and attach the config file
   int cfgFD;
   const char *cfn = fCfgFile.fName.c_str();
   if ((cfgFD = open(cfn, O_RDONLY, 0)) < 0) {
      fEDest->Say(0, "ProofdManager: Config: unable to open : ", cfn);
      return -1;
   }
   cfg.Attach(cfgFD);

   if (rcf) {
      // Park existing valid ROOT sys entries: this allows us to eliminate
      // those not wanted without re-validating those that we know are valid
      std::list<XrdROOT *>::iterator tri;
      if (fROOT.size() > 0) {
         for (tri = fROOT.begin(); tri != fROOT.end(); tri++) {
            if ((*tri)->IsValid())
               (*tri)->Park();
         }
      }
   }

   // For better notification
   XrdOucHash<XrdProofdDirective> *fst = (rcf) ? &fReConfigDirectives
                                               : &fConfigDirectives;
   XrdOucHash<XrdProofdDirective> *snd = (rcf) ? &fConfigDirectives : 0;
   // Process items
   char *var = 0, *val = 0;
   while ((var = cfg.GetMyFirstWord())) {
      if (!(strncmp("xpd.", var, 4)) && var[4]) {
         // xpd directive: process it
         var += 4;
         // Get the value
         val = cfg.GetToken();
         // Get the directive
         XrdProofdDirective *d = fst->Find(var);
         if (d) {
            // Process it
            d->DoDirective(val, &cfg, rcf);
         } else if (snd && (d = snd->Find(var))) {
            TRACE(XERR, "Config: directive xpd."<<var<<" cannot be re-configured");
         }
      } else if (var[0]) {
         // Check if we are interested in this non-xpd directive
         XrdProofdDirective *d = fst->Find(var);
         if (d) {
            // Process it
            d->DoDirective(0, &cfg, rcf);
         } else if (snd && (d = snd->Find(var))) {
            TRACE(XERR, "Config: directive "<<var<<" cannot be re-configured");
         }
      }
   }

   // Parse the config directives
   return ParseConfig(ui, rcf);
}

//__________________________________________________________________________
int XrdProofdManager::ParseConfig(XrdProofUI ui, bool rcf)
{
   // Parse the entered config directives.
   // Return 0 on success, -1 on error

   XrdOucString msg;
   msg = (rcf) ? "ProofdManager: ParseConfig: reconfiguring"
               : "ProofdManager: ParseConfig: configuring";
   fEDest->Say(0, msg.c_str());

   // Change/DonotChange ownership when logging clients
   fChangeOwn = (fMultiUser && getuid()) ? 0 : 1;

   // Work directory, if specified
   if (fWorkDir.length() > 0) {
      // Make sure it exists
      if (XrdProofdAux::AssertDir(fWorkDir.c_str(), ui, 1) != 0) {
         fEDest->Say(0, "ProofdManager: ParseConfig: unable to assert working dir: ",
                        fWorkDir.c_str());
         return -1;
      }
      fEDest->Say(0, "ProofdManager: ParseConfig: working directories under: ",
                     fWorkDir.c_str());
   }

   if (fSrvType != kXPD_WorkerServer || fSrvType == kXPD_AnyServer) {
      fEDest->Say(0, "ProofdManager: ParseConfig: PROOF config file: ",
                    ((fPROOFcfg.fName.length() > 0) ? fPROOFcfg.fName.c_str()
                                                    : "none"));

      if (fResourceType == kRTStatic) {
         // Initialize the list of workers if a static config has been required
         // Default file path, if none specified
         if (fPROOFcfg.fName.length() <= 0) {
            CreateDefaultPROOFcfg();
         } else {
            // Load file content in memory
            if (ReadPROOFcfg() != 0) {
               fEDest->Say(0, "ProofdManager: ParseConfig: unable to find valid information"
                              "in PROOF config file ", fPROOFcfg.fName.c_str());
               fPROOFcfg.fMtime = 0;
               return 0;
            }
         }
      }
   }

   // Initialize the security system if this is wanted
   if (!rcf) {
      if (fSecLib.length() <= 0)
         fEDest->Say(0, "XRD seclib not specified; strong authentication disabled");
      else {
         if (!(fCIA = LoadSecurity())) {
            fEDest->Emsg(0, "ProofdManager: ParseConfig: unable to load security system.");
            return -1;
         }
         fEDest->Emsg(0, "ProofdManager: ParseConfig: security library loaded");
      }
   }

   // Notify allow rules
   if (fSrvType == kXPD_WorkerServer || fSrvType == kXPD_MasterServer) {
      if (fMastersAllowed.size() > 0) {
         std::list<XrdOucString *>::iterator i;
         for (i = fMastersAllowed.begin(); i != fMastersAllowed.end(); ++i)
            fEDest->Say(0, "ProofdManager : ParseConfig: masters allowed to connect: ", (*i)->c_str());
      } else {
            fEDest->Say(0, "ProofdManager : ParseConfig: masters allowed to connect: any");
      }
   }

   // Notify change priority rules
   if (fPriorities.size() > 0) {
      std::list<XrdProofdPriority *>::iterator i;
      for (i = fPriorities.begin(); i != fPriorities.end(); ++i) {
         msg = "priority will be changed by ";
         msg += (*i)->fDeltaPriority;
         msg += " for user(s): ";
         msg += (*i)->fUser;
         fEDest->Say(0, "ProofdManager : ParseConfig: ", msg.c_str());
      }
   } else {
      fEDest->Say(0, "ProofdManager : ParseConfig: no priority changes requested");
   }

   // Pool and namespace
   fEDest->Say(0, "ProofdManager : ParseConfig: PROOF pool: ", fPoolURL.c_str());
   fEDest->Say(0, "ProofdManager : ParseConfig: PROOF pool namespace: ", fNamespace.c_str());

   // Initialize resource broker (if not worker)
   if (fSrvType != kXPD_WorkerServer) {

      // Scheduler instance
      if (!(fProofSched = LoadScheduler())) {
         fEDest->Say(0, "ProofdManager : ParseConfig: scheduler initialization failed");
         return 0;
      }

      if (!PROOFcfg() || strlen(PROOFcfg()) <= 0)
         // Enable user config files
         fWorkerUsrCfg = 1;
      const char *st[] = { "disabled", "enabled" };
      fEDest->Say(0, "ProofdManager : ParseConfig: user config files are ", st[fWorkerUsrCfg]);
   }

   // Shutdown options
   XrdOucString mp = "ProofdManager : ParseConfig: client sessions shutdown after disconnection";
   if (fShutdownOpt > 0) {
      if (fShutdownOpt == 1)
         mp = "ProofdManager : ParseConfig: client sessions kept idle for ";
      else if (fShutdownOpt == 2)
         mp = "ProofdManager : ParseConfig: client sessions kept for ";
      mp += fShutdownDelay;
      mp += " secs after disconnection";
   }
   fEDest->Say(0, mp.c_str());

   // Superusers: add default
   if (fSuperUsers.length() > 0)
      fSuperUsers += ",";
   fSuperUsers += fEffectiveUser;
   mp = "ProofdManager : ParseConfig: list of superusers: ";
   mp += fSuperUsers;
   fEDest->Say(0, mp.c_str());

   // Notify controlled mode, if such
   if (fOperationMode == kXPD_OpModeControlled) {
      fAllowedUsers += ',';
      fAllowedUsers += fSuperUsers;
      mp = "ProofdManager : ParseConfig: running in controlled access mode: users allowed: ";
      mp += fAllowedUsers;
      fEDest->Say(0, mp.c_str());
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
      fEDest->Say(0, "ProofdManager : ParseConfig: bare lib path for proofserv: ",
                     fBareLibPath.c_str());
   }

   // ROOT dirs
   if (rcf) {
      // Remove parked ROOT sys entries
      std::list<XrdROOT *>::iterator tri;
      if (fROOT.size() > 0) {
         for (tri = fROOT.begin(); tri != fROOT.end();) {
            if ((*tri)->IsParked()) {
               delete (*tri);
               tri = fROOT.erase(tri);
            } else {
               tri++;
            }
         }
      }
   } else {
      // Check the ROOT dirs
      if (fROOT.size() <= 0) {
         // None defined: use ROOTSYS as default, if any; otherwise we fail
         if (getenv("ROOTSYS")) {
            XrdROOT *rootc = new XrdROOT(getenv("ROOTSYS"), "");
            msg = "ProofdManager : ParseConfig: ROOT dist: \"";
            msg += rootc->Export();
            if (rootc->Validate()) {
               msg += "\" validated";
               fROOT.push_back(rootc);
            } else {
               msg += "\" could not be validated";
            }
            fEDest->Say(0, msg.c_str());
        }
         if (fROOT.size() <= 0) {
            fEDest->Say(0, "ProofdManager : ParseConfig: no ROOT dir defined;"
                           " ROOTSYS location missing - unloading");
            return 0;
         }
      }
   }

   // Groups
   if (!fGroupsMgr)
      // Create default group, if none explicitely requested
      fGroupsMgr = new XrdProofGroupMgr;

   if (rcf) {
      // Re-assign groups
      if (fGroupsMgr && fGroupsMgr->Num() > 0) {
         std::list<XrdProofdClient *>::iterator pci;
         for (pci = fProofdClients.begin(); pci != fProofdClients.end(); ++pci) {
            // Find first client
            XrdProofdProtocol *c = 0;
            int ic = 0;
            while (ic < (int) (*pci)->Clients()->size())
               if ((c = (*pci)->Clients()->at(ic++)))
                  break;
            if (c)
               (*pci)->SetGroup(fGroupsMgr->GetUserGroup(c->GetID(), c->GetGroupID()));
         }
      }
   }
   if (fGroupsMgr)
      fGroupsMgr->Print(0);

   // Scheduling option
   if (fGroupsMgr && fGroupsMgr->Num() > 1 && fSchedOpt != kXPD_sched_off) {
      mp = "ProofdManager : ParseConfig: worker sched based on: ";
      mp += (fSchedOpt == kXPD_sched_central) ? "central" : "local";
      mp += " priorities";
      fEDest->Say(0, mp.c_str());
   }

   // Done
   return 0;
}

//__________________________________________________________________________
int XrdProofdManager::Broadcast(int type, const char *msg,
                                XrdProofdResponse *r, bool notify)
{
   // Broadcast request to known potential sub-nodes.
   // Return 0 on success, -1 on error
   int rc = 0;

   TRACE(ACT, "Broadcast: enter: type: "<<type);

   // Loop over unique nodes
   std::list<XrdProofWorker *>::iterator iw = fNodes.begin();
   XrdProofWorker *w = 0;
   XrdClientMessage *xrsp = 0;
   while (iw != fNodes.end()) {
      if ((w = *iw) && w->fType != 'M') {
         // Do not send it to ourselves
         bool us = (((w->fHost.find("localhost") != STR_NPOS ||
                     fHost.find(w->fHost.c_str()) != STR_NPOS)) &&
                    (w->fPort == -1 || w->fPort == fPort)) ? 1 : 0;
         if (!us) {
            // Create 'url'
            XrdOucString u = fEffectiveUser;
            u += '@';
            u += w->fHost;
            if (w->fPort != -1) {
               u += ':';
               u += w->fPort;
            }
            // Type of server
            int srvtype = (w->fType != 'W') ? (kXR_int32) kXPD_MasterServer
                                            : (kXR_int32) kXPD_WorkerServer;
            TRACE(HDBG,"Broadcast: sending request to "<<u);
            // Send request
            if (!(xrsp = Send(u.c_str(), type, msg, srvtype, r, notify))) {
               TRACE(XERR,"Broadcast: problems sending request to "<<u);
            }
            // Cleanup answer
            SafeDelete(xrsp);
         }
      }
      // Next worker
      iw++;
   }

   // Done
   return rc;
}

//__________________________________________________________________________
XrdProofConn *XrdProofdManager::GetProofConn(const char *url)
{
   // Get a XrdProofConn for url; create a new one if not available

   XrdSysMutexHelper mhp(&fMutex);

   XrdProofConn *p = 0;
   if (fProofConnHash.Num() > 0) {
      if ((p = fProofConnHash.Find(url)) && (p->IsValid())) {
         // Valid connection exists
         TRACE(DBG,"GetProofConn: foudn valid connection for "<<url);
         return p;
      }
      // If the connection is invalid connection clean it up
      SafeDelete(p);
      fProofConnHash.Del(url);
   }

   // If not found create a new one
   XrdOucString buf = " Manager connection from ";
   buf += fHost;
   buf += "|ord:000";
   char m = 'A'; // log as admin

   // We try only once
   int maxtry_save = -1;
   int timewait_save = -1;
   XrdProofConn::GetRetryParam(maxtry_save, timewait_save);
   XrdProofConn::SetRetryParam(1, 1);

   // Request Timeout
   EnvPutInt(NAME_REQUESTTIMEOUT, fRequestTO);

   if ((p = new XrdProofConn(url, m, -1, -1, 0, buf.c_str()))) {
      if (p->IsValid())
         // Cache it
         fProofConnHash.Rep(url, p, 0, Hash_keepdata);
      else
         SafeDelete(p);
   }

   // Restore original retry parameters
   XrdProofConn::SetRetryParam(maxtry_save, timewait_save);

   // Done
   return p;
}

//__________________________________________________________________________
XrdClientMessage *XrdProofdManager::Send(const char *url, int type,
                                         const char *msg, int srvtype,
                                         XrdProofdResponse *r, bool notify)
{
   // Broadcast request to known potential sub-nodes.
   // Return 0 on success, -1 on error
   XrdClientMessage *xrsp = 0;

   TRACE(ACT, "Send: enter: type: "<<type);

   if (!url || strlen(url) <= 0)
      return xrsp;

   // Get a connection to the server
   XrdProofConn *conn = GetProofConn(url);

   XrdSysMutexHelper mhp(&fMutex);

   // For requests we try 4 times
   int maxtry_save = -1;
   int timewait_save = -1;
   XrdProofConn::GetRetryParam(maxtry_save, timewait_save);
   XrdProofConn::SetRetryParam(4, timewait_save);

   bool ok = 1;
   if (conn && conn->IsValid()) {
      XrdOucString notifymsg("Send: ");
      // Prepare request
      XPClientRequest reqhdr;
      const void *buf = 0;
      char **vout = 0;
      memset(&reqhdr, 0, sizeof(reqhdr));
      conn->SetSID(reqhdr.header.streamid);
      reqhdr.header.requestid = kXP_admin;
      reqhdr.proof.int1 = type;
      switch (type) {
         case kROOTVersion:
            notifymsg += "change-of-ROOT version request to ";
            notifymsg += url;
            notifymsg += " msg: ";
            notifymsg += msg;
            reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
            buf = (msg) ? (const void *)msg : buf;
            break;
         case kCleanupSessions:
            notifymsg += "cleanup request to ";
            notifymsg += url;
            notifymsg += " for user: ";
            notifymsg += msg;
            reqhdr.proof.int2 = (kXR_int32) srvtype;
            reqhdr.proof.sid = -1;
            reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
            buf = (msg) ? (const void *)msg : buf;
            break;
         default:
            ok = 0;
            TRACE(XERR,"Send: invalid request type "<<type);
            break;
      }

      // Notify the client that we are sending the request
      if (r && notify)
         r->Send(kXR_attn, kXPD_srvmsg, 0, (char *) notifymsg.c_str(), notifymsg.length());

      // Send over
      if (ok)
         xrsp = conn->SendReq(&reqhdr, buf, vout, "XrdProofdManager::Send");

      // Print error msg, if any
      if (r && !xrsp && conn->GetLastErr()) {
         XrdOucString cmsg = url;
         cmsg += ": ";
         cmsg += conn->GetLastErr();
         r->Send(kXR_attn, kXPD_srvmsg, (char *) cmsg.c_str(), cmsg.length());
      }

   } else {
      TRACE(XERR,"Send: could not open connection to "<<url);
      if (r) {
         XrdOucString cmsg = "failure attempting connection to ";
         cmsg += url;
         r->Send(kXR_attn, kXPD_srvmsg, (char *) cmsg.c_str(), cmsg.length());
      }
   }

   // Restore original retry parameters
   XrdProofConn::SetRetryParam(maxtry_save, timewait_save);

   // Done
   return xrsp;
}

//__________________________________________________________________________
void XrdProofdManager::CreateDefaultPROOFcfg()
{
   // Fill-in fWorkers for a localhost based on the number of
   // workers fNumLocalWrks.

   TRACE(ACT, "CreateDefaultPROOFcfg: enter");

   // Create a default master line
   XrdOucString mm("master ",128);
   mm += fHost;
   fWorkers.push_back(new XrdProofWorker(mm.c_str()));
   TRACE(DBG, "CreateDefaultPROOFcfg: added line: " << mm);

   // Create 'localhost' lines for each worker
   int nwrk = fNumLocalWrks;
   if (nwrk > 0) {
      mm = "worker localhost port=";
      mm += fPort;
      while (nwrk--) {
         fWorkers.push_back(new XrdProofWorker(mm.c_str()));
         TRACE(DBG, "CreateDefaultPROOFcfg: added line: " << mm);
      }
      // One line for the nodes
      fNodes.push_back(new XrdProofWorker(mm.c_str()));
   }

   XPDPRT("CreateDefaultPROOFcfg: done: "<<fWorkers.size()-1<<" workers");

   // We are done
   return;
}

//__________________________________________________________________________
XrdProofServProxy *XrdProofdManager::GetActiveSession(int pid)
{
   // Return active session with process ID pid, if any.

   XrdSysMutexHelper mhp(fMutex);

   XrdProofServProxy *srv = 0;

   std::list<XrdProofServProxy *>::iterator svi;
   for (svi = fActiveSessions.begin(); svi != fActiveSessions.end(); svi++) {
      if ((*svi)->IsValid() && ((*svi)->SrvID() == pid)) {
         srv = *svi;
         return srv;
      }
   }
   // Done
   return srv;
}

//__________________________________________________________________________
std::list<XrdProofWorker *> *XrdProofdManager::GetActiveWorkers()
{
   // Return the list of workers after having made sure that the info is
   // up-to-date

   XrdSysMutexHelper mhp(fMutex);

   if (fResourceType == kRTStatic && fPROOFcfg.fName.length() > 0) {
      // Check if there were any changes in the config file
      if (ReadPROOFcfg() != 0) {
         TRACE(XERR, "GetActiveWorkers: unable to read the configuration file");
         return (std::list<XrdProofWorker *> *)0;
      }
   }
   XPDPRT( "GetActiveWorkers: returning list with "<<fWorkers.size()<<" entries");

   return &fWorkers;
}

//__________________________________________________________________________
std::list<XrdProofWorker *> *XrdProofdManager::GetNodes()
{
   // Return the list of unique nodes after having made sure that the info is
   // up-to-date

   XrdSysMutexHelper mhp(fMutex);

   if (fResourceType == kRTStatic && fPROOFcfg.fName.length() > 0) {
      // Check if there were any changes in the config file
      if (ReadPROOFcfg() != 0) {
         TRACE(XERR, "GetNodes: unable to read the configuration file");
         return (std::list<XrdProofWorker *> *)0;
      }
   }
   XPDPRT( "GetNodes: returning list with "<<fNodes.size()<<" entries");

   return &fNodes;
}

//__________________________________________________________________________
int XrdProofdManager::ReadPROOFcfg()
{
   // Read PROOF config file and load the information in fWorkers.
   // NB: 'master' information here is ignored, because it is passed
   //     via the 'xpd.workdir' and 'xpd.image' config directives

   TRACE(ACT, "ReadPROOFcfg: enter: saved time of last modification: " <<
              fPROOFcfg.fMtime);

   // Check inputs
   if (fPROOFcfg.fName.length() <= 0)
      return -1;

   // Get the modification time
   struct stat st;
   if (stat(fPROOFcfg.fName.c_str(), &st) != 0)
      return -1;
   TRACE(DBG, "ReadPROOFcfg: enter: time of last modification: " << st.st_mtime);

   // File should be loaded only once
   if (st.st_mtime <= fPROOFcfg.fMtime)
      return 0;

   // Cleanup the worker list
   std::list<XrdProofWorker *>::iterator w = fWorkers.begin();
   while (w != fWorkers.end()) {
      delete *w;
      w = fWorkers.erase(w);
   }
   // Cleanup the nodes list
   fNodes.clear();

   // Save the modification time
   fPROOFcfg.fMtime = st.st_mtime;

   // Open the defined path.
   FILE *fin = 0;
   if (!(fin = fopen(fPROOFcfg.fName.c_str(), "r")))
      return -1;

   // Create a default master line
   XrdOucString mm("master ",128);
   mm += fHost;
   fWorkers.push_back(new XrdProofWorker(mm.c_str()));

   // Read now the directives
   int nw = 1;
   char lin[2048];
   while (fgets(lin,sizeof(lin),fin)) {
      // Skip empty lines
      int p = 0;
      while (lin[p++] == ' ') { ; } p--;
      if (lin[p] == '\0' || lin[p] == '\n')
         continue;

      // Skip comments
      if (lin[0] == '#')
         continue;

      // Remove trailing '\n';
      if (lin[strlen(lin)-1] == '\n')
         lin[strlen(lin)-1] = '\0';

      TRACE(DBG, "ReadPROOFcfg: found line: " << lin);

      const char *pfx[2] = { "master", "node" };
      if (!strncmp(lin, pfx[0], strlen(pfx[0])) ||
          !strncmp(lin, pfx[1], strlen(pfx[1]))) {
         // Init a master instance
         XrdProofWorker *pw = new XrdProofWorker(lin);
         if (pw->fHost == "localhost" ||
             pw->Matches(fHost.c_str())) {
            // Replace the default line (the first with what found in the file)
            XrdProofWorker *fw = fWorkers.front();
            fw->Reset(lin);
         }
         SafeDelete(pw);
     } else {
         // Build the worker object
         fWorkers.push_back(new XrdProofWorker(lin));
         nw++;
      }
   }

   // Close files
   fclose(fin);

   // Build the list of unique nodes (skip the master line);
   if (fWorkers.size() > 0) {
      w = fWorkers.begin();
      w++;
      for ( ; w != fWorkers.end(); w++) {
         bool add = 1;
         std::list<XrdProofWorker *>::iterator n;
         for (n = fNodes.begin() ; n != fNodes.end(); n++) {
            if ((*n)->Matches(*w)) {
               add = 0;
               break;
            }
         }
         if (add)
            fNodes.push_back(*w);
      }
   }
   TRACE(DBG, "ReadPROOFcfg: found " << fNodes.size() <<" unique nodes");

   // We are done
   return ((nw == 0) ? -1 : 0);
}

//______________________________________________________________________________
int XrdProofdManager::VerifyProcessByID(int pid, const char *pname)
{
   // Check if 'proofserv' (or a process named 'pname') process 'pid' is still
   // in the process table.
   // For {linux, sun, macosx} it uses the system info; for other systems it
   // invokes the command shell 'ps ax' via popen.
   // Return 1 if running, 0 if not running, -1 if the check could not be run.

   int rc = 0;

   TRACE(ACT, "VerifyProcessByID: enter: pid: "<<pid);

   // Check input consistency
   if (pid < 0) {
      TRACE(XERR, "VerifyProcessByID: invalid pid");
      return -1;
   }

   // Name
   const char *pn = (pname && strlen(pname) > 0) ? pname : "proofserv";

#if defined(linux)
   // Look for the relevant /proc dir
   XrdOucString fn("/proc/");
   fn += pid;
   fn += "/stat";
   FILE *ffn = fopen(fn.c_str(), "r");
   if (!ffn) {
      if (errno == ENOENT) {
         TRACE(DBG, "VerifyProcessByID: process does not exists anymore");
         return 0;
      } else {
         XrdOucString emsg("VerifyProcessByID: cannot open ");
         emsg += fn;
         emsg += ": errno: ";
         emsg += errno;
         TRACE(XERR, emsg.c_str());
         return -1;
      }
   }
   // Read status line
   char line[2048] = { 0 };
   if (fgets(line, sizeof(line), ffn)) {
      if (strstr(line, pn))
         // Still there
         rc = 1;
   } else {
      XrdOucString emsg("VerifyProcessByID: cannot read ");
      emsg += fn;
      emsg += ": errno: ";
      emsg += errno;
      TRACE(XERR, emsg.c_str());
      fclose(ffn);
      return -1;
   }
   // Close the file
   fclose(ffn);

#elif defined(__sun)

   // Look for the relevant /proc dir
   XrdOucString fn("/proc/");
   fn += pid;
   fn += "/psinfo";
   int ffd = open(fn.c_str(), O_RDONLY);
   if (ffd <= 0) {
      if (errno == ENOENT) {
         TRACE(DBG, "VerifyProcessByID: process does not exists anymore");
         return 0;
      } else {
         XrdOucString emsg("VerifyProcessByID: cannot open ");
         emsg += fn;
         emsg += ": errno: ";
         emsg += errno;
         TRACE(XERR, emsg.c_str());
         return -1;
      }
   }
   // Get the information
   psinfo_t psi;
   if (read(ffd, &psi, sizeof(psinfo_t)) != sizeof(psinfo_t)) {
      XrdOucString emsg("VerifyProcessByID: cannot read ");
      emsg += fn;
      emsg += ": errno: ";
      emsg += errno;
      TRACE(XERR, emsg.c_str());
      close(ffd);
      return -1;
   }

   // Verify now
   if (strstr(psi.pr_fname, pn))
      // The process is still there
      rc = 1;

   // Close the file
   close(ffd);

#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)

   // Get the proclist
   kinfo_proc *pl = 0;
   int np;
   int ern = 0;
   if ((ern = XrdProofdAux::GetMacProcList(&pl, np)) != 0) {
      XrdOucString emsg("VerifyProcessByID: cannot get the process list: errno: ");
      emsg += ern;
      TRACE(XERR, emsg.c_str());
      return -1;
   }

   // Loop over the list
   while (np--) {
      if (pl[np].kp_proc.p_pid == pid &&
          strstr(pl[np].kp_proc.p_comm, pn)) {
         // Process still exists
         rc = 1;
         break;
      }
   }
   // Cleanup
   free(pl);
#else
   // Use the output of 'ps ax' as a backup solution
   XrdOucString cmd = "ps ax | grep proofserv 2>/dev/null";
   if (pname && strlen(pname))
      cmd.replace("proofserv", pname);
   FILE *fp = popen(cmd.c_str(), "r");
   if (fp != 0) {
      char line[2048] = { 0 };
      while (fgets(line, sizeof(line), fp)) {
         if (pid == XrdProofdAux::GetLong(line)) {
            // Process still running
            rc = 1;
            break;
         }
      }
      pclose(fp);
   } else {
      // Error executing the command
      return -1;
   }
#endif
   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdManager::TrimTerminatedProcesses()
{
   // Check if the terminated processed have really exited the process
   // table; return number of processes still being terminated

   int np = 0;

   // Cleanup the list of terminated or killed processes
   XrdSysMutexHelper mtxh(&fMutex);
   if (fTerminatedProcess.size() > 0) {
      std::list<XrdProofdPInfo *>::iterator i;
      for (i = fTerminatedProcess.begin(); i != fTerminatedProcess.end();) {
         XrdProofdPInfo *xi = (*i);
         if (VerifyProcessByID(xi->pid, xi->pname.c_str()) == 0) {
            TRACE(HDBG,"TrimTerminatedProcesses: freeing: "<<xi<<" ("<<xi->pid<<", "<<xi->pname<<")");
            // Cleanup the integer
            delete *i;
            // Process has terminated: remove it from the list
            i = fTerminatedProcess.erase(i);
         } else {
            // Count
            np++;
            // Goto next
            i++;
         }
      }
   }

   // Done
   return np;
}

//______________________________________________________________________________
int XrdProofdManager::LogTerminatedProc(int pid)
{
   // Add 'pid' to the global list of processes for which termination was
   // requested.
   // returns 0 on success, -1 in case pid <= 0 .

   if (pid > 0) {
      {  XrdSysMutexHelper mtxh(&fMutex);
         fTerminatedProcess.push_back(new XrdProofdPInfo(pid, "proofserv"));
      }
      TRACE(DBG, "LogTerminatedProc: process ID "<<pid<<
                 " signalled and pushed back");
      return 0;
   }
   return -1;
}

//______________________________________________________________________________
bool XrdProofdManager::CheckMaster(const char *m)
{
   // Check if master 'm' is allowed to connect to this host
   bool rc = 1;

   XrdSysMutexHelper mtxh(&fMutex);
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
int XrdProofdManager::CheckUser(const char *usr,
                                 XrdProofUI &ui, XrdOucString &e)
{
   // Check if the user is allowed to use the system
   // Return 0 if OK, -1 if not.

   // No 'root' logins
   if (!usr || strlen(usr) <= 0) {
      e = "CheckUser: 'usr' string is undefined ";
      return -1;
   }

   // No 'root' logins
   if (strlen(usr) == 4 && !strcmp(usr, "root")) {
      e = "CheckUser: 'root' logins not accepted ";
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
      if (XrdProofdAux::GetUserInfo(geteuid(), ui) != 0) {
         e = "CheckUser: problems getting user info for id: ";
         e += (int)geteuid();
         return -1;
      }
   }

   // If we are in controlled mode we have to check if the user in the
   // authorized list; otherwise we fail. Privileged users are always
   // allowed to connect.
   if (fOperationMode == kXPD_OpModeControlled) {
      bool notok = 1;
      XrdOucString us;
      int from = 0;
      while ((from = fAllowedUsers.tokenize(us, from, ',')) != -1) {
         if (us == usr) {
            notok = 0;
            break;
         }
      }
      if (notok) {
         e = "CheckUser: controlled operations:"
             " user not currently authorized to log in: ";
         e += usr;
         return -1;
      }
   }

   // OK
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::ResolveKeywords(XrdOucString &s, XrdProofdClient *pcl)
{
   // Resolve special keywords in 's' for client 'pcl'. Recognized keywords
   //     <workdir>          root for working dirs
   //     <host>             local host name
   //     <user>             user name
   // Return the number of keywords resolved.

   int nk = 0;

   TRACE(HDBG,"ResolveKeywords: enter: "<<s<<" - WorkDir(): "<<WorkDir());

   // Parse <workdir>
   if (s.replace("<workdir>", WorkDir()))
      nk++;

   TRACE(HDBG,"ResolveKeywords: after <workdir>: "<<s);

   // Parse <host>
   if (s.replace("<host>", Host()))
      nk++;

   TRACE(HDBG,"ResolveKeywords: after <host>: "<<s);

   // Parse <user>
   if (pcl)
      if (s.replace("<user>", pcl->ID()))
         nk++;

   TRACE(HDBG,"ResolveKeywords: exit: "<<s);

   // We are done
   return nk;
}

//_____________________________________________________________________________
XrdProofSched *XrdProofdManager::LoadScheduler()
{
   // Load PROOF scheduler

   XrdProofSched *sched = 0;
   XrdOucString name, lib;

   const char *cfn = fCfgFile.fName.c_str();

   // Locate first the relevant directives in the config file
   if (cfn && strlen(cfn) > 0) {
      XrdOucStream cfg(fEDest, getenv("XRDINSTANCE"));
      // Open and attach the config file
      int cfgFD;
      if ((cfgFD = open(cfn, O_RDONLY, 0)) >= 0) {
         cfg.Attach(cfgFD);
         // Process items
         char *val = 0, *var = 0;
         while ((var = cfg.GetMyFirstWord())) {
            if (!(strcmp("xpd.sched", var))) {
               // Get the name
               val = cfg.GetToken();
               if (val && val[0]) {
                  name = val;
                  // Get the lib
                  val = cfg.GetToken();
                  if (val && val[0])
                     lib = val;
                  // We are done
                  break;
               }
            }
         }
      } else {
         XrdOucString m("failure opening config file (errno:");
         m += errno;
         m += "): ";
         TRACE(XERR, "LoadScheduler: "<< m);
      }
   }

   // If undefined or default init a default instance
   if (name == "default" || !(name.length() > 0 && lib.length() > 0)) {
      if ((name.length() <= 0 && lib.length() > 0) ||
          (name.length() > 0 && lib.length() <= 0)) {
         XrdOucString m("LoadScheduler: missing or incomplete info (name:");
         m += name;
         m += ", lib:";
         m += lib;
         m += ")";
         TRACE(DBG, m.c_str());
      }
      TRACE(DBG,"LoadScheduler: instantiating default scheduler");
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
      if (!(sched = (*ep)(cfn, this, fGroupsMgr, fEDest))) {
         TRACE(XERR, "LoadScheduler: unable to create scheduler object from " << lib);
         return (XrdProofSched *)0;
      }
   }
   // Check result
   if (!(sched->IsValid())) {
      TRACE(XERR, "LoadScheduler:"
                  " unable to instantiate the "<<sched->Name()<<" scheduler using "<< cfn);
      delete sched;
      return (XrdProofSched *)0;
   }
   // Notify
   XPDPRT("LoadScheduler: scheduler loaded: type: " << sched->Name());

   // All done
   return sched;
}

//_____________________________________________________________________________
XrdSecService *XrdProofdManager::LoadSecurity()
{
   // Load security framework and plugins, if not already done

   TRACE(ACT, "LoadSecurity: enter");

   const char *cfn = fCfgFile.fName.c_str();
   const char *seclib = fSecLib.c_str();

   // Make sure the input config file is defined
   if (!cfn) {
      if (fEDest) fEDest->Emsg("LoadSecurity","config file not specified");
      return 0;
   }

   // Open the security library
   void *lh = 0;
   if (!(lh = dlopen(seclib, RTLD_NOW))) {
      if (fEDest) fEDest->Emsg("LoadSecurity",dlerror(),"opening shared library",seclib);
      return 0;
   }

   // Get the server object creator
   XrdSecServLoader_t ep = 0;
   if (!(ep = (XrdSecServLoader_t)dlsym(lh, "XrdSecgetService"))) {
      if (fEDest) fEDest->Emsg("LoadSecurity", dlerror(),
                             "finding XrdSecgetService() in", seclib);
      return 0;
   }

   // Extract in a temporary file the directives prefixed "xpd.sec..." (filtering
   // out the prefix), "sec.protocol" and "sec.protparm"
   int nd = 0;
   char *rcfn = FilterSecConfig(nd);
   if (!rcfn) {
      if (nd == 0) {
         // No directives to be processed
         if (fEDest) fEDest->Emsg("LoadSecurity",
                                "no security directives: strong authentication disabled");
         return 0;
      }
      // Failure
      if (fEDest) fEDest->Emsg("LoadSecurity", "creating temporary config file");
      return 0;
   }

   // Get the server object
   XrdSecService *cia = 0;
   if (!(cia = (*ep)((fEDest ? fEDest->logger() : (XrdSysLogger *)0), rcfn))) {
      if (fEDest) fEDest->Emsg("LoadSecurity",
                             "Unable to create security service object via", seclib);
      return 0;
   }
   // Notify
   if (fEDest) fEDest->Emsg("LoadSecurity", "strong authentication enabled");

   // Unlink the temporary file and cleanup its path
   unlink(rcfn);
   delete[] rcfn;

   // All done
   return cia;
}

//__________________________________________________________________________
char *XrdProofdManager::FilterSecConfig(int &nd)
{
   // Grep directives of the form "xpd.sec...", "sec.protparm" and
   // "sec.protocol" from file 'cfn' and save them in a temporary file,
   // stripping off the prefix "xpd." when needed.
   // If any such directory is found, the full path of the temporary file
   // is returned, with the number of directives found in 'nd'.
   // Otherwise 0 is returned and '-errno' specified in nd.
   // The caller has the responsability to unlink the temporary file and
   // to release the memory allocated for the path.

   static const char *pfx[] = { "xpd.sec.", "sec.protparm", "sec.protocol" };
   char *rcfn = 0;

   TRACE(ACT, "FilterSecConfig: enter");

   const char *cfn = fCfgFile.fName.c_str();

   // Make sure that we got an input file path and that we can open the
   // associated path.
   FILE *fin = 0;
   if (!cfn || !(fin = fopen(cfn,"r"))) {
      nd = (errno > 0) ? -errno : -1;
      return rcfn;
   }

   // Read the directives: if an interesting one is found, we create
   // the output temporary file
   int fd = -1;
   char lin[2048];
   while (fgets(lin,sizeof(lin),fin)) {
      if (!strncmp(lin, pfx[0], strlen(pfx[0])) ||
          !strncmp(lin, pfx[1], strlen(pfx[1])) ||
          !strncmp(lin, pfx[2], strlen(pfx[2]))) {
         // Target directive found
         nd++;
         // Create the output file, if not yet done
         if (!rcfn) {
            rcfn = new char[fTMPdir.length() + strlen("/xpdcfn_XXXXXX") + 2];
            sprintf(rcfn, "%s/xpdcfn_XXXXXX", fTMPdir.c_str());
            if ((fd = mkstemp(rcfn)) < 0) {
               delete[] rcfn;
               nd = (errno > 0) ? -errno : -1;
               fclose(fin);
               rcfn = 0;
               return rcfn;
            }
         }
         XrdOucString slin = lin;
         // Strip the prefix "xpd."
         slin.replace("xpd.","");
         // Make keyword substitution
         ResolveKeywords(slin, 0);
         // Write the line to the output file
         XrdProofdAux::Write(fd, slin.c_str(), slin.length());
      }
   }

   // Close files
   fclose(fin);
   close(fd);

   return rcfn;
}

//__________________________________________________________________________
int XrdProofdManager::GetWorkers(XrdOucString &lw, XrdProofServProxy *xps)
{
   // Get a list of workers from the available resource broker
   int rc = 0;

   TRACE(ACT, "GetWorkers: enter");

   // We need the scheduler at this point
   if (!fProofSched) {
      fEDest->Emsg("GetWorkers", "Scheduler undefined");
      return -1;
   }

   // Query the scheduler for the list of workers
   std::list<XrdProofWorker *> wrks;
   fProofSched->GetWorkers(xps, &wrks);
   TRACE(DBG, "GetWorkers: list size: " << wrks.size());

   // The full list
   std::list<XrdProofWorker *>::iterator iw;
   for (iw = wrks.begin(); iw != wrks.end() ; iw++) {
      XrdProofWorker *w = *iw;
      // Add separator if not the first
      if (lw.length() > 0)
         lw += '&';
      // Add export version of the info
      lw += w->Export();
      // Count
      xps->AddWorker(w);
      w->fProofServs.push_back(xps);
      w->fActive++;
   }

   return rc;
}
//__________________________________________________________________________
static int GetGroupsInfo(const char *, XrdProofGroup *g, void *s)
{
   // Fill the global group structure

   XpdGroupGlobal_t *glo = (XpdGroupGlobal_t *)s;

   if (glo) {
      if (g->Active() > 0) {
         // Set the min/max priorities
         if (glo->prmin == -1 || g->Priority() < glo->prmin)
            glo->prmin = g->Priority();
         if (glo->prmax == -1 || g->Priority() > glo->prmax)
            glo->prmax = g->Priority();
         // Set the draft fractions
         if (g->Fraction() > 0) {
            g->SetFracEff((float)(g->Fraction()));
            glo->totfrac += (float)(g->Fraction());
         } else {
            glo->nofrac += 1;
         }
      }
   } else {
      // Not enough info: stop
      return 1;
   }

   // Check next
   return 0;
}

//__________________________________________________________________________
static int SetGroupFracEff(const char *, XrdProofGroup *g, void *s)
{
   // Check if user 'u' is memmebr of group 'grp'

   XpdGroupEff_t *eff = (XpdGroupEff_t *)s;

   if (eff && eff->glo) {
      XpdGroupGlobal_t *glo = eff->glo;
      if (g->Active() > 0) {
         if (eff->opt == 0) {
            float ef = g->Priority() / glo->prmin;
            g->SetFracEff(ef);
         } else if (eff->opt == 1) {
            if (g->Fraction() < 0) {
               float ef = ((100. - glo->totfrac) / glo->nofrac);
               g->SetFracEff(ef);
            }
         } else if (eff->opt == 2) {
            if (g->FracEff() < 0) {
               // Share eff->cut (default 5%) between those with undefined fraction
               float ef = (eff->cut / glo->nofrac);
               g->SetFracEff(ef);
            } else {
               // renormalize
               float ef = g->FracEff() * eff->norm;
               g->SetFracEff(ef);
            }
         }
      }
   } else {
      // Not enough info: stop
      return 1;
   }

   // Check next
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::SetGroupEffectiveFractions()
{
   // Go through the list of active groups (those having at least a non-idle
   // member) and determine the effective resource fraction on the base of
   // the scheduling option and of priorities or nominal fractions.
   // Return 0 in case of success, -1 in case of error, 1 if every group
   // has the same priority so that the system scheduler should do the job.

   if (!fGroupsMgr)
      return 0;

   // Scheduling option
   bool opri = (fSchedOpt != kXPD_sched_off) ? 1 : 0;

   // Loop over groupd
   XpdGroupGlobal_t glo = {-1., -1., 0, 0.};
   fGroupsMgr->Apply(GetGroupsInfo, &glo);

   XpdGroupEff_t eff = {0, &glo, 0.5, 1.};
   if (opri) {
      // Set effective fractions
      fGroupsMgr->ResetIter();
      eff.opt = 0;
      fGroupsMgr->Apply(SetGroupFracEff, &eff);

   } else {
      // In the fraction scheme we need to fill up with the remaining resources
      // if at least one lower bound was found. And of course we need to restore
      // unitarity, if it was broken

      if (glo.totfrac < 100. && glo.nofrac > 0) {
         eff.opt = 1;
         fGroupsMgr->Apply(SetGroupFracEff, &eff);
      } else if (glo.totfrac > 100) {
         // Leave 5% for unnamed or low priority groups
         eff.opt = 2;
         eff.norm = (glo.nofrac > 0) ? (100. - eff.cut)/glo.totfrac : 100./glo.totfrac ;
         fGroupsMgr->Apply(SetGroupFracEff, &eff);
      }
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::SetInflateFactors()
{
   // Recalculate inflate factors taking into account all active users
   // and their priorities. Return 0 on success, -1 otherwise.

   TRACE(SCHED,"---------------- SetInflateFactors ---------------------------");

   if (!fGroupsMgr || fGroupsMgr->Num() <= 1 ||  !IsSchedOn())
      // Nothing to do
      return 0;

   // At least two active session
   int nact = fActiveSessions.size();
   if (nact <= 1) {
      // Reset inflate
      if (nact == 1) {
         XrdProofServProxy *srv = fActiveSessions.front();
         srv->SetInflate(1000,1);
      }
      // Nothing else to do
      return 0;
   }

   TRACE(SCHED,"enter: "<< fGroupsMgr->Num()<<" groups, " <<
                           nact<<" active sessions");

   XrdSysMutexHelper mtxh(&fMutex);

   // Determine which groups are active and their effective fractions
   int rc = 0;
   if ((rc = SetGroupEffectiveFractions()) != 0) {
      // Failure
      TRACE(XERR,"SetInflateFactors: failure from SetGroupEffectiveFractions");
      return -1;
   }

   // Now create a list of active sessions sorted by decreasing effective fraction
   TRACE(SCHED,"--> creating a list of active sessions sorted by decreasing effective fraction ");
   float tf = 0.;
   std::list<XrdProofServProxy *>::iterator asvi, ssvi;
   std::list<XrdProofServProxy *> sorted;
   for (asvi = fActiveSessions.begin();
        asvi != fActiveSessions.end(); asvi++) {
      if ((*asvi)->IsValid() && ((*asvi)->Status() == kXPD_running)) {
         XrdProofdClient *c = (*asvi)->Parent()->C();
         XrdProofGroup *g = c->Group();
         TRACE(SCHED,"SetInflateFactors: group: "<<  g<<", client: "<<(*asvi)->Client());
         if (g && g->Active() > 0) {
            float ef = g->FracEff() / g->Active();
            int nsrv = c->WorkerProofServ() + c->MasterProofServ();
            TRACE(SCHED,"SetInflateFactors: FracEff: "<< g->FracEff()<<", Active: "<<g->Active()<<
                        ", nsrv: "<<nsrv);
            if (nsrv > 0) {
               ef /= nsrv;
               (*asvi)->SetFracEff(ef);
               tf += ef;
               bool pushed = 0;
               for (ssvi = sorted.begin() ; ssvi != sorted.end(); ssvi++) {
                   if (ef >= (*ssvi)->FracEff()) {
                      sorted.insert(ssvi, (*asvi));
                      pushed = 1;
                      break;
                   }
               }
               if (!pushed)
                  sorted.push_back((*asvi));
            } else {
               TRACE(XERR,"SetInflateFactors: "<<(*asvi)->Client()<<" ("<<c->ID()<<
                          "): no srv sessions for active client !!!"
                          " ===> Protocol error");
            }
         } else {
            if (g) {
               TRACE(XERR,"SetInflateFactors: "<<(*asvi)->Client()<<
                          ": inactive group for active session !!!"
                          " ===> Protocol error");
               g->Print();
            } else {
               TRACE(XERR,"SetInflateFactors: "<<(*asvi)->Client()<<
                          ": undefined group for active session !!!"
                          " ===> Protocol error");
            }
         }
      }
   }

   // Notify
   int i = 0;
   if (TRACING(SCHED) && TRACING(HDBG)) {
      for (ssvi = sorted.begin() ; ssvi != sorted.end(); ssvi++)
         XPDPRT("SetInflateFactors: "<< i++ <<" eff: "<< (*ssvi)->FracEff());
   }

   // Number of processors on this machine
   int ncpu = XrdProofdAux::GetNumCPUs();

   TRACE(SCHED,"--> calculating alpha factors (tf: "<<tf<<")");
   // Calculate alphas now
   float T = 0.;
   float *aa = new float[sorted.size()];
   int nn = sorted.size() - 1;
   i = nn;
   ssvi = sorted.end();
   while (ssvi != sorted.begin()) {
      --ssvi;
      // Normalized factor
      float f = (*ssvi)->FracEff() / tf;
      TRACE(SCHED, "    --> entry: "<< i<<" norm frac:"<< f);
      // The lowest priority gives the overall normalization
      if (i == nn) {
         aa[i] = (1. - f * (nn + 1)) / f ;
         T = (nn + 1 + aa[i]);
         TRACE(SCHED, "    --> aa["<<i<<"]: "<<aa[i]<<", T: "<<T);
      } else {
         float fr = f * T - 1.;
         float ar = 0;
         int j = 0;
         for (j = i+1 ; j < nn ; j++) {
            ar += (aa[j+1] - aa[j]) / (j+1);
         }
         aa[i] = aa[i+1] - (i+1) * (fr - ar);
         TRACE(SCHED, "    --> aa["<<i<<"]: "<<aa[i]<<", fr: "<<fr<<", ar: "<<ar);
      }

      // Round Robin scheduling to have time control
      (*ssvi)->SetSchedRoundRobin();

      // Inflate factor (taking into account the number of CPU's)
      int infl = (int)((1. + aa[i] /ncpu) * 1000 * fOverallInflate);
      TRACE(SCHED, "    --> inflate factor for client "<<
            (*ssvi)->Client()<<" is "<<infl<<"( aa["<<i<<"]: "<<aa[i]<<")");
      (*ssvi)->SetInflate(infl, 1);
      // Go to next
      i--;
   }
   TRACE(SCHED,"------------ End of SetInflateFactors ------------------------");

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::SetNiceValues(int opt)
{
   // Recalculate nice values taking into account all active users
   // and their priorities.
   // The type of sessions considered depend on 'opt':
   //    0          all active sessions
   //    1          master sessions
   //    2          worker sessions
   // Return 0 on success, -1 otherwise.

   TRACE(SCHED,"---------------- SetNiceValues ("<<opt<<") ---------------------------");

   if (!fGroupsMgr || fGroupsMgr->Num() <= 1 || !IsSchedOn())
      // Nothing to do
      return 0;

   // At least two active session
   int nact = fActiveSessions.size();
   TRACE(SCHED,"enter: "<< fGroupsMgr->Num()<<" groups, " << nact<<" active sessions");
   if (nact <= 1) {
      // Restore default values
      if (nact == 1) {
         XrdProofServProxy *srv = fActiveSessions.front();
         srv->SetProcessPriority();
      }
      // Nothing else to do
      TRACE(SCHED,"------------ End of SetNiceValues ------------------------");
      return 0;
   }

   XrdSysMutexHelper mtxh(&fMutex);

   // Determine which groups are active and their effective fractions
   int rc = 0;
   if ((rc = SetGroupEffectiveFractions()) != 0) {
      // Failure
      TRACE(XERR,"SetNiceValues: failure from SetGroupEffectiveFractions");
      TRACE(SCHED,"------------ End of SetNiceValues ------------------------");
      return -1;
   }

   // Now create a list of active sessions sorted by decreasing effective fraction
   TRACE(SCHED,"--> creating a list of active sessions sorted by decreasing effective fraction ");
   float tf = 0.;
   std::list<XrdProofServProxy *>::iterator asvi, ssvi;
   std::list<XrdProofServProxy *> sorted;
   for (asvi = fActiveSessions.begin();
        asvi != fActiveSessions.end(); asvi++) {
      bool act = (opt == 0) || (opt == 1 && !((*asvi)->SrvType() == kXPD_WorkerServer))
                            || (opt == 2 &&  ((*asvi)->SrvType() == kXPD_WorkerServer));
            TRACE(SCHED,"UpdatePriorities: server type: "<<(*asvi)->SrvType()<<" act:"<<act);
      if ((*asvi)->IsValid() && ((*asvi)->Status() == kXPD_running) && act) {
         XrdProofdClient *c = (*asvi)->Parent()->C();
         XrdProofGroup *g = c->Group();
         TRACE(SCHED,"SetNiceValues: group: "<<  g<<", client: "<<(*asvi)->Client());
         if (g && g->Active() > 0) {
            float ef = g->FracEff() / g->Active();
            int nsrv = c->WorkerProofServ() + c->MasterProofServ();
            TRACE(SCHED,"SetNiceValues: FracEff: "<< g->FracEff()<<", Active: "<<g->Active()<<
                        ", nsrv: "<<nsrv);
            if (nsrv > 0) {
               ef /= nsrv;
               (*asvi)->SetFracEff(ef);
               tf += ef;
               bool pushed = 0;
               for (ssvi = sorted.begin() ; ssvi != sorted.end(); ssvi++) {
                   if (ef >= (*ssvi)->FracEff()) {
                      sorted.insert(ssvi, (*asvi));
                      pushed = 1;
                      break;
                   }
               }
               if (!pushed)
                  sorted.push_back((*asvi));
            } else {
               TRACE(XERR,"SetNiceValues: "<<(*asvi)->Client()<<" ("<<c->ID()<<
                          "): no srv sessions for active client !!!"
                          " ===> Protocol error");
            }
         } else {
            if (g) {
               TRACE(XERR,"SetNiceValues: "<<(*asvi)->Client()<<
                          ": inactive group for active session !!!"
                          " ===> Protocol error");
               g->Print();
            } else {
               TRACE(XERR,"SetNiceValues: "<<(*asvi)->Client()<<
                          ": undefined group for active session !!!"
                          " ===> Protocol error");
            }
         }
      }
   }

   // Notify
   int i = 0;
   if (TRACING(SCHED) && TRACING(HDBG)) {
      for (ssvi = sorted.begin() ; ssvi != sorted.end(); ssvi++)
         XPDPRT("SetNiceValues: "<< i++ <<" eff: "<< (*ssvi)->FracEff());
   }

   TRACE(SCHED,"SetNiceValues: calculating nice values");

   // The first has the max priority
   ssvi = sorted.begin();
   float xmax = (*ssvi)->FracEff();
   if (xmax <= 0.) {
      TRACE(XERR,"SetNiceValues: negative or null max effective fraction: "<<xmax);
      return -1;
   }
   // This is for Unix
   int nice = 20 - fPriorityMax;
   (*ssvi)->SetProcessPriority(nice);
   // The others have priorities rescaled wrt their effective fractions
   ssvi++;
   while (ssvi != sorted.end()) {
      if ((*ssvi)->IsValid() && ((*ssvi)->Status() == kXPD_running)) {
         int xpri = (int) ((*ssvi)->FracEff() / xmax * (fPriorityMax - fPriorityMin))
                                                     + fPriorityMin;
         nice = 20 - xpri;
         TRACE(SCHED, "    --> nice value for client "<< (*ssvi)->Client()<<" is "<<nice);
         (*ssvi)->SetProcessPriority(nice);
      }
      ssvi++;
   }
   TRACE(SCHED,"------------ End of SetNiceValues ------------------------");

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::UpdatePriorities(bool forceupdate)
{
   // Update the priorities of the active sessions.
   // The priorities are taken from the priority file. The file is read only
   // if it has changed since the last check.
   // The action depends on the type of scheduling: in case of 'local' scheduling
   // the priorities of all sessions are updated; in case of 'central' scheduling
   // the master priority is updated and the values for the workers are broadcast.

   TRACE(SCHED,"---------------- UpdatePriorities ---------------------------");

   if (!fGroupsMgr || fGroupsMgr->Num() <= 1 ||  !IsSchedOn())
      // Nothing to do
      return 0;

   // At least two active session
   int nact = fActiveSessions.size();

   TRACE(SCHED, "UpdatePriorities: enter: "<< fGroupsMgr->Num()<<
                " groups, " << nact<<" active sessions");
   if (nact <= 1) {
      // Restore default values
      if (nact == 1) {
         XrdProofServProxy *srv = fActiveSessions.front();
         srv->SetProcessPriority();
      }
      // Nothing else to do
      TRACE(SCHED,"------------ End of UpdatePriorities ------------------------");
      return 0;
   }

   XrdSysMutexHelper mtxh(&fMutex);

   // Read priorities
   if (fGroupsMgr->ReadPriorities() != 0) {
      TRACE(SCHED, "UpdatePriorities: no new priority information"
                   " (or problems reading the file)")
      if (!forceupdate)
         return 0;
   }

   if (fSchedOpt == kXPD_sched_central) {
      // Communicate them to the sessions
      std::list<XrdProofServProxy *>::iterator asvi, ssvi;
      for (asvi = fActiveSessions.begin();
           asvi != fActiveSessions.end(); asvi++) {
            TRACE(SCHED,"UpdatePriorities: server type: "<<(*asvi)->SrvType());
         if ((*asvi)->IsValid() && ((*asvi)->Status() == kXPD_running) &&
             !((*asvi)->SrvType() == kXPD_WorkerServer)) {
            XrdProofdClient *c = (*asvi)->Parent()->C();
            XrdProofGroup *g = c->Group();
            TRACE(SCHED,"UpdatePriorities: group: "<<  g<<", client: "<<(*asvi)->Client());
            if (g && g->Active() > 0) {
               TRACE(SCHED,"UpdatePriorities: Priority: "<< g->Priority()<<" Active: "<<g->Active());
               int prio = (int) (g->Priority() * 100);
               (*asvi)->BroadcastPriority(prio);
            }
         }
      }
      // Reset the nice values of the master sessions
      if (SetNiceValues() != 0) {
         TRACE(XERR,"UpdatePriorities: problems setting the new nice values ");
         return -1;
      }
   } else {
      // Just set the new nice values to all sessions
      if (SetNiceValues() != 0) {
         TRACE(XERR,"UpdatePriorities: problems setting the new nice values ");
         return -1;
      }
   }
   TRACE(SCHED,"------------ End of UpdatePriorities ------------------------");

   // Done
   return 0;
}

//
// Special directive processors

//______________________________________________________________________________
int XrdProofdManager::ProcessDirective(XrdProofdDirective *d,
                                       char *val, XrdOucStream *cfg, bool rcf)
{
   // Update the priorities of the active sessions.

   if (!d)
      // undefined inputs
      return -1;

   if (d->fName == "shutdown") {
      return DoDirectiveShutdown(val, cfg, rcf);
   } else if (d->fName == "resource") {
      return DoDirectiveResource(val, cfg, rcf);
   } else if (d->fName == "groupfile") {
      return DoDirectiveGroupfile(val, cfg, rcf);
   } else if (d->fName == "priority") {
      return DoDirectivePriority(val, cfg, rcf);
   } else if (d->fName == "rootsys") {
      return DoDirectiveRootSys(val, cfg, rcf);
   } else if (d->fName == "maxoldlogs") {
      return DoDirectiveMaxOldLogs(val, cfg, rcf);
   } else if (d->fName == "allow") {
      return DoDirectiveAllow(val, cfg, rcf);
   } else if (d->fName == "allowedusers") {
      return DoDirectiveAllowedUsers(val, cfg, rcf);
   } else if (d->fName == "schedopt") {
      return DoDirectiveSchedOpt(val, cfg, rcf);
   } else if (d->fName == "role") {
      return DoDirectiveRole(val, cfg, rcf);
   } else if (d->fName == "multiuser") {
      return DoDirectiveMultiUser(val, cfg, rcf);
   } else if (d->fName == "adminreqto") {
      return DoDirectiveAdminReqTO(val, cfg, rcf);
   } else if (d->fName == "cron") {
      return DoDirectiveCron(val, cfg, rcf);
   } else if (d->fName == "xrd.protocol") {
      return DoDirectivePort(val, cfg, rcf);
   } else if (d->fName == "xrootd.seclib") {
      return DoDirectiveSecLib(val, cfg, rcf);
   }
   TRACE(XERR,"ProcessDirective: unknown directive: "<<d->fName);
   return -1;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveShutdown(char *val, XrdOucStream *cfg, bool)
{
   // Process 'shutdown' directive

   if (!val || !cfg)
      // undefined inputs
      return -1;

   int opt = -1;
   int delay = -1;

   // Shutdown option
   int dp = strtol(val,0,10);
   if (dp >= 0 && dp <= 2)
      opt = dp;
   // Shutdown delay
   if ((val = cfg->GetToken())) {
      int l = strlen(val);
      int f = 1;
      XrdOucString tval = val;
      // Parse
      if (val[l-1] == 's') {
         val[l-1] = 0;
      } else if (val[l-1] == 'm') {
         f = 60;
         val[l-1] = 0;
      } else if (val[l-1] == 'h') {
         f = 3600;
         val[l-1] = 0;
      } else if (val[l-1] < 48 || val[l-1] > 57) {
         f = -1;
      }
      if (f > 0) {
         int de = strtol(val,0,10);
         if (de > 0)
            delay = de * f;
      }
   }

   // Check deprecated 'if' directive
   if (Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, Host()) == 0)
         return 0;

   // Set the values
   fShutdownOpt = (opt > -1) ? opt : fShutdownOpt;
   fShutdownDelay = (delay > -1) ? delay : fShutdownDelay;

   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveResource(char *val, XrdOucStream *cfg, bool)
{
   // Process 'resource' directive

   if (!val || !cfg)
      // undefined inputs
      return -1;

   if (!strcmp("static",val)) {
      // We just take the path of the config file here; the
      // rest is used by the static scheduler
      fResourceType = kRTStatic;
      while ((val = cfg->GetToken()) && val[0]) {
         XrdOucString s(val);
         if (s.beginswith("ucfg:")) {
            fWorkerUsrCfg = s.endswith("yes") ? 1 : 0;
         } else if (s.beginswith("wmx:")) {
         } else if (s.beginswith("selopt:")) {
         } else {
            // Config file
            fPROOFcfg.fName = val;
            if (fPROOFcfg.fName.beginswith("sm:")) {
               fPROOFcfg.fName.replace("sm:","");
               fSuperMst = 1;
            }
            XrdProofdAux::Expand(fPROOFcfg.fName);
            // Make sure it exists and can be read
            if (access(fPROOFcfg.fName.c_str(), R_OK)) {
               TRACE(XERR,"DoDirectiveResource: configuration file cannot be read: "<<
                          fPROOFcfg.fName.c_str());
               fPROOFcfg.fName = "";
               fPROOFcfg.fMtime = 0;
               fSuperMst = 0;
            }
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveGroupfile(char *val, XrdOucStream *cfg, bool rcf)
{
   // Process 'groupfile' directive

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
      TRACE(XERR,"DoDirectiveGroupfile: groups manager already initialized: ignoring ");
      return -1;
   }
   fGroupsMgr = new XrdProofGroupMgr;
   fGroupsMgr->Config(val);
   return 0;
}


//______________________________________________________________________________
int XrdProofdManager::DoDirectivePriority(char *val, XrdOucStream *cfg, bool)
{
   // Process 'priority' directive

   if (!val || !cfg)
      // undefined inputs
      return -1;

   // Priority change directive: get delta_priority
   int dp = strtol(val,0,10);
   XrdProofdPriority *p = new XrdProofdPriority("*", dp);
   // Check if an 'if' condition is present
   if ((val = cfg->GetToken()) && !strncmp(val,"if",2)) {
      if ((val = cfg->GetToken()) && val[0]) {
         p->fUser = val;
      }
   }
   // Add to the list
   fPriorities.push_back(p);
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveRootSys(char *val, XrdOucStream *cfg, bool)
{
   // Process 'rootsys' directive

   if (!val || !cfg)
      // undefined inputs
      return -1;

   // Two tokens may be meaningful
   XrdOucString dir = val;
   val = cfg->GetToken();
   XrdOucString tag = val;
   bool ok = 1;
   if (tag == "if") {
      tag = "";
      // Conditional
      cfg->RetToken();
      ok = (XrdProofdAux::CheckIf(cfg, Host()) > 0) ? 1 : 0;
   }
   if (ok) {
      XrdROOT *rootc = new XrdROOT(dir.c_str(), tag.c_str());
      // Check if already validated
      std::list<XrdROOT *>::iterator ori;
      for (ori = fROOT.begin(); ori != fROOT.end(); ori++) {
         if ((*ori)->Match(rootc->Dir(), rootc->Tag())) {
            if ((*ori)->IsParked()) {
               (*ori)->SetValid();
               SafeDelete(rootc);
               break;
            }
         }
      }
      // If not, try validation
      if (rootc) {
         if (rootc->Validate()) {
            XPDPRT("DoDirectiveRootSys: validation OK for: "<<rootc->Export());
            // Add to the list
            fROOT.push_back(rootc);
         } else {
            XPDPRT("DoDirectiveRootSys: could not validate "<<rootc->Export());
            SafeDelete(rootc);
         }
      }
   }
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
   XrdProofdClient::SetMaxOldLogs(maxoldlogs);
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

   // Users allowed to use the cluster
   fAllowedUsers = val;
   fOperationMode = kXPD_OpModeControlled;
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveSchedOpt(char *val, XrdOucStream *cfg, bool)
{
   // Process 'schedopt' directive

   if (!val || !cfg)
      // undefined inputs
      return -1;

   float of = -1.;
   int pmin = -1;
   int pmax = -1;
   int opt = -1;
   // Defines scheduling options
   while (val && val[0]) {
      XrdOucString o = val;
      if (o.beginswith("overall:")) {
         // The overall inflating factor
         o.replace("overall:","");
         sscanf(o.c_str(), "%f", &of);
      } else if (o.beginswith("min:")) {
         // The overall inflating factor
         o.replace("min:","");
         sscanf(o.c_str(), "%d", &pmin);
      } else if (o.beginswith("max:")) {
         // The overall inflating factor
         o.replace("max:","");
         sscanf(o.c_str(), "%d", &pmax);
      } else {
         if (o == "central")
            opt = kXPD_sched_central;
         else if (o == "local")
            opt = kXPD_sched_local;
      }
      // Check deprecated 'if' directive
      if (Host() && cfg)
         if (XrdProofdAux::CheckIf(cfg, Host()) == 0)
            return 0;
      // Next
      val = cfg->GetToken();
   }

   // Set the values (we need to do it here to avoid setting wrong values
   // when a non-matching 'if' condition is found)
   if (of > -1.)
      fOverallInflate = (of >= 1.) ? of : fOverallInflate;
   if (pmin > -1)
      fPriorityMin = (pmin >= 1 && pmin <= 40) ? pmin : fPriorityMin;
   if (pmax > -1)
      fPriorityMax = (pmax >= 1 && pmax <= 40) ? pmax : fPriorityMax;
   if (opt > -1)
      fSchedOpt = opt;

   // Make sure that min is <= max
   if (fPriorityMin > fPriorityMax) {
      TRACE(XERR, "DoDirectiveSchedOpt: inconsistent value for fPriorityMin (> fPriorityMax) ["<<
                  fPriorityMin << ", "<<fPriorityMax<<"] - correcting");
      fPriorityMin = fPriorityMax;
   }

   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveRole(char *val, XrdOucStream *cfg, bool)
{
   // Process 'allowedusers' directive

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
      fSrvType = kXPD_MasterServer;
   } else if (tval == "worker") {
      fSrvType = kXPD_WorkerServer;
   }

   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectivePort(char *, XrdOucStream *cfg, bool)
{
   // Process 'xrd.protocol' directive to find the port

   if (!cfg)
      // undefined inputs
      return -1;

   // Get the value
   XrdOucString proto = cfg->GetToken();
   if (proto.length() > 0 && proto.beginswith("xproofd:")) {
      proto.replace("xproofd:","");
      fPort = strtol(proto.c_str(), 0, 10);
      fPort = (fPort < 0) ? XPD_DEF_PORT : fPort;
   }
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveSecLib(char *, XrdOucStream *cfg, bool)
{
   // Process 'xrootd.seclib' directive: take the path only if fSecLib
   // is not yet defined: in this way xpd.seclib will always be honoured first

   if (!cfg)
      // undefined inputs
      return -1;

   // Get the value
   XrdOucString lib = cfg->GetToken();
   if (lib.length() > 0 && fSecLib.length() <= 0)
      fSecLib = lib;
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
   int mu = strtol(val,0,10);
   fMultiUser = (mu == 1) ? 1 : fMultiUser;
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveAdminReqTO(char *val, XrdOucStream *cfg, bool)
{
   // Process 'adminreqto' directive

   if (!val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, Host()) == 0)
         return 0;

   // Timeout on requested broadcasted to workers; there are 4 attempts,
   // so the real timeout is 4 x fRequestTO
   int to = strtol(val, 0, 10);
   fRequestTO = (to > 0) ? to : fRequestTO;
   return 0;
}

//______________________________________________________________________________
int XrdProofdManager::DoDirectiveCron(char *val, XrdOucStream *, bool)
{
   // Process 'cron' directive

   if (!val)
      // undefined inputs
      return -1;

   // Cron freqeuncy
   int freq = strtol(val, 0, 10);
   if (freq > 0) {
      XPDPRT("DoDirectiveCron: setting frequency to "<<freq<<" sec");
      fCronFrequency = freq;
   }

   return 0;
}
