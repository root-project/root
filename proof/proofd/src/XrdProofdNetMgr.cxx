// @(#)root/proofd:$Id$
// Author: G. Ganis  Jan 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "XrdProofdPlatform.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdNetMgr                                                      //
//                                                                      //
// Authors: G. Ganis, CERN, 2008                                        //
//                                                                      //
// Manages connections between PROOF server daemons                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdProofdNetMgr.h"

#include "Xrd/XrdBuffer.hh"
#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPlatform.hh"

#include "XrdProofdClient.h"
#include "XrdProofdManager.h"
#include "XrdProofdProtocol.h"
#include "XrdProofdResponse.h"
#include "XrdProofWorker.h"

// Tracing utilities
#include "XrdProofdTrace.h"

#include <algorithm>
#include <limits>
#include <math.h>

//______________________________________________________________________________
int MessageSender(const char *msg, int len, void *arg)
{
   // Send up a message from the server

   XrdProofdResponse *r = (XrdProofdResponse *) arg;
   if (r) {
      return r->Send(kXR_attn, kXPD_srvmsg, 2, (char *) msg, len);
   }
   return -1;
}

//______________________________________________________________________________
XrdProofdNetMgr::XrdProofdNetMgr(XrdProofdManager *mgr,
                                 XrdProtocol_Config *pi, XrdSysError *e)
   : XrdProofdConfig(pi->ConfigFN, e)
{
   // Constructor
   fMgr = mgr;
   fResourceType = kRTNone;
   fPROOFcfg.fName = "";
   fPROOFcfg.fMtime = -1;
   fReloadPROOFcfg = 1;
   fDfltFallback = 0;
   fDfltWorkers.clear();
   fRegWorkers.clear();
   fWorkers.clear();
   fNodes.clear();
   fNumLocalWrks = XrdProofdAux::GetNumCPUs();
   fWorkerUsrCfg = 0;
   fRequestTO = 30;
   fBonjourEnabled = false;
#if defined(BUILD_BONJOUR)
   char *host = XrdNetDNS::getHostName();
   fBonjourName = host ? host : "";
   SafeFree(host);
   fBonjourCores = XrdProofdAux::GetNumCPUs();
   fBonjourRequestedSrvType = kBonjourSrvDisabled;
#endif

   // Configuration directives
   RegisterDirectives();
}

//__________________________________________________________________________
void XrdProofdNetMgr::RegisterDirectives()
{
   // Register config directives

   Register("adminreqto", new XrdProofdDirective("adminreqto", this, &DoDirectiveClass));
   Register("resource", new XrdProofdDirective("resource", this, &DoDirectiveClass));
   Register("worker", new XrdProofdDirective("worker", this, &DoDirectiveClass));
   Register("bonjour", new XrdProofdDirective("bonjour", this, &DoDirectiveClass));
   Register("localwrks", new XrdProofdDirective("localwrks", (void *)&fNumLocalWrks, &DoDirectiveInt));
}

//__________________________________________________________________________
XrdProofdNetMgr::~XrdProofdNetMgr()
{
   // Destructor

   // Cleanup the worker lists
   // (the nodes list points to the same object, no cleanup is needed)
   std::list<XrdProofWorker *>::iterator w = fRegWorkers.begin();
   while (w != fRegWorkers.end()) {
      delete *w;
      w = fRegWorkers.erase(w);
   }
   w = fDfltWorkers.begin();
   while (w != fDfltWorkers.end()) {
      delete *w;
      w = fDfltWorkers.erase(w);
   }
   fWorkers.clear();
}

//__________________________________________________________________________
int XrdProofdNetMgr::Config(bool rcf)
{
   // Run configuration and parse the entered config directives.
   // Return 0 on success, -1 on error
   XPDLOC(NMGR, "NetMgr::Config")

   // Lock the method to protect the lists.
   XrdSysMutexHelper mhp(fMutex);

   // Cleanup the worker list
   std::list<XrdProofWorker *>::iterator w = fWorkers.begin();
   while (w != fWorkers.end()) {
      delete *w;
      w = fWorkers.erase(w);
   }
   // Create a default master line
   XrdOucString mm("master ", 128);
   mm += fMgr->Host();
   mm += " port=";
   mm += fMgr->Port();
   fWorkers.push_back(new XrdProofWorker(mm.c_str()));

   // Run first the configurator
   if (XrdProofdConfig::Config(rcf) != 0) {
      XPDERR("problems parsing file ");
      return -1;
   }

   XrdOucString msg;
   msg = (rcf) ? "re-configuring" : "configuring";
   TRACE(ALL, msg);

   if (fMgr->SrvType() != kXPD_Worker || fMgr->SrvType() == kXPD_AnyServer) {
      TRACE(ALL, "PROOF config file: " <<
            ((fPROOFcfg.fName.length() > 0) ? fPROOFcfg.fName.c_str() : "none"));
      if (fResourceType == kRTStatic) {
         // Initialize the list of workers if a static config has been required
         // Default file path, if none specified
         bool dodefault = 1;
         if (fPROOFcfg.fName.length() > 0) {
            // Load file content in memory
            if (ReadPROOFcfg() == 0) {
               TRACE(ALL, "PROOF config file will " <<
                     ((fReloadPROOFcfg) ? "" : "not ") << "be reloaded upon change");
               dodefault = 0;
            } else {
               if (!fDfltFallback) {
                  XPDERR("unable to find valid information in PROOF config file " <<
                         fPROOFcfg.fName);
                  fPROOFcfg.fMtime = -1;
                  return 0;
               } else {
                  TRACE(ALL, "file " << fPROOFcfg.fName << " cannot be parsed: use default configuration to start with");
               }
            }
         }
         if (dodefault) {
            // Use default
            CreateDefaultPROOFcfg();
         }
      } else if (fResourceType == kRTNone && fWorkers.size() <= 1 && !fBonjourEnabled) {
         // Nothing defined: use default
         CreateDefaultPROOFcfg();
      }

      // Find unique nodes
      FindUniqueNodes();
   }

   // For connection to the other xproofds we try only once
   XrdProofConn::SetRetryParam(1, 1);
   // Request Timeout
   EnvPutInt(NAME_REQUESTTIMEOUT, fRequestTO);

   // Notification
   XPDFORM(msg, "%d worker nodes defined at start-up", fWorkers.size() - 1);
   TRACE(ALL, msg);

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdNetMgr::DoDirective(XrdProofdDirective *d,
                                 char *val, XrdOucStream *cfg, bool rcf)
{
   // Update the priorities of the active sessions.
   XPDLOC(NMGR, "NetMgr::DoDirective")

   if (!d)
      // undefined inputs
      return -1;

   if (d->fName == "resource") {
      return DoDirectiveResource(val, cfg, rcf);
   } else if (d->fName == "adminreqto") {
      return DoDirectiveAdminReqTO(val, cfg, rcf);
   } else if (d->fName == "worker") {
      return DoDirectiveWorker(val, cfg, rcf);
   } else if (d->fName == "bonjour") {
      return DoDirectiveBonjour(val, cfg, rcf);
   }

   TRACE(XERR, "unknown directive: " << d->fName);

   return -1;
}

//______________________________________________________________________________
int XrdProofdNetMgr::DoDirectiveBonjour(char *val, XrdOucStream *cfg, bool)
{
   XPDLOC(NMGR, "NetMgr::DoDirectiveBonjour");

   // Process 'bonjour' directive
   TRACE(DBG, "processing Bonjour directive");

   if (!val || !cfg)
      // undefined inputs
      return -1;

#if defined(BUILD_BONJOUR)
   const char * cp = NULL;

   // The first directive must be the 'bonjour role'.
   if (!strcmp("register", val) || !strcmp("publish", val)) {
      // register and publish are synonyms.
      fBonjourRequestedSrvType = kBonjourSrvRegister;
   } else if (!strcmp("discover", val) || !strcmp("browse", val)) {
      fBonjourRequestedSrvType = kBonjourSrvBrowse;
   } else if (!strcmp("both", val) || !strcmp("all", val)) {
      fBonjourRequestedSrvType = kBonjourSrvBoth;
   } else {
      TRACE(XERR, "Bonjour directive unknown");
      return -1;
   }

   // Continue reading words until the end of the logical line. This a descending
   // recursive parser (LR). Doing this in that way allows users to use a custom
   // order of directives and improves xrootd's if/else/fi compatibility.
   while ((val = cfg->GetWord()) != NULL) {
      // Construct an XrdString for a more confortable analysis.
      XrdOucString s(val);
      // If we have line, parse the directive according to the personal rules of
      // each one. Note that this method allows. It would be more elegant with
      // switch statment, but we need to check only the beginning of the words.
      if (s.beginswith("servicetype=")) {
         cp = index(val, '=');
         cp++;
         fBonjourServiceType.assign(cp, 0);
         TRACE(DBG, "custom service type is " << cp);
      } else if (s.beginswith("name=")) {
         cp = index(val, '=');
         cp++;
         fBonjourName.assign(cp, 0);
         TRACE(DBG, "custom Bonjour name is " << cp);
      } else if (s.beginswith("domain=")) {
         cp = index(val, '=');
         cp++;
         fBonjourDomain.assign(cp, 0);
         TRACE(DBG, "custom Bonjour domain is " << cp);
      } else if (s.beginswith("cores=")) {
         cp = index(val, '=');
         cp++;
         fBonjourCores = strtol(cp, NULL, 10); // atoi() is not thread-safe.
         if (fBonjourCores <= 0) {
            TRACE(XERR, "number of cores not valid: " << fBonjourCores);
            TRACE(XERR, "Bonjour module not loaded!");
            return -1;
         }
         TRACE(DBG, "custom number of cores is " << cp);
      } else {
         TRACE(XERR, "Bonjour directive unknown");
         cfg->RetToken();
         return -1;
      }
   }
   TRACE(DBG, "custom Bonjour name is '" << fBonjourName <<"'");

   // Check the compatibility of the roles and give a warning to the user.
   if (!XrdProofdNetMgr::CheckBonjourRoleCoherence(fMgr->SrvType(), GetBonjourRequestedServiceType())) {
      TRACE(XERR, "Warning: xpd.role directive and xpd.bonjour service selection are not compatible");
   }

   // Register the service on bonjour.
   return LoadBonjourModule(fBonjourRequestedSrvType);

#else

   TRACE(XERR, "Bonjour support is disabled");
   return -1;

#endif
}

//______________________________________________________________________________
void XrdProofdNetMgr::BalanceNodesOrder()
{
   // Indices (this will be used twice).
   list<XrdProofWorker *>::const_iterator iter, iter2;
   list<XrdProofWorker *>::iterator iter3; // Not const, less efficient.
   // Map to store information of the balancer.
   map<XrdProofWorker *, BalancerInfo> info;
   // Node with minimum number of workers distinct to 1.
   unsigned int min = UINT_MAX;
   // Total number of nodes and per iteration assignments.
   unsigned int total = 0, total_perit = 0;
   // Number of iterations to get every node filled.
   unsigned int total_added = 0;
   // Temporary list to store the balanced configuration
   list<XrdProofWorker *> tempNodes;
   // Flag for the search and destroy loop.
   bool deleted;

   // Fill the information store with the first data (number of nodes).
   for (iter = fNodes.begin(); iter != fNodes.end(); iter++) {
      // The next code is not the same as this:
      //info[*iter].available = count(fWorkers.begin(), fWorkers.end(), *iter);
      // The previous piece of STL code only checks the pointer of the value
      // stored on the list, altough it is more efficient, it needs that repeated
      // nodes point to the same object. To allow hybrid configurations, we are
      // doing a 'manually' matching since statically configured nodes are
      // created in multiple ways.
      info[*iter].available = 0;
      for (iter2 = fWorkers.begin(); iter2 != fWorkers.end(); iter2++) {
         if ((*iter)->Matches(*iter2)) {
            info[*iter].available++;
         }
      }
      info[*iter].added = 0;
      // Calculate the minimum greater than 1.
      if (info[*iter].available > 1 && info[*iter].available < min)
         min = info[*iter].available;
      // Calculate the totals.
      total += info[*iter].available;
   }

   // Now, calculate the number of workers to add in each iteration of the
   // round robin, scaling to the smaller number.
   for (iter = fNodes.begin(); iter != fNodes.end(); iter++) {
      if (info[*iter].available > 1) {
         info[*iter].per_iteration = (unsigned int)floor((double)info[*iter].available / (double)min);
      } else {
         info[*iter].per_iteration = 1;
      }
      // Calculate the totals.
      total_perit += info[*iter].per_iteration;
   }

   // Since we are going to substitute the list, don't forget to recover the
   // default node at the fist time.
   tempNodes.push_back(fWorkers.front());

   // Finally, do the round robin assignment of nodes.
   // Stop when every node has its workers processed.
   while (total_added < total) {
      for (map<XrdProofWorker *, BalancerInfo>::iterator i = info.begin(); i != info.end(); i++) {
         if (i->second.added < i->second.available) {
            // Be careful with the remainders (on prime number of nodes).
            unsigned int to_add = xrdmin(i->second.per_iteration,
                                        (i->second.available - i->second.added));
            // Then add the nodes.
            for (unsigned int j = 0; j < to_add; j++) {
               tempNodes.push_back(i->first);
            }
            i->second.added += to_add;
            total_added += to_add;
         }
      }
   }

   // Since we are mergin nodes in only one object, we must merge the current
   // sessions of the static nodes (that can be distinct objects that represent
   // the same node) and delete the orphaned objects. If, in the future, we can
   // assure that every worker has only one object in the list, this is not more
   // necessary. The things needed to change are the DoDirectiveWorker, it must
   // search for a node before inserting it, and in the repeat directive insert
   // the same node always. Also the default configuration methods (there are
   // two in this class) must be updated.
   iter3 = ++(fWorkers.begin());
   while (iter3 != fWorkers.end()) {
      deleted = false;
      // If the worker is not in the fWorkers list, we must process it. Note that
      // std::count() uses a plain comparison between values, in this case, we
      // are comparing pointers (numbers, at the end).
      if (count(++(tempNodes.begin()), tempNodes.end(), *iter3) == 0) {
         // Search for an object that matches with this in the temp list.
         for (iter2 = ++(tempNodes.begin()); iter2 != tempNodes.end(); ++iter2) {
            if ((*iter2)->Matches(*iter3)) {
               // Copy data and delete the *iter object.
               (*iter2)->MergeProofServs(*(*iter3));
               deleted = true;
               delete *iter3;
               fWorkers.erase(iter3++);
               break;
            }
         }
      }
      // Do not forget to increase the iterator.
      if (!deleted)
         iter3++;
   }

   // Then, substitute the current fWorkers list with the balanced one.
   fWorkers = tempNodes;
}

//______________________________________________________________________________
#if defined(BUILD_BONJOUR)
void * XrdProofdNetMgr::ProcessBonjourUpdate(void * context)
{
   XrdProofdNetMgr * mgr;
   std::list<XrdOucBonjourNode *> nodes;
   std::list<XrdOucBonjourNode *>::const_iterator idx;
   std::list<XrdProofWorker *>::iterator w, w2;
   const XrdOucBonjourNode * i;
   XrdProofWorker * worker;
   bool haveit;
   int cores = -1;
   int recordLength;

   XPDLOC(NMGR, "NetMgr::ProcessBonjourUpdate");
   TRACE(DBG, "Updating the network topology");

   mgr = static_cast<XrdProofdNetMgr *>(context);

   // Lock the method to protect the lists.
   XrdSysMutexHelper mhp(mgr->fMutex);

   // If there are no workers registered on the fRegWorkers list, this is the
   // first time we run this updater, so we must create the default node. If not
   // we can mark all the nodes as inactive and then look for the Bonjour updates.
   if (mgr->fRegWorkers.size() < 1) {
      XrdOucString mm("master ", 128);
      mm += mgr->fMgr->Host();
      mm += " port=";
      mm += mgr->fMgr->Port();
      mgr->fRegWorkers.push_back(new XrdProofWorker(mm.c_str()));
   } else {
      // Deactivate all current active workers
      w = mgr->fRegWorkers.begin();
      // Skip the master line
      w++;
      for (; w != mgr->fRegWorkers.end(); w++) {
         (*w)->fActive = false;
      }
   }

   // Update the list with the new nodes.
   // Get the list, and get it locked.
   mgr->fBonjourManager->LockNodeList();
   nodes = mgr->fBonjourManager->GetCurrentNodeList();

   for (idx = nodes.begin(); idx != nodes.end(); idx++) {
      // Get the current node.
      i = *idx;
      (*i).Print();
      // Must be not empty
      if (!i->GetHostName() || (i->GetHostName() && strlen(i->GetHostName()) <= 0)) {
         TRACE(ALL,"bonjour list node: empty!");
         continue;
      }
      TRACE(DBG, "parsing info for node: " << i->GetHostName() << ", port: " << i->GetPort());
      // Filter by service type getting rid of the trailing '.'.
      if (i->GetBonjourRecord().MatchesRegisteredType(mgr->GetBonjourServiceType())) {
         // Check if we have already it
         w = mgr->fRegWorkers.begin();
         w++;
         haveit = 0;
         while (w != mgr->fRegWorkers.end()) {
            TRACE(HDBG,"registerd node: "<< (*w)->fHost <<", port: "<<(*w)->fPort);
            if ((*w)->fHost == i->GetHostName() && (*w)->fPort == i->GetPort()) {
               (*w)->fActive = true;
               haveit = 1;

               // Check if the node is on the fWorkers list.
               if (std::find(mgr->fWorkers.begin(), mgr->fWorkers.end(), *w) == mgr->fWorkers.end()) {
                  // Check for the cores of the node.
                  if (i->GetBonjourRecord().GetTXTValue("cores", recordLength) != NULL) {
                     XrdOucString trimmed(i->GetBonjourRecord().GetTXTValue("cores", recordLength), recordLength);
                     cores = strtol(trimmed.c_str(), NULL, 10);
                  } else {
                     cores = 1;
                  }
                  // The node is returning from being not available, maybe with
                  // a different number of cores, so re-check the number of it.
                  TRACE(HDBG, " adding "<<cores<<" for worker '"<<(*w)->fHost<<"'");
                  for (int c = 0; c < cores; c++) {
                     // If we don't have the node on the fRegWorkers list, it will not
                     // be also on the de fWorkers list.
                     mgr->fWorkers.push_back(*w);
                  }
               } else {
                  TRACE(DBG, " worker(s) '"<<(*w)->fHost<<"' already in the list");
               }

               break;
            }
            // Go to next
            w++;
         }
         // If we do not have it, build a new worker object
         if (!haveit) {
            // Create the new node.
            worker = new XrdProofWorker();
            worker->fHost = i->GetHostName();
            worker->fPort = i->GetPort();
            worker->fActive = true;
            if (i->GetBonjourRecord().GetTXTValue("nodetype", recordLength) != NULL) {
               worker->fType = i->GetBonjourRecord().GetTXTValue("nodetype", recordLength)[0];
            }
            // Check for the cores of the node.
            const char *pbr = i->GetBonjourRecord().GetTXTValue("cores", recordLength);
            if (recordLength > 0) {
               char *pc = new char[recordLength + 1];
               memcpy(pc, pbr, recordLength);
               pc[recordLength] = 0;
               cores = strtol(pbr, NULL, 10);
               delete [] pc;
            } else {
               TRACE(ALL, "no information about the cores available ... skip");
               continue;
            }
            // Add the node to the list the times needed.
            mgr->fRegWorkers.push_back(worker);
            for (int c = 0; c < cores; c++) {
               // If we don't have the node on the fRegWorkers list, it will not
               // be also on the de fWorkers list.
               mgr->fWorkers.push_back(worker);
            }
         }
      }
   }

   // Remove the lock on the Bonjour list.
   mgr->fBonjourManager->UnLockNodeList();

   // Remove nodes not active from fWorkers list.
   w = mgr->fRegWorkers.begin();
   w++;
   while (w != mgr->fRegWorkers.end()) {
      if (!((*w)->fActive)) {
         mgr->fWorkers.remove(*w);
      }
      w++;
   }

   // Process list.
   mgr->FindUniqueNodes();

   // Balance order.
   mgr->BalanceNodesOrder();

   return NULL;
}
#endif

//______________________________________________________________________________
#if defined(BUILD_BONJOUR)
int XrdProofdNetMgr::LoadBonjourModule(int srvtype)
{
   XPDLOC(NMGR, "NetMgr::LoadBonjourModule");

   // Get the reference to the bonjour manager. Store it to optimze the
   // getInstance() usage.
   fBonjourManager = &(XrdOucBonjourFactory::FactoryByPlatform()->GetBonjourManager());

   // Register the service if needed.
   if (srvtype == kBonjourSrvRegister || srvtype == kBonjourSrvBoth) {
      // Default service name or a user customized
      XrdOucBonjourRecord record(GetBonjourName(), GetBonjourServiceType(), GetBonjourDomain());

      // Put the extra information on the record.
      if (XrdProofdProtocol::Mgr())
         switch (XrdProofdProtocol::Mgr()->SrvType()) {
            case kXPD_TopMaster:
            case kXPD_Master:
               record.AddTXTRecord("nodetype", "S");
               break;
            case kXPD_AnyServer: // Altough we can be master, publish as worker.
            case kXPD_Worker:
               record.AddTXTRecord("nodetype", "W");
               break;
            default:
               TRACE(XERR, "TXT node type is not known '" << XrdProofdProtocol::Mgr()->SrvType() << "'");
         }

      // Put the number of workers desired
      record.AddTXTRecord("cores", fBonjourCores);

      // Register the service.
      if (fBonjourManager->RegisterService(record, fMgr->Port()) == 0) {
         TRACE(ALL, "Bonjour service was published OK");
      } else {
         TRACE(XERR, "Bonjour service could not be published");
         return -1;
      }
   }

   // Subscribe to the discoverage thread.
   if (srvtype == kBonjourSrvBrowse || srvtype == kBonjourSrvBoth) {
      fBonjourEnabled = true;
      fBonjourManager->SubscribeForUpdates(GetBonjourServiceType(), ProcessBonjourUpdate, this);
   }

   return 0;
}
#endif

//______________________________________________________________________________
#if defined(BUILD_BONJOUR)
bool XrdProofdNetMgr::CheckBonjourRoleCoherence(int xrdRole, int bonjourSrvType)
{
   // Bonjour services:       Discover-Publish-Both   ALLOWED COMBINATIONS
   const bool allowed[4][3] = {{true,  true,  true }, // -1: AnyServer
                               {false, true,  false}, //  0: Worker
                               {true,  true,  true }, //  1: Submaster
                               {true,  false, false}};//  2: Master & Supermaster

   if (xrdRole < -1 || xrdRole > 2 || bonjourSrvType < -1 || bonjourSrvType > 2)
      return false;

   if (bonjourSrvType == kBonjourSrvDisabled)
      return true; // Avoids warnings when Bonjour is not enabled.

   // Add 1 to role since the constants defined in XProofProtocol.h are between
   // -1 and 2.
   return allowed[xrdRole + 1][bonjourSrvType];
}
#endif

//______________________________________________________________________________
int XrdProofdNetMgr::DoDirectiveAdminReqTO(char *val, XrdOucStream *cfg, bool)
{
   // Process 'adminreqto' directive

   if (!val)
      // undefined inputs
      return -1;

   // Check deprecated 'if' directive
   if (fMgr->Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, fMgr->Host()) == 0)
         return 0;

   // Timeout on requested broadcasted to workers; there are 4 attempts,
   // so the real timeout is 4 x fRequestTO
   int to = strtol(val, 0, 10);
   fRequestTO = (to > 0) ? to : fRequestTO;
   return 0;
}

//______________________________________________________________________________
int XrdProofdNetMgr::DoDirectiveResource(char *val, XrdOucStream *cfg, bool)
{
   // Process 'resource' directive
   XPDLOC(NMGR, "NetMgr::DoDirectiveResource")

   if (!val || !cfg)
      // undefined inputs
      return -1;

   if (!strcmp("static", val)) {
      // We just take the path of the config file here; the
      // rest is used by the static scheduler
      fResourceType = kRTStatic;
      while ((val = cfg->GetWord()) && val[0]) {
         XrdOucString s(val);
         if (s.beginswith("ucfg:")) {
            fWorkerUsrCfg = s.endswith("yes") ? 1 : 0;
         } else if (s.beginswith("reload:")) {
            fReloadPROOFcfg = (s.endswith("1") || s.endswith("yes")) ? 1 : 0;
         } else if (s.beginswith("dfltfallback:")) {
            fDfltFallback = (s.endswith("1") || s.endswith("yes")) ? 1 : 0;
         } else if (s.beginswith("wmx:")) {
         } else if (s.beginswith("selopt:")) {
         } else {
            // Config file
            fPROOFcfg.fName = val;
            if (fPROOFcfg.fName.beginswith("sm:")) {
               fPROOFcfg.fName.replace("sm:", "");
            }
            XrdProofdAux::Expand(fPROOFcfg.fName);
            // Make sure it exists and can be read
            if (access(fPROOFcfg.fName.c_str(), R_OK)) {
               if (errno == ENOENT) {
                  TRACE(ALL, "WARNING: configuration file does not exists: " << fPROOFcfg.fName);
               } else {
                  TRACE(XERR, "configuration file cannot be read: " << fPROOFcfg.fName);
                  fPROOFcfg.fName = "";
                  fPROOFcfg.fMtime = -1;
               }
            }
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
int XrdProofdNetMgr::DoDirectiveWorker(char *val, XrdOucStream *cfg, bool)
{
   // Process 'worker' directive
   XPDLOC(NMGR, "NetMgr::DoDirectiveWorker")

   if (!val || !cfg)
      // undefined inputs
      return -1;

   // Lock the method to protect the lists.
   XrdSysMutexHelper mhp(fMutex);

   // Get the full line (w/o heading keyword)
   cfg->RetToken();
   XrdOucString wrd(cfg->GetWord());
   if (wrd.length() > 0) {
      // Build the line
      XrdOucString line;
      char rest[2048] = {0};
      cfg->GetRest((char *)&rest[0], 2048);
      XPDFORM(line, "%s %s", wrd.c_str(), rest);
      // Parse it now
      if (wrd == "master" || wrd == "node") {
         // Init a master instance
         XrdProofWorker *pw = new XrdProofWorker(line.c_str());
         if (pw->fHost.beginswith("localhost") ||
             pw->Matches(fMgr->Host())) {
            // Replace the default line (the first with what found in the file)
            XrdProofWorker *fw = fWorkers.front();
            fw->Reset(line.c_str());
         }
         SafeDelete(pw);
      } else {
         // How many lines like this?
         int nr = 1;
         int ir = line.find("repeat=");
         if (ir != STR_NPOS) {
            XrdOucString r(line, ir + strlen("repeat="));
            r.erase(r.find(' '));
            nr = r.atoi();
            if (nr < 0 || !XPD_LONGOK(nr)) nr = 1;
            TRACE(DBG, "found repeat = " << nr);
         }
         while (nr--) {
            // Build the worker object
            XrdProofdMultiStr mline(line.c_str());
            if (mline.IsValid()) {
               TRACE(DBG, "found multi-line with: " << mline.N() << " tokens");
               for (int i = 0; i < mline.N(); i++) {
                  TRACE(HDBG, "found token: " << mline.Get(i));
                  fWorkers.push_back(new XrdProofWorker(mline.Get(i).c_str()));
               }
            } else {
               TRACE(DBG, "found line: " << line);
               fWorkers.push_back(new XrdProofWorker(line.c_str()));
            }
         }
      }
   }

   // Necessary for the balancer when Bonjour is enabled. Note that this balancer
   // can also be enabled with a static configuration. By this time is disabled
   // due to its experimental status.
   FindUniqueNodes();
   //BalanceNodesOrder();

   return 0;
}

//__________________________________________________________________________
int XrdProofdNetMgr::BroadcastCtrlC(const char *usr)
{
   // Broadcast a ctrlc interrupt
   // Return 0 on success, -1 on error
   XPDLOC(NMGR, "NetMgr::BroadcastCtrlC")

   int rc = 0;

   // Loop over unique nodes
   std::list<XrdProofWorker *>::iterator iw = fNodes.begin();
   XrdProofWorker *w = 0;
   while (iw != fNodes.end()) {
      if ((w = *iw) && w->fType != 'M') {
         // Do not send it to ourselves
         bool us = (((w->fHost.find("localhost") != STR_NPOS ||
                      XrdOucString(fMgr->Host()).find(w->fHost.c_str()) != STR_NPOS)) &&
                    (w->fPort == -1 || w->fPort == fMgr->Port())) ? 1 : 0;
         if (!us) {
            // Create 'url'
            XrdOucString u = (usr) ? usr : fMgr->EffectiveUser();
            u += '@';
            u += w->fHost;
            if (w->fPort != -1) {
               u += ':';
               u += w->fPort;
            }
            // Get a connection to the server
            XrdProofConn *conn = GetProofConn(u.c_str());
            if (conn && conn->IsValid()) {
               // Prepare request
               XPClientRequest reqhdr;
               memset(&reqhdr, 0, sizeof(reqhdr));
               conn->SetSID(reqhdr.header.streamid);
               reqhdr.proof.requestid = kXP_ctrlc;
               reqhdr.proof.sid = 0;
               reqhdr.proof.dlen = 0;
               // We need the right order
               if (XPD::clientMarshall(&reqhdr) != 0) {
                  TRACE(XERR, "problems marshalling request");
                  return -1;
               }
               if (conn->LowWrite(&reqhdr, 0, 0) != kOK) {
                  TRACE(XERR, "problems sending ctrl-c request to server " << u);
               }
               // Clean it up, to avoid leaving open tcp connection possibly going forever
               // into CLOSE_WAIT
               SafeDelete(conn);
            }
         } else {
            TRACE(DBG, "broadcast request for ourselves: ignore");
         }
      }
      // Next worker
      iw++;
   }

   // Done
   return rc;
}

//__________________________________________________________________________
int XrdProofdNetMgr::Broadcast(int type, const char *msg, const char *usr,
                               XrdProofdResponse *r, bool notify, int subtype)
{
   // Broadcast request to known potential sub-nodes.
   // Return 0 on success, -1 on error
   XPDLOC(NMGR, "NetMgr::Broadcast")

   unsigned int nok = 0;
   TRACE(REQ, "type: " << type);

   // Loop over unique nodes
   std::list<XrdProofWorker *>::iterator iw = fNodes.begin();
   XrdProofWorker *w = 0;
   XrdClientMessage *xrsp = 0;
   while (iw != fNodes.end()) {
      if ((w = *iw) && w->fType != 'M') {
         // Do not send it to ourselves
         bool us = (((w->fHost.find("localhost") != STR_NPOS ||
                      XrdOucString(fMgr->Host()).find(w->fHost.c_str()) != STR_NPOS)) &&
                    (w->fPort == -1 || w->fPort == fMgr->Port())) ? 1 : 0;
         if (!us) {
            // Create 'url'
            XrdOucString u = (usr) ? usr : fMgr->EffectiveUser();
            u += '@';
            u += w->fHost;
            if (w->fPort != -1) {
               u += ':';
               u += w->fPort;
            }
            // Type of server
            int srvtype = (w->fType != 'W') ? (kXR_int32) kXPD_Master
                          : (kXR_int32) kXPD_Worker;
            TRACE(HDBG, "sending request to " << u);
            // Send request
            if (!(xrsp = Send(u.c_str(), type, msg, srvtype, r, notify, subtype))) {
               TRACE(XERR, "problems sending request to " << u);
            } else {
               nok++;
            }
            // Cleanup answer
            SafeDelete(xrsp);
         } else {
            TRACE(DBG, "broadcast request for ourselves: ignore");
         }
      }
      // Next worker
      iw++;
   }

   // Done
   return (nok == fNodes.size()) ? 0 : -1;
}

//__________________________________________________________________________
XrdProofConn *XrdProofdNetMgr::GetProofConn(const char *url)
{
   // Get a XrdProofConn for url; create a new one if not available

   XrdProofConn *p = 0;

   // If not found create a new one
   XrdOucString buf = " Manager connection from ";
   buf += fMgr->Host();
   buf += "|ord:000";
   char m = 'A'; // log as admin

   {
      XrdSysMutexHelper mhp(fMutex);
      p = new XrdProofConn(url, m, -1, -1, 0, buf.c_str());
   }
   if (p && !(p->IsValid())) SafeDelete(p);

   // Done
   return p;
}

//__________________________________________________________________________
XrdClientMessage *XrdProofdNetMgr::Send(const char *url, int type,
                                        const char *msg, int srvtype,
                                        XrdProofdResponse *r, bool notify,
                                        int subtype)
{
   // Broadcast request to known potential sub-nodes.
   // Return 0 on success, -1 on error
   XPDLOC(NMGR, "NetMgr::Send")

   XrdClientMessage *xrsp = 0;
   TRACE(REQ, "type: " << type);

   if (!url || strlen(url) <= 0)
      return xrsp;

   // Get a connection to the server
   XrdProofConn *conn = GetProofConn(url);

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
         case kExec:
            notifymsg += "exec ";
            notifymsg += subtype;
            notifymsg += "request for ";
            notifymsg += msg;
            reqhdr.proof.int2 = (kXR_int32) subtype;
            reqhdr.proof.sid = -1;
            reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
            buf = (msg) ? (const void *)msg : buf;
            break;
         default:
            ok = 0;
            TRACE(XERR, "invalid request type " << type);
            break;
      }

      // Notify the client that we are sending the request
      if (r && notify)
         r->Send(kXR_attn, kXPD_srvmsg, 0, (char *) notifymsg.c_str(), notifymsg.length());

      // Activate processing of unsolicited responses
      conn->SetAsync(conn, &MessageSender, (void *)r);

      // Send over
      if (ok)
         xrsp = conn->SendReq(&reqhdr, buf, vout, "NetMgr::Send");

      // Deactivate processing of unsolicited responses
      conn->SetAsync(0, 0, (void *)0);

      // Print error msg, if any
      if (r && !xrsp && conn->GetLastErr()) {
         XrdOucString cmsg = url;
         cmsg += ": ";
         cmsg += conn->GetLastErr();
         r->Send(kXR_attn, kXPD_srvmsg, (char *) cmsg.c_str(), cmsg.length());
      }
      // Clean it up, to avoid leaving open tcp connection possibly going forever
      // into CLOSE_WAIT
      SafeDelete(conn);

   } else {
      TRACE(XERR, "could not open connection to " << url);
      if (r) {
         XrdOucString cmsg = "failure attempting connection to ";
         cmsg += url;
         r->Send(kXR_attn, kXPD_srvmsg, (char *) cmsg.c_str(), cmsg.length());
      }
   }

   // Done
   return xrsp;
}

//______________________________________________________________________________
bool XrdProofdNetMgr::IsLocal(const char *host, bool checkport)
{
   // Check if 'host' is this local host. If checkport is true,
   // matching of the local port with the one implied by host is also checked.
   // Return 1 if 'local', 0 otherwise

   int rc = 0;
   if (host && strlen(host) > 0) {
      XrdClientUrlInfo uu(host);
      if (uu.Port <= 0) uu.Port = 1093;
      // Fully qualified name
      char *fqn = XrdNetDNS::getHostName(uu.Host.c_str());
      if (fqn && (strstr(fqn, "localhost") || !strcmp(fqn, "127.0.0.1") ||
                  !strcmp(fMgr->Host(), fqn))) {
         if (!checkport || (uu.Port == fMgr->Port()))
            rc = 1;
      }
      SafeFree(fqn);
   }
   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdNetMgr::ReadBuffer(XrdProofdProtocol *p)
{
   // Process a readbuf request
   XPDLOC(NMGR, "NetMgr::ReadBuffer")

   int rc = 0;
   XPD_SETRESP(p, "ReadBuffer");

   XrdOucString emsg;

   // Unmarshall the data
   //
   kXR_int64 ofs = ntohll(p->Request()->readbuf.ofs);
   int len = ntohl(p->Request()->readbuf.len);

   // Find out the file name
   char *file = 0;
   char *filen = 0;
   char *pattern = 0;
   int dlen = p->Request()->header.dlen;
   int grep = ntohl(p->Request()->readbuf.int1);
   int blen = dlen;
   bool local = 0;
   if (dlen > 0 && p->Argp()->buff) {
      file = new char[dlen+1];
      memcpy(file, p->Argp()->buff, dlen);
      file[dlen] = 0;
      // Check if local
      XrdClientUrlInfo ui(file);
      if (ui.Host.length() > 0) {
         // Check locality
         local = XrdProofdNetMgr::IsLocal(ui.Host.c_str());
         if (local) {
            memcpy(file, ui.File.c_str(), ui.File.length());
            file[ui.File.length()] = 0;
            blen = ui.File.length();
            TRACEP(p, DBG, "file is LOCAL");
         }
      }
      // If grep, extract the pattern
      if (grep > 0) {
         // 'grep' operation: len is the length of the 'pattern' to be grepped
         pattern = new char[len + 1];
         int j = blen - len;
         int i = 0;
         while (j < blen)
            pattern[i++] = file[j++];
         pattern[i] = 0;
         filen = strdup(file);
         filen[blen - len] = 0;
         TRACEP(p, DBG, "grep operation " << grep << ", pattern:" << pattern);
      }
   } else {
      emsg = "file name not found";
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
      return 0;
   }
   if (grep) {
      TRACEP(p, REQ, "file: " << filen << ", ofs: " << ofs << ", len: " << len <<
             ", pattern: " << pattern);
   } else {
      TRACEP(p, REQ, "file: " << file << ", ofs: " << ofs << ", len: " << len);
   }

   // Get the buffer
   int lout = len;
   char *buf = 0;
   if (local) {
      if (grep > 0) {
         // Grep local file
         lout = blen; // initial length
         buf = ReadBufferLocal(filen, pattern, lout, grep);
      } else {
         // Read portion of local file
         buf = ReadBufferLocal(file, ofs, lout);
      }
   } else {
      // Read portion of remote file
      XrdClientUrlInfo u(file);
      u.User = p->Client()->User() ? p->Client()->User() : fMgr->EffectiveUser();
      buf = ReadBufferRemote(u.GetUrl().c_str(), file, ofs, lout, grep);
   }

   if (!buf) {
      if (lout > 0) {
         if (grep > 0) {
            if (TRACING(DBG)) {
               XPDFORM(emsg, "nothing found by 'grep' in %s, pattern: %s", filen, pattern);
               TRACEP(p, DBG, emsg);
            }
            response->Send();
            return 0;
         } else {
            XPDFORM(emsg, "could not read buffer from %s %s",
                    (local) ? "local file " : "remote file ", file);
            TRACEP(p, XERR, emsg);
            response->Send(kXR_InvalidRequest, emsg.c_str());
            return 0;
         }
      } else {
         // Just got an empty buffer
         if (TRACING(DBG)) {
            emsg = "nothing found in ";
            emsg += (grep > 0) ? filen : file;
            TRACEP(p, DBG, emsg);
         }
      }
   }

   // Send back to user
   response->Send(buf, lout);

   // Cleanup
   SafeFree(buf);
   SafeDelArray(file);
   SafeFree(filen);
   SafeDelArray(pattern);

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdNetMgr::LocateLocalFile(XrdOucString &file)
{
   // Locate the exact file path allowing for wildcards '*' in the file name.
   // In case of success, returns 0 and fills file wity the first matching instance.
   // Return -1 if no matching pat is found.

   XPDLOC(NMGR, "NetMgr::LocateLocalFile")

   // If no wild cards or empty, nothing to do
   if (file.length() <= 0 || file.find('*') == STR_NPOS) return 0;

   // Locate the file name and the dir
   XrdOucString fn, dn;
   int isl = file.rfind('/');
   if (isl != STR_NPOS) {
      fn.assign(file, isl + 1, -1);
      dn.assign(file, 0, isl);
   } else {
      fn = file;
      dn = "./";
   }

   XrdOucString emsg;
   // Scan the dir
   DIR *dirp = opendir(dn.c_str());
   if (!dirp) {
      XPDFORM(emsg, "cannot open '%s' - errno: %d", dn.c_str(), errno);
      TRACE(XERR, emsg.c_str());
      return -1;
   }
   struct dirent *ent = 0;
   XrdOucString sent;
   while ((ent = readdir(dirp))) {
      if (!strncmp(ent->d_name, ".", 1) || !strncmp(ent->d_name, "..", 2))
         continue;
      // Check the match
      sent = ent->d_name;
      if (sent.matches(fn.c_str()) > 0) break;
      sent = "";
   }
   closedir(dirp);

   // If found fill a new output
   if (sent.length() > 0) {
      XPDFORM(file, "%s%s", dn.c_str(), sent.c_str());
      return 0;
   }

   // Not found
   return -1;
}

//______________________________________________________________________________
char *XrdProofdNetMgr::ReadBufferLocal(const char *path, kXR_int64 ofs, int &len)
{
   // Read a buffer of length 'len' at offset 'ofs' of local file 'path'; the
   // returned buffer must be freed by the caller.
   // Wild cards '*' are allowed in the file name of 'path'; the first matching
   // instance is taken.
   // Returns 0 in case of error.
   XPDLOC(NMGR, "NetMgr::ReadBufferLocal")

   XrdOucString emsg;
   TRACE(REQ, "file: " << path << ", ofs: " << ofs << ", len: " << len);

   // Check input
   if (!path || strlen(path) <= 0) {
      TRACE(XERR, "path undefined!");
      return (char *)0;
   }

   // Locate the path resolving wild cards
   XrdOucString spath(path);
   if (LocateLocalFile(spath) != 0) {
      TRACE(XERR, "path cannot be resolved! (" << path << ")");
      return (char *)0;
   }
   const char *file = spath.c_str();

   // Open the file in read mode
   int fd = open(file, O_RDONLY);
   if (fd < 0) {
      emsg = "could not open ";
      emsg += file;
      TRACE(XERR, emsg);
      return (char *)0;
   }

   // Size of the output
   struct stat st;
   if (fstat(fd, &st) != 0) {
      emsg = "could not get size of file with stat: errno: ";
      emsg += (int)errno;
      TRACE(XERR, emsg);
      close(fd);
      return (char *)0;
   }
   off_t ltot = st.st_size;

   // Estimate offsets of the requested range
   // Start from ...
   kXR_int64 start = ofs;
   off_t fst = (start < 0) ? ltot + start : start;
   fst = (fst < 0) ? 0 : ((fst >= ltot) ? ltot - 1 : fst);
   // End at ...
   kXR_int64 end = fst + len;
   off_t lst = (end >= ltot) ? ltot : ((end > fst) ? end  : ltot);
   TRACE(DBG, "file size: " << ltot << ", read from: " << fst << " to " << lst);

   // Number of bytes to be read
   len = lst - fst;

   // Output buffer
   char *buf = (char *)malloc(len + 1);
   if (!buf) {
      emsg = "could not allocate enough memory on the heap: errno: ";
      emsg += (int)errno;
      XPDERR(emsg);
      close(fd);
      return (char *)0;
   }

   // Reposition, if needed
   if (fst >= 0)
      lseek(fd, fst, SEEK_SET);

   int left = len;
   int pos = 0;
   int nr = 0;
   do {
      while ((nr = read(fd, buf + pos, left)) < 0 && errno == EINTR)
         errno = 0;
      if (nr < 0) {
         TRACE(XERR, "error reading from file: errno: " << errno);
         break;
      }

      // Update counters
      pos += nr;
      left -= nr;

   } while (nr > 0 && left > 0);

   // Termination
   buf[len] = 0;
   TRACE(HDBG, "read " << nr << " bytes: " << buf);

   // Close file
   close(fd);

   // Done
   return buf;
}

//______________________________________________________________________________
char *XrdProofdNetMgr::ReadBufferLocal(const char *path,
                                       const char *pat, int &len, int opt)
{
   // Grep lines matching 'pat' form 'path'; the returned buffer (length in 'len')
   // must be freed by the caller.
   // Wild cards '*' are allowed in the file name of 'path'; the first matching
   // instance is taken.
   // Returns 0 in case of error.
   XPDLOC(NMGR, "NetMgr::ReadBufferLocal")

   XrdOucString emsg;
   TRACE(REQ, "file: " << path << ", pat: " << pat << ", len: " << len);

   // Check input
   if (!path || strlen(path) <= 0) {
      TRACE(XERR, "file path undefined!");
      return (char *)0;
   }

   // Locate the path resolving wild cards
   XrdOucString spath(path);
   if (LocateLocalFile(spath) != 0) {
      TRACE(XERR, "path cannot be resolved! (" << path << ")");
      return (char *)0;
   }
   const char *file = spath.c_str();

   // Size of the output
   struct stat st;
   if (stat(file, &st) != 0) {
      emsg = "could not get size of file with stat: errno: ";
      emsg += (int)errno;
      TRACE(XERR, emsg);
      return (char *)0;
   }
   off_t ltot = st.st_size;

   // The grep command
   char *cmd = 0;
   int lcmd = 0;
   if (pat && strlen(pat) > 0) {
      lcmd = strlen(pat) + strlen(file) + 20;
      cmd = new char[lcmd];
      if (opt == 2) {
         sprintf(cmd, "grep -v %s %s", pat, file);
      } else {
         sprintf(cmd, "grep %s %s", pat, file);
      }
   } else {
      lcmd = strlen(file) + 10;
      cmd = new char[lcmd];
      sprintf(cmd, "cat %s", file);
   }
   TRACE(DBG, "cmd: " << cmd);

   // Execute the command in a pipe
   FILE *fp = popen(cmd, "r");
   if (!fp) {
      emsg = "could not run '";
      emsg += cmd;
      emsg += "'";
      TRACE(XERR, emsg);
      delete[] cmd;
      return (char *)0;
   }
   delete[] cmd;

   // Read line by line
   len = 0;
   char *buf = 0;
   char line[2048];
   int bufsiz = 0, left = 0, lines = 0;
   while ((ltot > 0) && fgets(line, sizeof(line), fp)) {
      // Parse the line
      int llen = strlen(line);
      ltot -= llen;
      lines++;
      // (Re-)allocate the buffer
      if (!buf || (llen > left)) {
         int dsiz = 100 * ((int)((len + llen) / lines) + 1);
         dsiz = (dsiz > llen) ? dsiz : llen;
         bufsiz += dsiz;
         buf = (char *)realloc(buf, bufsiz + 1);
         left += dsiz;
      }
      if (!buf) {
         emsg = "could not allocate enough memory on the heap: errno: ";
         emsg += (int)errno;
         TRACE(XERR, emsg);
         pclose(fp);
         return (char *)0;
      }
      // Add line to the buffer
      memcpy(buf + len, line, llen);
      len += llen;
      left -= llen;
      if (TRACING(HDBG))
         fprintf(stderr, "line: %s", line);
   }

   // Check the result and terminate the buffer
   if (buf) {
      if (len > 0) {
         buf[len] = 0;
      } else {
         free(buf);
         buf = 0;
      }
   }

   // Close file
   pclose(fp);

   // Done
   return buf;
}

//______________________________________________________________________________
char *XrdProofdNetMgr::ReadBufferRemote(const char *url, const char *file,
                                        kXR_int64 ofs, int &len, int grep)
{
   // Send a read buffer request of length 'len' at offset 'ofs' for remote file
   // defined by 'url'; the returned buffer must be freed by the caller.
   // Returns 0 in case of error.
   XPDLOC(NMGR, "NetMgr::ReadBufferRemote")

   TRACE(REQ, "url: " << (url ? url : "undef") <<
         ", file: " << (file ? file : "undef") << ", ofs: " << ofs <<
         ", len: " << len << ", grep: " << grep);

   // Check input
   if (!file || strlen(file) <= 0) {
      TRACE(XERR, "file undefined!");
      return (char *)0;
   }
   XrdClientUrlInfo u(url);
   if (!url || strlen(url) <= 0) {
      // Use file as url
      u.TakeUrl(XrdOucString(file));
      if (u.User.length() <= 0) u.User = fMgr->EffectiveUser();
   }

   // Get a connection (logs in)
   XrdProofConn *conn = GetProofConn(u.GetUrl().c_str());

   char *buf = 0;
   if (conn && conn->IsValid()) {
      // Prepare request
      XPClientRequest reqhdr;
      memset(&reqhdr, 0, sizeof(reqhdr));
      conn->SetSID(reqhdr.header.streamid);
      reqhdr.header.requestid = kXP_readbuf;
      reqhdr.readbuf.ofs = ofs;
      reqhdr.readbuf.len = len;
      reqhdr.readbuf.int1 = grep;
      reqhdr.header.dlen = strlen(file);
      const void *btmp = (const void *) file;
      char **vout = &buf;
      // Send over
      XrdClientMessage *xrsp =
         conn->SendReq(&reqhdr, btmp, vout, "NetMgr::ReadBufferRemote");

      // If positive answer
      if (xrsp && buf && (xrsp->DataLen() > 0)) {
         len = xrsp->DataLen();
      } else {
         if (xrsp && !(xrsp->IsError()))
            // The buffer was just empty: do not call it error
            len = 0;
         SafeFree(buf);
      }

      // Clean the message
      SafeDelete(xrsp);
      // Clean it up, to avoid leaving open tcp connection possibly going forever
      // into CLOSE_WAIT
      SafeDelete(conn);
   }

   // Done
   return buf;
}

//______________________________________________________________________________
char *XrdProofdNetMgr::ReadLogPaths(const char *url, const char *msg, int isess)
{
   // Get log paths from next tier; used in multi-master setups
   // Returns 0 in case of error.
   XPDLOC(NMGR, "NetMgr::ReadLogPaths")

   TRACE(REQ, "url: " << (url ? url : "undef") <<
         ", msg: " << (msg ? msg : "undef") << ", isess: " << isess);

   // Check input
   if (!url || strlen(url) <= 0) {
      TRACE(XERR, "url undefined!");
      return (char *)0;
   }

   // Get a connection (logs in)
   XrdProofConn *conn = GetProofConn(url);

   char *buf = 0;
   if (conn && conn->IsValid()) {
      // Prepare request
      XPClientRequest reqhdr;
      memset(&reqhdr, 0, sizeof(reqhdr));
      conn->SetSID(reqhdr.header.streamid);
      reqhdr.header.requestid = kXP_admin;
      reqhdr.proof.int1 = kQueryLogPaths;
      reqhdr.proof.int2 = isess;
      reqhdr.proof.sid = -1;
      reqhdr.header.dlen = strlen(msg);
      const void *btmp = (const void *) msg;
      char **vout = &buf;
      // Send over
      XrdClientMessage *xrsp =
         conn->SendReq(&reqhdr, btmp, vout, "NetMgr::ReadLogPaths");

      // If positive answer
      if (xrsp && buf && (xrsp->DataLen() > 0)) {
         int len = xrsp->DataLen();
         buf = (char *) realloc((void *)buf, len + 1);
         if (buf)
            buf[len] = 0;
      } else {
         SafeFree(buf);
      }

      // Clean the message
      SafeDelete(xrsp);
      // Clean it up, to avoid leaving open tcp connection possibly going forever
      // into CLOSE_WAIT
      SafeDelete(conn);
   }

   // Done
   return buf;
}

//______________________________________________________________________________
char *XrdProofdNetMgr::ReadLogPaths(const char *msg, int isess)
{
   // Get log paths from next tier; used in multi-master setups
   // Returns 0 in case of error.
   XPDLOC(NMGR, "NetMgr::ReadLogPaths")

   TRACE(REQ, "msg: " << (msg ? msg : "undef") << ", isess: " << isess);

   char *buf = 0, *pbuf = buf;
   int len = 0;
   // Loop over unique nodes
   std::list<XrdProofWorker *>::iterator iw = fNodes.begin();
   XrdProofWorker *w = 0;
   while (iw != fNodes.end()) {
      if ((w = *iw)) {
         // Do not send it to ourselves
         bool us = (((w->fHost.find("localhost") != STR_NPOS ||
                      XrdOucString(fMgr->Host()).find(w->fHost.c_str()) != STR_NPOS)) &&
                    (w->fPort == -1 || w->fPort == fMgr->Port())) ? 1 : 0;
         if (!us) {
            // Create 'url'
            XrdOucString u = fMgr->EffectiveUser();
            u += '@';
            u += w->fHost;
            if (w->fPort != -1) {
               u += ':';
               u += w->fPort;
            }
            // Ask the node
            char *bmst = fMgr->NetMgr()->ReadLogPaths(u.c_str(), msg, isess);
            if (bmst) {
               len += strlen(bmst) + 1;
               buf = (char *) realloc((void *)buf, len);
               pbuf = buf + len - strlen(bmst) - 1;
               memcpy(pbuf, bmst, strlen(bmst) + 1);
               buf[len - 1] = 0;
               pbuf = buf + len;
               free(bmst);
            }            
         } else {
            TRACE(DBG, "request for ourselves: ignore");
         }
      }
      // Next worker
      iw++;
   }

   // Done
   return buf;
}

//__________________________________________________________________________
void XrdProofdNetMgr::CreateDefaultPROOFcfg()
{
   // Fill-in fWorkers for a localhost based on the number of
   // workers fNumLocalWrks.
   XPDLOC(NMGR, "NetMgr::CreateDefaultPROOFcfg")

   TRACE(DBG, "enter: local workers: " << fNumLocalWrks);

   // Lock the method to protect the lists.
   XrdSysMutexHelper mhp(fMutex);

   // Cleanup the worker list
   fWorkers.clear();
   // The first time we need to create the default workers
   if (fDfltWorkers.size() < 1) {
      // Create a default master line
      XrdOucString mm("master ", 128);
      mm += fMgr->Host();
      fDfltWorkers.push_back(new XrdProofWorker(mm.c_str()));

      // Create 'localhost' lines for each worker
      int nwrk = fNumLocalWrks;
      if (nwrk > 0) {
         mm = "worker localhost port=";
         mm += fMgr->Port();
         while (nwrk--) {
            fDfltWorkers.push_back(new XrdProofWorker(mm.c_str()));
            TRACE(DBG, "added line: " << mm);
         }
      }
   }

   // Copy the list
   std::list<XrdProofWorker *>::iterator w = fDfltWorkers.begin();
   for (; w != fDfltWorkers.end(); w++) {
      fWorkers.push_back(*w);
   }

   TRACE(DBG, "done: " << fWorkers.size() - 1 << " workers");

   // Find unique nodes
   FindUniqueNodes();

   // We are done
   return;
}

//__________________________________________________________________________
std::list<XrdProofWorker *> *XrdProofdNetMgr::GetActiveWorkers()
{
   // Return the list of workers after having made sure that the info is
   // up-to-date
   XPDLOC(NMGR, "NetMgr::GetActiveWorkers")

   XrdSysMutexHelper mhp(fMutex);

   if (fResourceType == kRTStatic && fPROOFcfg.fName.length() > 0) {
      // Check if there were any changes in the config file
      if (fReloadPROOFcfg && ReadPROOFcfg(1) != 0) {
         if (fDfltFallback) {
            // Use default settings
            CreateDefaultPROOFcfg();
            TRACE(DBG, "parsing of " << fPROOFcfg.fName << " failed: use default settings");
         } else {
            TRACE(XERR, "unable to read the configuration file");
            return (std::list<XrdProofWorker *> *)0;
         }
      }
   }
   TRACE(DBG,  "returning list with " << fWorkers.size() << " entries");

   if (TRACING(HDBG)) Dump();

   return &fWorkers;
}

//__________________________________________________________________________
void XrdProofdNetMgr::Dump()
{
   // Dump status
   const char *xpdloc = "NetMgr::Dump";

   XrdSysMutexHelper mhp(fMutex);

   XPDPRT("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
   XPDPRT("+ Active workers status");
   XPDPRT("+ Size: " << fWorkers.size());
   XPDPRT("+ ");

   std::list<XrdProofWorker *>::iterator iw;
   for (iw = fWorkers.begin(); iw != fWorkers.end(); iw++) {
      XPDPRT("+ wrk: " << (*iw)->fHost << ":" << (*iw)->fPort << " type:" << (*iw)->fType <<
             " active sessions:" << (*iw)->Active());
   }
   XPDPRT("+ ");
   XPDPRT("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
}

//__________________________________________________________________________
std::list<XrdProofWorker *> *XrdProofdNetMgr::GetNodes()
{
   // Return the list of unique nodes after having made sure that the info is
   // up-to-date
   XPDLOC(NMGR, "NetMgr::GetNodes")

   XrdSysMutexHelper mhp(fMutex);

   if (fResourceType == kRTStatic && fPROOFcfg.fName.length() > 0) {
      // Check if there were any changes in the config file
      if (fReloadPROOFcfg && ReadPROOFcfg(1) != 0) {
         if (fDfltFallback) {
            // Use default settings
            CreateDefaultPROOFcfg();
            TRACE(DBG, "parsing of " << fPROOFcfg.fName << " failed: use default settings");
         } else {
            TRACE(XERR, "unable to read the configuration file");
            return (std::list<XrdProofWorker *> *)0;
         }
      }
   }
   TRACE(DBG, "returning list with " << fNodes.size() << " entries");

   return &fNodes;
}

//__________________________________________________________________________
int XrdProofdNetMgr::ReadPROOFcfg(bool reset)
{
   // Read PROOF config file and load the information in fWorkers.
   // NB: 'master' information here is ignored, because it is passed
   //     via the 'xpd.workdir' and 'xpd.image' config directives
   XPDLOC(NMGR, "NetMgr::ReadPROOFcfg")

   TRACE(REQ, "saved time of last modification: " << fPROOFcfg.fMtime);

   // Lock the method to protect the lists.
   XrdSysMutexHelper mhp(fMutex);

   // Check inputs
   if (fPROOFcfg.fName.length() <= 0)
      return -1;

   // Get the modification time
   struct stat st;
   if (stat(fPROOFcfg.fName.c_str(), &st) != 0) {
      // If the file disappeared, reset the modification time so that we are sure
      // to reload it if it comes back
      if (errno == ENOENT) fPROOFcfg.fMtime = -1;
      if (!fDfltFallback) {
         TRACE(XERR, "unable to stat file: " << fPROOFcfg.fName << " - errno: " << errno);
      } else {
         TRACE(ALL, "file " << fPROOFcfg.fName << " cannot be parsed: use default configuration");
      }
      return -1;
   }
   TRACE(DBG, "time of last modification: " << st.st_mtime);

   // File should be loaded only once
   if (st.st_mtime <= fPROOFcfg.fMtime)
      return 0;

   // Save the modification time
   fPROOFcfg.fMtime = st.st_mtime;

   // Open the defined path.
   FILE *fin = 0;
   if (!(fin = fopen(fPROOFcfg.fName.c_str(), "r"))) {
      if (fWorkers.size() > 1) {
         TRACE(XERR, "unable to fopen file: " << fPROOFcfg.fName << " - errno: " << errno);
         TRACE(XERR, "continuing with existing list of workers.");
         return 0;
      } else {
         return -1;
      }
   }

   if (reset) {
      // Cleanup the worker list
      fWorkers.clear();
   }

   // Add default a master line if not yet there
   if (fRegWorkers.size() < 1) {
      XrdOucString mm("master ", 128);
      mm += fMgr->Host();
      fRegWorkers.push_back(new XrdProofWorker(mm.c_str()));
   } else {
      // Deactivate all current active workers
      std::list<XrdProofWorker *>::iterator w = fRegWorkers.begin();
      // Skip the master line
      w++;
      for (; w != fRegWorkers.end(); w++) {
         (*w)->fActive = 0;
      }
   }

   // Read now the directives
   int nw = 0;
   char lin[2048];
   while (fgets(lin, sizeof(lin), fin)) {
      // Skip empty lines
      int p = 0;
      while (lin[p++] == ' ') {
         ;
      }
      p--;
      if (lin[p] == '\0' || lin[p] == '\n')
         continue;

      // Skip comments
      if (lin[0] == '#')
         continue;

      // Remove trailing '\n';
      if (lin[strlen(lin)-1] == '\n')
         lin[strlen(lin)-1] = '\0';

      TRACE(DBG, "found line: " << lin);

      // Parse the line
      XrdProofWorker *pw = new XrdProofWorker(lin);

      const char *pfx[2] = { "master", "node" };
      if (!strncmp(lin, pfx[0], strlen(pfx[0])) ||
          !strncmp(lin, pfx[1], strlen(pfx[1]))) {
         // Init a master instance
         if (pw->fHost.beginswith("localhost") ||
             pw->Matches(fMgr->Host())) {
            // Replace the default line (the first with what found in the file)
            XrdProofWorker *fw = fRegWorkers.front();
            fw->Reset(lin);
         }
         // Ignore it
         SafeDelete(pw);
      } else {
         // Check if we have already it
         std::list<XrdProofWorker *>::iterator w = fRegWorkers.begin();
         // Skip the master line
         w++;
         bool haveit = 0;
         while (w != fRegWorkers.end()) {
            if (!((*w)->fActive)) {
               if ((*w)->fHost == pw->fHost && (*w)->fPort == pw->fPort) {
                  (*w)->fActive = 1;
                  haveit = 1;
                  break;
               }
            }
            // Go to next
            w++;
         }
         // If we do not have it, build a new worker object
         if (!haveit) {
            // Keep it
            fRegWorkers.push_back(pw);
         } else {
            // Drop it
            SafeDelete(pw);
         }
      }
   }

   // Copy the active workers
   std::list<XrdProofWorker *>::iterator w = fRegWorkers.begin();
   while (w != fRegWorkers.end()) {
      if ((*w)->fActive) {
         fWorkers.push_back(*w);
         nw++;
      }
      w++;
   }

   // Close files
   fclose(fin);

   // Find unique nodes
   if (reset)
      FindUniqueNodes();

   // We are done
   return ((nw == 0) ? -1 : 0);
}

//__________________________________________________________________________
int XrdProofdNetMgr::FindUniqueNodes()
{
   // Scan fWorkers for unique nodes (stored in fNodes).
   // Return the number of unque nodes.
   // NB: 'master' information here is ignored, because it is passed
   //     via the 'xpd.workdir' and 'xpd.image' config directives
   XPDLOC(NMGR, "NetMgr::FindUniqueNodes")

   TRACE(REQ, "# workers: " << fWorkers.size());

   // Cleanup the nodes list
   fNodes.clear();

   // Build the list of unique nodes (skip the master line);
   if (fWorkers.size() > 1) {
      std::list<XrdProofWorker *>::iterator w = fWorkers.begin();
      w++;
      for (; w != fWorkers.end(); w++) if ((*w)->fActive) {
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
   TRACE(REQ, "found " << fNodes.size() << " unique nodes");

   // We are done
   return fNodes.size();
}
