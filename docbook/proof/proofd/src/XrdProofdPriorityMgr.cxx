// @(#)root/proofd:$Id$
// Author: G. Ganis Feb 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdPriorityMgr                                                 //
//                                                                      //
// Author: G. Ganis, CERN, 2007                                         //
//                                                                      //
// Class managing session priorities.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "XrdProofdPlatform.h"

#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPriv.hh"

#include "XrdProofdAux.h"
#include "XrdProofdManager.h"
#include "XrdProofdPriorityMgr.h"
#include "XrdProofGroup.h"

// Tracing utilities
#include "XrdProofdTrace.h"

// Aux structures for scan through operations
typedef struct {
   XrdProofGroupMgr *fGroupMgr;
   std::list<XrdProofdSessionEntry *> *fSortedList;
   bool error;
} XpdCreateActiveList_t;

//--------------------------------------------------------------------------
//
// XrdProofdPriorityCron
//
// Function run in separate thread watching changes in session status
// frequency
//
//--------------------------------------------------------------------------
void *XrdProofdPriorityCron(void *p)
{
   // This is an endless loop to periodically check the system
   XPDLOC(PMGR, "PriorityCron")

   XrdProofdPriorityMgr *mgr = (XrdProofdPriorityMgr *)p;
   if (!(mgr)) {
      TRACE(REQ, "undefined manager: cannot start");
      return (void *)0;
   }

   while(1) {
      // We wait for processes to communicate a session status change
      int pollRet = mgr->Pipe()->Poll(-1);
      if (pollRet > 0) {
         int rc = 0;
         XpdMsg msg;
         if ((rc = mgr->Pipe()->Recv(msg)) != 0) {
            XPDERR("problems receiving message; errno: "<<-rc);
            continue;
         }
         // Parse type
         if (msg.Type() == XrdProofdPriorityMgr::kChangeStatus) {
            XrdOucString usr, grp;
            int opt, pid;
            rc = msg.Get(opt);
            rc = (rc == 0) ? msg.Get(usr) : rc;
            rc = (rc == 0) ? msg.Get(grp) : rc;
            rc = (rc == 0) ? msg.Get(pid) : rc;
            if (rc != 0) {
               XPDERR("kChangeStatus: problems parsing message : '"<<msg.Buf()<<"'; errno: "<<-rc);
               continue;
            }
            if (opt < 0) {
               // Remove
               mgr->RemoveSession(pid);
            } else if (opt > 0) {
               // Add
               mgr->AddSession(usr.c_str(), grp.c_str(), pid);
            } else {
               XPDERR("kChangeStatus: invalid opt: "<< opt);
            }
         } else if (msg.Type() == XrdProofdPriorityMgr::kSetGroupPriority) {
            XrdOucString grp;
            int prio = -1;
            rc = msg.Get(grp);
            rc = (rc == 0) ? msg.Get(prio) : rc;
            if (rc != 0) {
               XPDERR("kSetGroupPriority: problems parsing message; errno: "<<-rc);
               continue;
            }
            // Change group priority
            mgr->SetGroupPriority(grp.c_str(), prio);
         } else {
            XPDERR("unknown message type: "<< msg.Type());
         }
         // Communicate new priorities
         if (mgr->SetNiceValues() != 0) {
            XPDERR("problem setting nice values ");
         }
      }
   }

   // Should never come here
   return (void *)0;
}

//______________________________________________________________________________
XrdProofdPriorityMgr::XrdProofdPriorityMgr(XrdProofdManager *mgr,
                                           XrdProtocol_Config *pi, XrdSysError *e)
                    : XrdProofdConfig(pi->ConfigFN, e)
{
   // Constructor
   XPDLOC(PMGR, "XrdProofdPriorityMgr")

   fMgr = mgr;
   fSchedOpt = kXPD_sched_off;
   fPriorityMax = 20;
   fPriorityMin = 1;

   // Init pipe for the poller
   if (!fPipe.IsValid()) {
      TRACE(XERR, "unable to generate pipe for the priority poller");
      return;
   }

   // Configuration directives
   RegisterDirectives();
}

//__________________________________________________________________________
static int DumpPriorityChanges(const char *, XrdProofdPriority *p, void *s)
{
   // Reset the priority on entries
   XPDLOC(PMGR, "DumpPriorityChanges")

   XrdSysError *e = (XrdSysError *)s;

   if (p && e) {
      XrdOucString msg;
      XPDFORM(msg, "priority will be changed by %d for user(s): %s",
                   p->fDeltaPriority, p->fUser.c_str());
      TRACE(ALL, msg);
      // Check next
      return 0;
   }

   // Not enough info: stop
   return 1;
}

//__________________________________________________________________________
int XrdProofdPriorityMgr::Config(bool rcf)
{
   // Run configuration and parse the entered config directives.
   // Return 0 on success, -1 on error
   XPDLOC(PMGR, "PriorityMgr::Config")

   // Run first the configurator
   if (XrdProofdConfig::Config(rcf) != 0) {
      XPDERR("problems parsing file ");
      return -1;
   }

   XrdOucString msg;
   msg = (rcf) ? "re-configuring" : "configuring";
   TRACE(ALL, msg);

   // Notify change priority rules
   if (fPriorities.Num() > 0) {
      fPriorities.Apply(DumpPriorityChanges, (void *)fEDest);
   } else {
      TRACE(ALL, "no priority changes requested");
   }

   // Scheduling option
   if (fMgr->GroupsMgr() && fMgr->GroupsMgr()->Num() > 1 && fSchedOpt != kXPD_sched_off) {
      XPDFORM(msg, "worker sched based on '%s' priorities",
                   (fSchedOpt == kXPD_sched_central) ? "central" : "local");
      TRACE(ALL, msg);
   }

   if (!rcf) {
      // Start poller thread
      pthread_t tid;
      if (XrdSysThread::Run(&tid, XrdProofdPriorityCron,
                              (void *)this, 0, "PriorityMgr poller thread") != 0) {
         XPDERR("could not start poller thread");
         return 0;
      }
      TRACE(ALL, "poller thread started");
   }

   // Done
   return 0;
}

//__________________________________________________________________________
void XrdProofdPriorityMgr::RegisterDirectives()
{
   // Register directives for configuration

   Register("schedopt", new XrdProofdDirective("schedopt", this, &DoDirectiveClass));
   Register("priority", new XrdProofdDirective("priority", this, &DoDirectiveClass));
}

//______________________________________________________________________________
int XrdProofdPriorityMgr::DoDirective(XrdProofdDirective *d,
                                  char *val, XrdOucStream *cfg, bool rcf)
{
   // Update the priorities of the active sessions.
   XPDLOC(PMGR, "PriorityMgr::DoDirective")

   if (!d)
      // undefined inputs
      return -1;

   if (d->fName == "priority") {
      return DoDirectivePriority(val, cfg, rcf);
   } else if (d->fName == "schedopt") {
      return DoDirectiveSchedOpt(val, cfg, rcf);
   }
   TRACE(XERR, "unknown directive: "<<d->fName);
   return -1;
}

//______________________________________________________________________________
void XrdProofdPriorityMgr::SetGroupPriority(const char *grp, int priority)
{
   // Change group priority. Used when a master pushes a priority to a worker.

   XrdProofGroup *g = fMgr->GroupsMgr()->GetGroup(grp);
   if (g)
      g->SetPriority((float)priority);

   // Make sure scheduling is ON
   SetSchedOpt(kXPD_sched_central);

   // Done
   return;
}

//__________________________________________________________________________
static int ResetEntryPriority(const char *, XrdProofdSessionEntry *e, void *)
{
   // Reset the priority on entries

   if (e) {
      e->SetPriority();
      // Check next
      return 0;
   }

   // Not enough info: stop
   return 1;
}

//__________________________________________________________________________
static int CreateActiveList(const char *, XrdProofdSessionEntry *e, void *s)
{
   // Run thorugh entries to create the sorted list of active entries
   XPDLOC(PMGR, "CreateActiveList")

   XpdCreateActiveList_t *cal = (XpdCreateActiveList_t *)s;

   XrdOucString emsg;
   if (e && cal) {
      XrdProofGroupMgr *gm = cal->fGroupMgr;
      std::list<XrdProofdSessionEntry *> *sorted = cal->fSortedList;
      if (gm) {
         XrdProofGroup *g = gm->GetGroup(e->fGroup.c_str());
         if (g) {
            float ef = g->FracEff() / g->Active();
            int nsrv = g->Active(e->fUser.c_str());
            if (nsrv > 0) {
               ef /= nsrv;
               e->fFracEff = ef;
               bool pushed = 0;
               std::list<XrdProofdSessionEntry *>::iterator ssvi;
               for (ssvi = sorted->begin() ; ssvi != sorted->end(); ssvi++) {
                  if (ef >= (*ssvi)->fFracEff) {
                     sorted->insert(ssvi, e);
                     pushed = 1;
                     break;
                  }
               }
               if (!pushed)
                  sorted->push_back(e);
               // Go to next
               return 0;

            } else {
               emsg = "no srv sessions for active client";
            }
         } else {
            emsg = "group not found: "; emsg += e->fGroup.c_str();
         }
      } else {
         emsg = "group manager undefined";
      }
   } else {
      emsg = "input structure or entry undefined";
   }

   // Some problem
   if (cal) cal->error = 1;
   TRACE(XERR, (e ? e->fUser : "---") << ": protocol error: "<<emsg);
   return 1;
}

//______________________________________________________________________________
int XrdProofdPriorityMgr::SetNiceValues(int opt)
{
   // Recalculate nice values taking into account all active users
   // and their priorities.
   // The type of sessions considered depend on 'opt':
   //    0          all active sessions
   //    1          master sessions
   //    2          worker sessionsg21
   // Return 0 on success, -1 otherwise.
   XPDLOC(PMGR, "PriorityMgr::SetNiceValues")

   TRACE(REQ, "------------------- Start ----------------------");

   TRACE(REQ, "opt: "<<opt);

   if (!fMgr->GroupsMgr() || fMgr->GroupsMgr()->Num() <= 1 || !IsSchedOn()) {
      // Nothing to do
      TRACE(REQ, "------------------- End ------------------------");
      return 0;
   }

   // At least two active session
   int nact = fSessions.Num();
   TRACE(DBG,  fMgr->GroupsMgr()->Num()<<" groups, " << nact<<" active sessions");
   if (nact <= 1) {
      // Restore default values
      if (nact == 1)
         fSessions.Apply(ResetEntryPriority, 0);
      // Nothing else to do
      TRACE(REQ, "------------------- End ------------------------");
      return 0;
   }

   XrdSysMutexHelper mtxh(&fMutex);

   // Determine which groups are active and their effective fractions
   int rc = 0;
   if ((rc = fMgr->GroupsMgr()->SetEffectiveFractions(IsSchedOn())) != 0) {
      // Failure
      TRACE(XERR, "failure from SetEffectiveFractions");
      rc = -1;
   }

   // Now create a list of active sessions sorted by decreasing effective fraction
   TRACE(DBG,  "creating a list of active sessions sorted by decreasing effective fraction ");
   std::list<XrdProofdSessionEntry *> sorted;
   XpdCreateActiveList_t cal = { fMgr->GroupsMgr(), &sorted, 0 };
   if (rc == 0)
      fSessions.Apply(CreateActiveList, (void *)&cal);

   if (!cal.error) {
      // Notify
      int i = 0;
      std::list<XrdProofdSessionEntry *>::iterator ssvi;
      if (TRACING(HDBG)) {
         for (ssvi = sorted.begin() ; ssvi != sorted.end(); ssvi++)
            TRACE(HDBG, i++ <<" eff: "<< (*ssvi)->fFracEff);
      }

      TRACE(DBG,  "calculating nice values");

      // The first has the max priority
      ssvi = sorted.begin();
      float xmax = (*ssvi)->fFracEff;
      if (xmax > 0.) {
         // This is for Unix
         int nice = 20 - fPriorityMax;
         (*ssvi)->SetPriority(nice);
         // The others have priorities rescaled wrt their effective fractions
         ssvi++;
         while (ssvi != sorted.end()) {
            int xpri = (int) ((*ssvi)->fFracEff / xmax * (fPriorityMax - fPriorityMin))
                                                      + fPriorityMin;
            nice = 20 - xpri;
            TRACE(DBG,  "    --> nice value for client "<< (*ssvi)->fUser<<" is "<<nice);
            (*ssvi)->SetPriority(nice);
            ssvi++;
         }
      } else {
         TRACE(XERR, "negative or null max effective fraction: "<<xmax);
         rc = -1;
      }
   } else {
      TRACE(XERR, "failure from CreateActiveList");
      rc = -1;
   }
   TRACE(REQ, "------------------- End ------------------------");

   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdPriorityMgr::DoDirectivePriority(char *val, XrdOucStream *cfg, bool)
{
   // Process 'priority' directive

   if (!val || !cfg)
      // undefined inputs
      return -1;

   // Priority change directive: get delta_priority
   int dp = strtol(val,0,10);
   XrdProofdPriority *p = new XrdProofdPriority("*", dp);
   // Check if an 'if' condition is present
   if ((val = cfg->GetWord()) && !strncmp(val,"if",2)) {
      if ((val = cfg->GetWord()) && val[0]) {
         p->fUser = val;
      }
   }
   // Add to the list
   fPriorities.Rep(p->fUser.c_str(), p);
   return 0;
}

//______________________________________________________________________________
int XrdProofdPriorityMgr::DoDirectiveSchedOpt(char *val, XrdOucStream *cfg, bool)
{
   // Process 'schedopt' directive
   XPDLOC(PMGR, "PriorityMgr::DoDirectiveSchedOpt")

   if (!val || !cfg)
      // undefined inputs
      return -1;

   int pmin = -1;
   int pmax = -1;
   int opt = -1;
   // Defines scheduling options
   while (val && val[0]) {
      XrdOucString o = val;
      if (o.beginswith("min:")) {
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
      if (fMgr->Host() && cfg)
         if (XrdProofdAux::CheckIf(cfg, fMgr->Host()) == 0)
            return 0;
      // Next
      val = cfg->GetWord();
   }

   // Set the values (we need to do it here to avoid setting wrong values
   // when a non-matching 'if' condition is found)
   if (pmin > -1)
      fPriorityMin = (pmin >= 1 && pmin <= 40) ? pmin : fPriorityMin;
   if (pmax > -1)
      fPriorityMax = (pmax >= 1 && pmax <= 40) ? pmax : fPriorityMax;
   if (opt > -1)
      fSchedOpt = opt;

   // Make sure that min is <= max
   if (fPriorityMin > fPriorityMax) {
      TRACE(XERR, "inconsistent value for fPriorityMin (> fPriorityMax) ["<<
                  fPriorityMin << ", "<<fPriorityMax<<"] - correcting");
      fPriorityMin = fPriorityMax;
   }

   return 0;
}

//______________________________________________________________________________
int XrdProofdPriorityMgr::RemoveSession(int pid)
{
   // Remove from the active list the session with ID pid.
   // Return -ENOENT if not found, or 0.

   XrdOucString key; key += pid;
   return fSessions.Del(key.c_str());
}

//______________________________________________________________________________
int XrdProofdPriorityMgr::AddSession(const char *u, const char *g, int pid)
{
   // Add to the active list a session with ID pid. Check for duplications.
   // Returns 1 if the entry existed already and it has been replaced; or 0.

   int rc = 0;
   XrdOucString key; key += pid;
   XrdProofdSessionEntry *oldent = fSessions.Find(key.c_str());
   if (oldent) {
      rc = 1;
      fSessions.Rep(key.c_str(), new XrdProofdSessionEntry(u, g, pid));
   } else {
      fSessions.Add(key.c_str(), new XrdProofdSessionEntry(u, g, pid));
   }

   // Done
   return rc;
}

//__________________________________________________________________________
int XrdProofdPriorityMgr::SetProcessPriority(int pid, const char *user, int &dp)
{
   // Change priority of process pid belonging to user, if needed.
   // Return 0 on success, -errno in case of error
   XPDLOC(PMGR, "PriorityMgr::SetProcessPriority")

   // Change child process priority, if required
   if (fPriorities.Num() > 0) {
      XrdProofdPriority *pu = fPriorities.Find(user);
      if (pu) {
         dp = pu->fDeltaPriority;
         // Change the priority
         errno = 0;
         int priority = XPPM_NOPRIORITY;
         if ((priority = getpriority(PRIO_PROCESS, pid)) == -1 && errno != 0) {
            TRACE(XERR, "getpriority: errno: " << errno);
            return -errno;
         }
         // Set the priority
         int newp = priority + dp;
         XrdProofUI ui;
         XrdProofdAux::GetUserInfo(geteuid(), ui);
         XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
         if (XpdBadPGuard(pGuard, ui.fUid)) {
            TRACE(XERR, "could not get privileges");
            return -1;
         }
         TRACE(REQ, "got privileges ");
         errno = 0;
         if (setpriority(PRIO_PROCESS, pid, newp) != 0) {
            TRACE(XERR, "setpriority: errno: " << errno);
            return ((errno != 0) ? -errno : -1);
         }
         if ((getpriority(PRIO_PROCESS, pid)) != newp && errno != 0) {
            TRACE(XERR, "did not succeed: errno: " << errno);
            return -errno;
         }
      }
   }

   // We are done
   return 0;
}

//
// Small class to describe an active session
//
//______________________________________________________________________________
XrdProofdSessionEntry::XrdProofdSessionEntry(const char *u, const char *g, int pid)
                     : fUser(u), fGroup(g), fPid(pid)
{
   // Constructor
   XPDLOC(PMGR, "XrdProofdSessionEntry")

   fPriority = XPPM_NOPRIORITY;
   fDefaultPriority = XPPM_NOPRIORITY;
   errno = 0;
   int prio = getpriority(PRIO_PROCESS, pid);
   if (errno != 0) {
      TRACE(XERR, " getpriority: errno: " << errno);
      return;
   }
   fDefaultPriority = prio;
}

//______________________________________________________________________________
XrdProofdSessionEntry::~XrdProofdSessionEntry()
{
   // Destructor

   SetPriority(fDefaultPriority);
}

//______________________________________________________________________________
int XrdProofdSessionEntry::SetPriority(int priority)
{
   // Change process priority
   XPDLOC(PMGR, "SessionEntry::SetPriority")

   if (priority != XPPM_NOPRIORITY)
      priority = fDefaultPriority;

   if (priority != fPriority) {
      // Set priority to the default value
      XrdProofUI ui;
      XrdProofdAux::GetUserInfo(geteuid(), ui);
      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, ui.fUid)) {
         TRACE(XERR, "could not get privileges");
         return -1;
      }
      errno = 0;
      if (setpriority(PRIO_PROCESS, fPid, priority) != 0) {
         TRACE(XERR, "setpriority: errno: " << errno);
         return -1;
      }
      fPriority = priority;
   }

   // Done
   return 0;
}
