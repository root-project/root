// @(#)root/proofd:$Id$
// Author: G. Ganis  September 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofSched                                                        //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Interface for a PROOF scheduler.                                     //
// Alternative scheduler implementations should be provided as shared   //
// library containing an implementation of this class. The library      //
// must also provide a function to load an instance of this class       //
// with the following signature (see commented example below):          //
// extern "C" {                                                         //
//    XrdProofSched *XrdgetProofSched(const char *cfg,                  //
//                                    XrdProofdManager *mgr,            //
//                                    XrdProofGroupMgr *grpmgr,         //
//                                    XrdSysError *edest);              //
// }                                                                    //
// Here 'cfg' is the xrootd config file where directives to configure   //
// the scheduler are specified, 'mgr' is the instance of the cluster    //
// manager from where the scheduler can get info about the available    //
// workers and their status, 'grpmgr' is the instance of the group      //
// bringing the definition of the groups for this run, and 'edest' is   //
// instance of the error logger to be used.                             //
// The scheduler is identified by a name of max 16 chars.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <list>

#include "XProofProtocol.h"
#include "XrdProofdManager.h"
#include "XrdProofdNetMgr.h"
#include "XrdProofdProofServMgr.h"
#include "XrdProofGroup.h"
#include "XrdProofSched.h"
#include "XrdProofdProofServ.h"
#include "XrdProofWorker.h"

#include "XrdOuc/XrdOucString.hh"
#include "XrdOuc/XrdOucStream.hh"

#include "XpdSysError.h"

// Tracing
#include "XrdProofdTrace.h"

//
// Example of scheduler loader for an implementation called XrdProofSchedDyn
//
// extern "C" {
// //______________________________________________________________________________
// XrdProofSched *XrdgetProofSched(const char *cfg, XrdProofdManager *mgr,
//                                 XrdProofGroupMgr *grpmgr, XrdSysError *edest)
// {
//   // This scheduler is meant to live in a shared library. The interface below is
//   // used by the server to obtain a copy of the scheduler object.
//
//   XrdProofSchedDyn *pss = new XrdProofSchedDyn(mgr, grpmgr, edest);
//   if (pss && pss->Config(cfg) == 0) {
//      return (XrdProofSched *) pss;
//   }
//   if (pss)
//      delete pss;
//   return (XrdProofSched *)0;
// }}

//--------------------------------------------------------------------------
//
// XrdProofSchedCron
//
// Scheduler thread
//
//--------------------------------------------------------------------------
void *XrdProofSchedCron(void *p)
{
   // This is an endless loop to check the system periodically or when
   // triggered via a message in a dedicated pipe
   XPDLOC(SCHED, "SchedCron")

   XrdProofSched *sched = (XrdProofSched *)p;
   if (!(sched)) {
      TRACE(XERR, "undefined scheduler: cannot start");
      return (void *)0;
   }

   // Time of last session check
   int lastcheck = time(0), ckfreq = sched->CheckFrequency(), deltat = 0;
   while(1) {
      // We wait for processes to communicate a session status change
      if ((deltat = ckfreq - (time(0) - lastcheck)) <= 0)
         deltat = ckfreq;
      int pollRet = sched->Pipe()->Poll(deltat);

      if (pollRet > 0) {
         // Read message
         XpdMsg msg;
         int rc = 0;
         if ((rc = sched->Pipe()->Recv(msg)) != 0) {
            XPDERR("problems receiving message; errno: "<<-rc);
            continue;
         }
         // Parse type
         XrdOucString buf;
         if (msg.Type() == XrdProofSched::kReschedule) {

            TRACE(ALL, "received kReschedule");

            // Reschedule
            sched->Reschedule();

         } else {

            TRACE(XERR, "unknown type: "<<msg.Type());
            continue;
         }
      } else {
         // Notify
         TRACE(ALL, "running regular checks");
         // Run regular rescheduling checks
         sched->Reschedule();
         // Remember when ...
         lastcheck = time(0);
      }
   }

   // Should never come here
   return (void *)0;
}

//______________________________________________________________________________
static bool XpdWrkComp(XrdProofWorker *&lhs, XrdProofWorker *&rhs)
{
   // Compare two workers for sorting

   return ((lhs && rhs &&
            lhs->GetNActiveSessions() < rhs->GetNActiveSessions()) ? 1 : 0);
}

//______________________________________________________________________________
int DoSchedDirective(XrdProofdDirective *d, char *val, XrdOucStream *cfg, bool rcf)
{
   // Generic directive processor

   if (!d || !(d->fVal))
      // undefined inputs
      return -1;

   return ((XrdProofSched *)d->fVal)->ProcessDirective(d, val, cfg, rcf);
}

//______________________________________________________________________________
XrdProofSched::XrdProofSched(const char *name,
                             XrdProofdManager *mgr, XrdProofGroupMgr *grpmgr,
                             const char *cfn,  XrdSysError *e)
              : XrdProofdConfig(cfn, e)
{
   // Constructor

   fValid = 1;
   fMgr = mgr;
   fGrpMgr = grpmgr;
   fNextWrk = 1;
   fEDest = e;
   fUseFIFO = 0;
   ResetParameters();

   memset(fName, 0, kXPSMXNMLEN);
   if (name)
      memcpy(fName, name, kXPSMXNMLEN-1);

   // Configuration directives
   RegisterDirectives();
}

//__________________________________________________________________________
void XrdProofSched::RegisterDirectives()
{
   // Register directives for configuration

   Register("schedparam", new XrdProofdDirective("schedparam", this, &DoDirectiveClass));
   Register("resource", new XrdProofdDirective("resource", this, &DoDirectiveClass));
}

//______________________________________________________________________________
int XrdProofSched::DoDirective(XrdProofdDirective *d,
                               char *val, XrdOucStream *cfg, bool rcf)
{
   // Update the priorities of the active sessions.
   XPDLOC(SCHED, "Sched::DoDirective")

   if (!d)
      // undefined inputs
      return -1;

   if (d->fName == "schedparam") {
      return DoDirectiveSchedParam(val, cfg, rcf);
   } else if (d->fName == "resource") {
      return DoDirectiveResource(val, cfg, rcf);
   }
   TRACE(XERR,"unknown directive: "<<d->fName);
   return -1;
}


//______________________________________________________________________________
void XrdProofSched::ResetParameters()
{
   // Reset values for the configurable parameters

   fMaxSessions = -1;
   fMaxRunning = -1;
   fWorkerMax = -1;
   fWorkerSel = kSSORoundRobin;
   fOptWrksPerUnit = 1;
   fMinForQuery = 0;
   fNodesFraction = 0.5;
   fCheckFrequency = 30;
}

//______________________________________________________________________________
int XrdProofSched::Config(bool rcf)
{
   // Configure this instance using the content of file 'cfn'.
   // Return 0 on success, -1 in case of failure (file does not exists
   // or containing incoherent information).
   XPDLOC(SCHED, "Sched::Config")

   // Run first the configurator
   if (XrdProofdConfig::Config(rcf) != 0) {
      XPDERR("problems parsing file ");
      fValid = 0;
      return -1;
   }

   int rc = 0;

   XrdOucString msg;

   // Notify
   XPDFORM(msg, "maxsess: %d, maxrun: %d, maxwrks: %d, selopt: %d, fifo:%d",
                fMaxSessions, fMaxRunning, fWorkerMax, fWorkerSel, fUseFIFO);
   TRACE(DBG, msg);

   if (!rcf) {
      // Start cron thread
      pthread_t tid;
      if (XrdSysThread::Run(&tid, XrdProofSchedCron,
                           (void *)this, 0, "Scheduler cron thread") != 0) {
         XPDERR("could not start cron thread");
         fValid = 0;
         return 0;
      }
      TRACE(ALL, "cron thread started");
   }

   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofSched::Enqueue(XrdProofdProofServ *xps, XrdProofQuery *query)
{
   // Queue a query in the session; if this is the first querym enqueue also
   // the session
   XPDDOM(SCHED)

   if (xps->Enqueue(query) == 1) {
      std::list<XrdProofdProofServ *>::iterator ii;
      for (ii = fQueue.begin(); ii != fQueue.end(); ii++) {
         if ((*ii)->Status() == kXPD_running) break;
      }
      if (ii != fQueue.end()) {
         fQueue.insert(ii, xps);
      } else {
         fQueue.push_back(xps);
      }
   }
   if (TRACING(DBG)) DumpQueues("Enqueue");

   return 0;
}

//______________________________________________________________________________
void XrdProofSched::DumpQueues(const char *prefix)
{
   // Dump the content of the waiting sessions queue

   XPDLOC(SCHED, "DumpQueues")

   TRACE(ALL," ++++++++++++++++++++ DumpQueues ++++++++++++++++++++++++++++++++ ");
   if (prefix) TRACE(ALL, " +++ Called from: "<<prefix);
   TRACE(ALL," +++ # of waiting sessions: "<<fQueue.size()); 
   std::list<XrdProofdProofServ *>::iterator ii;
   int i = 0;
   for (ii = fQueue.begin(); ii != fQueue.end(); ii++) {
      TRACE(ALL," +++ #"<<++i<<" client:"<< (*ii)->Client()<<" # of queries: "<< (*ii)->Queries()->size());
   }
   TRACE(ALL," ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ");

   return;
}

//______________________________________________________________________________
XrdProofdProofServ *XrdProofSched::FirstSession()
{
   // Get first valid session.
   // The dataset information can be used to assign workers.
   XPDDOM(SCHED)

   if (fQueue.empty())
      return 0;
   XrdProofdProofServ *xps = fQueue.front();
   while (xps && !(xps->IsValid())) {
      fQueue.remove(xps);
      xps = fQueue.front();
   }
   if (TRACING(DBG)) DumpQueues("FirstSession");
   // The session will be removed after workers are assigned
   return xps;
}

//______________________________________________________________________________
int XrdProofSched::GetNumWorkers(XrdProofdProofServ *xps)
{
   // Calculate the number of workers to be used given the state of the cluster
   XPDLOC(SCHED, "Sched::GetNumWorkers")

   // Go through the list of hosts and see how many CPUs are not used.
   int nFreeCPUs = 0;
   std::list<XrdProofWorker *> *wrks = fMgr->NetMgr()->GetActiveWorkers();
   std::list<XrdProofWorker *>::iterator iter;
   for (iter = wrks->begin(); iter != wrks->end(); ++iter) {
      TRACE(DBG, (*iter)->fImage<<" : # act: "<<(*iter)->fProofServs.size());
      if ((*iter)->fType != 'M' && (*iter)->fType != 'S'
          && (int) (*iter)->fProofServs.size() < fOptWrksPerUnit)
         // add number of free slots
         nFreeCPUs += fOptWrksPerUnit - (*iter)->fProofServs.size();
   }

   float priority = 1;
   XrdProofGroup *grp = 0;
   if (fGrpMgr && xps->Group())
      grp = fGrpMgr->GetGroup(xps->Group());
   if (grp) {
      std::list<XrdProofdProofServ *> *sessions = fMgr->SessionMgr()->ActiveSessions();
      std::list<XrdProofdProofServ *>::iterator sesIter;
      float summedPriority = 0;
      for (sesIter = sessions->begin(); sesIter != sessions->end(); ++sesIter) {
         if ((*sesIter)->Group()) {
            XrdProofGroup *g = fGrpMgr->GetGroup((*sesIter)->Group());
            if (g)
               summedPriority += g->Priority();
         }
      }
      if (summedPriority > 0)
         priority = (grp->Priority() * sessions->size()) / summedPriority;
   }

   int nWrks = (int)(nFreeCPUs * fNodesFraction * priority);
   if (nWrks <= fMinForQuery) {
      nWrks = fMinForQuery;
   } else if (nWrks >= (int) wrks->size()) {
      nWrks = wrks->size() - 1;
   }
   TRACE(DBG, nFreeCPUs<<" : "<< nWrks);

   return nWrks;
}

//______________________________________________________________________________
int XrdProofSched::GetWorkers(XrdProofdProofServ *xps,
                              std::list<XrdProofWorker *> *wrks,
                              const char *querytag)
{
   // Get a list of workers that can be used by session 'xps'.
   // The return code is:
   //  -1     Some failure occured; cannot continue
   //   0     A new list has been assigned to the session 'xps' and
   //         returned in 'wrks'
   //   1     The list currently assigned to the session is the one
   //         to be used
   //   2     No worker could be assigned now; session should be queued

   XPDLOC(SCHED, "Sched::GetWorkers")

   int rc = 0;

   TRACE(REQ, "enter: query tag: "<< ((querytag) ? querytag : ""));

   // Static or dynamic
   bool isDynamic = 1;
   if (querytag && !strncmp(querytag, XPD_GW_Static, strlen(XPD_GW_Static) - 1)) {
      isDynamic = 0;
   }

   // Check if the current assigned list of workers is valid
   if (querytag && xps && xps->Workers()->Num() > 0) {
      if (TRACING(REQ)) xps->DumpQueries();
      const char *cqtag = (xps->CurrentQuery()) ? xps->CurrentQuery()->GetTag() : "undef";
      TRACE(REQ, "current query tag: "<< cqtag );
      if (!strcmp(querytag, cqtag)) {
         // Remove the query to be processed from the queue
         xps->RemoveQuery(cqtag);
         TRACE(REQ, "current assignment for session "<< xps->SrvPID() << " is valid");
         // Current assignement is valid
         return 1;
      }
   }

   // The caller must provide a list where to store the result
   if (!wrks)
      return -1;

   // If the session has already assigned workers or there are
   // other queries waiting - just enqueue
   // FIFO is enforced by dynamic mode so it is checked just in case
   if (isDynamic) {
      if (fUseFIFO && xps->Workers()->Num() > 0) {
         if (!xps->GetQuery(querytag))
            Enqueue(xps, new XrdProofQuery(querytag));
         if (TRACING(DBG)) xps->DumpQueries();
         // Signal enqueing
         TRACE(REQ, "session has already assigned workers: enqueue");
         return 2;
      }
   }

   // The current, full list
   std::list<XrdProofWorker *> *acws = 0;

   if (!fMgr || !(acws = fMgr->NetMgr()->GetActiveWorkers()))
      return -1;

   // Point to the master element
   XrdProofWorker *mst = acws->front();
   if (!mst)
      return -1;

   if (fWorkerSel == kSSOLoadBased) {
      // Dynamic scheduling: the scheduler will determine the #workers
      // to be used based on the current load and assign the least loaded ones

      // Sort the workers by the load
      XrdProofWorker::Sort(acws, XpdWrkComp);

      // Get the advised number
      int nw = GetNumWorkers(xps);

      if (nw > 0) {
         // The master first (stats are updated in XrdProofdProtocol::GetWorkers)
         wrks->push_back(mst);

         std::list<XrdProofWorker *>::iterator nxWrk = acws->begin();
         while (nw--) {
            nxWrk++;
            // Add export version of the info
            // (stats are updated in XrdProofdProtocol::GetWorkers)
            wrks->push_back(*nxWrk);
         }
      } else {
         // if no workers were assigned
         // enqueue or send a list with only the master (processing refused)
         if (fUseFIFO) {
            // Enqueue the query/session
            // the returned list of workers was not filled
            if (!xps->GetQuery(querytag))
               Enqueue(xps, new XrdProofQuery(querytag));
            if (TRACING(DBG)) xps->DumpQueries();
            // Signal enqueing
            TRACE(REQ, "no workers currently available: session enqueued");
            return 2;
         } else {
            // The master first (stats are updated in XrdProofdProtocol::GetWorkers)
            wrks->push_back(mst);
         }
      }
      // Done
      return 0;
   }

   // Check if the check on the max number of sessions is enabled
   // We need at least 1 master and a worker
   std::list<XrdProofWorker *> *acwseff = 0;
   int maxnum = (querytag && strcmp(querytag, XPD_GW_Static)) ? fMaxRunning : fMaxSessions;
   bool ok = 0;
   if (isDynamic) {
      if (maxnum > 0) {
         acwseff = new std::list<XrdProofWorker *>;
         std::list<XrdProofWorker *>::iterator xWrk = acws->begin();
         if ((*xWrk)->Active() < maxnum) {
            acwseff->push_back(*xWrk);
            xWrk++;
            for (; xWrk != acws->end(); xWrk++) {
               if ((*xWrk)->Active() < maxnum) {
                  acwseff->push_back(*xWrk);
                  ok = 1;
               }
            }
         } else if (!fUseFIFO) {
            TRACE(REQ, "max number of sessions reached - ("<< maxnum <<")");
         }
         // Check the result
         if (!ok) { delete acwseff; acwseff = 0; }
         acws = acwseff;
      }
   } else {
      if (maxnum > 0) {
         // This is over-conservative for sub-selectiob (random, or round-robin options)
         // A better solution should be implemented for that.
         int nactsess = mst->GetNActiveSessions();
         TRACE(REQ, "act sess ... " << nactsess);
         if (nactsess < maxnum) {
            ok = 1;
         } else if (!fUseFIFO) {
            TRACE(REQ, "max number of sessions reached - ("<< maxnum <<")");
         }
         // Check the result
         if (!ok) acws = acwseff;
      }
   }

   // Make sure that something has been found
   if (!acws || acws->size() <= 1) {
      if (fUseFIFO) {
         // Enqueue the query/session
         // the returned list of workers was not filled
         if (!xps->GetQuery(querytag))
            Enqueue(xps, new XrdProofQuery(querytag));
         if (TRACING(REQ)) xps->DumpQueries();
         // Notify enqueing
         TRACE(REQ, "no workers currently available: session enqueued");
         return 2;
      } else {
         TRACE(XERR, "no worker available: do nothing");
         if (acwseff) { delete acwseff; acwseff = 0; }
         return -1;
      }
   }

   // If the session has already assigned workers just return
   if (xps->Workers()->Num() > 0) {
      // Current assignement is valid
      return 1;
   }

   // The master first (stats are updated in XrdProofdProtocol::GetWorkers)
   wrks->push_back(mst);

   if (fWorkerMax > 0 && fWorkerMax < (int) acws->size()) {

      // Now the workers
      if (fWorkerSel == kSSORandom) {
         // Random: the first time init the machine
         static bool rndmInit = 0;
         if (!rndmInit) {
            const char *randdev = "/dev/urandom";
            int fd;
            unsigned int seed;
            if ((fd = open(randdev, O_RDONLY)) != -1) {
               if (read(fd, &seed, sizeof(seed)) != sizeof(seed)) {
                  TRACE(XERR, "problems reading seed; errno: "<< errno);
               }
               srand(seed);
               close(fd);
               rndmInit = 1;
            }
         }
         // Selection
         int nwt = acws->size();
         std::vector<int> walloc(nwt, 0);
         std::vector<XrdProofWorker *> vwrk(nwt);

         // Fill the vector with cumulative number of actives
         int namx = -1;
         int i = 1;
         std::list<XrdProofWorker *>::iterator iwk = acws->begin();
         iwk++; // Skip master
         for ( ; iwk != acws->end(); iwk++) {
            vwrk[i] = *iwk;
            int na = (*iwk)->Active();
            printf(" %d", na);
            walloc[i] = na + walloc[i-1];
            i++;
            namx = (na > namx) ? na : namx;
         }
         printf("\n");
         // Normalize
         for (i = 1; i < nwt; i++) {
            if (namx > 0)
               walloc[i] = namx*i - walloc[i] + i;
            else
               walloc[i] = i;
         }
         int natot = walloc[nwt - 1];

         int nw = fWorkerMax;
         while (nw--) {
            // Normalized number
            int maxAtt = 10000, natt = 0;
            int iw = -1;
            while ((iw < 1 || iw >= nwt) && natt < maxAtt) {
               int jw = rand() % natot;
               for (i = 0; i < nwt; i++) {
                  if (jw < walloc[i]) {
                     // re-normalize the weights for the higher index entries
                     int j = 0;
                     for (j = i; j < nwt; j++) {
                        if (walloc[j] > 0)
                           walloc[j]--;
                     }
                     natot--;
                     iw = i;
                     break;
                  }
               }
            }

            if (iw > -1) {
               // Add to the list (stats are updated in XrdProofdProtocol::GetWorkers)
               wrks->push_back(vwrk[iw]);
            } else {
               // Unable to generate the right number
               TRACE(XERR, "random generation failed");
               rc = -1;
               break;
            }
         }

      } else {
         if (fNextWrk >= (int) acws->size())
            fNextWrk = 1;
         int iw = 0;
         std::list<XrdProofWorker *>::iterator nxWrk = acws->begin();
         int nw = fWorkerMax;
         while (nw--) {
            while (iw != fNextWrk) {
               nxWrk++;
               iw++;
            }
            // Add export version of the info
            // (stats are updated in XrdProofdProtocol::GetWorkers)
            wrks->push_back(*nxWrk);
            // Update next worker index
            fNextWrk++;
            if (fNextWrk >= (int) acws->size()) {
               fNextWrk = 1;
               iw = 0;
               nxWrk = acws->begin();
            }
         }
      }
   } else {
      // The full list
      std::list<XrdProofWorker *>::iterator iw = acws->begin();
      iw++;
      while (iw != acws->end()) {
         // Add to the list (stats are updated in XrdProofdProtocol::GetWorkers)
         wrks->push_back(*iw);
         iw++;
      }
   }

   // Make sure that something has been found
   if (wrks->size() <= 1) {
      TRACE(XERR, "no worker found: do nothing");
      rc = -1;
   }

   // Cleanup
   if (acwseff) { delete acwseff; acwseff = 0; }

   return rc;
}

//______________________________________________________________________________
int XrdProofSched::Reschedule()
{
   // Consider starting some query from the queue.
   // to be called after some resources are free (e.g. by a finished query)
   // This method is doing the full transaction of finding the session to
   // resume, assigning it workers and sending a resume message.
   // In this way there is not possibility of interference with other GetWorkers
   // return 0 in case of success and -1 in case of an error
   XPDDOM(SCHED)

   if (fUseFIFO && TRACING(DBG)) DumpQueues("Reschedule");

   if (!fQueue.empty()) {
      // Any advanced scheduling algorithms can be done here

      XrdProofdProofServ *xps = FirstSession();
      XrdOucString wrks;
      // Call GetWorkers in the manager to mark the assignment.
      XrdOucString qtag;
      if (xps && xps->CurrentQuery()) {
         qtag = xps->CurrentQuery()->GetTag();
         if (qtag.beginswith(XPD_GW_Static)) {
            qtag = XPD_GW_Static;
            qtag.replace(":","");
         }
      }
      if (fMgr->GetWorkers(wrks, xps, qtag.c_str()) < 0 ) {
         // Something wrong
         return -1;
      } else {
         // Send buffer
         // if workers were assigned remove the session from the queue
         if (wrks.length() > 0 && wrks != XPD_GW_QueryEnqueued) {
            // Send the resume message: the workers will be send in response to a
            // GetWorkers message
            xps->Resume();
            // Acually remove the session from the queue
            fQueue.remove(xps);
            // Put the session at the end of the queue
            // > 1 because the query is kept in the queue until 2nd GetWorkers
            if (xps->Queries()->size() > 1)
               fQueue.push_back(xps);
            if (TRACING(DBG)) DumpQueues("Reschedule 2");
         } // else add workers to the running sessions (once it's possible)

      }

   } //else add workers to the running sessions (once it's possible)

   return 0;
}

//______________________________________________________________________________
int XrdProofSched::ExportInfo(XrdOucString &sbuf)
{
   // Fill sbuf with some info about our current status

   // Selection type
   const char *osel[] = { "all", "round-robin", "random", "load-based"};
   sbuf += "Selection: ";
   sbuf += osel[fWorkerSel+1];
   if (fWorkerSel > -1) {
      sbuf += ", max workers: ";
      sbuf += fWorkerMax; sbuf += " &";
   }

   // The full list
   std::list<XrdProofWorker *> *acws = fMgr->NetMgr()->GetActiveWorkers();
   std::list<XrdProofWorker *>::iterator iw;
   for (iw = acws->begin(); iw != acws->end(); ++iw) {
      sbuf += (*iw)->fType;
      sbuf += ": "; sbuf += (*iw)->fHost;
      if ((*iw)->fPort > -1) {
         sbuf += ":"; sbuf += (*iw)->fPort;
      } else
         sbuf += "     ";
      sbuf += "  sessions: "; sbuf += (*iw)->Active();
      sbuf += " &";
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofSched::ProcessDirective(XrdProofdDirective *d,
                                    char *val, XrdOucStream *cfg, bool rcf)
{
   // Update the priorities of the active sessions.
   XPDLOC(SCHED, "Sched::ProcessDirective")

   if (!d)
      // undefined inputs
      return -1;

   if (d->fName == "schedparam") {
      return DoDirectiveSchedParam(val, cfg, rcf);
   } else if (d->fName == "resource") {
      return DoDirectiveResource(val, cfg, rcf);
   }
   TRACE(XERR, "unknown directive: "<<d->fName);
   return -1;
}

//______________________________________________________________________________
int XrdProofSched::DoDirectiveSchedParam(char *val, XrdOucStream *cfg, bool)
{
   // Process 'schedparam' directive
   XPDLOC(SCHED, "Sched::DoDirectiveSchedParam")

   if (!val || !cfg)
      // undefined inputs
      return -1;

   // Get the parameters
   while (val && val[0]) {
      XrdOucString s(val);
      if (s.beginswith("wmx:")) {
         s.replace("wmx:","");
         fWorkerMax = strtol(s.c_str(), (char **)0, 10);
      } else if (s.beginswith("mxsess:")) {
         s.replace("mxsess:","");
         fMaxSessions = strtol(s.c_str(), (char **)0, 10);
      } else if (s.beginswith("mxrun:")) {
         s.replace("mxrun:","");
         fMaxRunning = strtol(s.c_str(), (char **)0, 10);
      } else if (s.beginswith("selopt:")) {
         if (s.endswith("random"))
            fWorkerSel = kSSORandom;
         else if (s.endswith("load"))
            fWorkerSel = kSSOLoadBased;
         else
            fWorkerSel = kSSORoundRobin;
      } else if (s.beginswith("fraction:")) {
         s.replace("fraction:","");
         fNodesFraction = strtod(s.c_str(), (char **)0);
      } else if (s.beginswith("optnwrks:")) {
         s.replace("optnwrks:","");
         fOptWrksPerUnit = strtol(s.c_str(), (char **)0, 10);
      } else if (s.beginswith("minforquery:")) {
         s.replace("minforquery:","");
         fMinForQuery = strtol(s.c_str(), (char **)0, 10);
      } else if (s.beginswith("queue:")) {
         if (s.endswith("fifo")) {
            fUseFIFO = 1;
         }
      } else if (strncmp(val, "default", 7)) {
         // This line applies to another scheduler
         ResetParameters();
         break;
      }
      val = cfg->GetWord();
   }

   // If the max number of sessions is limited then there is no lower bound
   // the number of workers per query
   if (fMaxSessions > 0) {
      fMinForQuery = 0;
      // And there is an upper limit on the number of running sessions
      if (fMaxRunning < 0 || fMaxRunning > fMaxSessions)
         fMaxRunning = fMaxSessions;
   }

   // The FIFO size make sense only in non-load based mode
   if (fWorkerSel == kSSOLoadBased && fMaxRunning > 0) {
      TRACE(ALL, "WARNING: in Load-Based mode the max number of sessions"
                 " to be run is determined dynamically");
   }

   return 0;
}

//______________________________________________________________________________
int XrdProofSched::DoDirectiveResource(char *val, XrdOucStream *cfg, bool)
{
   // Process 'resource' directive

   if (!val || !cfg)
      // undefined inputs
      return -1;

   // Get the scheduler name
   if (strncmp(val, "static", 6) && strncmp(val, "default", 7))
      return 0;
   // Get the values
   while ((val = cfg->GetWord()) && val[0]) {
      XrdOucString s(val);
      if (s.beginswith("wmx:")) {
         s.replace("wmx:","");
         fWorkerMax = strtol(s.c_str(), (char **)0, 10);
      } else if (s.beginswith("mxsess:")) {
         s.replace("mxsess:","");
         fMaxSessions = strtol(s.c_str(), (char **)0, 10);
      } else if (s.beginswith("selopt:")) {
         if (s.endswith("random"))
            fWorkerSel = kSSORandom;
         else
            fWorkerSel = kSSORoundRobin;
      }
   }
   return 0;
}
