// @(#)root/proofd:$Name:  $:$Id: XrdProofServProxy.cxx,v 1.17 2007/03/20 16:16:04 rdm Exp $
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string.h>
#include <unistd.h>
#include <sys/uio.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <list>
#include <map>

#include "XrdNet/XrdNet.hh"
#include "XrdSys/XrdSysPriv.hh"
#include "XrdProofServProxy.h"
#include "XrdProofdProtocol.h"
#include "XrdProofWorker.h"

// Tracing utils
#include "XrdProofdTrace.h"
static const char *TraceID = " ";
#define TRACEID TraceID
#ifndef SafeDelete
#define SafeDelete(x) { if (x) { delete x; x = 0; } }
#endif
#ifndef SafeDelArray
#define SafeDelArray(x) { if (x) { delete[] x; x = 0; } }
#endif

//__________________________________________________________________________
XrdProofServProxy::XrdProofServProxy()
{
   // Constructor

   fMutex = new XrdOucRecMutex;
   fLink = 0;
   fParent = 0;
   fPingSem = 0;
   fQueryNum = 0;
   fStartMsg = 0;
   fStatus = kXPD_idle;
   fSrvID = -1;
   fSrvType = kXPD_AnyServer;
   fID = -1;
   fIsShutdown = false;
   fIsValid = true;  // It is created for a valid server ...
   fProtVer = -1;
   fFileout = 0;
   fClient = 0;
   fTag = 0;
   fAlias = 0;
   fOrdinal = 0;
   fUserEnvs = 0;
   fClients.reserve(10);
   fROOT = 0;
   fRequirements = 0;
   fGroup = 0;
   fInflate = 1000;
   fSched = -1;
   fDefSched = -1;
   fDefSchedPriority = 0;
   fFracEff = 0.;
}

//__________________________________________________________________________
XrdProofServProxy::~XrdProofServProxy()
{
   // Destructor

   SafeDelete(fQueryNum);
   SafeDelete(fStartMsg);
   SafeDelete(fRequirements);
   SafeDelete(fPingSem);

   std::vector<XrdClientID *>::iterator i;
   for (i = fClients.begin(); i != fClients.end(); i++)
       if (*i)
          delete (*i);
   fClients.clear();

   // Cleanup worker info
   ClearWorkers();

   SafeDelArray(fClient);
   SafeDelArray(fFileout);
   SafeDelArray(fTag);
   SafeDelArray(fAlias);
   SafeDelArray(fOrdinal);
   SafeDelete(fMutex);
   SafeDelArray(fUserEnvs);
}

//__________________________________________________________________________
void XrdProofServProxy::ClearWorkers()
{
   // Decrease worker counters and clean-up the list

   XrdOucMutexHelper mhp(fMutex);

   // Decrease worker counters
   std::list<XrdProofWorker *>::iterator i;
   for (i = fWorkers.begin(); i != fWorkers.end(); i++)
       if (*i)
          (*i)->fActive--;
   fWorkers.clear();
}

//__________________________________________________________________________
void XrdProofServProxy::Reset()
{
   // Reset this instance
   XrdOucMutexHelper mhp(fMutex);

   fLink = 0;
   fParent = 0;
   SafeDelete(fQueryNum);
   SafeDelete(fRequirements);
   SafeDelete(fStartMsg);
   SafeDelete(fPingSem);
   fStatus = kXPD_idle;
   fSrvID = -1;
   fSrvType = kXPD_AnyServer;
   fID = -1;
   fIsShutdown = false;
   fIsValid = 0;
   fProtVer = -1;
   SafeDelArray(fClient);
   SafeDelArray(fFileout);
   SafeDelArray(fTag);
   SafeDelArray(fAlias);
   SafeDelArray(fOrdinal);
   SafeDelArray(fUserEnvs);
   fClients.clear();
   fROOT = 0;
   fGroup = 0;
   fInflate = 1000;
   fSched = -1;
   fDefSched = -1;
   fDefSchedPriority = 0;
   fFracEff = 0.;
   // Cleanup worker info
   ClearWorkers();
}

//__________________________________________________________________________
XrdClientID *XrdProofServProxy::GetClientID(int cid)
{
   // Get instance corresponding to cid
   //

   XrdOucMutexHelper mhp(fMutex);

   XrdClientID *csid = 0;
   TRACE(ACT,"XrdProofServProxy::GetClientID: cid: "<<cid<<
             ", size: "<<fClients.size());

   if (cid < 0) {
      TRACE(XERR,"XrdProofServProxy::GetClientID: negative ID: protocol error!");
      return csid;
   }

   // If in the allocate range reset the corresponding instance and
   // return it
   if (cid < (int)fClients.size()) {
      csid = fClients.at(cid);
      csid->Reset();
      return csid;
   }

   // If not, allocate a new one; we need to resize (double it)
   if (cid >= (int)fClients.capacity())
      fClients.reserve(2*fClients.capacity());

   // Allocate new elements (for fast access we need all of them)
   int ic = (int)fClients.size();
   for (; ic <= cid; ic++)
      fClients.push_back((csid = new XrdClientID()));

   TRACE(DBG,"XrdProofServProxy::GetClientID: cid: "<<cid<<
             ", new size: "<<fClients.size());

   // We are done
   return csid;
}

//__________________________________________________________________________
int XrdProofServProxy::GetFreeID()
{
   // Get next free client ID. If none is found, increase the vector size
   // and get the first new one

   XrdOucMutexHelper mhp(fMutex);

   int ic = 0;
   // Search for free places in the existing vector
   for (ic = 0; ic < (int)fClients.size() ; ic++) {
      if (fClients[ic] && (fClients[ic]->fP == 0))
         return ic;
   }

   // We need to resize (double it)
   if (ic >= (int)fClients.capacity())
      fClients.reserve(2*fClients.capacity());

   // Allocate new element
   fClients.push_back(new XrdClientID());

   // We are done
   return ic;
}

//__________________________________________________________________________
int XrdProofServProxy::GetNClients()
{
   // Get number of attached clients.

   XrdOucMutexHelper mhp(fMutex);

   int nc = 0;
   // Search for free places in the existing vector
   int ic = 0;
   for (ic = 0; ic < (int)fClients.size() ; ic++)
      if (fClients[ic] && fClients[ic]->IsValid())
         nc++;

   // We are done
   return nc;
}

//__________________________________________________________________________
const char *XrdProofServProxy::StatusAsString() const
{
   // Return a string describing the status

   const char *sst[] = { "idle", "running", "shutting-down", "unknown" };

   XrdOucMutexHelper mhp(fMutex);

   // Check status range
   int ist = fStatus;
   ist = (ist > kXPD_unknown) ? kXPD_unknown : ist;
   ist = (ist < kXPD_idle) ? kXPD_unknown : ist;

   // Done
   return sst[ist];
}

//__________________________________________________________________________
void XrdProofServProxy::SetCharValue(char **carr, const char *v, int l)
{
   // Store null-terminated string at v in *carr

   if (carr) {
      // Reset first
      SafeDelArray(*carr);
      // Store value, if any
      int len = 0;
      if (v && (len = (l > 0) ? l : strlen(v)) > 0) {
         *carr = new char[len+1];
         memcpy(*carr, v, len);
         (*carr)[len] = 0;
      }
   }
}

//______________________________________________________________________________
int XrdProofServProxy::SetShutdownTimer(int opt, int delay, bool on)
{
   // Start (on = TRUE) or stop (on = FALSE) the shutdown timer for the session.
   // Return 0 on success, -1 in case of error.
   int rc = -1;

   TRACE(ACT, "XrdProofServProxy::SetShutdownTimer: enter: on/off: "<<on);

   // Prepare buffer
   int len = 2 *sizeof(kXR_int32);
   char *buf = new char[len];
   // Option
   kXR_int32 itmp = (on) ? (kXR_int32)opt : -1;
   itmp = static_cast<kXR_int32>(htonl(itmp));
   memcpy(buf, &itmp, sizeof(kXR_int32));
   // Delay
   itmp = (on) ? (kXR_int32)delay : 0;
   itmp = static_cast<kXR_int32>(htonl(itmp));
   memcpy(buf + sizeof(kXR_int32), &itmp, sizeof(kXR_int32));
   // Send over
   if (ProofSrv()->Send(kXR_attn, kXPD_timer, buf, len) != 0) {
      TRACE(XERR, "XrdProofServProxy::SetShutdownTimer: "
                  "could not send shutdown info to proofsrv");
   } else {
      rc = 0;
      XrdOucString msg = "XrdProofServProxy::SetShutdownTimer: ";
      if (on) {
         if (delay > 0) {
            msg += "delayed (";
            msg += delay;
            msg += " secs) ";
         }
         msg += "shutdown notified to process ";
         msg += SrvID();
         if (opt == 1)
            msg += "; action: when idle";
         else if (opt == 2)
            msg += "; action: immediate";
         SetShutdown(1);
      } else {
         msg += "cancellation of shutdown action notified to process ";
         msg += SrvID();
         SetShutdown(0);
      }
      TRACE(DBG, msg.c_str());
   }
   // Cleanup
   delete[] buf;

   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofServProxy::TerminateProofServ()
{
   // Terminate the associated process.
   // A shutdown interrupt message is forwarded.
   // If add is TRUE (default) the pid is added to the list of processes
   // requested to terminate.
   // Return the pid of tyhe terminated process on success, -1 if not allowed
   // or other errors occured.

   TRACE(ACT, "XrdProofServProxy::TerminateProofServ: enter: " << Ordinal());

   // Send a terminate signal to the proofserv
   int pid = SrvID();
   if (pid > -1) {

      int type = 3;
      if (ProofSrv()->Send(kXR_attn, kXPD_interrupt, type) != 0)
         // Could not send: signal failure
         return -1;
      // For registration/monitoring purposes
      return pid;
   }

   // Failed
   return -1;
}

//______________________________________________________________________________
int XrdProofServProxy::VerifyProofServ(int timeout)
{
   // Check if the associated proofserv process is alive.
   // A ping message is sent and the reply waited for the internal timeout.
   // Return 1 if successful, 0 if reply was not received within the
   // internal timeout, -1 in case of error.
   int rc = -1;

   TRACE(ACT, "XrdProofServProxy::VerifyProofServ: enter");

   // Create semaphore
   CreatePingSem();

   // Propagate the ping request
   if (ProofSrv()->Send(kXR_attn, kXPD_ping, 0, 0) != 0) {
      TRACE(XERR, "XrdProofServProxy::VerifyProofServ: could not propagate ping to proofsrv");
      DeletePingSem();
      return rc;
   }

   // Wait for reply
   rc = 1;
   if (PingSem()->Wait(timeout) != 0) {
      XrdOucString msg = "XrdProofServProxy::VerifyProofServ: did not receive ping reply after ";
      msg += timeout;
      msg += " secs";
      TRACE(XERR, msg.c_str());
      rc = 0;
   }

   // Cleanup
   DeletePingSem();

   // Done
   return rc;
}
//__________________________________________________________________________
int XrdProofServProxy::ChangeProcessPriority(int dp)
{
   // Change priority of the server process by dp (positive or negative)
   // Returns 0 in case of success, -errno in case of error.

   TRACE(ACT, "ChangeProcessPriority: enter: pid: " << fSrvID << ", dp: " << dp);

   // No action requested
   if (dp == 0)
      return 0;

   // Get current priority; errno needs to be reset here, as -1
   // could be a legitimate priority value
   errno = 0;
   int priority = getpriority(PRIO_PROCESS, fSrvID);
   if (priority == -1 && errno != 0) {
      TRACE(XERR, "ChangeProcessPriority:"
                 " getpriority: errno: " << errno);
      return -errno;
   }

   // Reference priority
   int refpriority = priority + dp;

   // Chaneg the priority
   if (setpriority(PRIO_PROCESS, fSrvID, refpriority) != 0) {
      TRACE(XERR, "ChangeProcessPriority:"
                 " setpriority: errno: " << errno);
      return ((errno != 0) ? -errno : -1);
   }

   // Check that it worked out
   errno = 0;
   if ((priority = getpriority(PRIO_PROCESS, fSrvID)) == -1 && errno != 0) {
      TRACE(XERR, "ChangeProcessPriority:"
                 " getpriority: errno: " << errno);
      return -errno;
   }
   if (priority != refpriority) {
      TRACE(XERR, "ChangeProcessPriority:"
                 " unexpected result of action: found " << priority <<
                 ", expected "<<refpriority);
      errno = EPERM;
      return -errno;
   }

   // We are done
   return 0;
}

//__________________________________________________________________________
int XrdProofServProxy::SetInflate(int inflate, bool sendover)
{
   // Set the inflate factor for this session; this factor is used to
   // artificially inflate the processing time (by delaying new packet
   // requests) to control resource sharing.
   // If 'sendover' is TRUE the factor is communicated to proofserv,
   // otherwise is just stored.

   XrdOucMutexHelper mhp(fMutex);
   fInflate = inflate;

   if (sendover) {
      // Prepare buffer
      int len = sizeof(kXR_int32);
      char *buf = new char[len];
      kXR_int32 itmp = inflate;
      itmp = static_cast<kXR_int32>(htonl(itmp));
      memcpy(buf, &itmp, sizeof(kXR_int32));
      // Send over
      if (fProofSrv.Send(kXR_attn, kXPD_inflate, buf, len) != 0) {
         // Failure
         TRACE(XERR,"XrdProofServProxy::SetInflate: problems telling proofserv");
         return -1;
      }
      TRACE(DBG,"XrdProofServProxy::SetInflate: inflate factor set to "<<inflate);
   }
   // Done
   return 0;
}

//__________________________________________________________________________
void XrdProofServProxy::SetSrv(int pid)
{
   // Set the server PID. Also find the scheduling policy

   XrdOucMutexHelper mhp(fMutex);

   // The PID
   fSrvID = pid;

#if !defined(__FreeBSD__) && !defined(__OpenBSD__) && !defined(__APPLE__)
   // The scheduling policy
   fSched = sched_getscheduler(pid);
   fDefSched = fSched;
#endif

   // Done
   return;
}

//__________________________________________________________________________
int XrdProofServProxy::SetSchedRoundRobin(bool on)
{
   // Set the scheduling policy for process 'pid' to SCHED_RR (Round Robin)
   // if on is TRUE, or to the original one if on is FALSE.
   // Round Robin is needed to control the exact CPU time assigned to a
   // process. This is needed when the priority-based worker level load
   // balancing is enabled. Under Linux, the default policy SCHED_OTHER increases
   // dynamically the priority of sleeping processes so that load balancing
   // based on the slowdown of low priority sessions does not work.
   // Return 0 on success, -1 if any problem occured.

   TRACE(ACT, "SetSchedRoundRobin: enter: pid: " << fSrvID<<", ON: "<<on);
#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)
   TRACE(ACT, "SetSchedRoundRobin: functionality unsupported on this platform");
#else

   // Next depends on 'on'
   if (on) {

      const char *lab = "SetSchedRoundRobin: ON: ";

      // Nothing to do if already Round Robin
      if ((fDefSched = sched_getscheduler(fSrvID)) == SCHED_RR) {
         TRACE(DBG, lab << "current policy is already SCHED_RR - do nothing");
         return 0;
      }

      // Save current parameters
      sched_getparam(fSrvID, &fDefSchedParam);

      // Min RoundRobin priorities
      int pr_min = sched_get_priority_min(SCHED_RR);
      if (pr_min < 0) {
         TRACE(XERR, lab << "sched_get_priority_min: errno: "<<errno);
         return -1;
      }

      // Retrieve the privileges
      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);

      // Set new schema with minimal priority
      struct sched_param par;
      par.sched_priority = pr_min;
      if (sched_setscheduler(fSrvID, SCHED_RR, &par) != 0) {
         TRACE(XERR, lab << "sched_setscheduler: errno: "<<errno);
         return -1;
      }

      // We increase the nice level to avoid overloading the machine
      fDefSchedPriority = getpriority(PRIO_PROCESS, fSrvID);
      if (setpriority(PRIO_PROCESS, fSrvID, fDefSchedPriority + 5) != 0) {
         TRACE(XERR, lab << "setpriority: errno: "<<errno);
      }

      // Current policy
      fSched = fDefSched;

//         TRACE(DBG, lab << "scheduling policy set to SCHED_RR for process "<<fSrvID);
      XPDPRT(lab << "scheduling policy set to SCHED_RR for process "<<fSrvID);

   } else {

      const char *lab = "SetSchedRoundRobin: OFF: ";

      // Nothing to do if already done
      if ((fSched = sched_getscheduler(fSrvID)) == fDefSched) {
         TRACE(DBG, lab << "current policy the default one - do nothing");
         return 0;
      }

      // Retrieve the privileges
      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);

      // Set default policy and params
      if (sched_setscheduler(fSrvID, fDefSched, &fDefSchedParam) != 0) {
         TRACE(XERR, lab << "sched_setscheduler: errno: "<<errno);
         return -1;
      }

      // Reset initial scheduling priority
      if (setpriority(PRIO_PROCESS, fSrvID, fDefSchedPriority) != 0) {
         TRACE(XERR, lab << "setpriority: errno: "<<errno);
      }

      // Current policy
      fSched = fDefSched;

//         TRACE(DBG, lab << "scheduling policy set to "<<fSched<<" for process "<<fSrvID);
      XPDPRT(lab << "scheduling policy set to  "<<fSched<<" for process "<<fSrvID);
   }
#endif
   // Done
   return 0;
}
