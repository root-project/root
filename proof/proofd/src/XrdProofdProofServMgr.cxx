// @(#)root/proofd:$Id$
// Author: G. Ganis Jan 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdProofServMgr                                                  //
//                                                                      //
// Author: G. Ganis, CERN, 2008                                         //
//                                                                      //
// Class managing proofserv sessions manager.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "XrdProofdPlatform.h"

#ifdef OLDXRDOUC
#  include "XrdOuc/XrdOucError.hh"
#  include "XrdOuc/XrdOucLogger.hh"
#else
#  include "XrdSys/XrdSysError.hh"
#  include "XrdSys/XrdSysLogger.hh"
#endif

#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdPoll.hh"
#include "Xrd/XrdScheduler.hh"
#include "XrdNet/XrdNet.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "XrdOuc/XrdOucRash.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPriv.hh"
#include "XrdSut/XrdSutAux.hh"

#include "XrdProofdClient.h"
#include "XrdProofdClientMgr.h"
#include "XrdProofdManager.h"
#include "XrdProofdNetMgr.h"
#include "XrdProofdPriorityMgr.h"
#include "XrdProofdProofServMgr.h"
#include "XrdProofdProtocol.h"
#include "XrdProofdLauncher.h"
#include "XrdProofGroup.h"
#include "XrdProofSched.h"
#include "XrdROOT.h"

#include <map>

// Aux structures for scan through operations
typedef struct {
   XrdProofGroupMgr *fGroupMgr;
   int *fNBroadcast;
} XpdBroadcastPriority_t;
typedef struct {
   XrdProofdManager *fMgr;
   XrdProofdClient *fClient;
   FILE *fEnv;
   bool fExport;
} XpdWriteEnv_t;

#ifndef PutEnv
#define PutEnv(x,e) { if (e) { putenv(x); } else { delete[] x; } }
#endif

// Tracing utilities
#include "XrdProofdTrace.h"

static XpdManagerCron_t fManagerCron;

//--------------------------------------------------------------------------
//
// XrdProofdProofServCron
//
// Function run in separate thread watching changes in session status
// frequency
//
//--------------------------------------------------------------------------
void *XrdProofdProofServCron(void *p)
{
   // This is an endless loop to check the system periodically or when
   // triggered via a message in a dedicated pipe
   XPDLOC(SMGR, "ProofServCron")

   XpdManagerCron_t *mc = (XpdManagerCron_t *)p;
   XrdProofdProofServMgr *mgr = mc->fSessionMgr;
   XrdProofSched *sched = mc->fProofSched;
   if (!(mgr)) {
      TRACE(XERR,  "undefined session manager: cannot start");
      return (void *)0;
   }

   // Quicj checks for client disconnections: frequency (5 secs) and
   // flag for disconnections effectively occuring
   int quickcheckfreq = 5;
   int clnlostscale = 0;

   // Time of last full sessions check
   int lastrun = time(0);
   int lastcheck = lastrun, ckfreq = mgr->CheckFrequency(), waitt = 0;
   int deltat = ((int)(0.1*ckfreq) >= 1) ? (int)(0.1*ckfreq) : 1;
   int maxdelay = 5*ckfreq; // Force check after 5 times the check frequency
   mgr->SetNextSessionsCheck(lastcheck + ckfreq);
   TRACE(ALL, "next full sessions check in "<<ckfreq<<" secs");
   while(1) {
      // We check for client disconnections every 'quickcheckfreq' secs; we do
      // a full check every mgr->CheckFrequency() secs; we make sure that we
      // do not pass a negative value (meaning no timeout)
      waitt =  ckfreq - (time(0) - lastcheck);
      if (waitt > quickcheckfreq || waitt <= 0)
         waitt = quickcheckfreq;
      int pollRet = mgr->Pipe()->Poll(waitt);

      if (pollRet > 0) {
         // Read message
         XpdMsg msg;
         int rc = 0;
         if ((rc = mgr->Pipe()->Recv(msg)) != 0) {
            TRACE(XERR, "problems receiving message; errno: "<<-rc);
            continue;
         }
         // Parse type
         if (msg.Type() == XrdProofdProofServMgr::kSessionRemoval) {
            // A session has just gone: read process id
            XrdOucString fpid;
            if ((rc = msg.Get(fpid)) != 0) {
               TRACE(XERR, "kSessionRemoval: problems receiving process ID (buf: '"<<
                           msg.Buf()<<"'); errno: "<<-rc);
               continue;
            }
            XrdSysMutexHelper mhp(mgr->Mutex());
            // Remove it from the hash list
            mgr->DeleteFromSessions(fpid.c_str());
            // Move the entry to the terminated sessions area
            mgr->MvSession(fpid.c_str());
            // Notify the scheduler too
            if (sched) {
               if (sched->Pipe()->Post(XrdProofSched::kReschedule, 0) != 0) {
                  TRACE(XERR, "kSessionRemoval: problem posting the scheduler pipe");
               }
            }
            // Notify action
            TRACE(REQ, "kSessionRemoval: session: "<<fpid<<
                        " has been removed from the active list");
         } else if (msg.Type() == XrdProofdProofServMgr::kClientDisconnect) {
            // A client just disconnected: we free the slots in the proofserv sesssions and
            // we check the sessions status to see if any of them must be terminated
            // read process id
            int pid = 0;
            if ((rc = msg.Get(pid)) != 0) {
               TRACE(XERR, "kClientDisconnect: problems receiving process ID (buf: '"<<
                           msg.Buf()<<"'); errno: "<<-rc);
               continue;
            }
            TRACE(REQ, "kClientDisconnect: a client just disconnected: "<<pid);
            // Free slots in the proof serv instances
            mgr->DisconnectFromProofServ(pid);
            TRACE(DBG, "quick check of active sessions");
            // Quick check of active sessions in case of disconnections
            mgr->CheckActiveSessions(0);
        } else if (msg.Type() == XrdProofdProofServMgr::kCleanSessions) {
            // Request for cleanup all sessions of a client (or all clients)
            XpdSrvMgrCreateCnt cnt(mgr, XrdProofdProofServMgr::kCleanSessionsCnt);
            XrdOucString usr;
            rc = msg.Get(usr);
            int svrtype;
            rc = (rc == 0) ? msg.Get(svrtype) : rc;
            if (rc != 0) {
               TRACE(XERR, "kCleanSessions: problems parsing message (buf: '"<<
                           msg.Buf()<<"'); errno: "<<-rc);
               continue;
            }
            // Notify action
            TRACE(REQ, "kCleanSessions: request for user: '"<<usr<<"', server type: "<<svrtype);
            // Clean sessions
            mgr->CleanClientSessions(usr.c_str(), svrtype);
            // Check if there is any orphalin sessions and clean them up
            mgr->CleanupLostProofServ();
         } else if (msg.Type() == XrdProofdProofServMgr::kProcessReq) {
            // Process request from some client: if we are here it means they can go ahead
            mgr->ProcessSem()->Post();
         } else if (msg.Type() == XrdProofdProofServMgr::kChgSessionSt) {
            // Propagate cluster information to active sessions after one session changed its state
            mgr->BroadcastClusterInfo();
         } else {
            TRACE(XERR, "unknown type: "<<msg.Type());
            continue;
         }
      } else {

         // The current time
         int now = time(0);

         // If there is any activity in mgr->Process() we postpone the checks in 5 secs
         int cnt = mgr->CheckCounter(XrdProofdProofServMgr::kProcessCnt);
         if (cnt > 0) {
            if ((now - lastrun) < maxdelay) {
               // The current time
               lastcheck = now + 5 - ckfreq;
               mgr->SetNextSessionsCheck(now + 5);
               // Notify
               TRACE(ALL, "postponing sessions check (will retry in 5 secs)");
               continue;
            } else {
               // Max time without checks reached: force a check
               TRACE(ALL, "Max time without checks reached ("<<maxdelay<<"): force a session check");
               // Reset the counter
               mgr->UpdateCounter(XrdProofdProofServMgr::kProcessCnt, -cnt);
            }
         }

         bool full = (now > mgr->NextSessionsCheck() - deltat) ? 1 : 0;
         if (full) {
            // Run periodical full checks
            mgr->CheckActiveSessions();
            mgr->CheckTerminatedSessions();
            if (clnlostscale <= 0) {
               mgr->CleanupLostProofServ();
               clnlostscale = 10;
            } else {
               clnlostscale--;
            }
            // How many active sessions do we have
            int cursess = mgr->CurrentSessions(1);
            TRACE(ALL, cursess << " sessions are currently active");
            // Remember when ...
            lastrun = now;
            lastcheck = now;
            mgr->SetNextSessionsCheck(lastcheck + mgr->CheckFrequency());
            // Notify
            TRACE(ALL, "next sessions check in "<<mgr->CheckFrequency()<<" secs");
         } else {
            TRACE(HDBG, "nothing to do; "<<mgr->NextSessionsCheck()-now<<" secs to full check");
         }
      }
   }

   // Should never come here
   return (void *)0;
}

//--------------------------------------------------------------------------
//
// XrdProofdProofServRecover
//
// Function run in a separate thread waiting for session to recover after
// an abrupt shutdown
//
//--------------------------------------------------------------------------
void *XrdProofdProofServRecover(void *p)
{
   // Waiting for session to recover after an abrupt shutdown
   XPDLOC(SMGR, "ProofServRecover")

   XpdManagerCron_t *mc = (XpdManagerCron_t *)p;
   XrdProofdProofServMgr *mgr = mc->fSessionMgr;
   if (!(mgr)) {
      TRACE(XERR,  "undefined session manager: cannot start");
      return (void *)0;
   }

   // Recover active sessions
   int rc = mgr->RecoverActiveSessions();

   // Notify end of recovering
   if (rc > 0) {
      TRACE(ALL, "timeout recovering sessions: "<<rc<<" sessions not recovered");
   } else if (rc < 0) {
      TRACE(XERR, "some problem occured while recovering sessions");
   } else {
      TRACE(ALL, "recovering successfully terminated");
   }

   // Should never come here
   return (void *)0;
}

//______________________________________________________________________________
XrdProofdProofServMgr::XrdProofdProofServMgr(XrdProofdManager *mgr,
                                             XrdProtocol_Config *pi, XrdSysError *e)
                      : XrdProofdConfig(pi->ConfigFN, e), fProcessSem(0)
{
   // Constructor
   XPDLOC(SMGR, "XrdProofdProofServMgr")

   fMgr = mgr;
   fLogger = pi->eDest->logger();
   fInternalWait = 10;
   fActiveSessions.clear();
   fShutdownOpt = 1;
   fShutdownDelay = 0;
   fReconnectTime = -1;
   fReconnectTimeOut = 300;
   fNextSessionsCheck = -1;
   // Init internal counters
   for (int i = 0; i < PSMMAXCNTS; i++) {
      fCounters[i] = 0;
   }
   fCurrentSessions = 0;

   fSeqSessionN = 0;

   // Defaults can be changed via 'proofservmgr'
   fCheckFrequency = 30;
   fTerminationTimeOut = fCheckFrequency - 10;
   fVerifyTimeOut = 3 * fCheckFrequency;
   fRecoverTimeOut = 10;
   fCheckLost = 1;
   fUseFork = 1;
   fParentExecs = "xproofd,xrootd";

   // Recover-related quantities
   fRecoverClients = 0;
   fRecoverDeadline = -1;

   // Init pipe for the poller
   if (!fPipe.IsValid()) {
      TRACE(XERR, "unable to generate pipe for the session poller");
      return;
   }

   // Configuration directives
   RegisterDirectives();
}

//__________________________________________________________________________
int XrdProofdProofServMgr::Config(bool rcf)
{
   // Run configuration and parse the entered config directives.
   // Return 0 on success, -1 on error
   XPDLOC(SMGR, "ProofServMgr::Config")

   XrdSysMutexHelper mhp(fEnvsMutex);

   bool notify = (rcf) ? 0 : 1;
   if (rcf && ReadFile(0)) {
      // Cleanup lists of envs and RCs
      fProofServRCs.clear();
      fProofServEnvs.clear();
      // Notify possible new settings
      notify = 1;
   }

   // Run first the configurator
   if (XrdProofdConfig::Config(rcf) != 0) {
      TRACE(XERR, "problems parsing file ");
      return -1;
   }

   XrdOucString msg;
   msg = (rcf) ? "re-configuring" : "configuring";
   if (notify) XPDPRT(msg);

   // Notify timeout on internal communications
   XPDFORM(msg, "setting internal timeout to %d secs", fInternalWait);
   if (notify) XPDPRT(msg);

   // Shutdown options
   msg = "client sessions shutdown after disconnection";
   if (fShutdownOpt > 0) {
      XPDFORM(msg, "client sessions kept %sfor %d secs after disconnection",
                   (fShutdownOpt == 1) ? "idle " : "", fShutdownDelay);
   }
   if (notify) XPDPRT(msg);

   if (!rcf) {
      // Admin paths
      fActiAdminPath = fMgr->AdminPath();
      fActiAdminPath += "/activesessions";
      fTermAdminPath = fMgr->AdminPath();
      fTermAdminPath += "/terminatedsessions";

      // Make sure they exist
      XrdProofUI ui;
      XrdProofdAux::GetUserInfo(fMgr->EffectiveUser(), ui);
      if (XrdProofdAux::AssertDir(fActiAdminPath.c_str(), ui, 1) != 0) {
         TRACE(XERR, "unable to assert the admin path: "<<fActiAdminPath);
         fActiAdminPath = "";
         return -1;
      }
      XPDPRT("active sessions admin path set to: "<<fActiAdminPath);

      if (XrdProofdAux::AssertDir(fTermAdminPath.c_str(), ui, 1) != 0) {
         TRACE(XERR, "unable to assert the admin path "<<fTermAdminPath);
         fTermAdminPath = "";
         return -1;
      }
      XPDPRT("terminated sessions admin path set to "<<fTermAdminPath);
   }

   if (notify) {
      XPDPRT("RC settings: "<< fProofServRCs.size());
      if (fProofServRCs.size() > 0) {
         std::list<XpdEnv>::iterator ircs = fProofServRCs.begin();
         for ( ; ircs != fProofServRCs.end(); ircs++) { (*ircs).Print("rc"); }
      }
      XPDPRT("ENV settings: "<< fProofServEnvs.size());
      if (fProofServEnvs.size() > 0) {
         std::list<XpdEnv>::iterator ienvs = fProofServEnvs.begin();
         for ( ; ienvs != fProofServEnvs.end(); ienvs++) { (*ienvs).Print("env"); }
      }
   }

   // Notify sessions startup technology
   XPDFORM(msg, "using %s to start proofserv sessions", fUseFork ? "fork()" : "system()");
   if (notify) XPDPRT(msg);

   if (!rcf) {
      // Try to recover active session previously started
      int nr = -1;
      if ((nr = PrepareSessionRecovering()) < 0) {
         TRACE(XERR, "problems trying to recover active sessions");
      } else if (nr > 0) {
         XPDFORM(msg, "%d active sessions have been recovered", nr);
         XPDPRT(msg);
      }

      // Start cron thread
      pthread_t tid;
      // Fill manager pointers structure
      fManagerCron.fClientMgr = fMgr->ClientMgr();
      fManagerCron.fSessionMgr = this;
      if (XrdSysThread::Run(&tid, XrdProofdProofServCron,
                            (void *)&fManagerCron, 0, "ProofServMgr cron thread") != 0) {
         TRACE(XERR, "could not start cron thread");
         return 0;
      }
      XPDPRT("cron thread started");
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::AddSession(XrdProofdProtocol *p, XrdProofdProofServ *s)
{
   // Add new active session
   XPDLOC(SMGR, "ProofServMgr::AddSession")

   TRACE(REQ, "adding new active session ...");

   // Check inputs
   if (!s || !p || !p->Client()) {
      TRACE(XERR,"invalid inputs: "<<p<<", "<<s<<", "<<p->Client());
      return -1;
   }
   XrdProofdClient *c = p->Client();

   // Path
   XrdOucString path;
   XPDFORM(path, "%s/%s.%s.%d", fActiAdminPath.c_str(), c->User(), c->Group(), s->SrvPID());

   // Save session info to file
   XrdProofSessionInfo info(c, s);
   int rc = info.SaveToFile(path.c_str());

   return rc;
}

//______________________________________________________________________________
bool XrdProofdProofServMgr::IsSessionSocket(const char *fpid)
{
   // Checks is fpid is the path of a session UNIX socket
   // Returns TRUE is yes; cleans the socket if the session is gone.
   XPDLOC(SMGR, "ProofServMgr::IsSessionSocket")

   TRACE(REQ, "checking "<<fpid<<" ...");

   // Check inputs
   if (!fpid || strlen(fpid) <= 0) {
      TRACE(XERR, "invalid input: "<<fpid);
      return 0;
   }

   // Paths
   XrdOucString spath(fpid);
   if (!spath.endswith(".sock")) return 0;
   if (!spath.beginswith(fActiAdminPath.c_str())) {
      // We are given a partial path: create full paths
      XPDFORM(spath, "%s/%s", fActiAdminPath.c_str(), fpid);
   }
   XrdOucString apath = spath;
   apath.replace(".sock", "");

   // Check the admin path
   struct stat st;
   if (stat(apath.c_str(), &st) != 0 && (errno == ENOENT)) {
      // Remove the socket path if not during creation
      if (CheckCounter(kCreateCnt) <= 0) {
         unlink(spath.c_str());
         TRACE(REQ, "missing admin path: removing "<<spath<<" ...");
      }
   }

   // Done
   return 1;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::MvSession(const char *fpid)
{
   // Move session file from the active to the terminated areas
   XPDLOC(SMGR, "ProofServMgr::MvSession")

   TRACE(REQ, "moving "<<fpid<<" ...");

   // Check inputs
   if (!fpid || strlen(fpid) <= 0) {
      TRACE(XERR, "invalid input: "<<fpid);
      return -1;
   }

   // Paths
   XrdOucString opath(fpid), npath;
   if (!opath.beginswith(fActiAdminPath.c_str())) {
      // We are given a partial path: create full paths
      XPDFORM(opath, "%s/%s", fActiAdminPath.c_str(), fpid);
      opath.replace(".status", "");
   } else {
      // Full path: just create the new path
      opath.replace(".status", "");
   }
   // The target path
   npath = opath;
   npath.replace(fActiAdminPath.c_str(), fTermAdminPath.c_str());

   // Remove the socket path
   XrdOucString spath = opath;
   spath += ".sock";
   if (unlink(spath.c_str()) != 0 && errno != ENOENT)
      TRACE(XERR, "problems removing the UNIX socket path: "<<spath<<"; errno: "<<errno);
   spath.replace(".sock", ".status");
   if (unlink(spath.c_str()) != 0 && errno != ENOENT)
      TRACE(XERR, "problems removing the status file: "<<spath<<"; errno: "<<errno);

   // Move the file
   errno = 0;
   int rc = 0;
   if ((rc = rename(opath.c_str(), npath.c_str())) == 0 || (errno == ENOENT)) {
      if (!rc)
         // Record the time when we did this
         TouchSession(fpid, npath.c_str());
      return 0;
   }

   TRACE(XERR, "session pid file cannot be moved: "<<opath<<
              "; target file: "<<npath<<"; errno: "<<errno);
   return -1;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::RmSession(const char *fpid)
{
   // Remove session file from the terminated sessions area
   XPDLOC(SMGR, "ProofServMgr::RmSession")

   TRACE(REQ, "removing "<<fpid<<" ...");

   // Check inputs
   if (!fpid || strlen(fpid) <= 0) {
      TRACE(XERR, "invalid input: "<<fpid);
      return -1;
   }

   // Path
   XrdOucString path;
   XPDFORM(path, "%s/%s", fTermAdminPath.c_str(), fpid);

   // remove the file
   if (unlink(path.c_str()) == 0)
      return 0;

   TRACE(XERR, "session pid file cannot be unlinked: "<<
               path<<"; error: "<<errno);
   return -1;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::TouchSession(const char *fpid, const char *fpath)
{
   // Update the access time for the session pid file to the current time
   XPDLOC(SMGR, "ProofServMgr::TouchSession")

   TRACE(REQ, "touching "<<fpid<<", "<<fpath<<" ...");

   // Check inputs
   if (!fpid || strlen(fpid) <= 0) {
      TRACE(XERR, "invalid input: "<<fpid);
      return -1;
   }

   // Path
   XrdOucString path(fpath);
   if (!fpath || strlen(fpath) == 0)
      XPDFORM(path, "%s/%s.status", fActiAdminPath.c_str(), fpid);

   // Update file time stamps
   if (utime(path.c_str(), 0) == 0)
      return 0;

   TRACE(XERR, "time stamps for session pid file cannot be updated: "<<
               path<<"; error: "<<errno);
   return -1;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::VerifySession(const char *fpid,
                                         int to, const char *fpath)
{
   // Check if the session is alive, i.e. if it has recently touched its admin file.
   // Return 0 if alive, 1 if not-responding, -1 in case of error.
   // The timeout for verification is 'to' if positive, else fVerifyTimeOut;
   // the admin file is looked under 'fpath' if defined, else fActiAdminPath.
   XPDLOC(SMGR, "ProofServMgr::VerifySession")

   // Check inputs
   if (!fpid || strlen(fpid) <= 0) {
      TRACE(XERR, "invalid input: "<<fpid);
      return -1;
   }

   // Path
   XrdOucString path;
   if (fpath && strlen(fpath) > 0)
      XPDFORM(path, "%s/%s", fpath, fpid);
   else
      XPDFORM(path, "%s/%s", fActiAdminPath.c_str(), fpid);

   // Check first the new file but also the old one, for backward compatibility
   int deltat = -1;
   bool checkmore = 1;
   while (checkmore) {
      // Current settings
      struct stat st;
      if (stat(path.c_str(), &st)) {
         TRACE(XERR, "session status file cannot be stat'ed: "<<
                     path<<"; error: "<<errno);
         return -1;
      }
      // Check times
      int xto = (to > 0) ? to : fVerifyTimeOut;
      deltat = time(0) - st.st_mtime;
      if (deltat > xto) {
         if (path.endswith(".status")) {
            // Check the old one too
            path.erase(path.rfind(".status"));
         } else {
            // Dead
            TRACE(DBG, "admin path for session "<<fpid<<" hase not been touched"
                       " since at least "<< xto <<" secs");
            return 1;
         }
      } else {
         // We are done
         checkmore = 0;
      }
   }

   // Alive
   TRACE(DBG, "admin path for session "<<fpid<<" was touched " <<
              deltat <<" secs ago");
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::DeleteFromSessions(const char *fpid)
{
   // Delete from the hash list the session with ID pid.
   // Return -ENOENT if not found, or 0.
   XPDLOC(SMGR, "ProofServMgr::DeleteFromSessions")

   TRACE(REQ, "session: "<<fpid);

   // Check inputs
   if (!fpid || strlen(fpid) <= 0) {
      TRACE(XERR, "invalid input: "<<fpid);
      return -1;
   }

   XrdOucString key = fpid;
   key.replace(".status", "");
   key.erase(0, key.rfind('.') + 1);
   XrdProofdProofServ *xps = 0;
   { XrdSysMutexHelper mhp(fMutex); xps = fSessions.Find(key.c_str()); }
   if (xps) {
      // Tell other attached clients, if any, that this session is gone
      XrdOucString msg;
      XPDFORM(msg, "session: %s terminated by peer", fpid);
      TRACE(DBG, msg);
      // Reset this instance
      int tp = xps->Reset(msg.c_str(), kXPD_wrkmortem);
      // Update counters and lists
      XrdSysMutexHelper mhp(fMutex);
      if (tp == 1) fCurrentSessions--;
      // remove from the list of active sessions
      fActiveSessions.remove(xps);
   }
   int rc = -1;
   { XrdSysMutexHelper mhp(fMutex); rc = fSessions.Del(key.c_str()); }
   return rc;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::PrepareSessionRecovering()
{
   // Go through the active sessions admin path and prepare reconnection of those
   // still alive.
   // Called at start-up.
   XPDLOC(SMGR, "ProofServMgr::PrepareSessionRecovering")

   // Open dir
   DIR *dir = opendir(fActiAdminPath.c_str());
   if (!dir) {
      TRACE(XERR, "cannot open dir "<<fActiAdminPath<<" ; error: "<<errno);
      return -1;
   }
   TRACE(REQ, "preparing recovering of active sessions ...");

   // Scan the active sessions admin path
   fRecoverClients = new std::list<XpdClientSessions *>;
   struct dirent *ent = 0;
   while ((ent = (struct dirent *)readdir(dir))) {
      if (!strncmp(ent->d_name, ".", 1) || !strncmp(ent->d_name, "..", 2)) continue;
      // Get the session instance (skip non-digital entries)
      XrdOucString rest, a;
      int pid = XrdProofdAux::ParsePidPath(ent->d_name, rest, a);
      if (!XPD_LONGOK(pid) || pid <= 0) continue;
      if (a.length() > 0) continue;
      bool rmsession = 1;
      // Check if the process is still alive
      if (XrdProofdAux::VerifyProcessByID(pid) != 0) {
         if (ResolveSession(ent->d_name) == 0) {
            TRACE(DBG, "found active session: "<<pid);
            rmsession = 0;
         }
      }
      // Remove the session, if needed
      if (rmsession)
         MvSession(ent->d_name);
   }
   // Close the directory
   closedir(dir);

   // Start the recovering thread, if needed
   int nrc = 0;
   { XrdSysMutexHelper mhp(fRecoverMutex); nrc = fRecoverClients->size(); }
   if (nrc > 0) {
      // Start recovering thread
      pthread_t tid;
      // Fill manager pointers structure
      fManagerCron.fClientMgr = fMgr->ClientMgr();
      fManagerCron.fSessionMgr = this;
      fManagerCron.fProofSched = fMgr->ProofSched();
      if (XrdSysThread::Run(&tid, XrdProofdProofServRecover, (void *)&fManagerCron,
                            0, "ProofServMgr session recover thread") != 0) {
         TRACE(XERR, "could not start session recover thread");
         return 0;
      }
      XPDPRT("session recover thread started");
   } else {
      // End reconnect state if there is nothing to reconnect
      if (fMgr->ClientMgr() && fMgr->ClientMgr()->GetNClients() <= 0)
         SetReconnectTime(0);
   }

   // Done
   return 0;
}


//______________________________________________________________________________
int XrdProofdProofServMgr::RecoverActiveSessions()
{
   // Accept connections from sessions still alive. This is run in a dedicated
   // thread.
   // Returns -1 in case of failure, 0 if all alive sessions reconnected or the
   // numer of sessions not reconnected if the timeout (fRecoverTimeOut per client)
   // expired.
   XPDLOC(SMGR, "ProofServMgr::RecoverActiveSessions")

   int rc = 0;

   if (!fRecoverClients) {
      // Invalid input
      TRACE(XERR, "recovering clients list undefined");
      return -1;
   }

   int nrc = 0;
   { XrdSysMutexHelper mhp(fRecoverMutex); nrc = fRecoverClients->size(); }
   TRACE(REQ, "start recovering of "<<nrc<<" clients");

   // Recovering deadline
   { XrdSysMutexHelper mhp(fRecoverMutex);
     fRecoverDeadline = time(0) + fRecoverTimeOut * nrc; }

   // Respect the deadline
   int nr = 0;
   XpdClientSessions *cls = 0;
   bool go = true;
   while (go) {

      // Pickup the first one in the list
      { XrdSysMutexHelper mhp(fRecoverMutex); cls = fRecoverClients->front(); }
      if (cls) {
         SetReconnectTime();
         nr += Recover(cls);

         // If all client sessions reconnected remove the client from the list
         {  XrdSysMutexHelper mhp(cls->fMutex);
            if (cls->fProofServs.size() <= 0) {
               XrdSysMutexHelper mhpr(fRecoverMutex);
               fRecoverClients->remove(cls);
               // We may be over
               if ((nrc = fRecoverClients->size()) <= 0)
                  break;
            }
         }
      }
      TRACE(REQ, nrc<<" clients still to recover");

      // Check the deadline
      {  XrdSysMutexHelper mhp(fRecoverMutex);
         go = (time(0) < fRecoverDeadline) ? true : false; }
   }
   // End reconnect state
   SetReconnectTime(0);

   // If we reached the deadline, calculate the number of sessions not reconnected
   rc = 0;
   {  XrdSysMutexHelper mhp(fRecoverMutex);
      if (fRecoverClients->size() > 0) {
         std::list<XpdClientSessions* >::iterator ii = fRecoverClients->begin();
         for (; ii != fRecoverClients->end(); ii++) {
            rc += (*ii)->fProofServs.size();
         }
      }
   }

   // Delete the recovering clients list
   {  XrdSysMutexHelper mhp(fRecoverMutex);
      fRecoverClients->clear();
      delete fRecoverClients;
      fRecoverClients = 0;
      fRecoverDeadline = -1;
   }

   // Done
   return rc;
}

//______________________________________________________________________________
bool XrdProofdProofServMgr::IsClientRecovering(const char *usr, const char *grp,
                                               int &deadline)
{
   // Returns true (an the recovering deadline) if the client has sessions in
   // recovering state; returns false otherwise.
   // Called during for attach requests.
   XPDLOC(SMGR, "ProofServMgr::IsClientRecovering")

   if (!usr || !grp) {
      TRACE(XERR, "invalid inputs: usr: "<<usr<<", grp:"<<grp<<" ...");
      return false;
   }

   deadline = -1;
   int rc = false;
   {  XrdSysMutexHelper mhp(fRecoverMutex);
      if (fRecoverClients && fRecoverClients->size() > 0) {
         std::list<XpdClientSessions *>::iterator ii = fRecoverClients->begin();
         for (; ii != fRecoverClients->end(); ii++) {
            if ((*ii)->fClient && (*ii)->fClient->Match(usr, grp)) {
               rc = true;
               deadline = fRecoverDeadline;
               break;
            }
         }
      }
   }
   TRACE(DBG, "checking usr: "<<usr<<", grp:"<<grp<<" ... recovering? "<<
              rc<<", until: "<<deadline);

   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::CheckActiveSessions(bool verify)
{
   // Go through the active sessions admin path and make sure sessions are alive.
   // If 'verify' is true also ask the session to proof that they are alive
   // via asynchronous ping (the result will be done at next check).
   // Move those not responding in the terminated sessions admin path.
   XPDLOC(SMGR, "ProofServMgr::CheckActiveSessions")

   TRACE(REQ, "checking active sessions ...");

   // Open dir
   DIR *dir = opendir(fActiAdminPath.c_str());
   if (!dir) {
      TRACE(XERR, "cannot open dir "<<fActiAdminPath<<" ; error: "<<errno);
      return -1;
   }

   // Scan the active sessions admin path
   struct dirent *ent = 0;
   while ((ent = (struct dirent *)readdir(dir))) {
      if (!strncmp(ent->d_name, ".", 1) || !strncmp(ent->d_name, "..", 2)) continue;
      // If a socket path, make sure that the associated session still exists
      // and go to the next
      if (strstr(ent->d_name, ".sock") && IsSessionSocket(ent->d_name)) continue;
      // Get the session instance (skip non-digital entries)
      XrdOucString rest, key, after;
      int pid = XrdProofdAux::ParsePidPath(ent->d_name, rest, after);
      // If not a status path, go to the next
      if (after != "status") continue;
      // If not a good pid
      if (!XPD_LONGOK(pid) || pid <= 0) continue;
      key += pid;
      //
      XrdProofdProofServ *xps = 0;
      {  XrdSysMutexHelper mhp(fMutex);
         xps = fSessions.Find(key.c_str());
      }

      bool sessionalive = (VerifySession(ent->d_name) == 0) ? 1 : 0;
      bool rmsession = 0;
      if (xps) {
         if (!xps->IsValid() || !sessionalive) rmsession = 1;
      } else {
         // Session not yet registered, possibly starting
         // Skips checks the admin file verification was OK
         if (sessionalive) continue;
         rmsession = 1;
      }

      // For backward compatibility we need to check the session version
      bool oldvers = (xps && xps->ROOT() && xps->ROOT()->SrvProtVers() >= 18) ? 0 : 1;

      // If somebody is interested in this session, we give her/him some
      // more time by skipping the connected clients check this time
      int nc = -1;
      if (!rmsession)
         rmsession = xps->CheckSession(oldvers, IsReconnecting(),
                                       fShutdownOpt, fShutdownDelay, fMgr->ChangeOwn(), nc);

      // Verify the session: this just sends a request to the session
      // to touch the session file; all this will be done asynchronously;
      // the result will be checked next time.
      // We do not want further propagation at this stage.
      if (!rmsession && verify && !oldvers) {
         if (xps->VerifyProofServ(0) != 0) {
            // This means that the connection is already gone
            rmsession = 1;
         }
      }
      TRACE(REQ, "session: "<<ent->d_name<<"; nc: "<<nc<<"; rm: "<<rmsession);
      // Remove the session, if needed
      if (rmsession)
         MvSession(ent->d_name);
   }
   // Close the directory
   closedir(dir);

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::CheckTerminatedSessions()
{
   // Go through the terminated sessions admin path and make sure sessions they
   // are gone.
   // Hard-kill those still alive.
   XPDLOC(SMGR, "ProofServMgr::CheckTerminatedSessions")

   TRACE(REQ, "checking terminated sessions ...");

   // Open dir
   DIR *dir = opendir(fTermAdminPath.c_str());
   if (!dir) {
      TRACE(XERR, "cannot open dir "<<fTermAdminPath<<" ; error: "<<errno);
      return -1;
   }

   // Scan the terminated sessions admin path
   int now = -1;
   struct dirent *ent = 0;
   while ((ent = (struct dirent *)readdir(dir))) {
      if (!strncmp(ent->d_name, ".", 1) || !strncmp(ent->d_name, "..", 2)) continue;
      // Get the session instance (skip non-digital entries)
      XrdOucString rest, a;
      int pid = XrdProofdAux::ParsePidPath(ent->d_name, rest, a);
      if (!XPD_LONGOK(pid) || pid <= 0) continue;

      // Current time
      now = (now > 0) ? now : time(0);

      // Full path
      XrdOucString path;
      XPDFORM(path, "%s/%s", fTermAdminPath.c_str(), ent->d_name);

      // Check termination time
      struct stat st;
      int rcst = stat(path.c_str(), &st);
      TRACE(DBG, pid<<": rcst: "<<rcst<<", now - mtime: "<<now - st.st_mtime<<" secs")
      if ((now - st.st_mtime) > fTerminationTimeOut || rcst != 0) {
         // Check if the process is still alive
         if (XrdProofdAux::VerifyProcessByID(pid) != 0) {
            // Send again an hard-kill signal
            XrdProofSessionInfo info(path.c_str());
            XrdProofUI ui;
            XrdProofdAux::GetUserInfo(info.fUser.c_str(), ui);
            XrdProofdAux::KillProcess(pid, 1, ui, fMgr->ChangeOwn());
         } else {
            // Delete the entry
            RmSession(ent->d_name);
         }
      }
   }
   // Close the directory
   closedir(dir);

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::CleanClientSessions(const char *usr, int srvtype)
{
   // Go through the sessions admin path and clean all sessions belonging to 'usr'.
   // Move those not responding in the terminated sessions admin path.
   XPDLOC(SMGR, "ProofServMgr::CleanClientSessions")

   TRACE(REQ, "cleaning "<<usr<<" ...");

   // Check which client
   bool all = (!usr || strlen(usr) <= 0 || !strcmp(usr, "all")) ? 1 : 0;

   // Get user info
   XrdProofUI ui;
   if (!all)
      XrdProofdAux::GetUserInfo(usr, ui);
   XrdOucString path, rest, key, a;

   // We need lock to avoid session actions request while we are doing this
   XrdSysRecMutex *mtx = 0;
   if (all) {
      // Lock us all
      mtx = &fMutex;
   } else {
      // Lock the client
      XrdProofdClient *c = fMgr->ClientMgr()->GetClient(usr);
      if (c) mtx = c->Mutex();
   }

   std::list<int> tobedel;
   {  XrdSysMutexHelper mtxh(mtx);

      // Check the terminated session dir first
      DIR *dir = opendir(fTermAdminPath.c_str());
      if (!dir) {
         TRACE(XERR, "cannot open dir "<<fTermAdminPath<<" ; error: "<<errno);
      } else {
         // Go trough
         struct dirent *ent = 0;
         while ((ent = (struct dirent *)readdir(dir))) {
            // Skip basic entries
            if (!strncmp(ent->d_name, ".", 1) || !strncmp(ent->d_name, "..", 2)) continue;
            // Get the session instance
            int pid = XrdProofdAux::ParsePidPath(ent->d_name, rest, a);
            if (!XPD_LONGOK(pid) || pid <= 0) continue;
            // Read info from file and check that we are interested in this session
            XPDFORM(path, "%s/%s", fTermAdminPath.c_str(), ent->d_name);
            XrdProofSessionInfo info(path.c_str());
            // Check user
            if (!all && info.fUser != usr) continue;
            // Check server type
            if (srvtype != kXPD_AnyServer && info.fSrvType != srvtype) continue;
            // Refresh user info, if needed
            if (all)
               XrdProofdAux::GetUserInfo(info.fUser.c_str(), ui);
            // Check if the process is still alive
            if (XrdProofdAux::VerifyProcessByID(pid) != 0) {
               // Send a hard-kill signal
               XrdProofdAux::KillProcess(pid, 1, ui, fMgr->ChangeOwn());
            } else {
               // Delete the entry
               RmSession(ent->d_name);
            }
         }
         // Close the directory
         closedir(dir);
      }

      // Check the active session dir now
      dir = opendir(fActiAdminPath.c_str());
      if (!dir) {
         TRACE(XERR, "cannot open dir "<<fActiAdminPath<<" ; error: "<<errno);
         return -1;
      }

      // Scan the active sessions admin path
      struct dirent *ent = 0;
      while ((ent = (struct dirent *)readdir(dir))) {
         // Skip basic entries
         if (!strncmp(ent->d_name, ".", 1) || !strncmp(ent->d_name, "..", 2)) continue;
         // Get the session instance
         int pid = XrdProofdAux::ParsePidPath(ent->d_name, rest, a);
         if (a == "status") continue;
         if (!XPD_LONGOK(pid) || pid <= 0) continue;
         // Read info from file and check that we are interested in this session
         XPDFORM(path, "%s/%s", fActiAdminPath.c_str(), ent->d_name);
         XrdProofSessionInfo info(path.c_str());
         if (!all && info.fUser != usr) continue;
         // Check server type
         if (srvtype != kXPD_AnyServer && info.fSrvType != srvtype) continue;
         // Refresh user info, if needed
         if (all)
            XrdProofdAux::GetUserInfo(info.fUser.c_str(), ui);
         // Check if the process is still alive
         if (XrdProofdAux::VerifyProcessByID(pid) != 0) {
            // We will remove this later
            tobedel.push_back(pid);
            // Send a termination signal
            XrdProofdAux::KillProcess(pid, 0, ui, fMgr->ChangeOwn());
         }
         // Flag as terminated
         MvSession(ent->d_name);
      }
      // Close the directory
      closedir(dir);
   }

   // Cleanup fSessions
   std::list<int>::iterator ii = tobedel.begin();
   while (ii != tobedel.end()) {
      XPDFORM(key, "%d", *ii);
      XrdSysMutexHelper mhp(fMutex);
      fSessions.Del(key.c_str());
      ii++;
   }

   // Done
   return 0;
}

//__________________________________________________________________________
void XrdProofdProofServMgr::RegisterDirectives()
{
   // Register directives for configuration

   // Register special config directives
   Register("proofservmgr", new XrdProofdDirective("proofservmgr", this, &DoDirectiveClass));
   Register("putenv", new XrdProofdDirective("putenv", this, &DoDirectiveClass));
   Register("putrc", new XrdProofdDirective("putrc", this, &DoDirectiveClass));
   Register("shutdown", new XrdProofdDirective("shutdown", this, &DoDirectiveClass));
   // Register config directives for ints
   Register("intwait",
                  new XrdProofdDirective("intwait", (void *)&fInternalWait, &DoDirectiveInt));
   Register("reconnto",
                  new XrdProofdDirective("reconnto", (void *)&fReconnectTimeOut, &DoDirectiveInt));
   // Register config directives for strings
   Register("proofplugin",
                  new XrdProofdDirective("proofplugin", (void *)&fProofPlugin, &DoDirectiveString));
   Register("proofservparents",
                  new XrdProofdDirective("proofservparents", (void *)&fParentExecs, &DoDirectiveString));
}

//______________________________________________________________________________
int XrdProofdProofServMgr::DoDirective(XrdProofdDirective *d,
                                       char *val, XrdOucStream *cfg, bool rcf)
{
   // Update the priorities of the active sessions.
   XPDLOC(SMGR, "ProofServMgr::DoDirective")

   if (!d)
      // undefined inputs
      return -1;

   if (d->fName == "proofservmgr") {
      return DoDirectiveProofServMgr(val, cfg, rcf);
   } else if (d->fName == "putenv") {
      return DoDirectivePutEnv(val, cfg, rcf);
   } else if (d->fName == "putrc") {
      return DoDirectivePutRc(val, cfg, rcf);
   } else if (d->fName == "shutdown") {
      return DoDirectiveShutdown(val, cfg, rcf);
   }
   TRACE(XERR,"unknown directive: "<<d->fName);
   return -1;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::DoDirectiveProofServMgr(char *val, XrdOucStream *cfg, bool rcf)
{
   // Process 'proofswrvmgr' directive
   // eg: xpd.proofswrvmgr checkfq:120 termto:100 verifyto:5 recoverto:20
   XPDLOC(SMGR, "ProofServMgr::DoDirectiveProofServMgr")

   if (!val || !cfg)
      // undefined inputs
      return -1;

   if (rcf)
      // Do not reconfigure this (need to check what happens with the cron thread ...
      return 0;

   int checkfq = -1;
   int termto = -1;
   int verifyto = -1;
   int recoverto = -1;
   int checklost = 0;
   int usefork = 0;

   while (val) {
      XrdOucString tok(val);
      if (tok.beginswith("checkfq:")) {
         tok.replace("checkfq:", "");
         checkfq = strtol(tok.c_str(), 0, 10);
      } else if (tok.beginswith("termto:")) {
         tok.replace("termto:", "");
         termto = strtol(tok.c_str(), 0, 10);
      } else if (tok.beginswith("verifyto:")) {
         tok.replace("verifyto:", "");
         verifyto = strtol(tok.c_str(), 0, 10);
      } else if (tok.beginswith("recoverto:")) {
         tok.replace("recoverto:", "");
         recoverto = strtol(tok.c_str(), 0, 10);
      } else if (tok.beginswith("checklost:")) {
         tok.replace("checklost:", "");
         checklost = strtol(tok.c_str(), 0, 10);
      } else if (tok.beginswith("usefork:")) {
         tok.replace("usefork:", "");
         usefork = strtol(tok.c_str(), 0, 10);
      }
      // Get next
      val = cfg->GetWord();
   }

   // Check deprecated 'if' directive
   if (fMgr->Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, fMgr->Host()) == 0)
         return 0;

   // Set the values
   fCheckFrequency = (XPD_LONGOK(checkfq) && checkfq > 0) ? checkfq : fCheckFrequency;
   fTerminationTimeOut = (XPD_LONGOK(termto) && termto > 0) ? termto : fTerminationTimeOut;
   fVerifyTimeOut = (XPD_LONGOK(verifyto) && (verifyto > fCheckFrequency + 1))
                  ? verifyto : fVerifyTimeOut;
   fRecoverTimeOut = (XPD_LONGOK(recoverto) && recoverto > 0) ? recoverto : fRecoverTimeOut;
   if (XPD_LONGOK(checklost)) fCheckLost = (checklost != 0) ? 1 : 0;
   if (XPD_LONGOK(usefork)) fUseFork = (usefork != 0) ? 1 : 0;

   XrdOucString msg;
   XPDFORM(msg, "checkfq: %d s, termto: %d s, verifyto: %d s, recoverto: %d s, checklost: %d, usefork: %d",
            fCheckFrequency, fTerminationTimeOut, fVerifyTimeOut, fRecoverTimeOut, fCheckLost, fUseFork);
   TRACE(ALL, msg);

   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::DoDirectivePutEnv(char *val, XrdOucStream *cfg, bool)
{
   // Process 'putenv' directives

   if (!val)
      // undefined inputs
      return -1;

   // Parse env variables to be passed to 'proofserv':
   XrdOucString users, groups, rcval, rcnam;
   int smi = -1, smx = -1, vmi = -1, vmx = -1; 
   bool hex = 0;
   ExtractEnv(val, cfg, users, groups, rcval, rcnam, smi, smx, vmi, vmx, hex);

   // Adjust name of the variable
   int iequ = rcnam.find('=');
   if (iequ == STR_NPOS) return -1;
   rcnam.erase(iequ);
   
   // Fill entries
   FillEnvList(&fProofServEnvs, rcnam.c_str(), rcval.c_str(),
                                users.c_str(), groups.c_str(), smi, smx, vmi, vmx, hex);

   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::DoDirectivePutRc(char *val, XrdOucStream *cfg, bool)
{
   // Process 'putrc' directives.
   // Syntax:
   //    xpd.putrc  [u:<usr1>,<usr2>,...] [g:<grp1>,<grp2>,...] 
   //               [s:[svnmin][-][svnmax]] [v:[vermin][-][vermax]] RcVarName RcVarValue
   // NB: <usr1>,... and <grp1>,... may contain the wild card '*' 

   if (!val || !cfg)
      // undefined inputs
      return -1;
   
   // Parse rootrc variables to be passed to 'proofserv':
   XrdOucString users, groups, rcval, rcnam;
   int smi = -1, smx = -1, vmi = -1, vmx = -1; 
   bool hex = 0;
   ExtractEnv(val, cfg, users, groups, rcval, rcnam, smi, smx, vmi, vmx, hex);
   
   // Fill entries
   FillEnvList(&fProofServRCs, rcnam.c_str(), rcval.c_str(),
                               users.c_str(), groups.c_str(), smi, smx, vmi, vmx, hex);

   return 0;
}

//______________________________________________________________________________
void XrdProofdProofServMgr::ExtractEnv(char *val, XrdOucStream *cfg,
                                       XrdOucString &users, XrdOucString &groups,
                                       XrdOucString &rcval, XrdOucString &rcnam,
                                       int &smi, int &smx, int &vmi, int &vmx, bool &hex)
{
   // Extract env information from the stream 'cfg'

   XrdOucString ssvn, sver;
   int idash = -1; 
   while (val && val[0]) {
      if (!strncmp(val, "u:", 2)) {
         users = val;
         users.erase(0,2);
      } else if (!strncmp(val, "g:", 2)) {
         groups = val;
         groups.erase(0,2);
      } else if (!strncmp(val, "s:", 2)) {
         ssvn = val;
         ssvn.erase(0,2);
         idash = ssvn.find('-');
         if (idash != STR_NPOS) {
            if (ssvn.isdigit(0, idash-1)) smi = ssvn.atoi(0, idash-1);
            if (ssvn.isdigit(idash+1)) smx = ssvn.atoi(idash+1);
         } else {
            if (ssvn.isdigit()) smi = ssvn.atoi();
         }
      } else if (!strncmp(val, "v:", 2)) {
         sver = val;
         sver.erase(0,2);
         hex = 0;
         if (sver.beginswith('x')) {
            hex = 1;
            sver.erase(0,1);
         }
         idash = sver.find('-');
         if (idash != STR_NPOS) {
            if (sver.isdigit(0, idash-1)) vmi = sver.atoi(0, idash-1);
            if (sver.isdigit(idash+1)) vmx = sver.atoi(idash+1);
         } else {
            if (sver.isdigit()) vmi = sver.atoi();
         }
      } else {
        if (rcval.length() > 0) {
           rcval += ' ';
        } else {
           rcnam = val;
        }
        rcval += val;
      }
      val = cfg->GetWord();
   }
   // Done
   return;
}

//______________________________________________________________________________
void XrdProofdProofServMgr::FillEnvList(std::list<XpdEnv> *el, const char *nam, const char *val,
                                        const char *usrs, const char *grps,
                                        int smi, int smx, int vmi, int vmx, bool hex)
{
   // Fill env entry(ies) in the relevant list
   XPDLOC(SMGR, "ProofServMgr::FillEnvList")

   if (!el) {
      TRACE(ALL, "env list undefined!");
      return;
   }
   
   XrdOucString users(usrs), groups(grps);
   // Transform version numbers in the human unreadable format used internally (version code)
   if (vmi > 0) vmi = XpdEnv::ToVersCode(vmi, hex);
   if (vmx > 0) vmx = XpdEnv::ToVersCode(vmx, hex);
   // Create the entry
   XpdEnv xpe(nam, val, users.c_str(), groups.c_str(), smi, smx, vmi, vmx);
   if (users.length() > 0) {
      XrdOucString usr;
      int from = 0;
      while ((from = users.tokenize(usr, from, ',')) != -1) {
         if (usr.length() > 0) {
            if (groups.length() > 0) {
               XrdOucString grp;
               int fromg = 0;
               while ((fromg = groups.tokenize(grp, from, ',')) != -1) {
                  if (grp.length() > 0) {
                     xpe.Reset(nam, val, usr.c_str(), grp.c_str(), smi, smx, vmi, vmx);
                     el->push_back(xpe);
                  }
               }
            } else {
               xpe.Reset(nam, val, usr.c_str(), 0, smi, smx, vmi, vmx);
               el->push_back(xpe);
            }
         }
      }
   } else {
      if (groups.length() > 0) {
         XrdOucString grp;
         int fromg = 0;
         while ((fromg = groups.tokenize(grp, fromg, ',')) != -1) {
            if (grp.length() > 0) {
               xpe.Reset(nam, val, 0, grp.c_str(), smi, smx, vmi, vmx);
               el->push_back(xpe);
            }
         }
      } else {
         el->push_back(xpe);
      }
   }
   // Done
   return;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::DoDirectiveShutdown(char *val, XrdOucStream *cfg, bool)
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
   if ((val = cfg->GetWord())) {
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
   if (fMgr->Host() && cfg)
      if (XrdProofdAux::CheckIf(cfg, fMgr->Host()) == 0)
         return 0;

   // Set the values
   fShutdownOpt = (opt > -1) ? opt : fShutdownOpt;
   fShutdownDelay = (delay > -1) ? delay : fShutdownDelay;

   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::Process(XrdProofdProtocol *p)
{
   // Process manager request
   XPDLOC(SMGR, "ProofServMgr::Process")

   int rc = 1;
   XPD_SETRESP(p, "Process");

   TRACEP(p, REQ, "enter: req id: " << p->Request()->header.requestid << " (" <<
                XrdProofdAux::ProofRequestTypes(p->Request()->header.requestid) << ")");

   XrdSysMutexHelper mtxh(p->Client()->Mutex());

   // Once logged-in, the user can request the real actions
   XrdOucString emsg("Invalid request code: ");

   int twait = 20;

   if (Pipe()->Post(XrdProofdProofServMgr::kProcessReq, 0) != 0) {
      response->Send(kXR_ServerError,
                     "ProofServMgr::Process: error posting internal pipe for authorization to proceed");
      return 0;
   }
   if (fProcessSem.Wait(twait) != 0) {
      response->Send(kXR_ServerError,
                     "ProofServMgr::Process: timed-out waiting for authorization to proceed - retry later");
      return 0;
   }

   // This is needed to block the session checks
   XpdSrvMgrCreateCnt cnt(this, kProcessCnt);

   switch(p->Request()->header.requestid) {
   case kXP_create:
      return Create(p);
   case kXP_destroy:
      return Destroy(p);
   case kXP_attach:
      return Attach(p);
   case kXP_detach:
      return Detach(p);
   default:
      emsg += p->Request()->header.requestid;
      break;
   }

   // Whatever we have, it's not valid
   response->Send(kXR_InvalidRequest, emsg.c_str());
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::Attach(XrdProofdProtocol *p)
{
   // Handle a request to attach to an existing session
   XPDLOC(SMGR, "ProofServMgr::Attach")

   int psid = -1, rc = 0;
   XPD_SETRESP(p, "Attach");

   // Unmarshall the data
   psid = ntohl(p->Request()->proof.sid);
   TRACEP(p, REQ, "psid: "<<psid<<", CID = "<<p->CID());

   // The client instance must be defined
   XrdProofdClient *c = p->Client();
   if (!c) {
      TRACEP(p, XERR, "client instance undefined");
      response->Send(kXR_ServerError,"client instance undefined");
      return 0;
   }

   // Find server session; sessions maybe recovering, so we need to take
   // that into account
   XrdProofdProofServ *xps = 0;
   int now = time(0);
   int deadline = -1, defdeadline = now + fRecoverTimeOut;
   while ((deadline < 0) || (now < deadline)) {
      if (!(xps = c->GetServer(psid)) || !xps->IsValid()) {
         // If the client is recovering start regular checks
         if (!IsClientRecovering(c->User(), c->Group(), deadline)) {
            // Failure
            TRACEP(p, XERR, "session ID not found: "<<psid);
            response->Send(kXR_InvalidRequest,"session ID not found");
            return 0;
         } else {
            // Make dure we do not enter an infinite loop
            deadline = (deadline > 0) ? deadline : defdeadline;
            // Wait until deadline in 1 sec steps
            sleep(1);
            now++;
         }
      } else {
         // Found
         break;
      }
   }
   // If we deadline we should fail now
   if (!xps || !xps->IsValid()) {
      TRACEP(p, XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"session ID not found");
      return 0;
   }
   TRACEP(p, DBG, "xps: "<<xps<<", status: "<< xps->Status());

   // Stream ID
   unsigned short sid;
   memcpy((void *)&sid, (const void *)&(p->Request()->header.streamid[0]), 2);

   // We associate this instance to the corresponding slot in the
   // session vector of attached clients
   XrdClientID *csid = xps->GetClientID(p->CID());
   csid->SetP(p);
   csid->SetSid(sid);

   // Take parentship, if orphalin
   if (!(xps->Parent()))
      xps->SetParent(csid);

   // Notify to user
   int protvers = (xps && xps->ROOT()) ? xps->ROOT()->SrvProtVers() : -1;
   if (p->ConnType() == kXPD_ClientMaster) {
      // Send also back the data pool url
      XrdOucString dpu = fMgr->PoolURL();
      if (!dpu.endswith('/'))
         dpu += '/';
      dpu += fMgr->NameSpace();
      response->SendI(psid, protvers, (kXR_int16)XPROOFD_VERSBIN,
                           (void *) dpu.c_str(), dpu.length());
   } else
      response->SendI(psid, protvers, (kXR_int16)XPROOFD_VERSBIN);

   // Send saved start processing message, if not idle
   if (xps->Status() == kXPD_running && xps->StartMsg()) {
      TRACEP(p, XERR, "sending start process message ("<<xps->StartMsg()->fSize<<" bytes)");
      response->Send(kXR_attn, kXPD_msg,
                          xps->StartMsg()->fBuff, xps->StartMsg()->fSize);
   }

   // Over
   return 0;
}

//_________________________________________________________________________________
XrdProofdProofServ *XrdProofdProofServMgr::PrepareProofServ(XrdProofdProtocol *p,
                                                            XrdProofdResponse *r,
                                                            unsigned short &sid)
{
   // Allocate and prepare the XrdProofdProofServ object describing this session
   XPDLOC(SMGR, "ProofServMgr::PrepareProofServ")

   // Allocate next free server ID and fill in the basic stuff
   XrdProofdProofServ *xps = p->Client()->GetFreeServObj();
   xps->SetClient(p->Client()->User());
   xps->SetSrvType(p->ConnType());

   // Prepare the stream identifier
   memcpy((void *)&sid, (const void *)&(p->Request()->header.streamid[0]), 2);
   // We associate this instance to the corresponding slot in the
   // session vector of attached clients
   XrdClientID *csid = xps->GetClientID(p->CID());
   csid->SetSid(sid);
   csid->SetP(p);
   // Take parentship, if orphalin
   xps->SetParent(csid);

   // The ROOT version to be used
   xps->SetROOT(p->Client()->ROOT());
   XrdOucString msg;
   XPDFORM(msg, "using ROOT version: %s", xps->ROOT()->Export());
   TRACEP(p, REQ, msg);
   if (p->ConnType() == kXPD_ClientMaster) {
      // Notify the client if using a version different from the default one
      if (p->Client()->ROOT() != fMgr->ROOTMgr()->DefaultVersion()) {
         XPDFORM(msg, "++++ Using NON-default ROOT version: %s ++++\n", xps->ROOT()->Export());
         r->Send(kXR_attn, kXPD_srvmsg, (char *) msg.c_str(), msg.length());
      }
   }

   // Done
   return xps;
}

//_________________________________________________________________________________
void XrdProofdProofServMgr::ParseCreateBuffer(XrdProofdProtocol *p,
                                              XrdProofdProofServ *xps,
                                              XrdOucString &tag, XrdOucString &ord,
                                              XrdOucString &cffile,
                                              XrdOucString &uenvs, int &intwait)
{
   // Extract relevant quantities from the buffer received during a create request
   XPDLOC(SMGR, "ProofServMgr::ParseCreateBuffer")

   // Parse buffer
   char *buf = p->Argp()->buff;
   int   len = p->Request()->proof.dlen;

   // Extract session tag
   tag.assign(buf,0,len-1);

   TRACEP(p, DBG, "received buf: "<<tag);

   tag.erase(tag.find('|'));
   xps->SetTag(tag.c_str());
   TRACEP(p, DBG, "tag: "<<tag);

   // Extract ordinal number
   ord = "0";
   if ((p->ConnType() == kXPD_MasterWorker) || (p->ConnType() == kXPD_MasterMaster)) {
      ord.assign(buf,0,len-1);
      int iord = ord.find("|ord:");
      if (iord != STR_NPOS) {
         ord.erase(0,iord+5);
         ord.erase(ord.find("|"));
      } else
         ord = "0";
   }
   xps->SetOrdinal(ord.c_str());

   // Extract config file, if any (for backward compatibility)
   cffile.assign(buf,0,len-1);
   int icf = cffile.find("|cf:");
   if (icf != STR_NPOS) {
      cffile.erase(0,icf+4);
      cffile.erase(cffile.find("|"));
   } else
      cffile = "";

   // Extract user envs, if any
   uenvs.assign(buf,0,len-1);
   int ienv = uenvs.find("|envs:");
   if (ienv != STR_NPOS) {
      uenvs.erase(0,ienv+6);
      uenvs.erase(uenvs.find("|"));
      xps->SetUserEnvs(uenvs.c_str());
   } else
      uenvs = "";

   // Check if the user wants to wait more for the session startup
   intwait = fInternalWait;
   if (uenvs.length() > 0) {
      TRACEP(p, DBG, "user envs: "<<uenvs);
      int iiw = STR_NPOS;
      if ((iiw = uenvs.find("PROOF_INTWAIT=")) !=  STR_NPOS) {
         XrdOucString s(uenvs, iiw + strlen("PROOF_INTWAIT="));
         s.erase(s.find(','));
         if (s.isdigit()) {
            intwait = s.atoi();
            TRACEP(p, ALL, "startup internal wait set by user to "<<intwait);
         }
      }
   }
}

//_________________________________________________________________________________
int XrdProofdProofServMgr::CreateFork(XrdProofdProtocol *p)
{
   // Handle a request to create a new session
   XPDLOC(SMGR, "ProofServMgr::CreateFork")

   int psid = -1, rc = 0;
   XPD_SETRESP(p, "CreateFork");

   TRACEP(p, DBG, "enter");
   XrdOucString msg;

   XpdSrvMgrCreateGuard mcGuard;

   // Check if we are allowed to start a new session
   int mxsess = fMgr->ProofSched() ? fMgr->ProofSched()->MaxSessions() : -1;
   if (p->ConnType() == kXPD_ClientMaster && mxsess > 0) {
      XrdSysMutexHelper mhp(fMutex);
      int cursess = CurrentSessions();
      TRACEP(p,ALL," cursess: "<<cursess);
      if (mxsess <= cursess) {
         XPDFORM(msg, " ++++ Max number of sessions reached (%d) - please retry later ++++ \n", cursess); 
         response->Send(kXR_attn, kXPD_srvmsg, (char *) msg.c_str(), msg.length());
         response->Send(kXP_TooManySess, "cannot start a new session");
         return 0;
      }
      // If we fail this guarantees that the counters are decreased, if needed 
      mcGuard.Set(&fCurrentSessions);
   }

   // Update counter to control checks during creation
   XpdSrvMgrCreateCnt cnt(this, kCreateCnt);
   if (TRACING(DBG)) {
      int nc = CheckCounter(kCreateCnt);
      TRACEP(p, DBG, nc << " threads are creating a new session");
   }

   // Allocate and prepare the XrdProofdProofServ object describing this session
   unsigned short sid;
   XrdProofdProofServ *xps = PrepareProofServ(p, response, sid);
   psid = xps->ID();

   // Unmarshall log level
   int loglevel = ntohl(p->Request()->proof.int1);

   // Parse buffer
   int intwait;
   XrdOucString tag, ord, cffile, uenvs;
   ParseCreateBuffer(p, xps, tag, ord, cffile, uenvs, intwait); 

   // Notify
   TRACEP(p, DBG, "{ord,cfg,psid,cid,log}: {"<<ord<<","<<cffile<<","<<psid
                                             <<","<<p->CID()<<","<<loglevel<<"}");

   // Here we fork: for some weird problem on SMP machines there is a
   // non-zero probability for a deadlock situation in system mutexes.
   // The semaphore seems to have solved the problem.
   if (fForkSem.Wait(10) != 0) {
      xps->Reset();
      // Timeout acquire fork semaphore
      response->Send(kXP_ServerError, "timed-out acquiring fork semaphore");
      return 0;
   }

   // Pipe for child-to-parent communications during setup
   XrdProofdPipe fpc, fcp;
   if (!(fpc.IsValid()) || !(fcp.IsValid())) {
      xps->Reset();
      // Failure creating pipe
      response->Send(kXP_ServerError,
                     "unable to create pipes for communication during setup");
      return 0;
   }

   // Start setting up the unique tag and relevant dirs for this session
   ProofServEnv_t in = {xps, loglevel, cffile.c_str(), "", "", "", "", "", 1};
   GetTagDirs(0, p, xps, in.fSessionTag, in.fTopSessionTag, in.fSessionDir, in.fWrkDir);

   // Fork an agent process to handle this session
   int pid = -1;
   TRACEP(p, FORK,"Forking external proofsrv");
   if (!(pid = fMgr->Sched()->Fork("proofsrv"))) {

      // Finalize unique tag and relevant dirs for this session and create log file path
      GetTagDirs((int)getpid(),
                 p, xps, in.fSessionTag, in.fTopSessionTag, in.fSessionDir, in.fWrkDir);
      XPDFORM(in.fLogFile, "%s.log", in.fWrkDir.c_str());

      // Log to the session log file from now on
      if (fLogger) fLogger->Bind(in.fLogFile.c_str());
      TRACE(FORK, "log file: "<<in.fLogFile);

      XrdOucString pmsg = "child process ";
      pmsg += (int) getpid();
      TRACE(FORK, pmsg);

      // These files belongs to the client
      if (chown(in.fLogFile.c_str(), p->Client()->UI().fUid, p->Client()->UI().fGid) != 0)
         TRACE(XERR, "chown on '"<<in.fLogFile.c_str()<<"'; errno: "<<errno);

      XpdMsg xmsg;
      XrdOucString path, sockpath, emsg;

      // Receive the admin path from the parent
      if (fpc.Poll() < 0) {
         TRACE(XERR, "error while polling to receive the admin path from parent - EXIT" );
         exit(1);
      }
      if (fpc.Recv(xmsg) != 0) {
         TRACE(XERR, "error reading message while waiting for the admin path from parent - EXIT" );
         exit(1);
      }
      if (xmsg.Type() < 0) {
         TRACE(XERR, "the parent failed to setup the admin path - EXIT" );
         exit(1);
      }
      // Set the path w/o asserting the related files
      path = xmsg.Buf();
      xps->SetAdminPath(path.c_str(), 0);
      TRACE(FORK, "admin path: "<<path);

      xmsg.Reset();
      // Receive the sock path from the parent
      if (fpc.Poll() < 0) {
         TRACE(XERR, "error while polling to receive the sock path from parent - EXIT" );
         exit(1);
      }
      if (fpc.Recv(xmsg) != 0) {
         TRACE(XERR, "error reading message while waiting for the sock path from parent - EXIT" );
         exit(1);
      }
      if (xmsg.Type() < 0) {
         TRACE(XERR, "the parent failed to setup the sock path - EXIT" );
         exit(1);
      }
      // Set the UNIX sock path
      sockpath = xmsg.Buf();
      xps->SetUNIXSockPath(sockpath.c_str());
      TRACE(FORK, "UNIX sock path: "<<sockpath);

      // We set to the user ownerships and create relevant dirs
      bool asserdatadir = 1;
      int srvtype = xps->SrvType();
      TRACE(ALL,"srvtype = "<< srvtype);
      if (xps->SrvType() != kXPD_Worker && !strchr(fMgr->DataDirOpts(), 'M')) {
         asserdatadir = 0;
      } else if (xps->SrvType() == kXPD_Worker && !strchr(fMgr->DataDirOpts(), 'W')) {
         asserdatadir = 0;
      }
      const char *pord = asserdatadir ? ord.c_str() : 0;
      const char *ptag = asserdatadir ? in.fSessionTag.c_str() : 0;
      if (SetUserOwnerships(p, pord, ptag) != 0) {
         emsg = "SetUserOwnerships did not return OK - EXIT";
         TRACE(XERR, emsg);
         if (fcp.Post(0, emsg.c_str()) != 0)
            TRACE(XERR, "cannot write to internal pipe; errno: "<<errno);
         exit(1);
      }

      // We set to the user environment
      if (SetUserEnvironment(p) != 0) {
         emsg = "SetUserEnvironment did not return OK - EXIT";
         TRACE(XERR, emsg);
         if (fcp.Post(0, emsg.c_str()) != 0)
            TRACE(XERR, "cannot write to internal pipe; errno: "<<errno);
         exit(1);
      }

      char *argvv[6] = {0};

      char *sxpd = 0;
      if (fMgr && fMgr->AdminPath()) {
         // We add our admin path to be able to identify processes coming from us
         sxpd = new char[strlen(fMgr->AdminPath()) + strlen("xpdpath:") + 1];
         sprintf(sxpd, "xpdpath:%s", fMgr->AdminPath());
      } else {
         // We add our PID to be able to identify processes coming from us
         sxpd = new char[10];
         sprintf(sxpd, "%d", getppid());
      }

      // Log level
      char slog[10] = {0};
      sprintf(slog, "%d", loglevel);

      // start server
      argvv[0] = (char *) xps->ROOT()->PrgmSrv();
      argvv[1] = (char *)((p->ConnType() == kXPD_MasterWorker) ? "proofslave"
                       : "proofserv");
      argvv[2] = (char *)"xpd";
      argvv[3] = (char *)sxpd;
      argvv[4] = (char *)slog;
      argvv[5] = 0;

      // Set environment for proofserv
      if (SetProofServEnv(p, (void *)&in) != 0) {
         emsg = "SetProofServEnv did not return OK - EXIT";
         TRACE(XERR, emsg);
         if (fcp.Post(0, emsg.c_str()) != 0)
            TRACE(XERR, "cannot write to internal pipe; errno: "<<errno);
         exit(1);
      }
      TRACE(FORK, (int)getpid() << ": proofserv env set up");
      
      // Setup OK: now we go
      // Communicate the logfile path
      if (fcp.Post(1, xps->Fileout()) != 0) {
         TRACE(XERR, "cannot write log file path to internal pipe; errno: "<<errno);
         exit(1);
      }
      TRACE(FORK, (int)getpid()<< ": log file path communicated");

      // Unblock SIGUSR1 and SIGUSR2
      sigset_t myset;
      sigemptyset(&myset);
      sigaddset(&myset, SIGUSR1);
      sigaddset(&myset, SIGUSR2);
      pthread_sigmask(SIG_UNBLOCK, &myset, 0);

      // Close pipes
      fpc.Close();
      fcp.Close();

      TRACE(FORK, (int)getpid()<<": user: "<<p->Client()->User()<<
                  ", uid: "<<getuid()<<", euid:"<<geteuid()<<", psrv: "<<xps->ROOT()->PrgmSrv());
      // Run the program
      execv(xps->ROOT()->PrgmSrv(), argvv);

      // We should not be here!!!
      TRACE(XERR, "returned from execv: bad, bad sign !!! errno:" << (int)errno);
      exit(1);
   }

   // Wakeup colleagues
   fForkSem.Post();

   // parent process
   if (pid < 0) {
      xps->Reset();
      // Failure in forking
      response->Send(kXP_ServerError, "could not fork agent");
      return 0;
   }

   TRACEP(p, FORK,"Parent process: child is "<<pid);
   XrdOucString emsg;

   // Finalize unique tag and relevant dirs for this session and create log file path
   GetTagDirs((int)pid, p, xps, in.fSessionTag, in.fTopSessionTag, in.fSessionDir, in.fWrkDir);
   XPDFORM(in.fLogFile, "%s.log", in.fWrkDir.c_str());
   TRACEP(p, FORK, "log file: "<<in.fLogFile);

   // Log prefix
   XrdOucString npfx;
   XPDFORM(npfx, "%s-%s:", (p->ConnType() == kXPD_MasterWorker) ? "wrk" : "mst", xps->Ordinal());
   
   // Cleanup current socket, if any
   if (xps->UNIXSock()) {
      TRACEP(p, FORK,"current UNIX sock: "<<xps->UNIXSock() <<", path: "<<xps->UNIXSockPath());
      xps->DeleteUNIXSock();
   }

   // Admin and UNIX Socket Path (set path and create the socket); we need to
   // set and create them in here, otherwise the cleaning may remove the socket
   XrdOucString path, sockpath;
   XPDFORM(path, "%s/%s.%s.%d", fActiAdminPath.c_str(),
                                p->Client()->User(), p->Client()->Group(), pid);
   // Sock path under dedicated directory to avoid problems related to its length
   XPDFORM(sockpath, "%s/xpd.%d.%d", fMgr->SockPathDir(), fMgr->Port(), pid);
   struct sockaddr_un unserver;
   if (sockpath.length() > (int)(sizeof(unserver.sun_path) - 1)) {
      emsg = "socket path very long (";
      emsg += sockpath.length();
      emsg += "): this may lead to stack corruption!";
      emsg += " Use xpd.sockpathdir to change it";
      TRACEP(p, XERR, emsg.c_str());
   }
   int pathrc = 0;
   if (!pathrc && !(pathrc = xps->SetAdminPath(path.c_str(), 1))) {
      // Communicate the path to child
      if ((pathrc = fpc.Post(0, path.c_str())) != 0) {
         emsg = "failed to communicating path to child";
         XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
         TRACEP(p, XERR, emsg.c_str());
      }
   } else {
      emsg = "failed to setup child admin path";
      // Communicate failure to child
      if ((pathrc = fpc.Post(-1, path.c_str())) != 0) {
         emsg += ": failed communicating failure to child";
         XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
         TRACEP(p, XERR, emsg.c_str());
      }
   }
   // Now create the UNIX sock path
   if (!pathrc) {
      xps->SetUNIXSockPath(sockpath.c_str());
      if ((pathrc = xps->CreateUNIXSock(fEDest)) != 0) {
         // Failure
         emsg = "failure creating UNIX socket on " ;
         emsg += sockpath;
         XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
         TRACEP(p, XERR, emsg.c_str());
      }
   }
   if (!pathrc) {
      TRACEP(p, FORK,"UNIX sock: "<<xps->UNIXSockPath());
      if ((pathrc = chown(sockpath.c_str(), p->Client()->UI().fUid, p->Client()->UI().fGid)) != 0) {
         emsg = "failure changing ownership of the UNIX socket on " ;
         emsg += sockpath;
         emsg += "; errno: " ;
         emsg += errno;
         XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
         TRACEP(p, XERR, emsg.c_str());
      }
   }
   // Communicate sockpath or failure, if any 
   if (!pathrc) {
      // Communicate the path to child
      if ((pathrc = fpc.Post(0, sockpath.c_str())) != 0) {
         emsg = "failed to communicating path to child";
         XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
         TRACEP(p, XERR, emsg.c_str());
      }
   } else {
      emsg = "failed to setup child admin path";
      // Communicate failure to child
      if ((pathrc = fpc.Post(-1, sockpath.c_str())) != 0) {
         emsg += ": failed communicating failure to child";
         XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
         TRACEP(p, XERR, emsg.c_str());
      }
   }
   
   if (pathrc != 0) {
      // Failure
      xps->Reset();
      XrdProofdAux::KillProcess(pid, 1, p->Client()->UI(), fMgr->ChangeOwn());
      // Make sure that the log file path reaches the caller
      emsg += "|log:";
      emsg += in.fLogFile;
      emsg.insert(npfx, 0);
      response->Send(kXP_ServerError, emsg.c_str());
      return 0;
   }

   TRACEP(p, FORK, "waiting for client setup status ...");

   emsg = "proofserv setup";
   // Wait for the setup process on the pipe, 20 secs max (10 x 2000 millisecs): this
   // is enough to cover possible delays due to heavy load; the client will anyhow
   // retry a few times
   int ntry = 10, prc = 0, rst = -1;
   while (prc == 0 && ntry--) {
      // Poll for 2 secs
      if ((prc = fcp.Poll(2)) > 0) {
         // Got something: read the message out
         XpdMsg xmsg;
         if (fcp.Recv(xmsg) != 0) {
            emsg = "error receiving message from pipe";
            XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
            TRACEP(p, XERR, emsg.c_str());
            prc = -1;
            break;
         }
         // Status is the message type
         rst = xmsg.Type();
         // Read string, if any
         XrdOucString xbuf = xmsg.Buf();
         if (xbuf.length() <= 0) {
            emsg = "error reading buffer {logfile, error message} from message received on the pipe";
            XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
            TRACEP(p, XERR, emsg.c_str());
            prc = -1;
            break;
         }
         if (rst > 0) {
            // Set the log file
            xps->SetFileout(xbuf.c_str());
            // Set also the session tag
            XrdOucString stag(xbuf);
            stag.erase(stag.rfind('/'));
            stag.erase(0, stag.find("session-") + strlen("session-"));
            xps->SetTag(stag.c_str());

         } else {
            // Setup failed: save the error
            prc = -1;
            emsg = "failed: ";
            emsg += xbuf;
            XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
            TRACEP(p, XERR, emsg.c_str());
            break;
         }

      } else if (prc < 0) {
         emsg = "error receive status-of-setup from pipe";
         XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
         TRACEP(p, XERR, emsg.c_str());
         break;
      } else {
         TRACEP(p, FORK, "receiving status-of-setup from pipe: waiting 2 s ..."<<pid);
      }
   }

   // Close pipes
   fpc.Close();
   fcp.Close();

   // Notify the user
   if (prc <= 0) {
      // Timed-out or failed: we are done; if timed-out finalize the notification message
      emsg = "failure setting up proofserv" ;
      if (prc == 0) emsg += ": timed-out receiving status-of-setup from pipe";
      // Dump to the log file
      XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
      // Recycle the session object
      xps->Reset();
      XrdProofdAux::KillProcess(pid, 1, p->Client()->UI(), fMgr->ChangeOwn());
      // Make sure that the log file path reaches the caller
      emsg += "|log:";
      emsg += in.fLogFile;
      TRACEP(p, XERR, emsg.c_str());
      emsg.insert(npfx, 0);
      response->Send(kXP_ServerError, emsg.c_str());
      return 0;

   } else {
      // Setup was successful
      XrdOucString info;
      if (p->ConnType() == kXPD_ClientMaster) {
         // Send also back the data pool url
         info = fMgr->PoolURL();
         if (!info.endswith('/')) info += '/';
         info += fMgr->NameSpace();
      }
      // The log file path (so we do it independently of a successful session startup)
      info += "|log:";
      info += xps->Fileout();
      // Send it back
      response->SendI(psid, xps->ROOT()->SrvProtVers(), (kXR_int16)XPROOFD_VERSBIN,
                            (void *) info.c_str(), info.length());
   }

   // now we wait for the callback to be (successfully) established
   TRACEP(p, FORK, "server launched: wait for callback ");

   // Set ID
   xps->SetSrvPID(pid);

   // Wait for the call back
   if (AcceptPeer(xps, intwait, emsg) != 0) {
      emsg = "problems accepting callback: ";
      // Failure: kill the child process
      if (XrdProofdAux::KillProcess(pid, 0, p->Client()->UI(), fMgr->ChangeOwn()) != 0)
         emsg += "process could not be killed - pid: ";
      else
         emsg += "process killed - pid: ";
      emsg += (int)pid;
      // Dump to the log file
      XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
      // Reset the instance
      xps->Reset();
      // Notify
      TRACEP(p, XERR, emsg.c_str());
      emsg.insert(npfx, 0);
      response->Send(kXR_attn, kXPD_errmsg, (char *) emsg.c_str(), emsg.length());
      return 0;
   }
   // Set the group, if any
   xps->SetGroup(p->Client()->Group());

   // Change child process priority, if required
   int dp = 0;
   if (fMgr->PriorityMgr()->SetProcessPriority(xps->SrvPID(),
                                                        p->Client()->User(), dp) != 0) {
      TRACEP(p, XERR, "problems changing child process priority");
   } else if (dp > 0) {
      TRACEP(p, DBG, "priority of the child process changed by " << dp << " units");
   }

   XrdClientID *cid = xps->Parent();
   TRACEP(p, FORK, "xps: "<<xps<<", ClientID: "<<(int *)cid<<" (sid: "<<sid<<")"<<" NClients: "<<xps->GetNClients(1));

   // Record this session in the client sandbox
   if (p->Client()->Sandbox()->AddSession(xps->Tag()) == -1)
      TRACEP(p, REQ, "problems recording session in sandbox");

   // Success; avoid that the global counter is decreased
   mcGuard.Set(0);

   // Update the global session handlers
   XrdOucString key; key += pid;
   {  XrdSysMutexHelper mh(fMutex);
      fSessions.Add(key.c_str(), xps, 0, Hash_keepdata);
      fActiveSessions.push_back(xps);
   }
   AddSession(p, xps);

   // Check session validity
   if (!xps->IsValid()) {
      // Notify
      TRACEP(p, XERR, "PROOF session is invalid: protocol error? " <<emsg);
   }

   // Over
   return 0;
}
//_________________________________________________________________________________
int XrdProofdProofServMgr::CreateAdminPath(XrdProofdProofServ *xps,
                                           XrdProofdProtocol *p, int pid,
                                           XrdOucString &emsg)
{
   // Create the admin path for the starting session 
   // Return 0 on success, -1 on error (error message in 'emsg')

   XrdOucString path;
   bool assert = (pid > 0) ? 1 : 0;
   XPDFORM(path, "%s/%s.%s.", fActiAdminPath.c_str(),
                              p->Client()->User(), p->Client()->Group());
   if (pid > 0) path += pid;
   if (xps->SetAdminPath(path.c_str(), assert) != 0) {
      XPDFORM(emsg, "failure setting admin path '%s'", path.c_str());
      return -1;
   }
   // Done
   return 0;
}
 
//_________________________________________________________________________________
int XrdProofdProofServMgr::CreateSockPath(XrdProofdProofServ *xps,
                                          XrdProofdProtocol *p,
                                          unsigned int seq, XrdOucString &emsg)
{
   // Create the socket path for the starting session 
   // Return 0 on success, -1 on error (error message in 'emsg')
   XPDLOC(SMGR, "ProofServMgr::CreateSockPath")

   XrdOucString sockpath;
   // Sock path under dedicated directory to avoid problems related to its length
   XPDFORM(sockpath, "%s/xpd.%d.%d.%u", fMgr->SockPathDir(), fMgr->Port(), getpid(), seq);
   TRACEP(p, ALL, "socket path: " << sockpath);
   struct sockaddr_un unserver;
   if (sockpath.length() > (int)(sizeof(unserver.sun_path) - 1)) {
      XPDFORM(emsg, "socket path very long (%d): this may lead to stack corruption! ", sockpath.length());
      return -1;
   }
   // Now create the UNIX sock path and set its permissions
   xps->SetUNIXSockPath(sockpath.c_str());
   if (xps->CreateUNIXSock(fEDest) != 0) {
      // Failure
      XPDFORM(emsg, "failure creating UNIX socket on '%s'", sockpath.c_str());
      return -1;
   }
   if (chmod(sockpath.c_str(), 0755) != 0) {
      XPDFORM(emsg, "failure changing permissions of the UNIX socket on '%s'; errno: %d",
                    sockpath.c_str(), (int)errno);
      return -1;
   }

   // Done
   return 0;
}

//_________________________________________________________________________________
int XrdProofdProofServMgr::Create(XrdProofdProtocol *p)
{
   // Handle a request to create a new session
   XPDLOC(SMGR, "ProofServMgr::Create")

   int psid = -1, rc = 0;
   XPD_SETRESP(p, "Create");

   // Check if we have to use the version base on Fork
   bool forksess = (p->Client()->ROOT()->SrvProtVers() <= 32) ? 1 : 0;
   if (fUseFork || forksess) {
      if (!fUseFork || forksess) {
         TRACEP(p, ALL, "PROOF version requires use of fork(): calling CreateFork()");
      } else {
         TRACEP(p, ALL, "use of fork() enforced: calling CreateFork()");
      }
      return CreateFork(p);
   }
   
   TRACEP(p, DBG, "enter");
   XrdOucString msg;

   XpdSrvMgrCreateGuard mcGuard;

   // Check if we are allowed to start a new session
   int mxsess = fMgr->ProofSched() ? fMgr->ProofSched()->MaxSessions() : -1;
   if (p->ConnType() == kXPD_ClientMaster && mxsess > 0) {
      XrdSysMutexHelper mhp(fMutex);
      int cursess = CurrentSessions();
      TRACEP(p,ALL," cursess: "<<cursess);
      if (mxsess <= cursess) {
         XPDFORM(msg, " ++++ Max number of sessions reached (%d) - please retry later ++++ \n", cursess); 
         response->Send(kXR_attn, kXPD_srvmsg, (char *) msg.c_str(), msg.length());
         response->Send(kXP_TooManySess, "cannot start a new session");
         return 0;
      }
      // If we fail this guarantees that the counters are decreased, if needed 
      mcGuard.Set(&fCurrentSessions);
   }

   // Update counter to control checks during creation
   XpdSrvMgrCreateCnt cnt(this, kCreateCnt);
   if (TRACING(DBG)) {
      int nc = CheckCounter(kCreateCnt);
      TRACEP(p, DBG, nc << " threads are creating a new session");
   }

   // Allocate and prepare the XrdProofdProofServ object describing this session
   unsigned short sid;
   XrdProofdProofServ *xps = PrepareProofServ(p, response, sid);
   psid = xps->ID();

   // Unmarshall log level
   int loglevel = ntohl(p->Request()->proof.int1);

   // Parse buffer
   int intwait;
   XrdOucString tag, ord, cffile, uenvs;
   ParseCreateBuffer(p, xps, tag, ord, cffile, uenvs, intwait); 

   // Notify
   TRACEP(p, ALL, "{ord,cfg,psid,cid,log}: {"<<ord<<","<<cffile<<","<<psid
                                             <<","<<p->CID()<<","<<loglevel<<"}");

   // Start setting up the unique tag and relevant dirs for this session
   ProofServEnv_t in = {xps, loglevel, cffile.c_str(), "", "", "", "", "", 0};
   GetTagDirs(0, p, xps, in.fSessionTag, in.fTopSessionTag, in.fSessionDir, in.fWrkDir);

   XrdOucString emsg;

   // The sequential number for this session
   int seq = GetSeqSessionN();

   // Admin and UNIX Socket Path (set path and create the socket); we need to
   // set and create them in here, so that the starting process can use them
   if (CreateSockPath(xps, p, seq, emsg) != 0) {
      // Failure
      TRACEP(p, XERR, emsg.c_str());
      response->Send(kXP_ServerError, emsg.c_str());
      return 0;
   }
   // The partial admin path (needed by the launcher to be finalized inside ...)
   if (CreateAdminPath(xps, p, 0, emsg) != 0) {
      // Failure
      TRACEP(p, XERR, emsg.c_str());
      response->Send(kXP_ServerError, emsg.c_str());
      return 0;
   }

   // Create the RC-file and env-file paths for this session: for masters this will be temporary
   // located in the sandbox
   XrdOucString rcfile, envfile;
   if (p->ConnType() == kXPD_ClientMaster) {
      XPDFORM(rcfile, "%s.rootrc", in.fSessionDir.c_str());
      XPDFORM(envfile, "%s.env", in.fSessionDir.c_str());
   } else {
      const char *ndtype = (p->ConnType() == kXPD_MasterWorker) ? "worker" : "master";
      XPDFORM(rcfile, "%s/%s-%s-%s.rootrc", in.fSessionDir.c_str(), ndtype, xps->Ordinal(), in.fSessionTag.c_str());
      XPDFORM(envfile, "%s/%s-%s-%s.env", in.fSessionDir.c_str(), ndtype, xps->Ordinal(),  in.fSessionTag.c_str());
   }

   TRACE(ALL, "RC: "<< rcfile); 
   TRACE(ALL, "env: "<< envfile); 
   // Create the RC-file ...
   if (CreateProofServRootRc(p, &in, rcfile.c_str()) != 0) {
      // Failure: reset the instance
      xps->Reset();
      // Make sure that the log file path reaches the caller
      XPDFORM(emsg, "Problems creating RC-file '%s'", rcfile.c_str());
      TRACEP(p, XERR, emsg.c_str());
      response->Send(kXP_ServerError, emsg.c_str());
      return 0;
   }
   // Create the env-file ...
   if (CreateProofServEnvFile(p, &in, envfile.c_str(), rcfile.c_str()) != 0) {
      // Failure: reset the instance
      xps->Reset();
      // Remove the RC-file
      unlink(rcfile.c_str());
      // Make sure that the log file path reaches the caller
      XPDFORM(emsg, "Problems creating env-file '%s'", envfile.c_str());
      response->Send(kXP_ServerError, emsg.c_str());
      return 0;
   }

   // Session dir, startup errors file and log prefix
   XrdOucString sessdir, errlog, npfx;
   if (p->ConnType() == kXPD_MasterWorker) {
      XPDFORM(npfx, "wrk-%s:", xps->Ordinal());
      XPDFORM(sessdir, "%s/worker-%s-%s<pid>",
                        in.fSessionDir.c_str(), xps->Ordinal(), in.fSessionTag.c_str());
      XPDFORM(errlog, "%s-worker-%s-%s.errlog",
                        in.fSessionDir.c_str(), xps->Ordinal(), in.fSessionTag.c_str());
   } else {
      XPDFORM(npfx, "mst-%s:", xps->Ordinal());
      if (p->ConnType() == kXPD_MasterMaster) {
         XPDFORM(sessdir, "%s/master-%s-%s<pid>",
                           in.fSessionDir.c_str(), xps->Ordinal(), in.fSessionTag.c_str());
         XPDFORM(errlog, "%s/master-%s-%s.errlog",
                           in.fSessionDir.c_str(), xps->Ordinal(), in.fSessionTag.c_str());
      } else {
         XPDFORM(sessdir, "%s<pid>/master-%s-%s<pid>",
                           in.fSessionDir.c_str(), xps->Ordinal(), in.fSessionTag.c_str());
         XPDFORM(errlog, "%s.errlog", in.fSessionDir.c_str());
      }
   }
   TRACE(ALL, "sessdir: "<< sessdir); 
   TRACE(ALL, "errlog: "<< errlog); 
   
   // Launch the proofserv
   XrdNetPeer *peersrv = 0;
   int pid = -1;
   int launchrc = 0;
   ProofdLaunch_t launch_in = {fMgr, xps, loglevel, envfile, sessdir, errlog, intwait, 0};
   if (!(peersrv = p->Client()->Launcher()->Launch(&launch_in, pid))) {
      // Failure in creating proofserv
      emsg = "could not create proofserv";
      launchrc = -1;
   } else {
      // Launch was successful: proceed to setup the session ...
      TRACEP(p, FORK,"Parent process: child is "<<pid);
      // The admin path
      if (CreateAdminPath(xps, p, pid, emsg) != 0) {
         // Failure
         launchrc = -1;
      } else {
         TRACEP(p, ALL, "admin path: " << xps->AdminPath());
         // Finalize unique tag and relevant dirs for this session and create log file path
         GetTagDirs((int)pid, p, xps, in.fSessionTag, in.fTopSessionTag, in.fSessionDir, in.fWrkDir);
         XPDFORM(in.fLogFile, "%s.log", in.fWrkDir.c_str());
         TRACEP(p, FORK, "log file: "<<in.fLogFile);

         // Create or Update symlink to last session
         TRACE(DBG, "creating symlink");
         XrdOucString syml = p->Client()->Sandbox()->Dir();
         if (p->ConnType() == kXPD_MasterWorker)
            syml += "/last-worker-session";
         else
            syml += "/last-master-session";
         if (XrdProofdAux::SymLink(in.fSessionDir.c_str(), syml.c_str()) != 0) {
            XPDFORM(emsg, "problems creating symlink to last session (errno: %d)", errno);
            launchrc = -1;
         }
      }
   }
   if (launchrc == -1) {
      // Failure in creating proofserv
      xps->Reset();
      // Dump to the log file
      TRACEP(p, XERR, emsg.c_str());
      XrdProofdAux::LogEmsgToFile(errlog.c_str(), emsg.c_str(), npfx.c_str());
      emsg += "|log:";
      emsg += errlog;
      emsg.insert(npfx, 0);
      response->Send(kXP_ServerError, emsg.c_str());
      return 0;
   }

   // Setup was successful
   XrdOucString info;
   if (p->ConnType() == kXPD_ClientMaster) {
      // Send also back the data pool url
      info = fMgr->PoolURL();
      if (!info.endswith('/')) info += '/';
      info += fMgr->NameSpace();
   }
   // The log file path (so we do it independently of a successful session startup)
   info += "|log:";
   info += in.fLogFile;
   // Send it back
   response->SendI(psid, xps->ROOT()->SrvProtVers(), (kXR_int16)XPROOFD_VERSBIN,
                           (void *) info.c_str(), info.length());
   // Set ID
   xps->SetSrvPID(pid);

   // Now we establish the connection ...
   TRACEP(p, FORK, "server launched: wait for setup ...");

   // Setup the protocol serving this peer
   if (SetupProtocol(*peersrv, xps, emsg) != 0) {
      // Send content of errlog quick access to error information
      SendErrLog(errlog.c_str(), response);
      emsg += "could not assert connected peer: ";
      // Failure: kill the child process
      if (XrdProofdAux::KillProcess(pid, 0, p->Client()->UI(), fMgr->ChangeOwn()) != 0)
         emsg += "process could not be killed - pid: ";
      else
         emsg += "process killed - pid: ";
      emsg += (int)pid;
      // Dump to the log file
      XrdProofdAux::LogEmsgToFile(in.fLogFile.c_str(), emsg.c_str(), npfx.c_str());
      // Reset the instance
      xps->Reset();
      // Notify
      TRACEP(p, XERR, emsg.c_str());
      emsg.insert(npfx, 0);
      response->Send(kXR_attn, kXPD_errmsg, (char *) emsg.c_str(), emsg.length());
      return 0;
   }
   SafeDelete(peersrv);
   // Set the group, if any
   xps->SetGroup(p->Client()->Group());

   // Change child process priority, if required
   int dp = 0;
   if (fMgr->PriorityMgr()->SetProcessPriority(xps->SrvPID(),
                                                        p->Client()->User(), dp) != 0) {
      TRACEP(p, XERR, "problems changing child process priority");
   } else if (dp > 0) {
      TRACEP(p, DBG, "priority of the child process changed by " << dp << " units");
   }

   XrdClientID *cid = xps->Parent();
   TRACEP(p, FORK, "xps: "<<xps<<", ClientID: "<<(int *)cid<<" (sid: "<<sid<<")"<<" NClients: "<<xps->GetNClients(1));

   // Record this session in the client sandbox
   if (p->Client()->Sandbox()->AddSession(xps->Tag()) == -1)
      TRACEP(p, REQ, "problems recording session in sandbox");

   // Success; avoid that the global counter is decreased
   mcGuard.Set(0);

   // Update the global session handlers
   XrdOucString key; key += pid;
   {  XrdSysMutexHelper mh(fMutex);
      fSessions.Add(key.c_str(), xps, 0, Hash_keepdata);
      fActiveSessions.push_back(xps);
   }
   AddSession(p, xps);

   // Check session validity
   if (!xps->IsValid()) {
      // Notify
      TRACEP(p, XERR, "PROOF session is invalid: protocol error? " <<emsg);
   } else {
      // We can remove the statup error log
      if (unlink(errlog.c_str()) != 0)
         TRACEP(p, XERR, "problem unlinking "<<errlog<<" - errno: "<<errno);
   }

   // Over
   return 0;
}

//_________________________________________________________________________________
void XrdProofdProofServMgr::SendErrLog(const char *errlog, XrdProofdResponse *r)
{
   // Send content of errlog upstream asynchronously
   XPDLOC(SMGR, "ProofServMgr::SendErrLog")

   XrdOucString emsg("An error occured: the content of errlog follows:");
   r->Send(kXR_attn, kXPD_srvmsg, (char *) emsg.c_str(), emsg.length());
   emsg = "------------------------------------------------\n";
   r->Send(kXR_attn, kXPD_srvmsg, 2, (char *) emsg.c_str(), emsg.length());

   int ierr = open(errlog, O_RDONLY);
   if (ierr < 0) {
      XPDFORM(emsg, "cannot open '%s' (errno: %d)", errlog, errno);
      r->Send(kXR_attn, kXPD_srvmsg, 2, (char *) emsg.c_str(), emsg.length());
      return;
   }
   struct stat st;
   if (fstat(ierr, &st) != 0) {
      XPDFORM(emsg, "cannot stat '%s' (errno: %d)", errlog, errno);
      r->Send(kXR_attn, kXPD_srvmsg, 2, (char *) emsg.c_str(), emsg.length());
      close (ierr);
      return;
   }
   off_t len = st.st_size;
   TRACE(ALL, " reading "<<len<<" bytes from "<<errlog);
   ssize_t chunk = 2048, nb, nr;
   char buf[2048];
   ssize_t left = len;
   while (left > 0) {
      nb = (left > chunk) ? chunk : left;
      if ((nr = read(ierr, buf, nb)) < 0) {
         XPDFORM(emsg, "problems reading from '%s' (errno: %d)", errlog, errno);
         r->Send(kXR_attn, kXPD_srvmsg, 2, (char *) emsg.c_str(), emsg.length());
         close(ierr);
         return;
      }
      TRACE(ALL, buf);
      r->Send(kXR_attn, kXPD_srvmsg, 2, buf, nr);
      left -= nr;
   }
   close(ierr);
   emsg = "------------------------------------------------";
   r->Send(kXR_attn, kXPD_srvmsg, 2, (char *) emsg.c_str(), emsg.length());
      
   // Done
   return;
}

//_________________________________________________________________________________
int XrdProofdProofServMgr::ResolveSession(const char *fpid)
{
   // Handle a request to recover a session after stop&restart
   XPDLOC(SMGR, "ProofServMgr::ResolveSession")

   TRACE(REQ,  "resolving "<< fpid<<" ...");

   // Check inputs
   if (!fpid || strlen(fpid)<= 0 || !(fMgr->ClientMgr()) || !fRecoverClients) {
      TRACE(XERR, "invalid inputs: "<<fpid<<", "<<fMgr->ClientMgr()<<
                  ", "<<fRecoverClients);
      return -1;
   }

   // Path to the session file
   XrdOucString path;
   XPDFORM(path, "%s/%s", fActiAdminPath.c_str(), fpid);

   // Read info
   XrdProofSessionInfo si(path.c_str());

   // Check if recovering is supported
   if (si.fSrvProtVers < 18) {
      TRACE(DBG, "session does not support recovering: protocol "
                 <<si.fSrvProtVers<<" < 18");
      return -1;
   }

   // Create client instance
   XrdProofdClient *c = fMgr->ClientMgr()->GetClient(si.fUser.c_str(), si.fGroup.c_str(),
                                                     si.fUnixPath.c_str());
   if (!c) {
      TRACE(DBG, "client instance not initialized");
      return -1;
   }

   // Allocate the server object
   int psid = si.fID;
   XrdProofdProofServ *xps = c->GetServObj(psid);
   if (!xps) {
      TRACE(DBG, "server object not initialized");
      return -1;
   }

   // Fill info for this session
   si.FillProofServ(*xps, fMgr->ROOTMgr());
   if (xps->CreateUNIXSock(fEDest) != 0) {
      // Failure
      TRACE(XERR,"failure creating UNIX socket on " << xps->UNIXSockPath());
      xps->Reset();
      return -1;
   }

   // Set invalid as we are not yet connected
   xps->SetValid(0);

   // Add to the lists
   XrdSysMutexHelper mhp(fRecoverMutex);
   std::list<XpdClientSessions *>::iterator ii = fRecoverClients->begin();
   while (ii != fRecoverClients->end()) {
      if ((*ii)->fClient == c)
         break;
      ii++;
   }
   if (ii != fRecoverClients->end()) {
      (*ii)->fProofServs.push_back(xps);
   } else {
      XpdClientSessions *cl = new XpdClientSessions(c);
      cl->fProofServs.push_back(xps);
      fRecoverClients->push_back(cl);
   }

   // Done
   return 0;
}

//_________________________________________________________________________________
int XrdProofdProofServMgr::Recover(XpdClientSessions *cl)
{
   // Handle a request to recover a session after stop&restart for a specific client
   XPDLOC(SMGR, "ProofServMgr::Recover")

   if (!cl) {
      TRACE(XERR, "invalid input!");
      return 0;
   }

   TRACE(DBG,  "client: "<< cl->fClient->User());

   int nr = 0;
   XrdOucString emsg;
   XrdProofdProofServ *xps = 0;
   int nps = 0, npsref = 0;
   { XrdSysMutexHelper mhp(cl->fMutex); nps = cl->fProofServs.size(), npsref = nps; }
   while (nps--) {

      { XrdSysMutexHelper mhp(cl->fMutex); xps = cl->fProofServs.front();
        cl->fProofServs.remove(xps); cl->fProofServs.push_back(xps); }

      // Short steps of 1 sec
      if (AcceptPeer(xps, 1, emsg) != 0) {
         if (emsg == "timeout") {
            TRACE(DBG, "timeout while accepting callback");
         } else {
            TRACE(XERR, "problems accepting callback: "<<emsg);
         }
      } else {
         // Update the global session handlers
         XrdOucString key; key += xps->SrvPID();
         fSessions.Add(key.c_str(), xps, 0, Hash_keepdata);
         fActiveSessions.push_back(xps);
         xps->Protocol()->SetAdminPath(xps->AdminPath());
         // Remove from the temp list
         { XrdSysMutexHelper mhp(cl->fMutex); cl->fProofServs.remove(xps); }
         // Count
         nr++;
         // Notify
         if (TRACING(REQ)) {
            int pid = xps->SrvPID();
            int left = -1;
            { XrdSysMutexHelper mhp(cl->fMutex); left = cl->fProofServs.size(); }
            XPDPRT("session for "<<cl->fClient->User()<<"."<<cl->fClient->Group()<<
                   " successfully recovered ("<<left<<" left); pid: "<<pid);
         }
      }
   }

   // Over
   return nr;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::AcceptPeer(XrdProofdProofServ *xps,
                                      int to, XrdOucString &msg)
{
   // Accept a callback from a starting-up server and setup the related protocol
   // object. Used for old servers.
   // Return 0 if successful or -1 in case of failure.
   XPDLOC(SMGR, "ProofServMgr::AcceptPeer")

   // We will get back a peer to initialize a link
   XrdNetPeer peerpsrv;

   // Check inputs
   if (!xps || !xps->UNIXSock()) {
      XPDFORM(msg, "session pointer undefined or socket invalid: %p", xps);
      return -1;
   }
   TRACE(REQ, "waiting for server callback for "<<to<<" secs ... on "<<xps->UNIXSockPath());

   // Perform regular accept
   if (!(xps->UNIXSock()->Accept(peerpsrv, XRDNET_NODNTRIM, to))) {
      msg = "timeout";
      return -1;
   }

   // Setup the protocol serving this peer
   if (SetupProtocol(peerpsrv, xps, msg) != 0) {
      msg = "could not assert connected peer: ";
      return -1;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::SetupProtocol(XrdNetPeer &peerpsrv,
                                         XrdProofdProofServ *xps, XrdOucString &msg)
{
   // Setup the protocol object serving the peer described by 'peerpsrv'
   XPDLOC(SMGR, "ProofServMgr::SetupProtocol")

   // We will get back a peer to initialize a link
   XrdLink   *linkpsrv = 0;
   XrdProtocol *xp = 0;
   int lnkopts = 0;
   bool go = 1;

   // Make sure we have the full host name
   if (peerpsrv.InetName) free(peerpsrv.InetName);
   peerpsrv.InetName = XrdNetDNS::getHostName("localhost");

   // Allocate a new network object
   if (!(linkpsrv = XrdLink::Alloc(peerpsrv, lnkopts))) {
      msg = "could not allocate network object: ";
      go = 0;
   }

   if (go) {
      // Keep buffer after object goes away
      peerpsrv.InetBuff = 0;
      TRACE(DBG, "connection accepted: matching protocol ... ");
      // Get a protocol object off the stack (if none, allocate a new one)
      XrdProofdProtocol *p = new XrdProofdProtocol();
      if (!(xp = p->Match(linkpsrv))) {
         msg = "match failed: protocol error: ";
         go = 0;
      }
      delete p;
   }

   if (go) {
      // Save path into the protocol instance: it may be needed during Process
      XrdOucString apath(xps->AdminPath());
      apath += ".status";
      ((XrdProofdProtocol *)xp)->SetAdminPath(apath.c_str());
      // Take a short-cut and process the initial request as a sticky request
      if (xp->Process(linkpsrv) != 0) {
         msg = "handshake with internal link failed: ";
         go = 0;
      }
   }

   // Attach this link to the appropriate poller and enable it.
   if (go && !XrdPoll::Attach(linkpsrv)) {
      msg = "could not attach new internal link to poller: ";
      go = 0;
   }

   if (!go) {
      // Close the link
      if (linkpsrv)
         linkpsrv->Close();
      return -1;
   }

   // Tight this protocol instance to the link
   linkpsrv->setProtocol(xp);

   TRACE(REQ, "Protocol "<<xp<<" attached to link "<<linkpsrv<<" ("<< peerpsrv.InetName <<")");

   // Schedule it
   fMgr->Sched()->Schedule((XrdJob *)linkpsrv);

   // Save the protocol in the session instance
   xps->SetProtocol((XrdProofdProtocol *)xp);

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::Detach(XrdProofdProtocol *p)
{
   // Handle a request to detach from an existing session
   XPDLOC(SMGR, "ProofServMgr::Detach")

   int psid = -1, rc = 0;
   XPD_SETRESP(p, "Detach");

   // Unmarshall the data
   psid = ntohl(p->Request()->proof.sid);
   TRACEP(p, REQ, "psid: "<<psid);

   // Find server session
   XrdProofdProofServ *xps = 0;
   if (!p->Client() || !(xps = p->Client()->GetServer(psid))) {
      TRACEP(p, XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"session ID not found");
      return 0;
   }
   xps->FreeClientID(p->Pid());

   // Notify to user
   response->Send();

   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::Destroy(XrdProofdProtocol *p)
{
   // Handle a request to shutdown an existing session
   XPDLOC(SMGR, "ProofServMgr::Destroy")

   int psid = -1, rc = 0;
   XPD_SETRESP(p, "Destroy");

   // Unmarshall the data
   psid = ntohl(p->Request()->proof.sid);
   TRACEP(p, REQ, "psid: "<<psid);

   XrdOucString msg;

   // Find server session
   XrdProofdProofServ *xpsref = 0;
   if (psid > -1) {
      // Request for a specific session
      if (!p->Client() || !(xpsref = p->Client()->GetServer(psid))) {
         TRACEP(p, XERR, "reference session ID not found");
         response->Send(kXR_InvalidRequest,"reference session ID not found");
         return 0;
      }
      XPDFORM(msg, "session %d destroyed by %s", xpsref->SrvPID(), p->Link()->ID);
   } else {
      XPDFORM(msg, "all sessions destroyed by %s", p->Link()->ID);
   }

   // Terminate the servers
   p->Client()->TerminateSessions(kXPD_AnyServer, xpsref,
                                  msg.c_str(), Pipe(), fMgr->ChangeOwn());

   // Notify to user
   response->Send();

   // Over
   return 0;
}

//__________________________________________________________________________
static int WriteSessEnvs(const char *, XpdEnv *env, void *s)
{
   // Run thorugh entries to broadcast the relevant priority
   XPDLOC(SMGR, "WriteSessEnvs")

   XrdOucString emsg;
   
   XpdWriteEnv_t *xwe = (XpdWriteEnv_t *)s;  

   if (env && xwe && xwe->fMgr && xwe->fClient &&  xwe->fEnv) {
      if (env->fEnv.length() > 0) {
         // Resolve keywords
         xwe->fMgr->ResolveKeywords(env->fEnv, xwe->fClient);
         // Set the env now
         char *ev = new char[env->fEnv.length()+1];
         strncpy(ev, env->fEnv.c_str(), env->fEnv.length());
         ev[env->fEnv.length()] = 0;
         fprintf(xwe->fEnv, "%s\n", ev);
         TRACE(DBG, ev);
         PutEnv(ev, xwe->fExport);
      }
      // Go to next
      return 0;
   } else {
      emsg = "some input undefined";
   }

   // Some problem
   TRACE(XERR,"protocol error: "<<emsg);
   return 1;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::SetProofServEnvOld(XrdProofdProtocol *p, void *input)
{
   // Set environment for proofserv; old version preparing the environment for
   // proofserv protocol version <= 13. Needed for backward compatibility.
   XPDLOC(SMGR, "ProofServMgr::SetProofServEnvOld")

   char *ev = 0;

   // Check inputs
   if (!p || !p->Client() || !input) {
      TRACE(XERR, "at leat one input is invalid - cannot continue");
      return -1;
   }

   // Set basic environment for proofserv
   if (SetProofServEnv(fMgr, p->Client()->ROOT()) != 0) {
      TRACE(XERR, "problems setting basic environment - exit");
      return -1;
   }

   ProofServEnv_t *in = (ProofServEnv_t *)input;

   // Session proxy
   XrdProofdProofServ *xps = in->fPS;
   if (!xps) {
      TRACE(XERR, "unable to get instance of proofserv proxy");
      return -1;
   }
   int psid = xps->ID();
   TRACE(REQ,  "psid: "<<psid<<", log: "<<in->fLogLevel);

   // Work directory
   XrdOucString udir = p->Client()->Sandbox()->Dir();
   TRACE(DBG, "working dir for "<<p->Client()->User()<<" is: "<<udir);

   ev = new char[strlen("ROOTPROOFSESSDIR=") + in->fWrkDir.length() + 2];
   sprintf(ev, "ROOTPROOFSESSDIR=%s", in->fWrkDir.c_str());
   putenv(ev);
   TRACE(DBG, ev);

   // Log level
   ev = new char[strlen("ROOTPROOFLOGLEVEL=")+5];
   sprintf(ev, "ROOTPROOFLOGLEVEL=%d", in->fLogLevel);
   putenv(ev);
   TRACE(DBG, ev);

   // Ordinal number
   ev = new char[strlen("ROOTPROOFORDINAL=")+strlen(xps->Ordinal())+2];
   sprintf(ev, "ROOTPROOFORDINAL=%s", xps->Ordinal());
   putenv(ev);
   TRACE(DBG, ev);

   // ROOT Version tag if not the default one
   ev = new char[strlen("ROOTVERSIONTAG=")+strlen(p->Client()->ROOT()->Tag())+2];
   sprintf(ev, "ROOTVERSIONTAG=%s", p->Client()->ROOT()->Tag());
   putenv(ev);
   TRACE(DBG, ev);

   // Create the env file
   TRACE(DBG, "creating env file");
   XrdOucString envfile = in->fWrkDir;
   envfile += ".env";
   FILE *fenv = fopen(envfile.c_str(), "w");
   if (!fenv) {
      TRACE(XERR,
                  "unable to open env file: "<<envfile);
      return -1;
   }
   TRACE(DBG, "environment file: "<< envfile);

   // Forwarded sec credentials, if any
   if (p->AuthProt()) {

      // Additional envs possibly set by the protocol for next application
      XrdOucString secenvs(getenv("XrdSecENVS"));
      if (secenvs.length() > 0) {
         // Go through the list
         XrdOucString env;
         int from = 0;
         while ((from = secenvs.tokenize(env, from, ',')) != -1) {
            if (env.length() > 0) {
               // Set the env now
               ev = new char[env.length()+1];
               strncpy(ev, env.c_str(), env.length());
               ev[env.length()] = 0;
               putenv(ev);
               fprintf(fenv, "%s\n", ev);
               TRACE(DBG, ev);
            }
         }
      }

      // The credential buffer, if any
      XrdSecCredentials *creds = p->AuthProt()->getCredentials();
      if (creds) {
         int lev = strlen("XrdSecCREDS=")+creds->size;
         ev = new char[lev+1];
         strcpy(ev, "XrdSecCREDS=");
         memcpy(ev+strlen("XrdSecCREDS="), creds->buffer, creds->size);
         ev[lev] = 0;
         putenv(ev);
         TRACE(DBG, "XrdSecCREDS set");

         // If 'pwd', save AFS key, if any
         if (!strncmp(p->AuthProt()->Entity.prot, "pwd", 3)) {
            XrdOucString credsdir = udir;
            credsdir += "/.creds";
            // Make sure the directory exists
            if (!XrdProofdAux::AssertDir(credsdir.c_str(), p->Client()->UI(), fMgr->ChangeOwn())) {
               if (SaveAFSkey(creds, credsdir.c_str(), p->Client()->UI()) == 0) {
                  ev = new char[strlen("ROOTPROOFAFSCREDS=")+credsdir.length()+strlen("/.afs")+2];
                  sprintf(ev, "ROOTPROOFAFSCREDS=%s/.afs", credsdir.c_str());
                  putenv(ev);
                  fprintf(fenv, "ROOTPROOFAFSCREDS has been set\n");
                  TRACE(DBG, ev);
               } else {
                  TRACE(DBG, "problems in saving AFS key");
               }
            } else {
               TRACE(XERR, "unable to create creds dir: "<<credsdir);
               return -1;
            }
         }
      }
   }

   // Set ROOTSYS
   fprintf(fenv, "ROOTSYS=%s\n", xps->ROOT()->Dir());

   // Set conf dir
   fprintf(fenv, "ROOTCONFDIR=%s\n", xps->ROOT()->Dir());

   // Set TMPDIR
   fprintf(fenv, "ROOTTMPDIR=%s\n", fMgr->TMPdir());

   // Port (really needed?)
   fprintf(fenv, "ROOTXPDPORT=%d\n", fMgr->Port());

   // Work dir
   fprintf(fenv, "ROOTPROOFWORKDIR=%s\n", udir.c_str());

   // Session tag
   fprintf(fenv, "ROOTPROOFSESSIONTAG=%s\n", in->fSessionTag.c_str());

   // Whether user specific config files are enabled
   if (fMgr->NetMgr()->WorkerUsrCfg())
      fprintf(fenv, "ROOTUSEUSERCFG=1\n");

   // Set Open socket
   fprintf(fenv, "ROOTOPENSOCK=%s\n", xps->UNIXSockPath());

   // Entity
   fprintf(fenv, "ROOTENTITY=%s@%s\n", p->Client()->User(), p->Link()->Host());

   // Session ID
   fprintf(fenv, "ROOTSESSIONID=%d\n", psid);

   // Client ID
   fprintf(fenv, "ROOTCLIENTID=%d\n", p->CID());

   // Client Protocol
   fprintf(fenv, "ROOTPROOFCLNTVERS=%d\n", p->ProofProtocol());

   // Ordinal number
   fprintf(fenv, "ROOTPROOFORDINAL=%s\n", xps->Ordinal());

   // ROOT version tag if different from the default one
   if (getenv("ROOTVERSIONTAG"))
      fprintf(fenv, "ROOTVERSIONTAG=%s\n", getenv("ROOTVERSIONTAG"));

   // Config file
   if (in->fCfg.length() > 0)
      fprintf(fenv, "ROOTPROOFCFGFILE=%s\n", in->fCfg.c_str());

   // Log file in the log dir
   fprintf(fenv, "ROOTPROOFLOGFILE=%s\n", in->fLogFile.c_str());
   xps->SetFileout(in->fLogFile.c_str());

   // Additional envs (xpd.putenv directive)
   {  XrdSysMutexHelper mhp(fEnvsMutex);
      if (fProofServEnvs.size() > 0) {
         // Hash list of the directives applying to this {user, group, svn, version}
         XrdOucHash<XpdEnv> sessenvs;
         std::list<XpdEnv>::iterator ienvs = fProofServEnvs.begin();
         for ( ; ienvs != fProofServEnvs.end(); ienvs++) {
            int envmatch = (*ienvs).Matches(p->Client()->User(), p->Client()->Group(),
                                            p->Client()->ROOT()->SvnRevision(),
                                            p->Client()->ROOT()->VersionCode());
            if (envmatch >= 0) {
               XpdEnv *env = sessenvs.Find((*ienvs).fName.c_str());
               if (env) {
                  int envmtcex = env->Matches(p->Client()->User(), p->Client()->Group(),
                                              p->Client()->ROOT()->SvnRevision(),
                                              p->Client()->ROOT()->VersionCode());
                  if (envmatch > envmtcex) {
                     // Replace the entry
                     env = &(*ienvs);
                     sessenvs.Rep(env->fName.c_str(), env, 0, Hash_keepdata);
                  }
               } else {
                  // Add an entry
                  env = &(*ienvs);
                  sessenvs.Add(env->fName.c_str(), env, 0, Hash_keepdata);
               }
               TRACE(HDBG, "Adding: "<<(*ienvs).fEnv);
            }
         }
         XpdWriteEnv_t xpwe = {fMgr, p->Client(), fenv, in->fOld};
         sessenvs.Apply(WriteSessEnvs, (void *)&xpwe);
      }
   }

   // Set the user envs
   if (xps->UserEnvs() &&
       strlen(xps->UserEnvs()) && strstr(xps->UserEnvs(),"=")) {
      // The single components
      XrdOucString ue = xps->UserEnvs();
      XrdOucString env, namelist;
      int from = 0, ieq = -1;
      while ((from = ue.tokenize(env, from, ',')) != -1) {
         if (env.length() > 0 && (ieq = env.find('=')) != -1) {
            // Resolve keywords
            ResolveKeywords(env, in);
            ev = new char[env.length()+1];
            strncpy(ev, env.c_str(), env.length());
            ev[env.length()] = 0;
            putenv(ev);
            fprintf(fenv, "%s\n", ev);
            TRACE(DBG, ev);
            env.erase(ieq);
            if (namelist.length() > 0)
               namelist += ',';
            namelist += env;
         }
      }
      // The list of names, ','-separated
      ev = new char[strlen("PROOF_ALLVARS=") + namelist.length() + 2];
      sprintf(ev, "PROOF_ALLVARS=%s", namelist.c_str());
      putenv(ev);
      fprintf(fenv, "%s\n", ev);
      TRACE(DBG, ev);
   }

   // Close file
   fclose(fenv);

   // Create or Update symlink to last session
   TRACE(DBG, "creating symlink");
   XrdOucString syml = udir;
   if (p->ConnType() == kXPD_MasterWorker)
      syml += "/last-worker-session";
   else
      syml += "/last-master-session";
   if (XrdProofdAux::SymLink(in->fSessionDir.c_str(), syml.c_str()) != 0) {
      TRACE(XERR, "problems creating symlink to last session (errno: "<<errno<<")");
   }

   // We are done
   TRACE(DBG, "done");
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::SetProofServEnv(XrdProofdManager *mgr, XrdROOT *r)
{
   // Set basic environment accordingly to 'r'
   XPDLOC(SMGR, "ProofServMgr::SetProofServEnv")

   char *ev = 0;

   TRACE(REQ,  "ROOT dir: "<< (r ? r->Dir() : "*** undef ***"));

   if (r) {
      char *libdir = (char *) r->LibDir();
      char *ldpath = 0;
      if (mgr->BareLibPath() && strlen(mgr->BareLibPath()) > 0) {
         ldpath = new char[32 + strlen(libdir) + strlen(mgr->BareLibPath())];
         sprintf(ldpath, "%s=%s:%s", XPD_LIBPATH, libdir, mgr->BareLibPath());
      } else {
         ldpath = new char[32 + strlen(libdir)];
         sprintf(ldpath, "%s=%s", XPD_LIBPATH, libdir);
      }
      putenv(ldpath);
      // Set ROOTSYS
      char *rootsys = (char *) r->Dir();
      ev = new char[15 + strlen(rootsys)];
      sprintf(ev, "ROOTSYS=%s", rootsys);
      putenv(ev);

      // Set bin directory
      char *bindir = (char *) r->BinDir();
      ev = new char[15 + strlen(bindir)];
      sprintf(ev, "ROOTBINDIR=%s", bindir);
      putenv(ev);

      // Set conf dir
      char *confdir = (char *) r->DataDir();
      ev = new char[20 + strlen(confdir)];
      sprintf(ev, "ROOTCONFDIR=%s", confdir);
      putenv(ev);

      // Set TMPDIR
      ev = new char[20 + strlen(mgr->TMPdir())];
      sprintf(ev, "TMPDIR=%s", mgr->TMPdir());
      putenv(ev);

      // Done
      return 0;
   }

   // Bad input
   TRACE(XERR, "XrdROOT instance undefined!");
   return -1;
}

//______________________________________________________________________________
void XrdProofdProofServMgr::GetTagDirs(int pid,
                                       XrdProofdProtocol *p, XrdProofdProofServ *xps,
                                       XrdOucString &sesstag, XrdOucString &topsesstag,
                                       XrdOucString &sessiondir, XrdOucString &sesswrkdir)
{
   // Determine the unique tag and relevant dirs for this session
   XPDLOC(SMGR, "GetTagDirs")

   // Client sandbox
   XrdOucString udir = p->Client()->Sandbox()->Dir();

   if (pid == 0) {

      // Create the unique tag identify this session
      XrdOucString host = fMgr->Host();
      if (host.find(".") != STR_NPOS)
         host.erase(host.find("."));
      XPDFORM(sesstag, "%s-%d-", host.c_str(), (int)time(0));

      // Session dir
      topsesstag = sesstag;
      sessiondir = udir;
      if (p->ConnType() == kXPD_ClientMaster) {
         sessiondir += "/session-";
         sessiondir += sesstag;
      } else {
         sessiondir += "/";
         sessiondir += xps->Tag();
         topsesstag = xps->Tag();
         topsesstag.replace("session-","");
         // If the child, make sure the directory exists ...
         if (XrdProofdAux::AssertDir(sessiondir.c_str(), p->Client()->UI(),
                                    fMgr->ChangeOwn()) == -1) {
            TRACE(XERR, "problems asserting dir '"<<sessiondir<<"' - errno: "<<errno);  
            return;
         }
      }

   } else if (pid > 0) {

      // Finalize unique tag identifying this session
      sesstag += pid;

      // Session dir
      topsesstag = sesstag;
      if (p->ConnType() == kXPD_ClientMaster) {
         sessiondir += pid;
         xps->SetTag(sesstag.c_str());
      }

      // If the child, make sure the directory exists ...
      if (pid == (int) getpid()) {
         if (XrdProofdAux::AssertDir(sessiondir.c_str(), p->Client()->UI(),
                                    fMgr->ChangeOwn()) == -1) {
            return;
         }
      }

      // The session working dir depends on the role
      sesswrkdir = sessiondir;
      if (p->ConnType() == kXPD_MasterWorker) {
         XPDFORM(sesswrkdir, "%s/worker-%s-%s", sessiondir.c_str(), xps->Ordinal(), sesstag.c_str());
      } else {
         XPDFORM(sesswrkdir, "%s/master-%s-%s", sessiondir.c_str(), xps->Ordinal(), sesstag.c_str());
      }
   } else {
      TRACE(XERR, "negative pid ("<<pid<<"): should not have got here!");  
   }

   // Done
   return;
}

//__________________________________________________________________________
static int WriteSessRCs(const char *, XpdEnv *erc, void *f)
{
   // Run thorugh entries to broadcast the relevant priority
   XPDLOC(SMGR, "WriteSessRCs")

   XrdOucString emsg;
   FILE *frc = (FILE *)f;
   if (frc && erc) {
      XrdOucString rc = erc->fEnv;
      if (rc.length() > 0) {
         if (rc.find("Proof.DataSetManager") != STR_NPOS) {
            TRACE(ALL,"Proof.DataSetManager ignored: use xpd.datasetsrc to define dataset managers");
         } else {
            fprintf(frc, "%s\n", rc.c_str());
         }
      }
      // Go to next
      return 0;
   } else {
      emsg = "file or input entry undefined";
   }

   // Some problem
   TRACE(XERR,"protocol error: "<<emsg);
   return 1;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::SetProofServEnv(XrdProofdProtocol *p, void *input)
{
   // Set environment for proofserv
   XPDLOC(SMGR, "ProofServMgr::SetProofServEnv")

   // Check inputs
   if (!p || !p->Client() || !input) {
      TRACE(XERR, "at leat one input is invalid - cannot continue");
      return -1;
   }

   // Old proofservs expect different settings
   int rootvers = p->Client()->ROOT() ? p->Client()->ROOT()->SrvProtVers() : -1;
   TRACE(DBG, "rootvers: "<< rootvers);
   if (rootvers < 14 && rootvers > -1)
      return SetProofServEnvOld(p, input);

   ProofServEnv_t *in = (ProofServEnv_t *)input;

   // Session proxy
   XrdProofdProofServ *xps = in->fPS;
   if (!xps) {
      TRACE(XERR, "unable to get instance of proofserv proxy");
      return -1;
   }
   int psid = xps->ID();
   TRACE(REQ,  "psid: "<<psid<<", log: "<<in->fLogLevel);

   // Client sandbox
   XrdOucString udir = p->Client()->Sandbox()->Dir();
   TRACE(DBG, "sandbox for "<<p->Client()->User()<<" is: "<<udir);
   TRACE(DBG, "session unique tag "<<in->fSessionTag);
   TRACE(DBG, "session dir " << in->fSessionDir);
   TRACE(DBG, "session working dir:" << in->fWrkDir);

   // Log into the session it
   if (XrdProofdAux::ChangeToDir(in->fSessionDir.c_str(), p->Client()->UI(),
                                 fMgr->ChangeOwn()) != 0) {
      TRACE(XERR, "couldn't change directory to " << in->fSessionDir);
      return -1;
   }

   // Set basic environment for proofserv
   if (SetProofServEnv(fMgr, p->Client()->ROOT()) != 0) {
      TRACE(XERR, "problems setting basic environment - exit");
      return -1;
   }

   // Create the rootrc and env files
   TRACE(DBG, "creating env file");
   XrdOucString rcfile = in->fWrkDir;
   rcfile += ".rootrc";
   if (CreateProofServRootRc(p, in, rcfile.c_str()) != 0) {
      TRACE(XERR, "problems creating RC file "<<rcfile.c_str());
      return -1;
   }

   // Now save the exported env variables, for the record
   XrdOucString envfile = in->fWrkDir;
   envfile += ".env";
   if (CreateProofServEnvFile(p, in, envfile.c_str(), rcfile.c_str()) != 0) {
      TRACE(XERR, "problems creating environment file "<<envfile.c_str());
      return -1;
   }

   // Create or Update symlink to last session
   if (in->fOld) {
      TRACE(REQ, "creating symlink");
      XrdOucString syml = udir;
      if (p->ConnType() == kXPD_MasterWorker)
         syml += "/last-worker-session";
      else
         syml += "/last-master-session";
      if (XrdProofdAux::SymLink(in->fSessionDir.c_str(), syml.c_str()) != 0) {
         TRACE(XERR, "problems creating symlink to "
                     " last session (errno: "<<errno<<")");
      }
   }
   
   // We are done
   TRACE(REQ, "done");
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::CreateProofServEnvFile(XrdProofdProtocol *p, void *input,
                                                  const char *envfn, const char *rcfn)
{
   // Create in 'rcfn' the rootrc file for the proofserv being created
   // return 0 on success, -1 on error
   XPDLOC(SMGR, "ProofServMgr::CreateProofServEnvFile")

   // Check inputs
   if (!p || !input || (!envfn ||
      (envfn && strlen(envfn) <= 0)) || (!rcfn || (rcfn && strlen(rcfn) <= 0))) {
      TRACE(XERR, "invalid inputs!");
      return -1;
   }

   // Attach the structure
   ProofServEnv_t *in = (ProofServEnv_t *)input;
   
   // Session proxy
   XrdProofdProofServ *xps = in->fPS;
   if (!xps) {
      TRACE(XERR, "unable to get instance of proofserv proxy");
      return -1;
   }

   FILE *fenv = fopen(envfn, "w");
   if (!fenv) {
      TRACE(XERR, "unable to open env file: "<<envfn);
      return -1;
   }
   TRACE(REQ, "environment file: "<< envfn);

   char *ev = 0;
   // Forwarded sec credentials, if any
   if (p->AuthProt()) {

      // Additional envs possibly set by the protocol for next application
      XrdOucString secenvs(getenv("XrdSecENVS"));
      if (secenvs.length() > 0) {
         // Go through the list
         XrdOucString env;
         int from = 0;
         while ((from = secenvs.tokenize(env, from, ',')) != -1) {
            if (env.length() > 0) {
               // Set the env now
               ev = new char[env.length()+1];
               strncpy(ev, env.c_str(), env.length());
               ev[env.length()] = 0;
               fprintf(fenv, "%s\n", ev);
               TRACE(DBG, ev);
               PutEnv(ev, in->fOld);
            }
         }
      }

      // The credential buffer, if any
      XrdSecCredentials *creds = p->AuthProt()->getCredentials();
      if (creds) {
         int lev = strlen("XrdSecCREDS=")+creds->size;
         ev = new char[lev+1];
         strcpy(ev, "XrdSecCREDS=");
         memcpy(ev+strlen("XrdSecCREDS="), creds->buffer, creds->size);
         ev[lev] = 0;
         PutEnv(ev, in->fOld);
         TRACE(DBG, "XrdSecCREDS set");

         // If 'pwd', save AFS key, if any
         if (!strncmp(p->AuthProt()->Entity.prot, "pwd", 3)) {
            XrdOucString credsdir = p->Client()->Sandbox()->Dir();
            credsdir += "/.creds";
            // Make sure the directory exists
            if (!XrdProofdAux::AssertDir(credsdir.c_str(), p->Client()->UI(), fMgr->ChangeOwn())) {
               if (SaveAFSkey(creds, credsdir.c_str(), p->Client()->UI()) == 0) {
                  ev = new char[strlen("ROOTPROOFAFSCREDS=")+credsdir.length()+strlen("/.afs")+2];
                  sprintf(ev, "ROOTPROOFAFSCREDS=%s/.afs", credsdir.c_str());
                  fprintf(fenv, "ROOTPROOFAFSCREDS has been set\n");
                  TRACE(DBG, ev);
                  PutEnv(ev, in->fOld);
               } else {
                  TRACE(DBG, "problems in saving AFS key");
               }
            } else {
               TRACE(XERR, "unable to create creds dir: "<<credsdir);
               return -1;
            }
         }
      }
   }

   // Library path
   fprintf(fenv, "%s=%s\n", XPD_LIBPATH, getenv(XPD_LIBPATH));

   // ROOTSYS
   fprintf(fenv, "ROOTSYS=%s\n", xps->ROOT()->Dir());

   // Conf dir
   fprintf(fenv, "ROOTCONFDIR=%s\n", xps->ROOT()->Dir());

   // TMPDIR
   fprintf(fenv, "TMPDIR=%s\n", fMgr->TMPdir());

   // RC file
   if (in->fOld) {
      ev = new char[strlen("ROOTRCFILE=")+strlen(rcfn)+2];
      sprintf(ev, "ROOTRCFILE=%s", rcfn);
      fprintf(fenv, "%s\n", ev);
      TRACE(DBG, ev);
      PutEnv(ev, in->fOld);
   }

   // ROOT version tag (needed in building packages)
   ev = new char[strlen("ROOTVERSIONTAG=")+strlen(p->Client()->ROOT()->Tag())+2];
   sprintf(ev, "ROOTVERSIONTAG=%s", p->Client()->ROOT()->Tag());
   fprintf(fenv, "%s\n", ev);
   TRACE(DBG, ev);
   PutEnv(ev, in->fOld);

   // Log file in the log dir
   if (in->fOld) {
      ev = new char[strlen("ROOTPROOFLOGFILE=") + in->fLogFile.length() + 2];
      sprintf(ev, "ROOTPROOFLOGFILE=%s", in->fLogFile.c_str());
      fprintf(fenv, "%s\n", ev);
      xps->SetFileout(in->fLogFile.c_str());
      TRACE(DBG, ev);
      PutEnv(ev, in->fOld);
   }

   // Local data server
   XrdOucString locdatasrv;
   if (strlen(fMgr->RootdExe()) <= 0) {
      XPDFORM(locdatasrv, "root://%s", fMgr->Host());
   } else { 
      XPDFORM(locdatasrv, "rootd://%s:%d", fMgr->Host(), fMgr->Port());
   }
   ev = new char[strlen("LOCALDATASERVER=") + locdatasrv.length() + 2];
   sprintf(ev, "LOCALDATASERVER=%s", locdatasrv.c_str());
   fprintf(fenv, "%s\n", ev);
   TRACE(DBG, ev);
   PutEnv(ev, in->fOld);

   // Xrootd config file
   if (CfgFile()) {
      ev = new char[strlen("XRDCF=")+strlen(CfgFile())+2];
      sprintf(ev, "XRDCF=%s", CfgFile());
      fprintf(fenv, "%s\n", ev);
      TRACE(DBG, ev);
      PutEnv(ev, in->fOld);
   }

   // Additional envs (xpd.putenv directive)
   {  XrdSysMutexHelper mhp(fEnvsMutex);
      if (fProofServEnvs.size() > 0) {
         // Hash list of the directives applying to this {user, group, svn, version}
         XrdOucHash<XpdEnv> sessenvs;
         std::list<XpdEnv>::iterator ienvs = fProofServEnvs.begin();
         for ( ; ienvs != fProofServEnvs.end(); ienvs++) {
            int envmatch = (*ienvs).Matches(p->Client()->User(), p->Client()->Group(),
                                            p->Client()->ROOT()->SvnRevision(),
                                            p->Client()->ROOT()->VersionCode());
            if (envmatch >= 0) {
               XpdEnv *env = sessenvs.Find((*ienvs).fName.c_str());
               if (env) {
                  int envmtcex = env->Matches(p->Client()->User(), p->Client()->Group(),
                                              p->Client()->ROOT()->SvnRevision(),
                                              p->Client()->ROOT()->VersionCode());
                  if (envmatch > envmtcex) {
                     // Replace the entry
                     env = &(*ienvs);
                     sessenvs.Rep(env->fName.c_str(), env, 0, Hash_keepdata);
                  }
               } else {
                  // Add an entry
                  env = &(*ienvs);
                  sessenvs.Add(env->fName.c_str(), env, 0, Hash_keepdata);
               }
               TRACE(HDBG, "Adding: "<<(*ienvs).fEnv);
            }
         }
         XpdWriteEnv_t xpwe = {fMgr, p->Client(), fenv, in->fOld};
         sessenvs.Apply(WriteSessEnvs, (void *)&xpwe);
      }
   }
   // Set the user envs
   if (xps->UserEnvs() &&
       strlen(xps->UserEnvs()) && strstr(xps->UserEnvs(),"=")) {
      // The single components
      XrdOucString ue = xps->UserEnvs();
      XrdOucString env, namelist;
      int from = 0, ieq = -1;
      while ((from = ue.tokenize(env, from, ',')) != -1) {
         if (env.length() > 0 && (ieq = env.find('=')) != -1) {
            // Resolve keywords
            ResolveKeywords(env, in);
            ev = new char[env.length()+1];
            strncpy(ev, env.c_str(), env.length());
            ev[env.length()] = 0;
            fprintf(fenv, "%s\n", ev);
            TRACE(DBG, ev);
            PutEnv(ev, in->fOld);
            env.erase(ieq);
            if (namelist.length() > 0)
               namelist += ',';
            namelist += env;
         }
      }
      // The list of names, ','-separated
      ev = new char[strlen("PROOF_ALLVARS=") + namelist.length() + 2];
      sprintf(ev, "PROOF_ALLVARS=%s", namelist.c_str());
      fprintf(fenv, "%s\n", ev);
      TRACE(DBG, ev);
      PutEnv(ev, in->fOld);
   }

   // Close file
   fclose(fenv);
   
   // We are done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::CreateProofServRootRc(XrdProofdProtocol *p,
                                                 void *input, const char *rcfn)
{
   // Create in 'rcfn' the rootrc file for the proofserv being created
   // return 0 on success, -1 on error
   XPDLOC(SMGR, "ProofServMgr::CreateProofServRootRc")

   // Check inputs
   if (!p || !input || (!rcfn || (rcfn && strlen(rcfn) <= 0))) {
      TRACE(XERR, "invalid inputs!");
      return -1;
   }

   // Attach the structure
   ProofServEnv_t *in = (ProofServEnv_t *)input;
   
   // Session proxy
   XrdProofdProofServ *xps = in->fPS;
   if (!xps) {
      TRACE(XERR, "unable to get instance of proofserv proxy");
      return -1;
   }
   int psid = xps->ID();

   FILE *frc = fopen(rcfn, "w");
   if (!frc) {
      TRACE(XERR, "unable to open rootrc file: "<<rcfn);
      return -1;
   }
   // Symlink to session.rootrc
   if (in->fOld) {
      if (XrdProofdAux::SymLink(rcfn, "session.rootrc") != 0) {
         TRACE(XERR, "problems creating symlink to 'session.rootrc' (errno: "<<errno<<")");
      }
   }
   TRACE(REQ, "session rootrc file: "<< rcfn);

   // Port
   fprintf(frc, "# XrdProofdProtocol listening port\n");
   fprintf(frc, "ProofServ.XpdPort: %d\n", fMgr->Port());

   // Local root prefix
   if (fMgr->LocalROOT() && strlen(fMgr->LocalROOT()) > 0) {
      fprintf(frc, "# Prefix to be prepended to local paths\n");
      fprintf(frc, "Path.Localroot: %s\n", fMgr->LocalROOT());
   }

   // Data pool entry-point URL
   if (fMgr->PoolURL() && strlen(fMgr->PoolURL()) > 0) {
      XrdOucString purl(fMgr->PoolURL());
      if (!purl.endswith("/"))
         purl += "/";
      fprintf(frc, "# URL for the data pool entry-point\n");
      fprintf(frc, "ProofServ.PoolUrl: %s\n", purl.c_str());
   }

   // The session working dir depends on the role
   if (in->fOld) {
      fprintf(frc, "# The session working dir\n");
      fprintf(frc, "ProofServ.SessionDir: %s\n", in->fWrkDir.c_str());
   }

   // Log / Debug level
   fprintf(frc, "# Proof Log/Debug level\n");
   fprintf(frc, "Proof.DebugLevel: %d\n", in->fLogLevel);

   // Ordinal number
   fprintf(frc, "# Ordinal number\n");
   fprintf(frc, "ProofServ.Ordinal: %s\n", xps->Ordinal());

   // ROOT Version tag
   if (p->Client()->ROOT()) {
      fprintf(frc, "# ROOT Version tag\n");
      fprintf(frc, "ProofServ.RootVersionTag: %s\n", p->Client()->ROOT()->Tag());
   }
   // Proof group
   if (p->Client()->Group()) {
      fprintf(frc, "# Proof group\n");
      fprintf(frc, "ProofServ.ProofGroup: %s\n", p->Client()->Group());
   }

   //  Path to file with group information
   if (fMgr->GroupsMgr() && fMgr->GroupsMgr()->GetCfgFile()) {
      fprintf(frc, "# File with group information\n");
      fprintf(frc, "Proof.GroupFile: %s\n", fMgr->GroupsMgr()->GetCfgFile());
   }

   // Work dir
   XrdOucString udir = p->Client()->Sandbox()->Dir();
   fprintf(frc, "# Users sandbox\n");
   fprintf(frc, "ProofServ.Sandbox: %s\n", udir.c_str());

   // Image
   if (fMgr->Image() && strlen(fMgr->Image()) > 0) {
      fprintf(frc, "# Server image\n");
      fprintf(frc, "ProofServ.Image: %s\n", fMgr->Image());
   }

   // Session tag
   if (in->fOld) {
      fprintf(frc, "# Session tag\n");
      fprintf(frc, "ProofServ.SessionTag: %s\n", in->fTopSessionTag.c_str());
   }

   // Session admin path
   fprintf(frc, "# Session admin path\n");
   int proofvrs = (p->Client()->ROOT()) ? p->Client()->ROOT()->SrvProtVers() : -1;
   if (proofvrs < 0 || proofvrs < 27) {
      // Use the first version of the session status file
      fprintf(frc, "ProofServ.AdminPath: %s\n", xps->AdminPath());
   } else {
      if (in->fOld) {
         // New version with updated status
         fprintf(frc, "ProofServ.AdminPath: %s.status\n", xps->AdminPath());
      }
   }

   // Whether user specific config files are enabled
   if (fMgr->NetMgr()->WorkerUsrCfg()) {
      fprintf(frc, "# Whether user specific config files are enabled\n");
      fprintf(frc, "ProofServ.UseUserCfg: 1\n");
   }
   // Set Open socket
   fprintf(frc, "# Open socket\n");
   fprintf(frc, "ProofServ.OpenSock: %s\n", xps->UNIXSockPath());
   // Entity
   fprintf(frc, "# Entity\n");
   if (p->Client()->UI().fGroup.length() > 0)
      fprintf(frc, "ProofServ.Entity: %s:%s@%s\n",
              p->Client()->User(), p->Client()->UI().fGroup.c_str(), p->Link()->Host());
   else
      fprintf(frc, "ProofServ.Entity: %s@%s\n", p->Client()->User(), p->Link()->Host());


   // Session ID
   fprintf(frc, "# Session ID\n");
   fprintf(frc, "ProofServ.SessionID: %d\n", psid);

   // Client ID
   fprintf(frc, "# Client ID\n");
   fprintf(frc, "ProofServ.ClientID: %d\n", p->CID());

   // Client Protocol
   fprintf(frc, "# Client Protocol\n");
   fprintf(frc, "ProofServ.ClientVersion: %d\n", p->ProofProtocol());

   // Config file
   if (in->fCfg.length() > 0) {
      if (in->fCfg == "masteronly") {
         fprintf(frc, "# MasterOnly option\n");
         // Master Only setup
         fprintf(frc, "Proof.MasterOnly: 1\n");
      } else {
         fprintf(frc, "# Config file\n");
         // User defined
         fprintf(frc, "ProofServ.ProofConfFile: %s\n", in->fCfg.c_str());
      }
   } else {
      fprintf(frc, "# Config file\n");
      if (fMgr->IsSuperMst()) {
         fprintf(frc, "# Config file\n");
         fprintf(frc, "ProofServ.ProofConfFile: sm:\n");
      } else if (fProofPlugin.length() > 0) {
         fprintf(frc, "# Config file\n");
         fprintf(frc, "ProofServ.ProofConfFile: %s\n", fProofPlugin.c_str());
      }
   }

   // We set this to avoid blocking to much on xrdclient actions; they can be
   // oevrwritten with explicit putrc directives
   fprintf(frc, "# Default settings for XrdClient\n");
   fprintf(frc, "XNet.FirstConnectMaxCnt 3\n");
   fprintf(frc, "XNet.ConnectTimeout     5\n");

   // This is a workaround for a problem fixed in 5.24/00
   int vrscode = (p->Client()->ROOT()) ? p->Client()->ROOT()->VersionCode() : -1;
   if (vrscode > 0 && vrscode < XrdROOT::GetVersionCode(5,24,0)) {
      fprintf(frc, "# Force remote reading also for local files to avoid a wrong TTreeCache initialization\n");
      fprintf(frc, "Path.ForceRemote 1\n");
   }

   // Additional rootrcs (xpd.putrc directive)
   {  XrdSysMutexHelper mhp(fEnvsMutex);
      if (fProofServRCs.size() > 0) {
         fprintf(frc, "# Additional rootrcs (xpd.putrc directives)\n");
         // Hash list of the directives applying to this {user, group, svn, version}
         XrdOucHash<XpdEnv> sessrcs;
         std::list<XpdEnv>::iterator ircs = fProofServRCs.begin();
         for ( ; ircs != fProofServRCs.end(); ircs++) {
            int rcmatch = (*ircs).Matches(p->Client()->User(), p->Client()->Group(),
                                          p->Client()->ROOT()->SvnRevision(),
                                          p->Client()->ROOT()->VersionCode());
            if (rcmatch >= 0) {
               XpdEnv *rcenv = sessrcs.Find((*ircs).fName.c_str());
               if (rcenv) {
                  int rcmtcex = rcenv->Matches(p->Client()->User(), p->Client()->Group(),
                                               p->Client()->ROOT()->SvnRevision(),
                                               p->Client()->ROOT()->VersionCode());
                  if (rcmatch > rcmtcex) {
                     // Replace the entry
                     rcenv = &(*ircs);
                     sessrcs.Rep(rcenv->fName.c_str(), rcenv, 0, Hash_keepdata);
                  }
               } else {
                  // Add an entry
                  rcenv = &(*ircs);
                  sessrcs.Add(rcenv->fName.c_str(), rcenv, 0, Hash_keepdata);
               }
               TRACE(HDBG, "Adding: "<<(*ircs).fEnv);
            }
         }
         sessrcs.Apply(WriteSessRCs, (void *)frc);
      }
   }
   // If applicable, add dataset managers initiators
   if (fMgr->DataSetSrcs()->size() > 0) {
      fprintf(frc, "# Dataset sources\n");
      XrdOucString rc("Proof.DataSetManager: ");
      std::list<XrdProofdDSInfo *>::iterator ii;
      for (ii = fMgr->DataSetSrcs()->begin(); ii != fMgr->DataSetSrcs()->end(); ii++) {
         if (ii != fMgr->DataSetSrcs()->begin()) rc += ", ";
         rc += (*ii)->fType;
         rc += " dir:";
         rc += (*ii)->fUrl;
         rc += " opt:";
         rc += (*ii)->fOpts;
      }
      fprintf(frc, "%s\n", rc.c_str());
   }

   // If applicable, add datadir location
   if (fMgr->DataDir() && strlen(fMgr->DataDir()) > 0) {
      fprintf(frc, "# Data directory\n");
      XrdOucString rc;
      XPDFORM(rc, "ProofServ.DataDir: %s/%s/%s/%s/%s", fMgr->DataDir(),
                  p->Client()->Group(), p->Client()->User(), xps->Ordinal(),
                  in->fSessionTag.c_str());
      fprintf(frc, "%s\n", rc.c_str());
   }

   // Done with this
   fclose(frc);
   
   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::CleanupLostProofServ()
{
   // Cleanup (kill) all 'proofserv' processes which lost control from their
   // creator or controller daemon. We rely here on the information in the admin
   // path(s) (<xrd_admin>/.xproof.<port>).
   // This is called regurarly by the cron job to avoid having proofservs around.
   // Return number of process killed or -1 in case of any error.
   XPDLOC(SMGR, "ProofServMgr::CleanupLostProofServ")

   if (!fCheckLost) {
      TRACE(REQ,  "disabled ...");
      return 0;
   }

   TRACE(REQ,  "checking for orphalin proofserv processes ...");
   int nk = 0;

   // Get the list of existing proofserv processes from the process table
   std::map<int,XrdOucString> procs;
   if (XrdProofdAux::GetProcesses("proofserv", &procs) <= 0) {
      TRACE(DBG, " no proofservs around: nothing to do");
      return 0;
   }

   XrdProofUI ui;
   if (XrdProofdAux::GetUserInfo(fMgr->EffectiveUser(), ui) != 0) {
      TRACE(DBG, "problems getting info for user " << fMgr->EffectiveUser());
      return -1;
   }

   // Hash list of controlled and xrootd process
   XrdOucRash<int, int> controlled, xrdproc;

   // Hash list of sessions files loaded
   XrdOucHash<XrdOucString> sessionspaths;

   // For each process extract the information about the daemon supposed to be in control
   int pid, ia, a;
   XrdOucString cmd, apath, pidpath, sessiondir, emsg, rest, after;
   std::map<int,XrdOucString>::iterator ip;
   for (ip = procs.begin(); ip != procs.end(); ip++) {
      pid = ip->first;
      cmd = ip->second;
      if ((ia = cmd.find("xpdpath:")) != STR_NPOS) {
         cmd.tokenize(apath, ia, ' ');
         apath.replace("xpdpath:", "");
         if (apath.length() <= 0) {
            TRACE(ALL, "admin path not found; initial cmd line: "<<cmd);
            continue;
         }
         // Extract daemon PID and check that it is alive
         XPDFORM(pidpath, "%s/xrootd.pid", apath.c_str());
         TRACE(ALL, "pidpath: "<<pidpath);
         int xpid = XrdProofdAux::GetIDFromPath(pidpath.c_str(), emsg);
         int *alive = xrdproc.Find(xpid);
         if (!alive) {
            a = (XrdProofdAux::VerifyProcessByID(xpid, fParentExecs.c_str())) ? 1 : 0;
            xrdproc.Add(xpid, a);
            if (!(alive = xrdproc.Find(xpid))) {
               TRACE(ALL, "unable to add info in the Rash table!");
            }
         } else {
            a = *alive;
         }
         // If the daemon is still there check that the process has its entry in the
         // session path(s);
         bool ok = 0;
         if (a == 1) {
            const char *subdir[2] = {"activesessions", "terminatedsessions"};
            for (int i = 0; i < 2; i++) {
               XPDFORM(sessiondir, "%s/%s", apath.c_str(), subdir[i]);
               if (!sessionspaths.Find(sessiondir.c_str())) {
                  DIR *sdir = opendir(sessiondir.c_str());
                  if (!sdir) {
                     XPDFORM(emsg, "cannot open '%s' - errno: %d", apath.c_str(), errno);
                     TRACE(XERR, emsg.c_str());
                     continue;
                  }
                  struct dirent *sent = 0;
                  while ((sent = readdir(sdir))) {
                     if (!strncmp(sent->d_name, ".", 1) || !strncmp(sent->d_name, "..", 2))
                        continue;
                     // Get the pid
                     int ppid = XrdProofdAux::ParsePidPath(sent->d_name, rest, after);
                     // Add to the list
                     controlled.Add(ppid, ppid);
                  }
                  closedir(sdir);
                  sessionspaths.Add(sessiondir.c_str(), 0, 0, Hash_data_is_key);
               }
               ok = (controlled.Find(pid)) ? 1 : ok;
               // We are done, if the process is controlled
               if (ok) break;
            }
         }
         // If the process is not controlled we have to kill it
         if (!ok) {
            TRACE(ALL,"process: "<<pid<<" lost its controller: killing");
            if (XrdProofdAux::KillProcess(pid, 1, ui, fMgr->ChangeOwn()) == 0)
               nk++;
         }
      }

   }

   // Done
   return nk;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::CleanupProofServ(bool all, const char *usr)
{
   // Cleanup (kill) all 'proofserv' processes from the process table.
   // Only the processes associated with 'usr' are killed,
   // unless 'all' is TRUE, in which case all 'proofserv' instances are
   // terminated (this requires superuser privileges).
   // Super users can also terminated all processes fo another user (specified
   // via usr).
   // Return number of process notified for termination on success, -1 otherwise
   XPDLOC(SMGR, "ProofServMgr::CleanupProofServ")

   TRACE(REQ,  "all: "<<all<<", usr: " << (usr ? usr : "undef"));
   int nk = 0;

   // Name
   const char *pn = "proofserv";

   // Uid
   XrdProofUI ui;
   int refuid = -1;
   if (!all) {
      if (!usr) {
         TRACE(DBG, "usr must be defined for all = FALSE");
         return -1;
      }
      if (XrdProofdAux::GetUserInfo(usr, ui) != 0) {
         TRACE(DBG, "problems getting info for user " << usr);
         return -1;
      }
      refuid = ui.fUid;
   }

#if defined(linux)
   // Loop over the "/proc" dir
   DIR *dir = opendir("/proc");
   if (!dir) {
      XrdOucString emsg("cannot open /proc - errno: ");
      emsg += errno;
      TRACE(DBG, emsg.c_str());
      return -1;
   }

   struct dirent *ent = 0;
   while ((ent = readdir(dir))) {
      if (!strncmp(ent->d_name, ".", 1) || !strncmp(ent->d_name, "..", 2)) continue;
      if (DIGIT(ent->d_name[0])) {
         XrdOucString fn("/proc/", 256);
         fn += ent->d_name;
         fn += "/status";
         // Open file
         FILE *ffn = fopen(fn.c_str(), "r");
         if (!ffn) {
            XrdOucString emsg("cannot open file ");
            emsg += fn; emsg += " - errno: "; emsg += errno;
            TRACE(HDBG, emsg);
            continue;
         }
         // Read info
         bool xname = 1, xpid = 1, xppid = 1;
         bool xuid = (all) ? 0 : 1;
         int pid = -1;
         int ppid = -1;
         char line[2048] = { 0 };
         while (fgets(line, sizeof(line), ffn) &&
               (xname || xpid || xppid || xuid)) {
            // Check name
            if (xname && strstr(line, "Name:")) {
               if (!strstr(line, pn))
                  break;
               xname = 0;
            }
            if (xpid && strstr(line, "Pid:")) {
               pid = (int) XrdProofdAux::GetLong(&line[strlen("Pid:")]);
               xpid = 0;
            }
            if (xppid && strstr(line, "PPid:")) {
               ppid = (int) XrdProofdAux::GetLong(&line[strlen("PPid:")]);
               // Parent process must be us or be dead
               if (ppid != getpid() && XrdProofdAux::VerifyProcessByID(ppid, fParentExecs.c_str()))
                  // Process created by another running xrootd
                  break;
               xppid = 0;
            }
            if (xuid && strstr(line, "Uid:")) {
               int uid = (int) XrdProofdAux::GetLong(&line[strlen("Uid:")]);
               if (refuid == uid)
                  xuid = 0;
            }
         }
         // Close the file
         fclose(ffn);
         // If this is a good candidate, kill it
         if (!xname && !xpid && !xppid && !xuid) {

            bool muok = 1;
            if (fMgr->MultiUser() && !all) {
               // We need to check the user name: we may be the owner of somebody
               // else process; if not session is attached, we kill it
               muok = 0;
               XrdProofdProofServ *srv = GetActiveSession(pid);
               if (!srv || (srv && !strcmp(usr, srv->Client())))
                  muok = 1;
            }
            if (muok)
               if (XrdProofdAux::KillProcess(pid, 1, ui, fMgr->ChangeOwn()) == 0)
                  nk++;
         }
      }
   }
   // Close the directory
   closedir(dir);

#elif defined(__sun)

   // Loop over the "/proc" dir
   DIR *dir = opendir("/proc");
   if (!dir) {
      XrdOucString emsg("cannot open /proc - errno: ");
      emsg += errno;
      TRACE(DBG, emsg);
      return -1;
   }

   struct dirent *ent = 0;
   while ((ent = readdir(dir))) {
      if (!strncmp(ent->d_name, ".", 1) || !strncmp(ent->d_name, "..", 2)) continue;
      if (DIGIT(ent->d_name[0])) {
         XrdOucString fn("/proc/", 256);
         fn += ent->d_name;
         fn += "/psinfo";
         // Open file
         int ffd = open(fn.c_str(), O_RDONLY);
         if (ffd <= 0) {
            XrdOucString emsg("cannot open file ");
            emsg += fn; emsg += " - errno: "; emsg += errno;
            TRACE(HDBG, emsg);
            continue;
         }
         // Read info
         bool xname = 1;
         bool xuid = (all) ? 0 : 1;
         bool xppid = 1;
         // Get the information
         psinfo_t psi;
         if (read(ffd, &psi, sizeof(psinfo_t)) != sizeof(psinfo_t)) {
            XrdOucString emsg("cannot read ");
            emsg += fn; emsg += ": errno: "; emsg += errno;
            TRACE(XERR, emsg);
            close(ffd);
            continue;
         }
         // Close the file
         close(ffd);

         // Check name
         if (xname) {
            if (!strstr(psi.pr_fname, pn))
               continue;
            xname = 0;
         }
         // Check uid, if required
         if (xuid) {
            if (refuid == psi.pr_uid)
               xuid = 0;
         }
         // Parent process must be us or be dead
         int ppid = psi.pr_ppid;
         if (ppid != getpid() && XrdProofdAux::VerifyProcessByID(ppid, fParentExecs.c_str())) {
            // Process created by another running xrootd
            continue;
            xppid = 0;
         }

         // If this is a good candidate, kill it
         if (!xname && !xppid && !xuid) {
            bool muok = 1;
            if (fMgr->MultiUser() && !all) {
               // We need to check the user name: we may be the owner of somebody
               // else process; if no session is attached , we kill it
               muok = 0;
               XrdProofdProofServ *srv = GetActiveSession(psi.pr_pid);
               if (!srv || (srv && !strcmp(usr, srv->Client())))
                  muok = 1;
            }
            if (muok)
               if (XrdProofdAux::KillProcess(psi.pr_pid, 1, ui, fMgr->ChangeOwn()) == 0)
                  nk++;
         }
      }
   }
   // Close the directory
   closedir(dir);

#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)

   // Get the proclist
   kinfo_proc *pl = 0;
   int np;
   int ern = 0;
   if ((ern = XrdProofdAux::GetMacProcList(&pl, np)) != 0) {
      XrdOucString emsg("cannot get the process list: errno: ");
      emsg += ern;
      TRACE(XERR, emsg);
      return -1;
   }

   // Loop over the list
   int ii = np;
   while (ii--) {
      if (strstr(pl[ii].kp_proc.p_comm, pn)) {
         if (all || (int)(pl[ii].kp_eproc.e_ucred.cr_uid) == refuid) {
            // Parent process must be us or be dead
            int ppid = pl[ii].kp_eproc.e_ppid;
            bool xppid = 0;
            if (ppid != getpid()) {
               int jj = np;
               while (jj--) {
                  if (strstr(pl[jj].kp_proc.p_comm, "xrootd") &&
                      pl[jj].kp_proc.p_pid == ppid) {
                      xppid = 1;
                      break;
                  }
               }
            }
            if (!xppid) {
               bool muok = 1;
               if (fMgr->MultiUser() && !all) {
                  // We need to check the user name: we may be the owner of somebody
                  // else process; if no session is attached, we kill it
                  muok = 0;
                  XrdProofdProofServ *srv = GetActiveSession(pl[np].kp_proc.p_pid);
                  if (!srv || (srv && !strcmp(usr, srv->Client())))
                     muok = 1;
               }
               if (muok)
                  // Good candidate to be shot
                  if (XrdProofdAux::KillProcess(pl[np].kp_proc.p_pid, 1, ui, fMgr->ChangeOwn()))
                     nk++;
            }
         }
      }
   }
   // Cleanup
   free(pl);
#else
   // For the remaining cases we use 'ps' via popen to localize the processes

   // Build command
   XrdOucString cmd = "ps ";
   bool busr = 0;
   const char *cusr = (usr && strlen(usr) && fSuperUser) ? usr : fPClient->ID();
   if (all) {
      cmd += "ax";
   } else {
      cmd += "-U ";
      cmd += cusr;
      cmd += " -u ";
      cmd += cusr;
      cmd += " -f";
      busr = 1;
   }
   cmd += " | grep proofserv 2>/dev/null";

   // Our parent ID as a string
   char cpid[10];
   sprintf(cpid, "%d", getpid());

   // Run it ...
   XrdOucString pids = ":";
   FILE *fp = popen(cmd.c_str(), "r");
   if (fp != 0) {
      char line[2048] = { 0 };
      while (fgets(line, sizeof(line), fp)) {
         // Parse line: make sure that we are the parent
         char *px = strstr(line, "xpd");
         if (!px)
            // Not xpd: old proofd ?
            continue;
         char *pi = strstr(px+3, cpid);
         if (!pi) {
            // Not started by us: check if the parent is still running
            pi = px + 3;
            int ppid = (int) XrdProofdAux::GetLong(pi);
            TRACE(HDBG, "found alternative parent ID: "<< ppid);
            // If still running then skip
            if (XrdProofdAux::VerifyProcessByID(ppid, fParentExecs.c_str()))
               continue;
         }
         // Get pid now
         int from = 0;
         if (busr)
            from += strlen(cusr);
         int pid = (int) XrdProofdAux::GetLong(&line[from]);
         bool muok = 1;
         if (fMgr->MultiUser() && !all) {
            // We need to check the user name: we may be the owner of somebody
            // else process; if no session is attached, we kill it
            muok = 0;
            XrdProofdProofServ *srv = GetActiveSession(pid);
            if (!srv || (srv && !strcmp(usr, srv->Client())))
               muok = 1;
         }
         if (muok)
            // Kill it
            if (XrdProofdAux::KillProcess(pid, 1, ui, fMgr->ChangeOwn()) == 0)
               nk++;
      }
      pclose(fp);
   } else {
      // Error executing the command
      return -1;
   }
#endif

   // Done
   return nk;
}

//___________________________________________________________________________
int XrdProofdProofServMgr::SetUserOwnerships(XrdProofdProtocol *p,
                                             const char *ord, const char *stag)
{
   // Set user ownerships on some critical files or directories.
   // Return 0 on success, -1 if enything goes wrong.
   XPDLOC(SMGR, "ProofServMgr::SetUserOwnerships")

   TRACE(REQ, "enter");

   // If applicable, make sure that the private dataset dir for this user exists 
   // and has the right permissions
   if (fMgr->DataSetSrcs()->size() > 0) {
      XrdProofUI ui;
      XrdProofdAux::GetUserInfo(XrdProofdProtocol::EUidAtStartup(), ui);
      std::list<XrdProofdDSInfo *>::iterator ii;
      for (ii = fMgr->DataSetSrcs()->begin(); ii != fMgr->DataSetSrcs()->end(); ii++) {
         TRACE(ALL, "Checking dataset source: url:"<<(*ii)->fUrl<<", local:"
                                                   <<(*ii)->fLocal<<", rw:"<<(*ii)->fRW);
         if ((*ii)->fLocal && (*ii)->fRW) {
            XrdOucString d;
            XPDFORM(d, "%s/%s", ((*ii)->fUrl).c_str(), p->Client()->UI().fGroup.c_str());
            if (XrdProofdAux::AssertDir(d.c_str(), ui, fMgr->ChangeOwn()) == 0) {
               if (XrdProofdAux::ChangeMod(d.c_str(), 0777) == 0) {
                  XPDFORM(d, "%s/%s/%s", ((*ii)->fUrl).c_str(), p->Client()->UI().fGroup.c_str(),
                                                                p->Client()->UI().fUser.c_str());
                  if (XrdProofdAux::AssertDir(d.c_str(), p->Client()->UI(), fMgr->ChangeOwn()) == 0) {
                     if (XrdProofdAux::ChangeMod(d.c_str(), 0755) != 0) {
                        TRACE(XERR, "problems setting permissions 0755 on: "<<d);
                     }
                  } else {
                     TRACE(XERR, "problems asserting: "<<d);
                  }
               } else {
                  TRACE(XERR, "problems setting permissions 0777 on: "<<d);
               }
            } else {
               TRACE(XERR, "problems asserting: "<<d);
            }
         }
      }
   }

   // If applicable, make sure that the private data dir for this user exists 
   // and has the right permissions
   if (fMgr->DataDir() && strlen(fMgr->DataDir()) > 0 &&
       fMgr->DataDirOpts() && strlen(fMgr->DataDirOpts()) > 0 && ord && stag) {
      XrdProofUI ui;
      XrdProofdAux::GetUserInfo(XrdProofdProtocol::EUidAtStartup(), ui);
      XrdOucString dgr, dus[3];
      XPDFORM(dgr, "%s/%s", fMgr->DataDir(), p->Client()->UI().fGroup.c_str());
      if (XrdProofdAux::AssertDir(dgr.c_str(), ui, fMgr->ChangeOwn()) == 0) {
         if (XrdProofdAux::ChangeMod(dgr.c_str(), 0777) == 0) {
            unsigned int mode = 0755;
            if (strchr(fMgr->DataDirOpts(), 'g')) mode = 0775;
            if (strchr(fMgr->DataDirOpts(), 'a') || strchr(fMgr->DataDirOpts(), 'o')) mode = 0777;
            XPDFORM(dus[0], "%s/%s", dgr.c_str(), p->Client()->UI().fUser.c_str());
            XPDFORM(dus[1], "%s/%s", dus[0].c_str(), ord);
            XPDFORM(dus[2], "%s/%s", dus[1].c_str(), stag);
            for (int i = 0; i < 3; i++) {
               if (XrdProofdAux::AssertDir(dus[i].c_str(), p->Client()->UI(), fMgr->ChangeOwn()) == 0) {
                  if (XrdProofdAux::ChangeMod(dus[i].c_str(), mode) != 0) {
                     TRACE(XERR, "problems setting permissions "<< oct << mode<<" on: "<<dus[i]);
                  }
               } else {
                  TRACE(XERR, "problems asserting: "<<dus[i]);
                  break;
               }
            }
         } else {
            TRACE(XERR, "problems setting permissions 0777 on: "<<dgr);
         }
      } else {
         TRACE(XERR, "problems asserting: "<<dgr);
      }
   }

   if (fMgr->ChangeOwn()) {
      // Change ownership of '.creds'
      XrdOucString creds(p->Client()->Sandbox()->Dir());
      creds += "/.creds";
      if (XrdProofdAux::ChangeOwn(creds.c_str(), p->Client()->UI()) != 0) {
         TRACE(XERR, "can't change ownership of "<<creds);
         return -1;
      }
   }

   // We are done
   TRACE(REQ, "done");
   return 0;
}

//___________________________________________________________________________
int XrdProofdProofServMgr::SetUserEnvironment(XrdProofdProtocol *p)
{
   // Set user environment: set effective user and group ID of the process
   // to the ones of the owner of this protocol instnace and change working
   // dir to the sandbox.
   // Return 0 on success, -1 if enything goes wrong.
   XPDLOC(SMGR, "ProofServMgr::SetUserEnvironment")

   TRACE(REQ, "enter");

   if (XrdProofdAux::ChangeToDir(p->Client()->Sandbox()->Dir(),
                                 p->Client()->UI(), fMgr->ChangeOwn()) != 0) {
      TRACE(XERR, "couldn't change directory to "<< p->Client()->Sandbox()->Dir());
      return -1;
   }

   // set HOME env
   char *h = new char[8 + strlen(p->Client()->Sandbox()->Dir())];
   sprintf(h, "HOME=%s", p->Client()->Sandbox()->Dir());
   putenv(h);
   TRACE(DBG, "set "<<h);

   // set USER env
   char *u = new char[8 + strlen(p->Client()->User())];
   sprintf(u, "USER=%s", p->Client()->User());
   putenv(u);
   TRACE(DBG, "set "<<u);

   // Set access control list from /etc/initgroup
   // (super-user privileges required)
   TRACE(DBG, "setting ACLs");
   if (fMgr->ChangeOwn() && (int) geteuid() != p->Client()->UI().fUid) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, p->Client()->UI().fUid)) {
         TRACE(XERR, "could not get privileges");
         return -1;
      }

      initgroups(p->Client()->UI().fUser.c_str(), p->Client()->UI().fGid);
   }

   if (fMgr->ChangeOwn()) {
      // acquire permanently target user privileges
      TRACE(DBG, "acquiring target user identity: "<<(uid_t)p->Client()->UI().fUid<<
                                               ", "<<(gid_t)p->Client()->UI().fGid);
      if (XrdSysPriv::ChangePerm((uid_t)p->Client()->UI().fUid,
                                 (gid_t)p->Client()->UI().fGid) != 0) {
         TRACE(XERR, "can't acquire "<< p->Client()->UI().fUser <<" identity");
         return -1;
      }
   }

   // We are done
   TRACE(REQ, "done");
   return 0;
}

//______________________________________________________________________________
int XrdProofdProofServMgr::SaveAFSkey(XrdSecCredentials *c,
                                      const char *dir, XrdProofUI ui)
{
   // Save the AFS key, if any, for usage in proofserv in file 'dir'/.afs .
   // Return 0 on success, -1 on error.
   XPDLOC(SMGR, "ProofServMgr::SaveAFSkey")

   // Check file name
   if (!dir || strlen(dir) <= 0) {
      TRACE(XERR, "dir name undefined");
      return -1;
   }

   // Check credentials
   if (!c) {
      TRACE(XERR, "credentials undefined");
      return -1;
   }
   TRACE(REQ, "dir: "<<dir);

   // Decode credentials
   int lout = 0;
   char *out = new char[c->size];
   if (XrdSutFromHex(c->buffer, out, lout) != 0) {
      TRACE(XERR, "problems unparsing hex string");
      delete [] out;
      return -1;
   }

   // Locate the key
   char *key = out + 5;
   if (strncmp(key, "afs:", 4)) {
      TRACE(DBG, "string does not contain an AFS key");
      delete [] out;
      return 0;
   }
   key += 4;

   // Save to file, if not existing already
   XrdOucString fn = dir;
   fn += "/.afs";

   int rc = 0;
   struct stat st;
   if (stat(fn.c_str(), &st) != 0 && errno == ENOENT) {

      // Open the file, truncating if already existing
      int fd = open(fn.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
      if (fd <= 0) {
         TRACE(XERR, "problems creating file - errno: " << errno);
         delete [] out;
         return -1;
      }
      // Write out the key
      int lkey = lout - 9;
      if (XrdProofdAux::Write(fd, key, lkey) != lkey) {
         TRACE(XERR, "problems writing to file - errno: " << errno);
         rc = -1;
      }

      // Cleanup
      delete [] out;
      close(fd);
   } else {
      TRACE(XERR, "cannot stat existing file "<<fn<<" - errno: " << errno);
      delete [] out;
      return -1;
   }

   // Make sure the file is owned by the user
   if (XrdProofdAux::ChangeOwn(fn.c_str(), ui) != 0) {
      TRACE(XERR, "can't change ownership of "<<fn);
   }

   return rc;
}

//__________________________________________________________________________
XrdProofdProofServ *XrdProofdProofServMgr::GetActiveSession(int pid)
{
   // Return active session with process ID pid, if any.

   XrdOucString key; key += pid;
   return fSessions.Find(key.c_str());
}

//__________________________________________________________________________
static int BroadcastPriority(const char *, XrdProofdProofServ *ps, void *s)
{
   // Run thorugh entries to broadcast the relevant priority
   XPDLOC(SMGR, "BroadcastPriority")

   XpdBroadcastPriority_t *bp = (XpdBroadcastPriority_t *)s;

   int nb = *(bp->fNBroadcast);

   XrdOucString emsg;
   if (ps) {
      if (ps->IsValid() && (ps->Status() == kXPD_running) &&
        !(ps->SrvType() == kXPD_Master)) {
         XrdProofGroup *g = (ps->Group() && bp->fGroupMgr)
                          ? bp->fGroupMgr->GetGroup(ps->Group()) : 0;
         TRACE(DBG, "group: "<<  g<<", client: "<<ps->Client());
         if (g && g->Active() > 0) {
            TRACE(DBG, "priority: "<< g->Priority()<<" active: "<<g->Active());
            int prio = (int) (g->Priority() * 100);
            ps->BroadcastPriority(prio);
            nb++;
         }
      }
      // Go to next
      return 0;
   } else {
      emsg = "input entry undefined";
   }

   // Some problem
   TRACE(XERR,"protocol error: "<<emsg);
   return 1;
}

//__________________________________________________________________________
void XrdProofdProofServMgr::BroadcastClusterInfo()
{
   // Broadcast cluster info to the active sessions
   XPDLOC(SMGR, "ProofServMgr::BroadcastClusterInfo")

   TRACE(REQ, "enter");

   int tot = 0, act = 0;
   std::list<XrdProofdProofServ *>::iterator si = fActiveSessions.begin();
   while (si != fActiveSessions.end()) {
      if ((*si)->SrvType() != kXPD_Worker) {
         tot++;
         if ((*si)->Status() == kXPD_running) act++;
      }
      si++;
   }
   XPDPRT("tot: "<<tot<<", act: "<<act);
   // Now propagate
   si = fActiveSessions.begin();
   while (si != fActiveSessions.end()) {
      if ((*si)->Status() == kXPD_running) (*si)->SendClusterInfo(tot, act);
      si++;
   }
}

//__________________________________________________________________________
int XrdProofdProofServMgr::BroadcastPriorities()
{
   // Broadcast priorities to the active sessions.
   // Returns the number of sessions contacted.
   XPDLOC(SMGR, "ProofServMgr::BroadcastPriorities")

   TRACE(REQ, "enter");

   int nb = 0;
   XpdBroadcastPriority_t bp = { (fMgr ? fMgr->GroupsMgr() : 0), &nb };
   fSessions.Apply(BroadcastPriority, (void *)&bp);
   // Done
   return nb;
}

//__________________________________________________________________________
bool XrdProofdProofServMgr::IsReconnecting()
{
   // Return true if in reconnection state, i.e. during
   // that period during which clients are expected to reconnect.
   // Return false if the session is fully effective

   int rect = -1;
   if (fReconnectTime >= 0) {
      rect = time(0) - fReconnectTime;
      if (rect < fReconnectTimeOut)
         return true;
   }
   // Not reconnecting
   return false;
}

//__________________________________________________________________________
void XrdProofdProofServMgr::SetReconnectTime(bool on)
{
   // Change reconnecting status
   //

   XrdSysMutexHelper mhp(fMutex);

   if (on) {
      fReconnectTime = time(0);
   } else {
      fReconnectTime = -1;
   }
}

//__________________________________________________________________________
static int FreeClientID(const char *, XrdProofdProofServ *ps, void *s)
{
   // Run through entries to reset the disconnecting client slots
   XPDLOC(SMGR, "FreeClientID")

   int pid = *((int *)s);

   if (ps) {
      ps->FreeClientID(pid);
      // Go to next
      return 0;
   }

   // Some problem
   TRACE(XERR, "protocol error: undefined session!");
   return 1;
}

//__________________________________________________________________________
void XrdProofdProofServMgr::DisconnectFromProofServ(int pid)
{
   // Change reconnecting status
   //

   XrdSysMutexHelper mhp(fMutex);

   fSessions.Apply(FreeClientID, (void *)&pid);
}

//__________________________________________________________________________
static int CountTopMasters(const char *, XrdProofdProofServ *ps, void *s)
{
   // Run thorugh entries to count top-masters
   XPDLOC(SMGR, "CountTopMasters")

   int *ntm = (int *)s;

   XrdOucString emsg;
   if (ps) {
      if (ps->SrvType() == kXPD_TopMaster) (*ntm)++;
      // Go to next
      return 0;
   } else {
      emsg = "input entry undefined";
   }

   // Some problem
   TRACE(XERR,"protocol error: "<<emsg);
   return 1;
}

//__________________________________________________________________________
int XrdProofdProofServMgr::CurrentSessions(bool recalculate)
{
   // Return the number of current sessions (top masters)

   XPDLOC(SMGR, "ProofServMgr::CurrentSessions")

   TRACE(REQ, "enter");

   XrdSysMutexHelper mhp(fMutex);
   if (recalculate) {
      fCurrentSessions = 0;
      fSessions.Apply(CountTopMasters, (void *)&fCurrentSessions);
   }

   // Done
   return fCurrentSessions;
}

//__________________________________________________________________________
void XrdProofdProofServMgr::ResolveKeywords(XrdOucString &s, ProofServEnv_t *in)
{
   // Resolve some keywords in 's'
   //    <logfileroot>, <user>, <rootsys>

   if (!in) return;

   bool isWorker = 0;
   if (in->fPS->SrvType() == kXPD_Worker) isWorker = 1;

   // Log file
   if (!isWorker && s.find("<logfilemst>") != STR_NPOS) {
      XrdOucString lfr(in->fLogFile);
      if (lfr.endswith(".log")) lfr.erase(lfr.rfind(".log"));
      s.replace("<logfilemst>", lfr);
   } else if (isWorker && s.find("<logfilewrk>") != STR_NPOS) {
      XrdOucString lfr(in->fLogFile);
      if (lfr.endswith(".log")) lfr.erase(lfr.rfind(".log"));
      s.replace("<logfilewrk>", lfr);
   }

   // user
   if (getenv("USER") && s.find("<user>") != STR_NPOS) {
      XrdOucString usr(getenv("USER"));
      s.replace("<user>", usr);
   }

   // rootsys
   if (getenv("ROOTSYS") && s.find("<rootsys>") != STR_NPOS) {
      XrdOucString rootsys(getenv("ROOTSYS"));
      s.replace("<rootsys>", rootsys);
   }
}

//
// Auxilliary class to handle session pid files
//

//______________________________________________________________________________
XrdProofSessionInfo::XrdProofSessionInfo(XrdProofdClient *c, XrdProofdProofServ *s)
{
   // Construct from 'c' and 's'

   fLastAccess = 0;

   // Fill from the client instance
   fUser = c ? c->User() : "";
   fGroup = c ? c->Group() : "";

   // Fill from the server instance
   fPid = s ? s->SrvPID() : -1;
   fID = s ? s->ID() : -1;
   fSrvType = s ? s->SrvType() : -1;
   fStatus = s ? s->Status() : kXPD_unknown;
   fOrdinal = s ? s->Ordinal() : "";
   fTag = s ? s->Tag() : "";
   fAlias = s ? s->Alias() : "";
   fLogFile = s ? s->Fileout() : "";
   fROOTTag = (s && s->ROOT())? s->ROOT()->Tag() : "";
   fSrvProtVers = (s && s->ROOT()) ? s->ROOT()->SrvProtVers() : -1;
   fUserEnvs = s ? s->UserEnvs() : "";
   fAdminPath = s ? s->AdminPath() : "";
   fUnixPath = s ? s->UNIXSockPath() : "";
}

//______________________________________________________________________________
void XrdProofSessionInfo::FillProofServ(XrdProofdProofServ &s, XrdROOTMgr *rmgr)
{
   // Fill 's' fields using the stored info
   XPDLOC(SMGR, "SessionInfo::FillProofServ")

   s.SetClient(fUser.c_str());
   s.SetGroup(fGroup.c_str());
   if (fPid > 0)
      s.SetSrvPID(fPid);
   if (fID >= 0)
      s.SetID(fID);
   s.SetSrvType(fSrvType);
   s.SetStatus(fStatus);
   s.SetOrdinal(fOrdinal.c_str());
   s.SetTag(fTag.c_str());
   s.SetAlias(fAlias.c_str());
   s.SetFileout(fLogFile.c_str());
   if (rmgr) {
      if (rmgr->GetVersion(fROOTTag.c_str())) {
         s.SetROOT(rmgr->GetVersion(fROOTTag.c_str()));
      } else {
         TRACE(ALL, "ROOT version '"<< fROOTTag <<
                    "' not availabe anymore: setting the default");
         s.SetROOT(rmgr->DefaultVersion());
      }
   }
   s.SetUserEnvs(fUserEnvs.c_str());
   s.SetAdminPath(fAdminPath.c_str(), 0);
   s.SetUNIXSockPath(fUnixPath.c_str());
}

//______________________________________________________________________________
int XrdProofSessionInfo::SaveToFile(const char *file)
{
   // Save content to 'file'
   XPDLOC(SMGR, "SessionInfo::SaveToFile")

   // Check inputs
   if (!file || strlen(file) <= 0) {
      TRACE(XERR,"invalid input: "<<file);
      return -1;
   }

   // Create the file
   FILE *fpid = fopen(file, "w");
   if (fpid) {
      fprintf(fpid, "%s %s\n", fUser.c_str(), fGroup.c_str());
      fprintf(fpid, "%s\n", fUnixPath.c_str());
      fprintf(fpid, "%d %d %d\n", fPid, fID, fSrvType);
      fprintf(fpid, "%s %s %s\n", fOrdinal.c_str(), fTag.c_str(), fAlias.c_str());
      fprintf(fpid, "%s\n", fLogFile.c_str());
      fprintf(fpid, "%d %s\n", fSrvProtVers, fROOTTag.c_str());
      if (fUserEnvs.length() > 0)
         fprintf(fpid, "\n%s", fUserEnvs.c_str());
      fclose(fpid);

      // Make it writable by anyone (to allow the corresponding proofserv
      // to touch it for the asynchronous ping request)
      if (chmod(file, 0666) != 0) {
         TRACE(XERR, "could not change mode to 0666 on file "<<
                     file<<"; error: "<<errno);
      }

      return 0;
   }

   TRACE(XERR,"session pid file cannot be (re-)created: "<<
              file<<"; error: "<<errno);
   return -1;
}

//______________________________________________________________________________
void XrdProofSessionInfo::Reset()
{
   // Reset the content

   fLastAccess = 0;
   fUser = "";
   fGroup = "";
   fAdminPath = "";
   fUnixPath = "";
   fPid = -1;
   fStatus = kXPD_unknown;
   fID = -1;
   fSrvType = -1;
   fOrdinal = "";
   fTag = "";
   fAlias = "";
   fLogFile = "";
   fROOTTag = "";
   fSrvProtVers = -1;
   fUserEnvs = "";
}

//______________________________________________________________________________
int XrdProofSessionInfo::ReadFromFile(const char *file)
{
   // Read content from 'file'
   XPDLOC(SMGR, "SessionInfo::ReadFromFile")

   // Check inputs
   if (!file || strlen(file) <= 0) {
      TRACE(XERR,"invalid input: "<<file);
      return -1;
   }

   Reset();

   // Open the session file
   FILE *fpid = fopen(file,"r");
   if (fpid) {
      char line[4096];
      char v1[512], v2[512], v3[512];
      if (fgets(line, sizeof(line), fpid)) {
         if (sscanf(line, "%s %s", v1, v2) == 2) {
            fUser = v1;
            fGroup = v2;
         } else {
            TRACE(XERR,"warning: corrupted line? "<<line);
         }
      }
      if (fgets(line, sizeof(line), fpid)) {
         int l = strlen(line);
         if (line[l-1] == '\n') line[l-1] = '\0';
         fUnixPath = line;
      }
      if (fgets(line, sizeof(line), fpid)) {
         sscanf(line, "%d %d %d", &fPid, &fID, &fSrvType);
      }
      if (fgets(line, sizeof(line), fpid)) {
         int ns = 0;
         if ((ns = sscanf(line, "%s %s %s", v1, v2, v3)) >= 2) {
            fOrdinal = v1;
            fTag = v2;
            fAlias = (ns == 3) ? v3 : "";
         } else {
            TRACE(XERR,"warning: corrupted line? "<<line);
         }
      }
      if (fgets(line, sizeof(line), fpid)) {
         fLogFile = line;
      }
      if (fgets(line, sizeof(line), fpid)) {
         if (sscanf(line, "%d %s", &fSrvProtVers, v1) == 2) {
            fROOTTag = v1;
         } else {
            TRACE(XERR,"warning: corrupted line? "<<line);
         }
      }
      // All the remaining into fUserEnvs
      fUserEnvs = "";
      off_t lnow = lseek(fileno(fpid), (off_t) 0, SEEK_CUR);
      off_t ltot = lseek(fileno(fpid), (off_t) 0, SEEK_END);
      int left = (int)(ltot - lnow);
      int len = -1;
      do {
         int wanted = (left > 4095) ? 4095 : left;
         while ((len = read(fileno(fpid), line, wanted)) < 0 &&
                errno == EINTR)
            errno = 0;
         if (len < wanted) {
            break;
         } else {
            line[len] = '\0';
            fUserEnvs += line;
         }
         // Update counters
         left -= len;
      } while (len > 0 && left > 0);
      // Done
      fclose(fpid);
      // The file name is the admin path
      fAdminPath = file;
      // Fill access time
      struct stat st;
      if (!stat(file, &st))
         fLastAccess = st.st_atime;
   } else {
      TRACE(XERR,"session file cannot be open: "<< file<<"; error: "<<errno);
      return -1;
   }

   // Read the last status now if the session is active
   XrdOucString fs(file);
   fs += ".status";
   fpid = fopen(fs.c_str(),"r");
   if (fpid) {
      char line[64];
      if (fgets(line, sizeof(line), fpid)) {
         sscanf(line, "%d", &fStatus);
      }
      // Done
      fclose(fpid);
   } else {
      TRACE(DBG,"no session status file for: "<< fs<<"; session was probably terminated");
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XpdEnv::Matches(const char *usr, const char *grp, int svn, int ver)
{
   // Check if this env applies to 'usr', 'grp, 'svn', 'ver'.
   // Returns -1 if it does not match, >=0 if it matches. The value is a linear
   // combination of matching lengths for user and group, with a weight of 1000 for
   // the users one, so that an exact user match will always win.
   XPDLOC(SMGR, "XpdEnv::Matches")

   int nmtc = -1;
   // Check the user
   if (fUsers.length() > 0) {
      XrdOucString u(usr);
      if ((nmtc = u.matches(fUsers.c_str())) == 0) return -1;
   } else {
      nmtc = strlen(usr);
   }
   nmtc += 1000;   // Weigth of user name match
   // Check the group
   int nmtcg = -1;
   if (fGroups.length() > 0) {
      XrdOucString g(grp);
      if ((nmtcg = g.matches(fGroups.c_str())) == 0) return -1;
   } else {
      nmtcg = strlen(grp);
   }
   nmtc += nmtcg;

   TRACE(HDBG, fEnv <<", u:"<<usr<<", g:"<<grp<<" --> nmtc: "<<nmtc);

   // Check the subversion number
   TRACE(HDBG, fEnv <<", svn:"<<svn);
   if (fSvnMin > 0 && svn < fSvnMin) return -1; 
   if (fSvnMax > 0 && svn > fSvnMax) return -1; 

   // Check the version code
   TRACE(HDBG, fEnv <<", ver:"<<ver);
   if (fVerMin > 0 && ver < fVerMin) return -1; 
   if (fVerMax > 0 && ver > fVerMax) return -1; 
   
   // If we are here then it matches
   return nmtc;
}

//______________________________________________________________________________
int XpdEnv::ToVersCode(int ver, bool hex)
{
   // Transform version number ver (format patch + 100*minor + 10000*maj, e.g. 52706)
   // If 'hex' is true, the components are decoded as hex numbers
   
   int maj = -1, min = -1, ptc = -1, xv = ver;
   if (hex) {
      maj = xv / 65536;
      xv -= maj * 65536;
      min = xv / 256;
      ptc = xv - min * 256;
   } else {
      maj = xv / 10000;
      xv -= maj * 10000;
      min = xv / 100;
      ptc = xv - min * 100;
   }
   // Get the version code now
   int vc = (maj << 16) + (min << 8) + ptc;
   return vc;
}

//______________________________________________________________________________
void XpdEnv::Print(const char *what)
{
   // Print the content of this env
   XPDLOC(SMGR, what)
   
   XrdOucString vmi("-1"), vmx("-1");
   if (fVerMin > 0) {
      int maj = (fVerMin >> 16);
      int min = ((fVerMin - maj * 65536) >> 8);
      int ptc = fVerMin - maj * 65536 - min * 256;
      XPDFORM(vmi, "%d%d%d", maj, min, ptc);
   }
   if (fVerMax > 0) {
      int maj = (fVerMax >> 16);
      int min = ((fVerMax - maj * 65536) >> 8);
      int ptc = fVerMax - maj * 65536 - min * 256;
      XPDFORM(vmx, "%d%d%d", maj, min, ptc);
   }
   XrdOucString u("allusers"), g("allgroups");
   if (fUsers.length() > 0) u = fUsers;
   if (fGroups.length() > 0) u = fGroups;

   TRACE(ALL, "'"<<fEnv<<"' {"<<u<<"|"<<g<<
         "} svn:["<<fSvnMin<<","<<fSvnMax<<"] vers:["<<vmi<<","<<vmx<<"]");
}
