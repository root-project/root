// @(#)root/proofd:$Id$
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include <sys/stat.h>

#include "XrdNet/XrdNet.hh"

#include "XrdProofdAux.h"
#include "XrdProofdProofServ.h"
#include "XrdProofWorker.h"
#include "XrdProofSched.h"
#include "XrdProofdManager.h"

// Tracing utils
#include "XrdProofdTrace.h"

////////////////////////////////////////////////////////////////////////////////
/// Constructor

XrdProofdProofServ::XrdProofdProofServ()
{
   fMutex = new XrdSysRecMutex;
   fResponse = 0;
   fProtocol = 0;
   fParent = 0;
   fPingSem = 0;
   fStartMsg = 0;
   fStatus = kXPD_idle;
   fSrvPID = -1;
   fSrvType = kXPD_AnyServer;
   fPLiteNWrks = -1;
   fID = -1;
   fIsShutdown = false;
   fIsValid = true;  // It is created for a valid server ...
   fSkipCheck = false;
   fProtVer = -1;
   fNClients = 0;
   fClients.reserve(10);
   fDisconnectTime = -1;
   fSetIdleTime = time(0);
   fROOT = 0;
   // Strings
   fAdminPath = "";
   fAlias = "";
   fClient = "";
   fFileout = "";
   fGroup = "";
   fOrdinal = "";
   fTag = "";
   fUserEnvs = "";
   fUNIXSock = 0;
   fUNIXSockPath = "";
   fQueries.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

XrdProofdProofServ::~XrdProofdProofServ()
{
   SafeDel(fStartMsg);
   SafeDel(fPingSem);

   std::vector<XrdClientID *>::iterator i;
   for (i = fClients.begin(); i != fClients.end(); ++i)
       if (*i)
          delete (*i);
   fClients.clear();

   // Cleanup worker info
   ClearWorkers();

   // Cleanup queries info
   fQueries.clear();

   // Remove the associated UNIX socket path
   unlink(fUNIXSockPath.c_str());

   SafeDel(fMutex);
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease active session counters on worker w

static int DecreaseWorkerCounters(const char *, XrdProofWorker *w, void *x)
{
   XPDLOC(PMGR, "DecreaseWorkerCounters")

   XrdProofdProofServ *xps = (XrdProofdProofServ *)x;

   if (w && xps) {
      w->RemoveProofServ(xps);
      TRACE(REQ, w->fHost.c_str() <<" done");
      // Check next
      return 0;
   }

   // Not enough info: stop
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease active session counters on worker w

static int DumpWorkerCounters(const char *k, XrdProofWorker *w, void *)
{
   XPDLOC(PMGR, "DumpWorkerCounters")

   if (w) {
      TRACE(ALL, k <<" : "<<w->fHost.c_str()<<":"<<w->fPort <<" act: "<<w->Active());
      // Check next
      return 0;
   }

   // Not enough info: stop
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease worker counters and clean-up the list

void XrdProofdProofServ::ClearWorkers()
{
   XrdSysMutexHelper mhp(fMutex);

   // Decrease workers' counters and remove this from workers
   fWorkers.Apply(DecreaseWorkerCounters, this);
   fWorkers.Purge();
}

////////////////////////////////////////////////////////////////////////////////
/// Add a worker assigned to this session with label 'o'

void XrdProofdProofServ::AddWorker(const char *o, XrdProofWorker *w)
{
   if (!o || !w) return;

   XrdSysMutexHelper mhp(fMutex);

   fWorkers.Add(o, w, 0, Hash_keepdata);
}

////////////////////////////////////////////////////////////////////////////////
/// Release worker assigned to this session with label 'o'

void XrdProofdProofServ::RemoveWorker(const char *o)
{
   XPDLOC(SMGR, "ProofServ::RemoveWorker")

   if (!o) return;

   TRACE(DBG,"removing: "<<o);

   XrdSysMutexHelper mhp(fMutex);

   XrdProofWorker *w = fWorkers.Find(o);
   if (w) w->RemoveProofServ(this);
   fWorkers.Del(o);
   if (TRACING(HDBG)) fWorkers.Apply(DumpWorkerCounters, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Reset this instance, broadcasting a message to the clients.
/// return 1 if top master, 0 otherwise

int XrdProofdProofServ::Reset(const char *msg, int type)
{
   XPDLOC(SMGR, "ProofServ::Reset")

   int rc = 0;
   // Read the status file
   int st = -1;
   XrdOucString fn;
   XPDFORM(fn, "%s.status", fAdminPath.c_str());
   FILE *fpid = fopen(fn.c_str(), "r");
   if (fpid) {
      char line[64];
      if (fgets(line, sizeof(line), fpid)) {
         if (line[strlen(line)-1] == '\n') line[strlen(line)-1] = 0;
         st = atoi(line);
      } else {
         TRACE(XERR,"problems reading from file "<<fn);
      }
      fclose(fpid);
   }
   TRACE(DBG,"file: "<<fn<<", st:"<<st);
   XrdSysMutexHelper mhp(fMutex);
   // Broadcast msg
   if (st == 4) {
      Broadcast("idle-timeout", type);
   } else {
      Broadcast(msg, type);
   }
   // What kind of server is this?
   if (fSrvType == kXPD_TopMaster) rc = 1;
   // Reset instance
   Reset();
   // Done
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset this instance

void XrdProofdProofServ::Reset()
{
   XrdSysMutexHelper mhp(fMutex);

   fResponse = 0;
   fProtocol = 0;
   fParent = 0;
   SafeDel(fStartMsg);
   SafeDel(fPingSem);
   fSrvPID = -1;
   fID = -1;
   fIsShutdown = false;
   fIsValid = false;
   fSkipCheck = false;
   fProtVer = -1;
   fNClients = 0;
   fClients.clear();
   fDisconnectTime = -1;
   fSetIdleTime = -1;
   fROOT = 0;
   // Cleanup worker info
   ClearWorkers();
   // ClearWorkers depends on the fSrvType and fStatus
   fSrvType = kXPD_AnyServer;
   fPLiteNWrks = -1;
   fStatus = kXPD_idle;
   // Cleanup queries info
   fQueries.clear();
   // Strings
   fAdminPath = "";
   fAlias = "";
   fClient = "";
   fFileout = "";
   fGroup = "";
   fOrdinal = "";
   fTag = "";
   fUserEnvs = "";
   DeleteUNIXSock();
}

////////////////////////////////////////////////////////////////////////////////
/// Delete the current UNIX socket

void XrdProofdProofServ::DeleteUNIXSock()
{
   SafeDel(fUNIXSock);
   unlink(fUNIXSockPath.c_str());
   fUNIXSockPath = "";
}

////////////////////////////////////////////////////////////////////////////////
/// Return the value of fSkipCheck and reset it to false

bool XrdProofdProofServ::SkipCheck()
{
   XrdSysMutexHelper mhp(fMutex);

   bool rc = fSkipCheck;
   fSkipCheck = false;
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Get instance corresponding to cid

XrdClientID *XrdProofdProofServ::GetClientID(int cid)
{
   XPDLOC(SMGR, "ProofServ::GetClientID")

   XrdClientID *csid = 0;

   if (cid < 0) {
      TRACE(XERR, "negative ID: protocol error!");
      return csid;
   }

   XrdOucString msg;
   {  XrdSysMutexHelper mhp(fMutex);

      // Count new attached client
      fNClients++;

      // If in the allocate range reset the corresponding instance and
      // return it
      if (cid < (int)fClients.size()) {
         csid = fClients.at(cid);
         csid->Reset();

         // Notification message
         if (TRACING(DBG)) {
            XPDFORM(msg, "cid: %d, size: %d", cid, fClients.size());
         }
      }

      if (!csid) {
         // If not, allocate a new one; we need to resize (double it)
         if (cid >= (int)fClients.capacity())
            fClients.reserve(2*fClients.capacity());

         // Allocate new elements (for fast access we need all of them)
         int ic = (int)fClients.size();
         for (; ic <= cid; ic++)
            fClients.push_back((csid = new XrdClientID()));

         // Notification message
         if (TRACING(DBG)) {
            XPDFORM(msg, "cid: %d, new size: %d", cid, fClients.size());
         }
      }
   }
   TRACE(DBG, msg);

   // We are done
   return csid;
}

////////////////////////////////////////////////////////////////////////////////
/// Free instance corresponding to protocol connecting process 'pid'

int XrdProofdProofServ::FreeClientID(int pid)
{
   XPDLOC(SMGR, "ProofServ::FreeClientID")

   TRACE(DBG, "svrPID: "<<fSrvPID<< ", pid: "<<pid<<", session status: "<<
              fStatus<<", # clients: "<< fNClients);
   int rc = -1;
   if (pid <= 0) {
      TRACE(XERR, "undefined pid!");
      return rc;
   }
   if (!IsValid()) return rc;

   {  XrdSysMutexHelper mhp(fMutex);

      // Remove this from the list of clients
      std::vector<XrdClientID *>::iterator i;
      for (i = fClients.begin(); i != fClients.end(); ++i) {
         if ((*i) && (*i)->P()) {
            if ((*i)->P()->Pid() == pid || (*i)->P()->Pid() == -1) {
               if (fProtocol == (*i)->P()) {
                  SetProtocol(0);
                  SetConnection(0);
               }
               (*i)->Reset();
               if (fParent == (*i)) SetParent(0);
               fNClients--;
               // Record time of last disconnection
               if (fNClients <= 0)
                  fDisconnectTime = time(0);
               rc = 0;
               break;
            }
         }
      }
   }
   if (TRACING(REQ) && (rc == 0)) {
      int spid = SrvPID();
      TRACE(REQ, spid<<": slot for client pid: "<<pid<<" has been reset");
   }

   // Out of range
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Get the number of connected clients. If check is true check that
/// they are still valid ones and free the slots for the invalid ones

int XrdProofdProofServ::GetNClients(bool check)
{
   XrdSysMutexHelper mhp(fMutex);

   if (check) {
      fNClients = 0;
      // Remove this from the list of clients
      std::vector<XrdClientID *>::iterator i;
      for (i = fClients.begin(); i != fClients.end(); ++i) {
         if ((*i) && (*i)->P() && (*i)->P()->Link()) fNClients++;
      }
   }

   // Done
   return fNClients;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the time (in secs) all clients have been disconnected.
/// Return -1 if the session is running

int XrdProofdProofServ::DisconnectTime()
{
   XrdSysMutexHelper mhp(fMutex);

   int disct = -1;
   if (fDisconnectTime > 0)
      disct = time(0) - fDisconnectTime;
   return ((disct > 0) ? disct : -1);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the time (in secs) the session has been idle.
/// Return -1 if the session is running

int XrdProofdProofServ::IdleTime()
{
   XrdSysMutexHelper mhp(fMutex);

   int idlet = -1;
   if (fStatus == kXPD_idle)
      idlet = time(0) - fSetIdleTime;
   return ((idlet > 0) ? idlet : -1);
}

////////////////////////////////////////////////////////////////////////////////
/// Set status to idle and update the related time stamp
///

void XrdProofdProofServ::SetIdle()
{
   XrdSysMutexHelper mhp(fMutex);

   fStatus = kXPD_idle;
   fSetIdleTime = time(0);
}

////////////////////////////////////////////////////////////////////////////////
/// Set status to running and reset the related time stamp
///

void XrdProofdProofServ::SetRunning()
{
   XrdSysMutexHelper mhp(fMutex);

   fStatus = kXPD_running;
   fSetIdleTime = -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Broadcast message 'msg' at 'type' to the attached clients

void XrdProofdProofServ::Broadcast(const char *msg, int type)
{
   XPDLOC(SMGR, "ProofServ::Broadcast")

   // Backward-compatibility check
   int clproto = (type >= kXPD_wrkmortem) ? 18 : -1;

   XrdOucString m;
   int len = 0, nc = 0;
   if (msg && (len = strlen(msg)) > 0) {
      XrdProofdProtocol *p = 0;
      int ic = 0, ncz = 0, sid = -1;
      { XrdSysMutexHelper mhp(fMutex); ncz = (int) fClients.size(); }
      for (ic = 0; ic < ncz; ic++) {
         {  XrdSysMutexHelper mhp(fMutex);
            p = fClients.at(ic)->P();
            sid = fClients.at(ic)->Sid(); }
         // Send message
         if (p && XPD_CLNT_VERSION_OK(p, clproto)) {
            XrdProofdResponse *response = p->Response(sid);
            if (response) {
               response->Send(kXR_attn, (XProofActionCode)type, (void *)msg, len);
               nc++;
            } else {
               XPDFORM(m, "response instance for sid: %d not found", sid);
            }
         }
         if (m.length() > 0)
            TRACE(XERR, m);
         m = "";
      }
   }
   if (TRACING(DBG)) {
      XPDFORM(m, "type: %d, message: '%s' notified to %d clients", type, msg, nc);
      XPDPRT(m);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Terminate the associated process.
/// A shutdown interrupt message is forwarded.
/// If add is TRUE (default) the pid is added to the list of processes
/// requested to terminate.
/// Return the pid of tyhe terminated process on success, -1 if not allowed
/// or other errors occured.

int XrdProofdProofServ::TerminateProofServ(bool changeown)
{
   XPDLOC(SMGR, "ProofServ::TerminateProofServ")

   int pid = fSrvPID;
   TRACE(DBG, "ord: " << fOrdinal << ", pid: " << pid);

   // Send a terminate signal to the proofserv
   if (pid > -1) {
      XrdProofUI ui;
      XrdProofdAux::GetUserInfo(fClient.c_str(), ui);
      if (XrdProofdAux::KillProcess(pid, 0, ui, changeown) != 0) {
         TRACE(XERR, "ord: problems signalling process: "<<fSrvPID);
      }
      XrdSysMutexHelper mhp(fMutex);
      fIsShutdown = true;
   }

   // Failed
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the associated proofserv process is alive. This is done
/// asynchronously by asking the process to callback and proof its vitality.
/// We do not block here: the caller may setup a waiting structure if
/// required.
/// If forward is true, the process will forward the request to the following
/// tiers.
/// Return 0 if the request was send successfully, -1 in case of error.

int XrdProofdProofServ::VerifyProofServ(bool forward)
{
   XPDLOC(SMGR, "ProofServ::VerifyProofServ")

   TRACE(DBG, "ord: " << fOrdinal<< ", pid: " << fSrvPID);

   int rc = 0;
   XrdOucString msg;

   // Prepare buffer
   int len = sizeof(kXR_int32);
   char *buf = new char[len];
   // Option
   kXR_int32 ifw = (forward) ? (kXR_int32)1 : (kXR_int32)0;
   ifw = static_cast<kXR_int32>(htonl(ifw));
   memcpy(buf, &ifw, sizeof(kXR_int32));

   {  XrdSysMutexHelper mhp(fMutex);
      // Propagate the ping request
      if (!fResponse || fResponse->Send(kXR_attn, kXPD_ping, buf, len) != 0) {
         msg = "could not propagate ping to proofsrv";
         rc = -1;
      }
   }
   // Cleanup
   delete[] buf;

   // Notify errors, if any
   if (rc != 0)
      TRACE(XERR, msg);

   // Done
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Broadcast a new group priority value to the worker servers.
/// Called by masters.

int XrdProofdProofServ::BroadcastPriority(int priority)
{
   XPDLOC(SMGR, "ProofServ::BroadcastPriority")

   XrdSysMutexHelper mhp(fMutex);

   // Prepare buffer
   int len = sizeof(kXR_int32);
   char *buf = new char[len];
   kXR_int32 itmp = priority;
   itmp = static_cast<kXR_int32>(htonl(itmp));
   memcpy(buf, &itmp, sizeof(kXR_int32));
   // Send over
   if (!fResponse || fResponse->Send(kXR_attn, kXPD_priority, buf, len) != 0) {
      // Failure
      TRACE(XERR,"problems telling proofserv");
      SafeDelArray(buf);
      return -1;
   }
   SafeDelArray(buf);
   TRACE(DBG, "priority "<<priority<<" sent over");
   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Send data to client cid.

int XrdProofdProofServ::SendData(int cid, void *buff, int len)
{
   XPDLOC(SMGR, "ProofServ::SendData")

   TRACE(HDBG, "length: "<<len<<" bytes (cid: "<<cid<<")");

   int rs = 0;
   XrdOucString msg;

   // Get corresponding instance
   XrdClientID *csid = 0;
   {  XrdSysMutexHelper mhp(fMutex);
      if (cid < 0 || cid > (int)(fClients.size() - 1) || !(csid = fClients.at(cid))) {
         XPDFORM(msg, "client ID not found (cid: %d, size: %d)", cid, fClients.size());
         rs = -1;
      }
      if (!rs && !(csid->R())) {
         XPDFORM(msg, "client not connected: csid: %p, cid: %d, fSid: %d",
                       csid, cid, csid->Sid());
         rs = -1;
      }
   }

   //
   // The message is strictly for the client requiring it
   if (!rs) {
      rs = -1;
      XrdProofdResponse *response = csid->R() ? csid->R() : 0;
      if (response)
         if (!response->Send(kXR_attn, kXPD_msg, buff, len))
            rs = 0;
   } else {
      // Notify error
      TRACE(XERR, msg);
   }

   // Done
   return rs;
}

////////////////////////////////////////////////////////////////////////////////
/// Send data over the open client links of this session.
/// Used when all the connected clients are eligible to receive the message.

int XrdProofdProofServ::SendDataN(void *buff, int len)
{
   XPDLOC(SMGR, "ProofServ::SendDataN")

   TRACE(HDBG, "length: "<<len<<" bytes");

   XrdOucString msg;

   XrdSysMutexHelper mhp(fMutex);

   // Send to connected clients
   XrdClientID *csid = 0;
   int ic = 0;
   for (ic = 0; ic < (int) fClients.size(); ic++) {
      if ((csid = fClients.at(ic)) && csid->P()) {
         XrdProofdResponse *resp = csid->R();
         if (!resp || resp->Send(kXR_attn, kXPD_msg, buff, len) != 0)
            return -1;
      }
   }

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill buf with relevant info about this session

void XrdProofdProofServ::ExportBuf(XrdOucString &buf)
{
   XPDLOC(SMGR, "ProofServ::ExportBuf")

   buf = "";
   int id, status, nc;
   XrdOucString tag, alias;
   {  XrdSysMutexHelper mhp(fMutex);
      id = fID;
      status = fStatus;
      nc = fNClients;
      tag = fTag;
      alias = fAlias; }
   XPDFORM(buf, " | %d %s %s %d %d", id, tag.c_str(), alias.c_str(), status, nc);
   TRACE(HDBG, "buf: "<< buf);

   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Create UNIX socket for internal connections

int XrdProofdProofServ::CreateUNIXSock(XrdSysError *edest)
{
   XPDLOC(SMGR, "ProofServ::CreateUNIXSock")

   TRACE(DBG, "enter");

   // Make sure we do not have already a socket
   if (fUNIXSock) {
       TRACE(DBG,"UNIX socket exists already! ("<<fUNIXSockPath<<")");
       return 0;
   }

   // Create socket
   fUNIXSock = new XrdNet(edest);

   // Make sure the admin path exists
   if (fAdminPath.length() > 0) {
      FILE *fadm = fopen(fAdminPath.c_str(), "a");
      if (fadm) {
         fclose(fadm);
      } else {
         TRACE(XERR, "unable to open / create admin path "<< fAdminPath << "; errno = "<<errno);
         return -1;
      }
   }

   // Check the path
   bool ok = 0;
   if (unlink(fUNIXSockPath.c_str()) != 0 && (errno != ENOENT)) {
      XPDPRT("WARNING: path exists: unable to delete it:"
               " try to use it anyway " <<fUNIXSockPath);
      ok = 1;
   }

   // Create the path
   int fd = 0;
   if (!ok) {
      if ((fd = open(fUNIXSockPath.c_str(), O_EXCL | O_RDWR | O_CREAT, 0700)) < 0) {
         TRACE(XERR, "unable to create path: " <<fUNIXSockPath);
         return -1;
      }
      close(fd);
   }
   if (fd > -1) {
      // Change ownership
      if (fUNIXSock->Bind((char *)fUNIXSockPath.c_str())) {
         TRACE(XERR, " problems binding to UNIX socket; path: " <<fUNIXSockPath);
         return -1;
      } else
         TRACE(DBG, "path for UNIX for socket is " <<fUNIXSockPath);
   } else {
      TRACE(XERR, "unable to open / create path for UNIX socket; tried path "<< fUNIXSockPath);
      return -1;
   }

   // Change ownership if running as super-user
   if (!geteuid()) {
      XrdProofUI ui;
      XrdProofdAux::GetUserInfo(fClient.c_str(), ui);
      if (chown(fUNIXSockPath.c_str(), ui.fUid, ui.fGid) != 0) {
         TRACE(XERR, "unable to change ownership of the UNIX socket"<<fUNIXSockPath);
         return -1;
      }
   }

   // We are done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the admin path and make sure the file exists

int XrdProofdProofServ::SetAdminPath(const char *a, bool assert, bool setown)
{
   XPDLOC(SMGR, "ProofServ::SetAdminPath")

   XrdSysMutexHelper mhp(fMutex);

   fAdminPath = a;

   // If we are not asked to assert the file we are done
   if (!assert) return 0;

   // Check if the session file exists
   FILE *fpid = fopen(a, "a");
   if (fpid) {
      fclose(fpid);
   } else {
      TRACE(XERR, "unable to open / create admin path "<< fAdminPath << "; errno = "<<errno);
      return -1;
   }

   // Check if the status file exists
   XrdOucString fn;
   XPDFORM(fn, "%s.status", a);
   if ((fpid = fopen(fn.c_str(), "a"))) {
      fprintf(fpid, "%d", fStatus);
      fclose(fpid);
   } else {
      TRACE(XERR, "unable to open / create status path "<< fn << "; errno = "<<errno);
      return -1;
   }

   if (setown) {
      // Set the ownership of the status file to the user
      XrdProofUI ui;
      if (XrdProofdAux::GetUserInfo(fClient.c_str(), ui) != 0) {
         TRACE(XERR, "unable to get info for user "<<fClient<<"; errno = "<<errno);
         return -1;
      }
      if (XrdProofdAux::ChangeOwn(fn.c_str(), ui) != 0) {
         TRACE(XERR, "unable to give ownership of the status file "<< fn << " to user; errno = "<<errno);
         return -1;
      }
   }

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Send a resume message to the this session. It is assumed that the session
/// has at least one async query to process and will immediately send
/// a getworkers request (the workers are already assigned).

int XrdProofdProofServ::Resume()
{
   XPDLOC(SMGR, "ProofServ::Resume")

   TRACE(REQ, "ord: " << fOrdinal<< ", pid: " << fSrvPID);

   int rc = 0;
   XrdOucString msg;

   {  XrdSysMutexHelper mhp(fMutex);
      //
      if (!fResponse || fResponse->Send(kXR_attn, kXPD_resume, 0, 0) != 0) {
         msg = "could not propagate resume to proofsrv";
         rc = -1;
      }
   }

   // Notify errors, if any
   if (rc != 0)
      TRACE(XERR, msg);

   // Done
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease active session counters on worker w

static int ExportWorkerDescription(const char *k, XrdProofWorker *w, void *s)
{
   XPDLOC(PMGR, "ExportWorkerDescription")

   XrdOucString *wrks = (XrdOucString *)s;
   if (w && wrks) {
      // Master at the beginning
      if (w->fType == 'M') {
         if (wrks->length() > 0) wrks->insert('&',0);
         wrks->insert(w->Export(), 0);
      } else {
         // Add separator if not the first
         if (wrks->length() > 0)
            (*wrks) += '&';
         // Add export version of the info
         (*wrks) += w->Export(k);
      }
      TRACE(HDBG, k <<" : "<<w->fHost.c_str()<<":"<<w->fPort <<" act: "<<w->Active());
      // Check next
      return 0;
   }

   // Not enough info: stop
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Export the assigned workers in the format understood by proofserv

void XrdProofdProofServ::ExportWorkers(XrdOucString &wrks)
{
   XrdSysMutexHelper mhp(fMutex);
   wrks = "";
   fWorkers.Apply(ExportWorkerDescription, (void *)&wrks);
}

////////////////////////////////////////////////////////////////////////////////
/// Export the assigned workers in the format understood by proofserv

void XrdProofdProofServ::DumpQueries()
{
   XPDLOC(PMGR, "DumpQueries")

   XrdSysMutexHelper mhp(fMutex);

   TRACE(ALL," ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ");
   TRACE(ALL," +++ client: "<<fClient<<", session: "<< fSrvPID <<
             ", # of queries: "<< fQueries.size());
   std::list<XrdProofQuery *>::iterator ii;
   int i = 0;
   for (ii = fQueries.begin(); ii != fQueries.end(); ++ii) {
      i++;
      TRACE(ALL," +++ #"<<i<<" tag:"<< (*ii)->GetTag()<<" dset: "<<
                (*ii)->GetDSName()<<" size:"<<(*ii)->GetDSSize());
   }
   TRACE(ALL," ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ");
}

////////////////////////////////////////////////////////////////////////////////
/// Get query with tag form the list of queries

XrdProofQuery *XrdProofdProofServ::GetQuery(const char *tag)
{
   XrdProofQuery *q = 0;
   if (!tag || strlen(tag) <= 0) return q;

   XrdSysMutexHelper mhp(fMutex);

   if (fQueries.size() <= 0) return q;

   std::list<XrdProofQuery *>::iterator ii;
   for (ii = fQueries.begin(); ii != fQueries.end(); ++ii) {
      q = *ii;
      if (!strcmp(tag, q->GetTag())) break;
      q = 0;
   }
   // Done
   return q;
}

////////////////////////////////////////////////////////////////////////////////
/// remove query with tag form the list of queries

void XrdProofdProofServ::RemoveQuery(const char *tag)
{
   XrdProofQuery *q = 0;
   if (!tag || strlen(tag) <= 0) return;

   XrdSysMutexHelper mhp(fMutex);

   if (fQueries.size() <= 0) return;

   std::list<XrdProofQuery *>::iterator ii;
   for (ii = fQueries.begin(); ii != fQueries.end(); ++ii) {
      q = *ii;
      if (!strcmp(tag, q->GetTag())) break;
      q = 0;
   }
   // remove it
   if (q) {
      fQueries.remove(q);
      delete q;
   }

   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease active session counters on worker w

static int CountEffectiveSessions(const char *, XrdProofWorker *w, void *s)
{
   int *actw = (int *)s;
   if (w && actw) {
      *actw += w->GetNActiveSessions();
      // Check next
      return 0;
   }

   // Not enough info: stop
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the effective number of users on this session nodes
/// and communicate it to the master together with the total number
/// of sessions and the number of active sessions. for monitoring issues.

void XrdProofdProofServ::SendClusterInfo(int nsess, int nacti)
{
   XPDLOC(PMGR, "SendClusterInfo")

   // Only if we are active
   if (fWorkers.Num() <= 0) return;

   int actw = 0;
   fWorkers.Apply(CountEffectiveSessions, (void *)&actw);
   // The number of effective sessions * 1000
   int neffs = (actw*1000)/fWorkers.Num();
   TRACE(DBG, "# sessions: "<<nsess<<", # active: "<<nacti<<", # effective: "<<neffs/1000.);

   XrdSysMutexHelper mhp(fMutex);

   // Prepare buffer
   int len = 3*sizeof(kXR_int32);
   char *buf = new char[len];
   kXR_int32 off = 0;
   kXR_int32 itmp = nsess;
   itmp = static_cast<kXR_int32>(htonl(itmp));
   memcpy(buf + off, &itmp, sizeof(kXR_int32));
   off += sizeof(kXR_int32);
   itmp = nacti;
   itmp = static_cast<kXR_int32>(htonl(itmp));
   memcpy(buf + off, &itmp, sizeof(kXR_int32));
   off += sizeof(kXR_int32);
   itmp = neffs;
   itmp = static_cast<kXR_int32>(htonl(itmp));
   memcpy(buf + off, &itmp, sizeof(kXR_int32));
   // Send over
   if (!fResponse || fResponse->Send(kXR_attn, kXPD_clusterinfo, buf, len) != 0) {
      // Failure
      TRACE(XERR,"problems sending proofserv");
   }
   SafeDelArray(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the effective number of users on this session nodes
/// and communicate it to the master together with the total number
/// of sessions and the number of active sessions. for monitoring issues.

int XrdProofdProofServ::CheckSession(bool oldvers, bool isrec,
                                      int shutopt, int shutdel, bool changeown, int &nc)
{
   XPDLOC(PMGR, "SendClusterInfo")

   XrdOucString emsg;
   bool rmsession = 0;
   nc = -1;
   {  XrdSysMutexHelper mhp(fMutex);

      bool skipcheck = fSkipCheck;
      fSkipCheck = false;

      if (!skipcheck || oldvers) {
         nc = 0;
         // Remove this from the list of clients
         std::vector<XrdClientID *>::iterator i;
         for (i = fClients.begin(); i != fClients.end(); ++i) {
            if ((*i) && (*i)->P() && (*i)->P()->Link()) nc++;
         }
         // Check if we need to shutdown it
         if (nc <= 0 && (!isrec || oldvers)) {
            int idlet = -1, disct = -1, now = time(0);
            if (fStatus == kXPD_idle)
               idlet = now - fSetIdleTime;
            if (idlet <= 0) idlet = -1;
            if (fDisconnectTime > 0)
               disct = now - fDisconnectTime;
            if (disct <= 0) disct = -1;
            if ((fSrvType != kXPD_TopMaster) ||
                (shutopt == 1 && (idlet >= shutdel)) ||
                (shutopt == 2 && (disct >= shutdel))) {
               // Send a terminate signal to the proofserv
               if (fSrvPID > -1) {
                  XrdProofUI ui;
                  XrdProofdAux::GetUserInfo(fClient.c_str(), ui);
                  if (XrdProofdAux::KillProcess(fSrvPID, 0, ui, changeown) != 0) {
                     XPDFORM(emsg, "ord: problems signalling process: %d", fSrvPID);
                  }
                  fIsShutdown = true;
               }
               rmsession = 1;
            }
         }
      }
   }
   // Notify error, if any
   if (emsg.length() > 0) {
      TRACE(XERR,emsg.c_str());
   }
   // Done
   return rmsession;
}
