// @(#)root/proofd:$Id$
// Author: G. Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdClient                                                      //
//                                                                      //
// Author: G. Ganis, CERN, 2007                                         //
//                                                                      //
// Auxiliary class describing a PROOF client.                           //
// Used by XrdProofdProtocol.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <sys/stat.h>

#include "XrdNet/XrdNet.hh"
#include "XrdSys/XrdSysPriv.hh"

#include "XrdProofdClient.h"
#include "XrdProofdLauncher.h"
#include "XrdProofdProtocol.h"
#include "XrdProofdProofServ.h"
#include "XrdProofdProofServMgr.h"

#include "XrdProofdTrace.h"

//__________________________________________________________________________
XrdProofdClient::XrdProofdClient(XrdProofUI ui, bool master, bool changeown,
                                 XrdSysError *, const char *adminpath)
                : fSandbox(ui, master, changeown)
{
   // Constructor
   XPDLOC(CMGR, "Client::Client")

   fProofServs.clear();
   fClients.clear();
   fUI = ui;
   fROOT = 0;
   fIsValid = 0;
   fAskedToTouch = 0;
   fChangeOwn = changeown;

   // Make sure the admin path exists
   XPDFORM(fAdminPath, "%s/%s.%s", adminpath, ui.fUser.c_str(), ui.fGroup.c_str());
   struct stat st;
   if (stat(adminpath, &st) != 0) {
      TRACE(XERR, "problems stating admin path "<<adminpath<<"; errno = "<<errno);
      return;
   }
   XrdProofUI effui;
   XrdProofdAux::GetUserInfo(st.st_uid, effui);
   if (XrdProofdAux::AssertDir(fAdminPath.c_str(), effui, 1) != 0)
      return;

   // We must have a valid sandbox
   if (fSandbox.IsValid()) fIsValid = 1;
   
   // The session launcher (we may have a plugin here, on day ...)
   fLauncher = new XrdProofdLauncher(this);
}

//__________________________________________________________________________
XrdProofdClient::~XrdProofdClient()
{
   // Destructor

}

//__________________________________________________________________________
bool XrdProofdClient::Match(const char *usr, const char *grp)
{
   // return TRUE if this instance matches 'id' (and 'grp', if defined) 

   if (!fIsValid) return 0;

   bool rc = (usr && !strcmp(usr, User())) ? 1 : 0;
   if (rc && grp && strlen(grp) > 0)
      rc = (grp && Group() && !strcmp(grp, Group())) ? 1 : 0;

   return rc;
}

//__________________________________________________________________________
int XrdProofdClient::GetClientID(XrdProofdProtocol *p)
{
   // Get next free client ID. If none is found, increase the vector size
   // and get the first new one
   XPDLOC(CMGR, "Client::GetClientID")

   XrdClientID *cid = 0;
   int ic = 0, sz = 0;
   {  XrdSysMutexHelper mh(fMutex);
      if (!fIsValid) return -1;
      // Search for free places in the existing vector
      for (ic = 0; ic < (int)fClients.size() ; ic++) {
         if (fClients[ic] && !fClients[ic]->IsValid()) {
            cid = fClients[ic];
            cid->Reset();
            break;
         }
      }

      if (!cid) {
         // We need to resize (double it)
         if (ic >= (int)fClients.capacity())
            fClients.reserve(2*fClients.capacity());
 
         // Fill in new element
         cid = new XrdClientID();
         fClients.push_back(cid);
         sz = fClients.size();
      }
   }
   // Re-init for this protocol
   if (cid) {
      cid->SetP(p);
      // Reference Stream ID
      unsigned short sid;
      memcpy((void *)&sid, (const void *)&(p->Request()->header.streamid[0]), 2);
      cid->SetSid(sid);
   }

   TRACE(DBG, "size = "<<sz<<", ic = "<<ic);

   // We are done
   return ic;
}

//__________________________________________________________________________
int XrdProofdClient::ReserveClientID(int cid)
{
   // Reserve a client ID. If none is found, increase the vector size
   // and performe the needed initializations
   XPDLOC(CMGR, "Client::ReserveClientID")

   if (cid < 0)
      return -1;

   int sz = 0, newsz = 0;
   {  XrdSysMutexHelper mh(fMutex);
      if (!fIsValid) return -1;
      if (cid >= (int)fClients.size()) {

         // We need to resize (double it)
         newsz = fClients.capacity();
         if (cid >= (int)fClients.capacity()) {
            newsz = 2 * fClients.capacity();
            newsz = (cid < newsz) ? newsz : cid + 1;
            fClients.reserve(newsz);
         }

         // Fill in new elements
         while (cid >= (int)fClients.size())
            fClients.push_back(new XrdClientID());
      }
      sz = fClients.size();
   }

   TRACE(DBG, "cid = "<<cid<<", size = "<<sz<<", capacity = "<<newsz);

   // We are done
   return 0;
}

//__________________________________________________________________________
XrdProofdProofServ *XrdProofdClient::GetFreeServObj()
{
   // Get next free server ID. If none is found, increase the vector size
   // and get the first new one
   XPDLOC(CMGR, "Client::GetFreeServObj")

   int ic = 0, newsz = 0, sz = 0;
   XrdProofdProofServ *xps = 0;
   XrdOucString msg;
   {  XrdSysMutexHelper mh(fMutex);
      if (!fIsValid) return xps;

      // Search for free places in the existing vector
      for (ic = 0; ic < (int)fProofServs.size() ; ic++) {
         if (fProofServs[ic] && !(fProofServs[ic]->IsValid())) {
            fProofServs[ic]->SetValid();
            break;
         }
      }

      // If we did not find it, we resize the vector (double it)
      if (ic >= (int)fProofServs.capacity()) {
         newsz = 2 * fProofServs.capacity();
         fProofServs.reserve(newsz);
      }
      if (ic >= (int)fProofServs.size()) {
         // Allocate new element
         fProofServs.push_back(new XrdProofdProofServ());
      }
      sz = fProofServs.size();

      xps = fProofServs[ic];
      xps->SetValid();
      xps->SetID(ic);
   }

   // Notify
   if (TRACING(DBG)) {
      if (newsz > 0) {
         XPDFORM(msg, "new capacity = %d, size = %d, ic = %d, xps = %p",
                      newsz, sz, ic, xps);
      } else {
         XPDFORM(msg, "size = %d, ic = %d, xps = %p", sz, ic, xps);
      }
      XPDPRT(msg);
   }

   // We are done
   return xps;
}

//__________________________________________________________________________
XrdProofdProofServ *XrdProofdClient::GetServObj(int id)
{
   // Get server at 'id'. If needed, increase the vector size
   XPDLOC(CMGR, "Client::GetServObj")

   TRACE(DBG, "id: "<< id);

   if (id < 0) {
      TRACE(XERR, "invalid input: id: "<< id);
      return (XrdProofdProofServ *)0;
   }

   XrdOucString dmsg, emsg;
   XrdProofdProofServ *xps = 0;
   int siz = 0, cap = 0;
   {  XrdSysMutexHelper mh(fMutex);
      if (!fIsValid) return xps;
      siz = fProofServs.size();
      cap = fProofServs.capacity();
   }
   TRACE(DBG, "size = "<<siz<<"; capacity = "<<cap);

   bool newcap = 0;
   {  XrdSysMutexHelper mh(fMutex);
      if (!fIsValid) return xps;
      if (id < (int)fProofServs.size()) {
         if (!(xps = fProofServs[id])) {
            emsg = "instance in use or undefined! protocol error";
         }
      } else {
         // If we did not find it, we first resize the vector if needed (double it)
         if (id >= (int)fProofServs.capacity()) {
            int newsz = 2 * fProofServs.capacity();
            newsz = (id < newsz) ? newsz : id+1;
            fProofServs.reserve(newsz);
            cap = fProofServs.capacity();
            newcap = 1;
         }
         int nnew = id - fProofServs.size() + 1;
         while (nnew--)
            fProofServs.push_back(new XrdProofdProofServ());
         xps = fProofServs[id];
      }
   }
   xps->SetID(id);
   xps->SetValid();
   if (TRACING(DBG)) {
      {  XrdSysMutexHelper mh(fMutex);
         if (fIsValid) {
            siz = fProofServs.size();
            cap = fProofServs.capacity();
         }
      }
      TRACE(DBG, "size = "<<siz<<" (capacity = "<<cap<<"); id = "<<id);
   }

   // We are done
   return xps;
}

//______________________________________________________________________________
XrdProofdProofServ *XrdProofdClient::GetServer(XrdProofdProtocol *p)
{
   // Get server instance connected via 'p'
   XPDLOC(CMGR, "Client::GetServer")

   TRACE(DBG, "enter: p: " << p);


   XrdProofdProofServ *xps = 0;
   std::vector<XrdProofdProofServ *>::iterator ip;
   XrdSysMutexHelper mh(fMutex);
   if (!fIsValid) return xps;
   for (ip = fProofServs.begin(); ip != fProofServs.end(); ++ip) {
      xps = (*ip);
      if (xps && xps->SrvPID() == p->Pid())
         break;
      xps = 0;
   }
   // Done
   return xps;
}

//______________________________________________________________________________
XrdProofdProofServ *XrdProofdClient::GetServer(int psid)
{
   // Get from the vector server instance with ID psid

   XrdSysMutexHelper mh(fMutex);
   if (fIsValid && psid > -1 && psid < (int) fProofServs.size())
      return fProofServs.at(psid);
   // Done
   return (XrdProofdProofServ *)0;
}

//______________________________________________________________________________
void XrdProofdClient::EraseServer(int psid)
{
   // Erase server with id psid from the list
   XPDLOC(CMGR, "Client::EraseServer")

   TRACE(DBG, "enter: psid: " << psid);

   XrdProofdProofServ *xps = 0;
   std::vector<XrdProofdProofServ *>::iterator ip;
   XrdSysMutexHelper mh(fMutex);
   if (!fIsValid) return;

   for (ip = fProofServs.begin(); ip != fProofServs.end(); ++ip) {
      xps = *ip;
      if (xps && xps->Match(psid)) {
         // Reset (invalidate)
         xps->Reset();
         break;
      }
   }
}

//______________________________________________________________________________
int XrdProofdClient::GetTopServers()
{
   // Return the number of valid proofserv topmaster sessions in the list
   XPDLOC(CMGR, "Client::GetTopServers")

   int nv = 0;

   XrdProofdProofServ *xps = 0;
   std::vector<XrdProofdProofServ *>::iterator ip;
   XrdSysMutexHelper mh(fMutex);
   if (!fIsValid) return nv;
   for (ip = fProofServs.begin(); ip != fProofServs.end(); ++ip) {
      if ((xps = *ip) && xps->IsValid() && (xps->SrvType() == kXPD_TopMaster)) {
         TRACE(DBG,"found potentially valid topmaster session: pid "<<xps->SrvPID());
         nv++;
      }
   }

   // Done
   return nv;
}

//______________________________________________________________________________
int XrdProofdClient::ResetClientSlot(int ic)
{
   // Reset slot at 'ic'
   XPDLOC(CMGR, "Client::ResetClientSlot")

   TRACE(DBG, "enter: ic: " << ic);

   XrdSysMutexHelper mh(fMutex);
   if (fIsValid) {
      if (ic >= 0 && ic < (int) fClients.size()) {
         fClients[ic]->Reset();
         return 0;
      }
   }
   // Done
   return -1;
}

//______________________________________________________________________________
XrdProofdProtocol *XrdProofdClient::GetProtocol(int ic)
{
   // Reset slot at 'ic'
   XPDLOC(CMGR, "Client::GetProtocol")

   TRACE(DBG, "enter: ic: " << ic);

   XrdProofdProtocol *p = 0;

   XrdSysMutexHelper mh(fMutex);
   if (fIsValid) {
      if (ic >= 0 && ic < (int) fClients.size()) {
         p = fClients[ic]->P();
      }
   }
   // Done
   return p;
}

//______________________________________________________________________________
int XrdProofdClient::SetClientID(int cid, XrdProofdProtocol *p)
{
   // Set slot cid to instance 'p'
   XPDLOC(CMGR, "Client::SetClientID")

   TRACE(DBG, "cid: "<< cid <<", p: " << p);

   XrdSysMutexHelper mh(fMutex);
   if (!fIsValid) return -1;

   if (cid >= 0 && cid < (int) fClients.size()) {
      if (fClients[cid] && (fClients[cid]->P() != p))
         fClients[cid]->Reset();
      fClients[cid]->SetP(p);
      // Reference Stream ID
      unsigned short sid;
      memcpy((void *)&sid, (const void *)&(p->Request()->header.streamid[0]), 2);
      fClients[cid]->SetSid(sid);
      return 0;
   }

   // Not found
   return -1;
}

//______________________________________________________________________________
void XrdProofdClient::Broadcast(const char *msg)
{
   // Broadcast message 'msg' to the connected clients
   XPDLOC(CMGR, "Client::Broadcast")

   int len = 0;
   if (msg && (len = strlen(msg)) > 0) {

      // Notify the attached clients
      int ic = 0;
      XrdClientID *cid = 0;
      XrdSysMutexHelper mh(fMutex);
      for (ic = 0; ic < (int) fClients.size(); ic++) {
         if ((cid = fClients.at(ic)) && cid->P() && cid->P()->ConnType() == kXPD_ClientMaster) {

            if (cid->P()->Link()) {
               TRACE(ALL," sending to: "<<cid->P()->Link()->ID);
               XrdProofdResponse *response = cid->R();
               if (response)
                  response->Send(kXR_attn, kXPD_srvmsg, (char *) msg, len);
            }
         }
      }
   }
}

//______________________________________________________________________________
int XrdProofdClient::Touch(bool reset)
{
   // Send a touch the connected clients: this will remotely touch the associated
   // TSocket instance and schedule an asynchronous touch of the client admin file.
   // This request is only sent once per client: this is controlled by the flag
   // fAskedToTouch, whcih can reset to FALSE by calling this function with reset
   // TRUE.
   // Return 0 if the request is sent or if asked to reset.
   // Retunn 1 if the request was already sent.

   // If we are asked to reset, just do that and return
   if (reset) {
      fAskedToTouch = 0;
      return 0;
   }

   // If already asked to touch say it by return 1
   if (fAskedToTouch) return 1;
   
   // Notify the attached clients
   int ic = 0;
   XrdClientID *cid = 0;
   XrdSysMutexHelper mh(fMutex);
   for (ic = 0; ic < (int) fClients.size(); ic++) {
      // Do not send to old clients
      if ((cid = fClients.at(ic)) && cid->P() && cid->P()->ProofProtocol() > 17) {
         if (cid->P()->ConnType() != kXPD_Internal) {
            XrdProofdResponse *response = cid->R();
            if (response) response->Send(kXR_attn, kXPD_touch, (char *)0, 0);
         }
      }
   }
   // Do it only once a time
   fAskedToTouch = 1;
   // Done 
   return 0;
}

//______________________________________________________________________________
bool XrdProofdClient::VerifySession(XrdProofdProofServ *xps, XrdProofdResponse *r)
{
   // Quick verification of session 'xps' to avoid attaching clients to
   // non responding sessions. We do here a sort of loose ping.
   // Return true is responding, false otherwise.
   XPDLOC(CMGR, "Client::VerifySession")

   if (!xps || !(xps->IsValid())) {
      TRACE(XERR, " session undefined or invalid");
      return 0;
   }

   // Admin path
   XrdOucString path(xps->AdminPath());
   if (path.length() <= 0) {
      TRACE(XERR, "admin path is empty! - protocol error");
      return 0;
   }
   path += ".status";

   // Stat the admin file
   struct stat st0;
   if (stat(path.c_str(), &st0) != 0) {
      TRACE(XERR, "cannot stat admin path: "<<path);
      return 0;
   }
   int now = time(0);
   if (now >= st0.st_mtime && (now - st0.st_mtime) <= 1) return 1;
      TRACE(ALL, "admin path: "<<path<<", mtime: "<< st0.st_mtime << ", now: "<< now);

   // Take the pid
   int pid = xps->SrvPID();
   // If the session is alive ...
   if (XrdProofdAux::VerifyProcessByID(pid) != 0) {
      // Send the request (no further propagation)
      if (xps->VerifyProofServ(0) != 0) {
         TRACE(XERR, "could not send verify request to proofsrv");
         return 0;
      }
      // Wait for the action for fgMgr->SessionMgr()->VerifyTimeOut() secs,
      // checking every 1 sec
      XrdOucString notmsg;
      struct stat st1;
      int ns = 10;
      while (ns--) {
         if (stat(path.c_str(), &st1) == 0) {
            if (st1.st_mtime > st0.st_mtime) {
               return 1;
            }
         }
         // Wait 1 sec
         TRACE(HDBG, "waiting "<<ns<<" secs for session "<<pid<<
                     " to touch the admin path");
         if (r && ns == 5) {
            XPDFORM(notmsg, "verifying existing sessions, %d seconds ...", ns);
            r->Send(kXR_attn, kXPD_srvmsg, 0, (char *) notmsg.c_str(), notmsg.length());
         }
         sleep(1);
      }
   }

   // Verification Failed
   return 0;
}

//______________________________________________________________________________
void XrdProofdClient::SkipSessionsCheck(std::list<XrdProofdProofServ *> *active,
                                        XrdOucString &emsg, XrdProofdResponse *r)
{
   // Skip the next sessions status check. This is used, for example, when
   // somebody has shown interest in these sessions to give more time for the
   // reconnection.
   // If active is defined, the list of active sessions is filled.
   XPDLOC(CMGR, "Client::SkipSessionsCheck")

   XrdProofdProofServ *xps = 0;
   std::vector<XrdProofdProofServ *>::iterator ip;
   for (ip = fProofServs.begin(); ip != fProofServs.end(); ++ip) {
      if ((xps = *ip) && xps->IsValid() && (xps->SrvType() == kXPD_TopMaster)) {
         if (VerifySession(xps, r)) {
            xps->SetSkipCheck(); // Skip next validity check
            if (active) active->push_back(xps);
         } else {
            if (xps->SrvPID() > 0) {
               if (emsg.length() <= 0)
                  emsg = "ignoring (apparently) non-responding session(s): ";
               else
                  emsg += " ";
               emsg += xps->SrvPID();
            }
            TRACE(ALL,"session "<<xps->SrvPID()<<" does not react: dead?");
         }
      }
   }
   if (active)
      TRACE(HDBG, "found: " << active->size() << " sessions");

   // Over
   return;
}

//______________________________________________________________________________
XrdOucString XrdProofdClient::ExportSessions(XrdOucString &emsg,
                                             XrdProofdResponse *r)
{
   // Return a string describing the existing sessions

   XrdOucString out, buf;

   // Protect from next session check and get the list of actives
   std::list<XrdProofdProofServ *> active;
   SkipSessionsCheck(&active, emsg, r);

   // Fill info
   XrdProofdProofServ *xps = 0;
   out += (int) active.size();
   std::list<XrdProofdProofServ *>::iterator ia;
   for (ia = active.begin(); ia != active.end(); ++ia) {
      if ((xps = *ia) && xps->IsValid()) {
         xps->ExportBuf(buf);
         out += buf;
      }
   }

   // Over
   return out;
}

//______________________________________________________________________________
void XrdProofdClient::TerminateSessions(int srvtype, XrdProofdProofServ *ref,
                                        const char *msg, XrdProofdPipe *pipe,
                                        bool changeown)
{
   // Terminate client sessions; IDs of signalled processes are added to sigpid.
   XPDLOC(CMGR, "Client::TerminateSessions")

   // Loop over client sessions and terminated them
   int is = 0;
   XrdProofdProofServ *s = 0;
   for (is = 0; is < (int) fProofServs.size(); is++) {
      if ((s = fProofServs.at(is)) && s->IsValid() && (!ref || ref == s) &&
          (s->SrvType() == srvtype || (srvtype == kXPD_AnyServer))) {
         TRACE(DBG, "terminating " << s->SrvPID());

         if (srvtype == kXPD_TopMaster && msg && strlen(msg) > 0)
            // Tell other attached clients, if any, that this session is gone
            Broadcast(msg);

         // Sendout a termination signal
         s->TerminateProofServ(changeown);

         // Record this session in the sandbox as old session
         XrdOucString tag = "-";
         tag += s->SrvPID();
         if (fSandbox.GuessTag(tag, 1) == 0)
            fSandbox.RemoveSession(tag.c_str());

         // Tell the session manager that the session is gone
         if (pipe) {
            int rc = 0;
            XrdOucString buf(s->AdminPath());
            buf.erase(0, buf.rfind('/') + 1);
            TRACE(DBG,"posting kSessionRemoval with: '"<<buf<<"'");
            if ((rc = pipe->Post(XrdProofdProofServMgr::kSessionRemoval, buf.c_str())) != 0) {
               TRACE(XERR, "problem posting the pipe; errno: "<<-rc);
            }
         }

         // Reset this session
         s->Reset();
      }
   }
}

//__________________________________________________________________________
void XrdProofdClient::ResetSessions()
{
   // Reset this instance

   fAskedToTouch = 0;

   XrdSysMutexHelper mh(fMutex);
   std::vector<XrdProofdProofServ *>::iterator ip;
   for (ip = fProofServs.begin(); ip != fProofServs.end(); ip++) {
      // Reset (invalidate) the server instance
      if (*ip) (*ip)->Reset();
   }
}

