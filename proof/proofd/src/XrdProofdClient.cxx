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
#include "XrdProofdProtocol.h"
#include "XrdProofdProofServ.h"
#include "XrdProofdProofServMgr.h"

#include "XrdProofdTrace.h"

//__________________________________________________________________________
XrdProofdClient::XrdProofdClient(XrdProofUI ui, bool master, bool changeown,
                                 XrdSysError *edest, const char *adminpath)
                : fSandbox(ui, master, changeown)
{
   // Constructor

   fProofServs.clear();
   fClients.clear();
   fUI = ui;
   fUNIXSock = 0;
   fUNIXSockSaved = 0;
   fROOT = 0;
   fIsValid = 0;
   fChangeOwn = changeown;

   // Make sure the admin path exists
   fAdminPath.form("%s/%s.%s", adminpath, ui.fUser.c_str(), ui.fGroup.c_str());
   struct stat st;
   if (stat(adminpath, &st) != 0)
      return;
   XrdProofUI effui;
   XrdProofdAux::GetUserInfo(st.st_uid, effui);
   if (XrdProofdAux::AssertDir(fAdminPath.c_str(), effui, 1) != 0)
      return;

   // Create the UNIX socket
   if (fSandbox.IsValid())
      if (CreateUNIXSock(edest) == 0)
         fIsValid = 1;
}

//__________________________________________________________________________
XrdProofdClient::~XrdProofdClient()
{
   // Destructor

   // Unix socket
   SafeDel(fUNIXSock);
}

//__________________________________________________________________________
bool XrdProofdClient::Match(const char *usr, const char *grp)
{
   // return TRUE if this instance matches 'id' (and 'grp', if defined) 

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
      // Search for free places in the existing vector
      for (ic = 0; ic < (int)fClients.size() ; ic++) {
         if (!fClients[ic]) {
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
int XrdProofdClient::CreateUNIXSock(XrdSysError *edest)
{
   // Create UNIX socket for internal connections
   XPDLOC(CMGR, "Client::CreateUNIXSock")

   TRACE(DBG, "enter");


   // Make sure we do not have already a socket
   if (fUNIXSock) {
       TRACE(DBG,"UNIX socket exists already! (" <<
             fUNIXSockPath<<")");
       return 0;
   }

   // Inputs must make sense
   if (!edest) {
       TRACE(XERR,"invalid input: edest: " << (int *)edest);
       return -1;
   }

   // Create socket
   fUNIXSock = new XrdNet(edest);

   // Create path if needed
   fUNIXSockPath.form("%s/xpdsock", fAdminPath.c_str());
   fUNIXSockPath.replace("//", "/");
   bool rm = 0, ok = 0;
   struct stat st;
   if (stat(fUNIXSockPath.c_str(), &st) == 0) {
      if (!S_ISSOCK(st.st_mode))
         rm = 1;
      else
         ok = 1;
   } else {
      if (errno != ENOENT)
         rm = 1;
   }
   if (rm  && unlink(fUNIXSockPath.c_str()) != 0) {
      TRACE(XERR, "non-socket path exists: unable to delete it: " <<fUNIXSockPath);
      return -1;
   }

   // Create the path
   int fd = 0;
   if (!ok) {
      if ((fd = open(fUNIXSockPath.c_str(), O_EXCL | O_RDWR | O_CREAT)) < 0) {
         TRACE(XERR, "unable to create path: " <<fUNIXSockPath);
         return -1;
      }
      close(fd);
   }
   if (fd > -1) {
      if (fUNIXSock->Bind((char *)fUNIXSockPath.c_str())) {
         TRACE(XERR, " problems binding to UNIX socket; path: " <<fUNIXSockPath);
         return -1;
      } else
         TRACE(DBG, "path for UNIX for socket is " <<fUNIXSockPath);
   } else {
      TRACE(XERR, "unable to open / create path for UNIX socket; tried path "<< fUNIXSockPath);
      return -1;
   }

   // Set ownership of the socket file to the client
   XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
   if (XpdBadPGuard(pGuard, fUI.fUid) && fChangeOwn) {
      TRACE(XERR, "could not get privileges");
      return -1;
   }
   if (fChangeOwn) {
      if (chown(fUNIXSockPath.c_str(), fUI.fUid, fUI.fGid) == -1) {
         TRACE(XERR, "cannot set user ownership on UNIX socket (errno: "<<errno<<")");
         return -1;
      }
      // Make sure that it worked out
      if ((stat(fUNIXSockPath.c_str(), &st) != 0) || 
            (int) st.st_uid != fUI.fUid || (int) st.st_gid != fUI.fGid) {
         TRACE(XERR, "problems setting user ownership on UNIX socket");
         return -1;
      }
   }

   // We are done
   return 0;
}

//__________________________________________________________________________
void XrdProofdClient::SaveUNIXPath()
{
   // Save UNIX path in <SandBox>/.unixpath
   XPDLOC(CMGR, "Client::SaveUNIXPath")

   TRACE(DBG, "saved? "<<fUNIXSockSaved);

   // Make sure we do not have already a socket
   if (fUNIXSockSaved) {
      TRACE(DBG, "UNIX path saved already");
      return;
   }

   // Make sure we do not have already a socket
   if (fUNIXSockPath.length() <= 0) {
       TRACE(XERR, "UNIX path undefined!");
       return;
   }

   // File name
   XrdOucString fn = fSandbox.Dir();
   fn += "/.unixpath";

   // Open the file for appending
   FILE *fup = fopen(fn.c_str(), "a+");
   if (!fup) {
      TRACE(XERR, "cannot open file "<<fn<<" for appending (errno: "<<errno<<")");
      return;
   }

   // Lock the file
   lseek(fileno(fup), 0, SEEK_SET);
   if (lockf(fileno(fup), F_LOCK, 0) == -1) {
      TRACE(XERR, "cannot lock file "<<fn<<" (errno: "<<errno<<")");
      fclose(fup);
      return;
   }

   // Read content, if any
   char ln[1024], path[1024];
   int pid = -1;
   std::list<XrdOucString *> actln;
   while (fgets(ln, sizeof(ln), fup)) {
      // Get rid of '\n'
      if (ln[strlen(ln)-1] == '\n')
         ln[strlen(ln)-1] = '\0';
      // Skip empty or comment lines
      if (strlen(ln) <= 0 || ln[0] == '#')
         continue;
      // Get PID and path
      sscanf(ln, "%d %s", &pid, path);
      // Verify if still running
      int vrc = -1;
      if ((vrc = XrdProofdAux::VerifyProcessByID(pid, "xrootd")) != 0) {
         // Still there
         actln.push_back(new XrdOucString(ln));
      }
   }

   // Truncate the file
   if (ftruncate(fileno(fup), 0) == -1) {
      TRACE(XERR, "cannot truncate file "<<fn<<" (errno: "<<errno<<")");
      lseek(fileno(fup), 0, SEEK_SET);
      lockf(fileno(fup), F_ULOCK, 0);
      fclose(fup);
      return;
   }

   // If active sockets still exist, write out new composition
   if (actln.size() > 0) {
      std::list<XrdOucString *>::iterator i;
      for (i = actln.begin(); i != actln.end(); ++i) {
         fprintf(fup, "%s\n", (*i)->c_str());
         delete (*i);
      }
   }

   // Append the path and our process ID
   lseek(fileno(fup), 0, SEEK_END);
   fprintf(fup, "%d %s\n", getppid(), fUNIXSockPath.c_str());

   // Unlock the file
   lseek(fileno(fup), 0, SEEK_SET);
   if (lockf(fileno(fup), F_ULOCK, 0) == -1)
      TRACE(XERR, "cannot unlock file "<<fn<<" (errno: "<<errno<<")");

   // Close the file
   fclose(fup);

   // Path saved
   fUNIXSockSaved = 1;
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

      // Allocate new element
      fProofServs.push_back(new XrdProofdProofServ());
      sz = fProofServs.size();

      xps = fProofServs[ic];
      xps->SetValid();
      xps->SetID(ic);
   }

   // Notify
   if (TRACING(DBG)) {
      if (newsz > 0) {
         msg.form("new capacity = %d, size = %d, ic = %d, xps = %p",
                   newsz, sz, ic, xps);
      } else {
         msg.form("size = %d, ic = %d, xps = %p", sz, ic, xps);
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

   XrdSysMutexHelper mh(fMutex);

   TRACE(DBG, "size = "<<fProofServs.size()<<"; capacity = "<<fProofServs.capacity());

   if (id < (int)fProofServs.size()) {
      if (fProofServs[id]) {
         fProofServs[id]->SetValid();
         return fProofServs[id];
      } else {
         TRACE(XERR, "instance in use or undefined! protocol error");
         return (XrdProofdProofServ *)0;
      }
   }

   // If we did not find it, we first resize the vector if needed (double it)
   if (id >= (int)fProofServs.capacity()) {
      int newsz = 2 * fProofServs.capacity();
      newsz = (id < newsz) ? newsz : id+1;
      fProofServs.reserve(newsz);
      TRACE(DBG, "new capacity = "<<fProofServs.capacity());
   }
   int nnew = id - fProofServs.size() + 1;
   while (nnew--)
      fProofServs.push_back(new XrdProofdProofServ());

   XrdProofdProofServ *xps = fProofServs[id];
   xps->SetValid();
   xps->SetID(id);

   TRACE(DBG, "size = "<<fProofServs.size()<<"; id = "<<id);

   // We are done
   return xps;
}

//______________________________________________________________________________
XrdProofdProofServ *XrdProofdClient::GetServer(int psid)
{
   // Get from the vector server instance with id psid
   XPDLOC(CMGR, "Client::GetServer")

   TRACE(DBG, "psid: " << psid);

   XrdSysMutexHelper mh(fMutex);

   XrdProofdProofServ *xps = 0;
   std::vector<XrdProofdProofServ *>::iterator ip;
   for (ip = fProofServs.begin(); ip != fProofServs.end(); ++ip) {
      xps = *ip;
      if (xps && xps->Match(psid))
         break;
      xps = 0;
   }
   // Done
   return xps;
}

//______________________________________________________________________________
XrdProofdProofServ *XrdProofdClient::GetServer(XrdProofdProtocol *p)
{
   // Get server instance connected via 'p'
   XPDLOC(CMGR, "Client::GetServer")

   TRACE(DBG, "enter: p: " << p);

   XrdSysMutexHelper mh(fMutex);

   XrdProofdProofServ *xps = 0;
   std::vector<XrdProofdProofServ *>::iterator ip;
   for (ip = fProofServs.begin(); ip != fProofServs.end(); ++ip) {
      xps = (*ip);
      if (xps->SrvPID() == p->Pid())
         break;
      xps = 0;
   }
   // Done
   return xps;
}

//______________________________________________________________________________
XrdProofdProofServ *XrdProofdClient::GetProofServ(int psid)
{
   // Get from the vector server instance with ID psid

   XrdSysMutexHelper mh(fMutex);
   if (psid > -1 && psid < (int) fProofServs.size())
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

   XrdSysMutexHelper mh(fMutex);

   XrdProofdProofServ *xps = 0;
   std::vector<XrdProofdProofServ *>::iterator ip;
   for (ip = fProofServs.begin(); ip != fProofServs.end(); ++ip) {
      xps = *ip;
      if (xps && xps->Match(psid)) {
         fProofServs.erase(ip);
         break;
      }
   }
}

//______________________________________________________________________________
int XrdProofdClient::ResetClientSlot(int ic)
{
   // Reset slot at 'ic'
   XPDLOC(CMGR, "Client::ResetClientSlot")

   TRACE(DBG, "enter: ic: " << ic);

   XrdSysMutexHelper mh(fMutex);

   if (ic >= 0 && ic < (int) fClients.size()) {
      fClients[ic]->Reset();
      return 0;
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

   if (ic >= 0 && ic < (int) fClients.size()) {
      p = fClients[ic]->P();
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

   int len = 0;
   if (msg && (len = strlen(msg)) > 0) {

      // Notify the attached clients
      int ic = 0;
      XrdClientID *cid = 0;
      XrdSysMutexHelper mh(fMutex);
      for (ic = 0; ic < (int) fClients.size(); ic++) {
         if ((cid = fClients.at(ic)) && cid->P() && cid->P()->ConnType() == kXPD_ClientMaster) {
            XrdProofdResponse *response = cid->R();
            if (response)
               response->Send(kXR_attn, kXPD_srvmsg, (char *) msg, len);
         }
      }
   }
}

//______________________________________________________________________________
void XrdProofdClient::Touch()
{
   // Send a touch the connected clients: this will remotely touch the associated
   // TSocket instance and schedule an asynchronous touch of the client admin file

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
}

//______________________________________________________________________________
void XrdProofdClient::SkipSessionsCheck(std::list<XrdProofdProofServ *> *active)
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
         xps->SetSkipCheck(); // Skip next validity check
         if (active) active->push_back(xps);
      }
   }
   if (active)
      TRACE(HDBG, "found: " << active->size() << " sessions");

   // Over
   return;
}

//______________________________________________________________________________
XrdOucString XrdProofdClient::ExportSessions()
{
   // Return a string describing the existing sessions

   XrdOucString out, buf;

   // Protect from next session check and get the list of actives
   std::list<XrdProofdProofServ *> active;
   SkipSessionsCheck(&active);

   // Fill info
   XrdProofdProofServ *xps = 0;
   out.form("%d", active.size());
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

         if (msg && strlen(msg) > 0)
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

//___________________________________________________________________________
void XrdProofdClient::PostSessionRemoval(int fd, int pid)
{
   // Post removal of session 'pid'
   XPDLOC(CMGR, "Client::PostSessionRemoval")

   TRACE(DBG, "posting session removal to socket "<<fd);

   if (fd > 0) {
      int type = 0;
      if (write(fd, &type, sizeof(type)) !=  sizeof(type)) {
         TRACE(XERR, "problem sending message type on the pipe");
         return;
      }
      if (write(fd, &pid, sizeof(pid)) !=  sizeof(pid)) {
         TRACE(XERR, "problem sending pid on the pipe: "<<pid);
         return;
      }
   }
   // Done
   return;
}

