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
// XrdProofdClientMgr                                                   //
//                                                                      //
// Author: G. Ganis, CERN, 2008                                         //
//                                                                      //
// Class managing clients.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "XrdProofdPlatform.h"

#include "XrdProofdXrdVers.h"
#if ROOTXRDVERS < ROOT_OldXrdOuc
#  define XPD_LOG_01 OUC_LOG_01
#else
#  define XPD_LOG_01 SYS_LOG_01
#endif

#include "XpdSysError.h"

#include "Xrd/XrdBuffer.hh"
#ifdef ROOT_XrdFour
#include "XrdNet/XrdNetAddrInfo.hh"
#endif
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdSys/XrdSysPlugin.hh"

#include "XrdProofdClient.h"
#include "XrdProofdClientMgr.h"
#include "XrdProofdManager.h"
#include "XrdProofdProtocol.h"
#include "XrdProofGroup.h"
#include "XrdProofdProofServ.h"
#include "XrdProofdProofServMgr.h"
#include "XrdROOT.h"

// Tracing utilities
#include "XrdProofdTrace.h"

static XpdManagerCron_t fManagerCron;

// Security handle
typedef XrdSecService *(*XrdSecServLoader_t)(XrdSysLogger *, const char *cfn);

//--------------------------------------------------------------------------
//
// XrdProofdClientCron
//
// Client manager thread
//
////////////////////////////////////////////////////////////////////////////////
/// This is an endless loop to check the system periodically or when
/// triggered via a message in a dedicated pipe

void *XrdProofdClientCron(void *p)
{
   XPDLOC(CMGR, "ClientCron")

   XpdManagerCron_t *mc = (XpdManagerCron_t *)p;
   XrdProofdClientMgr *mgr = mc->fClientMgr;
   if (!(mgr)) {
      TRACE(REQ, "undefined client manager: cannot start");
      return (void *)0;
   }
   XrdProofdProofServMgr *smgr = mc->fSessionMgr;
   if (!(smgr)) {
      TRACE(REQ, "undefined session manager: cannot start");
      return (void *)0;
   }

   // Time of last session check
   int lastcheck = time(0), ckfreq = mgr->CheckFrequency(), deltat = 0;
   while(1) {
      // We wait for processes to communicate a session status change
      if ((deltat = ckfreq - (time(0) - lastcheck)) <= 0)
         deltat = ckfreq;
      int pollRet = mgr->Pipe()->Poll(deltat);

      if (pollRet > 0) {
         // Read message
         XpdMsg msg;
         int rc = 0;
         if ((rc = mgr->Pipe()->Recv(msg)) != 0) {
            XPDERR("problems receiving message; errno: "<<-rc);
            continue;
         }
         // Parse type
         //XrdOucString buf;
         if (msg.Type() == XrdProofdClientMgr::kClientDisconnect) {
            // obsolete
            TRACE(XERR, "obsolete type: XrdProofdClientMgr::kClientDisconnect");
         } else {
            TRACE(XERR, "unknown type: "<<msg.Type());
            continue;
         }
      } else {
         // Run regular checks
         mgr->CheckClients();
         // Remember when ...
         lastcheck = time(0);
      }
   }

   // Should never come here
   return (void *)0;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor

XrdProofdClientMgr::XrdProofdClientMgr(XrdProofdManager *mgr,
                                       XrdProtocol_Config *pi, XrdSysError *e)
                  : XrdProofdConfig(pi->ConfigFN, e), fSecPlugin(0)
{
   XPDLOC(CMGR, "XrdProofdClientMgr")

   fMutex = new XrdSysRecMutex;
   fMgr = mgr;
   fCIA = 0;
   fNDisconnected = 0;
   fReconnectTimeOut = 300;
   // Defaults can be changed via 'clientmgr'
   fActivityTimeOut = 1200;
   fCheckFrequency = 60;

   // Init pipe for manager thread
   if (!fPipe.IsValid()) {
      TRACE(XERR, "unable to generate the pipe");
      return;
   }

   // Configuration directives
   RegisterDirectives();
}

////////////////////////////////////////////////////////////////////////////////
/// Register directives for configuration

void XrdProofdClientMgr::RegisterDirectives()
{
   Register("clientmgr", new XrdProofdDirective("clientmgr", this, &DoDirectiveClass));
   Register("seclib", new XrdProofdDirective("seclib",
                                   (void *)&fSecLib, &DoDirectiveString, 0));
   Register("reconnto", new XrdProofdDirective("reconnto",
                               (void *)&fReconnectTimeOut, &DoDirectiveInt));
}

////////////////////////////////////////////////////////////////////////////////
/// Update the priorities of the active sessions.

int XrdProofdClientMgr::DoDirective(XrdProofdDirective *d,
                                    char *val, XrdOucStream *cfg, bool rcf)
{
   XPDLOC(SMGR, "ClientMgr::DoDirective")

   if (!d)
      // undefined inputs
      return -1;

   if (d->fName == "clientmgr") {
      return DoDirectiveClientMgr(val, cfg, rcf);
   }
   TRACE(XERR,"unknown directive: "<<d->fName);
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Process 'clientmgr' directive
/// eg: xpd.clientmgr checkfq:120 activityto:600

int XrdProofdClientMgr::DoDirectiveClientMgr(char *val, XrdOucStream *cfg, bool)
{
   XPDLOC(SMGR, "ClientMgr::DoDirectiveClientMgr")

   if (!val || !cfg)
      // undefined inputs
      return -1;

   int checkfq = -1;
   int activityto = -1;

   while (val) {
      XrdOucString tok(val);
      if (tok.beginswith("checkfq:")) {
         tok.replace("checkfq:", "");
         checkfq = strtol(tok.c_str(), 0, 10);
      } else if (tok.beginswith("activityto:")) {
         tok.replace("activityto:", "");
         activityto = strtol(tok.c_str(), 0, 10);
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
   fActivityTimeOut = (XPD_LONGOK(activityto) && activityto > 0) ? activityto : fActivityTimeOut;

   XrdOucString msg;
   XPDFORM(msg, "checkfq: %d s, activityto: %d s", fCheckFrequency, fActivityTimeOut);
   TRACE(ALL, msg);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Run configuration and parse the entered config directives.
/// Return 0 on success, -1 on error

int XrdProofdClientMgr::Config(bool rcf)
{
   XPDLOC(CMGR, "ClientMgr::Config")

   // Run first the configurator
   if (XrdProofdConfig::Config(rcf) != 0) {
      XPDERR("problems parsing file ");
      return -1;
   }

   XrdOucString msg;
   msg = (rcf) ? "re-configuring" : "configuring";
   TRACE(ALL, msg.c_str());

   // Admin paths
   fClntAdminPath = fMgr->AdminPath();
   fClntAdminPath += "/clients";

   // Make sure they exist
   XrdProofUI ui;
   XrdProofdAux::GetUserInfo(fMgr->EffectiveUser(), ui);
   if (XrdProofdAux::AssertDir(fClntAdminPath.c_str(), ui, 1) != 0) {
      XPDERR("unable to assert the clients admin path: "<<fClntAdminPath);
      fClntAdminPath = "";
      return -1;
   }
   TRACE(ALL, "clients admin path set to: "<<fClntAdminPath);

   // Init place holders for previous active clients, if any
   if (ParsePreviousClients(msg) != 0) {
      XPDERR("problems parsing previous active clients: "<<msg);
   }

   // Initialize the security system if this is wanted
   if (!rcf) {
      if (fSecLib.length() <= 0) {
         TRACE(ALL, "XRD seclib not specified; strong authentication disabled");
      } else {
         if (!(fCIA = LoadSecurity())) {
            XPDERR("unable to load security system.");
            return -1;
         }
         TRACE(ALL, "security library loaded");
      }
   }

   if (rcf) {
      // Re-assign groups
      if (fMgr->GroupsMgr() && fMgr->GroupsMgr()->Num() > 0) {
         std::list<XrdProofdClient *>::iterator pci;
         for (pci = fProofdClients.begin(); pci != fProofdClients.end(); ++pci)
            (*pci)->SetGroup(fMgr->GroupsMgr()->GetUserGroup((*pci)->User())->Name());
      }
   }

   if (!rcf) {
      // Start cron thread
      pthread_t tid;
      // Fill manager pointers structure
      fManagerCron.fClientMgr = this;
      fManagerCron.fSessionMgr = fMgr->SessionMgr();
      if (XrdSysThread::Run(&tid, XrdProofdClientCron,
                           (void *)&fManagerCron, 0, "ClientMgr cron thread") != 0) {
         XPDERR("could not start cron thread");
         return 0;
      }
      TRACE(ALL, "cron thread started");
   }

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process a login request

int XrdProofdClientMgr::Login(XrdProofdProtocol *p)
{
   XPDLOC(CMGR, "ClientMgr::Login")

   int rc = 0;
   XPD_SETRESP(p, "Login");

   TRACEP(p, HDBG, "enter");

   // If this server is explicitly required to be a worker node or a
   // submaster, check whether the requesting host is allowed to connect
   if (p->Request()->login.role[0] != 'i' &&
      (fMgr->SrvType() == kXPD_MasterWorker || fMgr->SrvType() == kXPD_Master)) {
      if (!fMgr->CheckMaster(p->Link()->Host())) {
         TRACEP(p, XERR,"master not allowed to connect - "
                        "ignoring request ("<<p->Link()->Host()<<")");
         response->Send(kXR_InvalidRequest,
                            "master not allowed to connect - request ignored");
         return 0;
      }
   }

   // Get user and group names for the entity requiring to login
   int i, pid;
   XrdOucString uname, gname, emsg;

   // If this is the second call (after authentication) we just need mapping
   if (p->Status() == XPD_NEED_MAP) {

      // Finalize the login, checking the if username is allowed to use the facility.
      // The username could have been set as part of the authentication process (for
      // example via a user mapping funtion or a grid-map file), so these checks have
      // to be done at this level.
      // (The XrdProofdClient instance is created in here, if everything else goes well)
      int rccc = 0;
      if ((rccc = CheckClient(p, 0, emsg)) != 0) {
         TRACEP(p, XERR, emsg);
         XErrorCode rcode = (rccc == -2) ? (XErrorCode) kXR_NotAuthorized
                                         : (XErrorCode) kXR_InvalidRequest;
         response->Send(rcode, emsg.c_str());
         response->Send(kXR_InvalidRequest, emsg.c_str());
         return 0;
      }

      // Acknowledge the client
      response->Send();
      p->SetStatus(XPD_LOGGEDIN);
      return MapClient(p, 0);
   }

   // Make sure the user is not already logged in
   if ((p->Status() & XPD_LOGGEDIN)) {
      response->Send(kXR_InvalidRequest, "duplicate login; already logged in");
      return 0;
   }

   TRACE(ALL," hostname: '"<<p->Link()->Host()<<"'");
   //
   // Check if in any-server mode (localhost connections always are)
   bool anyserver = (fMgr->SrvType() == kXPD_AnyServer ||
                     !strcmp(p->Link()->Host(), "localhost") ||
                     !strcmp(p->Link()->Host(), "127.0.0.0")) ? 1 : 0;

   // Find out the connection type: 'i', internal, means this is a proofsrv calling back.
   bool needauth = 0;
   bool ismaster = (fMgr->SrvType() == kXPD_TopMaster || fMgr->SrvType() == kXPD_Master) ? 1 : 0;
   switch (p->Request()->login.role[0]) {
   case 'A':
      p->SetConnType(kXPD_Admin);
      response->SetTag("adm");
      break;
   case 'i':
      p->SetConnType(kXPD_Internal);
      response->SetTag("int");
      break;
   case 'M':
      if (anyserver || ismaster) {
         p->SetConnType(kXPD_ClientMaster);
         needauth = 1;
         response->SetTag("m2c");
      } else {
         TRACEP(p, XERR,"top master mode not allowed - ignoring request");
         response->Send(kXR_InvalidRequest,
                            "Server not allowed to be top master - ignoring request");
         return 0;
      }
      break;
   case 'm':
      if (anyserver || ismaster) {
         p->SetConnType(kXPD_MasterMaster);
         needauth = 1;
         response->SetTag("m2m");
      } else {
         TRACEP(p, XERR,"submaster mode not allowed - ignoring request");
         response->Send(kXR_InvalidRequest,
                             "Server not allowed to be submaster - ignoring request");
         return 0;
      }
      break;
   case 'L':
      if (fMgr->SrvType() == kXPD_AnyServer || fMgr->RemotePLite()) {
         p->SetConnType(kXPD_MasterMaster);
         needauth = 1;
         response->SetTag("m2l");
         p->Request()->login.role[0] = 'm';
      } else {
         TRACEP(p, XERR,"PLite submaster mode not allowed - ignoring request");
         response->Send(kXR_InvalidRequest,
                             "Server not allowed to be PLite submaster - ignoring request");
         return 0;
      }
      break;
   case 's':
      if (anyserver || fMgr->SrvType() == kXPD_MasterWorker) {
         p->SetConnType(kXPD_MasterWorker);
         needauth = 1;
         response->SetTag("w2m");
      } else {
         TRACEP(p, XERR,"worker mode not allowed - ignoring request");
         response->Send(kXR_InvalidRequest,
                        "Server not allowed to be worker - ignoring request");
         return 0;
      }
      break;
   default:
      TRACEP(p, XERR, "unknown mode: '" << p->Request()->login.role[0] <<"'");
      response->Send(kXR_InvalidRequest, "Server type: invalide mode");
      return rc;
   }
   response->SetTraceID();

   // Unmarshall the data: process ID
   pid = (int)ntohl(p->Request()->login.pid);
   p->SetPid(pid);

   // Username
   char un[9];
   for (i = 0; i < (int)sizeof(un)-1; i++) {
      if (p->Request()->login.username[i] == '\0' || p->Request()->login.username[i] == ' ')
         break;
      un[i] = p->Request()->login.username[i];
   }
   un[i] = '\0';
   uname = un;

   // Longer usernames are in the attached buffer
   if (uname == "?>buf") {
      // Attach to buffer
      char *buf = p->Argp()->buff;
      int   len = p->Request()->login.dlen;
      // Extract username
      uname.assign(buf,0,len-1);
      int iusr = uname.find("|usr:");
      if (iusr == -1) {
         TRACEP(p, XERR,"long user name not found");
         response->Send(kXR_InvalidRequest, "long user name not found");
         return 0;
      }
      uname.erase(0,iusr+5);
      uname.erase(uname.find("|"));
   }

   // Extract group name, if specified (syntax is uname[:gname])
   int ig = uname.find(":");
   if (ig != -1) {
      gname.assign(uname, ig+1);
      uname.erase(ig);
      TRACEP(p, DBG, "requested group: "<<gname);
      // Save the requested group info in the protocol instance
      p->SetGroupIn(gname.c_str());
   }

   // Save the incoming username setting in the protocol instance
   p->SetUserIn(uname.c_str());

   // Establish IDs for this link
   p->Link()->setID(uname.c_str(), pid);
   p->SetTraceID();
   response->SetTraceID();
   p->SetClntCapVer(p->Request()->login.capver[0]);

   // Get the security token for this link. We will either get a token, a null
   // string indicating host-only authentication, or a null indicating no
   // authentication. We can then optimize of each case.
   if (needauth && fCIA) {
#ifdef ROOT_XrdFour
      const char *pp = fCIA->getParms(i, (XrdNetAddrInfo *) p->Link()->NetAddr());
#else
      const char *pp = fCIA->getParms(i, p->Link()->Name());
#endif
      if (pp && i ) {
         response->SendI((kXR_int32)XPROOFD_VERSBIN, (void *)pp, i);
         p->SetStatus((XPD_NEED_MAP | XPD_NEED_AUTH));
         return 0;
      } else if (pp) {
         p->SetAuthEntity();
      }
   }
   // Check the client at this point; the XrdProofdClient instance is created
   // in here, if everything else goes well
   int rccc = 0;
   if ((rccc = CheckClient(p, p->UserIn(), emsg)) != 0) {
      TRACEP(p, XERR, emsg);
      XErrorCode rcode = (rccc == -2) ? (XErrorCode) kXR_NotAuthorized
                                       : (XErrorCode) kXR_InvalidRequest;
      response->Send(rcode, emsg.c_str());
      return 0;
   }
   rc = response->SendI((kXR_int32)XPROOFD_VERSBIN);
   p->SetStatus(XPD_LOGGEDIN);

   // Map the client
   return MapClient(p, 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Perform checks on the client username. In case authentication is required
/// this is called afetr authentication.
/// Return 0 on success; on error, return -1 .

int XrdProofdClientMgr::CheckClient(XrdProofdProtocol *p,
                                    const char *user, XrdOucString &emsg)
{
   XPDLOC(CMGR, "ClientMgr::CheckClient")

   if (!p) {
      emsg = "protocol object undefined!";
      return -1;
   }

   XrdOucString uname(user), gname(p->GroupIn());
   if (!user) {
      if (p && p->AuthProt() && strlen(p->AuthProt()->Entity.name) > 0) {
         uname = p->AuthProt()->Entity.name;
      } else {
         emsg = "username not passed and not available in the protocol security entity - failing";
         return -1;
      }
   }

   // Check if user belongs to a group
   XrdProofGroup *g = 0;
   if (fMgr->GroupsMgr() && fMgr->GroupsMgr()->Num() > 0) {
      if (gname.length() > 0) {
         g = fMgr->GroupsMgr()->GetGroup(gname.c_str());
         if (!g) {
            XPDFORM(emsg, "group unknown: %s", gname.c_str());
            return -1;
         } else if (strncmp(g->Name(),"default",7) &&
                   !g->HasMember(uname.c_str())) {
            XPDFORM(emsg, "user %s is not member of group %s", uname.c_str(), gname.c_str());
            return -1;
         } else {
            if (TRACING(DBG)) {
               TRACEP(p, DBG,"group: "<<gname<<" found");
               g->Print();
            }
         }
      } else {
         g = fMgr->GroupsMgr()->GetUserGroup(uname.c_str());
         gname = g ? g->Name() : "default";
      }
   }

   // Here we check if the user is allowed to use the system
   // If not, we fail.
   XrdProofUI ui;
   bool su;
   if (fMgr->CheckUser(uname.c_str(), gname.c_str(), ui, emsg, su) != 0) {
      if (emsg.length() <= 0)
         XPDFORM(emsg, "Controlled access: user '%s', group '%s' not allowed to connect",
                       uname.c_str(), gname.c_str());
      return -2;
   }
   if (su) {
      // Update superuser flag
      p->SetSuperUser(su);
      TRACEP(p, DBG, "request from entity: "<<uname<<":"<<gname<<" (privileged)");
   } else {
      TRACEP(p, DBG, "request from entity: "<<uname<<":"<<gname);
   }

   // Attach-to / Create the XrdProofdClient instance for this user: if login
   // fails this will be removed at a later stage
   XrdProofdClient *c = GetClient(uname.c_str(), gname.c_str());
   if (c) {
      if (!c->ROOT())
         c->SetROOT(fMgr->ROOTMgr()->DefaultVersion());
      if (c->IsValid()) {
         // Set the group, if any
         c->SetGroup(gname.c_str());
      }
   } else {
      emsg = "unable to instantiate object for client ";
      emsg += uname;
      return -1;
   }
   // Save into the protocol instance
   p->SetClient(c);

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Process a login request

int XrdProofdClientMgr::MapClient(XrdProofdProtocol *p, bool all)
{
   XPDLOC(CMGR, "ClientMgr::MapClient")

   int rc = 0;
   XPD_SETRESP(p, "MapClient");

   XrdOucString msg;

   TRACEP(p, HDBG, "all: "<< all);

   // Attach to the client
   XrdProofdClient *pc = p->Client();

   // Map the existing session, if found
   if (!pc || !pc->IsValid()) {
      if (pc) {
         {  // Remove from the list
            XrdSysMutexHelper mh(fMutex);
            fProofdClients.remove(pc);
         }
         SafeDelete(pc);
         p->SetClient(0);
      }
      TRACEP(p, DBG, "cannot find valid instance of XrdProofdClient");
      response->Send(kXP_ServerError,
                     "MapClient: cannot find valid instance of XrdProofdClient");
      return 0;
   }

   // Flag for internal connections
   bool proofsrv = ((p->ConnType() == kXPD_Internal) && all) ? 1 : 0;

   // If call back from proofsrv, find out the target session
   short int psid = -1;
   char protver = -1;
   short int clientvers = -1;
   if (proofsrv) {
      memcpy(&psid, (const void *)&(p->Request()->login.reserved[0]), 2);
      if (psid < 0) {
         TRACEP(p, XERR, "proofsrv callback: sent invalid session id");
         response->Send(kXR_InvalidRequest,
                        "MapClient: proofsrv callback: sent invalid session id");
         return 0;
      }
      protver = p->Request()->login.capver[0];
      TRACEP(p, DBG, "proofsrv callback for session: " <<psid);
   } else {
      // Get PROOF version run by client
      memcpy(&clientvers, (const void *)&(p->Request()->login.reserved[0]), 2);
      TRACEP(p, DBG, "PROOF version run by client: " <<clientvers);
   }

   // If proofsrv, locate the target session
   if (proofsrv) {
      XrdProofdProofServ *psrv = pc->GetServer(psid);
      if (!psrv) {
         TRACEP(p, XERR, "proofsrv callback: wrong target session: "<<psid<<" : protocol error");
         response->Send(kXP_nosession, "MapClient: proofsrv callback:"
                                       " wrong target session: protocol error");
         return -1;
      } else {
         // Set the protocol version
         psrv->SetProtVer(protver);
         // Assign this link to it
         XrdProofdResponse *resp = p->Response(1);
         if (!resp) {
            TRACEP(p, XERR, "proofsrv callback: could not get XrdProofdResponse object");
            response->Send(kXP_nosession, "MapClient: proofsrv callback: memory issue?");
            return -1;            
         }
         psrv->SetConnection(resp);
         psrv->SetValid(1);
         // Set Trace ID
         XrdOucString tid;
         XPDFORM(tid, "xrd->%s", psrv->Ordinal());
         resp->SetTag(tid.c_str());
         resp->SetTraceID();
         TRACEI(resp->TraceID(), DBG, "proofsrv callback: link assigned to target session "<<psid);
      }
   } else {

      // Only one instance of this client can map at a time
      XrdSysMutexHelper mhc(pc->Mutex());

      // Make sure that the version is filled correctly (if an admin operation
      // was run before this may still be -1 on workers)
      p->SetProofProtocol(clientvers);

      // Check if we have already an ID for this client from a previous connection
      XrdOucString cpath;
      int cid = -1;
      if ((cid = CheckAdminPath(p, cpath, msg)) >= 0) {
         // Assign the slot
         pc->SetClientID(cid, p);
         // The index of the next free slot will be the unique ID
         p->SetCID(cid);
         // Remove the file indicating that this client was still disconnected
         XrdOucString discpath(cpath, 0, cpath.rfind("/cid"));
         discpath += "/disconnected";
         if (unlink(discpath.c_str()) != 0) {
            XPDFORM(msg, "warning: could not remove %s (errno: %d)", discpath.c_str(), errno);
            TRACEP(p, XERR, msg.c_str());
         }
         // Update counters
         if(fNDisconnected) fNDisconnected--;

      } else {
         // The index of the next free slot will be the unique ID
         p->SetCID(pc->GetClientID(p));
         // Create the client directory in the admin path
         if (CreateAdminPath(p, cpath, msg) != 0) {
            TRACEP(p, XERR, msg.c_str());
            fProofdClients.remove(pc);
            SafeDelete(pc);
            p->SetClient(0);
            response->Send(kXP_ServerError, msg.c_str());
            return 0;
         }
      }
      p->SetAdminPath(cpath.c_str());
      XPDFORM(msg, "client ID and admin paths created: %s", cpath.c_str());
      TRACEP(p, DBG, msg.c_str());

      TRACEP(p, DBG, "CID: "<<p->CID()<<", size: "<<pc->Size());
   }

   // Document this login
   if (!(p->Status() & XPD_NEED_AUTH)) {
      const char *srvtype[6] = {"ANY", "MasterWorker", "MasterMaster",
                                "ClientMaster", "Internal", "Admin"};
      XPDFORM(msg, "user %s logged-in%s; type: %s", pc->User(),
                   p->SuperUser() ? " (privileged)" : "", srvtype[p->ConnType()+1]);
      TRACEP(p, LOGIN, msg);
   }

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Create the client directory in the admin path

int XrdProofdClientMgr::CreateAdminPath(XrdProofdProtocol *p,
                                        XrdOucString &cpath, XrdOucString &emsg)
{
   if (!p || !p->Link()) {
      XPDFORM(emsg, "invalid inputs (p: %p)", p);
      return -1;
   }

   // Create link ID
   XrdOucString lid;
   XPDFORM(lid, "%s.%d", p->Link()->Host(), p->Pid());

   // Create the path now
   XPDFORM(cpath, "%s/%s", p->Client()->AdminPath(), lid.c_str());
   XrdProofUI ui;
   XrdProofdAux::GetUserInfo(fMgr->EffectiveUser(), ui);
   if (XrdProofdAux::AssertDir(cpath.c_str(), ui, 1) != 0) {
      XPDFORM(emsg, "error creating client admin path: %s", cpath.c_str());
      return -1;
   }
   // Save client ID for full recovery
   cpath += "/cid";
   FILE *fcid = fopen(cpath.c_str(), "w");
   if (fcid) {
      fprintf(fcid, "%d", p->CID());
      fclose(fcid);
   } else {
      XPDFORM(emsg, "error creating file for client id: %s", cpath.c_str());
      return -1;
   }
   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Check the old-clients admin for an existing entry for this client and
/// read the client ID;

int XrdProofdClientMgr::CheckAdminPath(XrdProofdProtocol *p,
                                       XrdOucString &cidpath, XrdOucString &emsg)
{
   emsg = "";
   if (!p) {
      XPDFORM(emsg, "CheckAdminPath: invalid inputs (p: %p)", p);
      return -1;
   }

   // Create link ID
   XrdOucString lid;
   XPDFORM(lid, "%s.%d", p->Link()->Host(), p->Pid());

   // Create the path now
   XPDFORM(cidpath, "%s/%s/cid", p->Client()->AdminPath(), lid.c_str());

   // Create disconnected path
   XrdOucString discpath;
   XPDFORM(discpath, "%s/%s/disconnected", p->Client()->AdminPath(), lid.c_str());

   // Check last access time of disconnected if available, otherwise cid
   bool expired = false;
   struct stat st;
   int rc = stat(discpath.c_str(), &st);
   if (rc != 0) rc = stat(cidpath.c_str(), &st);
   if (rc != 0 || (expired = ((int)(time(0) - st.st_atime) > fReconnectTimeOut))) {
      if (expired || (rc != 0 && errno != ENOENT)) {
         // Remove the file
         cidpath.replace("/cid", "");
         if (expired)
            XPDFORM(emsg, "CheckAdminPath: reconnection timeout expired: remove %s ",
                          cidpath.c_str());
         else
            XPDFORM(emsg, "CheckAdminPath: problems stat'ing %s (errno: %d): remove ",
                          cidpath.c_str(), errno);
         if (XrdProofdAux::RmDir(cidpath.c_str()) != 0)
            emsg += ": failure!";
      } else {
         XPDFORM(emsg, "CheckAdminPath: no such file %s", cidpath.c_str());
      }
      return -1;
   }

   // Get the client ID for full recovery
   return XrdProofdAux::GetIDFromPath(cidpath.c_str(), emsg);
}

////////////////////////////////////////////////////////////////////////////////
/// Client entries for the clients still connected when the daemon terminated

int XrdProofdClientMgr::ParsePreviousClients(XrdOucString &emsg)
{
   XPDLOC(CMGR, "ClientMgr::ParsePreviousClients")

   emsg = "";

   // Open dir
   DIR *dir = opendir(fClntAdminPath.c_str());
   if (!dir) {
      TRACE(XERR, "cannot open dir "<<fClntAdminPath<<" ; error: "<<errno);
      return -1;
   }
   TRACE(DBG, "creating holders for active clients ...");

   // Scan the active sessions admin path
   XrdOucString usrpath, cidpath, discpath, usr, grp;
   struct dirent *ent = 0;
   while ((ent = (struct dirent *)readdir(dir))) {
      // Skip the basic entries
      if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) continue;
      XPDFORM(usrpath, "%s/%s", fClntAdminPath.c_str(), ent->d_name);
      bool rm = 0;
      struct stat st;
      if (stat(usrpath.c_str(), &st) == 0) {
         usr = ent->d_name;
         grp = usr;
         usr.erase(usr.find('.'));
         grp.erase(0, grp.find('.')+1);
         TRACE(DBG, "found usr: "<<usr<<", grp: "<<grp);
         // Get client instance
         XrdProofdClient *c = GetClient(usr.c_str(), grp.c_str());
         if (!c) {
            XPDFORM(emsg, "ParsePreviousClients: could not get client instance"
                          " for {%s, %s}", usr.c_str(), grp.c_str());
            rm = 1;
         }
         // Open user sub-dir
         DIR *subdir = 0;
         if (!rm && !(subdir = opendir(usrpath.c_str()))) {
            TRACE(XERR, "cannot open dir "<<usrpath<<" ; error: "<<errno);
            rm = 1;
         }
         if (!rm) {
            bool xrm = 0;
            struct dirent *sent = 0;
            while ((sent = (struct dirent *)readdir(subdir))) {
               // Skip the basic entries
               if (!strcmp(sent->d_name, ".") || !strcmp(sent->d_name, "..")) continue;
               if (!strcmp(sent->d_name, "xpdsock")) continue;
               XPDFORM(cidpath, "%s/%s/cid", usrpath.c_str(), sent->d_name);
               // Check last access time
               if (stat(cidpath.c_str(), &st) != 0 ||
                  (int)(time(0) - st.st_atime) > fReconnectTimeOut) {
                  xrm = 1;
               }
               // Read the client ID and and reserve an entry in the related vector
               int cid = (!xrm) ? XrdProofdAux::GetIDFromPath(cidpath.c_str(), emsg) : -1;
               if (cid < 0)
                  xrm = 1;
               // Reserve an entry in the related vector
               if (!xrm && c->ReserveClientID(cid) != 0)
                  xrm = 1;
               // Flag this as disconnected
               if (!xrm) {
                  XPDFORM(discpath, "%s/%s/disconnected", usrpath.c_str(), sent->d_name);
                  FILE *fd = fopen(discpath.c_str(), "w");
                  if (!fd) {
                     TRACE(XERR, "unable to create path: " <<discpath);
                     xrm = 1;
                  } else {
                     fclose(fd);
                  }
                  if (!xrm)
                     fNDisconnected++;
               }
               // If it did not work remove the entry
               if (xrm) {
                  TRACE(DBG, "removing path: " <<cidpath);
                  cidpath.replace("/cid", "");
                  XPDFORM(emsg, "ParsePreviousClients: failure: remove %s ", cidpath.c_str());
                  if (XrdProofdAux::RmDir(cidpath.c_str()) != 0)
                     emsg += ": failure!";
               }
            }
         }
         if (subdir)
            closedir(subdir);
      } else {
         rm = 1;
      }
      // If it did not work remove the entry
      if (rm) {
         TRACE(DBG, "removing path: " <<usrpath);
         XPDFORM(emsg, "ParsePreviousClients: failure: remove %s ", usrpath.c_str());
         if (XrdProofdAux::RmDir(usrpath.c_str()) != 0)
            emsg += ": failure!";
      }
   }
   // Close the directory
   closedir(dir);

   // Notify the number of previously active clients now offline
   TRACE(DBG, "found "<<fNDisconnected<<" active clients");

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Regular checks of the client admin path; run by the cron job

int XrdProofdClientMgr::CheckClients()
{
   XPDLOC(CMGR, "ClientMgr::CheckClients")

   // Open dir
   DIR *dir = opendir(fClntAdminPath.c_str());
   if (!dir) {
      TRACE(XERR, "cannot open dir "<<fClntAdminPath<<" ; error: "<<errno);
      return -1;
   }
   TRACE(REQ, "checking active clients ...");

   // Scan the active sessions admin path
   int rc = 0;
   XrdOucString usrpath, cidpath, discpath;
   struct dirent *ent = 0;
   while ((ent = (struct dirent *)readdir(dir))) {
      // Skip the basic entries
      if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) continue;
      XPDFORM(usrpath, "%s/%s", fClntAdminPath.c_str(), ent->d_name);
      bool rm = 0;
      XrdProofdClient *c = 0;
      struct stat st, xst;
      if (stat(usrpath.c_str(), &xst) == 0) {
         // Find client instance
         XrdOucString usr, grp;
         XrdProofdAux::ParseUsrGrp(ent->d_name, usr, grp);
         if (!(c = GetClient(usr.c_str(), grp.c_str(), 0))) {
            TRACE(XERR, "instance for client "<<ent->d_name<<" not found!");
            rm = 1;
         }
         // Open user sub-dir
         DIR *subdir = 0;
         if (!rm && !(subdir = opendir(usrpath.c_str()))) {
            TRACE(XERR, "cannot open dir "<<usrpath<<" ; error: "<<errno);
            rm = 1;
         }
         if (!rm) {
            bool xrm = 0, xclose = 0;
            struct dirent *sent = 0;
            while ((sent = (struct dirent *)readdir(subdir))) {
               // Skip the basic entries
               if (!strcmp(sent->d_name, ".") || !strcmp(sent->d_name, "..")) continue;
               if (!strcmp(sent->d_name, "xpdsock")) continue;
               XPDFORM(discpath, "%s/%s/disconnected", usrpath.c_str(), sent->d_name);
               // Client admin path
               XPDFORM(cidpath, "%s/%s/cid", usrpath.c_str(), sent->d_name);
               // Check last access time
               if (stat(cidpath.c_str(), &st) == 0) {
                  // If in disconnected state, check if it needs to be cleaned
                  if (stat(discpath.c_str(), &xst) == 0) {
                     if ((int)(time(0) - st.st_atime) > fReconnectTimeOut) {
                        xrm = 1;
                     }
                  } else {
                     // If required, check the recent activity; if inactive since too long
                     // we ask the client to proof its vitality; but only once: next time
                     // we close the link
                     if (fActivityTimeOut > 0 &&
                         (int)(time(0) - st.st_atime) > fActivityTimeOut) {
                        if (c->Touch() == 1) {
                           // The client was already asked to proof its vitality
                           // during last cycle and it did not do it, so we close
                           // the link
                           xclose = 1;
                        }
                     }
                  }
               } else {
                  // No id info, clean
                  xrm = 1;
               }
               // If inactive since too long, close the associated link
               if (xclose) {
                  // Get the client id
                  XrdOucString emsg;
                  int cid = XrdProofdAux::GetIDFromPath(cidpath.c_str(), emsg);
                  if (cid >= 0) {
                     // Get the XrdProofdProtocol instance
                     XrdProofdProtocol *p = c->GetProtocol(cid);
                     if (p && p->Link()) {
                        // This client will try to reconnect, if alive, so give it
                        // some time by skipping the next sessions check
                        c->SkipSessionsCheck(0, emsg);
                        // Close the associated link; Recycle is called from there
                        p->Link()->Close();
                     } else {
                        TRACE(XERR, "protocol or link associated with ID "<<cid<<" are invalid");
                        xrm = 1;
                     }
                  } else {
                     TRACE(XERR, "could not resolve client id from "<<cidpath);
                     xrm = 1;
                  }
               }
               // If too old remove the entry
               if (xrm) {
                  discpath.replace("/disconnected", "");
                  TRACE(DBG, "removing path "<<discpath);
                  if ((rc = XrdProofdAux::RmDir(discpath.c_str())) != 0) {
                     TRACE(XERR, "problems removing "<<discpath<<"; error: "<<-rc);
                  }
               }
            }
         }
         if (subdir)
            closedir(subdir);
      } else {
         rm = 1;
      }
      // If it did not work remove the entry
      if (rm) {
         TRACE(DBG, "removing path: " <<usrpath);
         if ((rc = XrdProofdAux::RmDir(usrpath.c_str())) != 0) {
            TRACE(XERR, "problems removing "<<usrpath<<"; error: "<<-rc);
         }
      }
   }
   // Close the directory
   closedir(dir);

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Analyse client authentication info

int XrdProofdClientMgr::Auth(XrdProofdProtocol *p)
{
   XPDLOC(CMGR, "ClientMgr::Auth")

   XrdSecCredentials cred;
   XrdSecParameters *parm = 0;
   XrdOucErrInfo     eMsg;
   const char *eText;
   int rc = 1;
   XPD_SETRESP(p, "Auth");

   TRACEP(p, REQ, "enter");

   // Ignore authenticate requests if security turned off
   if (!fCIA)
      return response->Send();
   cred.size   = p->Request()->header.dlen;
   cred.buffer = p->Argp()->buff;

   // If we have no auth protocol, try to get it
   if (!p->AuthProt()) {
      XrdSecProtocol *ap = 0;
#ifdef ROOT_XrdFour
      XrdNetAddr netaddr(p->Link()->NetAddr());
#else
      struct sockaddr netaddr;
      p->Link()->Name(&netaddr);
#endif
      if (!(ap = fCIA->getProtocol(p->Link()->Host(), netaddr, &cred, &eMsg))) {
         eText = eMsg.getErrText(rc);
         TRACEP(p, XERR, "user authentication failed; "<<eText);
         response->Send(kXR_NotAuthorized, eText);
         return -EACCES;
      }
      p->SetAuthProt(ap);
      p->AuthProt()->Entity.tident = p->Link()->ID;
   }
   // Set the wanted login name
   size_t len = strlen("XrdSecLOGINUSER=")+strlen(p->UserIn())+2;
   char *u = new char[len];
   snprintf(u, len, "XrdSecLOGINUSER=%s", p->UserIn());
   putenv(u);

   // Now try to authenticate the client using the current protocol
   XrdOucString namsg;
   if (!(rc = p->AuthProt()->Authenticate(&cred, &parm, &eMsg))) {

      // Make sure that the user name that we want is allowed
      if (p->AuthProt()->Entity.name && strlen(p->AuthProt()->Entity.name) > 0) {
         if (p->UserIn() && strlen(p->UserIn()) > 0) {
            XrdOucString usrs(p->AuthProt()->Entity.name);
            SafeFree(p->AuthProt()->Entity.name);
            XrdOucString usr;
            int from = 0, rcmtc = -1;
            while ((from = usrs.tokenize(usr, from, ',')) != STR_NPOS) {
               // The first one by default, if no match is found
               if (!(p->AuthProt()->Entity.name))
                  p->AuthProt()->Entity.name = strdup(usr.c_str());
               if ((usr == p->UserIn())) {
                  free(p->AuthProt()->Entity.name);
                  p->AuthProt()->Entity.name = strdup(usr.c_str());
                  rcmtc = 0;
                  break;
               }
            }
            if (rcmtc != 0) {
               namsg = "logging as '";
               namsg += p->AuthProt()->Entity.name;
               namsg += "' instead of '";
               namsg += p->UserIn();
               namsg += "' following admin settings";
               TRACEP(p, LOGIN, namsg.c_str());
               namsg.insert("Warning: ", 0);
               response->Send(kXR_attn, kXPD_srvmsg, 2, (char *) namsg.c_str(), namsg.length());
            }
         } else {
            TRACEP(p, XERR, "user name is empty: protocol error?");
         }
      } else {
         TRACEP(p, XERR, "name of the authenticated entity is empty: protocol error?");
         rc = -1;
      }

      if (rc == 0) {
         const char *msg = (p->Status() & XPD_ADMINUSER) ? " admin login as " : " login as ";
         rc = response->Send();
         char status = p->Status();
         status &= ~XPD_NEED_AUTH;
         p->SetStatus(status);
         p->SetAuthEntity(&(p->AuthProt()->Entity));
         if (p->AuthProt()->Entity.name) {
            TRACEP(p, LOGIN, p->Link()->ID << msg << p->AuthProt()->Entity.name);
         } else {
            TRACEP(p, LOGIN, p->Link()->ID << msg << " nobody");
         }
         return rc;
      }
   }

   // If we need to continue authentication, tell the client as much
   if (rc > 0) {
      TRACEP(p, DBG, "more auth requested; sz: " <<(parm ? parm->size : 0));
      if (parm) {
         rc = response->Send(kXR_authmore, parm->buffer, parm->size);
         delete parm;
         return rc;
      }
      if (p->AuthProt()) {
         p->AuthProt()->Delete();
         p->SetAuthProt(0);
      }
      TRACEP(p, XERR, "security requested additional auth w/o parms!");
      response->Send(kXP_ServerError, "invalid authentication exchange");
      return -EACCES;
   }

   // We got an error, bail out
   if (p->AuthProt()) {
      p->AuthProt()->Delete();
      p->SetAuthProt(0);
   }
   eText = (namsg.length() > 0) ? namsg.c_str() : eMsg.getErrText(rc);
   TRACEP(p, XERR, "user authentication failed; "<<eText);
   response->Send(kXR_NotAuthorized, eText);
   return -EACCES;
}

////////////////////////////////////////////////////////////////////////////////
/// Load security framework and plugins, if not already done

XrdSecService *XrdProofdClientMgr::LoadSecurity()
{
   XPDLOC(CMGR, "ClientMgr::LoadSecurity")

   TRACE(REQ, "LoadSecurity");

   const char *cfn = CfgFile();
   const char *seclib = fSecLib.c_str();

   // Make sure the input config file is defined
   if (!cfn) {
      TRACE(XERR, "config file not specified");
      return 0;
   }

   // Create the plug-in instance
   if (!(fSecPlugin = new XrdSysPlugin((fEDest ? fEDest : (XrdSysError *)0), seclib))) {
      TRACE(XERR, "could not create plugin instance for "<<seclib);
      return (XrdSecService *)0;
   }

   // Get the function
   XrdSecServLoader_t ep = (XrdSecServLoader_t) fSecPlugin->getPlugin("XrdSecgetService");
   if (!ep) {
      TRACE(XERR, "could not find 'XrdSecgetService()' in "<<seclib);
      return (XrdSecService *)0;
   }

   // Extract in a temporary file the directives prefixed "xpd.sec..." (filtering
   // out the prefix), "sec.protocol" and "sec.protparm"
   int nd = 0;
   char *rcfn = FilterSecConfig(nd);
   if (!rcfn) {
      SafeDelete(fSecPlugin);
      if (nd == 0) {
         // No directives to be processed
         TRACE(XERR, "no security directives: strong authentication disabled");
         return 0;
      }
      // Failure
      TRACE(XERR, "creating temporary config file");
      return 0;
   }

   // Get the server object
   XrdSecService *cia = 0;
   if (!(cia = (*ep)((fEDest ? fEDest->logger() : (XrdSysLogger *)0), rcfn))) {
      TRACE(XERR, "Unable to create security service object via " << seclib);
      SafeDelete(fSecPlugin);
      unlink(rcfn);
      delete[] rcfn;
      return 0;
   }
   // Notify
   TRACE(ALL, "strong authentication enabled");

   // Unlink the temporary file and cleanup its path
   unlink(rcfn);
   delete[] rcfn;

   // All done
   return cia;
}

////////////////////////////////////////////////////////////////////////////////
/// Grep directives of the form "xpd.sec...", "sec.protparm" and
/// "sec.protocol" from file 'cfn' and save them in a temporary file,
/// stripping off the prefix "xpd." when needed.
/// If any such directory is found, the full path of the temporary file
/// is returned, with the number of directives found in 'nd'.
/// Otherwise 0 is returned and '-errno' specified in nd.
/// The caller has the responsability to unlink the temporary file and
/// to release the memory allocated for the path.

char *XrdProofdClientMgr::FilterSecConfig(int &nd)
{
   XPDLOC(CMGR, "ClientMgr::FilterSecConfig")

   static const char *pfx[] = { "xpd.sec.", "sec.protparm", "sec.protocol", "set" };
   char *rcfn = 0;

   TRACE(REQ, "enter");

   const char *cfn = CfgFile();

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
          !strncmp(lin, pfx[2], strlen(pfx[2])) ||
          !strncmp(lin, pfx[3], strlen(pfx[3]))) {
         // Target directive found
         nd++;
         // Create the output file, if not yet done
         if (!rcfn) {
            size_t len = strlen(fMgr->TMPdir()) + strlen("/xpdcfn_XXXXXX") + 2;
            rcfn = new char[len];
            snprintf(rcfn, len, "%s/xpdcfn_XXXXXX", fMgr->TMPdir());
            mode_t oldum = umask(022);
            if ((fd = mkstemp(rcfn)) < 0) {
               delete[] rcfn;
               nd = (errno > 0) ? -errno : -1;
               fclose(fin);
               rcfn = 0;
               oldum = umask(oldum);
               return rcfn;
            }
            oldum = umask(oldum);
         }
         XrdOucString slin = lin;
         // Strip the prefix "xpd."
         if (slin.beginswith("xpd.")) slin.replace("xpd.","");
         // Make keyword substitution
         fMgr->ResolveKeywords(slin, 0);
         // Write the line to the output file
         XrdProofdAux::Write(fd, slin.c_str(), slin.length());
      }
   }

   // Close files
   fclose(fin);
   if (fd >= 0) close(fd);

   return rcfn;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle request for localizing a client instance for {usr, grp} from the list.
/// Create a new instance, if required; for new instances, use the path at 'sock'
/// for the unix socket, or generate a new one, if sock = 0.

XrdProofdClient *XrdProofdClientMgr::GetClient(const char *usr, const char *grp,
                                               bool create)
{
   XPDLOC(CMGR, "ClientMgr::GetClient")

   TRACE(DBG, "usr: "<< (usr ? usr : "undef")<<", grp:"<<(grp ? grp : "undef"));

   XrdOucString dmsg, emsg;
   XrdProofdClient *c = 0;
   bool newclient = 0;
   std::list<XrdProofdClient *>::iterator i;

   {  XrdSysMutexHelper mh(fMutex);
      for (i = fProofdClients.begin(); i != fProofdClients.end(); ++i) {
         if ((c = *i) && c->Match(usr,grp)) break;
         c = 0;
      }
   }

   if (!c && create) {
      // Is this a potential user?
      XrdProofUI ui;
      bool su;
      if (fMgr->CheckUser(usr, grp, ui, emsg, su) == 0) {
         // Yes: create an (invalid) instance of XrdProofdClient:
         // It would be validated on the first valid login
         ui.fUser = usr;
         ui.fGroup = grp;
         bool full = (fMgr->SrvType() != kXPD_Worker)  ? 1 : 0;
         c = new XrdProofdClient(ui, full, fMgr->ChangeOwn(), fEDest, fClntAdminPath.c_str(), fReconnectTimeOut);
         newclient = 1;
         bool freeclient = 1;
         if (c && c->IsValid()) {
            // Locate and set the group, if any
            if (fMgr->GroupsMgr() && fMgr->GroupsMgr()->Num() > 0) {
               XrdProofGroup *g = fMgr->GroupsMgr()->GetUserGroup(usr, grp);
               if (g) {
                  c->SetGroup(g->Name());
               } else if (TRACING(XERR)) {
                  emsg = "group = "; emsg += grp; emsg += " nor found";
               }
            }
            {  XrdSysMutexHelper mh(fMutex);
               XrdProofdClient *nc = 0;
               for (i = fProofdClients.begin(); i != fProofdClients.end(); ++i) {
                  if ((nc = *i) && nc->Match(usr,grp)) break;
                  nc = 0;
                  newclient = 0;
               }
               if (!nc) {
                  // Add to the list
                  fProofdClients.push_back(c);
                  freeclient = 0;
               }
            }
            if (freeclient) {
               SafeDelete(c);
            } else if (TRACING(DBG)) {
               XPDFORM(dmsg, "instance for {client, group} = {%s, %s} created"
                             " and added to the list (%p)", usr, grp, c);
            }
         } else {
            if (TRACING(XERR)) {
               XPDFORM(dmsg, "instance for {client, group} = {%s, %s} is invalid", usr, grp);
            }
            SafeDelete(c);
         }
      } else {
         if (TRACING(XERR)) {
            XPDFORM(dmsg, "client '%s' unknown or unauthorized: %s", usr, emsg.c_str());
         }
      }
   }

   // Trim the sandbox, if needed
   if (c && !newclient) {
      if (c->TrimSessionDirs() != 0) {
         if (TRACING(XERR)) {
            XPDFORM(dmsg, "problems trimming client '%s' sandbox", usr);
         }
      }
   }

   if (dmsg.length() > 0) {
      if (TRACING(DBG)) {
         TRACE(DBG, dmsg);
      } else {
         if (emsg.length() > 0) TRACE(XERR, emsg);
         TRACE(XERR, dmsg);
      }
   }

   // Over
   return c;
}

////////////////////////////////////////////////////////////////////////////////
/// Broadcast message 'msg' to the connected instances of client 'clnt' or to all
/// connected instances if clnt == 0.

void XrdProofdClientMgr::Broadcast(XrdProofdClient *clnt, const char *msg)
{
   // The clients to notified
   std::list<XrdProofdClient *> *clnts;
   if (!clnt) {
      // The full list
      clnts = &fProofdClients;
   } else {
      clnts = new std::list<XrdProofdClient *>;
      clnts->push_back(clnt);
   }

   // Loop over them
   XrdProofdClient *c = 0;
   std::list<XrdProofdClient *>::iterator i;
   XrdSysMutexHelper mh(fMutex);
   for (i = clnts->begin(); i != clnts->end(); ++i) {
      if ((c = *i))
         c->Broadcast(msg);
   }

   // Cleanup, if needed
   if (clnt) delete clnts;
}

////////////////////////////////////////////////////////////////////////////////
/// Terminate sessions of client 'clnt' or to of all clients if clnt == 0.
/// The list of process IDs having been signalled is returned.

void XrdProofdClientMgr::TerminateSessions(XrdProofdClient *clnt, const char *msg,
                                           int srvtype)
{
   XPDLOC(CMGR, "ClientMgr::TerminateSessions")

   // The clients to notified
   bool all = 0;
   std::list<XrdProofdClient *> *clnts;
   if (!clnt) {
      // The full list
      clnts = &fProofdClients;
      all = 1;
   } else {
      clnts = new std::list<XrdProofdClient *>;
      clnts->push_back(clnt);
   }

   // If cleaning all, we send a unique meassge to scan the dirs in one go;
   // We first broadcast the message to connected clients.
   XrdProofdClient *c = 0;
   std::list<XrdProofdClient *>::iterator i;
   XrdSysMutexHelper mh(fMutex);
   for (i = clnts->begin(); i != clnts->end(); ++i) {
      if ((c = *i)) {
         // Notify the attached clients that we are going to cleanup
         c->Broadcast(msg);
      }
   }

   TRACE(DBG, "cleaning "<<all);

   if (fMgr && fMgr->SessionMgr()) {
      int rc = 0;
      XrdOucString buf;
      XPDFORM(buf, "%s %d", (all ? "all" : clnt->User()), srvtype);
      TRACE(DBG, "posting: "<<buf);
      if ((rc = fMgr->SessionMgr()->Pipe()->Post(XrdProofdProofServMgr::kCleanSessions,
                                                 buf.c_str())) != 0) {
         TRACE(XERR, "problem posting the pipe; errno: "<<-rc);
      }
   }

   // Reset the client instances
   for (i = clnts->begin(); i != clnts->end(); ++i) {
      if ((c = *i))
         c->ResetSessions();
   }

   // Cleanup, if needed
   if (clnt) delete clnts;
}
