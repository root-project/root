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
// XrdProofdAdmin                                                       //
//                                                                      //
// Author: G. Ganis, CERN, 2008                                         //
//                                                                      //
// Envelop class for admin services.                                    //
// Loaded as service by XrdProofdManager.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "XrdProofdPlatform.h"


#include "XpdSysError.h"

#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdScheduler.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdOuc/XrdOucStream.hh"

#include "XrdProofdAdmin.h"
#include "XrdProofdClient.h"
#include "XrdProofdClientMgr.h"
#include "XrdProofdManager.h"
#include "XrdProofdNetMgr.h"
#include "XrdProofdPriorityMgr.h"
#include "XrdProofdProofServMgr.h"
#include "XrdProofdProtocol.h"
#include "XrdProofGroup.h"
#include "XrdProofSched.h"
#include "XrdProofdProofServ.h"
#include "XrdROOT.h"

// Tracing utilities
#include "XrdProofdTrace.h"

//__________________________________________________________________________
static int ExportCpCmd(const char *k, XpdAdminCpCmd *cc, void *s)
{
   // Decrease active session counters on worker w
   XPDLOC(PMGR, "ExportCpCmd")

   XrdOucString *ccs = (XrdOucString *)s;
   if (cc && ccs) {
      if (ccs->length() > 0) *ccs += ",";
      *ccs += k;
      *ccs += ":";
      *ccs += cc->fCmd;
      TRACE(DBG, k <<" : "<<cc->fCmd<<" fmt: '"<<cc->fFmt<<"'");
      // Check next
      return 0;
   }

   // Not enough info: stop
   return 1;
}

//______________________________________________________________________________
XrdProofdAdmin::XrdProofdAdmin(XrdProofdManager *mgr,
                               XrdProtocol_Config *pi, XrdSysError *e)
                  : XrdProofdConfig(pi->ConfigFN, e)
{
   // Constructor

   fMgr = mgr;
   fExportPaths.clear();
   // Map of default copy commands supported / allowed, keyed by the protocol
   fAllowedCpCmds.Add("file", new XpdAdminCpCmd("cp","cp -rp %s %s",1));
   fAllowedCpCmds.Add("root", new XpdAdminCpCmd("xrdcp","xrdcp %s %s",1));
   fAllowedCpCmds.Add("xrd",  new XpdAdminCpCmd("xrdcp","xrdcp %s %s",1));
#if !defined(__APPLE__)
   fAllowedCpCmds.Add("http", new XpdAdminCpCmd("wget","wget %s -O %s",0));
   fAllowedCpCmds.Add("https", new XpdAdminCpCmd("wget","wget %s -O %s",0));
#else
   fAllowedCpCmds.Add("http", new XpdAdminCpCmd("curl","curl %s -o %s",0));
   fAllowedCpCmds.Add("https", new XpdAdminCpCmd("curl","curl %s -o %s",0));
#endif
   fCpCmds = "";
   fAllowedCpCmds.Apply(ExportCpCmd, (void *)&fCpCmds);

   // Configuration directives
   RegisterDirectives();
}

//__________________________________________________________________________
void XrdProofdAdmin::RegisterDirectives()
{
   // Register directives for configuration

   Register("exportpath", new XrdProofdDirective("exportpath", this, &DoDirectiveClass));
   Register("cpcmd", new XrdProofdDirective("cpcmd", this, &DoDirectiveClass));
}

//______________________________________________________________________________
int XrdProofdAdmin::Process(XrdProofdProtocol *p, int type)
{
   // Process admin request
   XPDLOC(ALL, "Admin::Process")

   int rc = 0;
   XPD_SETRESP(p, "Process");

   TRACEP(p, REQ, "req id: " << type << " ("<<
                  XrdProofdAux::AdminMsgType(type) << ")");

   XrdOucString emsg;
   switch (type) {
      case kQueryMssUrl:
         return QueryMssUrl(p);
      case kQuerySessions:
         return QuerySessions(p);
      case kQueryLogPaths:
         return QueryLogPaths(p);
      case kCleanupSessions:
         return CleanupSessions(p);
      case kSendMsgToUser:
         return SendMsgToUser(p);
      case kGroupProperties:
         return SetGroupProperties(p);
      case kGetWorkers:
         return GetWorkers(p);
      case kQueryWorkers:
         return QueryWorkers(p);
      case kQueryROOTVersions:
         return QueryROOTVersions(p);
      case kROOTVersion:
         return SetROOTVersion(p);
      case kSessionAlias:
         return SetSessionAlias(p);
      case kSessionTag:
         return SetSessionTag(p);
      case kReleaseWorker:
         return ReleaseWorker(p);
      case kExec:
         return Exec(p);
      case kGetFile:
         return GetFile(p);
      case kPutFile:
         return PutFile(p);
      case kCpFile:
         return CpFile(p);
      default:
         emsg += "Invalid type: ";
         emsg += type;
         break;
   }

   // Notify invalid request
   response->Send(kXR_InvalidRequest, emsg.c_str());

   // Done
   return 0;
}

//__________________________________________________________________________
int XrdProofdAdmin::Config(bool rcf)
{
   // Run configuration and parse the entered config directives.
   // Return 0 on success, -1 on error
   XPDLOC(ALL, "Admin::Config")

   // Run first the configurator
   if (XrdProofdConfig::Config(rcf) != 0) {
      XPDERR("problems parsing file ");
      return -1;
   }

   XrdOucString msg;
   msg = (rcf) ? "re-configuring" : "configuring";
   TRACE(ALL, msg.c_str());

   // Exported paths
   if (fExportPaths.size() > 0) {
      TRACE(ALL, "additional paths which can be browsed by all users: ");
      std::list<XrdOucString>::iterator is = fExportPaths.begin();
      while (is != fExportPaths.end()) { TRACE(ALL, "   "<<*is); is++; }
   }
   // Allowed / supported copy commands
   TRACE(ALL, "allowed/supported copy commands: "<<fCpCmds);

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::DoDirective(XrdProofdDirective *d,
                                    char *val, XrdOucStream *cfg, bool rcf)
{
   // Update the priorities of the active sessions.
   XPDLOC(SMGR, "Admin::DoDirective")

   if (!d)
      // undefined inputs
      return -1;

   if (d->fName == "exportpath") {
      return DoDirectiveExportPath(val, cfg, rcf);
   } else if (d->fName == "cpcmd") {
      return DoDirectiveCpCmd(val, cfg, rcf);
   }
   TRACE(XERR,"unknown directive: "<<d->fName);
   return -1;
}

//______________________________________________________________________________
int XrdProofdAdmin::DoDirectiveExportPath(char *val, XrdOucStream *cfg, bool)
{
   // Process 'exportpath' directives
   // eg: xpd.exportpath /tmp/data /data2/data

   XPDLOC(SMGR, "Admin::DoDirectiveExportPath")

   if (!val || !cfg)
      // undefined inputs
      return -1;

   TRACE(ALL,"val: "<<val);

   while (val) {
      XrdOucString tkns(val), tkn;
      int from = 0;
      while ((from = tkns.tokenize(tkn, from, ' ')) != STR_NPOS) {
         fExportPaths.push_back(tkn);
      }
      // Get next
      val = cfg->GetWord();
   }

   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::DoDirectiveCpCmd(char *val, XrdOucStream *cfg, bool)
{
   // Process 'cpcmd' directives
   // eg: xpd.cpcmd alien aliencp  fmt:"%s %s" put:0

   XPDLOC(SMGR, "Admin::DoDirectiveCpCmd")

   if (!val || !cfg)
      // undefined inputs
      return -1;

   XrdOucString proto, cpcmd, fmt;
   bool canput = 0, isfmt = 0, rm = 0;

   while (val) {
      XrdOucString tkn(val);
      if (proto.length() <= 0) {
         proto = tkn;
         if (proto.beginswith('-')) {
            rm = 1;
            proto.erase(0, 1);
            break;
         }
      } else if (cpcmd.length() <= 0) {
         cpcmd = tkn;
      } else if (tkn.beginswith("put:")) {
         isfmt = 0;
         if (tkn == "put:1") canput = 1;
      } else if (tkn.beginswith("fmt:")) {
         fmt.assign(tkn, 4, -1);
         isfmt = 1;
      } else {
         if (isfmt) {
            fmt += " ";
            fmt += tkn;
         }
      }
      // Get next
      val = cfg->GetWord();
   }

   if (rm) {
      // Remove the related entry
      fAllowedCpCmds.Del(proto.c_str());
   } else if (cpcmd.length() > 0 && fmt.length() > 0) {
      // Add or replace
      fmt.insert(" ", 0);
      fmt.insert(cpcmd, 0);
      fAllowedCpCmds.Rep(proto.c_str(), new XpdAdminCpCmd(cpcmd.c_str(),fmt.c_str(),canput));
   } else {
      TRACE(ALL, "incomplete information: ignoring!");
   }

   // Fill again the export string
   fCpCmds = "";
   fAllowedCpCmds.Apply(ExportCpCmd, (void *)&fCpCmds);

   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::QueryMssUrl(XrdProofdProtocol *p)
{
   // Handle request for the URL to the MSS attached to the cluster.
   // The reply contains also the namespace, i.e. proto://host:port//namespace
   XPDLOC(ALL, "Admin::QueryMssUrl")

   int rc = 0;
   XPD_SETRESP(p, "QueryMssUrl");

   XrdOucString msg = fMgr->PoolURL();
   msg += "/";
   msg += fMgr->NameSpace();

   TRACEP(p, DBG, "sending: "<<msg);

   // Send back to user
   response->Send((void *)msg.c_str(), msg.length()+1);

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::QueryROOTVersions(XrdProofdProtocol *p)
{
   // Handle request for list of ROOT versions
   XPDLOC(ALL, "Admin::QueryROOTVersions")

   int rc = 0;
   XPD_SETRESP(p, "QueryROOTVersions");

   XrdOucString msg = fMgr->ROOTMgr()->ExportVersions(p->Client()->ROOT());

   TRACEP(p, DBG, "sending: "<<msg);

   // Send back to user
   response->Send((void *)msg.c_str(), msg.length()+1);

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::SetROOTVersion(XrdProofdProtocol *p)
{
   // Handle request for changing the default ROOT version
   XPDLOC(ALL, "Admin::SetROOTVersion")

   int rc = 0;
   XPD_SETRESP(p, "SetROOTVersion");

   // Change default ROOT version
   const char *t = p->Argp() ? (const char *) p->Argp()->buff : "default";
   int len = p->Argp() ? p->Request()->header.dlen : strlen("default");
   XrdOucString tag(t,len);

   // If a user name is given separate it out and check if
   // we can do the operation
   XrdOucString usr;
   if (tag.beginswith("u:")) {
      usr = tag;
      usr.erase(usr.rfind(' '));
      usr.replace("u:","");
      // Isolate the tag
      tag.erase(0,tag.find(' ') + 1);
   }
   TRACEP(p, REQ, "usr: "<<usr<<", version tag: "<< tag);

   // If the action is requested for a user different from us we
   // must be 'superuser'
   XrdProofdClient *c = p->Client();
   XrdOucString grp;
   if (usr.length() > 0) {
      // Separate group info, if any
      if (usr.find(':') != STR_NPOS) {
         grp = usr;
         grp.erase(grp.rfind(':'));
         usr.erase(0,usr.find(':') + 1);
      } else {
         XrdProofGroup *g =
            (fMgr->GroupsMgr()) ? fMgr->GroupsMgr()->GetUserGroup(usr.c_str()) : 0;
         grp = g ? g->Name() : "default";
      }
      if (usr != p->Client()->User()) {
         if (!p->SuperUser()) {
            usr.insert("not allowed to change settings for usr '", 0);
            usr += "'";
            TRACEP(p, XERR, usr.c_str());
            response->Send(kXR_InvalidRequest, usr.c_str());
            return 0;
         }
         // Lookup the list
         if (!(c = fMgr->ClientMgr()->GetClient(usr.c_str(), grp.c_str()))) {
            // No: fail
            XrdOucString emsg("user not found or not allowed: ");
            emsg += usr;
            TRACEP(p, XERR, emsg.c_str());
            response->Send(kXR_InvalidRequest, emsg.c_str());
            return 0;
         }
      }
   }

   // Search in the list
   XrdROOT *r = fMgr->ROOTMgr()->GetVersion(tag.c_str());
   bool ok = r ? 1 : 0;
   if (!r && tag == "default") {
      // If not found we may have been requested to set the default version
      r = fMgr->ROOTMgr()->DefaultVersion();
      ok = r ? 1 : 0;
   }

   if (ok) {
      // Save the version in the client instance
      c->SetROOT(r);
      // Notify
      TRACEP(p, DBG, "default changed to "<<c->ROOT()->Tag()<<
                   " for {client, group} = {"<<usr<<", "<<grp<<"} ("<<c<<")");
      // Forward down the tree, if not leaf
      if (fMgr->SrvType() != kXPD_Worker) {
         XrdOucString buf("u:");
         buf += c->UI().fUser;
         buf += " ";
         buf += tag;
         int type = ntohl(p->Request()->proof.int1);
         fMgr->NetMgr()->Broadcast(type, buf.c_str(), p->Client()->User(), response);
      }
      // Acknowledge user
      response->Send();
   } else {
      tag.insert("tag '", 0);
      tag += "' not found in the list of available ROOT versions";
      TRACEP(p, XERR, tag.c_str());
      response->Send(kXR_InvalidRequest, tag.c_str());
   }

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::QueryWorkers(XrdProofdProtocol *p)
{
   // Handle request for getting the list of potential workers
   XPDLOC(ALL, "Admin::QueryWorkers")

   int rc = 0;
   XPD_SETRESP(p, "QueryWorkers");

   // Send back a list of potentially available workers
   XrdOucString sbuf(1024);
   fMgr->ProofSched()->ExportInfo(sbuf);

   // Send buffer
   char *buf = (char *) sbuf.c_str();
   int len = sbuf.length() + 1;
   TRACEP(p, DBG, "sending: "<<buf);

   // Send back to user
   response->Send(buf, len);

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::GetWorkers(XrdProofdProtocol *p)
{
   // Handle request for getting the best set of workers
   XPDLOC(ALL, "Admin::GetWorkers")

   int rc = 0;
   XPD_SETRESP(p, "GetWorkers");

   // Unmarshall the data
   int psid = ntohl(p->Request()->proof.sid);

   // Find server session
   XrdProofdProofServ *xps = 0;
   if (!p->Client() || !(xps = p->Client()->GetServer(psid))) {
      TRACEP(p, XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"session ID not found");
      return 0;
   }
   int pid = xps->SrvPID();
   TRACEP(p, REQ, "request from session "<<pid);

   // We should query the chosen resource provider
   XrdOucString wrks("");

   // Read the message associated with the request; needs to do like this because
   // of a bug in the XrdOucString constructor when length is 0
   XrdOucString msg;
   if (p->Request()->header.dlen > 0)
      msg.assign((const char *) p->Argp()->buff, 0, p->Request()->header.dlen);
   if (fMgr->GetWorkers(wrks, xps, msg.c_str()) < 0 ) {
      // Something wrong
      response->Send(kXR_InvalidRequest, "GetWorkers failed");
      return 0;
   }

   // Send buffer
   // In case the session was enqueued, pass an empty list.
   char *buf = (char *) wrks.c_str();
   int len = wrks.length() + 1;
   TRACEP(p, DBG, "sending: "<<buf);

   // Send back to user
   if (buf) {
      response->Send(buf, len);
   } else {
      // Something wrong
      response->Send(kXR_InvalidRequest, "GetWorkers failed");
      return 0;
   }

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::SetGroupProperties(XrdProofdProtocol *p)
{
   // Handle request for setting group properties
   XPDLOC(ALL, "Admin::SetGroupProperties")

   int rc = 1;
   XPD_SETRESP(p, "SetGroupProperties");

   // User's group
   int   len = p->Request()->header.dlen;
   char *grp = new char[len+1];
   memcpy(grp, p->Argp()->buff, len);
   grp[len] = 0;
   TRACEP(p, DBG, "request to change priority for group '"<< grp<<"'");

   // Make sure is the current one of the user
   if (strcmp(grp, p->Client()->UI().fGroup.c_str())) {
      TRACEP(p, XERR, "received group does not match the user's one");
      response->Send(kXR_InvalidRequest,
                     "SetGroupProperties: received group does not match the user's one");
      SafeDelArray(grp);
      return 0;
   }

   // The priority value
   int priority = ntohl(p->Request()->proof.int2);

   // Tell the priority manager
   if (fMgr && fMgr->PriorityMgr()) {
      XrdOucString buf;
      XPDFORM(buf, "%s %d", grp, priority);
      if (fMgr->PriorityMgr()->Pipe()->Post(XrdProofdPriorityMgr::kSetGroupPriority,
                                             buf.c_str()) != 0) {
         TRACEP(p, XERR, "problem sending message on the pipe");
         response->Send(kXR_ServerError,
                             "SetGroupProperties: problem sending message on the pipe");
         SafeDelArray(grp);
         return 0;
      }
   }

   // Notify
   TRACEP(p, REQ, "priority for group '"<< grp<<"' has been set to "<<priority);

   SafeDelArray(grp);

   // Acknowledge user
   response->Send();

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::SendMsgToUser(XrdProofdProtocol *p)
{
   // Handle request for sending a message to a user
   XPDLOC(ALL, "Admin::SendMsgToUser")

   int rc = 0;
   XPD_SETRESP(p, "SendMsgToUser");

   // Target client (default us)
   XrdProofdClient *tgtclnt = p->Client();
   XrdProofdClient *c = 0;
   std::list<XrdProofdClient *>::iterator i;

   // Extract the user name, if any
   int len = p->Request()->header.dlen;
   if (len <= 0) {
      // No message: protocol error?
      TRACEP(p, XERR, "no message");
      response->Send(kXR_InvalidRequest,"SendMsgToUser: no message");
      return 0;
   }

   XrdOucString cmsg((const char *)p->Argp()->buff, len);
   XrdOucString usr;
   if (cmsg.beginswith("u:")) {
      // Extract user
      int isp = cmsg.find(' ');
      if (isp != STR_NPOS) {
         usr.assign(cmsg, 2, isp-1);
         cmsg.erase(0, isp+1);
      }
      if (usr.length() > 0) {
         TRACEP(p, REQ, "request for user: '"<<usr<<"'");
         // Find the client instance
         bool clntfound = 0;
         if ((c = fMgr->ClientMgr()->GetClient(usr.c_str(), 0))) {
            tgtclnt = c;
            clntfound = 1;
         }
         if (!clntfound) {
            // No user: protocol error?
            TRACEP(p, XERR, "target client not found");
            response->Send(kXR_InvalidRequest,
                           "SendMsgToUser: target client not found");
            return 0;
         }
      }
   }
   // Recheck message length
   if (cmsg.length() <= 0) {
      // No message: protocol error?
      TRACEP(p, XERR, "no message after user specification");
      response->Send(kXR_InvalidRequest,
                          "SendMsgToUser: no message after user specification");
      return 0;
   }

   // Check if allowed
   if (!p->SuperUser()) {
      if (usr.length() > 0) {
         if (tgtclnt != p->Client()) {
            TRACEP(p, XERR, "not allowed to send messages to usr '"<<usr<<"'");
            response->Send(kXR_InvalidRequest,
                                "SendMsgToUser: not allowed to send messages to specified usr");
            return 0;
         }
      } else {
         TRACEP(p, XERR, "not allowed to send messages to connected users");
         response->Send(kXR_InvalidRequest,
                             "SendMsgToUser: not allowed to send messages to connected users");
         return 0;
      }
   } else {
      if (usr.length() <= 0) tgtclnt = 0;
   }

   // The clients to notified
   fMgr->ClientMgr()->Broadcast(tgtclnt, cmsg.c_str());

   // Acknowledge user
   response->Send();

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::QuerySessions(XrdProofdProtocol *p)
{
   // Handle request for list of sessions
   XPDLOC(ALL, "Admin::QuerySessions")

   int rc = 0;
   XPD_SETRESP(p, "QuerySessions");

   XrdOucString notmsg, msg;
   {  // This is needed to block the session checks
      XpdSrvMgrCreateCnt cnt(fMgr->SessionMgr(), XrdProofdProofServMgr::kProcessCnt);
      msg = p->Client()->ExportSessions(notmsg, response);
   }

   if (notmsg.length() > 0) {
      // Some sessions seem non-responding: notify the client
      response->Send(kXR_attn, kXPD_srvmsg, 0, (char *) notmsg.c_str(), notmsg.length());
   }

   TRACEP(p, DBG, "sending: "<<msg);

   // Send back to user
   response->Send((void *)msg.c_str(), msg.length()+1);

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::QueryLogPaths(XrdProofdProtocol *p)
{
   // Handle request for log paths 
   XPDLOC(ALL, "Admin::QueryLogPaths")

   int rc = 0;
   XPD_SETRESP(p, "QueryLogPaths");

   int ridx = ntohl(p->Request()->proof.int2);
   bool broadcast = (ntohl(p->Request()->proof.int3) == 1) ? 1 : 0;

   // Find out for which session is this request
   XrdOucString stag, master, user, ord, buf;
   int len = p->Request()->header.dlen;
   if (len > 0) {
      buf.assign(p->Argp()->buff,0,len-1);
      int im = buf.find("|master:");
      int iu = buf.find("|user:");
      int io = buf.find("|ord:");
      stag = buf;
      stag.erase(stag.find("|"));
      if (im != STR_NPOS) {
         master.assign(buf, im + strlen("|master:"));
         master.erase(master.find("|"));
      }
      if (iu != STR_NPOS) {
         user.assign(buf, iu + strlen("|user:"));
         user.erase(user.find("|"));
      }
      if (io != STR_NPOS) {
         ord.assign(buf, iu + strlen("|ord:"));
         ord.erase(user.find("|"));
      }
      if (stag.beginswith('*'))
         stag = "";
   }
   TRACEP(p, DBG, "master: "<<master<<", user: "<<user<<", ord: "<<ord<<", stag: "<<stag);

   XrdProofdClient *client = (user.length() > 0) ? 0 : p->Client();
   if (!client)
      // Find the client instance
      client = fMgr->ClientMgr()->GetClient(user.c_str(), 0);
   if (!client) {
      TRACEP(p, XERR, "query sess logs: client for '"<<user<<"' not found");
      response->Send(kXR_InvalidRequest,"QueryLogPaths: query log: client not found");
      return 0;
   }

   XrdOucString tag = (stag == "" && ridx >= 0) ? "last" : stag;
   if (stag == "" && client->Sandbox()->GuessTag(tag, ridx) != 0) {
      TRACEP(p, XERR, "query sess logs: session tag not found");
      response->Send(kXR_InvalidRequest,"QueryLogPaths: query log: session tag not found");
      return 0;
   }

   // Return message
   XrdOucString rmsg;

   if (master.length() <= 0) {
      // The session tag first
      rmsg += tag; rmsg += "|";
      // The pool URL second
      rmsg += fMgr->PoolURL(); rmsg += "|";
   }

   // Locate the local log file
   XrdOucString sdir(client->Sandbox()->Dir());
   sdir += "/session-";
   sdir += tag;

   // Open dir
   DIR *dir = opendir(sdir.c_str());
   if (!dir) {
      XrdOucString msg("cannot open dir ");
      msg += sdir; msg += " (errno: "; msg += errno; msg += ")";
      TRACEP(p, XERR, msg.c_str());
      response->Send(kXR_InvalidRequest, msg.c_str());
      return 0;
   }
   
   // Masters have the .workers file
   XrdOucString wfile(sdir);
   wfile += "/.workers";
   bool ismaster = (access(wfile.c_str(), F_OK) == 0) ? 1 : 0;
   
   // Scan the directory to add the top master (only if top master)
   XrdOucString xo;
   int ilog, idas;
   struct dirent *ent = 0;
   while ((ent = (struct dirent *)readdir(dir))) {         
      xo = ent->d_name;
      bool recordinfo = 0;
      if ((ilog = xo.find(".log")) != STR_NPOS) {
         xo.replace(".log", "");
         if ((idas = xo.find('-')) != STR_NPOS) xo.erase(0, idas + 1);
         if ((idas = xo.find('-')) != STR_NPOS) xo.erase(idas);
         if (ord.length() > 0 && (ord == xo)) {
            recordinfo = 1;
         } else {
            if (ismaster && !broadcast) {
               if (!strncmp(ent->d_name, "master-", 7)) recordinfo = 1;
            } else {
               recordinfo = 1;
            }
         }
         if (recordinfo) {
            rmsg += "|"; rmsg += xo;
            rmsg += " proof://"; rmsg += fMgr->Host(); rmsg += ':';
            rmsg += fMgr->Port(); rmsg += '/';
            rmsg += sdir; rmsg += '/'; rmsg += ent->d_name;
         }
      }
   }
   // Close dir
   closedir(dir);

   // If required and it makes sense, ask the underlying nodes
   if (broadcast && ismaster) {
      XrdOucString msg(tag);
      msg += "|master:";
      msg += fMgr->Host();
      msg += "|user:";
      msg += client->User();
      char *bmst = fMgr->NetMgr()->ReadLogPaths(msg.c_str(), ridx);
      if (bmst) {
         rmsg += bmst;
         free(bmst);
      }
   } else if (ismaster) {
      // Get info from the .workers file
      // Now open the workers file
      FILE *f = fopen(wfile.c_str(), "r");
      if (f) {
         char ln[2048];
         while (fgets(ln, sizeof(ln), f)) {
            if (ln[strlen(ln)-1] == '\n')
               ln[strlen(ln)-1] = 0;
            // Locate status and url
            char *ps = strchr(ln, ' ');
            if (ps) {
               *ps = 0;
               ps++;
               // Locate ordinal
               char *po = strchr(ps, ' ');
               if (po) {
                  po++;
                  // Locate path
                  char *pp = strchr(po, ' ');
                  if (pp) {
                     *pp = 0;
                     pp++;
                     // Record now
                     rmsg += "|"; rmsg += po; rmsg += " ";
                     if (master.length() > 0) {
                        rmsg += master;
                        rmsg += ",";
                     }
                     rmsg += ln; rmsg += '/';
                     rmsg += pp;
                     // Reposition on the file name
                     char *ppl = strrchr(pp, '/');
                     pp = (ppl) ? ppl : pp;
                     // If the line is for a submaster, we have to get the info
                     // about its workers
                     bool ismst = (strstr(pp, "master-")) ? 1 : 0;
                     if (ismst) {
                        XrdClientUrlInfo u((const char *)&ln[0]);
                        XrdOucString msg(stag);
                        msg += "|master:";
                        msg += ln;
                        msg += "|user:";
                        msg += u.User;
                        u.User = p->Client()->User() ? p->Client()->User() : fMgr->EffectiveUser();
                        char *bmst = fMgr->NetMgr()->ReadLogPaths(u.GetUrl().c_str(), msg.c_str(), ridx);
                        if (bmst) {
                           rmsg += bmst;
                           free(bmst);
                        }
                     }
                  }
               }
            }
         }
         fclose(f);
      }
   }

   // Send back to user
   response->Send((void *) rmsg.c_str(), rmsg.length()+1);

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::CleanupSessions(XrdProofdProtocol *p)
{
   // Handle request of
   XPDLOC(ALL, "Admin::CleanupSessions")

   int rc = 0;
   XPD_SETRESP(p, "CleanupSessions");

   XrdOucString cmsg;

   // Target client (default us)
   XrdProofdClient *tgtclnt = p->Client();

   // If super user we may be requested to cleanup everything
   bool all = 0;
   char *usr = 0;
   bool clntfound = 1;
   if (p->SuperUser()) {
      int what = ntohl(p->Request()->proof.int2);
      all = (what == 1) ? 1 : 0;

      if (!all) {
         // Get a user name, if any.
         // A super user can ask cleaning for clients different from itself
         char *buf = 0;
         int len = p->Request()->header.dlen;
         if (len > 0) {
            clntfound = 0;
            buf = p->Argp()->buff;
            len = (len < 9) ? len : 8;
         } else {
            buf = (char *) p->Client()->User();
            len = strlen(p->Client()->User());
         }
         if (len > 0) {
            usr = new char[len+1];
            memcpy(usr, buf, len);
            usr[len] = '\0';
            // Group info, if any
            char *grp = strstr(usr, ":");
            if (grp)
               *grp++ = 0;
            // Find the client instance
            XrdProofdClient *c = fMgr->ClientMgr()->GetClient(usr, grp);
            if (c) {
               tgtclnt = c;
               clntfound = 1;
            }
            TRACEP(p, REQ, "superuser, cleaning usr: "<< usr);
         }
      } else {
         tgtclnt = 0;
         TRACEP(p, REQ, "superuser, all sessions cleaned");
      }
   } else {
      // Define the user name for later transactions (their executed under
      // the admin name)
      int len = strlen(tgtclnt->User()) + 1;
      usr = new char[len+1];
      memcpy(usr, tgtclnt->User(), len);
      usr[len] = '\0';
   }

   // We cannot continue if we do not have anything to clean
   if (!clntfound) {
      TRACEP(p, DBG, "client '"<<usr<<"' has no sessions - do nothing");
   }

   // hard or soft (always hard for old clients)
   bool hard = (ntohl(p->Request()->proof.int3) == 1 || p->ProofProtocol() < 18) ? 1 : 0;
   const char *lab = hard ? "hard-reset" : "soft-reset";

   // Asynchronous notification to requester
   if (fMgr->SrvType() != kXPD_Worker) {
      XPDFORM(cmsg, "CleanupSessions: %s: signalling active sessions for termination", lab);
      response->Send(kXR_attn, kXPD_srvmsg, (char *) cmsg.c_str(), cmsg.length());
   }

   // Send a termination request to client sessions
   XPDFORM(cmsg, "CleanupSessions: %s: cleaning up client: requested by: %s", lab, p->Link()->ID);
   int srvtype = ntohl(p->Request()->proof.int2);
   fMgr->ClientMgr()->TerminateSessions(tgtclnt, cmsg.c_str(), srvtype);

   // Forward down the tree only if not leaf
   if (hard && fMgr->SrvType() != kXPD_Worker) {

      // Asynchronous notification to requester
      XPDFORM(cmsg, "CleanupSessions: %s: forwarding the reset request to next tier(s) ", lab);
      response->Send(kXR_attn, kXPD_srvmsg, 0, (char *) cmsg.c_str(), cmsg.length());

      int type = ntohl(p->Request()->proof.int1);
      fMgr->NetMgr()->Broadcast(type, usr, p->Client()->User(), response, 1);
   }

   // Wait just a bit before testing the activity of the session manager
   sleep(1);

   // Additional waiting (max 10 secs) depends on the activity of the session manager
   int twait = 10;
   while (twait-- > 0 &&
          fMgr->SessionMgr()->CheckCounter(XrdProofdProofServMgr::kCleanSessionsCnt) > 0) {
      if (twait < 7) {
         XPDFORM(cmsg, "CleanupSessions: %s: wait %d more seconds for completion ...", lab, twait);
         response->Send(kXR_attn, kXPD_srvmsg, 0, (char *) cmsg.c_str(), cmsg.length());
      }
      sleep(1);
   }

   // Cleanup usr
   SafeDelArray(usr);

   // Acknowledge user
   response->Send();

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::SetSessionAlias(XrdProofdProtocol *p)
{
   // Handle request for setting the session alias
   XPDLOC(ALL, "Admin::SetSessionAlias")

   int rc = 0;
   XPD_SETRESP(p, "SetSessionAlias");

   //
   // Specific info about a session
   int psid = ntohl(p->Request()->proof.sid);
   XrdProofdProofServ *xps = 0;
   if (!p->Client() || !(xps = p->Client()->GetServer(psid))) {
      TRACEP(p, XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"SetSessionAlias: session ID not found");
      return 0;
   }

   // Set session alias
   const char *msg = (const char *) p->Argp()->buff;
   int   len = p->Request()->header.dlen;
   if (len > kXPROOFSRVALIASMAX - 1)
      len = kXPROOFSRVALIASMAX - 1;

   // Save tag
   if (len > 0 && msg) {
      xps->SetAlias(msg);
      if (TRACING(DBG)) {
         XrdOucString alias(xps->Alias());
         TRACEP(p, DBG, "session alias set to: "<<alias);
      }
   }

   // Acknowledge user
   response->Send();

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::SetSessionTag(XrdProofdProtocol *p)
{
   // Handle request for setting the session tag
   XPDLOC(ALL, "Admin::SetSessionTag")

   int rc = 0;
   XPD_SETRESP(p, "SetSessionTag");
   //
   // Specific info about a session
   int psid = ntohl(p->Request()->proof.sid);
   XrdProofdProofServ *xps = 0;
   if (!p->Client() || !(xps = p->Client()->GetServer(psid))) {
      TRACEP(p, XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"SetSessionTag: session ID not found");
      return 0;
   }

   // Set session tag
   const char *msg = (const char *) p->Argp()->buff;
   int   len = p->Request()->header.dlen;
   if (len > kXPROOFSRVTAGMAX - 1)
      len = kXPROOFSRVTAGMAX - 1;

   // Save tag
   if (len > 0 && msg) {
      xps->SetTag(msg);
      if (TRACING(DBG)) {
         XrdOucString tag(xps->Tag());
         TRACEP(p, DBG, "session tag set to: "<<tag);
      }
   }

   // Acknowledge user
   response->Send();

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::ReleaseWorker(XrdProofdProtocol *p)
{
   // Handle request for releasing a worker
   XPDLOC(ALL, "Admin::ReleaseWorker")

   int rc = 0;
   XPD_SETRESP(p, "ReleaseWorker");
   //
   // Specific info about a session
   int psid = ntohl(p->Request()->proof.sid);
   XrdProofdProofServ *xps = 0;
   if (!p->Client() || !(xps = p->Client()->GetServer(psid))) {
      TRACEP(p, XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"ReleaseWorker: session ID not found");
      return 0;
   }

   // Set session tag
   const char *msg = (const char *) p->Argp()->buff;
   int   len = p->Request()->header.dlen;
   if (len > kXPROOFSRVTAGMAX - 1)
      len = kXPROOFSRVTAGMAX - 1;

   // Save tag
   if (len > 0 && msg) {
      xps->RemoveWorker(msg);
      TRACEP(p, DBG, "worker \""<<msg<<"\" released");
      if (TRACING(HDBG)) fMgr->NetMgr()->Dump();
   }

   // Acknowledge user
   response->Send();

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::CheckForbiddenChars(const char *s)
{
   // Check is 's' contains any of the forbidden chars '(){};'
   // Return 0 if OK (no forbidden chars), -1 in not OK

   int len = 0;
   if (!s || (len = strlen(s)) <= 0) return 0;

   int j = len;
   while (j--) {
      char c = s[j];
      if (c == '(' || c == ')' || c == '{' || c == '}' || c == ';') {
         return -1;
      }
   }
   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::Exec(XrdProofdProtocol *p)
{
   // Handle request of cleaning parts of the sandbox

   XPDLOC(ALL, "Admin::Exec")

   // Commands; must be synchronized with EAdminExecType in XProofProtocol.h
#if !defined(__APPLE__)
   const char *cmds[] = { "rm", "ls", "more", "grep", "tail", "md5sum", "stat", "find" };
#else
   const char *cmds[] = { "rm", "ls", "more", "grep", "tail", "md5", "stat", "find" };
#endif
   const char *actcmds[] = { "remove", "access", "open", "open", "open", "open", "stat", "find"};

   int rc = 0;
   XPD_SETRESP(p, "Exec");

   XrdOucString emsg;

   // Target client (default us)
   XrdProofdClient *tgtclnt = p->Client();
   if (!tgtclnt) {
      emsg = "client instance not found";
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
      return 0;
   }

   // Action type
   int action = ntohl(p->Request()->proof.int2);
   if (action < kRm || action > kFind) {
      emsg = "unknown action type: ";
      emsg += action;
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
      return 0;
   }

   // Parse the string
   int dlen = p->Request()->header.dlen;
   XrdOucString msg, node, path, opt;
   if (dlen > 0 && p->Argp()->buff) {
      msg.assign((const char *)p->Argp()->buff, 0, dlen);
      // Parse
      emsg = "";
      int from = 0;
      if ((from = msg.tokenize(node, from, '|')) != -1) {
         if ((from = msg.tokenize(path, from, '|')) != -1) {
            from = msg.tokenize(opt, from, '|');
         } else {
            emsg = "'path' not found in message";
         }
      } else {
         emsg = "'node' not found in message";
      }
      if (emsg.length() > 0) {
         TRACEP(p, XERR, emsg);
         response->Send(kXR_InvalidRequest, emsg.c_str());
         return 0;
      }
   }

   // Path and opt cannot contain multiple commands (e.g. file; rm *)
   if (CheckForbiddenChars(path.c_str()) != 0) {
      emsg = "none of the characters '(){};' are allowed in path string ("; emsg += path; emsg += ")";
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
      return 0;
   }
   if (CheckForbiddenChars(opt.c_str()) != 0) {
      emsg = "none of the characters '(){};' are allowed in opt string ("; emsg += opt; emsg += ")";
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
      return 0;
   }

   // Check if we have to forward this request
   XrdOucString result;
   bool islocal = fMgr->NetMgr()->IsLocal(node.c_str(), 1);
   if (fMgr->SrvType() != kXPD_Worker) {
      int type = ntohl(p->Request()->proof.int1);
      if (node == "all") {
         if (action == kStat || action == kMd5sum) {
            emsg = "action cannot be run in mode 'all' - running on master only";
            response->Send(kXR_attn, kXPD_srvmsg, 2, (char *)emsg.c_str(), emsg.length());
         } else {
            fMgr->NetMgr()->Broadcast(type, msg.c_str(), p->Client()->User(), response, 0, action);
         }
      } else if (!islocal) {
         // Create 'url'
         XrdOucString u = (p->Client()->User()) ? p->Client()->User() : fMgr->EffectiveUser();
         u += '@';
         u += node;
         TRACEP(p, HDBG, "sending request to "<<u);
         // Send request
         XrdClientMessage *xrsp;
         if (!(xrsp = fMgr->NetMgr()->Send(u.c_str(), type, msg.c_str(), 0, response, 0, action))) {
            TRACEP(p, XERR, "problems sending request to "<<u);
         } else {
            if (action == kStat || action == kMd5sum) {
               // Extract the result
               result.assign((const char *) xrsp->GetData(), 0, xrsp->DataLen());
            } else if (action == kRm) {
               // Send 'OK'
               result = "OK";
            }
         }
         // Cleanup answer
         SafeDel(xrsp);
      }
   }

   // We may not have been requested to execute the command
   if (node != "all" && !islocal) {
      // We are done: acknowledge user ...
      if (result.length() > 0) {
         response->Send(result.c_str());
      } else {
         response->Send();
      }
      // ... and go
      return 0;
   }

   // Here we execute the request
   XrdOucString cmd, pfx(fMgr->Host());
   pfx += ":"; pfx += fMgr->Port();

   // Notify the client
   if (node != "all") {
      if (action != kStat && action != kMd5sum && action != kRm) {
         emsg = "Node: "; emsg += pfx;
         emsg += "\n-----";
         response->Send(kXR_attn, kXPD_srvmsg, 2, (char *)emsg.c_str(), emsg.length());
      }
      pfx = "";
   } else {
      pfx += "| ";
   }

   // Get the full path, check if in sandbox and if the user is allowed
   // to access it
   XrdOucString fullpath(path);
   bool sandbox = 0;
   bool haswild = (fullpath.find('*') != STR_NPOS) ? 1 : 0;
   int check = (action == kMore || action == kTail ||
                action == kGrep || action == kMd5sum) ? 2 : 1;
   if ((action == kRm || action == kLs) && haswild) check = 0;
   int rccp = 0;
   struct stat st;
   if ((rccp = CheckPath(p->SuperUser(), tgtclnt->Sandbox()->Dir(),
                         fullpath, check, sandbox, &st, emsg)) != 0) {
      if (rccp == -2) {
         emsg = cmds[action];
         emsg += ": cannot ";
         emsg += actcmds[action];
         emsg += " `";
         emsg += fullpath;
         emsg += "': No such file or directory";
      } else if (rccp == -3) {
         emsg = cmds[action];
         emsg += ": cannot stat ";
         emsg += fullpath;
         emsg += ": errno: ";
         emsg += (int) errno;
      } else if (rccp == -4) {
         emsg = cmds[action];
         emsg += ": ";
         emsg += fullpath;
         emsg += ": Is not a regular file";
      }
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
      return 0;
   }

   // Additional checks for remove requests
   if (action == kRm) {
      // Ownership required and no support for wild cards for absolute paths
      if (!sandbox) {
         if (haswild) {
            emsg = "not allowed to rm with wild cards on path: ";
            emsg += fullpath;
            TRACEP(p, XERR, emsg);
            response->Send(kXR_InvalidRequest, emsg.c_str());
            return 0;
         }
         if ((int) st.st_uid != tgtclnt->UI().fUid || (int) st.st_gid != tgtclnt->UI().fGid) {
            emsg = "rm on path: ";
            emsg += fullpath;
            emsg += " requires ownership; path owned by: (";
            emsg += (int) st.st_uid; emsg += ",";
            emsg += (int) st.st_gid; emsg += ")";
            TRACEP(p, XERR, emsg);
            response->Send(kXR_InvalidRequest, emsg.c_str());
            return 0;
         }
      } else {
         // Will not allow to remove basic sandbox sub-dirs
         const char *sbdir[5] = {"queries", "packages", "cache", "datasets", "data"};
         while (fullpath.endswith('/'))
            fullpath.erasefromend(1);
         XrdOucString sball(tgtclnt->Sandbox()->Dir()), sball1 = sball;
         sball += "/*"; sball1 += "/*/";
         if (fullpath == sball || fullpath == sball1) {
            emsg = "removing all sandbox directory is not allowed: ";
            emsg += fullpath;
            TRACEP(p, XERR, emsg);
            response->Send(kXR_InvalidRequest, emsg.c_str());
            return 0;
         }
         int kk = 5;
         while (kk--) {
            if (fullpath.endswith(sbdir[kk])) {
               emsg = "removing a basic sandbox directory is not allowed: ";
               emsg += fullpath;
               TRACEP(p, XERR, emsg);
               response->Send(kXR_InvalidRequest, emsg.c_str());
               return 0;
            }
         }
      }

      // Prepare the command
      cmd = cmds[action];
      if (opt.length() <= 0) opt = "-f";
      cmd += " "; cmd += opt;
      cmd += " "; cmd += fullpath;
      cmd += " 2>&1";

   } else {

      XrdOucString rederr;
      cmd = cmds[action];
      switch (action) {
         case kLs:
            if (opt.length() <= 0) opt = "-C";
            rederr = " 2>&1";
            break;
         case kMore:
         case kGrep:
         case kTail:
         case kFind:
            rederr = " 2>&1";
            break;
         case kStat:
            cmd = "";
            opt = "";
            break;
         case kMd5sum:
            opt = "";
            rederr = " 2>&1";
            break;
         default:
            emsg = "undefined action: ";
            emsg = action;
            emsg = " - protocol error!";
            TRACEP(p, XERR, emsg);
            response->Send(kXR_ServerError, emsg.c_str());
            break;
      }
      if (action != kFind) {
         if (cmd.length() > 0) cmd += " ";
         if (opt.length() > 0) { cmd += opt; cmd += " ";}
         cmd += fullpath;
      } else {
         cmd += " "; cmd += fullpath;
         if (opt.length() > 0) { cmd += " "; cmd += opt; }
      }
      if (rederr.length() > 0) cmd += rederr;
   }

   // Run the command now
   emsg = pfx;
   if (ExecCmd(p, response, action, cmd.c_str(), emsg) != 0) {
      TRACEP(p, XERR, emsg);
      response->Send(kXR_ServerError, emsg.c_str());
   } else {
      // Done
      switch (action) {
         case kStat:
         case kMd5sum:
            response->Send(emsg.c_str());
            break;
         case kRm:
            response->Send("OK");
            break;
         default:
            response->Send();
            break;
      }
   }

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::ExecCmd(XrdProofdProtocol *p, XrdProofdResponse *r,
                         int action, const char *cmd, XrdOucString &emsg)
{
   // Low-level execution handler. The commands must be executed in user space.
   // We do that by forking and logging as user in the forked instance. The
   // parent will just send over te messages received from the user-child via
   // the pipe.
   // Return 0 on success, -1 on error

   XPDLOC(ALL, "Admin::ExecCmd")

   int rc = 0;
   XrdOucString pfx = emsg;
   emsg = "";

   // We do it via the shell
   if (!cmd || strlen(cmd) <= 0) {
      emsg = "undefined command!";
      return -1;
   }

   // Pipe for child-to-parent communications
   XrdProofdPipe pp;
   if (!pp.IsValid()) {
      emsg = "cannot create the pipe";
      return -1;
   }

   // Fork a test agent process to handle this session
   TRACEP(p, DBG, "forking to execute in the private sandbox");
   int pid = -1;
   if (!(pid = fMgr->Sched()->Fork("adminexeccmd"))) {
      // Child process
      // We set to the user environment as we must to run the command
      // in the user space
      if (fMgr->SessionMgr()->SetUserEnvironment(p) != 0) {
         emsg = "SetUserEnvironment did not return OK";
         rc = 1;
      } else {
         // Execute the command
         if (action == kStat) {
            struct stat st;
            if ((stat(cmd, &st)) != 0) {
               if (errno == ENOENT) {
                  emsg += "stat: cannot stat `";
                  emsg += cmd;
                  emsg += "': No such file or directory";
               } else {
                  emsg += "stat: cannot stat ";
                  emsg += cmd;
                  emsg += ": errno: ";
                  emsg += (int) errno;
               }
            } else {
               // Fill the buffer and go
               char msg[256];
               int  islink = S_ISLNK(st.st_mode);
               snprintf(msg, 256, "%ld %ld %d %d %d %lld %ld %d", (long)st.st_dev,
                        (long)st.st_ino, st.st_mode, (int)(st.st_uid),
                        (int)(st.st_gid), (kXR_int64)st.st_size, st.st_mtime, islink);
               emsg = msg;
            }
         } else {
            // Execute the command in a pipe
            FILE *fp = popen(cmd, "r");
            if (!fp) {
               emsg = "could not run '"; emsg += cmd; emsg += "'";
               rc = 1;
            } else {
               // Read line by line
               int pfxlen = pfx.length();
               int len = 0;
               char line[2048];
               char buf[1024];
               int bufsiz = 1024, left = bufsiz - 1, lines = 0;
               while (fgets(line, sizeof(line), fp)) {
                  // Parse the line
                  int llen = strlen(line);
                  lines++;
                  // If md5sum, we need to parse only the first line
                  if (lines == 1 && action == kMd5sum) {
                     if (line[llen-1] == '\n') {
                        line[llen-1] = '\0';
                        llen--;
                     }
#if !defined(__APPLE__)
                     // The first token
                     XrdOucString sl(line);
                     sl.tokenize(emsg, 0, ' ');
#else
                     // The last token
                     XrdOucString sl(line), tkn;
                     int from = 0;
                     while ((from = sl.tokenize(tkn, from, ' ')) != STR_NPOS) {
                        emsg = tkn;
                     }
#endif
                     break;
                  }
                  // Send over this part, if no more space
                  if ((llen + pfxlen) > left) {
                     buf[len] = '\0';
                     if (buf[len-1] == '\n') buf[len-1] = '\0';
                     if (r->Send(kXR_attn, kXPD_srvmsg, 2, (char *) &buf[0], len) != 0) {
                        emsg = "error sending message to requester";
                        rc = 1;
                        break;
                     }
                     buf[0] = 0;
                     len = 0;
                     left = bufsiz -1;
                  }
                  // Add prefix to the buffer, if any
                  if (pfxlen > 0) {
                     memcpy(buf+len, pfx.c_str(), pfxlen);
                     len += pfxlen;
                     left -= pfxlen;
                  }
                  // Add line to the buffer
                  memcpy(buf+len, line, llen);
                  len += llen;
                  left -= llen;
                  // Check if we have been interrupted
                  if (lines > 0 && !(lines % 10)) {
                     char b[1];
                     if (p->Link()->Peek(&b[0], 1, 0) == 1) {
                        p->Process(p->Link());
                        if (p->IsCtrlC()) break;
                     }
                  }
               }
               // Send the last bunch
               if (len > 0) {
                  buf[len] = '\0';
                  if (buf[len-1] == '\n') buf[len-1] = '\0';
                  if (r->Send(kXR_attn, kXPD_srvmsg, 2, (char *) &buf[0], len) != 0) {
                     emsg = "error sending message to requester";
                     rc = 1;
                  }
               }
               // Close the pipe
               int rcpc = 0;
               if ((rcpc = pclose(fp)) == -1) {
                  emsg = "could not close the command pipe";
                  rc = 1;
               }
               if (WEXITSTATUS(rcpc) != 0) {
                  emsg = "failure: return code: ";
                  emsg += (int) WEXITSTATUS(rcpc);
                  rc = 1;
               }
            }
         }
      }
      // Send error, if any
      if (rc == 1) {
         // Post Error
         if (pp.Post(-1, emsg.c_str()) != 0) rc = 1;
      }

      // End-Of-Transmission
      if (pp.Post(0, emsg.c_str()) != 0) rc = 1;

      // Done
      exit(rc);
   }

   // Parent process
   if (pid < 0) {
      emsg = "forking failed - errno: "; emsg += (int) errno;
      return -1;
   }

   // now we wait for the callback to be (successfully) established
   TRACEP(p, DBG, "forking OK: wait for information");

   // Read status-of-setup from pipe
   int prc = 0, rst = -1;
   // We wait for 60 secs max among transfers
   while (rst < 0 && rc >= 0) {
      while ((prc = pp.Poll(60)) > 0) {
         XpdMsg msg;
         if (pp.Recv(msg) != 0) {
            emsg = "error receiving message from pipe";
            return -1;
         }
         // Status is the message type
         rst = msg.Type();
         // Read string, if any
         XrdOucString buf;
         if (rst < 0) {
            buf = msg.Buf();
            if (buf.length() <= 0) {
               emsg = "error reading string from received message";
               return -1;
            }
            // Store error message
            emsg = buf;
         } else {
            if (action == kMd5sum || action == kStat) {
               buf = msg.Buf();
               if (buf.length() <= 0) {
                  emsg = "error reading string from received message";
                 return -1;
               }
               // Store md5sum
               emsg = buf;
            }
            // Done
            break;
         }
      }
      if (prc == 0) {
         emsg = "timeout from poll";
         return -1;
      } else if (prc < 0) {
         emsg = "error from poll - errno: "; emsg += -prc;
         return -1;
      }
   }

   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdAdmin::CheckPath(bool superuser, const char *sbdir,
                              XrdOucString &fullpath, int check, bool &sandbox,
                              struct stat *st, XrdOucString &emsg)
{
   // Handle request for sending a file

   if (!sbdir || strlen(sbdir) <= 0) {
      emsg = "CheckPath: sandbox dir undefined!";
      return -1;
   }

   // Get the full path and check if in sandbox
   XrdOucString path(fullpath);
   sandbox = 0;
   if (path.beginswith('/')) {
      fullpath = path;
      if (fullpath.beginswith(sbdir)) sandbox = 1;
   } else {
      if (path.beginswith("../")) path.erase(0,2);
      if (path.beginswith("./") || path.beginswith("~/")) path.erase(0,1);
      if (!path.beginswith("/")) path.insert('/',0);
      fullpath = sbdir;
      fullpath += path;
      sandbox = 1;
   }
   fullpath.replace("//","/");

   // If the path is absolute, we must check a normal user is allowed to browse
   if (!sandbox && !superuser) {
      bool notfound = 1;
      std::list<XrdOucString>::iterator si = fExportPaths.begin();
      while (si != fExportPaths.end()) {
         if (path.beginswith((*si).c_str())) {
            notfound = 0;
            break;
         }
         si++;
      }
      if (notfound) {
         emsg = "CheckPath: not allowed to run the requested action on ";
         emsg += path;
         return -1;
      }
   }

   if (check > 0 && st) {
      // Check if the file exists
      if (stat(fullpath.c_str(), st) != 0) {
         if (errno == ENOENT) {
            return -2;
         } else {
            return -3;
         }
      }

      // Certain actions require a regular file
      if ((check == 2) && !S_ISREG(st->st_mode)) return -4;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::GetFile(XrdProofdProtocol *p)
{
   // Handle request for sending a file

   XPDLOC(ALL, "Admin::GetFile")

   int rc = 0;
   XPD_SETRESP(p, "GetFile");

   XrdOucString emsg;

   // Target client (default us)
   XrdProofdClient *tgtclnt = p->Client();
   if (!tgtclnt) {
      emsg = "client instance not found";
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
      return 0;
   }

   // Parse the string
   int dlen = p->Request()->header.dlen;
   XrdOucString path;
   if (dlen > 0 && p->Argp()->buff) {
      path.assign((const char *)p->Argp()->buff, 0, dlen);
      if (path.length() <= 0) {
         TRACEP(p, XERR, "path missing!");
         response->Send(kXR_InvalidRequest, "path missing!");
         return 0;
      }
   }

   // Get the full path, check if in sandbox and if the user is allowed
   // to access it
   XrdOucString fullpath(path);
   bool sandbox = 0, check = 2;
   int rccp = 0;
   struct stat st;
   if ((rccp = CheckPath(p->SuperUser(), tgtclnt->Sandbox()->Dir(),
                         fullpath, check, sandbox, &st, emsg)) != 0) {
      if (rccp == -2) {
         emsg = "Cannot open `";
         emsg += fullpath;
         emsg += "': No such file or directory";
      } else if (rccp == -3) {
         emsg = "Cannot stat `";
         emsg += fullpath;
         emsg += "': errno: ";
         emsg += (int) errno;
      } else if (rccp == -4) {
         emsg = fullpath;
         emsg += " is not a regular file";
      }
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
      return 0;
   }

   // Pipe for child-to-parent communications
   XrdProofdPipe pp;
   if (!pp.IsValid()) {
      emsg = "cannot create the pipe for internal communications";
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
   }

   // Fork a test agent process to handle this session
   TRACEP(p, DBG, "forking to execute in the private sandbox");
   int pid = -1;
   if (!(pid = fMgr->Sched()->Fork("admingetfile"))) {

      // Child process
      // We set to the user environment as we must to run the command
      // in the user space
      if (fMgr->SessionMgr()->SetUserEnvironment(p) != 0) {
         emsg = "SetUserEnvironment did not return OK";
         rc = 1;
      } else {

         // Open the file
         int fd = open(fullpath.c_str(), O_RDONLY);
         if (fd < 0) {
            emsg = "cannot open file: ";
            emsg += fullpath;
            emsg += " - errno:";
            emsg += (int) errno;
            TRACEP(p, XERR, emsg);
            response->Send(kXR_ServerError, emsg.c_str());
            rc = 1;

         } else {
            // Send the size as OK message
            char sizmsg[64];
            snprintf(sizmsg, 64, "%lld", (kXR_int64) st.st_size);
            response->Send((const char *) &sizmsg[0]);
            TRACEP(p, XERR, "size is "<<sizmsg<<" bytes");

            // Now we send the content
            const int kMAXBUF = 16384;
            char buf[kMAXBUF];
            off_t pos = 0;
            lseek(fd, pos, SEEK_SET);

            while (rc == 0 && pos < st.st_size) {
               off_t left = st.st_size - pos;
               if (left > kMAXBUF) left = kMAXBUF;

               int siz;
               while ((siz = read(fd, &buf[0], left)) < 0 && errno == EINTR)
                  errno = 0;
               if (siz < 0 || siz != left) {
                  emsg = "error reading from file: errno: ";
                  emsg += (int) errno;
                  rc = 1;
                  break;
               }

               int src = 0;
               if ((src = response->Send(kXR_attn, kXPD_msg, (void *)&buf[0], left)) != 0) {
                  emsg = "error reading from file: errno: ";
                  emsg += src;
                  rc = 1;
                  break;
               }
               // Re-position
               pos += left;
               // Reset the timeout
               if (pp.Post(0, "") != 0) {
                  rc = 1;
                  break;
               }
            }
            // Close the file
            close(fd);
            // Send error, if any
            if (rc != 0) {
               TRACEP(p, XERR, emsg);
               response->Send(kXR_attn, kXPD_srvmsg, 0, (char *) emsg.c_str(), emsg.length());
            }
         }
      }

      // Send error, if any
      if (rc == 1) {
         // Post Error
         if (pp.Post(-1, emsg.c_str()) != 0) rc = 1;
      } else {
         // End-Of-Transmission
         if (pp.Post(1, "") != 0) rc = 1;
      }

      // Done
      exit(rc);
   }

   // Parent process
   if (pid < 0) {
      emsg = "forking failed - errno: "; emsg += (int) errno;
      TRACEP(p, XERR, emsg);
      response->Send(kXR_ServerError, emsg.c_str());
      return 0;
   }

   // The parent is done: wait for the child
   TRACEP(p, DBG, "forking OK: execution will continue in the child process");

   // Wait for end-of-operations from pipe
   int prc = 0, rst = 0;
   // We wait for 60 secs max among transfers
   while (rst == 0 && rc >= 0) {
      while ((prc = pp.Poll(60)) > 0) {
         XpdMsg msg;
         if (pp.Recv(msg) != 0) {
            emsg = "error receiving message from pipe";
            return -1;
         }
         // Status is the message type
         rst = msg.Type();
         // Read string, if any
         if (rst < 0) {
            // Error
            rc = -1;
            // Store error message
            emsg = msg.Buf();
            if (emsg.length() <= 0) {
               emsg = "error reading string from received message";
            }
            // We stop here
            break;
         } else if (rst > 0) {
            // We are done
            break;
         }
      }
      if (prc == 0) {
         emsg = "timeout from poll";
         rc = -1;
      } else if (prc < 0) {
         emsg = "error from poll - errno: "; emsg += -prc;
         rc = -1;
      }
   }

   // The parent is done
   TRACEP(p, DBG, "execution over: "<< ((rc == 0) ? "ok" : "failed"));

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::PutFile(XrdProofdProtocol *p)
{
   // Handle request for recieving a file

   XPDLOC(ALL, "Admin::PutFile")

   int rc = 0;
   XPD_SETRESP(p, "PutFile");

   XrdOucString emsg;

   // Target client (default us)
   XrdProofdClient *tgtclnt = p->Client();
   if (!tgtclnt) {
      emsg = "client instance not found";
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
      return 0;
   }

   // Parse the string
   kXR_int64 size = -1;
   int dlen = p->Request()->header.dlen;
   XrdOucString cmd, path, ssiz, opt;
   if (dlen > 0 && p->Argp()->buff) {
      cmd.assign((const char *)p->Argp()->buff, 0, dlen);
      if (cmd.length() <= 0) {
         TRACEP(p, XERR, "input buffer missing!");
         response->Send(kXR_InvalidRequest, "input buffer missing!");
         return 0;
      }
      int from = 0;
      if ((from = cmd.tokenize(path, from, ' ')) < 0) {
         TRACEP(p, XERR, "cannot resolve path!");
         response->Send(kXR_InvalidRequest, "cannot resolve path!");
         return 0;
      }
      if ((from = cmd.tokenize(ssiz, from, ' ')) < 0) {
         TRACEP(p, XERR, "cannot resolve word with size!");
         response->Send(kXR_InvalidRequest, "cannot resolve word with size!");
         return 0;
      }
      // Extract size
      size = atoll(ssiz.c_str());
      if (size < 0) {
         TRACEP(p, XERR, "cannot resolve size!");
         response->Send(kXR_InvalidRequest, "cannot resolve size!");
         return 0;
      }
      // Any option?
      cmd.tokenize(opt, from, ' ');
   }
   TRACEP(p, DBG, "path: '"<<path<<"'; size: "<<size<<" bytes; opt: '"<<opt<<"'");

   // Default open and mode flags
   kXR_unt32 openflags = O_WRONLY | O_TRUNC | O_CREAT;
   kXR_unt32 modeflags = 0600;

   // Get the full path and check if in sandbox and if the user is allowed
   // to create/access it
   XrdOucString fullpath(path);
   bool sandbox = 0, check = 1;
   struct stat st;
   int rccp = 0;
   if ((rccp = CheckPath(p->SuperUser(), tgtclnt->Sandbox()->Dir(),
                         fullpath, check, sandbox, &st, emsg)) != 0) {
      if (rccp == -3) {
         emsg = "File `";
         emsg += fullpath;
         emsg += "' exists but cannot be stat: errno: ";
         emsg += (int) errno;
      }
      if (rccp != -2) {
         TRACEP(p, XERR, emsg);
         response->Send(kXR_InvalidRequest, emsg.c_str());
         return 0;
      }
   } else {
      // File exists: either force deletion or fail
      if (opt == "force") {
         openflags = O_WRONLY | O_TRUNC;
      } else {
         emsg = "file'";
         emsg += fullpath;
         emsg += "' exists; user option 'force' to override it";
         TRACEP(p, XERR, emsg);
         response->Send(kXR_InvalidRequest, emsg.c_str());
         return 0;
      }
   }

   // Pipe for child-to-parent communications
   XrdProofdPipe pp;
   if (!pp.IsValid()) {
      emsg = "cannot create the pipe for internal communications";
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
   }

   // Fork a test agent process to handle this session
   TRACEP(p, DBG, "forking to execute in the private sandbox");
   int pid = -1;
   if (!(pid = fMgr->Sched()->Fork("adminputfile"))) {
      // Child process
      // We set to the user environment as we must to run the command
      // in the user space
      if (fMgr->SessionMgr()->SetUserEnvironment(p) != 0) {
         emsg = "SetUserEnvironment did not return OK";
         rc = 1;
      } else {
         // Open the file
         int fd = open(fullpath.c_str(), openflags, modeflags);
         if (fd < 0) {
            emsg = "cannot open file: ";
            emsg += fullpath;
            emsg += " - errno: ";
            emsg += (int) errno;
            TRACEP(p, XERR, emsg);
            response->Send(kXR_ServerError, emsg.c_str());
            rc = 1;
         } else {
            // We read in the content sent by the client
            rc = 0;
            response->Send("OK");
            // Receive the file
            const int kMAXBUF = XrdProofdProtocol::MaxBuffsz();
            // Get a buffer
            XrdBuffer *argp = XrdProofdProtocol::GetBuff(kMAXBUF);
            if (!argp) {
               emsg = "cannot get buffer to read data out";
               rc = 1;
            }
            int r;
            kXR_int64 filesize = 0, left = 0;
            while (rc == 0 && filesize < size) {
               left = size - filesize;
               if (left > kMAXBUF) left = kMAXBUF;
               // Read a bunch of data
               TRACEP(p, ALL, "receiving "<<left<<" ...");
               if ((rc = p->GetData("data", argp->buff, left))) {
                  XrdProofdProtocol::ReleaseBuff(argp);
                  emsg = "cannot read data out";
                  rc = 1;
                  break;
               }
               // Update counters
               filesize += left;
               // Write to local file
               char *b = argp->buff;
               r = left;
               while (r) {
                  int w = 0;
                  while ((w = write(fd, b, r)) < 0 && errno == EINTR)
                     errno = 0;
                  if (w < 0) {
                     emsg = "error writing to unit: ";
                     emsg += fd;
                     rc = 1;
                     break;
                  }
                  r -= w;
                  b += w;
               }
               // Reset the timeout
               if (pp.Post(0, "") != 0) {
                  rc = 1;
                  break;
               }
            }
            // Close the file
            close(fd);
            // Release the buffer
            XrdProofdProtocol::ReleaseBuff(argp);
            // Send error, if any
            if (rc != 0) {
               TRACEP(p, XERR, emsg);
               response->Send(kXR_attn, kXPD_srvmsg, 0, (char *) emsg.c_str(), emsg.length());
            }
         }
      }
      // Send error, if any
      if (rc == 1) {
         // Post Error
         if (pp.Post(-1, emsg.c_str()) != 0) rc = 1;
      } else {
         // End-Of-Transmission
         if (pp.Post(1, "") != 0) rc = 1;
      }
      // Done
      exit(rc);
   }

   // Parent process
   if (pid < 0) {
      emsg = "forking failed - errno: "; emsg += (int) errno;
      TRACEP(p, XERR, emsg);
      response->Send(kXR_ServerError, emsg.c_str());
      return 0;
   }

   // The parent is done: wait for the child
   TRACEP(p, DBG, "forking OK: execution will continue in the child process");

   // Wait for end-of-operations from pipe
   int prc = 0, rst = 0;
   // We wait for 60 secs max among transfers
   while (rst == 0 && rc >= 0) {
      while ((prc = pp.Poll(60)) > 0) {
         XpdMsg msg;
         if (pp.Recv(msg) != 0) {
            emsg = "error receiving message from pipe";
            return -1;
         }
         // Status is the message type
         rst = msg.Type();
         // Read string, if any
         if (rst < 0) {
            // Error
            rc = -1;
            // Store error message
            emsg = msg.Buf();
            if (emsg.length() <= 0) {
               emsg = "error reading string from received message";
            }
            // We stop here
            break;
         } else if (rst > 0) {
            // We are done
            break;
         }
      }
      if (prc == 0) {
         emsg = "timeout from poll";
         rc = -1;
      } else if (prc < 0) {
         emsg = "error from poll - errno: "; emsg += -prc;
         rc = -1;
      }
   }

   // The parent is done
   TRACEP(p, DBG, "execution over: "<< ((rc == 0) ? "ok" : "failed"));

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdAdmin::CpFile(XrdProofdProtocol *p)
{
   // Handle request for copy files from / to the sandbox

   XPDLOC(ALL, "Admin::CpFile")

   int rc = 0;
   XPD_SETRESP(p, "CpFile");

   XrdOucString emsg;

   // Target client (default us)
   XrdProofdClient *tgtclnt = p->Client();
   if (!tgtclnt) {
      emsg = "client instance not found";
      TRACEP(p, XERR, emsg);
      response->Send(kXR_InvalidRequest, emsg.c_str());
      return 0;
   }

   // Parse the string
   int dlen = p->Request()->header.dlen;
   XrdOucString buf, src, dst, fmt;
   if (dlen > 0 && p->Argp()->buff) {
      buf.assign((const char *)p->Argp()->buff, 0, dlen);
      if (buf.length() <= 0) {
         TRACEP(p, XERR, "input buffer missing!");
         response->Send(kXR_InvalidRequest, "input buffer missing!");
         return 0;
      }
      int from = 0;
      if ((from = buf.tokenize(src, from, ' ')) < 0) {
         TRACEP(p, XERR, "cannot resolve src path!");
         response->Send(kXR_InvalidRequest, "cannot resolve src path!");
         return 0;
      }
      if ((from = buf.tokenize(dst, from, ' ')) < 0) {
         TRACEP(p, XERR, "cannot resolve dst path!");
         response->Send(kXR_InvalidRequest, "cannot resolve dst path!");
         return 0;
      }
      // The rest, if any, is the format string (including options)
      fmt.assign(buf, from);
   }
   TRACEP(p, DBG, "src: '"<<src<<"'; dst: '"<<dst<<"'; fmt: '"<<fmt<<"'");

   // Check paths
   bool locsrc = 1;
   XrdClientUrlInfo usrc(src.c_str());
   if (usrc.Proto.length() > 0 && usrc.Proto != "file") {
      locsrc = 0;
      if (!fAllowedCpCmds.Find(usrc.Proto.c_str())) {
         TRACEP(p, XERR, "protocol for source file not supported");
         response->Send(kXR_InvalidRequest, "protocol for source file not supported");
         return 0;
      }
   }
   if (usrc.Proto == "file") src = usrc.File;
   bool locdst = 1;
   XrdClientUrlInfo udst(dst.c_str());
   if (udst.Proto.length() > 0 && udst.Proto != "file") {
      locdst = 0;
      if (!fAllowedCpCmds.Find(udst.Proto.c_str())) {
         TRACEP(p, XERR, "protocol for destination file not supported");
         response->Send(kXR_InvalidRequest, "protocol for destination file not supported");
         return 0;
      }
   }
   if (udst.Proto == "file") dst = udst.File;

   // Locate the remote protocol, if any
   bool loc2loc = 1;
   bool loc2rem = 0;
   bool rem2loc = 0;
   XpdAdminCpCmd *xc = 0;
   if (!locsrc && !locdst) {
      // Files cannot be both remote
      TRACEP(p, XERR, "At least destination or source must be local");
      response->Send(kXR_InvalidRequest, "At least destination or source must be local");
      return 0;
   } else if (!locdst) {
      // Find the requested protocol and check if we can put
      xc = fAllowedCpCmds.Find(udst.Proto.c_str());
      if (!xc->fCanPut) {
         TRACEP(p, XERR, "not allowed to create destination file with the chosen protocol");
         response->Send(kXR_InvalidRequest, "not allowed to create destination file with the chosen protocol");
         return 0;
      }
      loc2loc = 0;
      loc2rem = 1;
   } else if (!locsrc) {
      // Find the requested protocol
      xc = fAllowedCpCmds.Find(usrc.Proto.c_str());
      loc2loc = 0;
      rem2loc = 1;
   } else {
      // Default local protocol
      xc = fAllowedCpCmds.Find("file");
   }

   // Check the local paths
   XrdOucString srcpath(src), dstpath(dst);
   bool sbsrc = 0, sbdst = 0;
   struct stat stsrc, stdst;
   int rccpsrc = 0, rccpdst = 0;
   if (loc2loc || loc2rem) {
      if ((rccpsrc = CheckPath(p->SuperUser(), tgtclnt->Sandbox()->Dir(),
                               srcpath, 2, sbsrc, &stsrc, emsg)) != 0) {
         if (rccpsrc == -2) {
            emsg = xc->fCmd;
            emsg += ": cannot open `";
            emsg += srcpath;
            emsg += "': No such file or directory";
         } else if (rccpsrc == -3) {
            emsg = xc->fCmd;
            emsg += ": cannot stat ";
            emsg += srcpath;
            emsg += ": errno: ";
            emsg += (int) errno;
         } else if (rccpsrc == -4) {
            emsg = xc->fCmd;
            emsg += ": ";
            emsg += srcpath;
            emsg += ": Is not a regular file";
         }
         TRACEP(p, XERR, emsg);
         response->Send(kXR_InvalidRequest, emsg.c_str());
         return 0;
      }
   }
   if (loc2loc || rem2loc) {
      if ((rccpdst = CheckPath(p->SuperUser(), tgtclnt->Sandbox()->Dir(),
                               dstpath, 0, sbdst, &stdst, emsg)) != 0) {
         if (rccpdst == -2) {
            emsg = xc->fCmd;
            emsg += ": cannot open `";
            emsg += dstpath;
            emsg += "': No such file or directory";
         } else if (rccpdst == -3) {
            emsg = xc->fCmd;
            emsg += ": cannot stat ";
            emsg += dstpath;
            emsg += ": errno: ";
            emsg += (int) errno;
         } else if (rccpdst == -4) {
            emsg = xc->fCmd;
            emsg += ": ";
            emsg += dstpath;
            emsg += ": Is not a regular file";
         }
         TRACEP(p, XERR, emsg);
         response->Send(kXR_InvalidRequest, emsg.c_str());
         return 0;
      }
   }

   // Check the format string
   if (fmt.length() <= 0) {
      fmt = xc->fFmt;
   } else {
      if (!fmt.beginswith(xc->fCmd)) {
         fmt.insert(" ", 0);
         fmt.insert(xc->fCmd, 0);
      }
      if (fmt.find("%s") == STR_NPOS) {
         fmt.insert(" %s %s", -1);
      }
   }

   // Create the command now
   XrdOucString cmd;
   XrdProofdAux::Form(cmd, fmt.c_str(), srcpath.c_str(), dstpath.c_str());
   cmd += " 2>&1";
   TRACEP(p, DBG, "Executing command: " << cmd);

   // Pipe for child-to-parent communications
   XrdProofdPipe pp;
   if (!pp.IsValid()) {
      emsg = "cannot create the pipe";
      TRACEP(p, XERR, emsg);
      response->Send(kXR_ServerError, emsg.c_str());
      return 0;
   }

   // Fork a test agent process to handle this session
   TRACEP(p, DBG, "forking to execute in the private sandbox");
   int pid = -1;
   if (!(pid = fMgr->Sched()->Fork("admincpfile"))) {
      // Child process
      // We set to the user environment as we must to run the command
      // in the user space
      if (fMgr->SessionMgr()->SetUserEnvironment(p) != 0) {
         emsg = "SetUserEnvironment did not return OK";
         rc = 1;
      } else {
         // Execute the command in a pipe
         FILE *fp = popen(cmd.c_str(), "r");
         if (!fp) {
            emsg = "could not run '"; emsg += cmd; emsg += "'";
            rc = 1;
         } else {
            // Read line by line
            char line[2048];
            while (fgets(line, sizeof(line), fp)) {
               // Parse the line
               int llen = strlen(line);
               if (llen > 0 && line[llen-1] == '\n') {
                  line[llen-1] = '\0';
                  llen--;
               }
               // Real-time sending (line-by-line)
               if (llen > 0 &&
                   response->Send(kXR_attn, kXPD_srvmsg, 4, (char *) &line[0], llen) != 0) {
                  emsg = "error sending message to requester";
                  rc = 1;
                  break;
               }
               // Check if we have been interrupted
               char b[1];
               if (p->Link()->Peek(&b[0], 1, 0) == 1) {
                  p->Process(p->Link());
                  if (p->IsCtrlC()) break;
               }
               // Reset timeout
               if (pp.Post(0, "") != 0) {
                  rc = 1;
                  break;
               }
            }
            // Close the pipe if not in error state (otherwise we may block here)
            int rcpc = 0;
            if ((rcpc = pclose(fp)) == -1) {
               emsg = "error while trying to close the command pipe";
               rc = 1;
            }
            if (WEXITSTATUS(rcpc) != 0) {
               emsg = "return code: ";
               emsg += (int) WEXITSTATUS(rcpc);
               rc = 1;
            }
            // Close the notification messages
            char cp[1] = {'\n'};
            if (response->Send(kXR_attn, kXPD_srvmsg, 3, (char *) &cp[0], 1) != 0) {
               emsg = "error sending progress notification to requester";
               rc = 1;
            }
         }
      }
      // Send error, if any
      if (rc == 1) {
         // Post Error
         if (pp.Post(-1, emsg.c_str()) != 0) rc = 1;
      }

      // End-Of-Transmission
      if (pp.Post(1, "") != 0) rc = 1;

      // Done
      exit(rc);
   }

   // Parent process
   if (pid < 0) {
      emsg = "forking failed - errno: "; emsg += (int) errno;
      return -1;
   }

   // now we wait for the callback to be (successfully) established
   TRACEP(p, DBG, "forking OK: wait for execution");

   // Read status-of-setup from pipe
   int prc = 0, rst = 0;
   // We wait for 60 secs max among transfers
   while (rst == 0 && rc >= 0) {
      while ((prc = pp.Poll(60)) > 0) {
         XpdMsg msg;
         if (pp.Recv(msg) != 0) {
            emsg = "error receiving message from pipe";;
            rc = -1;
         }
         // Status is the message type
         rst = msg.Type();
         // Read string, if any
         if (rst < 0) {
            // Error
            rc = -1;
            // Store error message
            emsg = msg.Buf();
            if (emsg.length() <= 0)
               emsg = "error reading string from received message";
         } else if (rst == 1) {
            // Done
            break;
         }
      }
      if (prc == 0) {
         emsg = "timeout from poll";
         rc = -1;
      } else if (prc < 0) {
         emsg = "error from poll - errno: "; emsg += -prc;
         rc = -1;
      }
   }

   // The parent is done
   TRACEP(p, DBG, "execution over: "<< ((rc == 0) ? "ok" : "failed"));

   if (rc != 0) {
      emsg.insert("failure: ", 0);
      TRACEP(p, XERR, emsg);
      response->Send(kXR_ServerError, emsg.c_str());
   } else {
      response->Send("OK");
   }

   // Done
   return 0;
}
