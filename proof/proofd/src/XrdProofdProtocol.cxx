// @(#)root/proofd:$Id$
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdProtocol                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// XrdProtocol implementation to coordinate 'proofserv' applications.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdProofdPlatform.h"

#include "XpdSysError.h"
#include "XpdSysLogger.h"

#include "XrdSys/XrdSysPriv.hh"
#include "XrdOuc/XrdOucStream.hh"

#include "XrdVersion.hh"
#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdScheduler.hh"

#include "XrdProofdClient.h"
#include "XrdProofdClientMgr.h"
#include "XrdProofdConfig.h"
#include "XrdProofdManager.h"
#include "XrdProofdNetMgr.h"
#include "XrdProofdPriorityMgr.h"
#include "XrdProofdProofServMgr.h"
#include "XrdProofdProtocol.h"
#include "XrdProofdResponse.h"
#include "XrdProofdProofServ.h"
#include "XrdProofSched.h"
#include "XrdROOT.h"
#include "rpdconn.h"

// Tracing utils
#include "XrdProofdTrace.h"
XrdOucTrace          *XrdProofdTrace = 0;

// Loggers: we need two to avoid deadlocks
static XrdSysLogger   gMainLogger;

//
// Static area: general protocol managing section
int                   XrdProofdProtocol::fgCount    = 0;
XpdObjectQ            XrdProofdProtocol::fgProtStack("ProtStack",
                                                     "xproofd protocol anchor");
XrdSysRecMutex        XrdProofdProtocol::fgBMutex;    // Buffer management mutex
XrdBuffManager       *XrdProofdProtocol::fgBPool    = 0;
int                   XrdProofdProtocol::fgMaxBuffsz= 0;
XrdSysError           XrdProofdProtocol::fgEDest(0, "xpd");
XrdSysLogger         *XrdProofdProtocol::fgLogger   = 0;
//
// Static area: protocol configuration section
bool                  XrdProofdProtocol::fgConfigDone = 0;
//
int                   XrdProofdProtocol::fgReadWait = 0;
// Cluster manager
XrdProofdManager     *XrdProofdProtocol::fgMgr = 0;

// Effective uid
int                   XrdProofdProtocol::fgEUidAtStartup = -1;

// Local definitions
#define MAX_ARGS 128

// Macros used to set conditional options
#ifndef XPDCOND
#define XPDCOND(n,ns) ((n == -1 && ns == -1) || (n > 0 && n >= ns))
#endif
#ifndef XPDSETSTRING
#define XPDSETSTRING(n,ns,c,s) \
 { if (XPDCOND(n,ns)) { \
     SafeFree(c); c = strdup(s.c_str()); ns = n; }}
#endif

#ifndef XPDADOPTSTRING
#define XPDADOPTSTRING(n,ns,c,s) \
  { char *t = 0; \
    XPDSETSTRING(n, ns, t, s); \
    if (t && strlen(t)) { \
       SafeFree(c); c = t; \
  } else \
       SafeFree(t); }
#endif

#ifndef XPDSETINT
#define XPDSETINT(n,ns,i,s) \
 { if (XPDCOND(n,ns)) { \
     i = strtol(s.c_str(),0,10); ns = n; }}
#endif

typedef struct {
   kXR_int32 ptyp;  // must be always 0 !
   kXR_int32 rlen;
   kXR_int32 pval;
   kXR_int32 styp;
} hs_response_t;

typedef struct ResetCtrlcGuard {
   XrdProofdProtocol *xpd;
   int                type;
   ResetCtrlcGuard(XrdProofdProtocol *p, int t) : xpd(p), type(t) { }
   ~ResetCtrlcGuard() { if (xpd && type != kXP_ctrlc) xpd->ResetCtrlC(); }
} ResetCtrlcGuard_t;

//
// Derivation of XrdProofdConfig to read the port from the config file
class XrdProofdProtCfg : public XrdProofdConfig {
public:
   int  fPort; // The port on which we listen
   XrdProofdProtCfg(const char *cfg, XrdSysError *edest = 0);
   int  DoDirective(XrdProofdDirective *, char *, XrdOucStream *, bool);
   void RegisterDirectives();
};

////////////////////////////////////////////////////////////////////////////////
/// Constructor

XrdProofdProtCfg::XrdProofdProtCfg(const char *cfg, XrdSysError *edest)
                 : XrdProofdConfig(cfg, edest)
{
   fPort = -1;
   RegisterDirectives();
}

////////////////////////////////////////////////////////////////////////////////
/// Register directives for configuration

void XrdProofdProtCfg::RegisterDirectives()
{
   Register("port", new XrdProofdDirective("port", this, &DoDirectiveClass));
   Register("xrd.protocol", new XrdProofdDirective("xrd.protocol", this, &DoDirectiveClass));
}

////////////////////////////////////////////////////////////////////////////////
/// Parse directives

int XrdProofdProtCfg::DoDirective(XrdProofdDirective *d,
                                  char *val, XrdOucStream *cfg, bool)
{
   if (!d) return -1;

   XrdOucString port(val);
   if (d->fName == "xrd.protocol") {
      port = cfg->GetWord();
      port.replace("xproofd:", "");
   } else if (d->fName != "port") {
      return -1;
   }
   if (port.length() > 0) {
      fPort = strtol(port.c_str(), 0, 10);
   }
   fPort = (fPort < 0) ? XPD_DEF_PORT : fPort;
   return 0;
}

#if (ROOTXRDVERS >= 300030000)
XrdVERSIONINFO(XrdgetProtocol,xproofd);
XrdVERSIONINFO(XrdgetProtocolPort,xproofd);
#endif

extern "C" {
////////////////////////////////////////////////////////////////////////////////
/// This protocol is meant to live in a shared library. The interface below is
/// used by the server to obtain a copy of the protocol object that can be used
/// to decide whether or not a link is talking a particular protocol.

XrdProtocol *XrdgetProtocol(const char *, char *parms, XrdProtocol_Config *pi)
{
   // Return the protocol object to be used if static init succeeds
   if (XrdProofdProtocol::Configure(parms, pi)) {

      return (XrdProtocol *) new XrdProofdProtocol(pi);
   }
   return (XrdProtocol *)0;
}

////////////////////////////////////////////////////////////////////////////////
/// This function is called early on to determine the port we need to use. The
/// The default is ostensibly 1093 but can be overidden; which we allow.

int XrdgetProtocolPort(const char * /*pname*/, char * /*parms*/, XrdProtocol_Config *pi)
{
      // Default XPD_DEF_PORT (1093)
      int port = XPD_DEF_PORT;

      if (pi) {
         XrdProofdProtCfg pcfg(pi->ConfigFN, pi->eDest);
         // Init some relevant quantities for tracing
         XrdProofdTrace = new XrdOucTrace(pi->eDest);
         pcfg.Config(0);

         if (pcfg.fPort > 0) {
            port = pcfg.fPort;
         } else {
            port = (pi && pi->Port > 0) ? pi->Port : XPD_DEF_PORT;
         }
      }
      return port;
}}

////////////////////////////////////////////////////////////////////////////////
/// Protocol constructor

XrdProofdProtocol::XrdProofdProtocol(XrdProtocol_Config *pi)
   : XrdProtocol("xproofd protocol handler"), fProtLink(this)
{
   fLink = 0;
   fArgp = 0;
   fPClient = 0;
   fSecClient = 0;
   fAuthProt = 0;
   fResponses.reserve(10);

   fStdErrFD = (pi && pi->eDest) ? pi->eDest->baseFD() : fileno(stderr);

   // Instantiate a Proofd protocol object
   Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// Get response instance corresponding to stream ID 'sid'

XrdProofdResponse *XrdProofdProtocol::Response(kXR_unt16 sid)
{
   XPDLOC(ALL, "Protocol::Response")

   TRACE(HDBG, "sid: "<<sid<<", size: "<<fResponses.size());

   if (sid > 0)
      if (sid <= fResponses.size())
         return fResponses[sid-1];

   return (XrdProofdResponse *)0;
}

////////////////////////////////////////////////////////////////////////////////
/// Create new response instance for stream ID 'sid'

XrdProofdResponse *XrdProofdProtocol::GetNewResponse(kXR_unt16 sid)
{
   XPDLOC(ALL, "Protocol::GetNewResponse")

   XrdOucString msg;
   XPDFORM(msg, "sid: %d", sid);
   if (sid > 0) {
      if (sid > fResponses.size()) {
         if (sid > fResponses.capacity()) {
            int newsz = (sid < 2 * fResponses.capacity()) ? 2 * fResponses.capacity() : sid+1 ;
            fResponses.reserve(newsz);
            if (TRACING(DBG)) {
               msg += " new capacity: ";
               msg += (int) fResponses.capacity();
            }
        }
         int nnew = sid - fResponses.size();
         while (nnew--)
            fResponses.push_back(new XrdProofdResponse());
         if (TRACING(DBG)) {
            msg += "; new size: ";
            msg += (int) fResponses.size();
         }
      }
   } else {
      TRACE(XERR,"wrong sid: "<<sid);
      return (XrdProofdResponse *)0;
   }

   TRACE(DBG, msg);

   // Done
   return fResponses[sid-1];
}

////////////////////////////////////////////////////////////////////////////////
/// Check whether the request matches this protocol

XrdProtocol *XrdProofdProtocol::Match(XrdLink *lp)
{
   XPDLOC(ALL, "Protocol::Match")

   struct ClientInitHandShake hsdata;
   char  *hsbuff = (char *)&hsdata;

   static hs_response_t hsresp = {0, 0, kXR_int32(htonl(XPROOFD_VERSBIN)), 0};

   XrdProtocol *xp = nullptr;
   int dlen;
   TRACE(HDBG, "enter");

   XrdOucString emsg;
   // Peek at the first 20 bytes of data
   if ((dlen = lp->Peek(hsbuff,sizeof(hsdata),fgReadWait)) != sizeof(hsdata)) {
      if (dlen <= 0) lp->setEtext("Match: handshake not received");
      if (dlen == 12) {
         // Check if it is a request to open a file via 'rootd', unsupported
         hsdata.first = ntohl(hsdata.first);
         if (hsdata.first == 8) {
            emsg = "rootd-file serving not supported any-longer";
         }
         if (emsg.length() > 0) {
            lp->setEtext(emsg.c_str());
         } else {
            lp->setEtext("link transfered");
         }
         return xp;
      }
      TRACE(XERR, "peeked incomplete or empty information! (dlen: "<<dlen<<" bytes)");
      return xp;
   }

   // If this is is not our protocol, we check if it a data serving request via xrootd
   hsdata.third  = ntohl(hsdata.third);
   if (dlen != sizeof(hsdata) ||  hsdata.first || hsdata.second
       || !(hsdata.third == 1) || hsdata.fourth || hsdata.fifth) {

      // Check if it is a request to open a file via 'xrootd'
      if (fgMgr->Xrootd() && (xp = fgMgr->Xrootd()->Match(lp))) {
         TRACE(ALL, "matched xrootd protocol on link: serving a file");
      } else {
         TRACE(XERR, "failed to match any known or enabled protocol");
      }
      return xp;
   }

   // Respond to this request with the handshake response
   if (!lp->Send((char *)&hsresp, sizeof(hsresp))) {
      lp->setEtext("Match: handshake failed");
      TRACE(XERR, "handshake failed");
      return xp;
   }

   // We can now read all 20 bytes and discard them (no need to wait for it)
   int len = sizeof(hsdata);
   if (lp->Recv(hsbuff, len) != len) {
      lp->setEtext("Match: reread failed");
      TRACE(XERR, "reread failed");
      return xp;
   }

   // Get a protocol object off the stack (if none, allocate a new one)
   XrdProofdProtocol *xpp = nullptr;
   if (!(xpp = fgProtStack.Pop()))
      xpp = new XrdProofdProtocol();

   // Bind the protocol to the link and return the protocol
   xpp->fLink = lp;
   snprintf(xpp->fSecEntity.prot, XrdSecPROTOIDSIZE, "host");
   xpp->fSecEntity.host = strdup((char *)lp->Host());

   // Dummy data used by 'proofd'
   kXR_int32 dum[2];
   if (xpp->GetData("dummy",(char *)&dum[0],sizeof(dum)) != 0) {
      xpp->Recycle(0,0,0);
   }

   xp = (XrdProtocol *) xpp;

   // We are done
   return xp;
}

////////////////////////////////////////////////////////////////////////////////
/// Return statistics info about the protocol.
/// Not really implemented yet: this is a reduced XrdXrootd version.

int XrdProofdProtocol::Stats(char *buff, int blen, int)
{
   static char statfmt[] = "<stats id=\"xproofd\"><num>%ld</num></stats>";

   // If caller wants only size, give it to them
   if (!buff)
      return sizeof(statfmt)+16;

   // We have only one statistic -- number of successful matches
   return snprintf(buff, blen, statfmt, fgCount);
}

////////////////////////////////////////////////////////////////////////////////
/// Reset static and local vars

void XrdProofdProtocol::Reset()
{
   // Init local vars
   fLink      = 0;
   fPid       = -1;
   fArgp      = 0;
   fStatus    = 0;
   fClntCapVer = 0;
   fConnType  = kXPD_ClientMaster;
   fSuperUser = 0;
   fPClient   = 0;
   fUserIn    = "";
   fGroupIn   = "";
   fCID       = -1;
   fTraceID   = "";
   fAdminPath = "";
   if (fAuthProt) {
      fAuthProt->Delete();
      fAuthProt = 0;
   }
   fSecEntity = XrdSecEntity();
   // Cleanup existing XrdProofdResponse objects
   std::vector<XrdProofdResponse *>::iterator ii = fResponses.begin(); // One per each logical connection
   while (ii != fResponses.end()) {
      (*ii)->Reset();
      ++ii;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Protocol configuration tool
/// Function: Establish configuration at load time.
/// Output: 1 upon success or 0 otherwise.

int XrdProofdProtocol::Configure(char *parms, XrdProtocol_Config *pi)
{
   XPDLOC(ALL, "Protocol::Configure")

   XrdOucString mp;

   // Only once
   if (fgConfigDone)
      return 1;
   fgConfigDone = 1;

   // Copy out the special info we want to use at top level
   fgLogger = pi->eDest->logger();
   fgEDest.logger(fgLogger);
   if (XrdProofdTrace) delete XrdProofdTrace; // It could have been initialized in XrdgetProtocolPort
   XrdProofdTrace = new XrdOucTrace(&fgEDest);
   fgBPool        = pi->BPool;
   fgReadWait     = pi->readWait;

   // Pre-initialize some i/o values
   fgMaxBuffsz = fgBPool->MaxSize();

   // Schedule protocol object cleanup; the maximum number of objects
   // and the max age are taken from XrdXrootdProtocol: this may need
   // some optimization in the future.
#if 1
   fgProtStack.Set(pi->Sched, XrdProofdTrace, TRACE_MEM);
   fgProtStack.Set((pi->ConnMax/3 ? pi->ConnMax/3 : 30), 60*60);
#else
   fgProtStack.Set(pi->Sched, 3600);
#endif

   // Default tracing options: always trace logins and errors for all
   // domains; if the '-d' option was specified on the command line then
   // trace also REQ and FORM.
   // NB: these are superseeded by settings in the config file (xpd.trace)
   XrdProofdTrace->What = TRACE_DOMAINS;
   TRACESET(XERR, 1);
   TRACESET(LOGIN, 1);
   TRACESET(RSP, 0);
   if (pi->DebugON)
      XrdProofdTrace->What |= (TRACE_REQ | TRACE_FORK);

   // Work as root to avoid contineous changes of the effective user
   // (users are logged in their box after forking)
   fgEUidAtStartup = geteuid();
   if (!getuid()) XrdSysPriv::ChangePerm((uid_t)0, (gid_t)0);

   // Process the config file for directives meaningful to us
   // Create and Configure the manager
   fgMgr = new XrdProofdManager(parms, pi, &fgEDest);
   if (fgMgr->Config(0)) return 0;
   mp = "global manager created";
   TRACE(ALL, mp);

   // Issue herald indicating we configured successfully
   TRACE(ALL, "xproofd protocol version "<<XPROOFD_VERSION<<
              " build "<<XrdVERSION<<" successfully loaded");

   // Return success
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Process the information received on the active link.
/// (We ignore the argument here)

int XrdProofdProtocol::Process(XrdLink *)
{
   XPDLOC(ALL, "Protocol::Process")

   int rc = 0;
   TRACET(TraceID(), DBG, "instance: " << this);

   // Read the next request header
   if ((rc = GetData("request", (char *)&fRequest, sizeof(fRequest))) != 0)
      return rc;
   TRACET(TraceID(), HDBG, "after GetData: rc: " << rc);

   // Deserialize the data
   fRequest.header.requestid = ntohs(fRequest.header.requestid);
   fRequest.header.dlen      = ntohl(fRequest.header.dlen);

   // Get response object
   kXR_unt16 sid;
   memcpy((void *)&sid, (const void *)&(fRequest.header.streamid[0]), 2);
   XrdProofdResponse *response = 0;
   if (!(response = Response(sid))) {
      if (!(response = GetNewResponse(sid))) {
         TRACET(TraceID(), XERR, "could not get Response instance for rid: "<< sid);
         return rc;
      }
   }
   // Set the stream ID for the reply
   response->Set(fRequest.header.streamid);
   response->Set(fLink);

   TRACET(TraceID(), REQ, "sid: " << sid << ", req id: " << fRequest.header.requestid <<
                " (" << XrdProofdAux::ProofRequestTypes(fRequest.header.requestid)<<
                ")" << ", dlen: " <<fRequest.header.dlen);

   // Every request has an associated data length. It better be >= 0 or we won't
   // be able to know how much data to read.
   if (fRequest.header.dlen < 0) {
      response->Send(kXR_ArgInvalid, "Process: Invalid request data length");
      return fLink->setEtext("Process: protocol data length error");
   }

   // Read any argument data at this point, except when the request is to forward
   // a buffer: the argument may have to be segmented and we're not prepared to do
   // that here.
   if (fRequest.header.requestid != kXP_sendmsg && fRequest.header.dlen) {
      if ((fArgp = GetBuff(fRequest.header.dlen+1, fArgp)) == 0) {
         response->Send(kXR_ArgTooLong, "fRequest.argument is too long");
         return rc;
      }
      if ((rc = GetData("arg", fArgp->buff, fRequest.header.dlen)))
         return rc;
      fArgp->buff[fRequest.header.dlen] = '\0';
   }

   // Continue with request processing at the resume point
   return Process2();
}

////////////////////////////////////////////////////////////////////////////////
/// Local processing method: here the request is dispatched to the appropriate
/// method

int XrdProofdProtocol::Process2()
{
   XPDLOC(ALL, "Protocol::Process2")

   int rc = 0;
   XPD_SETRESP(this, "Process2");

   TRACET(TraceID(), REQ, "req id: " << fRequest.header.requestid << " (" <<
                XrdProofdAux::ProofRequestTypes(fRequest.header.requestid) << ")");

   ResetCtrlcGuard_t ctrlcguard(this, fRequest.header.requestid);

   // If the user is logged in check if the wanted action is to be done by us
   if (fStatus && (fStatus & XPD_LOGGEDIN)) {
      // Record time of the last action
      TouchAdminPath();
      // We must have a client instance if here
      if (!fPClient) {
         TRACET(TraceID(), XERR, "client undefined!!! ");
         response->Send(kXR_InvalidRequest,"client undefined!!! ");
         return 0;
      }
      bool formgr = 0;
      switch(fRequest.header.requestid) {
         case kXP_ctrlc:
            rc = CtrlC();
            break;
         case kXP_touch:
            // Reset the asked-to-touch flag, if it was never set
            fPClient->Touch(1);
            break;
         case kXP_interrupt:
            rc = Interrupt();
            break;
         case kXP_ping:
            rc = Ping();
            break;
         case kXP_sendmsg:
            rc = SendMsg();
            break;
         case kXP_urgent:
            rc = Urgent();
            break;
         default:
            formgr = 1;
      }
      if (!formgr) {
         // Check the link
         if (!fLink || (fLink->FDnum() <= 0)) {
            TRACE(XERR, "link is undefined! ");
            return -1;
         }
         return rc;
      }
   }

   // The request is for the manager
   rc = fgMgr->Process(this);
   // Check the link
   if (!fLink || (fLink->FDnum() <= 0)) {
      TRACE(XERR, "link is undefined! ");
      return -1;
   }
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Recycle call. Release the instance and give it back to the stack.

void XrdProofdProtocol::Recycle(XrdLink *, int, const char *)
{
   XPDLOC(ALL, "Protocol::Recycle")

   const char *srvtype[6] = {"ANY", "MasterWorker", "MasterMaster",
                             "ClientMaster", "Internal", "Admin"};
   XrdOucString buf;

   // Document the disconnect
   if (fPClient)
      XPDFORM(buf, "user %s disconnected; type: %s", fPClient->User(),
                   srvtype[fConnType+1]);
   else
      XPDFORM(buf, "user disconnected; type: %s", srvtype[fConnType+1]);
   TRACET(TraceID(), LOGIN, buf);

   // If we have a buffer, release it
   if (fArgp) {
      fgBPool->Release(fArgp);
      fArgp = 0;
   }

   // Locate the client instance
   XrdProofdClient *pmgr = fPClient;

   if (pmgr) {
      if (!Internal()) {

         TRACE(REQ,"External disconnection of protocol associated with pid "<<fPid);

         // Write disconnection file
         XrdOucString discpath(fAdminPath);
         discpath.replace("/cid", "/disconnected");
         FILE *fd = fopen(discpath.c_str(), "w");
         if (!fd && errno != ENOENT) {
            TRACE(XERR, "unable to create path: " <<discpath<<" (errno: "<<errno<<")");
         } else if (fd) {
            fclose(fd);
         }

         // Remove protocol and response from attached client/proofserv instances
         // Set reconnect flag if proofserv instances attached to this client are still running
         pmgr->ResetClientSlot(fCID);
         if(fgMgr && fgMgr->SessionMgr()) {
            XrdSysMutexHelper mhp(fgMgr->SessionMgr()->Mutex());

            fgMgr->SessionMgr()->DisconnectFromProofServ(fPid);
            if((fConnType == 0) && fgMgr->SessionMgr()->Alive(this)) {
               TRACE(REQ, "Non-destroyed proofserv processes attached to this protocol ("<<this<<
                          "), setting reconnect time");
               fgMgr->SessionMgr()->SetReconnectTime(true);
            }
            fgMgr->SessionMgr()->CheckActiveSessions(0);
         } else {
            TRACE(XERR, "No XrdProofdMgr ("<<fgMgr<<") or SessionMgr ("
                        <<(fgMgr ? fgMgr->SessionMgr() : (void *) -1)<<")")
         }

      } else {

         // Internal connection: we need to remove this instance from the list
         // of proxy servers and to notify the attached clients.
         // Tell the session manager that this session has gone
         if (fgMgr && fgMgr->SessionMgr()) {
            XrdSysMutexHelper mhp(fgMgr->SessionMgr()->Mutex());
            TRACE(HDBG, "fAdminPath: "<<fAdminPath);
            buf.assign(fAdminPath, fAdminPath.rfind('/') + 1, -1);
            fgMgr->SessionMgr()->DeleteFromSessions(buf.c_str());
            // Move the entry to the terminated sessions area
            fgMgr->SessionMgr()->MvSession(buf.c_str());
         }
         else {
            TRACE(XERR,"No XrdProofdMgr ("<<fgMgr<<") or SessionMgr ("<<fgMgr->SessionMgr()<<")")
         }
      }
   }
   // Set fields to starting point (debugging mostly)
   Reset();

   // Push ourselves on the stack
   fgProtStack.Push(&fProtLink);
#if 0
   if(fgProtStack.Push(&fProtLink) != 0) {
      XrdProofdProtocol *xp = fProtLink.objectItem();
      fProtLink.setItem(0);
      delete xp;
   }
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// Allocate a buffer to handle quantum bytes; if argp points to an existing
/// buffer, its size is checked and re-allocated if needed

XrdBuffer *XrdProofdProtocol::GetBuff(int quantum, XrdBuffer *argp)
{
   XPDLOC(ALL, "Protocol::GetBuff")

   TRACE(HDBG, "len: "<<quantum);

   // If we are given an existing buffer, we keep it if we use at least half
   // of it; otherwise we take a smaller one
   if (argp) {
      if (quantum >= argp->bsize / 2 && quantum <= argp->bsize)
         return argp;
   }

   // Release the buffer if too small
   XrdSysMutexHelper mh(fgBMutex);
   if (argp)
      fgBPool->Release(argp);

   // Obtain a new one
   if ((argp = fgBPool->Obtain(quantum)) == 0) {
      TRACE(XERR, "could not get requested buffer (size: "<<quantum<<
                  ") = insufficient memory");
   } else {
      TRACE(HDBG, "quantum: "<<quantum<<
                  ", buff: "<<(void *)(argp->buff)<<", bsize:"<<argp->bsize);
   }

   // Done
   return argp;
}

////////////////////////////////////////////////////////////////////////////////
/// Release a buffer previously allocated via GetBuff

void XrdProofdProtocol::ReleaseBuff(XrdBuffer *argp)
{
   XrdSysMutexHelper mh(fgBMutex);
   fgBPool->Release(argp);
}

////////////////////////////////////////////////////////////////////////////////
/// Get data from the open link

int XrdProofdProtocol::GetData(const char *dtype, char *buff, int blen)
{
   XPDLOC(ALL, "Protocol::GetData")

   int rlen;

   // Read the data but reschedule the link if we have not received all of the
   // data within the timeout interval.
   TRACET(TraceID(), HDBG, "dtype: "<<(dtype ? dtype : " - ")<<", blen: "<<blen);

   // No need to lock:the link is disable while we are here
   rlen = fLink->Recv(buff, blen, fgReadWait);
   if (rlen  < 0) {
      if (rlen != -ENOMSG && rlen != -ECONNRESET) {
         XrdOucString emsg = "link read error: errno: ";
         emsg += -rlen;
         TRACET(TraceID(), XERR, emsg.c_str());
         return (fLink ? fLink->setEtext(emsg.c_str()) : -1);
      } else {
         TRACET(TraceID(), HDBG, "connection closed by peer (errno: "<<-rlen<<")");
         return -1;
      }
   }
   if (rlen < blen) {
      TRACET(TraceID(), DBG, dtype << " timeout; read " <<rlen <<" of " <<blen <<" bytes - rescheduling");
      return 1;
   }
   TRACET(TraceID(), HDBG, "rlen: "<<rlen);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Send data over the open link. Segmentation is done here, if required.

int XrdProofdProtocol::SendData(XrdProofdProofServ *xps,
                                kXR_int32 sid, XrdSrvBuffer **buf, bool savebuf)
{
   XPDLOC(ALL, "Protocol::SendData")

   int rc = 0;

   TRACET(TraceID(), HDBG, "length: "<<fRequest.header.dlen<<" bytes ");

   // Buffer length
   int len = fRequest.header.dlen;

   // Quantum size
   int quantum = (len > fgMaxBuffsz ? fgMaxBuffsz : len);

   // Get a buffer
   XrdBuffer *argp = XrdProofdProtocol::GetBuff(quantum);
   if (!argp) return -1;

   // Now send over all of the data as unsolicited messages
   XrdOucString msg;
   while (len > 0) {

      XrdProofdResponse *response = (sid > -1) ? xps->Response() : 0;

      if ((rc = GetData("data", argp->buff, quantum))) {
         { XrdSysMutexHelper mh(fgBMutex); fgBPool->Release(argp); }
         return -1;
      }
      if (buf && !(*buf) && savebuf)
         *buf = new XrdSrvBuffer(argp->buff, quantum, 1);
      // Send
      if (sid > -1) {
         if (TRACING(HDBG))
            XPDFORM(msg, "EXT: server ID: %d, sending: %d bytes", sid, quantum);
         if (!response || response->Send(kXR_attn, kXPD_msgsid, sid,
                                         argp->buff, quantum) != 0) {
            { XrdSysMutexHelper mh(fgBMutex); fgBPool->Release(argp); }
            XPDFORM(msg, "EXT: server ID: %d, problems sending: %d bytes to server",
                         sid, quantum);
            TRACET(TraceID(), XERR, msg);
            return -1;
         }
      } else {

         // Get ID of the client
         int cid = ntohl(fRequest.sendrcv.cid);
         if (TRACING(HDBG))
            XPDFORM(msg, "INT: client ID: %d, sending: %d bytes", cid, quantum);
         if (xps->SendData(cid, argp->buff, quantum) != 0) {
            { XrdSysMutexHelper mh(fgBMutex); fgBPool->Release(argp); }
            XPDFORM(msg, "INT: client ID: %d, problems sending: %d bytes to client",
                         cid, quantum);
            TRACET(TraceID(), XERR, msg);
            return -1;
         }
      }
      TRACET(TraceID(), HDBG, msg);
      // Next segment
      len -= quantum;
      if (len < quantum)
         quantum = len;
   }

   // Release the buffer
   { XrdSysMutexHelper mh(fgBMutex); fgBPool->Release(argp); }

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Send data over the open client links of session 'xps'.
/// Used when all the connected clients are eligible to receive the message.
/// Segmentation is done here, if required.

int XrdProofdProtocol::SendDataN(XrdProofdProofServ *xps,
                                 XrdSrvBuffer **buf, bool savebuf)
{
   XPDLOC(ALL, "Protocol::SendDataN")

   int rc = 0;

   TRACET(TraceID(), HDBG, "length: "<<fRequest.header.dlen<<" bytes ");

   // Buffer length
   int len = fRequest.header.dlen;

   // Quantum size
   int quantum = (len > fgMaxBuffsz ? fgMaxBuffsz : len);

   // Get a buffer
   XrdBuffer *argp = XrdProofdProtocol::GetBuff(quantum);
   if (!argp) return -1;

   // Now send over all of the data as unsolicited messages
   while (len > 0) {
      if ((rc = GetData("data", argp->buff, quantum))) {
         XrdProofdProtocol::ReleaseBuff(argp);
         return -1;
      }
      if (buf && !(*buf) && savebuf)
         *buf = new XrdSrvBuffer(argp->buff, quantum, 1);

      // Send to connected clients
      if (xps->SendDataN(argp->buff, quantum) != 0) {
         XrdProofdProtocol::ReleaseBuff(argp);
         return -1;
      }

      // Next segment
      len -= quantum;
      if (len < quantum)
         quantum = len;
   }

   // Release the buffer
   XrdProofdProtocol::ReleaseBuff(argp);

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle a request to forward a message to another process

int XrdProofdProtocol::SendMsg()
{
   XPDLOC(ALL, "Protocol::SendMsg")

   static const char *crecv[5] = {"master proofserv", "top master",
                                  "client", "undefined", "any"};
   int rc = 0;

   XPD_SETRESP(this, "SendMsg");

   // Unmarshall the data
   int psid = ntohl(fRequest.sendrcv.sid);
   int opt = ntohl(fRequest.sendrcv.opt);

   XrdOucString msg;
   // Find server session
   XrdProofdProofServ *xps = 0;
   if (!fPClient || !(xps = fPClient->GetServer(psid))) {
      XPDFORM(msg, "%s: session ID not found: %d", (Internal() ? "INT" : "EXT"), psid);
      TRACET(TraceID(), XERR, msg.c_str());
      response->Send(kXR_InvalidRequest, msg.c_str());
      return 0;
   }

   // Message length
   int len = fRequest.header.dlen;

   if (!Internal()) {

      if (TRACING(HDBG)) {
         // Notify
         XPDFORM(msg, "EXT: sending %d bytes to proofserv (psid: %d, xps: %p, status: %d,"
                     " cid: %d)", len, psid, xps, xps->Status(), fCID);
         TRACET(TraceID(), HDBG, msg.c_str());
      }

      // Send to proofsrv our client ID
      if (fCID == -1) {
         TRACET(TraceID(), REQ, "EXT: error getting clientSID");
         response->Send(kXP_ServerError,"EXT: getting clientSID");
         return 0;
      }
      if (SendData(xps, fCID)) {
         TRACET(TraceID(), REQ, "EXT: error sending message to proofserv");
         response->Send(kXP_reconnecting,"EXT: sending message to proofserv");
         return 0;
      }

      // Notify to user
      response->Send();

   } else {

      if (TRACING(HDBG)) {
          // Notify
          XPDFORM(msg, "INT: sending %d bytes to client/master (psid: %d, xps: %p, status: %d)",
                       len, psid, xps, xps->Status());
          TRACET(TraceID(), HDBG, msg.c_str());
      }
      bool saveStartMsg = 0;
      XrdSrvBuffer *savedBuf = 0;
      // Additional info about the message
      if (opt & kXPD_setidle) {
         TRACET(TraceID(), DBG, "INT: setting proofserv in 'idle' state");
         xps->SetStatus(kXPD_idle);
         PostSession(-1, fPClient->UI().fUser.c_str(),
                         fPClient->UI().fGroup.c_str(), xps);
      } else if (opt & kXPD_querynum) {
         TRACET(TraceID(), DBG, "INT: got message with query number");
      } else if (opt & kXPD_startprocess) {
         TRACET(TraceID(), DBG, "INT: setting proofserv in 'running' state");
         xps->SetStatus(kXPD_running);
         PostSession(1, fPClient->UI().fUser.c_str(),
                        fPClient->UI().fGroup.c_str(), xps);
         // Save start processing message for later clients
         xps->DeleteStartMsg();
         saveStartMsg = 1;
      } else if (opt & kXPD_logmsg) {
         // We broadcast log messages only not idle to catch the
         // result from processing
         if (xps->Status() == kXPD_running) {
            TRACET(TraceID(), DBG, "INT: broadcasting log message");
            opt |= kXPD_fb_prog;
         }
      }
      bool fbprog = (opt & kXPD_fb_prog);

      if (!fbprog) {
         //
         // The message is strictly for the client requiring it
         if (SendData(xps, -1, &savedBuf, saveStartMsg) != 0) {
            response->Send(kXP_reconnecting,
                           "SendMsg: INT: session is reconnecting: retry later");
            return 0;
         }
      } else {
         // Send to all connected clients
         if (SendDataN(xps, &savedBuf, saveStartMsg) != 0) {
            response->Send(kXP_reconnecting,
                           "SendMsg: INT: session is reconnecting: retry later");
            return 0;
         }
      }
      // Save start processing messages, if required
      if (saveStartMsg)
         xps->SetStartMsg(savedBuf);

      if (TRACING(DBG)) {
         int ii = xps->SrvType();
         if (ii > 3) ii = 3;
         if (ii < 0) ii = 4;
         XPDFORM(msg, "INT: message sent to %s (%d bytes)", crecv[ii], len);
         TRACET(TraceID(), DBG, msg);
      }
      // Notify to proofsrv
      response->Send();
   }

   // Over
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle generic request of a urgent message to be forwarded to the server

int XrdProofdProtocol::Urgent()
{
   XPDLOC(ALL, "Protocol::Urgent")

   unsigned int rc = 0;

   XPD_SETRESP(this, "Urgent");

   // Unmarshall the data
   int psid = ntohl(fRequest.proof.sid);
   int type = ntohl(fRequest.proof.int1);
   int int1 = ntohl(fRequest.proof.int2);
   int int2 = ntohl(fRequest.proof.int3);

   TRACET(TraceID(), REQ, "psid: "<<psid<<", type: "<< type);

   // Find server session
   XrdProofdProofServ *xps = 0;
   if (!fPClient || !(xps = fPClient->GetServer(psid))) {
      TRACET(TraceID(), XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"Urgent: session ID not found");
      return 0;
   }

   TRACET(TraceID(), DBG, "xps: "<<xps<<", status: "<<xps->Status());

   // Check ID matching
   if (!xps->Match(psid)) {
      response->Send(kXP_InvalidRequest,"Urgent: IDs do not match - do nothing");
      return 0;
   }

   // Check the link to the session
   if (!xps->Response()) {
      response->Send(kXP_InvalidRequest,"Urgent: session response object undefined - do nothing");
      return 0;
   }

   // Prepare buffer
   int len = 3 *sizeof(kXR_int32);
   char *buf = new char[len];
   // Type
   kXR_int32 itmp = static_cast<kXR_int32>(htonl(type));
   memcpy(buf, &itmp, sizeof(kXR_int32));
   // First info container
   itmp = static_cast<kXR_int32>(htonl(int1));
   memcpy(buf + sizeof(kXR_int32), &itmp, sizeof(kXR_int32));
   // Second info container
   itmp = static_cast<kXR_int32>(htonl(int2));
   memcpy(buf + 2 * sizeof(kXR_int32), &itmp, sizeof(kXR_int32));
   // Send over
   if (xps->Response()->Send(kXR_attn, kXPD_urgent, buf, len) != 0) {
      response->Send(kXP_ServerError,
                     "Urgent: could not propagate request to proofsrv");
      return 0;
   }

   // Notify to user
   response->Send();
   TRACET(TraceID(), DBG, "request propagated to proofsrv");

   // Over
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle an interrupt request

int XrdProofdProtocol::Interrupt()
{
   XPDLOC(ALL, "Protocol::Interrupt")

   int rc = 0;

   XPD_SETRESP(this, "Interrupt");

   // Unmarshall the data
   int psid = ntohl(fRequest.interrupt.sid);
   int type = ntohl(fRequest.interrupt.type);
   TRACET(TraceID(), REQ, "psid: "<<psid<<", type:"<<type);

   // Find server session
   XrdProofdProofServ *xps = 0;
   if (!fPClient || !(xps = fPClient->GetServer(psid))) {
      TRACET(TraceID(), XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"Interrupt: session ID not found");
      return 0;
   }

   if (xps) {

      // Check ID matching
      if (!xps->Match(psid)) {
         response->Send(kXP_InvalidRequest,"Interrupt: IDs do not match - do nothing");
         return 0;
      }

      XrdOucString msg;
      XPDFORM(msg, "xps: %p, link ID: %s, proofsrv PID: %d",
                   xps, xps->Response()->TraceID(), xps->SrvPID());
      TRACET(TraceID(), DBG, msg.c_str());

      // Propagate the type as unsolicited
      if (xps->Response()->Send(kXR_attn, kXPD_interrupt, type) != 0) {
         response->Send(kXP_ServerError,
                        "Interrupt: could not propagate interrupt code to proofsrv");
         return 0;
      }

      // Notify to user
      response->Send();
      TRACET(TraceID(), DBG, "interrupt propagated to proofsrv");
   }

   // Over
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle a ping request.
/// For internal connections, ping is done asynchronously to avoid locking
/// problems; the session checker verifies that the admin file has been touched
/// recently enough; touching is done in Process2, so we have nothing to do here

int XrdProofdProtocol::Ping()
{
   XPDLOC(ALL, "Protocol::Ping")

   int rc = 0;
   if (Internal()) {
      if (TRACING(HDBG)) {
         XPD_SETRESP(this, "Ping");
         TRACET(TraceID(),  HDBG, "INT: nothing to do ");
      }
      return 0;
   }
   XPD_SETRESP(this, "Ping");

   // Unmarshall the data
   int psid = ntohl(fRequest.sendrcv.sid);
   int asyncopt = ntohl(fRequest.sendrcv.opt);

   TRACET(TraceID(), REQ, "psid: "<<psid<<", async: "<<asyncopt);

   // For connections to servers find the server session; manager connections
   // (psid == -1) do not have any session attached
   XrdProofdProofServ *xps = 0;
   if (!fPClient || (psid > -1 && !(xps = fPClient->GetServer(psid)))) {
      TRACET(TraceID(),  XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"session ID not found");
      return 0;
   }

   // For manager connections we are done
   kXR_int32 pingres = (psid > -1) ? 0 : 1;
   if (psid > -1 && xps && xps->IsValid()) {

      TRACET(TraceID(),  DBG, "EXT: psid: "<<psid);

      // This is the max time we will privide an answer
      kXR_int32 checkfq = fgMgr->SessionMgr()->CheckFrequency();

      // If asynchronous return the timeout for an answer
      if (asyncopt == 1) {
         TRACET(TraceID(), DBG, "EXT: async: notifying timeout to client: "<<checkfq<<" secs");
         response->Send(kXR_ok, checkfq);
      }

      // Admin path
      XrdOucString path(xps->AdminPath());
      if (path.length() <= 0) {
         TRACET(TraceID(),  XERR, "EXT: admin path is empty! - protocol error");
         if (asyncopt == 0)
            response->Send(kXP_ServerError, "EXT: admin path is empty! - protocol error");
         return 0;
      }
      path += ".status";

      // Current time
      int now = time(0);

      // Stat the admin file
      struct stat st0;
      if (stat(path.c_str(), &st0) != 0) {
         TRACET(TraceID(),  XERR, "EXT: cannot stat admin path: "<<path);
         if (asyncopt == 0)
            response->Send(kXP_ServerError, "EXT: cannot stat admin path");
         return 0;
      }

      // Take the pid
      int pid = xps->SrvPID();
      // If the session is alive ...
      if (XrdProofdAux::VerifyProcessByID(pid) != 0) {
         // If it as not touched during the last ~checkfq secs we ask for a refresh
         if ((now - st0.st_mtime) > checkfq - 5) {
            // Send the request (asking for further propagation)
            if (xps->VerifyProofServ(1) != 0) {
               TRACET(TraceID(),  XERR, "EXT: could not send verify request to proofsrv");
               if (asyncopt == 0)
                  response->Send(kXP_ServerError, "EXT: could not verify reuqest to proofsrv");
               return 0;
            }
            // Wait for the action for checkfq secs, checking every 1 sec
            struct stat st1;
            int ns = checkfq;
            while (ns--) {
               if (stat(path.c_str(), &st1) == 0) {
                  if (st1.st_mtime > st0.st_mtime) {
                     pingres = 1;
                     break;
                  }
               }
               // Wait 1 sec
               TRACET(TraceID(), DBG, "EXT: waiting "<<ns<<" secs for session "<<pid<<
                                 " to touch the admin path");
               sleep(1);
            }

         } else {
            // Session is alive
            pingres = 1;
         }
      } else {
         // Session is dead
         pingres = 0;
      }

      // Notify the client
      TRACET(TraceID(), DBG, "EXT: notified the result to client: "<<pingres);
      if (asyncopt == 0) {
         response->Send(kXR_ok, pingres);
      } else {
         // Prepare buffer for asynchronous notification
         int len = sizeof(kXR_int32);
         char *buf = new char[len];
         // Option
         kXR_int32 ifw = (kXR_int32)0;
         ifw = static_cast<kXR_int32>(htonl(ifw));
         memcpy(buf, &ifw, sizeof(kXR_int32));
         response->Send(kXR_attn, kXPD_ping, buf, len);
      }
      return 0;
   } else if (psid > -1)  {
      // This is a failure for connections to sessions
      TRACET(TraceID(), XERR, "session ID not found: "<<psid);
   }

   // Send the result
   response->Send(kXR_ok, pingres);

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Post change of session status

void XrdProofdProtocol::PostSession(int on, const char *u, const char *g,
                                    XrdProofdProofServ *xps)
{
   XPDLOC(ALL, "Protocol::PostSession")

   // Tell the priority manager
   if (fgMgr && fgMgr->PriorityMgr()) {
      int pid = (xps) ? xps->SrvPID() : -1;
      if (pid < 0) {
         TRACE(XERR, "undefined session or process id");
         return;
      }
      XrdOucString buf;
      XPDFORM(buf, "%d %s %s %d", on, u, g, pid);

      if (fgMgr->PriorityMgr()->Pipe()->Post(XrdProofdPriorityMgr::kChangeStatus,
                                             buf.c_str()) != 0) {
         TRACE(XERR, "problem posting the prority manager pipe");
      }
   }
   // Tell the scheduler
   if (fgMgr && fgMgr->ProofSched()) {
      if (on == -1 && xps && xps->SrvType() == kXPD_TopMaster) {
         TRACE(DBG, "posting the scheduler pipe");
         if (fgMgr->ProofSched()->Pipe()->Post(XrdProofSched::kReschedule, 0) != 0) {
            TRACE(XERR, "problem posting the scheduler pipe");
         }
      }
   }
   // Tell the session manager
   if (fgMgr && fgMgr->SessionMgr()) {
      if (fgMgr->SessionMgr()->Pipe()->Post(XrdProofdProofServMgr::kChgSessionSt, 0) != 0) {
         TRACE(XERR, "problem posting the session manager pipe");
      }
   }
   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Recording time of the last request on this instance

void XrdProofdProtocol::TouchAdminPath()
{
   XPDLOC(ALL, "Protocol::TouchAdminPath")

   XPD_SETRESPV(this, "TouchAdminPath");
   TRACET(TraceID(), HDBG, fAdminPath);

   if (fAdminPath.length() > 0) {
      int rc = 0;
      if ((rc = XrdProofdAux::Touch(fAdminPath.c_str())) != 0) {
         // In the case the file was not found and the connetion is internal
         // try also the terminated sessions, as the file could have been moved
         // in the meanwhile
         XrdOucString apath = fAdminPath;
         if (rc == -ENOENT && Internal()) {
            apath.replace("/activesessions/", "/terminatedsessions/");
            apath.replace(".status", "");
            rc = XrdProofdAux::Touch(apath.c_str());
         }
         if (rc != 0 && rc != -ENOENT) {
            const char *type = Internal() ? "internal" : "external";
            TRACET(TraceID(), XERR, type<<": problems touching "<<apath<<"; errno: "<<-rc);
         }
      }
   }
   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Set and propagate a Ctrl-C request

int XrdProofdProtocol::CtrlC()
{
   XPDLOC(ALL, "Protocol::CtrlC")

   TRACET(TraceID(), ALL, "handling request");

   { XrdSysMutexHelper mhp(fCtrlcMutex);
      fIsCtrlC = 1;
   }

   // Propagate now
   if (fgMgr) {
      if (fgMgr->SrvType() != kXPD_Worker) {
         if (fgMgr->NetMgr()) {
            fgMgr->NetMgr()->BroadcastCtrlC(Client()->User());
         }
      }
   }

   // Over
   return 0;
}
