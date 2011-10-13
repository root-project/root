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

#ifdef OLDXRDOUC
#  include "XrdOuc/XrdOucError.hh"
#  include "XrdOuc/XrdOucLogger.hh"
#else
#  include "XrdSys/XrdSysError.hh"
#  include "XrdSys/XrdSysLogger.hh"
#endif
#include "XrdSys/XrdSysPriv.hh"
#include "XrdOuc/XrdOucStream.hh"

#include "XrdVersion.hh"
#include "Xrd/XrdBuffer.hh"
#include "XrdNet/XrdNetDNS.hh"

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
XrdObjectQ<XrdProofdProtocol>
                      XrdProofdProtocol::fgProtStack("ProtStack",
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

//__________________________________________________________________________
XrdProofdProtCfg::XrdProofdProtCfg(const char *cfg, XrdSysError *edest)
                 : XrdProofdConfig(cfg, edest)
{
   // Constructor

   fPort = -1;
   RegisterDirectives();
}

//__________________________________________________________________________
void XrdProofdProtCfg::RegisterDirectives()
{
   // Register directives for configuration

   Register("port", new XrdProofdDirective("port", this, &DoDirectiveClass));
   Register("xrd.protocol", new XrdProofdDirective("xrd.protocol", this, &DoDirectiveClass));
}

//______________________________________________________________________________
int XrdProofdProtCfg::DoDirective(XrdProofdDirective *d,
                                  char *val, XrdOucStream *cfg, bool)
{
   // Parse directives

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


extern "C" {
//_________________________________________________________________________________
XrdProtocol *XrdgetProtocol(const char *, char *parms, XrdProtocol_Config *pi)
{
   // This protocol is meant to live in a shared library. The interface below is
   // used by the server to obtain a copy of the protocol object that can be used
   // to decide whether or not a link is talking a particular protocol.

   // Return the protocol object to be used if static init succeeds
   if (XrdProofdProtocol::Configure(parms, pi)) {

      return (XrdProtocol *) new XrdProofdProtocol(pi);
   }
   return (XrdProtocol *)0;
}

//_________________________________________________________________________________
int XrdgetProtocolPort(const char * /*pname*/, char * /*parms*/, XrdProtocol_Config *pi)
{
      // This function is called early on to determine the port we need to use. The
      // The default is ostensibly 1093 but can be overidden; which we allow.

      XrdProofdProtCfg pcfg(pi->ConfigFN, pi->eDest);
      // Init some relevant quantities for tracing
      XrdProofdTrace = new XrdOucTrace(pi->eDest);
      pcfg.Config(0);

      // Default XPD_DEF_PORT (1093)
      int port = XPD_DEF_PORT;

      if (pcfg.fPort > 0) {
         port = pcfg.fPort;
      } else {
         port = (pi && pi->Port > 0) ? pi->Port : XPD_DEF_PORT;
      }

      return port;
}}

//__________________________________________________________________________________
XrdProofdProtocol::XrdProofdProtocol(XrdProtocol_Config *pi)
   : XrdProtocol("xproofd protocol handler"), fProtLink(this)
{
   // Protocol constructor
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

//______________________________________________________________________________
XrdProofdResponse *XrdProofdProtocol::Response(kXR_unt16 sid)
{
   // Get response instance corresponding to stream ID 'sid'
   XPDLOC(ALL, "Protocol::Response")

   TRACE(HDBG, "sid: "<<sid<<", size: "<<fResponses.size());

   if (sid > 0)
      if (sid <= fResponses.size())
         return fResponses[sid-1];

   return (XrdProofdResponse *)0;
}

//______________________________________________________________________________
XrdProofdResponse *XrdProofdProtocol::GetNewResponse(kXR_unt16 sid)
{
   // Create new response instance for stream ID 'sid'
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

//______________________________________________________________________________
XrdProtocol *XrdProofdProtocol::Match(XrdLink *lp)
{
   // Check whether the request matches this protocol
   XPDLOC(ALL, "Protocol::Match")

   struct ClientInitHandShake hsdata;
   char  *hsbuff = (char *)&hsdata;

   static hs_response_t hsresp = {0, 0, htonl(XPROOFD_VERSBIN), 0};

   XrdProofdProtocol *xp;
   int dlen;
   TRACE(HDBG, "enter");

   XrdOucString emsg;
   // Peek at the first 20 bytes of data
   if ((dlen = lp->Peek(hsbuff,sizeof(hsdata),fgReadWait)) != sizeof(hsdata)) {
      if (dlen <= 0) lp->setEtext("Match: handshake not received");
      if (dlen == 12) {
         // Check if it is a request to open a file via 'rootd'
         hsdata.first = ntohl(hsdata.first);
         if (hsdata.first == 8) {
            if (strlen(fgMgr->RootdExe()) > 0) {
               if (fgMgr->IsRootdAllowed((const char *)lp->Host())) {
                  TRACE(ALL, "matched rootd protocol on link: executing "<<fgMgr->RootdExe());
                  XrdOucString em;
                  if (StartRootd(lp, em) != 0) {
                     emsg = "rootd: failed to start daemon: ";
                     emsg += em;
                  }
               } else {
                  XPDFORM(emsg, "rootd-file serving not authorized for host '%s'", lp->Host());
               }
            } else {
               emsg = "rootd-file serving not enabled";
            }
         }
         if (emsg.length() > 0) {
            lp->setEtext(emsg.c_str());
         } else {
            lp->setEtext("link transfered");
         }
         return (XrdProtocol *)0;
      }
      TRACE(XERR, "peeked incomplete or empty information! (dlen: "<<dlen<<" bytes)");
      return (XrdProtocol *)0;
   }

   // Verify that this is our protocol
   hsdata.third  = ntohl(hsdata.third);
   if (dlen != sizeof(hsdata) ||  hsdata.first || hsdata.second
       || !(hsdata.third == 1) || hsdata.fourth || hsdata.fifth) return 0;

   // Respond to this request with the handshake response
   if (!lp->Send((char *)&hsresp, sizeof(hsresp))) {
      lp->setEtext("Match: handshake failed");
      TRACE(XERR, "handshake failed");
      return (XrdProtocol *)0;
   }

   // We can now read all 20 bytes and discard them (no need to wait for it)
   int len = sizeof(hsdata);
   if (lp->Recv(hsbuff, len) != len) {
      lp->setEtext("Match: reread failed");
      TRACE(XERR, "reread failed");
      return (XrdProtocol *)0;
   }

   // Get a protocol object off the stack (if none, allocate a new one)
   if (!(xp = fgProtStack.Pop()))
      xp = new XrdProofdProtocol();

   // Bind the protocol to the link and return the protocol
   xp->fLink = lp;
   strcpy(xp->fSecEntity.prot, "host");
   xp->fSecEntity.host = strdup((char *)lp->Host());

   // Dummy data used by 'proofd'
   kXR_int32 dum[2];
   if (xp->GetData("dummy",(char *)&dum[0],sizeof(dum)) != 0) {
      xp->Recycle(0,0,0);
      return (XrdProtocol *)0;
   }

   // We are done
   return (XrdProtocol *)xp;
}

//_____________________________________________________________________________
int XrdProofdProtocol::StartRootd(XrdLink *lp, XrdOucString &emsg)
{
   // Transfer the connection to a rootd daemon to serve a file access request
   // Return 0 on success, -1 on failure
   XPDLOC(ALL, "Protocol::StartRootd")

   const char *prog = fgMgr->RootdExe();
   const char **progArg = fgMgr->RootdArgs();

   if (fgMgr->RootdFork()) {

      // Start rootd using fork()
      
      pid_t pid;
      if ((pid = fgMgr->Sched()->Fork(lp->Name()))) {
         if (pid < 0) {
            emsg = "rootd fork failed";
            return -1;
         }
         return 0;
      }
      // In the child ...
      
      // Restablish standard error for the program we will exec
      dup2(fStdErrFD, STDERR_FILENO);
      close(fStdErrFD);

      // Force stdin/out to point to the socket FD (this will also bypass the
      // close on exec setting for the socket)
      dup2(lp->FDnum(), STDIN_FILENO);
      dup2(lp->FDnum(), STDOUT_FILENO);

      // Do the exec
      execv((const char *)prog, (char * const *)progArg);
      TRACE(XERR, "rootd: Oops! Exec(" <<prog <<") failed; errno: " <<errno);
      _exit(17);

   } else {
      
      // Start rootd using system + proofexecv
   
      // ROOT version
      XrdROOT *roo = fgMgr->ROOTMgr()->DefaultVersion();
      if (!roo) {
         emsg = "ROOT version undefined!";
         return -1;
      }
      // The path to the executable
      XrdOucString pexe;
      XPDFORM(pexe, "%s/proofexecv", roo->BinDir());
      if (access(pexe.c_str(), X_OK) != 0) {
         XPDFORM(emsg, "path '%s' does not exist or is not executable (errno: %d)",
                     pexe.c_str(), (int)errno);
         return -1;
      }

      // Start the proofexecv
      XrdOucString cmd, exp;
      XPDFORM(cmd, "export ROOTBINDIR=\"%s\"; %s 20 0 %s %s", roo->BinDir(),
                  pexe.c_str(), fgMgr->RootdUnixSrv()->path(), prog);
      int n = 1;
      while (progArg[n] != 0) {
         cmd += " "; cmd += progArg[n]; n++;
      }
      cmd += " &";
      TRACE(HDBG, cmd);
      if (system(cmd.c_str()) == -1) {
         XPDFORM(emsg, "failure from 'system' (errno: %d)", (int)errno);
         return -1;
      }

      // Accept a connection from the second server
      int err;
      rpdunix *uconn = fgMgr->RootdUnixSrv()->accept(-1, &err);
      if (!uconn || !uconn->isvalid(0)) {
         XPDFORM(emsg, "failure accepting callback (errno: %d)", -err);
         if (uconn) delete uconn;
         return -1;
      }
      TRACE(HDBG, "proofexecv connected!");

      int rcc = 0;
      // Transfer the open descriptor to be used in rootd
      int fd = dup(lp->FDnum());
      if (fd < 0 || (rcc = uconn->senddesc(fd)) != 0) {
         XPDFORM(emsg, "failure sending descriptor '%d' (original: %d); (errno: %d)", fd, lp->FDnum(), -rcc);
         if (uconn) delete uconn;
         return -1;
      }
      // Close the connection to the parent
      delete uconn;
   }

   // Done
   return 0;
}
   
//_____________________________________________________________________________
int XrdProofdProtocol::Stats(char *buff, int blen, int)
{
   // Return statistics info about the protocol.
   // Not really implemented yet: this is a reduced XrdXrootd version.

   static char statfmt[] = "<stats id=\"xproofd\"><num>%ld</num></stats>";

   // If caller wants only size, give it to him
   if (!buff)
      return sizeof(statfmt)+16;

   // We have only one statistic -- number of successful matches
   return snprintf(buff, blen, statfmt, fgCount);
}

//______________________________________________________________________________
void XrdProofdProtocol::Reset()
{
   // Reset static and local vars

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
   memset(&fSecEntity, 0, sizeof(fSecEntity));
   // Cleanup existing XrdProofdResponse objects
   std::vector<XrdProofdResponse *>::iterator ii = fResponses.begin(); // One per each logical connection
   while (ii != fResponses.end()) {
      delete *ii;
      ii++;
   }
   fResponses.clear();
}

//______________________________________________________________________________
int XrdProofdProtocol::Configure(char *, XrdProtocol_Config *pi)
{
   // Protocol configuration tool
   // Function: Establish configuration at load time.
   // Output: 1 upon success or 0 otherwise.
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
   fgProtStack.Set(pi->Sched, XrdProofdTrace, TRACE_MEM);
   fgProtStack.Set((pi->ConnMax/3 ? pi->ConnMax/3 : 30), 60*60);

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
   fgMgr = new XrdProofdManager(pi, &fgEDest);
   if (fgMgr->Config(0)) return 0;
   mp = "global manager created";
   TRACE(ALL, mp);

   // Issue herald indicating we configured successfully
   TRACE(ALL, "xproofd protocol version "<<XPROOFD_VERSION<<
              " build "<<XrdVERSION<<" successfully loaded");

   // Return success
   return 1;
}

//______________________________________________________________________________
int XrdProofdProtocol::Process(XrdLink *)
{
   // Process the information received on the active link.
   // (We ignore the argument here)
   XPDLOC(ALL, "Protocol::Process")

   int rc = 0;
   TRACEP(this, DBG, "instance: " << this);

   // Read the next request header
   if ((rc = GetData("request", (char *)&fRequest, sizeof(fRequest))) != 0)
      return rc;
   TRACEP(this, HDBG, "after GetData: rc: " << rc);

   // Deserialize the data
   fRequest.header.requestid = ntohs(fRequest.header.requestid);
   fRequest.header.dlen      = ntohl(fRequest.header.dlen);

   // Get response object
   kXR_unt16 sid;
   memcpy((void *)&sid, (const void *)&(fRequest.header.streamid[0]), 2);
   XrdProofdResponse *response = 0;
   if (!(response = Response(sid))) {
      if (!(response = GetNewResponse(sid))) {
         TRACEP(this, XERR, "could not get Response instance for rid: "<< sid);
         return rc;
      }
   }
   // Set the stream ID for the reply
   response->Set(fRequest.header.streamid);
   response->Set(fLink);

   TRACEP(this, REQ, "sid: " << sid << ", req id: " << fRequest.header.requestid <<
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

//______________________________________________________________________________
int XrdProofdProtocol::Process2()
{
   // Local processing method: here the request is dispatched to the appropriate
   // method
   XPDLOC(ALL, "Protocol::Process2")

   int rc = 0;
   XPD_SETRESP(this, "Process2");

   TRACEP(this, REQ, "req id: " << fRequest.header.requestid << " (" <<
                XrdProofdAux::ProofRequestTypes(fRequest.header.requestid) << ")");

   ResetCtrlcGuard_t ctrlcguard(this, fRequest.header.requestid);

   // If the user is logged in check if the wanted action is to be done by us
   if (fStatus && (fStatus & XPD_LOGGEDIN)) {
      // Record time of the last action
      TouchAdminPath();
      // We must have a client instance if here
      if (!fPClient) {
         TRACEP(this, XERR, "client undefined!!! ");
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

//______________________________________________________________________
void XrdProofdProtocol::Recycle(XrdLink *, int, const char *)
{
   // Recycle call. Release the instance and give it back to the stack.
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
   TRACEP(this, LOGIN, buf);

   // If we have a buffer, release it
   if (fArgp) {
      fgBPool->Release(fArgp);
      fArgp = 0;
   }

   // Locate the client instance
   XrdProofdClient *pmgr = fPClient;

   if (pmgr) {


      if (!Internal()) {

         // Signal the client manager that a client has just gone
         if (fgMgr && fgMgr->ClientMgr()) {
            TRACE(HDBG, "fAdminPath: "<<fAdminPath);
            XPDFORM(buf, "%s %p %d %d", fAdminPath.c_str(), pmgr, fCID, fPid);
            TRACE(DBG, "sending to ClientMgr: "<<buf);
            fgMgr->ClientMgr()->Pipe()->Post(XrdProofdClientMgr::kClientDisconnect, buf.c_str());
         }

      } else {

         // Internal connection: we need to remove this instance from the list
         // of proxy servers and to notify the attached clients.
         // Tell the session manager that this session has gone
         if (fgMgr && fgMgr->SessionMgr()) {
            TRACE(HDBG, "fAdminPath: "<<fAdminPath);
            buf.assign(fAdminPath, fAdminPath.rfind('/') + 1, -1);
            TRACE(DBG, "sending to ProofServMgr: "<<buf);
            fgMgr->SessionMgr()->Pipe()->Post(XrdProofdProofServMgr::kSessionRemoval, buf.c_str());
         }
      }
   }

   // Set fields to starting point (debugging mostly)
   Reset();

   // Push ourselves on the stack
   fgProtStack.Push(&fProtLink);
}

//______________________________________________________________________________
XrdBuffer *XrdProofdProtocol::GetBuff(int quantum, XrdBuffer *argp)
{
   // Allocate a buffer to handle quantum bytes; if argp points to an existing
   // buffer, its size is checked and re-allocated if needed
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

//______________________________________________________________________________
void XrdProofdProtocol::ReleaseBuff(XrdBuffer *argp)
{
   // Release a buffer previously allocated via GetBuff

   XrdSysMutexHelper mh(fgBMutex);
   fgBPool->Release(argp);
}

//______________________________________________________________________________
int XrdProofdProtocol::GetData(const char *dtype, char *buff, int blen)
{
   // Get data from the open link
   XPDLOC(ALL, "Protocol::GetData")

   int rlen;

   // Read the data but reschedule the link if we have not received all of the
   // data within the timeout interval.
   TRACEP(this, HDBG, "dtype: "<<(dtype ? dtype : " - ")<<", blen: "<<blen);

   // No need to lock:the link is disable while we are here
   rlen = fLink->Recv(buff, blen, fgReadWait);
   if (rlen  < 0) {
      if (rlen != -ENOMSG && rlen != -ECONNRESET) {
         XrdOucString emsg = "link read error: errno: ";
         emsg += -rlen;
         TRACEP(this, XERR, emsg.c_str());
         return (fLink ? fLink->setEtext(emsg.c_str()) : -1);
      } else {
         TRACEP(this, HDBG, "connection closed by peer (errno: "<<-rlen<<")");
         return -1;
      }
   }
   if (rlen < blen) {
      TRACEP(this, DBG, dtype << " timeout; read " <<rlen <<" of " <<blen <<" bytes - rescheduling");
      return 1;
   }
   TRACEP(this, HDBG, "rlen: "<<rlen);

   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::SendData(XrdProofdProofServ *xps,
                                kXR_int32 sid, XrdSrvBuffer **buf, bool savebuf)
{
   // Send data over the open link. Segmentation is done here, if required.
   XPDLOC(ALL, "Protocol::SendData")

   int rc = 0;

   TRACEP(this, HDBG, "length: "<<fRequest.header.dlen<<" bytes ");

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
            TRACEP(this, XERR, msg);
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
            TRACEP(this, XERR, msg);
            return -1;
         }
      }
      TRACEP(this, HDBG, msg);
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

//______________________________________________________________________________
int XrdProofdProtocol::SendDataN(XrdProofdProofServ *xps,
                                 XrdSrvBuffer **buf, bool savebuf)
{
   // Send data over the open client links of session 'xps'.
   // Used when all the connected clients are eligible to receive the message.
   // Segmentation is done here, if required.
   XPDLOC(ALL, "Protocol::SendDataN")

   int rc = 0;

   TRACEP(this, HDBG, "length: "<<fRequest.header.dlen<<" bytes ");

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

//_____________________________________________________________________________
int XrdProofdProtocol::SendMsg()
{
   // Handle a request to forward a message to another process
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
      TRACEP(this, XERR, msg.c_str());
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
         TRACEP(this, HDBG, msg.c_str());
      }

      // Send to proofsrv our client ID
      if (fCID == -1) {
         TRACEP(this, REQ, "EXT: error getting clientSID");
         response->Send(kXP_ServerError,"EXT: getting clientSID");
         return 0;
      }
      if (SendData(xps, fCID)) {
         TRACEP(this, REQ, "EXT: error sending message to proofserv");
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
          TRACEP(this, HDBG, msg.c_str());
      }
      bool saveStartMsg = 0;
      XrdSrvBuffer *savedBuf = 0;
      // Additional info about the message
      if (opt & kXPD_setidle) {
         TRACEP(this, DBG, "INT: setting proofserv in 'idle' state");
         xps->SetStatus(kXPD_idle);
         PostSession(-1, fPClient->UI().fUser.c_str(),
                         fPClient->UI().fGroup.c_str(), xps);
      } else if (opt & kXPD_querynum) {
         TRACEP(this, DBG, "INT: got message with query number");
      } else if (opt & kXPD_startprocess) {
         TRACEP(this, DBG, "INT: setting proofserv in 'running' state");
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
            TRACEP(this, DBG, "INT: broadcasting log message");
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
         TRACEP(this, DBG, msg);
      }
      // Notify to proofsrv
      response->Send();
   }

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Urgent()
{
   // Handle generic request of a urgent message to be forwarded to the server
   XPDLOC(ALL, "Protocol::Urgent")

   unsigned int rc = 0;

   XPD_SETRESP(this, "Urgent");

   // Unmarshall the data
   int psid = ntohl(fRequest.proof.sid);
   int type = ntohl(fRequest.proof.int1);
   int int1 = ntohl(fRequest.proof.int2);
   int int2 = ntohl(fRequest.proof.int3);

   TRACEP(this, REQ, "psid: "<<psid<<", type: "<< type);

   // Find server session
   XrdProofdProofServ *xps = 0;
   if (!fPClient || !(xps = fPClient->GetServer(psid))) {
      TRACEP(this, XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"Urgent: session ID not found");
      return 0;
   }

   TRACEP(this, DBG, "xps: "<<xps<<", status: "<<xps->Status());

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
   TRACEP(this, DBG, "request propagated to proofsrv");

   // Over
   return 0;
}

//___________________________________________________________________________
int XrdProofdProtocol::Interrupt()
{
   // Handle an interrupt request
   XPDLOC(ALL, "Protocol::Interrupt")

   int rc = 0;

   XPD_SETRESP(this, "Interrupt");

   // Unmarshall the data
   int psid = ntohl(fRequest.interrupt.sid);
   int type = ntohl(fRequest.interrupt.type);
   TRACEP(this, REQ, "psid: "<<psid<<", type:"<<type);

   // Find server session
   XrdProofdProofServ *xps = 0;
   if (!fPClient || !(xps = fPClient->GetServer(psid))) {
      TRACEP(this, XERR, "session ID not found: "<<psid);
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
      TRACEP(this, DBG, msg.c_str());

      // Propagate the type as unsolicited
      if (xps->Response()->Send(kXR_attn, kXPD_interrupt, type) != 0) {
         response->Send(kXP_ServerError,
                        "Interrupt: could not propagate interrupt code to proofsrv");
         return 0;
      }

      // Notify to user
      response->Send();
      TRACEP(this, DBG, "interrupt propagated to proofsrv");
   }

   // Over
   return 0;
}

//___________________________________________________________________________
int XrdProofdProtocol::Ping()
{
   // Handle a ping request.
   // For internal connections, ping is done asynchronously to avoid locking
   // problems; the session checker verifies that the admin file has been touched
   // recently enough; touching is done in Process2, so we have nothing to do here
   XPDLOC(ALL, "Protocol::Ping")

   int rc = 0;
   if (Internal()) {
      if (TRACING(HDBG)) {
         XPD_SETRESP(this, "Ping");
         TRACEP(this,  HDBG, "INT: nothing to do ");
      }
      return 0;
   }
   XPD_SETRESP(this, "Ping");

   // Unmarshall the data
   int psid = ntohl(fRequest.sendrcv.sid);
   int asyncopt = ntohl(fRequest.sendrcv.opt);

   TRACEP(this, REQ, "psid: "<<psid<<", async: "<<asyncopt);

   // For connections to servers find the server session; manager connections
   // (psid == -1) do not have any session attached 
   XrdProofdProofServ *xps = 0;
   if (!fPClient || (psid > -1 && !(xps = fPClient->GetServer(psid)))) {
      TRACEP(this,  XERR, "session ID not found: "<<psid);
      response->Send(kXR_InvalidRequest,"session ID not found");
      return 0;
   }

   // For manager connections we are done
   kXR_int32 pingres = (psid > -1) ? 0 : 1;
   if (psid > -1 && xps && xps->IsValid()) {

      TRACEP(this,  DBG, "EXT: psid: "<<psid);

      // This is the max time we will privide an answer
      kXR_int32 checkfq = fgMgr->SessionMgr()->CheckFrequency();

      // If asynchronous return the timeout for an answer
      if (asyncopt == 1) {
         TRACEP(this, DBG, "EXT: async: notifying timeout to client: "<<checkfq<<" secs");
         response->Send(kXR_ok, checkfq);
      }

      // Admin path
      XrdOucString path(xps->AdminPath());
      if (path.length() <= 0) {
         TRACEP(this,  XERR, "EXT: admin path is empty! - protocol error");
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
         TRACEP(this,  XERR, "EXT: cannot stat admin path: "<<path);
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
               TRACEP(this,  XERR, "EXT: could not send verify request to proofsrv");
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
               TRACEP(this, DBG, "EXT: waiting "<<ns<<" secs for session "<<pid<<
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
      TRACEP(this, DBG, "EXT: notified the result to client: "<<pingres);
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
      TRACEP(this, XERR, "session ID not found: "<<psid);
   }

   // Send the result
   response->Send(kXR_ok, pingres);

   // Done
   return 0;
}

//___________________________________________________________________________
void XrdProofdProtocol::PostSession(int on, const char *u, const char *g,
                                    XrdProofdProofServ *xps)
{
   // Post change of session status
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

//___________________________________________________________________________
void XrdProofdProtocol::TouchAdminPath()
{
   // Recording time of the last request on this instance
   XPDLOC(ALL, "Protocol::TouchAdminPath")

   XPD_SETRESPV(this, "TouchAdminPath");
   TRACEP(this, HDBG, fAdminPath);

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
         if (rc != 0) {
            const char *type = Internal() ? "internal" : "external";
            TRACEP(this, XERR, type<<": problems touching "<<apath<<"; errno: "<<-rc);
         }
      }
   }
   // Done
   return;
}

//______________________________________________________________________________
int XrdProofdProtocol::CtrlC()
{
   // Set and propagate a Ctrl-C request
   XPDLOC(ALL, "Protocol::CtrlC")

   TRACEP(this, ALL, "handling request");

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
