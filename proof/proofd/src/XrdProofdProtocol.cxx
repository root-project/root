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
#  include "XrdOuc/XrdOucTimer.hh"
#  define XPD_LOG_01 OUC_LOG_01
#else
#  include "XrdSys/XrdSysError.hh"
#  include "XrdSys/XrdSysLogger.hh"
#  include "XrdSys/XrdSysTimer.hh"
#  define XPD_LOG_01 SYS_LOG_01
#endif

#include "XrdVersion.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdSys/XrdSysPriv.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdOuc/XrdOucReqID.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSut/XrdSutAux.hh"
#include "XrdNet/XrdNet.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "Xrd/XrdLink.hh"
#include "Xrd/XrdPoll.hh"
#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdScheduler.hh"

#include "XrdProofConn.h"
#include "XrdProofdClient.h"
#include "XrdProofdProtocol.h"
#include "XrdProofSched.h"
#include "XrdProofServProxy.h"
#include "XrdProofWorker.h"
#include "XrdROOT.h"

#include "RConfigure.h"

// Tracing utils
#include "XrdProofdTrace.h"
XrdOucTrace          *XrdProofdTrace = 0;
static const char    *gTraceID = " ";

// Static variables
static XrdOucReqID   *XrdProofdReqID = 0;

// Loggers: we need two to avoid deadlocks
static XrdSysLogger   gMainLogger;

//
// Static area: general protocol managing section
int                   XrdProofdProtocol::fgCount    = 0;
XrdSysRecMutex        XrdProofdProtocol::fgXPDMutex;
XrdObjectQ<XrdProofdProtocol>
                      XrdProofdProtocol::fgProtStack("ProtStack",
                                                     "xproofd protocol anchor");
XrdBuffManager       *XrdProofdProtocol::fgBPool    = 0;
int                   XrdProofdProtocol::fgMaxBuffsz= 0;
XrdScheduler         *XrdProofdProtocol::fgSched    = 0;
XrdSysError           XrdProofdProtocol::fgEDest(0, "xpd");
XrdSysLogger          XrdProofdProtocol::fgMainLogger;

//
// Static area: protocol configuration section
XrdProofdFile         XrdProofdProtocol::fgCfgFile;
bool                  XrdProofdProtocol::fgConfigDone = 0;
//
XrdSysSemWait         XrdProofdProtocol::fgForkSem;   // To serialize fork requests
//
int                   XrdProofdProtocol::fgReadWait = 0;
int                   XrdProofdProtocol::fgInternalWait = 30; // seconds
// Proofserv configuration
XrdOucString          XrdProofdProtocol::fgProofServEnvs; // Additional envs to be exported before proofserv
XrdOucString          XrdProofdProtocol::fgProofServRCs; // Additional rcs to be passed to proofserv
// Cluster manager
XrdProofdManager      XrdProofdProtocol::fgMgr;
// COnfig directives
XrdOucHash<XrdProofdDirective> XrdProofdProtocol::fgConfigDirectives; // Config directives
XrdOucHash<XrdProofdDirective> XrdProofdProtocol::fgReConfigDirectives; // Re-configurable directives

// Local definitions
#define MAX_ARGS 128
#define TRACEID gTraceID

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

//______________________________________________________________________________
int DoProtocolDirective(XrdProofdDirective *d,
                        char *val, XrdOucStream *cfg, bool rcf)
{
   // Generic directive processor

   if (!d || !val)
      // undefined inputs
      return -1;

   return XrdProofdProtocol::ProcessDirective(d, val, cfg, rcf);
}

//--------------------------------------------------------------------------
//
// XrdProofdCron
//
// Function run in separate thread to run periodic checks, ... at a tunable
// frequency
//
//--------------------------------------------------------------------------
void *XrdProofdCron(void *p)
{
   // This is an endless loop to periodically check the system

   XrdProofdManager *mgr = (XrdProofdManager *)p;
   if (!(mgr)) {
      TRACE(REQ, "XrdProofdCron: undefined manager: cannot start");
      return (void *)0;
   }

   TRACE(REQ, "XrdProofdCron: started with frequency "<<mgr->CronFrequency()<<" sec");

   while(1) {
      // Wait a while
      XrdSysTimer::Wait(mgr->CronFrequency() * 1000);
      // Do something here
      TRACE(REQ, "XrdProofdCron: running periodical checks");
      // Reconfig
      XrdProofdProtocol::Reconfig();
   }

   // Should never come here
   return (void *)0;
}

//__________________________________________________________________________________
void XrdProofdProtocol::Reconfig()
{
   // Reconfig

   XrdSysMutexHelper mtxh(fgMgr.Mutex());
   // Trim the list of processes asked for termination
   fgMgr.TrimTerminatedProcesses();
   // Reconfigure the manager
   fgMgr.Config(0, 1);
   // Reconfigure protocol
   Config(0, 1);
   // Broadcast updated priorities to the active sessions
   fgMgr.UpdatePriorities();
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

      // Issue herald
      char msg[256];
      sprintf(msg,"xproofd: protocol V %s successfully loaded", XPROOFD_VERSION);
      pi->eDest->Say(0, msg);

      return (XrdProtocol *) new XrdProofdProtocol();
   }
   return (XrdProtocol *)0;
}

//_________________________________________________________________________________
int XrdgetProtocolPort(const char * /*pname*/, char * /*parms*/, XrdProtocol_Config *pi)
{
      // This function is called early on to determine the port we need to use. The
      // The default is ostensibly 1093 but can be overidden; which we allow.

      // Default XPD_DEF_PORT (1093)
      int port = (pi && pi->Port > 0) ? pi->Port : XPD_DEF_PORT;
      return port;
}}

//__________________________________________________________________________________
XrdProofdProtocol::XrdProofdProtocol()
   : XrdProtocol("xproofd protocol handler"), fProtLink(this)
{
   // Protocol constructor
   fLink = 0;
   fArgp = 0;
   fClientID = 0;
   fGroupID = 0;
   fPClient = 0;
   fClient = 0;
   fAuthProt = 0;
   fBuff = 0;

   // Instantiate a Proofd protocol object
   Reset();
}

//______________________________________________________________________________
XrdProtocol *XrdProofdProtocol::Match(XrdLink *lp)
{
   // Check whether the request matches this protocol

   struct ClientInitHandShake hsdata;
   char  *hsbuff = (char *)&hsdata;

   static hs_response_t hsresp = {0, 0, htonl(XPROOFD_VERSBIN), 0};

   XrdProofdProtocol *xp;
   int dlen;

   // Peek at the first 20 bytes of data
   if ((dlen = lp->Peek(hsbuff,sizeof(hsdata),fgReadWait)) != sizeof(hsdata)) {
      if (dlen <= 0) lp->setEtext("Match: handshake not received");
      return (XrdProtocol *)0;
   }

   // Verify that this is our protocol
   hsdata.third  = ntohl(hsdata.third);
   if (dlen != sizeof(hsdata) ||  hsdata.first || hsdata.second
       || !(hsdata.third == 1) || hsdata.fourth || hsdata.fifth) return 0;

   // Respond to this request with the handshake response
   if (!lp->Send((char *)&hsresp, sizeof(hsresp))) {
      lp->setEtext("Match: handshake failed");
      return (XrdProtocol *)0;
   }

   // We can now read all 20 bytes and discard them (no need to wait for it)
   int len = sizeof(hsdata);
   if (lp->Recv(hsbuff, len) != len) {
      lp->setEtext("Match: reread failed");
      return (XrdProtocol *)0;
   }

   // Get a protocol object off the stack (if none, allocate a new one)
   if (!(xp = fgProtStack.Pop()))
      xp = new XrdProofdProtocol();

   // Bind the protocol to the link and return the protocol
   xp->fLink = lp;
   strcpy(xp->fEntity.prot, "host");
   xp->fEntity.host = strdup((char *)lp->Host());

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
   fArgp      = 0;
   fStatus    = 0;
   SafeDelArray(fClientID);
   SafeDelArray(fGroupID);
   fUI.Reset();
   fCapVer    = 0;
   fSrvType   = kXPD_TopMaster;
   fTopClient = 0;
   fSuperUser = 0;
   fPClient   = 0;
   fCID       = -1;
   fClient    = 0;
   SafeDelete(fClient);
   if (fAuthProt) {
      fAuthProt->Delete();
      fAuthProt = 0;
   }
   memset(&fEntity, 0, sizeof(fEntity));
   fTopClient = 0;
   fSuperUser = 0;
   fBuff      = 0;
   fBlen      = 0;
   fBlast     = 0;
   // Magic numbers cut & pasted from Xrootd
   fhcPrev    = 13;
   fhcMax     = 28657;
   fhcNext    = 21;
   fhcNow     = 13;
   fhalfBSize = 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Configure(char *, XrdProtocol_Config *pi)
{
   // Protocol configuration tool
   // Function: Establish configuration at load time.
   // Output: 1 upon success or 0 otherwise.

   XrdOucString mp;

   // Only once
   if (fgConfigDone)
      return 1;
   fgConfigDone = 1;

   // Copy out the special info we want to use at top level
   fgEDest.logger(&fgMainLogger);
   XrdProofdTrace = new XrdOucTrace(&fgEDest);
   fgSched        = pi->Sched;
   fgBPool        = pi->BPool;
   fgReadWait     = pi->readWait;

   // Debug flag
   TRACESET(XERR, 1);
   if (pi->DebugON)
      XrdProofdTrace->What |= (TRACE_REQ | TRACE_LOGIN | TRACE_FORK | TRACE_DBG);

   // Process the config file for directives meaningful to us
   if (pi->ConfigFN) {
      // Register (re-)config directives 
      RegisterConfigDirectives();
      RegisterReConfigDirectives();
      // Save path for re-configuration checks
      fgCfgFile.fName = pi->ConfigFN;
      XrdProofdAux::Expand(fgCfgFile.fName);
      // Configure tracing
      if (TraceConfig(fgCfgFile.fName.c_str()))
         return 0;
      // Configure the manager
      if (fgMgr.Config(fgCfgFile.fName.c_str(), 0, &fgEDest))
         return 0;
      // Configure the protocol
      if (Config(fgCfgFile.fName.c_str(), 0))
         return 0;
   }

   char msgs[256];
   sprintf(msgs,"Proofd : Configure: mgr: %p", &fgMgr);
   fgEDest.Say(0, msgs);

   // Notify port
   mp = "Proofd : Configure: listening on port ";
   mp += fgMgr.Port();
   fgEDest.Say(0, mp.c_str());

   // Pre-initialize some i/o values
   fgMaxBuffsz = fgBPool->MaxSize();

   // Notify timeout on internal communications
   mp = "Proofd : Configure: setting internal timeout to (secs): ";
   mp += fgInternalWait;
   fgEDest.Say(0, mp.c_str());

   // Notify temporary directory
   fgEDest.Say(0, "Proofd : Configure: using temp dir: ", fgMgr.TMPdir());

   // Initialize the security system if this is wanted
   if (!fgMgr.CIA())
      fgEDest.Say(0, "XRD seclib not specified; strong authentication disabled");

   // Notify role
   const char *roles[] = { "any", "worker", "submaster", "master" };
   fgEDest.Say(0, "Proofd : Configure: role set to: ", roles[fgMgr.SrvType()+1]);

   // Schedule protocol object cleanup; the maximum number of objects
   // and the max age are taken from XrdXrootdProtocol: this may need
   // some optimization in the future.
   fgProtStack.Set(pi->Sched, XrdProofdTrace, TRACE_MEM);
   fgProtStack.Set((pi->ConnMax/3 ? pi->ConnMax/3 : 30), 60*60);

   // Initialize the request ID generation object
   XrdProofdReqID = new XrdOucReqID((int)fgMgr.Port(), pi->myName,
                                    XrdNetDNS::IPAddr(pi->myAddr));

   // Start cron thread, if required
   if (fgMgr.Cron() == 1) {
      pthread_t tid;
      if (XrdSysThread::Run(&tid, XrdProofdCron,
                            (void *)&fgMgr, 0, "Proof cron thread") != 0) {
         fgEDest.Say(0, "Proofd : Configure: could not start cron thread");
         return 0;
      }
      fgEDest.Say(0, "Proofd : Configure: cron thread started");
   }

   // Indicate we configured successfully
   fgEDest.Say(0, "XProofd protocol version " XPROOFD_VERSION
               " build " XrdVERSION " successfully loaded.");

   // Return success
   return 1;
}

//______________________________________________________________________________
int XrdProofdProtocol::TraceConfig(const char *cfn)
{
   // Scan the config file for tracing settings

   TRACE(ACT, "TraceConfig: enter: file: " <<cfn);

   XrdOucStream cfg(&fgEDest, getenv("XRDINSTANCE"));

   // Open and attach the config file
   int cfgFD;
   if ((cfgFD = open(cfn, O_RDONLY, 0)) < 0)
      return fgEDest.Emsg("Config", errno, "open config file", cfn);
   cfg.Attach(cfgFD);

   // Process items
   char *val = 0, *var = 0;
   while ((var = cfg.GetMyFirstWord())) {
      if (!(strncmp("xpd.trace", var, 9))) {
         // Get the value
         val = cfg.GetToken();
         if (val && val[0]) {
            // Specifies tracing options. Valid keywords are:
            //   req            trace protocol requests             [on]*
            //   login          trace details about login requests  [on]*
            //   act            trace internal actions              [off]
            //   rsp            trace server replies                [off]
            //   fork           trace proofserv forks               [on]*
            //   dbg            trace details about actions         [off]
            //   hdbg           trace more details about actions    [off]
            //   err            trace errors                        [on]
            //   sched          trace details about scheduling      [off]
            //   admin          trace admin requests                [on]*
            //   all            trace everything
            //
            // Defaults are shown in brackets; '*' shows the default when the '-d'
            // option is passed on the command line. Each option may be
            // optionally prefixed by a minus sign to turn off the setting.
            // Order matters: 'all' in last position enables everything; in first
            // position is corrected by subsequent settings
            //
            while (val && val[0]) {
               bool on = 1;
               if (val[0] == '-') {
                  on = 0;
                  val++;
               }
               if (!strcmp(val,"req")) {
                  TRACESET(REQ, on);
               } else if (!strcmp(val,"login")) {
                  TRACESET(LOGIN, on);
               } else if (!strcmp(val,"act")) {
                  TRACESET(ACT, on);
               } else if (!strcmp(val,"rsp")) {
                  TRACESET(RSP, on);
               } else if (!strcmp(val,"fork")) {
                  TRACESET(FORK, on);
               } else if (!strcmp(val,"dbg")) {
                  TRACESET(DBG, on);
               } else if (!strcmp(val,"hdbg")) {
                  TRACESET(HDBG, on);
               } else if (!strcmp(val,"err")) {
                  TRACESET(XERR, on);
               } else if (!strcmp(val,"sched")) {
                  TRACESET(SCHED, on);
               } else if (!strcmp(val,"admin")) {
                  TRACESET(ADMIN, on);
               } else if (!strcmp(val,"all")) {
                  // Everything
                  XrdProofdTrace->What = (on) ? TRACE_ALL : 0;
               }
               // Next
               val = cfg.GetToken();
            }
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Config(const char *cfn, bool rcf)
{
   // Scan the config file

   TRACE(ACT, "Config: enter: file: " << (cfn ? cfn : ((rcf) ? "unchanged" : "undef")));

   if (fgCfgFile.fName.length() <= 0 && (!cfn || strlen(cfn) <= 0)) {
      // Done
      TRACE(XERR, "Config: no config file!");
      return -1;
   }

   // Did the file changed ?
   if (cfn) {
      if (fgCfgFile.fName.length() <= 0 ||
         (fgCfgFile.fName.length() > 0 && fgCfgFile.fName != cfn)) {
         fgCfgFile.fName = cfn;
         XrdProofdAux::Expand(fgCfgFile.fName);
         fgCfgFile.fMtime = 0;
      }
   } else if (!cfn) {
      // Link to the file
      cfn = fgCfgFile.fName.c_str();
   }

   // Get the modification time
   struct stat st;
   if (stat(cfn, &st) != 0)
      return -1;
   TRACE(DBG, "Config: file: " << cfn);
   TRACE(DBG, "Config: time of last modification: " << st.st_mtime);

   // File should be loaded only once
   if (st.st_mtime <= fgCfgFile.fMtime)
      return 0;

   // Save the modification time
   fgCfgFile.fMtime = st.st_mtime;

   // This part must be modified in atomic way
   XrdSysMutexHelper mhp(fgXPDMutex);

   // Reconfigure tracing
   if (rcf)
      TraceConfig(fgCfgFile.fName.c_str());

   XrdOucStream cfg(&fgEDest, getenv("XRDINSTANCE"));
   char *var;
   int cfgFD, NoGo = 0;

   // Open and attach the config file
   if ((cfgFD = open(cfn, O_RDONLY, 0)) < 0)
      return fgEDest.Emsg("Config", errno, "open config file", cfn);
   cfg.Attach(cfgFD);

   if (rcf) {
      // Reset the strings to the default values
      fgProofServEnvs = "";
      fgProofServRCs = "";
      fgInternalWait = 30;
   }

   // Communicate the host name to the config directives, so that the (deprecated)
   // old style 'if' condition can be handled
   fgConfigDirectives.Apply(SetHostInDirectives, (void *)fgMgr.Host());
   fgReConfigDirectives.Apply(SetHostInDirectives, (void *)fgMgr.Host());

   XrdOucHash<XrdProofdDirective> *fst = (rcf) ? &fgReConfigDirectives
                                               : &fgConfigDirectives;
   XrdOucHash<XrdProofdDirective> *snd = (rcf) ? &fgConfigDirectives : 0;

   char *val = 0;
   while ((var = cfg.GetMyFirstWord())) {
      if (!(strncmp("xpd.", var, 4)) && var[4]) {
         var += 4;
         // Get the value
         val = cfg.GetToken();
         // Get the directive
         XrdProofdDirective *d = fst->Find(var);
         if (d) {
            // Process it
            d->DoDirective(val, &cfg, rcf);
         } else if (snd && (d = snd->Find(var))) {
            TRACE(XERR, "Config: directive xpd."<<var<<" cannot be re-configured");
         }
      }
   }

   return NoGo;
}

//__________________________________________________________________________
void XrdProofdProtocol::RegisterConfigDirectives()
{
   // Register directives for configuration

   // Register special config directives
   fgConfigDirectives.Add("putenv",
      new XrdProofdDirective("putenv", 0, &DoProtocolDirective));
   fgConfigDirectives.Add("putrc",
      new XrdProofdDirective("putrc", 0, &DoProtocolDirective));
   // Register config directives for ints
   fgConfigDirectives.Add("intwait",
      new XrdProofdDirective("intwait", (void *)&fgInternalWait, &DoDirectiveInt));
}

//__________________________________________________________________________
void XrdProofdProtocol::RegisterReConfigDirectives()
{
   // Register directives for configuration

   // Register special config directives
   fgReConfigDirectives.Add("putenv",
      new XrdProofdDirective("putenv", 0, &DoProtocolDirective));
   fgReConfigDirectives.Add("putrc",
      new XrdProofdDirective("putrc", 0, &DoProtocolDirective));
   // Register config directives for ints
   fgReConfigDirectives.Add("intwait",
      new XrdProofdDirective("intwait", (void *)&fgInternalWait, &DoDirectiveInt));
}

//______________________________________________________________________________
int XrdProofdProtocol::ProcessDirective(XrdProofdDirective *d,
                                        char *val, XrdOucStream *cfg, bool rcf)
{
   // Update the priorities of the active sessions.

   if (!d)
      // undefined inputs
      return -1;

   if (d->fName == "putenv") {
      return DoDirectivePutEnv(val, cfg, rcf);
   } else if (d->fName == "putrc") {
      return DoDirectivePutRc(val, cfg, rcf);
   }
   TRACE(XERR,"ProcessDirective: unknown directive: "<<d->fName);
   return -1;
}

//______________________________________________________________________________
int XrdProofdProtocol::DoDirectivePutEnv(char *val, XrdOucStream *, bool)
{
   // Process 'putenv' directives

   if (!val)
      // undefined inputs
      return -1;

   // Env variable to exported to 'proofserv'
   if (fgProofServEnvs.length() > 0)
      fgProofServEnvs += ',';
   fgProofServEnvs += val;

   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::DoDirectivePutRc(char *val, XrdOucStream *cfg, bool)
{
   // Process 'putenv' directives

   if (!val || !cfg)
      // undefined inputs
      return -1;

   // rootrc variable to be passed to 'proofserv':
   if (fgProofServRCs.length() > 0)
      fgProofServRCs += ',';
   fgProofServRCs += val;
   while ((val = cfg->GetToken()) && val[0]) {
      fgProofServRCs += ' ';
      fgProofServRCs += val;
   }

   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Process(XrdLink *)
{
   // Process the information received on the active link.
   // (We ignore the argument here)

   int rc = 0;
   TRACEP(REQ, "Process: enter: instance: " << this);

   // Read the next request header
   if ((rc = GetData("request", (char *)&fRequest, sizeof(fRequest))) != 0)
      return rc;
   TRACEP(DBG, "Process: after GetData: rc: " << rc);

   // Deserialize the data
   fRequest.header.requestid = ntohs(fRequest.header.requestid);
   fRequest.header.dlen      = ntohl(fRequest.header.dlen);

   // The stream ID for the reply
   { XrdSysMutexHelper mh(fResponse.fMutex);
      fResponse.Set(fRequest.header.streamid);
      fResponse.Set(fLink);
   }
   unsigned short sid;
   memcpy((void *)&sid, (const void *)&(fRequest.header.streamid[0]), 2);
   TRACEP(DBG, "Process: sid: " << sid <<
               ", req: " <<fRequest.header.requestid <<
               ", dlen: " <<fRequest.header.dlen);

   // Every request has an associated data length. It better be >= 0 or we won't
   // be able to know how much data to read.
   if (fRequest.header.dlen < 0) {
      fResponse.Send(kXR_ArgInvalid, "Process: Invalid request data length");
      return fLink->setEtext("Process: protocol data length error");
   }

   // Read any argument data at this point, except when the request is to forward
   // a buffer: the argument may have to be segmented and we're not prepared to do
   // that here.
   if (fRequest.header.requestid != kXP_sendmsg && fRequest.header.dlen) {
      if (GetBuff(fRequest.header.dlen+1) != 1) {
         fResponse.Send(kXR_ArgTooLong, "fRequest.argument is too long");
         return 0;
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

   TRACEP(REQ, "Process2: enter: req id: " << fRequest.header.requestid);

   // If the user is not yet logged in, restrict what the user can do
   if (!fStatus || !(fStatus & XPD_LOGGEDIN))
      switch(fRequest.header.requestid) {
      case kXP_auth:
         return Auth();
      case kXP_login:
         return Login();
      default:
         TRACEP(XERR,"Process2: invalid request: " <<fRequest.header.requestid);
         fResponse.Send(kXR_InvalidRequest,"Invalid request; user not logged in");
         return fLink->setEtext("protocol sequence error 1");
      }

   // Once logged-in, the user can request the real actions
   XrdOucString emsg("Invalid request code: ");
   switch(fRequest.header.requestid) {
   case kXP_create:
      if (fSrvType != kXPD_Admin)
         return Create();
      else
         emsg += "'admin' role not allowd to process 'create'";
      break;
   case kXP_destroy:
      if (fSrvType != kXPD_Admin)
         return Destroy();
      else
         emsg += "'admin' role not allowd to process 'destroy'";
      break;
   case kXP_sendmsg:
      return SendMsg();
   case kXP_attach:
       if (fSrvType != kXPD_Admin)
         return Attach();
      else
         emsg += "'admin' role not allowd to process 'attach'";
      break;
   case kXP_detach:
      if (fSrvType != kXPD_Admin)
         return Detach();
      else
         emsg += "'admin' role not allowd to process 'detach'";
      break;
   case kXP_admin:
      return Admin();
   case kXP_interrupt:
      if (fSrvType != kXPD_Admin)
         return Interrupt();
      else
         emsg += "'admin' role not allowd to process 'interrupt'";
      break;
   case kXP_ping:
      return Ping();
   case kXP_urgent:
      return Urgent();
   case kXP_readbuf:
      return ReadBuffer();
   default:
      emsg += fRequest.header.requestid;
      break;
   }

   // Whatever we have, it's not valid
   fResponse.Send(kXR_InvalidRequest, emsg.c_str());
   return 0;
}

//______________________________________________________________________
void XrdProofdProtocol::Recycle(XrdLink *, int, const char *)
{
   // Recycle call. Release the instance and give it back to the stack.

   const char *srvtype[6] = {"ANY", "Worker", "Master",
                             "TopMaster", "Internal", "Admin"};

   // Document the disconnect
   TRACEI(REQ,"Recycle: enter: instance: " <<this<<", type: "<<srvtype[fSrvType+1]);

   // If we have a buffer, release it
   if (fArgp) {
      fgBPool->Release(fArgp);
      fArgp = 0;
   }

   // Flag for internal connections: those deserve a different treatment
   bool proofsrv = (fSrvType == kXPD_Internal) ? 1 : 0;

   // Locate the client instance
   XrdProofdClient *pmgr = fPClient;

   if (pmgr) {

      if (!proofsrv) {

         // Reset the corresponding client slot in the list of this client
         // Count the remaining top clients
         int nc = 0;
         int ic = 0;
         for (ic = 0; ic < (int) pmgr->Clients()->size(); ic++) {
            if (this == pmgr->Clients()->at(ic))
               pmgr->ResetClient(ic);
            else if (pmgr->Clients()->at(ic) && pmgr->Clients()->at(ic)->fTopClient)
               nc++;
         }

         // If top master ...
         if (fSrvType == kXPD_TopMaster) {
            XrdSysMutexHelper mtxh(pmgr->Mutex());
            // Loop over servers sessions associated to this client and update
            // their attached client vectors
            if (pmgr->ProofServs()->size() > 0) {
               XrdProofServProxy *psrv = 0;
               int is = 0;
               for (is = 0; is < (int) pmgr->ProofServs()->size(); is++) {
                  if ((psrv = pmgr->ProofServs()->at(is))) {
                     // Release CIDs in attached sessions: loop over attached clients
                     XrdClientID *cid = 0;
                     int iic = 0;
                     for (iic = 0; iic < (int) psrv->Clients()->size(); iic++) {
                        if ((cid = psrv->Clients()->at(iic))) {
                           if (cid->P() == this)
                              cid->Reset();
                        }
                     }
                  }
               }
            }

            // If no more clients schedule a shutdown at the PROOF session
            // by the sending the appropriate information
            if (nc <= 0 && pmgr->ProofServs()->size() > 0) {
               XrdProofServProxy *psrv = 0;
               int is = 0;
               for (is = 0; is < (int) pmgr->ProofServs()->size(); is++) {
                  if ((psrv = pmgr->ProofServs()->at(is)) && psrv->IsValid() &&
                       psrv->SrvType() == kXPD_TopMaster &&
                      (psrv->Status() == kXPD_idle || psrv->Status() == kXPD_running)) {
                     if (psrv->SetShutdownTimer(fgMgr.ShutdownOpt(), fgMgr.ShutdownDelay()) != 0) {
                        // Just notify locally: link is closed!
                        XrdOucString msg("Recycle: could not send shutdown info to proofsrv");
                        TRACEI(XERR, msg.c_str());
                     }
                  }
               }
            }

         } else {

            // We cannot continue if the top master went away: we cleanup the session
            XrdSysMutexHelper mtxh(pmgr->Mutex());
            if (pmgr->ProofServs()->size() > 0) {
               XrdProofServProxy *psrv = 0;
               int is = 0;
               for (is = 0; is < (int) pmgr->ProofServs()->size(); is++) {
                  if ((psrv = pmgr->ProofServs()->at(is)) && psrv->IsValid()
                      && psrv->SrvType() != kXPD_TopMaster) {

                     TRACEI(HDBG, "Recycle: found: " << psrv << " (t:"<<psrv->SrvType() <<
                                  ",nc:"<<psrv->Clients()->size()<<")");

                     XrdSysMutexHelper xpmh(psrv->Mutex());

                     // Send a terminate signal to the proofserv
                     if (fgMgr.LogTerminatedProc(psrv->TerminateProofServ()) < 0)
                        // Try hard kill
                        fgMgr.LogTerminatedProc(KillProofServ(psrv->SrvID(), 1));

                     // Reset instance
                     psrv->Reset();
                  }
               }
            }
         }

      } else {

         // Internal connection: we need to remove this instance from the list
         // of proxy servers and to notify the attached clients.
         // Loop over servers sessions associated to this client and locate
         // the one corresponding to this proofserv instance
         XrdSysMutexHelper mtxh(pmgr->Mutex());
         if (pmgr->ProofServs()->size() > 0) {
            XrdProofServProxy *psrv = 0;
            int is = 0;
            for (is = 0; is < (int) pmgr->ProofServs()->size(); is++) {
               if ((psrv = pmgr->ProofServs()->at(is)) && (psrv->Link() == fLink)) {

               TRACEI(HDBG, "Recycle: found: " << psrv << " (v:" << psrv->IsValid() <<
                            ",t:"<<psrv->SrvType() << ",nc:"<<psrv->Clients()->size()<<")");

                  XrdSysMutexHelper xpmh(psrv->Mutex());

                  // Tell other attached clients, if any, that this session is gone
                  if (psrv->Clients()->size() > 0) {
                     char msg[512] = {0};
                     snprintf(msg, 512, "Recycle: session: %s terminated by peer",
                                         psrv->Tag());
                     int len = strlen(msg);
                     int ic = 0;
                     XrdProofdProtocol *p = 0;
                     for (ic = 0; ic < (int) psrv->Clients()->size(); ic++) {
                        // Send message
                        if ((p = psrv->Clients()->at(ic)->P())) {
                           unsigned short sid;
                           p->fResponse.GetSID(sid);
                           p->fResponse.Set(psrv->Clients()->at(ic)->Sid());
                           p->fResponse.Send(kXR_attn, kXPD_errmsg, msg, len);
                           p->fResponse.Set(sid);
                        }
                     }
                  }

                  // Send a terminate signal to the proofserv
                  fgMgr.LogTerminatedProc(KillProofServ(psrv->SrvID()));

                  // Reset instance
                  psrv->Reset();
               }
            }
         }
      }
   }

   // Set fields to starting point (debugging mostly)
   Reset();

   // Push ourselves on the stack
   fgProtStack.Push(&fProtLink);
}

//______________________________________________________________________________
int XrdProofdProtocol::Login()
{
   // Process a login request

   int rc = 1;

   TRACEP(REQ, "Login: enter");

   // Check if there was any change in the configuration
   XrdProofdProtocol::Config(0, 1);

   // If this server is explicitely required to be a worker node or a
   // submaster, check whether the requesting host is allowed to connect
   if (fRequest.login.role[0] != 'i' &&
      (fgMgr.SrvType() == kXPD_WorkerServer || fgMgr.SrvType() == kXPD_MasterServer)) {
      if (!fgMgr.CheckMaster(fLink->Host())) {
         TRACEP(XERR,"Login: master not allowed to connect - "
                    "ignoring request ("<<fLink->Host()<<")");
         fResponse.Send(kXR_InvalidRequest,
                    "Login: master not allowed to connect - request ignored");
         return rc;
      }
   }

   // If this is the second call (after authentication) we just need
   // mapping
   if (fStatus == XPD_NEED_MAP) {

      // Check if this is a priviliged client
      char *p = 0;
      if ((p = (char *) strstr(fgMgr.SuperUsers(), fClientID))) {
         if (p == fgMgr.SuperUsers() || (p > fgMgr.SuperUsers() && *(p-1) == ',')) {
            if (!(strncmp(p, fClientID, strlen(fClientID)))) {
               fSuperUser = 1;
               TRACEP(LOGIN,"Login: privileged user ");
            }
         }
      }
      // Acknowledge the client
      fResponse.Send();
      fStatus = XPD_LOGGEDIN;
      return MapClient(0);
   }

   // Make sure the user is not already logged in
   if ((fStatus & XPD_LOGGEDIN)) {
      fResponse.Send(kXR_InvalidRequest, "duplicate login; already logged in");
      return rc;
   }

   int i, pid;
   XrdOucString uname, gname;

   // Unmarshall the data
   pid = (int)ntohl(fRequest.login.pid);
   char un[9];
   for (i = 0; i < (int)sizeof(un)-1; i++) {
      if (fRequest.login.username[i] == '\0' || fRequest.login.username[i] == ' ')
         break;
      un[i] = fRequest.login.username[i];
   }
   un[i] = '\0';
   uname = un;

   // Longer usernames are in the attached buffer
   if (uname == "?>buf") {
      // Attach to buffer
      char *buf = fArgp->buff;
      int   len = fRequest.login.dlen;
      // Extract username
      uname.assign(buf,0,len-1);
      int iusr = uname.find("|usr:");
      if (iusr == -1) {
         TRACEP(XERR,"Login: long user name not found");
         fResponse.Send(kXR_InvalidRequest,"Login: long user name not found");
         return rc;
      }
      uname.erase(0,iusr+5);
      uname.erase(uname.find("|"));
   }

   // Extract group name, if specified (syntax is uname[:gname])
   int ig = uname.find(":");
   if (ig != -1) {
      gname.assign(uname, ig+1);
      uname.erase(ig);
      TRACEP(DBG,"Login: requested group: "<<gname);
   }

   // Here we check if the user is allowed to use the system
   // If not, we fail.
   XrdOucString emsg;
   if (fgMgr.CheckUser(uname.c_str(), fUI, emsg) != 0) {
      emsg.insert(": ", 0);
      emsg.insert(uname, 0);
      emsg.insert("Login: ClientID not allowed: ", 0);
      TRACEP(XERR, emsg.c_str());
      fResponse.Send(kXR_InvalidRequest, emsg.c_str());
      return rc;
   }

   // Check if user belongs to a group
   if (fgMgr.GroupsMgr() && fgMgr.GroupsMgr()->Num() > 0) {
      XrdProofGroup *g = 0;
      if (gname.length() > 0) {
         g = fgMgr.GroupsMgr()->GetGroup(gname.c_str());
         if (!g) {
            emsg = "Login: group unknown: ";
            emsg += gname;
            TRACEP(XERR, emsg.c_str());
            fResponse.Send(kXR_InvalidRequest, emsg.c_str());
            return rc;
         } else if (strncmp(g->Name(),"default",7) &&
                   !g->HasMember(uname.c_str())) {
            emsg = "Login: user ";
            emsg += uname;
            emsg += " is not member of group ";
            emsg += gname;
            TRACEP(XERR, emsg.c_str());
            fResponse.Send(kXR_InvalidRequest, emsg.c_str());
            return rc;
         } else {
            if (TRACING(DBG)) {
               TRACEP(DBG,"Login: group: "<<gname<<" found");
               g->Print();
            }
         }
      } else {
         g = fgMgr.GroupsMgr()->GetUserGroup(uname.c_str());
         gname = g ? g->Name() : "default";
      }
   }

   // Establish the ID for this link
   fLink->setID(uname.c_str(), pid);
   fCapVer = fRequest.login.capver[0];

   // Establish the ID for this client
   fClientID = new char[uname.length()+4];
   strcpy(fClientID, uname.c_str());
   TRACEI(LOGIN,"Login: ClientID = " << fClientID);

   // Establish the group ID for this client
   fGroupID = new char[gname.length()+4];
   strcpy(fGroupID, gname.c_str());
   TRACEI(LOGIN,"Login: GroupID = " << fGroupID);

   // Assert the workdir directory ...
   fUI.fWorkDir = fUI.fHomeDir;
   if (fgMgr.WorkDir() && strlen(fgMgr.WorkDir()) > 0) {
      // The user directory path will be <workdir>/<user>
      fUI.fWorkDir = fgMgr.WorkDir();
      if (!fUI.fWorkDir.endswith('/'))
         fUI.fWorkDir += "/";
      fUI.fWorkDir += fClientID;
   } else {
      // Default: $HOME/proof
      if (!fUI.fWorkDir.endswith('/'))
         fUI.fWorkDir += "/";
      fUI.fWorkDir += "proof";
      if (fUI.fUser != fClientID) {
         fUI.fWorkDir += "/";
         fUI.fWorkDir += fClientID;
      }
   }
   TRACEI(LOGIN,"Login: work dir = " << fUI.fWorkDir);

   // Make sure the directory exists
   if (XrdProofdAux::AssertDir(fUI.fWorkDir.c_str(), fUI, fgMgr.ChangeOwn()) == -1) {
      emsg = "Login: unable to create work dir: ";
      emsg += fUI.fWorkDir;
      TRACEP(XERR, emsg);
      fResponse.Send(kXP_ServerError, emsg.c_str());
      return rc;
   }

   // If strong authentication is required ...
   if (fgMgr.CIA()) {
      // ... make sure that the directory for credentials exists in the sandbox ...
      XrdOucString credsdir = fUI.fWorkDir;
      credsdir += "/.creds";
      // Acquire user identity
      XrdSysPrivGuard pGuard((uid_t)fUI.fUid, (gid_t)fUI.fGid);
      if (!pGuard.Valid()) {
         emsg = "Login: could not get privileges to create credential dir ";
         emsg += credsdir;
         TRACEP(XERR, emsg);
         fResponse.Send(kXP_ServerError, emsg.c_str());
         return rc;
      }
      if (XrdProofdAux::AssertDir(credsdir.c_str(), fUI, fgMgr.ChangeOwn()) == -1) {
         emsg = "Login: unable to create credential dir: ";
         emsg += credsdir;
         TRACEP(XERR, emsg);
         fResponse.Send(kXP_ServerError, emsg.c_str());
         return rc;
      }
   }

   // Find out the server type: 'i', internal, means this is a proofsrv calling back.
   // For the time being authentication is required for clients only.
   bool needauth = 0;
   bool ismaster = (fgMgr.SrvType() == kXPD_TopMaster ||
                    fgMgr.SrvType() == kXPD_MasterServer) ? 1 : 0;
   switch (fRequest.login.role[0]) {
   case 'A':
      fSrvType = kXPD_Admin;
      fResponse.Set("adm: ");
      break;
   case 'i':
      fSrvType = kXPD_Internal;
      fResponse.Set("int: ");
      break;
   case 'M':
      if (fgMgr.SrvType() == kXPD_AnyServer || ismaster) {
         fTopClient = 1;
         fSrvType = kXPD_TopMaster;
         needauth = 1;
         fResponse.Set("m2c: ");
      } else {
         TRACEP(XERR,"Login: top master mode not allowed - ignoring request");
         fResponse.Send(kXR_InvalidRequest,
                        "Server not allowed to be top master - ignoring request");
         return rc;
      }
      break;
   case 'm':
      if (fgMgr.SrvType() == kXPD_AnyServer || ismaster) {
         fSrvType = kXPD_MasterServer;
         needauth = 1;
         fResponse.Set("m2m: ");
      } else {
         TRACEP(XERR,"Login: submaster mode not allowed - ignoring request");
         fResponse.Send(kXR_InvalidRequest,
                        "Server not allowed to be submaster - ignoring request");
         return rc;
      }
      break;
   case 's':
      if (fgMgr.SrvType() == kXPD_AnyServer || fgMgr.SrvType() == kXPD_WorkerServer) {
         fSrvType = kXPD_WorkerServer;
         needauth = 1;
         fResponse.Set("w2m: ");
      } else {
         TRACEP(XERR,"Login: worker mode not allowed - ignoring request");
         fResponse.Send(kXR_InvalidRequest,
                        "Server not allowed to be worker - ignoring request");
         return rc;
      }
      break;
   default:
      TRACEP(XERR, "Login: unknown mode: '" << fRequest.login.role[0] <<"'");
      fResponse.Send(kXR_InvalidRequest, "Server type: invalide mode");
      return rc;
   }

   // Get the security token for this link. We will either get a token, a null
   // string indicating host-only authentication, or a null indicating no
   // authentication. We can then optimize of each case.
   if (needauth && fgMgr.CIA()) {
      const char *pp = fgMgr.CIA()->getParms(i, fLink->Name());
      if (pp && i ) {
         fResponse.Send((kXR_int32)XPROOFD_VERSBIN, (void *)pp, i);
         fStatus = (XPD_NEED_MAP | XPD_NEED_AUTH);
         return rc;
      } else {
         fResponse.Send((kXR_int32)XPROOFD_VERSBIN);
         fStatus = XPD_LOGGEDIN;
         if (pp) {
            fEntity.tident = fLink->ID;
            fClient = &fEntity;
         }
      }
   } else {
      rc = fResponse.Send((kXR_int32)XPROOFD_VERSBIN);
      fStatus = XPD_LOGGEDIN;

      // Check if this is a priviliged client
      char *p = 0;
      if ((p = (char *) strstr(fgMgr.SuperUsers(), fClientID))) {
         if (p == fgMgr.SuperUsers() || (p > fgMgr.SuperUsers() && *(p-1) == ',')) {
            if (!(strncmp(p, fClientID, strlen(fClientID)))) {
               fSuperUser = 1;
               TRACEI(LOGIN,"Login: privileged user ");
            }
         }
      }
   }

   // Map the client
   return MapClient(1);
}

//______________________________________________________________________________
int XrdProofdProtocol::MapClient(bool all)
{
   // Process a login request
   int rc = 1;

   TRACEI(REQ,"MapClient: enter");

   // Flag for internal connections
   bool proofsrv = ((fSrvType == kXPD_Internal) && all) ? 1 : 0;

   // If call back from proofsrv, find out the target session
   short int psid = -1;
   char protver = -1;
   short int clientvers = -1;
   if (proofsrv) {
      memcpy(&psid, (const void *)&(fRequest.login.reserved[0]), 2);
      if (psid < 0) {
         TRACEP(XERR,"MapClient: proofsrv callback: sent invalid session id");
         fResponse.Send(kXR_InvalidRequest,
                        "MapClient: proofsrv callback: sent invalid session id");
         return rc;
      }
      protver = fRequest.login.capver[0];
      TRACEI(DBG,"MapClient: proofsrv callback for session: " <<psid);
   } else {
      // Get PROOF version run by client
      memcpy(&clientvers, (const void *)&(fRequest.login.reserved[0]), 2);
      TRACEI(DBG,"MapClient: PROOF version run by client: " <<clientvers);
   }

   // Now search for an existing manager session for this ClientID
   XrdProofdClient *pmgr = 0;
   TRACEI(DBG,"MapClient: # of clients: "<<fgMgr.ProofdClients()->size());
   // This part may be not thread safe
   {  XrdSysMutexHelper mtxh(&fgXPDMutex);
      if (fgMgr.ProofdClients()->size() > 0) {
         std::list<XrdProofdClient *>::iterator i;
         for (i = fgMgr.ProofdClients()->begin(); i != fgMgr.ProofdClients()->end(); ++i) {
            if ((pmgr = *i) && pmgr->Match(fClientID, fGroupID))
               break;
            TRACEI(HDBG, "MapClient: client: "<<pmgr->ID()<< ", group: "<<
                         ((pmgr->Group()) ? pmgr->Group()->Name() : "---"));
            pmgr = 0;
         }
      }
   }

   // Map the existing session, if found
   if (pmgr && pmgr->IsValid()) {
      // Save as reference proof mgr
      fPClient = pmgr;
      TRACEI(DBG,"MapClient: matching client: "<<pmgr);

      // If proofsrv, locate the target session
      if (proofsrv) {
         XrdProofServProxy *psrv = 0;
         int is = 0;
         for (is = 0; is < (int) pmgr->ProofServs()->size(); is++) {
            if ((psrv = pmgr->ProofServs()->at(is)) && psrv->Match(psid))
               break;
            psrv = 0;
         }
         if (!psrv) {
            TRACEP(XERR, "MapClient: proofsrv callback:"
                        " wrong target session: protocol error");
            fResponse.Send(kXP_nosession, "MapClient: proofsrv callback:"
                           " wrong target session: protocol error");
            return -1;
         } else {
            // Set the protocol version
            psrv->SetProtVer(protver);
            // Assign this link to it
            psrv->SetLink(fLink);
            psrv->ProofSrv()->Set(fRequest.header.streamid);
            psrv->ProofSrv()->Set(fLink);
            // Set Trace ID
            XrdOucString tid(" : xrd->");
            tid += psrv->Ordinal();
            tid += " ";
            psrv->ProofSrv()->Set(tid.c_str());
            TRACEI(DBG,"MapClient: proofsrv callback:"
                       " link assigned to target session "<<psid);
         }
      } else {

         // Make sure that the version is filled correctly (if an admin operation
         // was run before this may still be -1 on workers)
         pmgr->SetClientVers(clientvers);

         // The index of the next free slot will be the unique ID
         fCID = pmgr->GetClientID(this);

         // If any PROOF session in shutdown state exists, stop the related
         // shutdown timers
         if (pmgr->ProofServs()->size() > 0) {
            XrdProofServProxy *psrv = 0;
            int is = 0;
            for (is = 0; is < (int) pmgr->ProofServs()->size(); is++) {
               if ((psrv = pmgr->ProofServs()->at(is)) &&
                    psrv->IsValid() && (psrv->SrvType() == kXPD_TopMaster) &&
                    psrv->IsShutdown()) {
                  if (psrv->SetShutdownTimer(fgMgr.ShutdownOpt(), fgMgr.ShutdownDelay(), 0) != 0) {
                     XrdOucString msg("MapClient: could not stop shutdown timer in proofsrv ");
                     msg += psrv->SrvID();
                     msg += "; status: ";
                     msg += psrv->StatusAsString();
                     fResponse.Send(kXR_attn, kXPD_srvmsg, (void *) msg.c_str(), msg.length());
                  }
               }
            }
         }
      }

   } else {

      // Proofsrv callbacks need something to attach to
      if (proofsrv) {
         TRACEI(XERR, "MapClient: proofsrv callback:"
                     " no manager to attach to: protocol error");
         return -1;
      }

      // This part may be not thread safe
      {  XrdSysMutexHelper mtxh(&fgXPDMutex);

         // Make sure that no zombie proofserv is around
         CleanupProofServ(0, fClientID);
         if (!pmgr) {
            // No existing session: create a new one
            pmgr = new XrdProofdClient(fClientID, clientvers, fUI);
            pmgr->SetROOT(fgMgr.ROOT()->front());
            // Locate and set the group, if any
            if (fgMgr.GroupsMgr() && fgMgr.GroupsMgr()->Num() > 0)
               pmgr->SetGroup(fgMgr.GroupsMgr()->GetUserGroup(fClientID, fGroupID));
            // Add to the list
            fgMgr.ProofdClients()->push_back(pmgr);
         } else {
            // An instance not yet valid exists already: fill it
            pmgr->SetClientVers(clientvers);
            pmgr->SetWorkdir(fUI.fWorkDir.c_str());
            if (!(pmgr->ROOT()))
               pmgr->SetROOT(fgMgr.ROOT()->front());
         }
         // Save as reference proof mgr
         fPClient = pmgr;
      }

      // No existing session: create a new one
      if (pmgr && (pmgr->CreateUNIXSock(&fgEDest, fgMgr.TMPdir()) == 0)) {

         TRACEI(DBG,"MapClient: NEW client: "<<pmgr<<
                    ", group: "<<((pmgr->Group()) ? pmgr->Group()->Name() : "???"));

         // The index of the next free slot will be the unique ID
         fCID = pmgr->GetClientID(this);

         // Reference Stream ID
         unsigned short sid;
         memcpy((void *)&sid, (const void *)&(fRequest.header.streamid[0]), 2);
         pmgr->SetRefSid(sid);

         // Check if old sessions are still flagged as active
         XrdOucString tobemv;

         // Get list of session working dirs flagged as active,
         // and check if they have to be deactivated
         std::list<XrdOucString *> sactlst;
         if (pmgr->GetSessionDirs(1, &sactlst) == 0) {
            std::list<XrdOucString *>::iterator i;
            for (i = sactlst.begin(); i != sactlst.end(); ++i) {
               char *p = (char *) strrchr((*i)->c_str(), '-');
               if (p) {
                  int pid = strtol(p+1, 0, 10);
                  if (!fgMgr.VerifyProcessByID(pid)) {
                     tobemv += (*i)->c_str();
                     tobemv += '|';
                  }
               }
            }
         }
         // Clean up the list
         sactlst.clear();

         // To avoid dead locks we must close the file and do the mv actions after
         XrdOucString fnact = fUI.fWorkDir;
         fnact += "/.sessions";
         FILE *f = fopen(fnact.c_str(), "r");
         if (f) {
            char ln[1024];
            while (fgets(ln, sizeof(ln), f)) {
               if (ln[strlen(ln)-1] == '\n')
                  ln[strlen(ln)-1] = 0;
               char *p = strrchr(ln, '-');
               if (p) {
                  int pid = strtol(p+1, 0, 10);
                  if (!fgMgr.VerifyProcessByID(pid)) {
                     tobemv += ln;
                     tobemv += '|';
                  }
               }
            }
            fclose(f);
         }

         // Instance can be considered valid by now
         pmgr->SetValid();

         TRACEI(DBG,"MapClient: client "<<pmgr<<" added to the list (ref sid: "<< sid<<")");

         XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
         if (XpdBadPGuard(pGuard, fUI.fUid) && fgMgr.ChangeOwn()) {
            TRACEI(XERR, "MapClient: could not get privileges");
            return -1;
         }

         // Mv inactive sessions, if needed
         if (tobemv.length() > 0) {
            char del = '|';
            XrdOucString tag;
            int from = 0;
            while ((from = tobemv.tokenize(tag, from, del)) != -1) {
               if (fPClient->MvOldSession(tag.c_str()) == -1)
                  TRACEI(REQ, "MapClient: problems recording session as old in sandbox");
            }
         }

         // Set ownership of the socket file to the client
         if (fgMgr.ChangeOwn()) {
            if (chown(pmgr->UNIXSockPath(), fUI.fUid, fUI.fGid) == -1) {
               TRACEI(XERR, "MapClient: cannot set user ownership"
                            " on UNIX socket (errno: "<<errno<<")");
               return -1;
            }
            // Make sure that it worked out
            struct stat st;
            if ((stat(pmgr->UNIXSockPath(), &st) != 0) || 
                (int) st.st_uid != fUI.fUid || (int) st.st_gid != fUI.fGid) {
               TRACEI(XERR, "MapClient: problems setting user ownership"
                            " on UNIX socket");
               return -1;
            }
         }

      } else {
         // Remove from the list
         fgMgr.ProofdClients()->remove(pmgr);
         SafeDelete(pmgr);
         fPClient = 0;
         TRACEP(DBG,"MapClient: cannot instantiate XrdProofdClient");
         fResponse.Send(kXP_ServerError,
                        "MapClient: cannot instantiate XrdProofdClient");
         return rc;
      }
   }

   if (!proofsrv) {
      TRACEI(DBG,"MapClient: fCID: "<<fCID<<", size: "<<fPClient->Clients()->size()<<
                 ", capacity: "<<fPClient->Clients()->capacity());
   }

   // Document this login
   if (!(fStatus & XPD_NEED_AUTH))
      fgEDest.Log(XPD_LOG_01, ":MapClient", fLink->ID, "login");

   return 0;
}

//_____________________________________________________________________________
int XrdProofdProtocol::Auth()
{
   // Analyse client authentication info

   struct sockaddr netaddr;
   XrdSecCredentials cred;
   XrdSecParameters *parm = 0;
   XrdOucErrInfo     eMsg;
   const char *eText;
   int rc;

   TRACEP(REQ,"Auth: enter");

   // Ignore authenticate requests if security turned off
   if (!fgMgr.CIA())
      return fResponse.Send();
   cred.size   = fRequest.header.dlen;
   cred.buffer = fArgp->buff;

   // If we have no auth protocol, try to get it
   if (!fAuthProt) {
      fLink->Name(&netaddr);
      if (!(fAuthProt = fgMgr.CIA()->getProtocol(fLink->Host(), netaddr, &cred, &eMsg))) {
         eText = eMsg.getErrText(rc);
         TRACEP(XERR,"Auth: user authentication failed; "<<eText);
         fResponse.Send(kXR_NotAuthorized, eText);
         return -EACCES;
      }
      fAuthProt->Entity.tident = fLink->ID;
   }

   // Now try to authenticate the client using the current protocol
   XrdOucString namsg;
   if (!(rc = fAuthProt->Authenticate(&cred, &parm, &eMsg))) {

      // Make sure that the user name that we want is allowed
      if (fAuthProt->Entity.name && strlen(fAuthProt->Entity.name) > 0) {
         rc  = -1;
         if (fClientID && strlen(fClientID) > 0) {
            XrdOucString usrs(fAuthProt->Entity.name);
            XrdOucString usr;
            int from = 0;
            while ((from = usrs.tokenize(usr, from, ',')) != STR_NPOS) {
               if ((usr == (const char *)fClientID)) {
                  free(fAuthProt->Entity.name);
                  fAuthProt->Entity.name = strdup(usr.c_str());
                  rc = 0;
                  break;
               }
            }
            if (rc != 0) {
               namsg = "user ";
               namsg += fClientID;
               namsg += " not authorized to connect";
               TRACEP(XERR, namsg.c_str());
            }
         } else {
            TRACEP(XERR, "user name is empty: protocol error?");
         }
      } else {
         TRACEP(XERR, "name of the authenticated entity is empty: protocol error?");
         rc = -1;
      }

      if (rc == 0) {
         const char *msg = (fStatus & XPD_ADMINUSER) ? " admin login as " : " login as ";
         rc = fResponse.Send();
         fStatus &= ~XPD_NEED_AUTH;
         fClient = &fAuthProt->Entity;
         if (fClient->name) {
            TRACEP(LOGIN, fLink->ID << msg << fClient->name);
         } else {
            TRACEP(LOGIN, fLink->ID << msg << " nobody");
         }
         return rc;
      }
   }

   // If we need to continue authentication, tell the client as much
   if (rc > 0) {
      TRACEP(DBG, "Auth: more auth requested; sz: " <<(parm ? parm->size : 0));
      if (parm) {
         rc = fResponse.Send(kXR_authmore, parm->buffer, parm->size);
         delete parm;
         return rc;
      }
      if (fAuthProt) {
         fAuthProt->Delete();
         fAuthProt = 0;
      }
      TRACEP(XERR,"Auth: security requested additional auth w/o parms!");
      fResponse.Send(kXR_ServerError,"invalid authentication exchange");
      return -EACCES;
   }

   // We got an error, bail out
   if (fAuthProt) {
      fAuthProt->Delete();
      fAuthProt = 0;
   }
   eText = (namsg.length() > 0) ? namsg.c_str() : eMsg.getErrText(rc);
   TRACEP(XERR, "Auth: user authentication failed; "<<eText);
   fResponse.Send(kXR_NotAuthorized, eText);
   return -EACCES;
}

//______________________________________________________________________________
int XrdProofdProtocol::GetBuff(int quantum)
{
   // Allocate a buffer to handle quantum bytes

   TRACE(ACT, "GetBuff: enter");

   // The current buffer may be sufficient for the current needs
   if (!fArgp || quantum > fArgp->bsize)
      fhcNow = fhcPrev;
   else if (quantum >= fhalfBSize || fhcNow-- > 0)
      return 1;
   else if (fhcNext >= fhcMax)
      fhcNow = fhcMax;
   else {
      int tmp = fhcPrev;
      fhcNow = fhcNext;
      fhcPrev = fhcNext;
      fhcNext = tmp + fhcNext;
   }

   // We need a new buffer
   if (fArgp)
      fgBPool->Release(fArgp);
   if ((fArgp = fgBPool->Obtain(quantum)))
      fhalfBSize = fArgp->bsize >> 1;
   else
      return fResponse.Send(kXR_NoMemory, "insufficient memory for requested buffer");

   // Success
   return 1;
}

//______________________________________________________________________________
int XrdProofdProtocol::GetData(const char *dtype, char *buff, int blen)
{
   // Get data from the open link

   int rlen;

   // Read the data but reschedule the link if we have not received all of the
   // data within the timeout interval.
   TRACEI(ACT, "GetData: dtype: "<<(dtype ? dtype : " - ")<<", blen: "<<blen);

   rlen = fLink->Recv(buff, blen, fgReadWait);

   if (rlen  < 0) {
      if (rlen != -ENOMSG) {
         XrdOucString emsg = "GetData: link read error: errno: ";
         emsg += -rlen;
         TRACEI(XERR, emsg.c_str());
         return fLink->setEtext(emsg.c_str());
      } else {
         TRACEI(DBG, "GetData: connection closed by peer (errno: "<<-rlen<<")");
         return -1;
      }
   }
   if (rlen < blen) {
      fBuff = buff+rlen; fBlen = blen-rlen;
      TRACEI(XERR, "GetData: " << dtype <<
                  " timeout; read " <<rlen <<" of " <<blen <<" bytes");
      return 1;
   }
   TRACEI(DBG, "GetData: rlen: "<<rlen);

   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Attach()
{
   // Handle a request to attach to an existing session

   int psid = -1, rc = 1;

   // Unmarshall the data
   psid = ntohl(fRequest.proof.sid);
   TRACEI(REQ, "Attach: psid: "<<psid<<", fCID = "<<fCID);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
       !(xps = fPClient->ProofServs()->at(psid))) {
      TRACEP(XERR, "Attach: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
      return rc;
   }
   TRACEP(DBG, "Attach: xps: "<<xps<<", status: "<< xps->Status());

   // Stream ID
   unsigned short sid;
   memcpy((void *)&sid, (const void *)&(fRequest.header.streamid[0]), 2);

   // We associate this instance to the corresponding slot in the
   // session vector of attached clients
   XrdClientID *csid = xps->GetClientID(fCID);
   csid->SetP(this);
   csid->SetSid(sid);

   // Take parentship, if orphalin
   if (!(xps->Parent()))
      xps->SetParent(csid);

   // Notify to user
   if (fSrvType == kXPD_TopMaster) {
      // Send also back the data pool url
      XrdOucString dpu = fgMgr.PoolURL();
      if (!dpu.endswith('/'))
         dpu += '/';
      dpu += fgMgr.NameSpace();
      fResponse.Send(psid, xps->ROOT()->SrvProtVers(), (kXR_int16)XPROOFD_VERSBIN,
                     (void *) dpu.c_str(), dpu.length());
   } else
      fResponse.Send(psid, xps->ROOT()->SrvProtVers(), (kXR_int16)XPROOFD_VERSBIN);

   // Send saved query num message
   if (xps->QueryNum()) {
      TRACEP(XERR, "Attach: sending query num message ("<<
                  xps->QueryNum()->fSize<<" bytes)");
      fResponse.Send(kXR_attn, kXPD_msg,
                     xps->QueryNum()->fBuff, xps->QueryNum()->fSize);
   }
   // Send saved start processing message, if not idle
   if (xps->Status() == kXPD_running && xps->StartMsg()) {
      TRACEP(XERR, "Attach: sending start process message ("<<
                  xps->StartMsg()->fSize<<" bytes)");
      fResponse.Send(kXR_attn, kXPD_msg,
                     xps->StartMsg()->fBuff, xps->StartMsg()->fSize);
   }

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Detach()
{
   // Handle a request to detach from an existing session

   int psid = -1, rc = 1;

   XrdSysMutexHelper mh(fMutex);

   // Unmarshall the data
   psid = ntohl(fRequest.proof.sid);
   TRACEI(REQ, "Detach: psid: "<<psid);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
       !(xps = fPClient->ProofServs()->at(psid))) {
      TRACEP(XERR, "Detach: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
      return rc;
   }
   TRACEP(DBG, "Detach: xps: "<<xps<<", status: "<< xps->Status()<<
               ", # clients: "<< xps->Clients()->size());

   XrdSysMutexHelper xpmh(xps->Mutex());

   // Remove this from the list of clients
   std::vector<XrdClientID *>::iterator i;
   for (i = xps->Clients()->begin(); i != xps->Clients()->end(); ++i) {
      if (*i) {
         if ((*i)->P() == this) {
            delete (*i);
            xps->Clients()->erase(i);
            break;
         }
      }
   }

   // Notify to user
   fResponse.Send();

   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Destroy()
{
   // Handle a request to shutdown an existing session

   int psid = -1, rc = 1;

   int kpid = -1;

   {  XrdSysMutexHelper mh(fPClient->Mutex());

      // Unmarshall the data
      psid = ntohl(fRequest.proof.sid);
      TRACEI(REQ, "Destroy: psid: "<<psid);

      // Find server session
      XrdProofServProxy *xpsref = 0;
      if (psid > -1) {
         // Request for a specific session
         if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
            !(xpsref = fPClient->ProofServs()->at(psid))) {
            TRACEP(XERR, "Destroy: reference session ID not found");
            fResponse.Send(kXR_InvalidRequest,"reference session ID not found");
            return rc;
         }
      }

      // Loop over servers
      XrdProofServProxy *xps = 0;
      int is = 0;
      for (is = 0; is < (int) fPClient->ProofServs()->size(); is++) {

         if ((xps = fPClient->ProofServs()->at(is)) && (xpsref == 0 || xps == xpsref)) {

            TRACEI(DBG, "Destroy: xps: "<<xps<<", status: "<< xps->Status()<<", pid: "<<xps->SrvID());

            {  XrdSysMutexHelper xpmh(xps->Mutex());

               if (xps->SrvType() == kXPD_TopMaster) {
                  // Tell other attached clients, if any, that this session is gone
                  if (fTopClient && xps->Clients()->size() > 0) {
                     char msg[512] = {0};
                     snprintf(msg, 512, "Destroy: session: %s destroyed by: %s",
                              xps->Tag(), fLink->ID);
                     int len = strlen(msg);
                     int ic = 0;
                     XrdProofdProtocol *p = 0;
                     for (ic = 0; ic < (int) xps->Clients()->size(); ic++) {
                        if ((p = xps->Clients()->at(ic)->P()) &&
                           (p != this) && p->fTopClient) {
                           unsigned short sid;
                           p->fResponse.GetSID(sid);
                           p->fResponse.Set(xps->Clients()->at(ic)->Sid());
                           p->fResponse.Send(kXR_attn, kXPD_srvmsg, msg, len);
                           p->fResponse.Set(sid);
                        }
                     }
                  }
               }

               // Send a terminate signal to the proofserv
               if ((kpid = xps->TerminateProofServ()) < 0)
                  kpid = KillProofServ(xps->SrvID(), 1);

               // Reset instance
               xps->Reset();

               // If single delete we are done
               if ((xpsref != 0 && (xps == xpsref)))
                  break;
            }
         }

      }
   }

   // Register the termination
   fgMgr.LogTerminatedProc(kpid);

   // Notify to user
   fResponse.Send();

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::SaveAFSkey(XrdSecCredentials *c, const char *dir)
{
   // Save the AFS key, if any, for usage in proofserv in file 'dir'/.afs .
   // Return 0 on success, -1 on error.

   // Check file name
   if (!dir || strlen(dir) <= 0) {
      TRACE(XERR, "SaveAFSkey: dir name undefined");
      return -1;
   }

   // Check credentials
   if (!c) {
      TRACE(XERR, "SaveAFSkey: credentials undefined");
      return -1;
   }

   // Decode credentials
   int lout = 0;
   char *out = new char[c->size];
   if (XrdSutFromHex(c->buffer, out, lout) != 0) {
      TRACE(XERR, "SaveAFSkey: problems unparsing hex string");
      delete [] out;
      return -1;
   }

   // Locate the key
   char *key = out + 5;
   if (strncmp(key, "afs:", 4)) {
      TRACE(DBG, "SaveAFSkey: string does not contain an AFS key");
      delete [] out;
      return 0;
   }
   key += 4;

   // Filename
   XrdOucString fn = dir;
   fn += "/.afs";
   // Open the file, truncatin g if already existing
   int fd = open(fn.c_str(), O_WRONLY | O_CREAT | O_TRUNC);
   if (fd <= 0) {
      TRACE(XERR, "SaveAFSkey: problems creating file - errno: " << errno);
      delete [] out;
      return -1;
   }
   // Make sure it is protected
   if (fchmod(fd, 0600) != 0) {
      TRACE(XERR, "SaveAFSkey: problems setting file permissions to 0600 - errno: " << errno);
      delete [] out;
      close(fd);
      return -1;
   }
   // Write out the key
   int rc = 0;
   int lkey = lout - 9;
   if (XrdProofdAux::Write(fd, key, lkey) != lkey) {
      TRACE(XERR, "SaveAFSkey: problems writing to file - errno: " << errno);
      rc = -1;
   }

   // Cleanup
   delete [] out;
   close(fd);
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::SetProofServEnvOld(int psid, int loglevel, const char *cfg)
{
   // Set environment for proofserv; old version preparing the environment for
   // proofserv protocol version <= 13. Needed for backward compatibility.

   char *ev = 0;

   MTRACE(REQ,  "xpd:child: ", "SetProofServEnv: enter: psid: "<<psid<<
                      ", log: "<<loglevel);

   // Make sure the principal client is defined
   if (!fPClient) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnv: principal client undefined - cannot continue");
      return -1;
   }

   // Set basic environment for proofserv
   if (SetProofServEnv(fPClient->ROOT()) != 0) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnvOld: problems setting basic environment - exit");
      return -1;
   }

   // Session proxy
   XrdProofServProxy *xps = fPClient->ProofServs()->at(psid);
   if (!xps) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnvOld: unable to get instance of proofserv proxy");
      return -1;
   }

   // Work directory
   XrdOucString udir = fPClient->Workdir();
   MTRACE(DBG, "xpd:child: ",
               "SetProofServEnvOld: working dir for "<<fClientID<<" is: "<<udir);

   // Session tag
   char hn[64], stag[512];
#if defined(XPD__SUNCC)
   sysinfo(SI_HOSTNAME, hn, sizeof(hn));
#else
   gethostname(hn, sizeof(hn));
#endif
   XrdOucString host = hn;
   if (host.find(".") != STR_NPOS)
      host.erase(host.find("."));
   sprintf(stag,"%s-%d-%d",host.c_str(),(int)time(0),getpid());

   // Session dir
   XrdOucString logdir = udir;
   if (fSrvType == kXPD_TopMaster) {
      logdir += "/session-";
      logdir += stag;
      xps->SetTag(stag);
   } else {
      logdir += "/";
      logdir += xps->Tag();
   }
   MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: log dir "<<logdir);
   // Make sure the directory exists
   if (XrdProofdAux::AssertDir(logdir.c_str(), fUI, fgMgr.ChangeOwn()) == -1) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnvOld: unable to create log dir: "<<logdir);
      return -1;
   }
   // The session dir (sandbox) depends on the role
   XrdOucString sessdir = logdir;
   if (fSrvType == kXPD_WorkerServer)
      sessdir += "/worker-";
   else
      sessdir += "/master-";
   sessdir += xps->Ordinal();
   sessdir += "-";
   sessdir += stag;
   ev = new char[strlen("ROOTPROOFSESSDIR=")+sessdir.length()+2];
   sprintf(ev, "ROOTPROOFSESSDIR=%s", sessdir.c_str());
   putenv(ev);
   MTRACE(DBG,  "xpd:child: ", "SetProofServEnvOld: "<<ev);

   // Log level
   ev = new char[strlen("ROOTPROOFLOGLEVEL=")+5];
   sprintf(ev, "ROOTPROOFLOGLEVEL=%d", loglevel);
   putenv(ev);
   MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: "<<ev);

   // Ordinal number
   ev = new char[strlen("ROOTPROOFORDINAL=")+strlen(xps->Ordinal())+2];
   sprintf(ev, "ROOTPROOFORDINAL=%s", xps->Ordinal());
   putenv(ev);
   MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: "<<ev);

   // ROOT Version tag if not the default one
   ev = new char[strlen("ROOTVERSIONTAG=")+strlen(fPClient->ROOT()->Tag())+2];
   sprintf(ev, "ROOTVERSIONTAG=%s", fPClient->ROOT()->Tag());
   putenv(ev);
   MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: "<<ev);

   // Create the env file
   MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: creating env file");
   XrdOucString envfile = sessdir;
   envfile += ".env";
   FILE *fenv = fopen(envfile.c_str(), "w");
   if (!fenv) {
      MTRACE(XERR, "xpd:child: ",
                  "SetProofServEnvOld: unable to open env file: "<<envfile);
      return -1;
   }
   MTRACE(DBG, "xpd:child: ",
               "SetProofServEnvOld: environment file: "<< envfile);

   // Forwarded sec credentials, if any
   if (fAuthProt) {

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
               MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: "<<ev);
            }
         }
      }

      // The credential buffer, if any
      XrdSecCredentials *creds = fAuthProt->getCredentials();
      if (creds) {
         int lev = strlen("XrdSecCREDS=")+creds->size;
         ev = new char[lev+1];
         strcpy(ev, "XrdSecCREDS=");
         memcpy(ev+strlen("XrdSecCREDS="), creds->buffer, creds->size);
         ev[lev] = 0;
         putenv(ev);
         MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: XrdSecCREDS set");

         // If 'pwd', save AFS key, if any
         if (!strncmp(fAuthProt->Entity.prot, "pwd", 3)) {
            XrdOucString credsdir = udir;
            credsdir += "/.creds";
            // Make sure the directory exists
            if (!XrdProofdAux::AssertDir(credsdir.c_str(), fUI, fgMgr.ChangeOwn())) {
               if (SaveAFSkey(creds, credsdir.c_str()) == 0) {
                  ev = new char[strlen("ROOTPROOFAFSCREDS=")+credsdir.length()+strlen("/.afs")+2];
                  sprintf(ev, "ROOTPROOFAFSCREDS=%s/.afs", credsdir.c_str());
                  putenv(ev);
                  fprintf(fenv, "ROOTPROOFAFSCREDS has been set\n");
                  MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: " << ev);
               } else {
                  MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: problems in saving AFS key");
               }
            } else {
               MTRACE(XERR, "xpd:child: ",
                            "SetProofServEnvOld: unable to create creds dir: "<<credsdir);
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
   fprintf(fenv, "ROOTTMPDIR=%s\n", fgMgr.TMPdir());

   // Port (really needed?)
   fprintf(fenv, "ROOTXPDPORT=%d\n", fgMgr.Port());

   // Work dir
   fprintf(fenv, "ROOTPROOFWORKDIR=%s\n", udir.c_str());

   // Session tag
   fprintf(fenv, "ROOTPROOFSESSIONTAG=%s\n", stag);

   // Whether user specific config files are enabled
   if (fgMgr.WorkerUsrCfg())
      fprintf(fenv, "ROOTUSEUSERCFG=1\n");

   // Set Open socket
   fprintf(fenv, "ROOTOPENSOCK=%s\n", fPClient->UNIXSockPath());

   // Entity
   fprintf(fenv, "ROOTENTITY=%s@%s\n", fClientID, fLink->Host());

   // Session ID
   fprintf(fenv, "ROOTSESSIONID=%d\n", psid);

   // Client ID
   fprintf(fenv, "ROOTCLIENTID=%d\n", fCID);

   // Client Protocol
   fprintf(fenv, "ROOTPROOFCLNTVERS=%d\n", fPClient->Version());

   // Ordinal number
   fprintf(fenv, "ROOTPROOFORDINAL=%s\n", xps->Ordinal());

   // ROOT version tag if different from the default one
   if (getenv("ROOTVERSIONTAG"))
      fprintf(fenv, "ROOTVERSIONTAG=%s\n", getenv("ROOTVERSIONTAG"));

   // Config file
   if (cfg && strlen(cfg) > 0)
      fprintf(fenv, "ROOTPROOFCFGFILE=%s\n", cfg);

   // Log file in the log dir
   XrdOucString logfile = sessdir;
   logfile += ".log";
   fprintf(fenv, "ROOTPROOFLOGFILE=%s\n", logfile.c_str());
   xps->SetFileout(logfile.c_str());

   // Additional envs (xpd.putenv directive)
   if (fgProofServEnvs.length() > 0) {
      // Go through the list
      XrdOucString env;
      int from = 0;
      while ((from = fgProofServEnvs.tokenize(env, from, ',')) != -1) {
         if (env.length() > 0) {
            // Resolve keywords
            fgMgr.ResolveKeywords(env, fPClient);
            // Set the env now
            ev = new char[env.length()+1];
            strncpy(ev, env.c_str(), env.length());
            ev[env.length()] = 0;
            putenv(ev);
            fprintf(fenv, "%s\n", ev);
            MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: "<<ev);
         }
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
            ev = new char[env.length()+1];
            strncpy(ev, env.c_str(), env.length());
            ev[env.length()] = 0;
            putenv(ev);
            fprintf(fenv, "%s\n", ev);
            MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: "<<ev);
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
      MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: "<<ev);
   }

   // Close file
   fclose(fenv);

   // Create or Update symlink to last session
   TRACEI(DBG, "SetProofServEnvOld: creating symlink");
   XrdOucString syml = udir;
   if (fSrvType == kXPD_WorkerServer)
      syml += "/last-worker-session";
   else
      syml += "/last-master-session";
   if (XrdProofdAux::SymLink(logdir.c_str(), syml.c_str()) != 0) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnvOld: problems creating symlink to "
                    " last session (errno: "<<errno<<")");
   }

   // We are done
   MTRACE(DBG, "xpd:child: ", "SetProofServEnvOld: done");
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::SetProofServEnv(XrdROOT *r)
{
   // Set basic environment accordingly to 'r'

   char *ev = 0;

   MTRACE(REQ, "xpd:child: ",
               "SetProofServEnv: enter: ROOT dir: "<< (r ? r->Dir() : "*** undef ***"));

   if (r) {
      char *rootsys = (char *) r->Dir();
#ifndef ROOTLIBDIR
      char *ldpath = 0;
      if (fgMgr.BareLibPath() && strlen(fgMgr.BareLibPath()) > 0) {
         ldpath = new char[32 + strlen(rootsys) + strlen(fgMgr.BareLibPath())];
         sprintf(ldpath, "%s=%s/lib:%s", XPD_LIBPATH, rootsys, fgMgr.BareLibPath());
      } else {
         ldpath = new char[32 + strlen(rootsys)];
         sprintf(ldpath, "%s=%s/lib", XPD_LIBPATH, rootsys);
      }
      putenv(ldpath);
#endif
      // Set ROOTSYS
      ev = new char[15 + strlen(rootsys)];
      sprintf(ev, "ROOTSYS=%s", rootsys);
      putenv(ev);

      // Set conf dir
      ev = new char[20 + strlen(rootsys)];
      sprintf(ev, "ROOTCONFDIR=%s", rootsys);
      putenv(ev);

      // Set TMPDIR
      ev = new char[20 + strlen(fgMgr.TMPdir())];
      sprintf(ev, "TMPDIR=%s", fgMgr.TMPdir());
      putenv(ev);

      // Done
      return 0;
   }

   // Bad input
   MTRACE(REQ,  "xpd:child: ", "SetProofServEnv: XrdROOT instance undefined!");
   return -1;
}

//______________________________________________________________________________
int XrdProofdProtocol::SetProofServEnv(int psid, int loglevel, const char *cfg)
{
   // Set environment for proofserv

   char *ev = 0;

   MTRACE(REQ,  "xpd:child: ", "SetProofServEnv: enter: psid: "<<psid<<
                      ", log: "<<loglevel);

   // Make sure the principal client is defined
   if (!fPClient) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnv: principal client undefined - cannot continue");
      return -1;
   }

   // Old proofservs expect different settings
   if (fPClient->ROOT() && fPClient->ROOT()->SrvProtVers() < 14)
      return SetProofServEnvOld(psid, loglevel, cfg);

   // Session proxy
   XrdProofServProxy *xps = fPClient->ProofServs()->at(psid);
   if (!xps) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnv: unable to get instance of proofserv proxy");
      return -1;
   }

   // Client sandbox
   XrdOucString udir = fPClient->Workdir();
   MTRACE(DBG, "xpd:child: ",
               "SetProofServEnv: sandbox for "<<fClientID<<" is: "<<udir);

   // Create and log into the directory reserved to this session:
   // the unique tag will identify it
   char hn[64], stag[512];
#if defined(XPD__SUNCC)
   sysinfo(SI_HOSTNAME, hn, sizeof(hn));
#else
   gethostname(hn, sizeof(hn));
#endif
   XrdOucString host = hn;
   if (host.find(".") != STR_NPOS)
      host.erase(host.find("."));
   sprintf(stag,"%s-%d-%d",host.c_str(),(int)time(0),getpid());

   // Session dir
   XrdOucString topstag = stag;
   XrdOucString sessiondir = udir;
   if (fSrvType == kXPD_TopMaster) {
      sessiondir += "/session-";
      sessiondir += stag;
      xps->SetTag(stag);
   } else {
      sessiondir += "/";
      sessiondir += xps->Tag();
      topstag = xps->Tag();
      topstag.replace("session-","");
   }
   MTRACE(DBG, "xpd:child: ", "SetProofServEnv: session dir "<<sessiondir);
   // Make sure the directory exists ...
   if (XrdProofdAux::AssertDir(sessiondir.c_str(), fUI, fgMgr.ChangeOwn()) == -1) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnv: unable to create log dir: "<<sessiondir);
      return -1;
   }
   // ... and log into it
   if (XrdProofdAux::ChangeToDir(sessiondir.c_str(), fUI, fgMgr.ChangeOwn()) != 0) {
      MTRACE(XERR, "xpd:child: ", "SetProofServEnv: couldn't change directory to "<<
                   sessiondir);
      return -1;
   }

   // Set basic environment for proofserv
   if (SetProofServEnv(fPClient->ROOT()) != 0) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnv: problems setting basic environment - exit");
      return -1;
   }

   // The session working dir depends on the role
   XrdOucString swrkdir = sessiondir;
   if (fSrvType == kXPD_WorkerServer)
      swrkdir += "/worker-";
   else
      swrkdir += "/master-";
   swrkdir += xps->Ordinal();
   swrkdir += "-";
   swrkdir += stag;

   // Create the rootrc and env files
   MTRACE(DBG, "xpd:child: ", "SetProofServEnv: creating env file");
   XrdOucString rcfile = swrkdir;
   rcfile += ".rootrc";
   FILE *frc = fopen(rcfile.c_str(), "w");
   if (!frc) {
      MTRACE(XERR, "xpd:child: ",
                  "SetProofServEnv: unable to open rootrc file: "<<rcfile);
      return -1;
   }
   // Symlink to session.rootrc
   if (XrdProofdAux::SymLink(rcfile.c_str(), "session.rootrc") != 0) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnv: problems creating symlink to"
                    "'session.rootrc' (errno: "<<errno<<")");
   }
   MTRACE(DBG, "xpd:child: ",
               "SetProofServEnv: session rootrc file: "<< rcfile);

   // Port
   fprintf(frc,"# XrdProofdProtocol listening port\n");
   fprintf(frc, "ProofServ.XpdPort: %d\n", fgMgr.Port());

   // Local root prefix
   if (fgMgr.LocalROOT() && strlen(fgMgr.LocalROOT()) > 0) {
      fprintf(frc,"# Prefix to be prepended to local paths\n");
      fprintf(frc, "Path.Localroot: %s\n", fgMgr.LocalROOT());
   }

   // The session working dir depends on the role
   fprintf(frc,"# The session working dir\n");
   fprintf(frc,"ProofServ.SessionDir: %s\n", swrkdir.c_str());

   // Log / Debug level
   fprintf(frc,"# Proof Log/Debug level\n");
   fprintf(frc,"Proof.DebugLevel: %d\n", loglevel);

   // Ordinal number
   fprintf(frc,"# Ordinal number\n");
   fprintf(frc,"ProofServ.Ordinal: %s\n", xps->Ordinal());

   // ROOT Version tag
   if (fPClient->ROOT()) {
      fprintf(frc,"# ROOT Version tag\n");
      fprintf(frc,"ProofServ.RootVersionTag: %s\n", fPClient->ROOT()->Tag());
   }
   // Proof group
   if (fPClient->Group()) {
      fprintf(frc,"# Proof group\n");
      fprintf(frc,"ProofServ.ProofGroup: %s\n", fPClient->Group()->Name());
   }

   //  Path to file with group information
   if (fgMgr.GroupsMgr() && fgMgr.GroupsMgr()->GetCfgFile()) {
      fprintf(frc,"# File with group information\n");
      fprintf(frc, "Proof.GroupFile: %s\n", fgMgr.GroupsMgr()->GetCfgFile());
   }

   // Work dir
   fprintf(frc,"# Users sandbox\n");
   fprintf(frc, "ProofServ.Sandbox: %s\n", udir.c_str());

   // Image
   if (fgMgr.Image() && strlen(fgMgr.Image()) > 0) {
      fprintf(frc,"# Server image\n");
      fprintf(frc, "ProofServ.Image: %s\n", fgMgr.Image());
   }

   // Session tag
   fprintf(frc,"# Session tag\n");
   fprintf(frc, "ProofServ.SessionTag: %s\n", topstag.c_str());

   // Whether user specific config files are enabled
   if (fgMgr.WorkerUsrCfg()) {
      fprintf(frc,"# Whether user specific config files are enabled\n");
      fprintf(frc, "ProofServ.UseUserCfg: 1\n");
   }
   // Set Open socket
   fprintf(frc,"# Open socket\n");
   fprintf(frc, "ProofServ.OpenSock: %s\n", fPClient->UNIXSockPath());
   // Entity
   fprintf(frc,"# Entity\n");
   if (fGroupID && strlen(fGroupID) > 0)
      fprintf(frc, "ProofServ.Entity: %s:%s@%s\n", fClientID, fGroupID, fLink->Host());
   else
      fprintf(frc, "ProofServ.Entity: %s@%s\n", fClientID, fLink->Host());


   // Session ID
   fprintf(frc,"# Session ID\n");
   fprintf(frc, "ProofServ.SessionID: %d\n", psid);

   // Client ID
   fprintf(frc,"# Client ID\n");
   fprintf(frc, "ProofServ.ClientID: %d\n", fCID);

   // Client Protocol
   fprintf(frc,"# Client Protocol\n");
   fprintf(frc, "ProofServ.ClientVersion: %d\n", fPClient->Version());

   // Config file
   if (cfg && strlen(cfg) > 0) {
      fprintf(frc,"# Config file\n");
      // User defined
      fprintf(frc, "ProofServ.ProofConfFile: %s\n", cfg);
   } else {
      if (fgMgr.IsSuperMst()) {
         fprintf(frc,"# Config file\n");
         fprintf(frc, "ProofServ.ProofConfFile: sm:\n");
      } else if (fgMgr.ProofPlugin() && strlen(fgMgr.ProofPlugin())) {
         fprintf(frc,"# Config file\n");
         fprintf(frc, "ProofServ.ProofConfFile: %s\n", fgMgr.ProofPlugin());
      }
   }

   // Additional rootrcs (xpd.putrc directive)
   if (fgProofServRCs.length() > 0) {
      fprintf(frc,"# Additional rootrcs (xpd.putrc directives)\n");
      // Go through the list
      XrdOucString rc;
      int from = 0;
      while ((from = fgProofServRCs.tokenize(rc, from, ',')) != -1)
         if (rc.length() > 0)
            fprintf(frc, "%s\n", rc.c_str());
   }

   // Done with this
   fclose(frc);

   // Now save the exported env variables, for the record
   XrdOucString envfile = swrkdir;
   envfile += ".env";
   FILE *fenv = fopen(envfile.c_str(), "w");
   if (!fenv) {
      MTRACE(XERR, "xpd:child: ",
                  "SetProofServEnv: unable to open env file: "<<envfile);
      return -1;
   }
   MTRACE(DBG, "xpd:child: ", "SetProofServEnv: environment file: "<< envfile);

   // Forwarded sec credentials, if any
   if (fAuthProt) {

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
               MTRACE(DBG, "xpd:child: ", "SetProofServEnv: "<<ev);
            }
         }
      }

      // The credential buffer, if any
      XrdSecCredentials *creds = fAuthProt->getCredentials();
      if (creds) {
         int lev = strlen("XrdSecCREDS=")+creds->size;
         ev = new char[lev+1];
         strcpy(ev, "XrdSecCREDS=");
         memcpy(ev+strlen("XrdSecCREDS="), creds->buffer, creds->size);
         ev[lev] = 0;
         putenv(ev);
         MTRACE(DBG, "xpd:child: ", "SetProofServEnv: XrdSecCREDS set");

         // If 'pwd', save AFS key, if any
         if (!strncmp(fAuthProt->Entity.prot, "pwd", 3)) {
            XrdOucString credsdir = udir;
            credsdir += "/.creds";
            // Make sure the directory exists
            if (!XrdProofdAux::AssertDir(credsdir.c_str(), fUI, fgMgr.ChangeOwn())) {
               if (SaveAFSkey(creds, credsdir.c_str()) == 0) {
                  ev = new char[strlen("ROOTPROOFAFSCREDS=")+credsdir.length()+strlen("/.afs")+2];
                  sprintf(ev, "ROOTPROOFAFSCREDS=%s/.afs", credsdir.c_str());
                  putenv(ev);
                  fprintf(fenv, "ROOTPROOFAFSCREDS has been set\n");
                  MTRACE(DBG, "xpd:child: ", "SetProofServEnv: " << ev);
               } else {
                  MTRACE(DBG, "xpd:child: ", "SetProofServEnv: problems in saving AFS key");
               }
            } else {
               MTRACE(XERR, "xpd:child: ",
                            "SetProofServEnv: unable to create creds dir: "<<credsdir);
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
   fprintf(fenv, "TMPDIR=%s\n", fgMgr.TMPdir());

   // ROOT version tag (needed in building packages)
   ev = new char[strlen("ROOTVERSIONTAG=")+strlen(fPClient->ROOT()->Tag())+2];
   sprintf(ev, "ROOTVERSIONTAG=%s", fPClient->ROOT()->Tag());
   putenv(ev);
   fprintf(fenv, "%s\n", ev);

   // Log file in the log dir
   XrdOucString logfile = swrkdir;
   logfile += ".log";
   ev = new char[strlen("ROOTPROOFLOGFILE=")+logfile.length()+2];
   sprintf(ev, "ROOTPROOFLOGFILE=%s", logfile.c_str());
   putenv(ev);
   fprintf(fenv, "%s\n", ev);
   xps->SetFileout(logfile.c_str());

   // Xrootd config file
   ev = new char[strlen("XRDCF=")+fgCfgFile.fName.length()+2];
   sprintf(ev, "XRDCF=%s", fgCfgFile.fName.c_str());
   putenv(ev);
   fprintf(fenv, "%s\n", ev);

   // Additional envs (xpd.putenv directive)
   if (fgProofServEnvs.length() > 0) {
      // Go through the list
      XrdOucString env;
      int from = 0;
      while ((from = fgProofServEnvs.tokenize(env, from, ',')) != -1) {
         if (env.length() > 0) {
            // Resolve keywords
            fgMgr.ResolveKeywords(env, fPClient);
            // Set the env now
            ev = new char[env.length()+1];
            strncpy(ev, env.c_str(), env.length());
            ev[env.length()] = 0;
            putenv(ev);
            fprintf(fenv, "%s\n", ev);
            MTRACE(DBG, "xpd:child: ", "SetProofServEnv: "<<ev);
         }
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
            ev = new char[env.length()+1];
            strncpy(ev, env.c_str(), env.length());
            ev[env.length()] = 0;
            putenv(ev);
            fprintf(fenv, "%s\n", ev);
            MTRACE(DBG, "xpd:child: ", "SetProofServEnv: "<<ev);
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
      MTRACE(DBG, "xpd:child: ", "SetProofServEnv: "<<ev);
   }

   // Close file
   fclose(fenv);

   // Create or Update symlink to last session
   TRACEI(DBG, "SetProofServEnv: creating symlink");
   XrdOucString syml = udir;
   if (fSrvType == kXPD_WorkerServer)
      syml += "/last-worker-session";
   else
      syml += "/last-master-session";
   if (XrdProofdAux::SymLink(sessiondir.c_str(), syml.c_str()) != 0) {
      MTRACE(XERR, "xpd:child: ",
                   "SetProofServEnv: problems creating symlink to "
                    " last session (errno: "<<errno<<")");
   }

   // We are done
   MTRACE(DBG, "xpd:child: ", "SetProofServEnv: done");
   return 0;
}

//_________________________________________________________________________________
int XrdProofdProtocol::Create()
{
   // Handle a request to create a new session

   int psid = -1, rc = 1;

   TRACEI(REQ, "Create: enter");
   XrdSysMutexHelper mh(fPClient->Mutex());

   // Allocate next free server ID and fill in the basic stuff
   psid = fPClient->GetFreeServID();
   XrdProofServProxy *xps = fPClient->ProofServs()->at(psid);
   xps->SetClient((const char *)fClientID);
   xps->SetID(psid);
   xps->SetSrvType(fSrvType);

   // Prepare the stream identifier
   unsigned short sid;
   memcpy((void *)&sid, (const void *)&(fRequest.header.streamid[0]), 2);
   // We associate this instance to the corresponding slot in the
   // session vector of attached clients
   XrdClientID *csid = xps->GetClientID(fCID);
   csid->SetP(this);
   csid->SetSid(sid);
   // Take parentship, if orphalin
   xps->SetParent(csid);

   // Unmarshall log level
   int loglevel = ntohl(fRequest.proof.int1);

   // Parse buffer
   char *buf = fArgp->buff;
   int   len = fRequest.proof.dlen;

   // Extract session tag
   XrdOucString tag(buf,len);

   TRACEI(DBG, "Create: received buf: "<<tag);

   tag.erase(tag.find('|'));
   xps->SetTag(tag.c_str());
   TRACEI(DBG, "Create: tag: "<<tag);

   // Extract ordinal number
   XrdOucString ord = "0";
   if ((fSrvType == kXPD_WorkerServer) || (fSrvType == kXPD_MasterServer)) {
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
   XrdOucString cffile;
   cffile.assign(buf,0,len-1);
   int icf = cffile.find("|cf:");
   if (icf != STR_NPOS) {
      cffile.erase(0,icf+4);
      cffile.erase(cffile.find("|"));
   } else
      cffile = "";

   // Extract user envs, if any
   XrdOucString uenvs;
   uenvs.assign(buf,0,len-1);
   int ienv = uenvs.find("|envs:");
   if (ienv != STR_NPOS) {
      uenvs.erase(0,ienv+6);
      uenvs.erase(uenvs.find("|"));
      xps->SetUserEnvs(uenvs.c_str());
   } else
      uenvs = "";

   // The ROOT version to be used
   xps->SetROOT(fPClient->ROOT());
   XPDPRT("Create: using ROOT version: "<<xps->ROOT()->Export());
   if (fSrvType == kXPD_TopMaster) {
      // Notify the client if using a version different from the default one
      if (fPClient->ROOT() != fgMgr.ROOT()->front()) {
         XrdOucString msg("++++ Using NON-default ROOT version: ");
         msg += xps->ROOT()->Export();
         msg += " ++++\n";
         fResponse.Send(kXR_attn, kXPD_srvmsg, (char *) msg.c_str(), msg.length());
      }
   }

   // Notify
   TRACEI(DBG, "Create: {ord,cfg,psid,cid,log}: {"<<ord<<","<<cffile<<","<<psid
                                                  <<","<<fCID<<","<<loglevel<<"}");
   if (uenvs.length() > 0)
      TRACEI(DBG, "Create: user envs: "<<uenvs);

   // Here we fork: for some weird problem on SMP machines there is a
   // non-zero probability for a deadlock situation in system mutexes.
   // The semaphore seems to have solved the problem.
   if (fgForkSem.Wait(10) != 0) {
      xps->Reset();
      // Timeout acquire fork semaphore
      fResponse.Send(kXR_ServerError, "timed-out acquiring fork semaphore");
      return rc;
   }

   // Pipe to communicate status of setup
   int fp[2];
   if (pipe(fp) != 0) {
      xps->Reset();
      // Failure creating pipe
      fResponse.Send(kXR_ServerError,
                     "unable to create pipe for status-of-setup communication");
      return rc;
   }

   // Fork an agent process to handle this session
   int pid = -1;
   TRACEI(FORK,"Forking external proofsrv: UNIX sock: "<<fPClient->UNIXSockPath());
   if (!(pid = fgSched->Fork("proofsrv"))) {

      fPClient->Mutex()->UnLock();

      int setupOK = 0;

      XrdOucString pmsg = "child process ";
      pmsg += (int) getpid();
      MTRACE(FORK, "xpd: ", pmsg.c_str());

      // We set to the user environment
      if (SetUserEnvironment() != 0) {
         MTRACE(XERR, "xpd:child: ",
                      "Create: SetUserEnvironment did not return OK - EXIT");
         write(fp[1], &setupOK, sizeof(setupOK));
         close(fp[0]);
         close(fp[1]);
         exit(1);
      }

      char *argvv[6] = {0};

      // We add our PID to be able to identify processes coming from us
      char cpid[10] = {0};
      sprintf(cpid, "%d", getppid());

      // Log level
      char clog[10] = {0};
      sprintf(clog, "%d", loglevel);

      // start server
      argvv[0] = (char *) xps->ROOT()->PrgmSrv();
      argvv[1] = (char *)((fSrvType == kXPD_WorkerServer) ? "proofslave"
                       : "proofserv");
      argvv[2] = (char *)"xpd";
      argvv[3] = (char *)cpid;
      argvv[4] = (char *)clog;
      argvv[5] = 0;

      // Set environment for proofserv
      if (SetProofServEnv(psid, loglevel, cffile.c_str()) != 0) {
         MTRACE(XERR, "xpd:child: ",
                      "Create: SetProofServEnv did not return OK - EXIT");
         write(fp[1], &setupOK, sizeof(setupOK));
         close(fp[0]);
         close(fp[1]);
         exit(1);
      }

      // Setup OK: now we go
      // Communicate the logfile path
      int lfout = strlen(xps->Fileout());
      write(fp[1], &lfout, sizeof(lfout));
      if (lfout > 0) {
         int n, ns = 0;
         char *b = (char *) xps->Fileout();
         for (n = 0; n < lfout; n += ns) {
            if ((ns = write(fp[1], b + n, lfout - n)) <= 0) {
               MTRACE(XERR, "xpd:child: ",
                            "Create: SetProofServEnv did not return OK - EXIT");
               write(fp[1], &setupOK, sizeof(setupOK));
               close(fp[0]);
               close(fp[1]);
               exit(1);
            }
         }
      }

      // Cleanup
      close(fp[0]);
      close(fp[1]);

      MTRACE(LOGIN,"xpd:child: ", "Create: fClientID: "<<fClientID<<
                         ", uid: "<<getuid()<<", euid:"<<geteuid());
      // Run the program
      execv(xps->ROOT()->PrgmSrv(), argvv);

      // We should not be here!!!
      MERROR("xpd:child: ", "Create: returned from execv: bad, bad sign !!!");
      exit(1);
   }

   TRACEP(FORK,"Parent process: child is "<<pid);

   // Wakeup colleagues
   fgForkSem.Post();

   // parent process
   if (pid < 0) {
      xps->Reset();
      // Failure in forking
      fResponse.Send(kXR_ServerError, "could not fork agent");
      close(fp[0]);
      close(fp[1]);
      return rc;
   }

   // Read status-of-setup from pipe
   XrdOucString emsg;
   int setupOK = 0;
   struct pollfd fds_r;
   fds_r.fd = fp[0];
   fds_r.events = POLLIN;
   int pollRet = 0;
   // We wait for 14 secs max (7 x 2000 millisecs): this is enough to
   // cover possible delays due to heavy load; the client will anyhow
   // retry a few times
   int ntry = 7;
   while (pollRet == 0 && ntry--) {
      while ((pollRet = poll(&fds_r, 1, 2000)) < 0 &&
             (errno == EINTR)) { }
      if (pollRet == 0)
         TRACE(FORK,"Create: "
                    "receiving status-of-setup from pipe: waiting 2 s ..."<<pid);
   }
   if (pollRet > 0) {
      if (read(fp[0], &setupOK, sizeof(setupOK)) == sizeof(setupOK)) {
         // now we wait for the callback to be (successfully) established
         if (setupOK > 0) {
            // Receive path of the log file
            int lfout = setupOK;
            char *b = new char[lfout + 1];
            int n, nr = 0;
            for (n = 0; n < lfout; n += nr) {
               while ((nr = read(fp[0], b + n, lfout - n)) == -1 && errno == EINTR)
                  errno = 0;   // probably a SIGCLD that was caught
               if (nr == 0)
                  break;          // EOF
               if (nr < 0) {
                  // Failure
                  setupOK= -1;
                  emsg += ": failure receiving logfile path";
                  break;
               }
            }
            if (setupOK > 0) {
               b[lfout] = 0;
               xps->SetFileout(b);
               // Set also the session tag
               XrdOucString stag(b);
               stag.erase(stag.rfind('/'));
               stag.erase(0, stag.find("session-") + strlen("session-"));
               xps->SetTag(stag.c_str());
            }
            delete[] b;
         } else {
            emsg += ": proofserv startup failed";
         }
      } else {
         emsg += ": problems receiving status-of-setup after forking";
      }
   } else {
      if (pollRet == 0) {
         emsg += ": timed-out receiving status-of-setup from pipe";
      } else {
         emsg += ": failed to receive status-of-setup from pipe";
      }
   }

   // Cleanup
   close(fp[0]);
   close(fp[1]);

   // Notify to user
   if (setupOK > 0) {
      if (fSrvType == kXPD_TopMaster) {
         // Send also back the data pool url
         XrdOucString dpu = fgMgr.PoolURL();
         if (!dpu.endswith('/'))
            dpu += '/';
         dpu += fgMgr.NameSpace();
         fResponse.Send(psid, xps->ROOT()->SrvProtVers(), (kXR_int16)XPROOFD_VERSBIN,
                       (void *) dpu.c_str(), dpu.length());
      } else
         fResponse.Send(psid, xps->ROOT()->SrvProtVers(), (kXR_int16)XPROOFD_VERSBIN);
   } else {
      // Failure
      emsg += ": failure setting up proofserv" ;
      xps->Reset();
      KillProofServ(pid, 1);
      fResponse.Send(kXR_ServerError, emsg.c_str());
      return rc;
   }
   // UNIX Socket is saved now
   fPClient->SetUNIXSockSaved();

   // now we wait for the callback to be (successfully) established
   TRACEP(FORK, "Create: server launched: wait for callback ");

   // We will get back a peer to initialize a link
   XrdNetPeer peerpsrv;
   XrdLink   *linkpsrv = 0;
   int lnkopts = 0;

   // Perform regular accept
   if (!(fPClient->UNIXSock()->Accept(peerpsrv, XRDNET_NODNTRIM, fgInternalWait))) {

      // We need the right privileges to do this
      XrdOucString msg("did not receive callback: ");
      if (KillProofServ(pid, 1) != 0)
         msg += "process could not be killed";
      else
         msg += "process killed";
      fResponse.Send(kXR_attn, kXPD_errmsg, (char *) msg.c_str(), msg.length());

      xps->Reset();
      return rc;
   }
   // Make sure we have the full host name
   if (peerpsrv.InetName) {
      char *ptmp = peerpsrv.InetName;
      peerpsrv.InetName = XrdNetDNS::getHostName("localhost");
      free(ptmp);
   }

   // Allocate a new network object
   if (!(linkpsrv = XrdLink::Alloc(peerpsrv, lnkopts))) {

      // We need the right privileges to do this
      XrdOucString msg("could not allocate network object: ");
      if (KillProofServ(pid, 0) != 0)
         msg += "process could not be killed";
      else
         msg += "process killed";
      fResponse.Send(kXR_attn, kXPD_errmsg, (char *) msg.c_str(), msg.length());

      xps->Reset();
      return rc;

   } else {

      // Keep buffer after object goes away
      peerpsrv.InetBuff = 0;
      TRACEP(DBG, "Accepted connection from " << peerpsrv.InetName);

      // Get a protocol object off the stack (if none, allocate a new one)
      XrdProtocol *xp = Match(linkpsrv);
      if (!xp) {

         // We need the right privileges to do this
         XrdOucString msg("match failed: protocol error: ");
         if (KillProofServ(pid, 0) != 0)
            msg += "process could not be killed";
         else
            msg += "process killed";
         fResponse.Send(kXR_attn, kXPD_errmsg, (char *) msg.c_str(), msg.length());

         linkpsrv->Close();
         xps->Reset();
         return rc;
      }

      // Take a short-cut and process the initial request as a sticky request
      xp->Process(linkpsrv);
      if (xp->Process(linkpsrv) != 0) {
         // We need the right privileges to do this
         XrdOucString msg("handshake with internal link failed: ");
         if (KillProofServ(pid, 0) != 0)
            msg += "process could not be killed";
         else
            msg += "process killed";
         fResponse.Send(kXR_attn, kXPD_errmsg, (char *) msg.c_str(), msg.length());

         linkpsrv->Close();
         xps->Reset();
         return rc;
      }

      // Attach this link to the appropriate poller and enable it.
      if (!XrdPoll::Attach(linkpsrv)) {

         // We need the right privileges to do this
         XrdOucString msg("could not attach new internal link to poller: ");
         if (KillProofServ(pid, 0) != 0)
            msg += "process could not be killed";
         else
            msg += "process killed";
         fResponse.Send(kXR_attn, kXPD_errmsg, (char *) msg.c_str(), msg.length());

         linkpsrv->Close();
         xps->Reset();
         return rc;
      }

      // Tight this protocol instance to the link
      linkpsrv->setProtocol(xp);

      // Schedule it
      fgSched->Schedule((XrdJob *)linkpsrv);
   }

   // Set ID
   xps->SetSrv(pid);

   // Set the group, if any
   xps->SetGroup(fPClient->Group());

   // Change child process priority, if required
   if (fgMgr.Priorities()->size() > 0) {
      XrdOucString usr(fClientID);
      int dp = 0;
      int nmmx = -1;
      std::list<XrdProofdPriority *>::iterator i;
      for (i = fgMgr.Priorities()->begin(); i != fgMgr.Priorities()->end(); ++i) {
         int nm = usr.matches((*i)->fUser.c_str());
         if (nm >= nmmx) {
            nmmx = nm;
            dp = (*i)->fDeltaPriority;
         }
      }
      if (nmmx > -1) {
         // Changing child process priority for this user
         int newp = xps->GetDefaultProcessPriority() + dp;
         if (xps->SetProcessPriority(newp) != 0) {
            TRACEI(XERR, "Create: problems changing child process priority");
         } else {
            TRACEI(DBG, "Create: priority of the child process changed by "
                        << dp << " units");
         }
      }
   }
   XrdClientID *cid = xps->Parent();
   TRACEI(DBG, "Create: xps: "<<xps<<", ClientID: "<<(int *)cid<<" (sid: "<<sid<<")");

   // Record this session in the sandbox
   if (fSrvType != kXPD_Internal) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, fUI.fUid)) {
         TRACEI(REQ, "Create: could not get privileges to run AddNewSession");
      } else {
         if (fPClient->AddNewSession(xps->Tag()) == -1)
            TRACEI(REQ, "Create: problems recording session in sandbox");
      }
   }

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::SendData(XrdProofdResponse *resp,
                                kXR_int32 sid, XrdSrvBuffer **buf)
{
   // Send data over the open link. Segmentation is done here, if required.

   int rc = 1;

   TRACEI(ACT, "SendData: enter: length: "<<fRequest.header.dlen<<" bytes ");

   // Buffer length
   int len = fRequest.header.dlen;

   // Quantum size
   int quantum = (len > fgMaxBuffsz ? fgMaxBuffsz : len);

   // Make sure we have a large enough buffer
   if (!fArgp || quantum < fhalfBSize || quantum > fArgp->bsize) {
      if ((rc = GetBuff(quantum)) <= 0)
         return rc;
   } else if (fhcNow < fhcNext)
      fhcNow++;

   // Now send over all of the data as unsolicited messages
   while (len > 0) {
      if ((rc = GetData("data", fArgp->buff, quantum)))
         return rc;
      if (buf && !(*buf))
         *buf = new XrdSrvBuffer(fArgp->buff, quantum, 1);
      // Send
      if (sid > -1) {
         if (resp->Send(kXR_attn, kXPD_msgsid, sid, fArgp->buff, quantum))
            return 1;
      } else {
         if (resp->Send(kXR_attn, kXPD_msg, fArgp->buff, quantum))
            return 1;
      }
      // Next segment
      len -= quantum;
      if (len < quantum)
         quantum = len;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::SendDataN(XrdProofServProxy *xps,
                                 XrdSrvBuffer **buf)
{
   // Send data over the open client links of session 'xps'.
   // Used when all the connected clients are eligible to receive the message.
   // Segmentation is done here, if required.

   int rc = 1;

   TRACEI(ACT, "SendDataN: enter: length: "<<fRequest.header.dlen<<" bytes ");

   // Buffer length
   int len = fRequest.header.dlen;

   // Quantum size
   int quantum = (len > fgMaxBuffsz ? fgMaxBuffsz : len);

   // Make sure we have a large enough buffer
   if (!fArgp || quantum < fhalfBSize || quantum > fArgp->bsize) {
      if ((rc = GetBuff(quantum)) <= 0)
         return rc;
   } else if (fhcNow < fhcNext)
      fhcNow++;

   // Now send over all of the data as unsolicited messages
   while (len > 0) {
      if ((rc = GetData("data", fArgp->buff, quantum)))
         return rc;
      if (buf && !(*buf))
         *buf = new XrdSrvBuffer(fArgp->buff, quantum, 1);
      // Broadcast
      XrdClientID *csid = 0;
      int ic = 0;
      for (ic = 0; ic < (int) xps->Clients()->size(); ic++) {
         if ((csid = xps->Clients()->at(ic)) && csid->P()) {
            XrdProofdResponse& resp = csid->P()->fResponse;
            int rs = 0;
            {  XrdSysMutexHelper mhp(resp.fMutex);
               unsigned short sid;
               resp.GetSID(sid);
               TRACEI(HDBG, "SendDataN: INTERNAL: this sid: "<<sid<<
                            "; client sid:"<<csid->Sid());
               resp.Set(csid->Sid());
               rs = resp.Send(kXR_attn, kXPD_msg, fArgp->buff, quantum);
               resp.Set(sid);
            }
            if (rs)
               return 1;
         }
      }

      // Next segment
      len -= quantum;
      if (len < quantum)
         quantum = len;
   }

   // Done
   return 0;
}

//_____________________________________________________________________________
int XrdProofdProtocol::SendMsg()
{
   // Handle a request to forward a message to another process

   static const char *crecv[4] = {"master proofserv", "top master",
                                  "client", "undefined"};
   int rc = 1;
   if (!fPClient) {
      TRACEP(XERR, "SendMsg: client undefined!!! ");
      fResponse.Send(kXR_InvalidRequest,"SendMsg: client undefined!!! ");
      return rc;
   }

   XrdSysMutexHelper mhc(fPClient->Mutex());
   XrdSysMutexHelper mh(fResponse.fMutex);

   // Unmarshall the data
   int psid = ntohl(fRequest.sendrcv.sid);
   int opt = ntohl(fRequest.sendrcv.opt);
   bool external = !(opt & kXPD_internal);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
       !(xps = fPClient->ProofServs()->at(psid))) {
      TRACEP(XERR, "SendMsg: session ID not found: "<< psid);
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
      return rc;
   }

   // Forward message as unsolicited
   int len = fRequest.header.dlen;

   // Notify
   TRACEP(DBG, "SendMsg: psid: "<<psid<<", xps: "<<xps<<", status: "<<xps->Status()<<
               ", cid: "<<fCID);

   if (external) {

      if (opt & kXPD_process) {
         TRACEP(DBG, "SendMsg: EXT: setting proofserv in 'running' state");
         xps->SetStatus(kXPD_running);
         // Update global list of active sessions
         fgMgr.AddActiveSession(xps);
         // Update counters in client instance
         fPClient->CountSession(1, (xps->SrvType() == kXPD_WorkerServer));
         // Notify
         TRACE(SCHED, fPClient->ID()<<": kXPD_process: act w: "<<
                      fPClient->WorkerProofServ() <<": act m: "<<
                      fPClient->MasterProofServ())
         // Update group info, if any
         XrdSysMutexHelper mtxh(&fgXPDMutex);
         if (fPClient->Group())
            fPClient->Group()->Count((const char *) fPClient->ID());
         // Updated priorities to the active sessions
         fgMgr.UpdatePriorities(1);
      }

      // Send to proofsrv our client ID
      if (fCID == -1) {
         fResponse.Send(kXR_ServerError,"EXT: getting clientSID");
         return rc;
      }
      if (SendData(xps->ProofSrv(), fCID)) {
         fResponse.Send(kXR_ServerError,"EXT: sending message to proofserv");
         return rc;
      }
      // Notify to user
      fResponse.Send();
      TRACEP(DBG, "SendMsg: EXT: message sent to proofserv ("<<len<<" bytes)");

   } else {

      bool saveStartMsg = 0;
      XrdSrvBuffer *savedBuf = 0;
      // Additional info about the message
      if (opt & kXPD_setidle) {
         TRACEP(DBG, "SendMsg: INT: setting proofserv in 'idle' state");
         xps->SetStatus(kXPD_idle);
         xps->SetSchedRoundRobin(0);
         // Update global list of active sessions
         fgMgr.RemoveActiveSession(xps);
         // Clean start processing message, if any
         xps->DeleteStartMsg();
         // Update counters in client instance
         fPClient->CountSession(-1, (xps->SrvType() == kXPD_WorkerServer));
         // Notify
         TRACE(SCHED, fPClient->ID()<<": kXPD_setidle: act w: "<<
                      fPClient->WorkerProofServ()<<": act m: "<<
                      fPClient->MasterProofServ())
         // Update group info, if any
         XrdSysMutexHelper mtxh(&fgXPDMutex);
         if (fPClient->Group()) {
            if (!(fPClient->WorkerProofServ()+fPClient->MasterProofServ()))
               fPClient->Group()->Count((const char *) fPClient->ID(), -1);
         }
         // Updated priorities to the active sessions
         fgMgr.UpdatePriorities(1);

      } else if (opt & kXPD_querynum) {
         TRACEI(DBG, "SendMsg: INT: got message with query number");
         // Save query num message for later clients
         savedBuf = xps->QueryNum();
      } else if (opt & kXPD_startprocess) {
         TRACEI(DBG, "SendMsg: INT: setting proofserv in 'running' state");
         xps->SetStatus(kXPD_running);
         // Save start processing message for later clients
         xps->DeleteStartMsg();
         saveStartMsg = 1;
      } else if (opt & kXPD_logmsg) {
         // We broadcast log messages only not idle to catch the
         // result from processing
         if (xps->Status() == kXPD_running) {
            TRACEI(DBG, "SendMsg: INT: broadcasting log message");
            opt |= kXPD_fb_prog;
         }
      }
      bool fbprog = (opt & kXPD_fb_prog);

      if (!fbprog) {
         // Get ID of the client
         int cid = ntohl(fRequest.sendrcv.cid);
         TRACEP(DBG, "SendMsg: INT: client ID: "<<cid);

         // Get corresponding instance
         XrdClientID *csid = 0;
         if (!xps || !INRANGE(cid, xps->Clients()) ||
             !(csid = xps->Clients()->at(cid))) {
            TRACEP(XERR, "SendMsg: INT: client ID not found (cid: "<<cid<<
                        ", size: "<<xps->Clients()->size()<<")");
            fResponse.Send(kXR_InvalidRequest,"Client ID not found");
            return rc;
         }
         if (!csid || !(csid->P())) {
            TRACEP(XERR, "SendMsg: INT: client not connected: csid: "<<csid<<
                        ", cid: "<<cid<<", fSid: " << csid->Sid());
            // Notify to proofsrv
            fResponse.Send();
            return rc;
         }

         //
         // The message is strictly for the client requiring it
         int rs = 0;
         {  XrdSysMutexHelper mhp(csid->P()->fResponse.fMutex);
            unsigned short sid;
            csid->P()->fResponse.GetSID(sid);
            TRACEP(DBG, "SendMsg: INT: this sid: "<<sid<<
                        ", client sid: "<<csid->Sid());
            csid->P()->fResponse.Set(csid->Sid());
            rs = SendData(&(csid->P()->fResponse), -1, &savedBuf);
            csid->P()->fResponse.Set(sid);
         }
         if (rs) {
            fResponse.Send(kXR_ServerError,
                           "SendMsg: INT: sending message to client"
                           " or master proofserv");
            return rc;
         }
      } else {
         // Send to all connected clients
         if (SendDataN(xps, &savedBuf)) {
            fResponse.Send(kXR_ServerError,
                           "SendMsg: INT: sending message to client"
                           " or master proofserv");
            return rc;
         }
      }
      // Save start processing messages, if required
      if (saveStartMsg)
         xps->SetStartMsg(savedBuf);

      TRACEP(DBG, "SendMsg: INT: message sent to "<<crecv[xps->SrvType()]<<
                  " ("<<len<<" bytes)");
      // Notify to proofsrv
      fResponse.Send();
   }

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Urgent()
{
   // Handle generic request of a urgent message to be forwarded to the server
   unsigned int rc = 1;

   // Unmarshall the data
   int psid = ntohl(fRequest.proof.sid);
   int type = ntohl(fRequest.proof.int1);
   int int1 = ntohl(fRequest.proof.int2);
   int int2 = ntohl(fRequest.proof.int3);

   TRACEP(REQ, "Urgent: enter: psid: "<<psid<<", type: "<< type);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
       !(xps = fPClient->ProofServs()->at(psid))) {
      TRACEP(XERR, "Urgent: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"Urgent: session ID not found");
      return rc;
   }

   TRACEP(DBG, "Urgent: xps: "<<xps<<", status: "<<xps->Status());

   // Check ID matching
   if (!xps->Match(psid)) {
      fResponse.Send(kXP_InvalidRequest,"Urgent: IDs do not match - do nothing");
      return rc;
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
   if (xps->ProofSrv()->Send(kXR_attn, kXPD_urgent, buf, len) != 0) {
      fResponse.Send(kXP_ServerError,
                     "Urgent: could not propagate request to proofsrv");
      return rc;
   }

   // Notify to user
   fResponse.Send();
   TRACEP(DBG, "Urgent: request propagated to proofsrv");

   // Over
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Admin()
{
   // Handle generic request of administrative type

   int rc = 1;

   XrdSysMutexHelper mhc(fPClient->Mutex());

   // Unmarshall the data
   //
   int psid = ntohl(fRequest.proof.sid);
   int type = ntohl(fRequest.proof.int1);

   TRACEI(REQ, "Admin: enter: type: "<<type<<", psid: "<<psid);

   if (type == kQuerySessions) {

      XrdProofServProxy *xps = 0;
      int ns = 0;
      std::vector<XrdProofServProxy *>::iterator ip;
      for (ip = fPClient->ProofServs()->begin(); ip != fPClient->ProofServs()->end(); ++ip)
         if ((xps = *ip) && xps->IsValid() && (xps->SrvType() == kXPD_TopMaster)) {
            ns++;
            TRACEI(XERR, "Admin: found: " << xps << "(" << xps->IsValid() <<")");
         }

      // Generic info about all known sessions
      int len = (kXPROOFSRVTAGMAX+kXPROOFSRVALIASMAX+30)* (ns+1);
      char *buf = new char[len];
      if (!buf) {
         TRACEP(XERR, "Admin: no resources for results");
         fResponse.Send(kXR_NoMemory, "Admin: out-of-resources for results");
         return rc;
      }
      sprintf(buf, "%d", ns);

      xps = 0;
      for (ip = fPClient->ProofServs()->begin(); ip != fPClient->ProofServs()->end(); ++ip) {
         if ((xps = *ip) && xps->IsValid() && (xps->SrvType() == kXPD_TopMaster)) {
            sprintf(buf,"%s | %d %s %s %d %d",
                    buf, xps->ID(), xps->Tag(), xps->Alias(),
                    xps->Status(), xps->GetNClients());
         }
      }
      TRACEP(DBG, "Admin: sending: "<<buf);

      // Send back to user
      fResponse.Send(buf,strlen(buf)+1);
      if (buf) delete[] buf;

   } else if (type == kQueryLogPaths) {

      int ridx = ntohl(fRequest.proof.int2);

      // Find out for which session is this request
      XrdOucString stag, master, user, buf;
      int len = fRequest.header.dlen;
      if (len > 0) {
         buf.assign(fArgp->buff,0,len-1);
         int im = buf.find("|master:");
         int iu = buf.find("|user:");
         stag = buf;
         stag.erase(stag.find("|"));
         if (im != STR_NPOS) {
            master.assign(buf, im + strlen("|master:"));
            master.erase(master.find("|"));
            TRACEP(DBG,"Admin: master: "<<master);
         }
         if (iu != STR_NPOS) {
            user.assign(buf, iu + strlen("|user:"));
            user.erase(user.find("|"));
            TRACEP(DBG,"Admin: user: "<<user);
         }
         if (stag.beginswith('*'))
            stag = "";
      }

      XrdProofdClient *client = (user.length() > 0) ? 0 : fPClient;
      if (!client) {
         // Find the client instance
         std::list<XrdProofdClient *>::iterator i;
         for (i = fgMgr.ProofdClients()->begin(); i != fgMgr.ProofdClients()->end(); ++i) {
            if ((client = *i) && client->Match(user.c_str(),0)) {
               break;
            }
            client = 0;
         }
      }
      if (!client) {
         TRACEP(XERR, "Admin: query sess logs: client for '"<<user<<"' not found");
         fResponse.Send(kXR_InvalidRequest,"Admin: query log: client not found");
         return rc;
      }

      XrdOucString tag = (stag == "" && ridx >= 0) ? "last" : stag;
      if (stag == "" && client->GuessTag(tag, ridx) != 0) {
         TRACEP(XERR, "Admin: query sess logs: session tag not found");
         fResponse.Send(kXR_InvalidRequest,"Admin: query log: session tag not found");
         return rc;
      }

      // Return message
      XrdOucString rmsg;

      if (master.length() <= 0) {
         // The session tag first
         rmsg += tag; rmsg += "|";
         // The pool URL second
         rmsg += fgMgr.PoolURL(); rmsg += "|";
      }

      // Locate the local log file
      XrdOucString sdir(client->Workdir());
      sdir += "/session-";
      sdir += tag;

      // Open dir
      DIR *dir = opendir(sdir.c_str());
      if (!dir) {
         XrdOucString msg("Admin: cannot open dir ");
         msg += sdir; msg += " (errno: "; msg += errno; msg += ")";
         TRACEP(XERR, msg.c_str());
         fResponse.Send(kXR_InvalidRequest, msg.c_str());
         return rc;
      }
      // Scan the directory to add the top master (only if top master)
      if (master.length() <= 0) {
         bool found = 0;
         struct dirent *ent = 0;
         while ((ent = (struct dirent *)readdir(dir))) {
            if (!strncmp(ent->d_name, "master-", 7) &&
               strstr(ent->d_name, ".log")) {
               rmsg += "|0 proof://"; rmsg += fgMgr.Host(); rmsg += ':';
               rmsg += fgMgr.Port(); rmsg += '/';
               rmsg += sdir; rmsg += '/'; rmsg += ent->d_name;
               found = 1;
               break;
            }
         }
      }
      // Close dir
      closedir(dir);

      // Now open the workers file
      XrdOucString wfile(sdir);
      wfile += "/.workers";
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
                     // If the line is for a submaster, we have to get the info
                     // about its workers
                     bool ismst = (strstr(pp, "master-")) ? 1 : 0;
                     if (ismst) {
                        XrdOucString msg(stag);
                        msg += "|master:";
                        msg += ln;
                        msg += "|user:";
                        msg += XrdClientUrlInfo(ln).User;
                        char *bmst = ReadLogPaths((const char *)&ln[0], msg.c_str(), ridx);
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

      // Send back to user
      fResponse.Send((void *) rmsg.c_str(), rmsg.length()+1);

   } else if (type == kCleanupSessions) {

      XrdOucString cmsg;

      // Target client (default us)
      XrdProofdClient *tgtclnt = fPClient;

      // Server type to clean
      int srvtype = ntohl(fRequest.proof.int2);

      // If super user we may be requested to cleanup everything
      bool all = 0;
      char *usr = 0;
      bool clntfound = 1;
      if (fSuperUser) {
         int what = ntohl(fRequest.proof.int2);
         all = (what == 1) ? 1 : 0;

         if (!all) {
            // Get a user name, if any.
            // A super user can ask cleaning for clients different from itself
            char *buf = 0;
            int len = fRequest.header.dlen;
            if (len > 0) {
               clntfound = 0;
               buf = fArgp->buff;
               len = (len < 9) ? len : 8;
            } else {
               buf = fClientID;
               len = strlen(fClientID);
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
               XrdProofdClient *c = 0;
               std::list<XrdProofdClient *>::iterator i;
               for (i = fgMgr.ProofdClients()->begin(); i != fgMgr.ProofdClients()->end(); ++i) {
                  if ((c = *i) && c->Match(usr,grp)) {
                     tgtclnt = c;
                     clntfound = 1;
                     break;
                  }
               }
               TRACEI(ADMIN, "Admin: CleanupSessions: superuser, cleaning usr: "<< usr);
            }
         } else {
            TRACEI(ADMIN, "Admin: CleanupSessions: superuser, all sessions cleaned");
         }
      } else {
         // Define the user name for later transactions (their executed under
         // the admin name)
         int len = strlen(tgtclnt->ID()) + 1;
         usr = new char[len+1];
         memcpy(usr, tgtclnt->ID(), len);
         usr[len] = '\0';
      }

      // We cannot continue if we do not have anything to clean
      if (!clntfound) {
         TRACEI(ADMIN, "Admin: client '"<<usr<<"' has no sessions - do nothing");
      }

      // The clients to cleaned
      std::list<XrdProofdClient *> *clnts;
      if (all) {
         // The full list
         clnts = fgMgr.ProofdClients();
      } else {
         clnts = new std::list<XrdProofdClient *>;
         clnts->push_back(tgtclnt);
      }

      std::list<XrdProofdClient *>::iterator i;
      XrdProofdClient *c = 0;
      if (clntfound) {
         // List of process IDs asked to terminate
         std::list<int *> signalledpid;

         // Loop over them
         c = 0;
         for (i = clnts->begin(); i != clnts->end(); ++i) {
            if ((c = *i)) {

               // This part may be not thread safe
               XrdSysMutexHelper mh(c->Mutex());

               // Notify the attached clients that we are going to cleanup
               XrdOucString msg = "Admin: CleanupSessions: cleaning up client: requested by: ";
               msg += fLink->ID;
               int ic = 0;
               XrdProofdProtocol *p = 0;
               for (ic = 0; ic < (int) c->Clients()->size(); ic++) {
                  if ((p = c->Clients()->at(ic)) && (p != this) && p->fTopClient) {
                     unsigned short sid;
                     p->fResponse.GetSID(sid);
                     p->fResponse.Set(c->RefSid());
                     p->fResponse.Send(kXR_attn, kXPD_srvmsg, (char *) msg.c_str(), msg.length());
                     p->fResponse.Set(sid);
                     // Close the link, so that the associated protocol instance
                     // can be recycled
                     p->fLink->Close();
                  }
               }

               // Asynchronous notification to requester
               if (fgMgr.SrvType() != kXPD_WorkerServer) {
                  cmsg = "Reset: signalling active sessions for termination";
                  fResponse.Send(kXR_attn, kXPD_srvmsg, (char *) cmsg.c_str(), cmsg.length());
               }

               // Loop over client sessions and terminated them
               int is = 0;
               XrdProofServProxy *s = 0;
               for (is = 0; is < (int) c->ProofServs()->size(); is++) {
                  if ((s = c->ProofServs()->at(is)) && s->IsValid() &&
                     s->SrvType() == srvtype) {
                     int *pid = new int;
                     *pid = s->SrvID();
                     TRACEI(HDBG, "Admin: CleanupSessions: terminating " << *pid);
                     if (s->TerminateProofServ() < 0) {
                        if (KillProofServ(*pid, 0) == 0)
                           signalledpid.push_back(pid);
                     } else
                        signalledpid.push_back(pid);
                     // Reset session proxy
                     s->Reset();
                  }
               }
            }
         }

         // Asynchronous notification to requester
         if (fgMgr.SrvType() != kXPD_WorkerServer) {
            cmsg = "Reset: verifying termination status (may take up to 10 seconds)";
            fResponse.Send(kXR_attn, kXPD_srvmsg, 0, (char *) cmsg.c_str(), cmsg.length());
         }

         // Now we give sometime to sessions to terminate (10 sec).
         // We check the status every second
         int nw = 10;
         int nleft = signalledpid.size();
         while (nw-- && nleft > 0) {

            // Loop over the list of processes requested to terminate
            std::list<int *>::iterator ii;
            for (ii = signalledpid.begin(); ii != signalledpid.end(); )
               if (fgMgr.VerifyProcessByID(*(*ii)) == 0) {
                  nleft--;
                  delete (*ii);
                  ii = signalledpid.erase(ii);
               } else
                  ++ii;

            // Wait a bit before retrying
            sleep(1);
         }
      }

      // Lock the interested client mutexes (no action is allowed while
      // doing this
      if (clnts) {
         for (i = clnts->begin(); i != clnts->end(); ++i)
            if ((c = *i))
               c->Mutex()->Lock();
      }

      // Asynchronous notification to requester
      if (fgMgr.SrvType() != kXPD_WorkerServer) {
         cmsg = "Reset: terminating the remaining sessions ...";
         fResponse.Send(kXR_attn, kXPD_srvmsg, 0, (char *) cmsg.c_str(), cmsg.length());
      }

      // Now we cleanup what left (any zombies or super resistent processes)
      int ncln = CleanupProofServ(all, usr);
      if (ncln > 0) {
         // Asynchronous notification to requester
         cmsg = "Reset: wait 5 seconds for completion ...";
         fResponse.Send(kXR_attn, kXPD_srvmsg, 0, (char *) cmsg.c_str(), cmsg.length());
         sleep(5);
      }

      // Cleanup all possible sessions around
      // (forward down the tree only if not leaf)
      if (fgMgr.SrvType() != kXPD_WorkerServer) {

         // Asynchronous notification to requester
         cmsg = "Reset: forwarding the reset request to next tier(s) ";
         fResponse.Send(kXR_attn, kXPD_srvmsg, 0, (char *) cmsg.c_str(), cmsg.length());

         fgMgr.Broadcast(type, usr, &fResponse, 1);
      }

      // Unlock the locked client mutexes
      if (clnts) {
         for (i = clnts->begin(); i != clnts->end(); ++i)
            if ((c = *i))
               c->Mutex()->UnLock();
      }

      // Cleanup usr
      SafeDelArray(usr);

      // Acknowledge user
      fResponse.Send();

   } else if (type == kSendMsgToUser) {

      // Target client (default us)
      XrdProofdClient *tgtclnt = fPClient;
      XrdProofdClient *c = 0;
      std::list<XrdProofdClient *>::iterator i;

      // Extract the user name, if any
      int len = fRequest.header.dlen;
      if (len <= 0) {
         // No message: protocol error?
         TRACEP(XERR, "Admin: kSendMsgToUser: no message");
         fResponse.Send(kXR_InvalidRequest,"Admin: kSendMsgToUser: no message");
         return rc;
      }

      XrdOucString cmsg((const char *)fArgp->buff, len);
      XrdOucString usr;
      if (cmsg.beginswith("u:")) {
         // Extract user
         int isp = cmsg.find(' ');
         if (isp != STR_NPOS) {
            usr.assign(cmsg, 2, isp-1);
            cmsg.erase(0, isp+1);
         }
         if (usr.length() > 0) {
            TRACEP(DBG, "Admin: kSendMsgToUser: request for user: '"<<usr<<"'");
            // Find the client instance
            bool clntfound = 0;
            for (i = fgMgr.ProofdClients()->begin(); i != fgMgr.ProofdClients()->end(); ++i) {
               if ((c = *i) && c->Match(usr.c_str())) {
                  tgtclnt = c;
                  clntfound = 1;
                  break;
               }
            }
            if (!clntfound) {
               // No message: protocol error?
               TRACEP(XERR, "Admin: kSendMsgToUser: target client not found");
               fResponse.Send(kXR_InvalidRequest,
                              "Admin: kSendMsgToUser: target client not found");
               return rc;
            }
         }
      }
      // Recheck message length
      if (cmsg.length() <= 0) {
         // No message: protocol error?
         TRACEP(XERR, "Admin: kSendMsgToUser: no message after user specification");
         fResponse.Send(kXR_InvalidRequest,
                        "Admin: kSendMsgToUser: no message after user specification");
         return rc;
      }

      // Check if allowed
      bool all = 0;
      if (!fSuperUser) {
         if (usr.length() > 0) {
            if (tgtclnt != fPClient) {
               TRACEP(XERR, "Admin: kSendMsgToUser: not allowed to send messages to usr '"<<usr<<"'");
               fResponse.Send(kXR_InvalidRequest,
                           "Admin: kSendMsgToUser: not allowed to send messages to specified usr");
               return rc;
            }
         } else {
            TRACEP(XERR, "Admin: kSendMsgToUser: not allowed to send messages to connected users");
            fResponse.Send(kXR_InvalidRequest,
                           "Admin: kSendMsgToUser: not allowed to send messages to connected users");
            return rc;
         }
      } else {
         if (usr.length() <= 0)
            all = 1;
      }

      // The clients to notified
      std::list<XrdProofdClient *> *clnts;
      if (all) {
         // The full list
         clnts = fgMgr.ProofdClients();
      } else {
         clnts = new std::list<XrdProofdClient *>;
         clnts->push_back(tgtclnt);
      }

      // Loop over them
      c = 0;
      for (i = clnts->begin(); i != clnts->end(); ++i) {
         if ((c = *i)) {

            // This part may be not thread safe
            XrdSysMutexHelper mh(c->Mutex());

            // Notify the attached clients
            int ic = 0;
            XrdProofdProtocol *p = 0;
            for (ic = 0; ic < (int) c->Clients()->size(); ic++) {
               if ((p = c->Clients()->at(ic)) && p->fTopClient) {
                  unsigned short sid;
                  p->fResponse.GetSID(sid);
                  p->fResponse.Set(c->RefSid());
                  p->fResponse.Send(kXR_attn, kXPD_srvmsg, (char *) cmsg.c_str(), cmsg.length());
                  p->fResponse.Set(sid);
               }
            }
         }
      }

      // Acknowledge user
      fResponse.Send();

   } else if (type == kSessionTag) {

      //
      // Specific info about a session
      XrdProofServProxy *xps = 0;
      if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
          !(xps = fPClient->ProofServs()->at(psid))) {
         TRACEP(XERR, "Admin: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"Admin: session ID not found");
         return rc;
      }

      // Set session tag
      const char *msg = (const char *) fArgp->buff;
      int   len = fRequest.header.dlen;
      if (len > kXPROOFSRVTAGMAX - 1)
         len = kXPROOFSRVTAGMAX - 1;

      // Save tag
      if (len > 0 && msg) {
         xps->SetTag(msg, len);
         if (TRACING(ADMIN)) {
            TRACEI(DBG, "Admin: session tag set to: "<<xps->Tag());
         }
      }

      // Acknowledge user
      fResponse.Send();

   } else if (type == kSessionAlias) {

      //
      // Specific info about a session
      XrdProofServProxy *xps = 0;
      if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
          !(xps = fPClient->ProofServs()->at(psid))) {
         TRACEP(XERR, "Admin: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"Admin: session ID not found");
         return rc;
      }

      // Set session alias
      const char *msg = (const char *) fArgp->buff;
      int   len = fRequest.header.dlen;
      if (len > kXPROOFSRVALIASMAX - 1)
         len = kXPROOFSRVALIASMAX - 1;

      // Save tag
      if (len > 0 && msg) {
         xps->SetAlias(msg, len);
         if (TRACING(ADMIN)) {
            TRACEP(DBG, "Admin: session alias set to: "<<xps->Alias());
         }
      }

      // Acknowledge user
      fResponse.Send();

   } else if (type == kGroupProperties) {

      XrdSysMutexHelper mh(fPClient->Mutex());
      //
      // Specific info about a session
      XrdProofServProxy *xps = 0;
      if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
          !(xps = fPClient->ProofServs()->at(psid))) {
         TRACEP(XERR, "Admin: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"Admin: session ID not found");
         return rc;
      }

      // User's group
      int   len = fRequest.header.dlen;
      char *grp = new char[len+1];
      memcpy(grp, fArgp->buff, len);
      grp[len] = 0;
      TRACEP(ADMIN, "Admin: request to change priority for group '"<< grp<<"'");

      // Make sure is the current one of the user
      XrdProofGroup *g = xps->Group();
      if (g && strcmp(grp, g->Name())) {
         TRACEP(XERR, "Admin: received group does not match the user's one");
         fResponse.Send(kXR_InvalidRequest,
                      "Admin: received group does not match the user's one");
         return rc;
      }

      // Set the priority
      int priority = ntohl(fRequest.proof.int2);
      g->SetPriority((float)priority);

      // Make sure scheduling is ON
      fgMgr.SetSchedOpt(kXPD_sched_central);

      // Set the new nice values to worker sessions
      if (fgMgr.SetNiceValues(2) != 0) {
         TRACE(XERR,"Admin: problems setting the new nice values ");
         fResponse.Send(kXR_InvalidRequest,
                      "Admin: problems setting the new nice values ");
         return rc;
      }

      // Notify
      TRACEP(ADMIN, "Admin: priority for group '"<< grp<<"' has been set to "<<priority);

      // Acknowledge user
      fResponse.Send();

   } else if (type == kGetWorkers) {

      // Find server session
      XrdProofServProxy *xps = 0;
      if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
          !(xps = fPClient->ProofServs()->at(psid))) {
         TRACEP(XERR, "Admin: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"session ID not found");
         return rc;
      }

      // We should query the chosen resource provider
      XrdOucString wrks;

      if (fgMgr.GetWorkers(wrks, xps) !=0 ) {
         // Something wrong
         fResponse.Send(kXR_InvalidRequest,"Admin: GetWorkers failed");
         return rc;
      } else {
         // Send buffer
         char *buf = (char *) wrks.c_str();
         int len = wrks.length() + 1;
         TRACEP(DBG, "Admin: GetWorkers: sending: "<<buf);

         // Send back to user
         fResponse.Send(buf, len);
      }
   } else if (type == kQueryWorkers) {

      // Send back a list of potentially available workers
      XrdOucString sbuf(1024);
      fgMgr.ProofSched()->ExportInfo(sbuf);

      // Send buffer
      char *buf = (char *) sbuf.c_str();
      int len = sbuf.length() + 1;
      TRACEP(DBG, "Admin: QueryWorkers: sending: "<<buf);

      // Send back to user
      fResponse.Send(buf, len);

   } else if (type == kQueryROOTVersions) {

      // The total length first
      int len = 0;
      std::list<XrdROOT *>::iterator ip;
      for (ip = fgMgr.ROOT()->begin(); ip != fgMgr.ROOT()->end(); ++ip) {
         len += strlen((*ip)->Export());
         len += 5;
      }

      // Generic info about all known sessions
      char *buf = new char[len+2];
      char *pw = buf;
      for (ip = fgMgr.ROOT()->begin(); ip != fgMgr.ROOT()->end(); ++ip) {
         if (fPClient->ROOT() == *ip)
            memcpy(pw, "  * ", 4);
         else
            memcpy(pw, "    ", 4);
         pw += 4;
         const char *ex = (*ip)->Export();
         int lex = strlen(ex);
         memcpy(pw, ex, lex);
         pw[lex] = '\n';
         pw += (lex+1);
      }
      *pw = 0;
      TRACEP(DBG, "Admin: sending: "<<buf);

      // Send back to user
      fResponse.Send(buf,strlen(buf)+1);
      if (buf) delete[] buf;

   } else if (type == kROOTVersion) {

      // Change default ROOT version
      const char *t = fArgp ? (const char *) fArgp->buff : "default";
      int len = fArgp ? fRequest.header.dlen : strlen("default");
      XrdOucString tag(t,len);

      // If a user name is given separate it out and check if
      // we can do the operation
      XrdOucString usr;
      if (tag.beginswith("u:")) {
         usr = tag;
         usr.erase(usr.rfind(' '));
         usr.replace("u:","");
         TRACEI(DBG, "Admin: ROOTVersion: request is for user: "<< usr);
         // Isolate the tag
         tag.erase(0,tag.find(' ') + 1);
      }
      TRACEP(ADMIN, "Admin: ROOTVersion: version tag: "<< tag);

      // If the action is requested for a user different from us we
      // must be 'superuser'
      XrdProofdClient *c = fPClient;
      XrdOucString grp;
      if (usr.length() > 0) {
         // Separate group info, if any
         if (usr.find(':') != STR_NPOS) {
            grp = usr;
            grp.erase(grp.rfind(':'));
            usr.erase(0,usr.find(':') + 1);
         } else {
            XrdProofGroup *g =
               (fgMgr.GroupsMgr()) ? fgMgr.GroupsMgr()->GetUserGroup(usr.c_str()) : 0;
            grp = g ? g->Name() : "default";
         }
         if (usr != fPClient->ID()) {
            if (!fSuperUser) {
               usr.insert("Admin: not allowed to change settings for usr '", 0);
               usr += "'";
               TRACEI(XERR, usr.c_str());
               fResponse.Send(kXR_InvalidRequest, usr.c_str());
               return rc;
            }
            // Lookup the list
            c = 0;
            std::list<XrdProofdClient *>::iterator i;
            for (i = fgMgr.ProofdClients()->begin(); i != fgMgr.ProofdClients()->end(); ++i) {
               if ((*i)->Match(usr.c_str(), grp.c_str())) {
                  c = (*i);
                  break;
               }
            }
            if (!c) {
               // Is this a potential user?
               XrdOucString emsg;
               XrdProofUI ui;
               if (fgMgr.CheckUser(usr.c_str(), ui, emsg) != 0) {
                  // No: fail
                  emsg.insert(": ", 0);
                  emsg.insert(usr, 0);
                  emsg.insert("Admin: user not found: ", 0);
                  TRACEP(XERR, emsg.c_str());
                  fResponse.Send(kXR_InvalidRequest, emsg.c_str());
                  return rc;
               } else {
                  // Yes: create an (invalid) instance of XrdProofdClient:
                  // It would be validated on the first valid login
                  c = new XrdProofdClient(usr.c_str(), (short int) -1, ui);
                  // Locate and set the group, if any
                  if (fgMgr.GroupsMgr() && fgMgr.GroupsMgr()->Num() > 0)
                     c->SetGroup(fgMgr.GroupsMgr()->GetUserGroup(usr.c_str(), grp.c_str()));
                  // Add to the list
                  fgMgr.ProofdClients()->push_back(c);
                  TRACEP(DBG, "Admin: instance for {client, group} = {"<<usr<<", "<<
                              grp<<"} created and added to the list ("<<c<<")");
               }
            }
         }
      }

      // Search in the list
      bool ok = 0;
      std::list<XrdROOT *>::iterator ip;
      for (ip = fgMgr.ROOT()->begin(); ip != fgMgr.ROOT()->end(); ++ip) {
         if ((*ip)->MatchTag(tag.c_str())) {
            c->SetROOT(*ip);
            ok = 1;
            break;
         }
      }

      // If not found we may have been requested to set the default version
      if (!ok && tag == "default") {
         c->SetROOT(*fgMgr.ROOT()->begin());
         ok = 1;
      }

      if (ok) {
         // Notify
         TRACEP(ADMIN, "Admin: default changed to "<<c->ROOT()->Tag()<<
                       " for {client, group} = {"<<usr<<", "<<grp<<"} ("<<c<<")");
         // Forward down the tree, if not leaf
         if (fgMgr.SrvType() != kXPD_WorkerServer) {
            XrdOucString buf("u:");
            buf += c->ID();
            buf += " ";
            buf += tag;
            fgMgr.Broadcast(type, buf.c_str(), &fResponse);
         }
         // Acknowledge user
         fResponse.Send();
      } else {
         tag.insert("Admin: tag '", 0);
         tag += "' not found in the list of available ROOT versions";
         TRACEP(XERR, tag.c_str());
         fResponse.Send(kXR_InvalidRequest, tag.c_str());
      }
   } else {
      TRACEP(XERR, "Admin: unknown request type");
      fResponse.Send(kXR_InvalidRequest,"Admin: unknown request type");
      return rc;
   }

   // Over
   return 0;
}

//___________________________________________________________________________
int XrdProofdProtocol::Interrupt()
{
   // Handle an interrupt request

   unsigned int rc = 1;

   // Unmarshall the data
   int psid = ntohl(fRequest.interrupt.sid);
   int type = ntohl(fRequest.interrupt.type);
   TRACEI(REQ, "Interrupt: psid: "<<psid<<", type:"<<type);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
       !(xps = fPClient->ProofServs()->at(psid))) {
      TRACEP(XERR, "Interrupt: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"nterrupt: session ID not found");
      return rc;
   }

   if (xps) {

      // Check ID matching
      if (!xps->Match(psid)) {
         fResponse.Send(kXP_InvalidRequest,"Interrupt: IDs do not match - do nothing");
         return rc;
      }

      TRACEP(DBG, "Interrupt: xps: "<<xps<<", internal link "<<xps->Link()<<
                  ", proofsrv ID: "<<xps->SrvID());

      // Propagate the type as unsolicited
      if (xps->ProofSrv()->Send(kXR_attn, kXPD_interrupt, type) != 0) {
         fResponse.Send(kXP_ServerError,
                        "Interrupt: could not propagate interrupt code to proofsrv");
         return rc;
      }

      // Notify to user
      fResponse.Send();
      TRACEP(DBG, "Interrupt: interrupt propagated to proofsrv");
   }

   // Over
   return 0;
}

//___________________________________________________________________________
int XrdProofdProtocol::Ping()
{
   // Handle a ping request

   int rc = 1;

   // Unmarshall the data
   int psid = ntohl(fRequest.sendrcv.sid);
   int opt = ntohl(fRequest.sendrcv.opt);

   TRACEI(REQ, "Ping: psid: "<<psid<<", opt: "<<opt);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (!fPClient || !INRANGE(psid,fPClient->ProofServs()) ||
       !(xps = fPClient->ProofServs()->at(psid))) {
      TRACEP(XERR, "Ping: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
      return rc;
   }

   kXR_int32 pingres = 0;
   if (xps) {
      TRACEI(DBG, "Ping: xps: "<<xps<<", status: "<<xps->Status());

      // Type of connection
      bool external = !(opt & kXPD_internal);

      if (external) {
         TRACEI(DBG, "Ping: EXT: psid: "<<psid);

         // Send the request
         if ((pingres = (kXR_int32) xps->VerifyProofServ(fgInternalWait)) == -1) {
            TRACEP(XERR, "Ping: EXT: could not verify proofsrv");
            fResponse.Send(kXR_ServerError, "EXT: could not verify proofsrv");
            return rc;
         }

         // Notify the client
         TRACEP(DBG, "Ping: EXT: ping notified to client");
         fResponse.Send(kXR_ok, pingres);
         return rc;

      } else {
         TRACEI(DBG, "Ping: INT: psid: "<<psid);

         // If a semaphore is waiting, post it
         if (xps->PingSem())
            xps->PingSem()->Post();


         // Just notify to user
         pingres = 1;
         TRACEP(DBG, "Ping: INT: ping notified to client");
         fResponse.Send(kXR_ok, pingres);
         return rc;
      }
   }

   // Failure
   TRACEP(XERR, "Ping: session ID not found");
   fResponse.Send(kXR_ok, pingres);
   return 0;
}

//___________________________________________________________________________
int XrdProofdProtocol::SetUserEnvironment()
{
   // Set user environment: set effective user and group ID of the process
   // to the ones of the owner of this protocol instnace and change working
   // dir to the sandbox.
   // Return 0 on success, -1 if enything goes wrong.

   MTRACE(ACT, "xpd:child: ", "SetUserEnvironment: enter");

   if (XrdProofdAux::ChangeToDir(fPClient->Workdir(), fUI, fgMgr.ChangeOwn()) != 0) {
      MTRACE(XERR, "xpd:child: ", "SetUserEnvironment: couldn't change directory to "<<
                   fPClient->Workdir());
      return -1;
   }

   // set HOME env
   char *h = new char[8 + strlen(fPClient->Workdir())];
   sprintf(h, "HOME=%s", fPClient->Workdir());
   putenv(h);
   MTRACE(XERR, "xpd:child: ", "SetUserEnvironment: set "<<h);

   // Set access control list from /etc/initgroup
   // (super-user privileges required)
   MTRACE(DBG, "xpd:child: ", "SetUserEnvironment: setting ACLs");
   if (fgMgr.ChangeOwn() && (int) geteuid() != fUI.fUid) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, fUI.fUid)) {
         MTRACE(XERR, "xpd:child: ", "SetUserEnvironment: could not get privileges");
         return -1;
      }

      initgroups(fUI.fUser.c_str(), fUI.fGid);
   }

   if (fgMgr.ChangeOwn()) {
      // acquire permanently target user privileges
      MTRACE(DBG, "xpd:child: ", "SetUserEnvironment: acquire target user identity");
      if (XrdSysPriv::ChangePerm((uid_t)fUI.fUid, (gid_t)fUI.fGid) != 0) {
         MTRACE(XERR, "xpd:child: ",
                      "SetUserEnvironment: can't acquire "<< fUI.fUser <<" identity");
         return -1;
      }
   }

   // Save UNIX path in the sandbox for later cleaning
   // (it must be done after sandbox login)
   fPClient->SaveUNIXPath();

   // We are done
   MTRACE(DBG, "xpd:child: ", "SetUserEnvironment: done");
   return 0;
}


//______________________________________________________________________________
int XrdProofdProtocol::CleanupProofServ(bool all, const char *usr)
{
   // Cleanup (kill) all 'proofserv' processes from the process table.
   // Only the processes associated with 'usr' are killed,
   // unless 'all' is TRUE, in which case all 'proofserv' instances are
   // terminated (this requires superuser privileges).
   // Super users can also terminated all processes fo another user (specified
   // via usr).
   // Return number of process notified for termination on success, -1 otherwise

   TRACE(ACT, "CleanupProofServ: enter: all: "<<all<<
               ", usr: " << (usr ? usr : "undef"));
   int nk = 0;

   // Name
   const char *pn = "proofserv";

   // Uid
   int refuid = -1;
   if (!all) {
      if (!usr) {
         TRACE(DBG, "CleanupProofServ: usr must be defined for all = FALSE");
         return -1;
      }
      XrdProofUI ui;
      if (XrdProofdAux::GetUserInfo(usr, ui) != 0) {
         TRACE(DBG, "CleanupProofServ: problems getting info for user " << usr);
         return -1;
      }
      refuid = ui.fUid;
   }

#if defined(linux)
   // Loop over the "/proc" dir
   DIR *dir = opendir("/proc");
   if (!dir) {
      XrdOucString emsg("CleanupProofServ: cannot open /proc - errno: ");
      emsg += errno;
      TRACE(DBG, emsg.c_str());
      return -1;
   }

   struct dirent *ent = 0;
   while ((ent = readdir(dir))) {
      if (DIGIT(ent->d_name[0])) {
         XrdOucString fn("/proc/", 256);
         fn += ent->d_name;
         fn += "/status";
         // Open file
         FILE *ffn = fopen(fn.c_str(), "r");
         if (!ffn) {
            XrdOucString emsg("CleanupProofServ: cannot open file ");
            emsg += fn; emsg += " - errno: "; emsg += errno;
            TRACE(HDBG, emsg.c_str());
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
               if (ppid != getpid() &&
                   fgMgr.VerifyProcessByID(ppid, "xrootd"))
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
            if (fgMgr.MultiUser() && !all) {
               // We need to check the user name: we may be the owner of somebody
               // else process; if not session is attached, we kill it
               muok = 0;
               XrdProofServProxy *srv = fgMgr.GetActiveSession(pid);
               if (!srv || (srv && !strcmp(usr, srv->Client())))
                  muok = 1;
            }
            if (muok)
               if (KillProofServ(pid, 1) == 0)
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
      XrdOucString emsg("CleanupProofServ: cannot open /proc - errno: ");
      emsg += errno;
      TRACE(DBG, emsg.c_str());
      return -1;
   }

   struct dirent *ent = 0;
   while ((ent = readdir(dir))) {
      if (DIGIT(ent->d_name[0])) {
         XrdOucString fn("/proc/", 256);
         fn += ent->d_name;
         fn += "/psinfo";
         // Open file
         int ffd = open(fn.c_str(), O_RDONLY);
         if (ffd <= 0) {
            XrdOucString emsg("CleanupProofServ: cannot open file ");
            emsg += fn; emsg += " - errno: "; emsg += errno;
            TRACE(HDBG, emsg.c_str());
            continue;
         }
         // Read info
         bool xname = 1;
         bool xuid = (all) ? 0 : 1;
         bool xppid = 1;
         // Get the information
         psinfo_t psi;
         if (read(ffd, &psi, sizeof(psinfo_t)) != sizeof(psinfo_t)) {
            XrdOucString emsg("CleanupProofServ: cannot read ");
            emsg += fn; emsg += ": errno: "; emsg += errno;
            TRACE(XERR, emsg.c_str());
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
         if (ppid != getpid() &&
             fgMgr.VerifyProcessByID(ppid, "xrootd")) {
             // Process created by another running xrootd
             continue;
             xppid = 0;
         }

         // If this is a good candidate, kill it
         if (!xname && !xppid && !xuid) {
            bool muok = 1;
            if (fgMgr.MultiUser() && !all) {
               // We need to check the user name: we may be the owner of somebody
               // else process; if no session is attached , we kill it
               muok = 0;
               XrdProofServProxy *srv = fgMgr.GetActiveSession(psi.pr_pid);
               if (!srv || (srv && !strcmp(usr, srv->Client())))
                  muok = 1;
            }
            if (muok)
               if (KillProofServ(psi.pr_pid, 1) == 0)
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
      XrdOucString emsg("CleanupProofServ: cannot get the process list: errno: ");
      emsg += ern;
      TRACE(XERR, emsg.c_str());
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
               if (fgMgr.MultiUser() && !all) {
                  // We need to check the user name: we may be the owner of somebody
                  // else process; if no session is attached, we kill it
                  muok = 0;
                  XrdProofServProxy *srv = fgMgr.GetActiveSession(pl[np].kp_proc.p_pid);
                  if (!srv || (srv && !strcmp(usr, srv->Client())))
                     muok = 1;
               }
               if (muok)
                  // Good candidate to be shot
                  if (KillProofServ(pl[np].kp_proc.p_pid, 1))
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
   const char *cusr = (usr && strlen(usr) && fSuperUser) ? usr : (const char *)fClientID;
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
            TRACE(HDBG, "CleanupProofServ: found alternative parent ID: "<< ppid);
            // If still running then skip
            if (fgMgr.VerifyProcessByID(ppid, "xrootd"))
               continue;
         }
         // Get pid now
         int from = 0;
         if (busr)
            from += strlen(cusr);
         int pid = (int) XrdProofdAux::GetLong(&line[from]);
         bool muok = 1;
         if (fgMgr.MultiUser() && !all) {
            // We need to check the user name: we may be the owner of somebody
            // else process; if no session is attached, we kill it
            muok = 0;
            XrdProofServProxy *srv = fgMgr.GetActiveSession(pid);
            if (!srv || (srv && !strcmp(usr, srv->Client())))
               muok = 1;
         }
         if (muok)
            // Kill it
            if (KillProofServ(pid, 1) == 0)
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

//______________________________________________________________________________
int XrdProofdProtocol::KillProofServ(int pid, bool forcekill)
{
   // Kill the process 'pid'.
   // A SIGTERM is sent, unless 'kill' is TRUE, in which case a SIGKILL is used.
   // If add is TRUE (default) the pid is added to the list of processes
   // requested to terminate.
   // Return 0 on success, -1 if not allowed or other errors occured.

   TRACE(ACT, "KillProofServ: enter: pid: "<<pid<< ", forcekill: "<< forcekill);

   if (pid > 0) {
      // We need the right privileges to do this
      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (XpdBadPGuard(pGuard, fUI.fUid) && fgMgr.ChangeOwn()) {
         XrdOucString msg = "KillProofServ: could not get privileges";
         TRACE(XERR, msg.c_str());
         return -1;
      } else {
         bool signalled = 1;
         if (forcekill) {
            // Hard shutdown via SIGKILL
            if (kill(pid, SIGKILL) != 0) {
               if (errno != ESRCH) {
                  XrdOucString msg = "KillProofServ: kill(pid,SIGKILL) failed for process: ";
                  msg += pid;
                  msg += " - errno: ";
                  msg += errno;
                  TRACE(XERR, msg.c_str());
                  return -1;
               }
               signalled = 0;
            }
         } else {
            // Softer shutdown via SIGTERM
            if (kill(pid, SIGTERM) != 0) {
               if (errno != ESRCH) {
                  XrdOucString msg = "KillProofServ: kill(pid,SIGTERM) failed for process: ";
                  msg += pid;
                  msg += " - errno: ";
                  msg += errno;
                  TRACE(XERR, msg.c_str());
                  return -1;
               }
               signalled = 0;
            }
         }
         // Add to the list of termination attempts
         if (signalled) {
            // Avoid after notification after this point: it may create dead-locks
            if (fPClient) {
               // Record this session in the sandbox as old session
               XrdOucString tag = "-";
               tag += pid;
               if (fPClient->GuessTag(tag, 1, 0) == 0)
                  fPClient->MvOldSession(tag.c_str(), 0);
            }
         } else {
            TRACE(DBG, "KillProofServ: process ID "<<pid<<" not found in the process table");
         }
      }
   } else {
      return -1;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::ReadBuffer()
{
   // Process a readbuf request

   int rc = 1;
   XrdOucString emsg;

   // Find out the file name
   char *url = 0;
   char *file = 0;
   int dlen = fRequest.header.dlen;
   if (dlen > 0 && fArgp->buff) {
      int flen = dlen;
      int ulen = 0;
      int offs = 0;
      char *p = (char *) strstr(fArgp->buff, ",");
      if (p) {
         ulen = (int) (p - fArgp->buff);
         url = new char[ulen+1];
         memcpy(url, fArgp->buff, ulen);
         url[ulen] = 0;
         offs = ulen + 1;
         flen -= offs;
      }
      file = new char[flen+1];
      memcpy(file, fArgp->buff+offs, flen);
      file[flen] = 0;
   } else {
      emsg = "ReadBuffer: file name not not found";
      TRACEP(XERR, emsg);
      fResponse.Send(kXR_InvalidRequest, emsg.c_str());
      return rc;
   }

   // Unmarshall the data
   //
   kXR_int64 ofs = ntohll(fRequest.readbuf.ofs);
   int len = ntohl(fRequest.readbuf.len);
   TRACEI(REQ, "ReadBuffer: file: "<<file<<", ofs: "<<ofs<<", len: "<<len);

   // Check if local
   bool local = 0;
   int blen = dlen;
   XrdClientUrlInfo ui(file);
   if (ui.Host.length() > 0) {
      // Fully qualified name
      char *fqn = XrdNetDNS::getHostName(ui.Host.c_str());
      if (fqn && (strstr(fqn, "localhost") ||
                 !strcmp(fqn, "127.0.0.1") ||
                 !strcmp(fgMgr.Host(),fqn))) {
         memcpy(file, ui.File.c_str(), ui.File.length());
         file[ui.File.length()] = 0;
         blen = ui.File.length();
         local = 1;
         TRACEI(DBG, "ReadBuffer: file is LOCAL");
      }
      SafeFree(fqn);
   }

   // Get the buffer
   int lout = len;
   char *buf = 0;
   char *filen = 0;
   char *pattern = 0;
   int grep = ntohl(fRequest.readbuf.int1);
   if (grep > 0) {
      // 'grep' operation: len is the length of the 'pattern' to be grepped
      pattern = new char[len + 1];
      int j = blen - len;
      int i = 0;
      while (j < blen)
         pattern[i++] = file[j++];
      pattern[i] = 0;
      filen = strdup(file);
      filen[blen - len] = 0;
      TRACEI(DBG, "ReadBuffer: grep operation "<<grep<<", pattern:"<<pattern);
   }
   if (local) {
      if (grep > 0) {
         // Grep local file
         lout = blen; // initial length
         buf = ReadBufferLocal(filen, pattern, lout, grep);
      } else {
         // Read portion of local file
         buf = ReadBufferLocal(file, ofs, lout);
      }
   } else {
      // Read portion of remote file
      buf = ReadBufferRemote(url, file, ofs, lout, grep);
   }

   if (!buf) {
      if (lout > 0) {
         if (grep > 0) {
            if (TRACING(DBG)) {
               emsg = "ReadBuffer: nothing found by 'grep' in ";
               emsg += filen;
               emsg += ", pattern: ";
               emsg += pattern;
               TRACEP(DBG, emsg);
            }
            fResponse.Send();
            return rc;
         } else {
            emsg = "ReadBuffer: could not read buffer from ";
            emsg += (local) ? "local file " : "remote file ";
            emsg += file;
            TRACEP(XERR, emsg);
            fResponse.Send(kXR_InvalidRequest, emsg.c_str());
            return rc;
         }
      } else {
         // Just got an empty buffer
         if (TRACING(DBG)) {
            emsg = "ReadBuffer: nothing found in ";
            emsg += file;
            TRACEP(DBG, emsg);
         }
      }
   }

   // Send back to user
   fResponse.Send(buf, lout);

   // Cleanup
   SafeFree(buf);
   SafeDelArray(file);
   SafeFree(filen);
   SafeDelArray(pattern);

   // Done
   return rc;
}

//______________________________________________________________________________
char *XrdProofdProtocol::ReadBufferLocal(const char *file, kXR_int64 ofs, int &len)
{
   // Read a buffer of length 'len' at offset 'ofs' of local file 'file'; the
   // returned buffer must be freed by the caller.
   // Returns 0 in case of error.

   XrdOucString emsg;
   TRACEI(ACT, "ReadBufferLocal: file: "<<file<<", ofs: "<<ofs<<", len: "<<len);

   // Check input
   if (!file || strlen(file) <= 0) {
      TRACEI(XERR, "ReadBufferLocal: file path undefined!");
      return (char *)0;
   }

   // Open the file in read mode
   int fd = open(file, O_RDONLY);
   if (fd < 0) {
      emsg = "ReadBufferLocal: could not open ";
      emsg += file;
      TRACEI(XERR, emsg);
      return (char *)0;
   }

   // Size of the output
   struct stat st;
   if (fstat(fd, &st) != 0) {
      emsg = "ReadBufferLocal: could not get size of file with stat: errno: ";
      emsg += (int)errno;
      TRACEI(XERR, emsg);
      close(fd);
      return (char *)0;
   }
   off_t ltot = st.st_size;

   // Estimate offsets of the requested range
   // Start from ...
   kXR_int64 start = ofs;
   off_t fst = (start < 0) ? ltot + start : start;
   fst = (fst < 0) ? 0 : ((fst >= ltot) ? ltot - 1 : fst);
   // End at ...
   kXR_int64 end = fst + len;
   off_t lst = (end >= ltot) ? ltot : ((end > fst) ? end  : ltot);
   TRACEI(DBG, "ReadBufferLocal: file size: "<<ltot<<
               ", read from: "<<fst<<" to "<<lst);

   // Number of bytes to be read
   len = lst - fst;

   // Output buffer
   char *buf = (char *)malloc(len + 1);
   if (!buf) {
      emsg = "ReadBufferLocal: could not allocate enough memory on the heap: errno: ";
      emsg += (int)errno;
      XPDERR(emsg);
      close(fd);
      return (char *)0;
   }

   // Reposition, if needed
   if (fst >= 0)
      lseek(fd, fst, SEEK_SET);

   int left = len;
   int pos = 0;
   int nr = 0;
   do {
      while ((nr = read(fd, buf + pos, left)) < 0 && errno == EINTR)
         errno = 0;
      TRACEI(HDBG, "ReadBufferLocal: read "<<nr<<" bytes: "<< buf);
      if (nr < 0) {
         TRACEI(XERR, "ReadBufferLocal: error reading from file: errno: "<< errno);
         break;
      }

      // Update counters
      pos += nr;
      left -= nr;

   } while (nr > 0 && left > 0);

   // Termination
   buf[len] = 0;

   // Close file
   close(fd);

   // Done
   return buf;
}

//______________________________________________________________________________
char *XrdProofdProtocol::ReadBufferLocal(const char *file,
                                         const char *pat, int &len, int opt)
{
   // Grep lines matching 'pat' form 'file'; the returned buffer (length in 'len')
   // must be freed by the caller.
   // Returns 0 in case of error.

   XrdOucString emsg;
   TRACEI(ACT, "ReadBufferLocal: file: "<<file<<", pat: "<<pat<<", len: "<<len);

   // Check input
   if (!file || strlen(file) <= 0) {
      TRACEI(XERR, "ReadBufferLocal: file path undefined!");
      return (char *)0;
   }

   // Size of the output
   struct stat st;
   if (stat(file, &st) != 0) {
      emsg = "ReadBufferLocal: could not get size of file with stat: errno: ";
      emsg += (int)errno;
      TRACEI(XERR, emsg);
      return (char *)0;
   }
   off_t ltot = st.st_size;

   XrdOucString cmd("grep ");
   if (pat && strlen(pat) > 0) {
      if (opt == 2) cmd += " -v ";
      cmd += pat;
      cmd += " ";
   } else {
      cmd = "cat ";
   }
   cmd += file;
   TRACEI(ACT, "ReadBufferLocal: cmd: "<<cmd);

   // Execute the command in a pipe
   FILE *fp = popen(cmd.c_str(), "r");
   if (!fp) {
      emsg = "ReadBufferLocal: could not open ";
      emsg += file;
      TRACEI(XERR, emsg);
      return (char *)0;
   }

   // Read line by line
   len = 0;
   char *buf = 0;
   char line[2048];
   int bufsiz = 0, left = 0, lines = 0;
   while ((ltot > 0) && fgets(line, sizeof(line), fp)) {
      // Parse the line
      int llen = strlen(line);
      ltot -= llen;
      lines++;
      // (Re-)allocate the buffer
      if (!buf || (llen > left)) {
         int dsiz = 100 * ((int) ((len + llen) / lines) + 1);
         dsiz = (dsiz > llen) ? dsiz : llen;
         bufsiz += dsiz;
         buf = (char *)realloc(buf, bufsiz + 1);
         left += dsiz;
      }
      if (!buf) {
         emsg = "ReadBufferLocal: could not allocate enough memory on the heap: errno: ";
         emsg += (int)errno;
         XPDERR(emsg);
         pclose(fp);
         return (char *)0;
      }
      // Add line to the buffer
      memcpy(buf+len, line, llen);
      len += llen;
      left -= llen;
      if (TRACING(HDBG))
         fprintf(stderr, "line: %s", line);
   }

   // Check the result and terminate the buffer
   if (buf) {
      if (len > 0) {
         buf[len] = 0;
      } else {
         free(buf);
         buf = 0;
      }
   }

   // Close the pipe
   pclose(fp);
   // Done
   return buf;
}

//______________________________________________________________________________
char *XrdProofdProtocol::ReadBufferRemote(const char *url, const char *file,
                                          kXR_int64 ofs, int &len, int grep)
{
   // Send a read buffer request of length 'len' at offset 'ofs' for remote file
   // defined by 'url'; the returned buffer must be freed by the caller.
   // Returns 0 in case of error.

   TRACEI(ACT, "ReadBufferRemote: url: "<<(url ? url : "undef")<<
               ", file: "<<(file ? file : "undef")<<", ofs: "<<ofs<<
               ", len: "<<len<<", grep: "<<grep);

   // Check input
   if (!file || strlen(file) <= 0) {
      TRACEI(XERR, "ReadBufferRemote: file undefined!");
      return (char *)0;
   }
   if (!url || strlen(url) <= 0) {
      // Use file as url
      url = file;
   }

   // We log in as the effective user to minimize the number of connections to the
   // other servers
   XrdClientUrlInfo u(url);
   u.User = fgMgr.EffectiveUser();
   XrdProofConn *conn = fgMgr.GetProofConn(u.GetUrl().c_str());

   char *buf = 0;
   if (conn && conn->IsValid()) {
      // Prepare request
      XPClientRequest reqhdr;
      memset(&reqhdr, 0, sizeof(reqhdr));
      conn->SetSID(reqhdr.header.streamid);
      reqhdr.header.requestid = kXP_readbuf;
      reqhdr.readbuf.ofs = ofs;
      reqhdr.readbuf.len = len;
      reqhdr.readbuf.int1 = grep;
      reqhdr.header.dlen = strlen(file);
      const void *btmp = (const void *) file;
      char **vout = &buf;
      // Send over
      XrdClientMessage *xrsp =
         conn->SendReq(&reqhdr, btmp, vout, "XrdProofdProtocol::ReadBufferRemote");

      // If positive answer
      if (xrsp && buf && (xrsp->DataLen() > 0)) {
         len = xrsp->DataLen();
      } else {
         if (xrsp && !(xrsp->IsError()))
            // The buffer was just empty: do not call it error
            len = 0;
         SafeFree(buf);
      }

      // Clean the message
      SafeDelete(xrsp);
   }

   // Done
   return buf;
}

//______________________________________________________________________________
char *XrdProofdProtocol::ReadLogPaths(const char *url, const char *msg, int isess)
{
   // Get log paths from next tier; used in multi-master setups
   // Returns 0 in case of error.

   TRACEI(ACT, "ReadLogPaths: url: "<<(url ? url : "undef")<<
               ", msg: "<<(msg ? msg : "undef")<<", isess: "<<isess);

   // Check input
   if (!url || strlen(url) <= 0) {
      TRACEI(XERR, "ReadLogPaths: url undefined!");
      return (char *)0;
   }

   // We log in as the effective user to minimize the number of connections to the
   // other servers
   XrdClientUrlInfo u(url);
   u.User = fgMgr.EffectiveUser();
   XrdProofConn *conn = fgMgr.GetProofConn(u.GetUrl().c_str());

   char *buf = 0;
   if (conn && conn->IsValid()) {
      // Prepare request
      XPClientRequest reqhdr;
      memset(&reqhdr, 0, sizeof(reqhdr));
      conn->SetSID(reqhdr.header.streamid);
      reqhdr.header.requestid = kXP_admin;
      reqhdr.proof.int1 = kQueryLogPaths;
      reqhdr.proof.int2 = isess;
      reqhdr.proof.sid = -1;
      reqhdr.header.dlen = strlen(msg);
      const void *btmp = (const void *) msg;
      char **vout = &buf;
      // Send over
      XrdClientMessage *xrsp =
         conn->SendReq(&reqhdr, btmp, vout, "XrdProofdProtocol::ReadLogPaths");

      // If positive answer
      if (xrsp && buf && (xrsp->DataLen() > 0)) {
         int len = xrsp->DataLen();
         buf = (char *) realloc((void *)buf, len+1);
         if (buf)
            buf[len] = 0;
      } else {
         SafeFree(buf);
      }

      // Clean the message
      SafeDelete(xrsp);
   }

   // Done
   return buf;
}
