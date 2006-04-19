// @(#)root/proofd:$Name:  $:$Id: XrdProofdProtocol.cxx,v 1.9 2006/04/17 21:11:32 rdm Exp $
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/stat.h>
#include <pwd.h>
#include <sys/resource.h>

// Bypass Solaris ELF madness
//
#if (defined(SUNCC) || defined(SUN))
#include <sys/isa_defs.h>
#if defined(_ILP32) && (_FILE_OFFSET_BITS != 32)
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 32
#undef  _LARGEFILE_SOURCE
#endif
#endif

#include <dlfcn.h>
#if !defined(__APPLE__)
#include <link.h>
#endif

#if defined(linux) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__) || defined(__OpenBSD__) || \
    defined(__APPLE__) || defined(__MACH__) || defined(cygwingcc)
#include <grp.h>
#include <sys/types.h>
#endif

// To ignore zombie childs
#include <signal.h>
#include <sys/param.h>
#if defined(__sun) || defined(__sgi)
#  include <fcntl.h>
#endif
#include <sys/wait.h>
#if defined(__FreeBSD__) || defined(__OpenBSD__) || \
    defined(__APPLE__) || defined(__hpux)
#define USE_SIGCHLD
#if !defined(__hpux)
#define	SIGCLD SIGCHLD
#endif
#endif

#include "XrdVersion.hh"
#include "XrdSys/XrdSysPriv.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdOuc/XrdOucError.hh"
#include "XrdOuc/XrdOucReqID.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdNet/XrdNet.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "Xrd/XrdLink.hh"
#include "Xrd/XrdPoll.hh"
#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdScheduler.hh"

#include "XrdProofdProtocol.h"

#include "config.h"

// Tracing utils
#include "XrdProofdTrace.h"
XrdOucTrace          *XrdProofdTrace = 0;
static const char    *gTraceID = " ";

// Static variables
static XrdOucReqID   *XrdProofdReqID = 0;

// Globals initialization
int                   XrdProofdProtocol::fgCount    = 0;
bool                  XrdProofdProtocol::fgConfigDone = 0;
int                   XrdProofdProtocol::fghcMax    = 28657; // const for now
XrdBuffManager       *XrdProofdProtocol::fgBPool    = 0;
int                   XrdProofdProtocol::fgMaxBuffsz= 0;
int                   XrdProofdProtocol::fgReadWait = 0;
int                   XrdProofdProtocol::fgInternalWait = 5; // seconds
int                   XrdProofdProtocol::fgPort     = 0;
int                   XrdProofdProtocol::fgMaxSessions = -1;
XrdSecService        *XrdProofdProtocol::fgCIA      = 0;
char                 *XrdProofdProtocol::fgSecLib   = 0;
char                 *XrdProofdProtocol::fgPrgmSrv  = 0;
int                   XrdProofdProtocol::fgSrvProtVers = -1;
char                 *XrdProofdProtocol::fgROOTsys  = 0;
char                 *XrdProofdProtocol::fgTMPdir   = 0;
char                 *XrdProofdProtocol::fgImage    = 0;
char                 *XrdProofdProtocol::fgWorkDir  = 0;
EResourceType         XrdProofdProtocol::fgResourceType = kRTStatic;
std::list<XrdOucString *> XrdProofdProtocol::fgMastersAllowed;
std::list<XrdProofdPriority *> XrdProofdProtocol::fgPriorities;
kXR_int32             XrdProofdProtocol::fgSrvType  = kXPD_AnyServer;
XrdOucString          XrdProofdProtocol::fgLocalHost;
char                 *XrdProofdProtocol::fgPoolURL = 0;
char                 *XrdProofdProtocol::fgNamespace = strdup("/proofpool");

// Static configuration section
char                 *XrdProofdProtocol::fgPROOFcfg = 0; // PROOF static configuration
int                   XrdProofdProtocol::fgWorkerMax = -1; // max number or workers per user
EStaticSelOpt         XrdProofdProtocol::fgWorkerSel = kSSORoundRobin; // selection option
bool                  XrdProofdProtocol::fgWorkerUsrCfg = 0; // user cfg files enabled / disabled
std::vector<XrdProofWorker *> XrdProofdProtocol::fgWorkers;  // list of possible workers

XrdScheduler         *XrdProofdProtocol::fgSched    = 0;
XrdOucError           XrdProofdProtocol::fgEDest(0, "Proofd");
std::list<XrdProofClient *> XrdProofdProtocol::fgProofClients;  // keeps track of all users
XrdOucMutex           XrdProofdProtocol::fgXPDMutex;
XrdObjectQ<XrdProofdProtocol>
XrdProofdProtocol::fgProtStack("ProtStack", "xproofd protocol anchor");

// Local definitions
#define MAX_ARGS 128
#define TRACELINK lp
#define TRACEID gTraceID
#ifndef SafeDelete
#define SafeDelete(x) { if (x) { delete x; x = 0; } }
#endif
#ifndef SafeDelArray
#define SafeDelArray(x) { if (x) { delete[] x; x = 0; } }
#endif
#ifndef SafeFree
#define SafeFree(x) { if (x) free(x); x = 0; }
#endif
#ifndef INRANGE
#define INRANGE(x,y) ((x >= 0) && (x < (int)y.size()))
#endif

// Macros used to set conditional options
#ifndef XPDCOND
#define XPDCOND(n,ns) ((n == -1 && ns == -1) || (n > 0 && n >= ns))
#endif
#ifndef XPDSETSTRING
#define XPDSETSTRING(n,ns,c,s) \
 { if (XPDCOND(n,ns)) { \
     SafeFree(c); c = strdup(s.c_str()); ns = n; }}
#endif
#ifndef XPDSETINT
#define XPDSETINT(n,ns,i,s) \
 { if (XPDCOND(n,ns)) { \
     i = strtol(s.c_str(),0,10); ns = n; }}
#endif

typedef struct {
   kXR_unt16 streamid;
   kXR_unt16 status;
   kXR_int32 rlen;
   kXR_int32 pval;
   kXR_int32 styp;
} hs_response_t;

// setresuid and setresgid
#if !defined(__hpux) && !defined(linux) && !defined(__FreeBSD__) && \
    !defined(__OpenBSD__) || defined(cygwingcc)
static int setresgid(gid_t r, gid_t e, gid_t) {
   if (setgid(r) == -1)
      return -1;
   return setegid(e);
}
static int setresuid(uid_t r, uid_t e, uid_t) {
   if (setuid(r) == -1)
      return -1;
   return seteuid(e);
}
#else
#if defined(linux) && !defined(R__HAS_SETRESUID)
extern "C" {
   int setresgid(gid_t r, gid_t e, gid_t s);
   int setresuid(uid_t r, uid_t e, uid_t s);
}
#endif
#endif

// Security handle
typedef XrdSecService *(*XrdSecServLoader_t)(XrdOucLogger *, const char *cfn);

//__________________________________________________________________________
int XrdProofdProtocol::ChangeProcessPriority(int pid, int dp)
{
   // Change priority of process pid by dp (positive or negative)
   // Returns 0 in case of success, -errno in case of error.

   // No action requested
   if (dp == 0)
      return 0;

   // Get current priority; errno needs to be reset here, as -1
   // could be a legitimate priority value
   errno = 0;
   int priority = getpriority(PRIO_PROCESS, pid);
   if (priority == -1 && errno != 0) {
      TRACE(REQ, "XrdProofdProtocol::ChangeProcessPriority:"
                 " getpriority: errno: " << errno);
      return -errno;
   }

   // Reference priority
   int refpriority = priority + dp;

   // Chaneg the priority
   if (setpriority(PRIO_PROCESS, pid, refpriority) != 0) {
      TRACE(REQ, "XrdProofdProtocol::ChangeProcessPriority:"
                 " setpriority: errno: " << errno);
      return ((errno != 0) ? -errno : -1);
   }

   // Check that it worked out
   errno = 0;
   if ((priority = getpriority(PRIO_PROCESS, pid)) == -1 && errno != 0) {
      TRACE(REQ, "XrdProofdProtocol::ChangeProcessPriority:"
                 " getpriority: errno: " << errno);
      return -errno;
   }
   if (priority != refpriority) {
      TRACE(REQ, "XrdProofdProtocol::ChangeProcessPriority:"
                 " unexpected result of action: found " << priority <<
                 ", expected "<<refpriority);
      errno = EPERM;
      return -errno;
   }

   // We are done
   return 0;
}

//__________________________________________________________________________
int XrdProofdProtocol::SetSrvProtVers()
{
   // Start a trial server application to test forking and get the version
   // of the protocol run by the PROOF server.
   // Return 0 if everything goes well, -1 in cse of any error.

   // Make sure the application path has been defined
   if (!fgPrgmSrv|| strlen(fgPrgmSrv) <= 0) {
      PRINT("SetSrvProtVers: error:"
            " path to PROOF server application undefined - exit");
      return -1;
   }

   // Make sure the temporary directory has been defined
   if (!fgTMPdir || strlen(fgTMPdir) <= 0) {
      PRINT("SetSrvProtVers: error:"
            " path to temporary directory undefined - exit");
      return -1;
   }

   // Make sure the temporary directory has been defined
   if (!fgROOTsys || strlen(fgROOTsys) <= 0) {
      PRINT("SetSrvProtVers: error: ROOTSYS undefined - exit");
      return -1;
   }

   // Pipe to communicate the protocol number
   int fp[2];
   if (pipe(fp) != 0) {
      PRINT("SetSrvProtVers: error: unable to generate pipe for"
            " PROOT protocol number communication");
      return -1;
   }

   // Fork a test agent process to handle this session
   int pid = -1;
   if (!(pid = fgSched->Fork("proofsrv"))) {

      char *argvv[5] = {0};

      // start server
      argvv[0] = (char *)fgPrgmSrv;
      argvv[1] = (char *)"proofserv";
      argvv[2] = (char *)"xpd";
      argvv[3] = (char *)"test";
      argvv[4] = 0;

      // Set conf dir
      char *ev = new char[20 + strlen(fgROOTsys)];
      sprintf(ev, "ROOTCONFDIR=%s", fgROOTsys);
      putenv(ev);

      // Set Open socket
      ev = new char[25];
      sprintf(ev, "ROOTOPENSOCK=%d", fp[1]);
      putenv(ev);

      // Close standard units
      close(2);
      close(1);
      close(0);

      // Run the program
      execv(fgPrgmSrv, argvv);

      // We should not be here!!!
      PRINT("SetSrvProtVers: error: returned from execv: bad, bad sign !!!");
      exit(1);
   }

   // parent process
   if (pid < 0) {
      PRINT("SetSrvProtVers: error: forking failed - exit");
      close(fp[0]);
      close(fp[1]);
      return -1;
   }

   // now we wait for the callback to be (successfully) established
   TRACE(REQ, "SetSrvProtVers: test server launched: wait for protocol ");

   // Read protocol
   int proto = -1;
   if (read(fp[0], &proto, sizeof(proto)) != sizeof(proto)) {
      PRINT("SetSrvProtVers: error:"
            " problems receiving PROOF server protocol number");
      return -1;
   }
   fgSrvProtVers = ntohl(proto);

   // Cleanup
   close(fp[0]);
   close(fp[1]);

   // We are done
   return 0;
}

//_____________________________________________________________________________
XrdSecService *XrdProofdProtocol::LoadSecurity(char *seclib, char *cfn)
{
   // Load security framework and plugins, if not already done

   // Make sure the input config file is defined
   if (!cfn) {
      fgEDest.Emsg("LoadSecurity","config file not specified");
      return 0;
   }

   // Open the security library
   void *lh = 0;
   if (!(lh = dlopen(seclib, RTLD_NOW))) {
      fgEDest.Emsg("LoadSecurity",dlerror(),"opening shared library",seclib);
      return 0;
   }

   // Get the server object creator
   XrdSecServLoader_t ep = 0;
   if (!(ep = (XrdSecServLoader_t)dlsym(lh, "XrdSecgetService"))) {
      fgEDest.Emsg("LoadSecurity", dlerror(),
                  "finding XrdSecgetService() in", seclib);
      return 0;
   }

   // Extract in a temporary file the directives prefixed "xpd.sec..." (filtering
   // out the prefix), "sec.protocol" and "sec.protparm"
   int nd = 0;
   char *rcfn = FilterSecConfig(cfn, nd);
   if (!rcfn) {
      if (nd == 0) {
         // No directives to be processed
         fgEDest.Emsg("LoadSecurity",
                     "no security directives: strong authentication disabled");
         return 0;
      }
      // Failure
      fgEDest.Emsg("LoadSecurity", "creating temporary config file");
      return 0;
   }

   // Get the server object
   XrdSecService *cia = 0;
   if (!(cia = (*ep)(fgEDest.logger(), rcfn))) {
      fgEDest.Emsg("LoadSecurity",
                  "Unable to create security service object via", seclib);
      return 0;
   }
   // Notify
   fgEDest.Emsg("LoadSecurity", "strong authentication enabled");

   // Unlink the temporary file and cleanup its path
   unlink(rcfn);
   delete[] rcfn;

   // All done
   return cia;
}

#if defined(USE_SIGCHLD)
//______________________________________________________________________________
static void SigChild(int)
{
   int         pid;
#if defined(__hpux) || defined(__FreeBSD__) || defined(__OpenBSD__) || \
    defined(__APPLE__)
   int status;
#else
   union wait  status;
#endif

   while ((pid = wait3(&status, WNOHANG, 0)) > 0)
      ;
}
#endif

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

      if (pi->Port < 0)
         return 1093;
      return pi->Port;
}}

//__________________________________________________________________________________
XrdProofdProtocol::XrdProofdProtocol()
   : XrdProtocol("xproofd protocol handler"), fProtLink(this)
{
   // Protocol constructor

   // Init local vars
   fLink      = 0;
   fArgp      = 0;
   fClientID  = 0;
   fPClient   = 0;
   fClient    = 0;
   fAuthProt  = 0;
   fBuff      = 0;
   fTopClient = 0;
   fSrvType   = kXPD_TopMaster;
   fUNIXSock = 0;
   fUNIXSockPath = 0;

   // Instantiate a Proofd protocol object
   Reset();
}

//______________________________________________________________________________
char *XrdProofdProtocol::Expand(char *p)
{
   // Expand path 'p' relative to:
   //     $HOME               if begins with ~/
   //     <user>'s $HOME      if begins with ~<user>/
   //     $PWD                if does not begin with '/' or '~'
   // The returned array of chars is the result of reallocation
   // of the input one

   // Make sure there soething to expand
   if (!p || strlen(p) <= 0 || p[0] == '/')
      return p;

   char *po = p;

   // Relative to the local location
   if (p[0] != '~') {
      if (getenv("PWD")) {
         int lpwd = strlen(getenv("PWD"));
         int lp = strlen(p);
         po = (char *) malloc(lp + lpwd + 2);
         if (po) {
            memcpy(po, getenv("PWD"), lpwd);
            memcpy(po+lpwd+1, p, lp);
            po[lpwd] = '/';
            free(p);
         } else
            po = p;
      }
      return po;
   }

   // Relative to $HOME or <user>'s $HOME
   if (p[0] == '~') {
      char *pu = p+1;
      char *pd = strchr(pu,'/');
      *pd++ = '\0';
      // Get the correct user structure
      struct passwd *pw = 0;
      if (strlen(pu) > 0)
         pw = getpwnam(pu);
      else
         pw = getpwuid(getuid());
      if (pw) {
         int ldir = strlen(pw->pw_dir);
         int lpd = strlen(pd);
         po = (char *) malloc(lpd + ldir + 2);
         if (po) {
            memcpy(po, pw->pw_dir, ldir);
            memcpy(po+ldir+1, pd, lpd);
            po[ldir] = '/';
            po[lpd + ldir + 1] = 0;
            free(p);
         } else
            po = p;
      }
      return po;
   }

   // We are done
   return po;
}

//______________________________________________________________________________
XrdProtocol *XrdProofdProtocol::Match(XrdLink *lp)
{
   // Check whether the request matches this protocol

   struct ClientInitHandShake hsdata;
   char  *hsbuff = (char *)&hsdata;

   static hs_response_t hsresp = {0, 0, htonl(1), htonl(XPROOFD_VERSBIN), 0};

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
   xp->fResponse.Set(lp);
   strcpy(xp->fEntity.prot, "host");
   xp->fEntity.host = strdup((char *)lp->Host());

   // Dummy data use dby 'proofd'
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

   fStatus = 0;
   fArgp   = 0;
   fLink   = 0;
   fhcPrev = 13;
   fhcNext = 21;
   fhcNow  = 13;

   // Default mode is query
   fPClient = 0;

   // This will be the unique client identifier
   if (fClientID)
      delete[] fClientID;
   fClientID = 0;

   fClient = 0;
   if (fAuthProt) {
      fAuthProt->Delete();
      fAuthProt = 0;
   }
   memset(&fEntity, 0, sizeof(fEntity));

   // Unix socket
   SafeDelete(fUNIXSock);
   if (fUNIXSockPath) {
      unlink(fUNIXSockPath);
      delete[] fUNIXSockPath;
   }
   fUNIXSockPath = 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Configure(char *parms, XrdProtocol_Config *pi)
{
   // Protocol configuration tool
   // Function: Establish configuration at load time.
   // Output: 1 upon success or 0 otherwise.

   // Only once
   if (fgConfigDone)
      return 1;
   fgConfigDone = 1;

   // Copy out the special info we want to use at top level
   fgEDest.logger(pi->eDest->logger());
   XrdProofdTrace = new XrdOucTrace(&fgEDest);
   fgSched        = pi->Sched;
   fgBPool        = pi->BPool;
   fgReadWait     = pi->readWait;
   fgPort         = pi->Port;

   // Debug flag
   if (pi->DebugON)
      XrdProofdTrace->What = TRACE_ALL;

   // Local FQDN
   char *host = XrdNetDNS::getHostName();
   fgLocalHost = host ? host : "";
   SafeFree(host);
   // Default pool entry point is this host
   fgPoolURL = strdup(fgLocalHost.c_str());

   // Pre-initialize some i/o values
   fgMaxBuffsz = fgBPool->MaxSize();

   // Process the config file for directives meaningful to us
   if (pi->ConfigFN && XrdProofdProtocol::Config(pi->ConfigFN))
      return 0;

   // Now process and configuration parameters: if we are not run as
   // default protocol those specified on the xrd.protocol line have
   // priority
   char *pe = parms;

   // Find out main ROOT directory
   char *rootsys = parms ? (char *)strstr(parms, "rootsys:") : 0;
   if (rootsys) {
      rootsys += 8;
      pe = (char *)strstr(rootsys, " ");
      if (pe) *pe = 0;
   } else if (!fgROOTsys) {
      // Try also the ROOTSYS env
      if (getenv("ROOTSYS")) {
         rootsys = getenv("ROOTSYS");
      } else {
         fgEDest.Say(0, "Configure: ROOTSYS location missing - unloading");
         return 0;
      }
   }
   if (rootsys) {
      SafeFree(fgROOTsys);
      fgROOTsys = strdup(rootsys);
   }
   fgEDest.Say(0, "Configure: using ROOTSYS: ", fgROOTsys);

   // External server application to be launched
   fgPrgmSrv = (char *) malloc(strlen(fgROOTsys) + strlen("/bin/proofserv") + 2);
   if (!fgPrgmSrv) {
      fgEDest.Say(0, "Configure:"
                  " could not allocate space for the server application path");
      return 0;
   }
   sprintf(fgPrgmSrv, "%s/bin/proofserv", fgROOTsys);
   fgEDest.Say(0, "Configure: PROOF server application: ", fgPrgmSrv);

   // Find out timeout on internal communications
   char *pw = pe ? (char *)strstr(pe+1, "intwait:") : 0;
   if (pw) {
      pe = (char *)strstr(pw, " ");
      if (pe) *pe = 0;
      fgInternalWait = strtol(pw+8, 0, 10);
      fgEDest.Say(0, "Configure: setting internal timeout to (secs): ", pw+8);
   }

   // Find out if a specific temporary directory is required
   char *tmp = parms ? (char *)strstr(parms, "tmp:") : 0;
   if (tmp)
      tmp += 5;
   fgTMPdir = tmp ? strdup(tmp) : strdup("/tmp");
   fgEDest.Say(0, "Configure: using temp dir: ", fgTMPdir);

   // Initialize the security system if this is wanted
   if (!fgSecLib)
      fgEDest.Say(0, "XRD seclib not specified; strong authentication disabled");
   else {
      TRACE(DEBUG, "Loading security library " <<fgSecLib);
      if (!(fgCIA = XrdProofdProtocol::LoadSecurity(fgSecLib, pi->ConfigFN))) {
         fgEDest.Emsg(0, "Configure: unable to load security system.");
         return 0;
      }
   }

   // Notify role
   const char *roles[] = { "any", "worker", "submaster", "master" };
   fgEDest.Say(0, "Configure: role set to: ", roles[fgSrvType+1]);

   // Notify allow rules
   if (fgSrvType == kXPD_WorkerServer || fgSrvType == kXPD_MasterServer) {
      if (fgMastersAllowed.size() > 0) {
         std::list<XrdOucString *>::iterator i;
         for (i = fgMastersAllowed.begin(); i != fgMastersAllowed.end(); ++i)
            fgEDest.Say(0, "Configure: masters allowed to connect: ", (*i)->c_str());
      } else {
            fgEDest.Say(0, "Configure: masters allowed to connect: any");
      }
   }

   // Notify change priority rules
   if (fgPriorities.size() > 0) {
      std::list<XrdProofdPriority *>::iterator i;
      for (i = fgPriorities.begin(); i != fgPriorities.end(); ++i) {
         XrdOucString msg("priority will be changed by ");
         msg += (*i)->fDeltaPriority;
         msg += " for user(s): ";
         msg += (*i)->fUser;
         fgEDest.Say(0, "Configure: ", msg.c_str());
      }
   } else {
      fgEDest.Say(0, "Configure: no priority changes requested");
   }

   // Check image
   if (!fgImage) {
      // Use the local host name
      fgImage = XrdNetDNS::getHostName();
   }
   fgEDest.Say(0, "Configure: image set to: ", fgImage);

   if (fgSrvType != kXPD_WorkerServer || fgSrvType == kXPD_AnyServer) {
      // Pool and namespace
      fgEDest.Say(0, "Configure: PROOF pool: ", fgPoolURL);
      fgEDest.Say(0, "Configure: PROOF pool namespace: ", fgNamespace);

      if (fgResourceType == kRTStatic) {
         // Initialize the list of workers if a static config has been required
         // Default file path, if none specified
         if (!fgPROOFcfg) {
            const char *cfg = "/proof/etc/proof.conf";
            fgPROOFcfg = new char[strlen(fgROOTsys)+strlen(cfg)+1];
            sprintf(fgPROOFcfg, "%s%s", fgROOTsys, cfg);
            // Check if the file exists and is readable
            if (access(fgPROOFcfg, R_OK)) {
               fgEDest.Say(0, "Configure: PROOF config file cannot be read: ", fgPROOFcfg);
               SafeFree(fgPROOFcfg);
               return 0;
            }
         }
         fgEDest.Say(0, "Configure: PROOF config file: ", fgPROOFcfg);
         // Load file content in memory
         if (ReadPROOFcfg() != 0) {
            fgEDest.Say(0, "Configure: unable to find valid information"
                           "in PROOF config file", fgPROOFcfg);
            SafeFree(fgPROOFcfg);
            return 0;
         }
         const char *st[] = { "disabled", "enabled" };
         fgEDest.Say(0, "Configure: user config files are ", st[fgWorkerUsrCfg]);
     }
   }


   // Test forking and get PROOF server protocol version
   if (SetSrvProtVers() < 0) {
      fgEDest.Say(0, "Configure: forking test failed");
      return 0;
   }
   XrdOucString mp("Configure: PROOF server protocol number: ");
   mp += fgSrvProtVers;
   fgEDest.Say(0, mp.c_str());

   // Schedule protocol object cleanup
   fgProtStack.Set(pi->Sched, XrdProofdTrace, TRACE_MEM);
   fgProtStack.Set(pi->ConnOptn, pi->ConnLife);

   // Initialize the request ID generation object
   XrdProofdReqID = new XrdOucReqID((int)fgPort, pi->myName,
                                    XrdNetDNS::IPAddr(pi->myAddr));

   // Indicate we configured successfully
   fgEDest.Say(0, "XProofd protocol version " XPROOFD_VERSION
               " build " XrdVERSION " successfully loaded.");

   // Return success
   return 1;
}

//______________________________________________________________________________
bool XrdProofdProtocol::CheckMaster(const char *m)
{
   // Check if master 'm' is allowed to connect to this host
   bool rc = 1;

   if (fgMastersAllowed.size() > 0) {
      rc = 0;
      XrdOucString wm(m);
      std::list<XrdOucString *>::iterator i;
      for (i = fgMastersAllowed.begin(); i != fgMastersAllowed.end(); ++i) {
         if (wm.matches((*i)->c_str())) {
            rc = 1;
            break;
         }
      }
   }

   // We are done
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::CheckIf(XrdOucStream *s)
{
   // Check existence and match condition of an 'if' directive
   // If none (valid) is found, return -1.
   // Else, return number of chars matching.

   // There must be an 'if'
   char *val = s ? s->GetToken() : 0;
   if (!val || strncmp(val,"if",2))
      return -1;

   // check value if any
   val = s->GetToken();
   if (!val)
      return -1;

   // Notify
   TRACE(DEBUG, "CheckIf: <pattern>: " <<val);

   // Return number of chars matching
   return fgLocalHost.matches((const char *)val);
}

//______________________________________________________________________________
int XrdProofdProtocol::Config(const char *cfn)
{
   // Scan the config file

   XrdOucStream Config(&fgEDest, getenv("XRDINSTANCE"));
   char *var;
   int cfgFD, NoGo = 0;
   int nmRole = -1, nmRootSys = -1, nmTmp = -1, nmInternalWait = -1,
       nmMaxSessions = -1, nmImage = -1, nmWorkDir = -1,
       nmPoolUrl = -1, nmNamespace = -1;

   // Open and attach the config file
   if ((cfgFD = open(cfn, O_RDONLY, 0)) < 0)
      return fgEDest.Emsg("Config", errno, "open config file", cfn);
   Config.Attach(cfgFD);

   // Process items
   char mess[512];
   char *val = 0;
   while ((var = Config.GetMyFirstWord())) {
      if (!(strncmp("xrootd.seclib", var, 13))) {
         if ((val = Config.GetToken()) && val[0]) {
            SafeFree(fgSecLib);
            fgSecLib = strdup(val);
         }
      } else if (!(strncmp("xpd.", var, 4)) && var[4]) {
         var += 4;
         // Get the value
         val = Config.GetToken();
         if (val && val[0]) {
            sprintf(mess,"Processing '%s = %s [if <pattern>]'", var, val);
            TRACE(DEBUG, "Config: " <<mess);
            // Treat first those not supporting 'if <pattern>'
            if (!strcmp("resource",var)) {
               // Specifies the resource broker
               if (!strcmp("static",val)) {
                  /* Using a config file; format of the remaining tokens is
                  // [<cfg_file>] [ucfg:<user_cfg_opt>] \
                  //              [wmx:<max_workers>] [selopt:<selection_mode>]
                  // where:
                  //         <cfg_file>          path to the config file
                  //                            [$ROOTSYS/proof/etc/proof.conf]
                  //         <user_cfg_opt>     "yes"/"no" enables/disables user
                  //                            private config files ["no"].
                  //                            If enable, the default file path
                  //                            is $HOME/.proof.conf (can be changed
                  //                            as option to TProof::Open() ).
                  //         <max_workers>       maximum number of workers per user
                  //                            [all]
                  //         <selection_mode>   selection mode in case not all the
                  //                            workers have to be allocated.
                  //                            Options: "rndm", "rrobin"
                  //                            ["rrobin"] */
                  fgResourceType = kRTStatic;
                  while ((val = Config.GetToken()) && val[0]) {
                     XrdOucString s(val);
                     if (s.beginswith("ucfg:")) {
                        fgWorkerUsrCfg = s.endswith("yes") ? 1 : 0;
                     } else if (s.beginswith("wmx:")) {
                        s.replace("wmx:","");
                        fgWorkerMax = strtol(s.c_str(), (char **)0, 10);
                     } else if (s.beginswith("selopt:")) {
                        fgWorkerSel = kSSORoundRobin;
                        if (s.endswith("random"))
                           fgWorkerSel = kSSORandom;
                     } else {
                        // Config file
                        SafeFree(fgPROOFcfg);
                        fgPROOFcfg = strdup(val);
                        fgPROOFcfg = Expand(fgPROOFcfg);
                        // Make sure it exists and can be read
                        if (access(fgPROOFcfg, R_OK)) {
                           fgEDest.Say(0, "Config: configuration file cannot be read: ", fgPROOFcfg);
                           SafeFree(fgPROOFcfg);
                        }
                     }
                  }
               }
            } else if (!strcmp("priority",var)) {
               // Priority change directive: get delta_priority
               int dp = strtol(val,0,10);
               XrdProofdPriority *p = new XrdProofdPriority("*", dp);
               // Check if an 'if' condition is present
               if ((val = Config.GetToken()) && !strncmp(val,"if",2)) {
                  if ((val = Config.GetToken()) && val[0]) {
                     p->fUser = val;
                  }
               }
               // Add to the list
               fgPriorities.push_back(p);
            } else if (!strcmp("seclib",var)) {
               // Record the path
               SafeFree(fgSecLib);
               fgSecLib = strdup(val);
            //
            // The following ones support the 'if <pattern>'
            } else {
               // Save 'val' first
               XrdOucString tval = val;
               // Number of matching chars: the parameter will be updated only
               // if condition is absent or equivalent/better matching
               int nm = CheckIf(&Config);
               // Now check
               if (!strcmp("rootsys", var)) {
                  // ROOTSYS path
                  XPDSETSTRING(nm, nmRootSys, fgROOTsys, tval);
               } else if (!strcmp("tmp",var)) {
                  // TMP directory
                  XPDSETSTRING(nm, nmTmp, fgTMPdir, tval);
               } else if (!strcmp("intwait",var)) {
                  // Internal time out
                  XPDSETINT(nm, nmInternalWait, fgInternalWait, tval);
               } else if (!strcmp("maxsessions",var)) {
                  // Max number of sessions per user
                  XPDSETINT(nm, nmMaxSessions, fgMaxSessions, tval);
               } else if (!strcmp("image",var)) {
                  // Image name of this server
                  XPDSETSTRING(nm, nmImage, fgImage, tval);
              } else if (!strcmp("workdir",var)) {
                  // Workdir for this server
                  XPDSETSTRING(nm, nmWorkDir, fgWorkDir, tval);
               } else if (!strcmp("allow",var)) {
                  // Masters allowed to connect
                  if (nm == -1 || nm > 0)
                     fgMastersAllowed.push_back(new XrdOucString(tval));
               } else if (!strcmp("poolurl",var)) {
                  // Local pool entry point
                  XPDSETSTRING(nm, nmPoolUrl, fgPoolURL, tval);
               } else if (!strcmp("namespace",var)) {
                  // Local pool entry point
                  XPDSETSTRING(nm, nmNamespace, fgNamespace, tval);
               } else if (!strcmp("role",var)) {
                  // Role this server
                  if (XPDCOND(nm, nmRole)) {
                     if (tval == "master")
                        fgSrvType = kXPD_TopMaster;
                     else if (tval == "submaster")
                        fgSrvType = kXPD_MasterServer;
                     else if (tval == "worker")
                        fgSrvType = kXPD_WorkerServer;
                     // New reference
                     nmRole = nm;
                  }
               }
            }
         } else {
            sprintf(mess,"%s not specified", var);
            fgEDest.Emsg("Config", mess);
         }
      }
   }
   return NoGo;
}

#undef  TRACELINK
#define TRACELINK fLink
#undef  RESPONSE
#define RESPONSE fResponse
//______________________________________________________________________________
int XrdProofdProtocol::Process(XrdLink *)
{
   // Process the information received on teh active link.
   // (We ignore the argument here)

   int rc = 0;
   TRACEP(REQ,"Process: XrdProofdProtocol =" <<this);

   // Read the next request header
   if ((rc = GetData("request",(char *)&fRequest,sizeof(fRequest))) != 0)
      return rc;
   TRACEP(REQ,"Process: sizeof(Request) = " <<sizeof(fRequest)<<", rc = "<<rc);

   // Deserialize the data
   TRACEP(REQ, "Process: " <<
               " req=" <<fRequest.header.requestid <<" dlen=" <<fRequest.header.dlen);
   fRequest.header.requestid = ntohs(fRequest.header.requestid);
   fRequest.header.dlen      = ntohl(fRequest.header.dlen);

   // The stream ID for the reply
   { XrdOucMutexHelper mh(fResponse.fMutex);
      fResponse.Set(fRequest.header.streamid);
   }
   unsigned short sid;
   memcpy((void *)&sid, (const void *)&(fRequest.header.streamid[0]), 2);
   TRACEP(REQ, "Process: sid=" << sid <<
               " req=" <<fRequest.header.requestid <<" dlen=" <<fRequest.header.dlen);

   // Every request has an associated data length. It better be >= 0 or we won't
   // be able to know how much data to read.
   if (fRequest.header.dlen < 0) {
      fResponse.Send(kXR_ArgInvalid, "Process: Invalid request data length");
      return fLink->setEtext("Process: protocol data length error");
   }

   // Read any argument data at this point, except when the request is a write.
   // The argument may have to be segmented and we're not prepared to do that here.
   if (fRequest.header.requestid != kXR_write && fRequest.header.dlen) {
      if (!fArgp || fRequest.header.dlen+1 > fArgp->bsize) {
         if (fArgp)
            fgBPool->Release(fArgp);
         if (!(fArgp = fgBPool->Obtain(fRequest.header.dlen+1))) {
            fResponse.Send(kXR_ArgTooLong, "fRequest.argument is too long");
            return 0;
         }
         fhcNow = fhcPrev; fhalfBSize = fArgp->bsize >> 1;
      }
      if ((rc = GetData("arg", fArgp->buff, fRequest.header.dlen)))
         return rc;
      fArgp->buff[fRequest.header.dlen] = '\0';
   }
   TRACEP(REQ,"Process: fArgp->buff = "<< (fArgp ? fArgp->buff : ""));

   // Continue with request processing at the resume point
   return Process2();
}

//______________________________________________________________________________
int XrdProofdProtocol::Process2()
{
   // Local processing method: here the request is dispatched to the appropriate
   // method

   // If the user is not yet logged in, restrict what the user can do
   if (!fStatus || !(fStatus & XPD_LOGGEDIN))
      switch(fRequest.header.requestid) {
      case kXP_auth:
         return Auth();
      case kXP_login:
         return Login();
      default:
         TRACEP(REQ,"Process2: invalid request: " <<fRequest.header.requestid);
         fResponse.Send(kXR_InvalidRequest,"Invalid request; user not logged in");
         return fLink->setEtext("protocol sequence error 1");
      }

   // Once logged-in, the user can request the real actions
   switch(fRequest.header.requestid) {
   case kXP_create:
      return Create();
   case kXP_destroy:
      return Destroy();
   case kXP_sendmsg:
      return SendMsg();
   case kXP_attach:
      return Attach();
   case kXP_detach:
      return Detach();
   case kXP_admin:
      return Admin();
   case kXP_interrupt:
      return Interrupt();
   case kXP_ping:
      return Ping();
   default:
      break;
   }

   // Whatever we have, it's not valid
   fResponse.Send(kXR_InvalidRequest, "Invalid request code");
   return 0;
}

//______________________________________________________________________
void XrdProofdProtocol::Recycle(XrdLink *, int, const char *)
{
   // Recycle call. Release the instance and give it back to the stack.

   // Document the disconnect
   TRACEP(REQ,"Recycle: XrdProofdProtocol =" <<this<<" : recycling");

   // This part may be not thread safe
   XrdOucMutexHelper mtxh(&fgXPDMutex);

   // If we have a buffer, release it
   if (fArgp) {
      fgBPool->Release(fArgp);
      fArgp = 0;
   }

   // Drop this instance from the list
   XrdProofClient *pmgr = 0;
   if (fgProofClients.size() > 0) {
      std::list<XrdProofClient *>::iterator i;
      for (i = fgProofClients.begin(); i != fgProofClients.end(); ++i) {
         if ((pmgr = *i) && pmgr->Match(fClientID))
               break;
         pmgr = 0;
      }
   }

   // Loop over servers sessions associated to this client and update
   // their attached client vectors
   if (pmgr && pmgr->fProofServs.size() > 0) {
      XrdProofServProxy *psrv = 0;
      int is = 0;
      for (is = 0; is < (int) pmgr->fProofServs.size(); is++) {
         if ((psrv = pmgr->fProofServs.at(is))) {
            // Release CIDs in attached sessions: loop over attached clients
            XrdClientID *cid = 0;
            int ic = 0;
            for (ic = 0; ic < (int) psrv->fClients.size(); ic++) {
               if ((cid = psrv->fClients.at(ic))) {
                  if (cid->fP == this)
                     cid->Reset();
               }
            }
         }
      }
   }

   // Reset the corresponding client slot in the list of this client
   if (pmgr) {
      int ic = 0;
      for (ic = 0; ic < (int) pmgr->fClients.size(); ic++) {
         if (this == pmgr->fClients.at(ic))
            pmgr->fClients[ic] = 0;
      }
   }

   // Set fields to starting point (debugging mostly)
   Reset();

   // Push ourselves on the stack
   fgProtStack.Push(&fProtLink);
}

//__________________________________________________________________________
char *XrdProofdProtocol::FilterSecConfig(const char *cfn, int &nd)
{
   // Grep directives of the form "xpd.sec...", "sec.protparm" and
   // "sec.protocol" from file 'cfn' and save them in a temporary file,
   // stripping off the prefix "xpd." when needed.
   // If any such directory is found, the full path of the temporary file
   // is returned, with the number of directives found in 'nd'.
   // Otherwise 0 is returned and '-errno' specified in nd.
   // The caller has the responsability to unlink the temporary file and
   // to release the memory allocated for the path.

   static const char *pfx[] = { "xpd.sec.", "sec.protparm", "sec.protocol" };
   char *rcfn = 0;

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
          !strncmp(lin, pfx[2], strlen(pfx[2]))) {
         // Target directive found
         nd++;
         // Create the output file, if not yet done
         if (!rcfn) {
            rcfn = new char[strlen(fgTMPdir) + strlen("/xpdcfn_XXXXXX") + 2];
            sprintf(rcfn, "%s/xpdcfn_XXXXXX", fgTMPdir);
            if ((fd = mkstemp(rcfn)) < 0) {
               delete[] rcfn;
               nd = (errno > 0) ? -errno : -1;
               fclose(fin);
               rcfn = 0;
               return rcfn;
            }
         }

         // Starting char
         int ns = (!strncmp(lin, pfx[0], strlen(pfx[0]))) ? 4 : 0 ;
         // Write the line to the output file, stripping the prefix "xpd."
         while (write(fd, &lin[ns], strlen(lin)-ns) < 0 && errno == EINTR)
            errno = 0;

      }
   }

   // Close files
   fclose(fin);
   close(fd);

   return rcfn;
}

//__________________________________________________________________________
int XrdProofdProtocol::ReadPROOFcfg()
{
   // Read PROOF config file and load the information in memory in the
   // fgWorkerList.
   // NB: 'master' information here is ignored, because it is passed
   //     via the 'xpd.workdir' and 'xpd.image' config directives
   static bool alreadyRead = 0;

   // File should be loaded only once
   if (alreadyRead)
      return 0;

   // Open the defined path.
   FILE *fin = 0;
   if (!fgPROOFcfg || !(fin = fopen(fgPROOFcfg, "r")))
      return -1;
   alreadyRead = 1;

   // Reserve some space
   int allocsz = 10;
   fgWorkers.reserve(allocsz);

   // Create a default master line
   XrdOucString mm("master ",128);
   mm += fgImage; mm += " image="; mm += fgImage;
   if (fgWorkDir) {
      mm += " workdir="; mm += fgWorkDir;
   }
   fgWorkers.push_back(new XrdProofWorker(mm.c_str()));

   // Read now the directives
   int nw = 1;
   char lin[2048];
   while (fgets(lin,sizeof(lin),fin)) {
      // Skip empty lines
      int p = 0;
      while (lin[p++] == ' ') { ; } p--;
      if (lin[p] == '\0' || lin[p] == '\n')
         continue;

      // Skip comments
      if (lin[0] == '#')
         continue;

      // Remove trailing '\n';
      if (lin[strlen(lin)-1] == '\n')
         lin[strlen(lin)-1] = '\0';

      TRACE(DEBUG, "ReadPROOFcfg: found line: " << lin);

      const char *pfx[2] = { "master", "node" };
      if (!strncmp(lin, pfx[0], strlen(pfx[0])) ||
          !strncmp(lin, pfx[1], strlen(pfx[1]))) {
         // Init a worker instance
         XrdProofWorker *pw = new XrdProofWorker(lin);
         if (fgLocalHost.matches(pw->fHost.c_str())) {
            // Replace the default line (the first with what found in the file)
            fgWorkers[0]->Reset(lin);
            // If the image was not specified use the default
            if (fgWorkers[0]->fImage == "")
               fgWorkers[0]->fImage = fgImage;
         }
         SafeDelete(pw); 
     } else {
         // If not, allocate a new one; we need to resize (double it)
         if (nw >= (int)fgWorkers.capacity())
            fgWorkers.reserve(fgWorkers.capacity() + allocsz);

         // Build the worker object
         fgWorkers.push_back(new XrdProofWorker(lin));

         nw++;
      }
   }

   // Close files
   fclose(fin);

   // We are done
   return ((nw == 0) ? -1 : 0);
}

//__________________________________________________________________________
int XrdProofdProtocol::GetWorkers(XrdOucString &lw, XrdProofServProxy *xps)
{
   // Get a list of workers from the available resource broker
   int rc = 0;

   // Static
   if (fgResourceType == kRTStatic) {
      if (fgWorkerMax > 0 && fgWorkerMax < (int) fgWorkers.size()) {

         // Partial list: the master line first
         lw += fgWorkers[0]->Export();
         // Add separator
         lw += '&';
         // Count
         xps->AddWorker(fgWorkers[0]);
         fgWorkers[0]->fProofServs.push_back(xps);
         fgWorkers[0]->fActive++;

         // Now the workers
         if (fgWorkerSel == kSSORandom) {
            // Random: the first time init the machine
            static bool rndmInit = 0;
            if (!rndmInit) {
               const char *randdev = "/dev/urandom";
               int fd;
               unsigned int seed;
               if ((fd = open(randdev, O_RDONLY)) != -1) {
                  read(fd, &seed, sizeof(seed));
                  srand(seed);
                  close(fd);
                  rndmInit = 1;
               }
            }
            // Selection
            std::vector<int> walloc(fgWorkers.size(), 0);
            int nw = fgWorkerMax;
            while (nw--) {
               // Normalized number
               int maxAtt = 10000, natt = 0;
               int iw = -1;
               while ((iw < 0 || iw >= (int)fgWorkers.size()) && natt < maxAtt) {
                  iw = rand() % fgWorkers.size();
                  if (iw > 0 && iw < (int)fgWorkers.size() && walloc[iw] == 0) {
                     walloc[iw] = 1;
                  } else {
                     natt++;
                     iw = -1;
                  }
               }

               if (iw > -1) {
                  // Add export version of the info
                  lw += fgWorkers[iw]->Export();
                  // Add separator
                  lw += '&';
                  // Count
                  xps->AddWorker(fgWorkers[iw]);
                  fgWorkers[iw]->fProofServs.push_back(xps);
                  fgWorkers[iw]->fActive++;
               } else {
                  // Unable to generate the right number
                  fgEDest.Emsg("GetWorkers", "Random generation failed");
                  rc = -1;
                  break;
               }
            }

         } else {
            // The first one is the master line
            static int rrNext = 1;
            // Round-robin (default)
            // Make sure the first one is in the range
            if (rrNext > (int)(fgWorkers.size()-1))
               rrNext = 1;
            // Create the serialized string to be communicated to proofserv
            int nw = fgWorkerMax;
            while (nw--) {
               // Add export version of the info
               lw += fgWorkers[rrNext]->Export();
               // Add separator
               lw += '&';
               // Count
               xps->AddWorker(fgWorkers[rrNext]);
               fgWorkers[rrNext]->fProofServs.push_back(xps);
               fgWorkers[rrNext++]->fActive++;
               if (rrNext > (int)(fgWorkers.size()-1))
                  rrNext = 1;
            }
         }
      } else {
         // The full list
         int iw = 0;
         for (; iw < (int)fgWorkers.size() ; iw++) {
            // Add export version of the info
            lw += fgWorkers[iw]->Export();
            // Add separator
            lw += '&';
            // Count
            xps->AddWorker(fgWorkers[iw]);
            fgWorkers[iw]->fProofServs.push_back(xps);
            fgWorkers[iw]->fActive++;
         }
      }
   } else {
      fgEDest.Emsg("GetWorkers", "Resource type implemented: do nothing");
      rc = -1;
   }
   return rc;
}

//__________________________________________________________________________
int XrdProofdProtocol::GetFreeServID()
{
   // Get next free server ID. If none is found, increase the vector size
   // and get the first new one

   XrdOucMutexHelper mh(fMutex);

   TRACEP(REQ,"GetFreeServID: size = "<<fPClient->fProofServs.size()<<
              "; capacity = "<<fPClient->fProofServs.capacity());
   int ic = 0;
   // Search for free places in the existing vector
   for (ic = 0; ic < (int)fPClient->fProofServs.size() ; ic++) {
      if (fPClient->fProofServs[ic] && !(fPClient->fProofServs[ic]->IsValid())) {
         fPClient->fProofServs[ic]->SetValid();
         return ic;
      }
   }

   // We may need to resize (double it)
   if (ic >= (int)fPClient->fProofServs.capacity()) {
      int newsz = 2 * fPClient->fProofServs.capacity();
      fPClient->fProofServs.reserve(newsz);
   }

   // Allocate new element
   fPClient->fProofServs.push_back(new XrdProofServProxy());

   TRACEP(REQ,"GetFreeServID: size = "<<fPClient->fProofServs.size()<<
              "; new capacity = "<<fPClient->fProofServs.capacity()<<"; ic = "<<ic);

   // We are done
   return ic;
}

//_______________________________________________________________________________
XrdProofServProxy *XrdProofdProtocol::GetServer(int psid)
{
   // Search the vector for a matching server

   XrdOucMutexHelper mh(fMutex);

   XrdProofServProxy *xps = 0;
   std::vector<XrdProofServProxy *>::iterator ip;
   for (ip = fPClient->fProofServs.begin(); ip != fPClient->fProofServs.end(); ++ip) {
      xps = *ip;
      if (xps && xps->Match(psid))
         break;
      xps = 0;
   }

   return xps;
}

//______________________________________________________________________________
void XrdProofdProtocol::EraseServer(int psid)
{
   // Erase server with id psid from the list

   XrdOucMutexHelper mh(fMutex);

   XrdProofServProxy *xps = 0;
   std::vector<XrdProofServProxy *>::iterator ip;
   for (ip = fPClient->fProofServs.begin(); ip != fPClient->fProofServs.end(); ++ip) {
      xps = *ip;
      if (xps && xps->Match(psid)) {
         fPClient->fProofServs.erase(ip);
         break;
      }
   }
}

//______________________________________________________________________________
int XrdProofdProtocol::Login()
{
   // Process a login request

   int rc = 1;

   // If this server is explicitely required to be a worker node or a
   // submaster, check whether the requesting host is allowed to connect
   if (fRequest.login.role[0] != 'i' &&
       fgSrvType == kXPD_WorkerServer || fgSrvType == kXPD_MasterServer) {
      if (!CheckMaster(fLink->Host())) {
         TRACEP(REQ,"Login: master not allowed to connect - "
                    "ignoring request ("<<fLink->Host()<<")");
         fResponse.Send(kXR_InvalidRequest,
                    "Login: master not allowed to connect - request ignored");
         return rc;
      }
   }

   // If this is the second call (after authentication) we just need
   // mapping
   if (fStatus == XPD_NEED_MAP) {
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
   char uname[9];

   // Unmarshall the data
   pid = (int)ntohl(fRequest.login.pid);
   for (i = 0; i < (int)sizeof(uname)-1; i++) {
      if (fRequest.login.username[i] == '\0' || fRequest.login.username[i] == ' ')
         break;
      uname[i] = fRequest.login.username[i];
   }
   uname[i] = '\0';

   // Establish the ID for this link
   fLink->setID(uname, pid);
   fCapVer = fRequest.login.capver[0];

   // Establish the ID for this client
   fClientID = new char[strlen(uname)+4];
   if (fClientID) {
      sprintf(fClientID, "%s", uname);
      TRACEP(REQ,"Login: ClientID =" << fClientID);
   } else {
      TRACEP(REQ,"Login: no space for ClientID");
      fResponse.Send(kXR_NoMemory, "Login: ClientID: out-of-resources");
      return rc;
   }

   // Find out the server type: 'i', internal, means this is a proofsrv calling back.
   // For the time being authentication is required for clients only.
   bool needauth = 0;
   switch (fRequest.login.role[0]) {
   case 'i':
      fSrvType = kXPD_Internal;
      break;
   case 'M':
      if (fgSrvType == kXPD_AnyServer || fgSrvType == kXPD_TopMaster) {
         fTopClient = 1;
         fSrvType = kXPD_TopMaster;
         needauth = 1;
      } else {
         TRACEP(REQ,"Login: top master mode not allowed - ignoring request");
         fResponse.Send(kXR_InvalidRequest,
                        "Server not allowed to be top master - ignoring request");
         return rc;
      }
      break;
   case 'm':
      if (fgSrvType == kXPD_AnyServer || fgSrvType == kXPD_MasterServer) {
         fSrvType = kXPD_MasterServer;
      } else {
         TRACEP(REQ,"Login: submaster mode not allowed - ignoring request");
         fResponse.Send(kXR_InvalidRequest,
                        "Server not allowed to be submaster - ignoring request");
         return rc;
      }
      break;
   case 's':
      if (fgSrvType == kXPD_AnyServer || fgSrvType == kXPD_WorkerServer) {
         fSrvType = kXPD_WorkerServer;
      } else {
         TRACEP(REQ,"Login: worker mode not allowed - ignoring request");
         fResponse.Send(kXR_InvalidRequest,
                        "Server not allowed to be worker - ignoring request");
         return rc;
      }
      break;
   default:
      TRACEP(REQ,"Login: unknown mode: '" << fRequest.login.role[0] <<"'");
      fResponse.Send(kXR_InvalidRequest,"Server type: invalide mode");
      return rc;
   }

   // Get the security token for this link. We will either get a token, a null
   // string indicating host-only authentication, or a null indicating no
   // authentication. We can then optimize of each case.
   if (needauth && fgCIA) {
      const char *pp = fgCIA->getParms(i, fLink->Name());
      if (pp && i ) {
         fResponse.Send((void *)pp, i);
         fStatus = (XPD_NEED_MAP | XPD_NEED_AUTH);
         return rc;
      } else {
         fResponse.Send();
         fStatus = XPD_LOGGEDIN;
         if (pp) {
            fEntity.tident = fLink->ID;
            fClient = &fEntity;
         }
      }
   } else {
      rc = fResponse.Send();
      fStatus = XPD_LOGGEDIN;
   }

   // Map the client
   return MapClient(1);
}

//______________________________________________________________________________
int XrdProofdProtocol::MapClient(bool all)
{
   // Process a login request
   int rc = 1;

   // Flag for internal connections
   bool proofsrv = ((fSrvType == kXPD_Internal) && all) ? 1 : 0;

   // If call back from proofsrv, find out the target session
   short int psid = -1;
   char protver = -1;
   if (proofsrv) {
      memcpy(&psid, (const void *)&(fRequest.login.reserved[0]), 2);
      if (psid < 0) {
         fResponse.Send(kXR_InvalidRequest,
                        "MapClient: proofsrv callback: sent invalid session id");
         return rc;
      }
      protver = fRequest.login.capver[0];
      TRACEP(REQ,"MapClient: proofsrv callback for session: " <<psid);
   }

   // This part may be not thread safe
   XrdOucMutexHelper mtxh(&fgXPDMutex);

   // Now search for an existing manager session for this ClientID
   XrdProofClient *pmgr = 0;
   TRACEP(REQ,"MapClient: # of clients: "<<fgProofClients.size());
   if (fgProofClients.size() > 0) {
      std::list<XrdProofClient *>::iterator i;
      for (i = fgProofClients.begin(); i != fgProofClients.end(); ++i) {
         if ((pmgr = *i) && pmgr->Match(fClientID))
            break;
         pmgr = 0;
      }
   }

   // Map the existing session, if found
   if (pmgr) {
      // Save as reference proof mgr
      fPClient = pmgr;
      TRACEP(REQ,"MapClient: matching client: "<<pmgr);

      // If proofsrv, locate the target session
      if (proofsrv) {
         XrdProofServProxy *psrv = 0;
         int is = 0;
         for (is = 0; is < (int) pmgr->fProofServs.size(); is++) {
            if ((psrv = pmgr->fProofServs.at(is)) && psrv->Match(psid))
               break;
            psrv = 0;
         }
         if (!psrv) {
            fResponse.Send(kXP_nosession, "MapClient: proofsrv callback:"
                           " wrong target session: protocol error");
            return rc;
         } else {
            // Set the protocol version
            psrv->fProtVer = protver;
            // Assign this link to it
            psrv->fLink = fLink;
            psrv->fProofSrv.Set(fLink);
            psrv->fProofSrv.Set(fRequest.header.streamid);
            TRACEP(REQ,"MapClient: proofsrv callback: link assigned to target session "<<psid);
         }
      } else {

         // The index of the next free slot will be the unique ID
         fCID = pmgr->GetClientID(this);
      }

   } else {

      // Proofsrv callbacks need something to attach to
      if (proofsrv) {
         fResponse.Send(kXP_nomanager,
                        "MapClient: proofsrv callback:"
                        " no manager to attach to: protocol error");
         return rc;
      }

      // No existing session: create a new one
      pmgr = new XrdProofClient(this);
      if (pmgr) {
         TRACEP(REQ,"MapClient: NEW client: "<<pmgr);

         // The index of the next free slot will be the unique ID
         fCID = pmgr->GetClientID(this);

         // Add to the list
         fgProofClients.push_back(pmgr);
         TRACEP(REQ,"MapClient: client "<<pmgr<<" added to the list");
         // Save as reference proof mgr
         fPClient = pmgr;
      }
   }

   if (!proofsrv) {
      TRACEP(REQ,"MapClient: fCID = "<<fCID<<"; size = "<<fPClient->fClients.size()<<
              "; capacity = "<<fPClient->fClients.capacity());
   }

   // UNIX socket for internal communications (to/from proofsrv)
   if (!fUNIXSock && !proofsrv) {
      fUNIXSock = new XrdNet(&fgEDest);
      fUNIXSockPath = new char[strlen(fgTMPdir)+strlen("/xpdsock_XXXXXX")+2];
      sprintf(fUNIXSockPath,"%s/xpdsock_XXXXXX",fgTMPdir);
      int fd = mkstemp(fUNIXSockPath);
      if (fd > -1) {
         close(fd);
         if (fUNIXSock->Bind(fUNIXSockPath)) {
            PRINT("MapClient: warning:"
                  " problems binding to UNIX socket; path: " <<fUNIXSockPath);
            return 0;
         } else
            TRACEP(REQ,"MapClient: path for UNIX for socket is " <<fUNIXSockPath);
      } else {
         PRINT("MapClient: unable to generate unique"
               " path for UNIX socket; tried path " << fUNIXSockPath);
         return 0;
      }

      // Set ownership of the socket file to the client
      if (getuid() == 0) {
         struct passwd *pw = getpwnam(fClientID);
         if (!pw) {
            PRINT("MapClient: client unknown to getpwnam");
            return 0;
         }
         if (chown(fUNIXSockPath, pw->pw_uid, pw->pw_gid) == -1) {
            PRINT("MapClient: cannot set user ownership on UNIX socket");
            return 0;
         }
      }
   }

   // Document this login
   if (!(fStatus & XPD_NEED_AUTH))
      fgEDest.Log(OUC_LOG_01, ":MapClient", fLink->ID, "login");

   return rc;
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

   TRACEP(REQ,"Auth: processing authentication request");

   // Ignore authenticate requests if security turned off
   if (!fgCIA)
      return fResponse.Send();
   cred.size   = fRequest.header.dlen;
   cred.buffer = fArgp->buff;

   // If we have no auth protocol, try to get it
   if (!fAuthProt) {
      fLink->Name(&netaddr);
      if (!(fAuthProt = fgCIA->getProtocol(fLink->Host(), netaddr, &cred, &eMsg))) {
         eText = eMsg.getErrText(rc);
         TRACEP(REQ,"Auth: user authentication failed; "<<eText);
         fResponse.Send(kXR_NotAuthorized, eText);
         return -EACCES;
      }
      fAuthProt->Entity.tident = fLink->ID;
   }

   // Now try to authenticate the client using the current protocol
   if (!(rc = fAuthProt->Authenticate(&cred, &parm, &eMsg))) {
      const char *msg = (fStatus & XPD_ADMINUSER ? "admin login as" : "login as");
      rc = fResponse.Send();
      fStatus &= ~XPD_NEED_AUTH;
      fClient = &fAuthProt->Entity;
      if (fClient->name)
         fgEDest.Log(OUC_LOG_01, ":Auth", fLink->ID, msg, fClient->name);
      else
         fgEDest.Log(OUC_LOG_01, ":Auth", fLink->ID, msg, " nobody");
      return rc;
   }

   // If we need to continue authentication, tell the client as much
   if (rc > 0) {
      TRACEP(REQ, "Auth: more auth requested; sz=" <<(parm ? parm->size : 0));
      if (parm) {
         rc = fResponse.Send(kXR_authmore, parm->buffer, parm->size);
         delete parm;
         return rc;
      }
      if (fAuthProt) {
         fAuthProt->Delete();
         fAuthProt = 0;
      }
      TRACEP(ALL,"Auth: security requested additional auth w/o parms!");
      fResponse.Send(kXR_ServerError,"invalid authentication exchange");
      return -EACCES;
   }

   // We got an error, bail out
   if (fAuthProt) {
      fAuthProt->Delete();
      fAuthProt = 0;
   }
   eText = eMsg.getErrText(rc);
   TRACEP(ALL,"Auth: user authentication failed; "<<eText);
   fResponse.Send(kXR_NotAuthorized, eText);
   return -EACCES;
}

//______________________________________________________________________________
int XrdProofdProtocol::GetData(const char *dtype, char *buff, int blen)
{
   // Get data from the open link

   int rlen;

   // Read the data but reschedule the link if we have not received all of the
   // data within the timeout interval.
   TRACEP(REQ, "GetData: dtype: "<<dtype<<", blen: "<<blen);

   rlen = fLink->Recv(buff, blen, fgReadWait);

   if (rlen  < 0)
      if (rlen != -ENOMSG) {
         TRACEP(REQ, "GetData: link read error");
         return fLink->setEtext("link read error");
      } else {
         TRACEP(REQ, "GetData: connection closed by peer");
         return -1;
      }
   if (rlen < blen) {
      fBuff = buff+rlen; fBlen = blen-rlen;
      TRACEP(REQ, "GetData: " << dtype <<
                  " timeout; read " <<rlen <<" of " <<blen <<" bytes");
      return 1;
   }
   TRACEP(REQ, "GetData: rlen: "<<rlen);

   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Attach()
{
   // Handle a request to attach to an existing session

   int psid = -1, rc = 1;

   // Unmarshall the data
   psid = ntohl(fRequest.proof.sid);
   TRACEP(REQ, "Attach: psid = "<<psid<<"; fCID = "<<fCID);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (INRANGE(psid,fPClient->fProofServs)) {
      xps = fPClient->fProofServs.at(psid);
   } else {
      TRACEP(REQ, "Attach: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
   }

   if (xps) {
      TRACEP(REQ, "Attach: xps: "<<xps<<", status: "<< xps->Status());

      // Stream ID
      unsigned short sid;
      memcpy((void *)&sid, (const void *)&(fRequest.header.streamid[0]), 2);

      // We associate this instance to the corresponding slot in the
      // session vector of attached clients
      XrdClientID *csid = xps->GetClientID(fCID);
      csid->fP = this;
      csid->fSid = sid;

      // Take parentship, if orphalin
      if (!(xps->fParent))
         xps->fParent = csid;

      // Notify to user
      fResponse.Send(psid, (kXR_int32)XPROOFD_VERSBIN, fClientID, strlen(fClientID));

      // Send saved query num message
      if (xps->fQueryNum) {
         TRACEP(REQ, "Attach: sending query num message ("<<
                     xps->fQueryNum->fSize<<" bytes)");
         fResponse.Send(kXR_attn, kXPD_msg,
                        xps->fQueryNum->fBuff, xps->fQueryNum->fSize);
      }
      // Send saved start processing message, if not idle
      if (xps->fStatus == kXPD_running && xps->fStartMsg) {
         TRACEP(REQ, "Attach: sending start process message ("<<
                     xps->fStartMsg->fSize<<" bytes)");
         fResponse.Send(kXR_attn, kXPD_msg,
                        xps->fStartMsg->fBuff, xps->fStartMsg->fSize);
      }

   } else {
      TRACEP(REQ, "Attach: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
   }

   // Over
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::Detach()
{
   // Handle a request to detach from an existing session

   int psid = -1, rc = 1;

   XrdOucMutexHelper mh(fMutex);

   // Unmarshall the data
   psid = ntohl(fRequest.proof.sid);
   TRACEP(REQ, "Detach: psid: "<<psid);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (INRANGE(psid,fPClient->fProofServs)) {
      xps = fPClient->fProofServs.at(psid);
   } else {
      TRACEP(REQ, "Detach: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
   }

   if (xps) {
      TRACEP(REQ, "Detach: xps: "<<xps<<", status: "<< xps->Status());

      XrdOucMutexHelper xpmh(xps->Mutex());

      // Remove this from the list of clients
      std::vector<XrdClientID *>::iterator i;
      TRACEP(REQ, "Detach: xps: "<<xps<<", # clients: "<< (xps->fClients).size());
      for (i = (xps->fClients).begin(); i != (xps->fClients).end(); ++i) {
         if (*i) {
            if ((*i)->fP == this) {
               delete (*i);
               xps->fClients.erase(i);
               break;
            }
         }
      }

      // Notify to user
      fResponse.Send();

   } else {
      // Notify to user
      fResponse.Send("Detach: server session not found: already cleaned?");
   }
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::Destroy()
{
   // Handle a request to shutdown an existing session

   int psid = -1, rc = 1;

   XrdOucMutexHelper mh(fPClient->fMutex);

   // Unmarshall the data
   psid = ntohl(fRequest.proof.sid);
   TRACEP(REQ, "Destroy: psid: "<<psid);

   // Find server session
   XrdProofServProxy *xpsref = 0;
   if (psid > -1) {
      // Request for a specific session
      if (INRANGE(psid,fPClient->fProofServs)) {
         xpsref = fPClient->fProofServs.at(psid);
      } else {
         TRACEP(REQ, "Destroy: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"session ID not found");
      }
   }

   // Loop over servers
   XrdProofServProxy *xps = 0;
   int is = 0;
   for (is = 0; is < (int) fPClient->fProofServs.size(); is++) {

      if ((xps = fPClient->fProofServs.at(is)) && (xpsref == 0 || xps == xpsref)) {

         TRACEP(REQ, "Destroy: xps: "<<xps<<", status: "<< xps->Status());

         {  XrdOucMutexHelper xpmh(xps->Mutex());

            if (xps->fSrvType == kXPD_TopMaster) {
               // Tell other attached clients, if any, that this session is gone
               if (fTopClient && xps->fClients.size() > 0) {
                  char msg[512] = {0};
                  snprintf(msg, 512, "Destroy: session: %s destroyed by: %s",
                           xps->Tag(), fLink->ID);
                  int len = strlen(msg);
                  int ic = 0;
                  XrdProofdProtocol *p = 0;
                  for (ic = 0; ic < (int) xps->fClients.size(); ic++) {
                     if ((p = xps->fClients.at(ic)->fP) &&
                         (p != this) && p->fTopClient)
                        p->fResponse.Send(kXR_attn, kXPD_srvmsg, msg, len);
                  }
               }
            }

            // Send a terminate signal to the proofserv
            if (xps->SrvID() > -1) {
               int type = 3;
               if (xps->fProofSrv.Send(kXR_attn, kXPD_interrupt, type) != 0) {
                  fResponse.Send(kXP_ServerError,
                                 "Destroy: could not send hard interrupt to proofsrv");
                  return rc;
               }
            }

            // Reset instance
            xps->Reset();

            // If single delete we are done
            if ((xpsref != 0 && (xps == xpsref)))
               break;
         }
      }

   }

   // Notify to user
   fResponse.Send();

   // Over
   return rc;
}

//______________________________________________________________________________
void XrdProofdProtocol::SetProofServEnv(int psid)
{
   // Set environment for proofserv
   char *ev = 0;

   // Set ROOTSYS
   ev = new char[15 + strlen(fgROOTsys)];
   sprintf(ev, "ROOTSYS=%s", fgROOTsys);
   putenv(ev);

   // Set conf dir
   ev = new char[20 + strlen(fgROOTsys)];
   sprintf(ev, "ROOTCONFDIR=%s", fgROOTsys);
   putenv(ev);

   // Set TMPDIR
   ev = new char[20 + strlen(fgTMPdir)];
   sprintf(ev, "ROOTTMPDIR=%s", fgTMPdir);
   putenv(ev);

   // Set Open socket
   ev = new char[20 + strlen(fUNIXSockPath)];
   sprintf(ev, "ROOTOPENSOCK=%s", fUNIXSockPath);
   putenv(ev);

   // Entity
   ev = new char[strlen(fClientID)+strlen(fLink->Host())+20];
   sprintf(ev, "ROOTENTITY=%s@%s", fClientID, fLink->Host());
   putenv(ev);

   // Session ID
   ev = new char[25];
   sprintf(ev, "ROOTSESSIONID=%d", psid);
   putenv(ev);

   // Client ID
   ev = new char[25];
   sprintf(ev, "ROOTCLIENTID=%d", fCID);
   putenv(ev);

   // Port (really needed?)
   ev = new char[25];
   sprintf(ev, "ROOTXPDPORT=%d", fgPort);
   putenv(ev);

   // Whether user specific config files are enabled
   if (fgWorkerUsrCfg) {
      ev = new char[strlen("ROOTUSEUSERCFG")+5];
      sprintf(ev, "ROOTUSEUSERCFG=1");
      putenv(ev);
   }

#ifndef ROOTLIBDIR
   char *ldpath = 0;
#if defined(__hpux) || defined(_HIUX_SOURCE)
   if (getenv("SHLIB_PATH")) {
      ldpath = new char[32+strlen(fgROOTsys)+strlen(getenv("SHLIB_PATH"))];
      sprintf(ldpath, "SHLIB_PATH=%s/lib:%s", fgROOTsys, getenv("SHLIB_PATH"));
   } else {
      ldpath = new char[32+strlen(fgROOTsys)];
      sprintf(ldpath, "SHLIB_PATH=%s/lib", fgROOTsys);
   }
#elif defined(_AIX)
   if (getenv("LIBPATH")) {
      ldpath = new char[32+strlen(fgROOTsys)+strlen(getenv("LIBPATH"))];
      sprintf(ldpath, "LIBPATH=%s/lib:%s", fgROOTsys, getenv("LIBPATH"));
   } else {
      ldpath = new char[32+strlen(fgROOTsys)];
      sprintf(ldpath, "LIBPATH=%s/lib", fgROOTsys);
   }
#elif defined(__APPLE__)
   if (getenv("DYLD_LIBRARY_PATH")) {
      ldpath = new char[32+strlen(fgROOTsys)+strlen(getenv("DYLD_LIBRARY_PATH"))];
      sprintf(ldpath, "DYLD_LIBRARY_PATH=%s/lib:%s",
              fgROOTsys, getenv("DYLD_LIBRARY_PATH"));
   } else {
      ldpath = new char[32+strlen(fgROOTsys)];
      sprintf(ldpath, "DYLD_LIBRARY_PATH=%s/lib", fgROOTsys);
   }
#else
   if (getenv("LD_LIBRARY_PATH")) {
      ldpath = new char[32+strlen(fgROOTsys)+strlen(getenv("LD_LIBRARY_PATH"))];
      sprintf(ldpath, "LD_LIBRARY_PATH=%s/lib:%s",
              fgROOTsys, getenv("LD_LIBRARY_PATH"));
   } else {
      ldpath = new char[32+strlen(fgROOTsys)];
      sprintf(ldpath, "LD_LIBRARY_PATH=%s/lib", fgROOTsys);
   }
#endif
   putenv(ldpath);
#endif
}

//_________________________________________________________________________________
int XrdProofdProtocol::Create()
{
   // Handle a request to create a new session

   int psid = -1, rc = 1;

   // Allocate next free server ID and fill in the basic stuff
   psid = GetFreeServID();
   XrdProofServProxy *xps = fPClient->fProofServs.at(psid);
   xps->SetID(psid);
   xps->SetSrvType(fSrvType);

   // Notify
   TRACEP(REQ, "Create: new psid: "<<psid<<"; client ID: "<<fCID);

   // Fork an agent process to handle this session
   int pid = -1;
   if (!(pid = fgSched->Fork("proofsrv"))) {

      char *argvv[4] = {0};

      // start server
      argvv[0] = (char *)fgPrgmSrv;
      argvv[1] = (char *)((fSrvType == kXPD_WorkerServer) ? "proofslave"
                          : "proofserv");
      argvv[2] = (char *)"xpd";
      argvv[3] = 0;

      // Set environment for proofserv
      SetProofServEnv(psid);

      // We set to the user environment
      if (SetUserEnvironment(fClientID) != 0) {
         PRINT("Create: SetUserEnvironment did not return OK - EXIT");
         exit(1);
      }

      // Close standard units
      close(2);
      close(1);
      close(0);

      // Run the program
      execv(fgPrgmSrv, argvv);

      // We should not be here!!!
      TRACEP(REQ, "Create: returned from execv: bad, bad sign !!!");
      exit(1);
   }

   // parent process
   if (pid < 0) {
      SafeDelete(xps);
      // Failure in forking
      fResponse.Send(kXR_ServerError, "could not fork agent");
      return rc;
   }

   // now we wait for the callback to be (successfully) established
   TRACEP(REQ, "Create: server launched: wait for callback ");

   // We will get back a peer to initialize a link
   XrdNetPeer peerpsrv;
   XrdLink   *linkpsrv = 0;
   int lnkopts = 0;

   // Perform regular accept
   if (!(fUNIXSock->Accept(peerpsrv, XRDNET_NODNTRIM, fgInternalWait))) {
      // Try kill
      if (kill(pid, SIGKILL) == 0)
         fResponse.Send(kXP_ServerError, "did not receive callback: process killed");
      else
         fResponse.Send(kXP_ServerError, "did not receive callback:"
                        " process could not be killed");
      xps->Reset();
      return rc;
   }

   // Allocate a new network object
   if (!(linkpsrv = XrdLink::Alloc(peerpsrv, lnkopts))) {
      kill(pid, SIGKILL);
      xps->Reset();
      fResponse.Send(kXP_ServerError, "could not allocate network object");
      return rc;

   } else {

      // Keep buffer after object goes away
      peerpsrv.InetBuff = 0;
      TRACE(REQ, "Accepted connection from " << peerpsrv.InetName);

      // Get a protocol object off the stack (if none, allocate a new one)
      XrdProtocol *xp = Match(linkpsrv);
      if (!xp) {
         kill(pid, SIGKILL);
         fResponse.Send(kXP_ServerError, "Match failed: protocol error");
         linkpsrv->Close();
         xps->Reset();
         return rc;
      }

      // Take a short-cut and process the initial request as a sticky request
      xp->Process(linkpsrv);

      // Attach this link to the appropriate poller and enable it.
      if (!XrdPoll::Attach(linkpsrv)) {
         kill(pid, SIGKILL);
         fResponse.Send(kXP_ServerError, "could not attach new internal link to poller");
         linkpsrv->Close();
         xps->Reset();
         return rc;
      }

      // Tight this protocol instance to the link
      linkpsrv->setProtocol(xp);

      // Schedule it
      fgSched->Schedule((XrdJob *)linkpsrv);
   }

   // Ignore childs when they terminate, so they do not become zombies
   SetIgnoreZombieChild();

   // Change child process priority, if required
   if (fgPriorities.size() > 0) {
      XrdOucString usr(fClientID);
      int dp = 0;
      int nmmx = -1;
      std::list<XrdProofdPriority *>::iterator i;
      for (i = fgPriorities.begin(); i != fgPriorities.end(); ++i) {
         int nm = usr.matches((*i)->fUser.c_str());
         if (nm >= nmmx) {
            nmmx = nm;
            dp = (*i)->fDeltaPriority;
         }
      }
      if (nmmx > -1) {
         // Changing child process priority for this user
         if (ChangeProcessPriority(pid, dp) != 0) {
            TRACEP(REQ, "Create: problems changing child process priority");
         } else {
            TRACEP(REQ, "Create: priority of the child process changed by " << dp << " units");
         }
      }
   }

   // Set ID
   xps->SetSrv(pid);

   // Stream ID
   unsigned short sid;
   memcpy((void *)&sid, (const void *)&(fRequest.header.streamid[0]), 2);

   // We associate this instance to the corresponding slot in the
   // session vector of attached clients
   XrdClientID *csid = xps->GetClientID(fCID);
   csid->fP = this;
   csid->fSid = sid;

   // Take parentship, if orphalin
   xps->fParent = csid;

   TRACEP(REQ, "Create: ClientID: "<<(int *)(xps->fParent)<<" (sid: "<<sid<<")");

   // Notify the client
   fResponse.Send(psid,(kXR_int32)XPROOFD_VERSBIN, fClientID, strlen(fClientID));

   // Over
   return rc;
}

//______________________________________________________________________________
void XrdProofdProtocol::SetIgnoreZombieChild()
{
   // Do want to have childs become zombies and clog up the system.
   // With SysV all we need to do is ignore the signal.
   // With BSD, however, we have to catch each signal
   // and execute the wait3() system call.
   // Code copied & pasted from rpdutils/src/daemons.cxx .

#ifdef USE_SIGCHLD
   signal(SIGCLD, SigChild);
#else
#if defined(__alpha) && !defined(linux)
   struct sigaction oldsigact, sigact;
   sigact.sa_handler = SIG_IGN;
   sigemptyset(&sigact.sa_mask);
   sigact.sa_flags = SA_NOCLDWAIT;
   sigaction(SIGCHLD, &sigact, &oldsigact);
#elif defined(__sun)
   sigignore(SIGCHLD);
#else
   signal(SIGCLD, SIG_IGN);
#endif
#endif
}

//_____________________________________________________________________________
int XrdProofdProtocol::SendMsg()
{
   // Handle a request to forward a message to another process

   static const char *crecv[4] = {"master proofserv", "top master",
                                  "client", "undefined"};
   int rc = 1;

   XrdOucMutexHelper mh(fResponse.fMutex);

   // Unmarshall the data
   int psid = ntohl(fRequest.sendrcv.sid);
   int opt = ntohl(fRequest.sendrcv.opt);
   TRACEP(REQ, "SendMsg: psid: "<<psid<<"; opt: "<< opt);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (INRANGE(psid,fPClient->fProofServs)) {
      xps = fPClient->fProofServs.at(psid);
   } else {
      TRACEP(REQ, "SendMsg: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
   }

   if (xps) {
      TRACEP(REQ, "SendMsg: xps: "<<xps<<", status: "<<xps->Status());

      // Type of connection
      bool external = !(opt & kXPD_internal);

      // Forward message as unsolicited
      char *msg = fArgp->buff;
      int   len = fRequest.header.dlen;

      if (external) {
         TRACEP(REQ, "SendMsg: EXTERNAL; psid: "<<psid);

         // Send to proofsrv our client ID
         if (fCID == -1) {
            fResponse.Send(kXR_ServerError,"external: getting clientSID");
            return rc;
         }
         TRACEP(REQ, "SendMsg: EXTERNAL; fCID: " << fCID);
         if (xps->fProofSrv.Send(kXR_attn, kXPD_msgsid, fCID, msg, len)) {
            fResponse.Send(kXR_ServerError,"external: sending message to proofsrv");
            return rc;
         }
         // Notify to user
         fResponse.Send();
         TRACEP(REQ, "SendMsg: external: message sent to proofserv ("<<len<<" bytes)");

      } else {
         TRACEP(REQ, "SendMsg: INTERNAL; psid: "<<psid);

         // Additional info about the message
         if (opt & kXPD_setidle) {
            TRACEP(REQ, "SendMsg: internal: setting proofserv in 'idle' state");
            xps->fStatus = kXPD_idle;
            // Clean start processing message, if any
            if (xps->fStartMsg) {
               delete xps->fStartMsg;
               xps->fStartMsg = 0;
            }
         } else if (opt & kXPD_querynum) {
            TRACEP(REQ, "SendMsg: internal: got message with query number");
            // Save query num message for later clients
            if (xps->fQueryNum)
               delete xps->fQueryNum;
            xps->fQueryNum = new XrdSrvBuffer(msg, len, 1);
         } else if (opt & kXPD_startprocess) {
            TRACEP(REQ, "SendMsg: internal: setting proofserv in 'running' state");
            xps->fStatus = kXPD_running;
            // Save start processing message for later clients
            if (xps->fStartMsg)
               delete xps->fStartMsg;
            xps->fStartMsg = new XrdSrvBuffer(msg, len, 1);
         } else if (opt & kXPD_logmsg) {
            // We broadcast log messages only not idle to catch the
            // result from processing
            if (xps->fStatus == kXPD_running) {
               TRACEP(REQ, "SendMsg: internal: broadcasting log message");
               opt |= kXPD_fb_prog;
            }
         }
         bool fbprog = (opt & kXPD_fb_prog);

         if (!fbprog) {
            // Get ID of the client
            int cid = ntohl(fRequest.sendrcv.cid);
            TRACEP(REQ, "SendMsg: INTERNAL; cid: "<<cid);

            // Get corresponding instance
            XrdClientID *csid = 0;
            if (INRANGE(cid,xps->fClients)) {
               csid = xps->fClients.at(cid);
            } else {
               TRACEP(REQ, "SendMsg: client ID not found (cid = "<<cid<<
                           "; size = "<<xps->fClients.size()<<")");
               fResponse.Send(kXR_InvalidRequest,"Client ID not found");
            }
            if (!csid || !(csid->fP)) {
               TRACEP(REQ, "SendMsg: INTERNAL: client not connected ");
               // Notify to proofsrv
               fResponse.Send();
               return rc;
            }

            //
            // The message is strictly for the client requiring it
            int rs = 0;
            {  XrdOucMutexHelper mhp(csid->fP->fResponse.fMutex);
               unsigned short sid;
               csid->fP->fResponse.GetSID(sid);
               TRACEP(REQ, "SendMsg: internal: this sid: "<<sid<<
                           "; client sid:"<<csid->fSid);
               csid->fP->fResponse.Set(csid->fSid);
               rs = csid->fP->fResponse.Send(kXR_attn, kXPD_msg, msg, len);
               csid->fP->fResponse.Set(sid);
            }
            if (rs) {
               fResponse.Send(kXR_ServerError,
                              "SendMsg: internal: sending message to client"
                              " or master proofserv");
               return rc;
            }
         } else {
            // Send to all connected clients
            XrdClientID *csid = 0;
            int ic = 0;
            for (ic = 0; ic < (int) xps->fClients.size(); ic++) {
               if ((csid = xps->fClients.at(ic)) && csid->fP) {
                  int rs = 0;
                  {  XrdOucMutexHelper mhp(csid->fP->fResponse.fMutex);
                     unsigned short sid;
                     csid->fP->fResponse.GetSID(sid);
                     TRACEP(REQ, "SendMsg: internal: this sid: "<<sid<<
                                 "; client sid:"<<csid->fSid);
                     csid->fP->fResponse.Set(csid->fSid);
                     rs = csid->fP->fResponse.Send(kXR_attn, kXPD_msg, msg, len);
                     csid->fP->fResponse.Set(sid);
                  }
                  if (rs) {
                     fResponse.Send(kXR_ServerError,
                                    "SendMsg: internal: sending message to client"
                                    " or master proofserv");
                     return rc;
                  }
               }
            }
         }
         TRACEP(REQ, "SendMsg: internal: message sent to "<<crecv[xps->fSrvType]<<
                     " ("<<len<<" bytes)");
         // Notify to proofsrv
         fResponse.Send();
      }

   } else {
      TRACEP(REQ, "SendMsg: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
   }

   // Over
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::Admin()
{
   // Handle generic request of administrative type

   // Should be the same as in proofx/inc/TXSocket.h
   enum EAdminMsgType { kQuerySessions = 1000,
                        kSessionTag, kSessionAlias, kGetWorkers, kQueryWorkers};

   int rc = 1;

   // Unmarshall the data
   //
   int type = ntohl(fRequest.proof.int1);
   int psid = ntohl(fRequest.proof.sid);
   TRACEP(REQ, "Admin: type: "<<type<<", psid: "<<psid);

   if (type == kQuerySessions) {

      XrdProofServProxy *xps = 0;
      int ns = 0;
      std::vector<XrdProofServProxy *>::iterator ip;
      for (ip = fPClient->fProofServs.begin(); ip != fPClient->fProofServs.end(); ++ip)
         if ((xps = *ip) && xps->IsValid() && (xps->fSrvType == kXPD_TopMaster))
            ns++;

      // Generic info about all known sessions
      int len = (kXPROOFSRVTAGMAX+kXPROOFSRVALIASMAX+30)* (ns+1);
      char *buf = new char[len];
      if (!buf) {
         TRACEP(REQ, "Admin: no resources for results");
         fResponse.Send(kXR_NoMemory,"Admin: out-of-resources for results");
         return rc;
      }
      sprintf(buf, "%d", ns);

      xps = 0;
      for (ip = fPClient->fProofServs.begin(); ip != fPClient->fProofServs.end(); ++ip) {
         if ((xps = *ip) && xps->IsValid() && (xps->fSrvType == kXPD_TopMaster)) {
            sprintf(buf,"%s | %d %s %s %d %d",
                    buf, xps->fID, xps->fTag, xps->fAlias,
                    ((xps->fStatus) ? 0 : 1), xps->GetNClients());
         }
      }
      TRACEP(REQ, "Admin: sending: "<<buf);

      // Send back to user
      fResponse.Send(buf,strlen(buf)+1);
      if (buf) delete[] buf;

   } else if (type == kSessionTag) {

      //
      // Specific info about a session
      XrdProofServProxy *xps = 0;
      if (INRANGE(psid,fPClient->fProofServs)) {
         xps = fPClient->fProofServs.at(psid);
      } else {
         TRACEP(REQ, "Admin: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"Admin: session ID not found");
      }

      // Set session tag
      char *msg = fArgp->buff;
      int   len = fRequest.header.dlen;
      if (len > kXPROOFSRVTAGMAX - 1)
         len = kXPROOFSRVTAGMAX - 1;

      // Save tag
      if (len > 0 && msg) {
         strncpy(xps->fTag, msg, len);
         xps->fTag[len] = 0;
         TRACEP(REQ, "Admin: session tag set to: "<<xps->fTag);
      }

      // Acknowledge user
      fResponse.Send();

   } else if (type == kSessionAlias) {

      //
      // Specific info about a session
      XrdProofServProxy *xps = 0;
      if (INRANGE(psid,fPClient->fProofServs)) {
         xps = fPClient->fProofServs.at(psid);
      } else {
         TRACEP(REQ, "Admin: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"Admin: session ID not found");
      }

      // Set session alias
      char *msg = fArgp->buff;
      int   len = fRequest.header.dlen;
      if (len > kXPROOFSRVALIASMAX - 1)
         len = kXPROOFSRVALIASMAX - 1;

      // Save tag
      if (len > 0 && msg) {
         strncpy(xps->fAlias, msg, len);
         xps->fAlias[len] = 0;
         TRACEP(REQ, "Admin: session alias set to: "<<xps->fAlias);
      }

      // Acknowledge user
      fResponse.Send();

   } else if (type == kGetWorkers) {

      // Find server session
      XrdProofServProxy *xps = 0;
      if (INRANGE(psid,fPClient->fProofServs)) {
         xps = fPClient->fProofServs.at(psid);
      } else {
         TRACEP(REQ, "Admin: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"session ID not found");
      }

      // We should query the chosen resource provider
      XrdOucString wrks;
      if (GetWorkers(wrks, xps) !=0 ) {
         // Something wrong
         fResponse.Send(kXR_InvalidRequest,"Admin: GetWorkers failed");
      } else {
         // Send buffer
         char *buf = (char *) wrks.c_str();
         int len = wrks.length() + 1;
         TRACEP(REQ, "Admin: GetWorkers: sending: "<<buf);

         // Send back to user
         fResponse.Send(buf, len);
      }
   } else if (type == kQueryWorkers) {

      // Send back a list of potentially available workers
      XrdOucString sbuf(1024);

      // Selection type
      const char *osel[] = { "all", "round-robin", "random"};
      sbuf += "Selection: ";
      sbuf += osel[fgWorkerSel+1];
      if (fgWorkerSel > -1) {
         sbuf += ", max workers: ";
         sbuf += fgWorkerMax; sbuf += " &";
      }

      // The full list
      int iw = 0;
      for (iw = 0; iw < (int)fgWorkers.size() ; iw++) {
         sbuf += fgWorkers[iw]->fType;
         sbuf += ": "; sbuf += fgWorkers[iw]->fHost;
         if (fgWorkers[iw]->fPort > -1) {
            sbuf += ":"; sbuf += fgWorkers[iw]->fPort;
         } else
            sbuf += "     ";
         sbuf += "  sessions: "; sbuf += fgWorkers[iw]->fActive;
         sbuf += " &";
      }

      // Send buffer
      char *buf = (char *) sbuf.c_str();
      int len = sbuf.length() + 1;
      TRACEP(REQ, "Admin: QueryWorkers: sending: "<<buf);

      // Send back to user
      fResponse.Send(buf, len);

   } else {
      TRACEP(REQ, "Admin: unknown request type");
      fResponse.Send(kXR_InvalidRequest,"Admin: unknown request type");
      return rc;
   }

   // Over
   return rc;
}

//___________________________________________________________________________
int XrdProofdProtocol::Interrupt()
{
   // Handle an interrupt request

   unsigned int rc = 1;

   // Unmarshall the data
   int psid = ntohl(fRequest.interrupt.sid);
   int type = ntohl(fRequest.interrupt.type);
   TRACEP(REQ, "Interrupt: psid: "<<psid<<" type:"<<type);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (INRANGE(psid,fPClient->fProofServs)) {
      xps = fPClient->fProofServs.at(psid);
   } else {
      TRACEP(REQ, "Interrupt: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"nterrupt: session ID not found");
   }

   if (xps) {
      TRACEP(REQ, "Interrupt: xps: "<<xps<<", status: "<<xps->Status());

      // Check ID matching
      if (!xps->Match(psid)) {
         fResponse.Send(kXP_InvalidRequest,"Interrupt: IDs do not match - do nothing");
         return rc;
      }

      TRACEP(REQ, "Interrupt: xps: "<<xps<<", internal link "<<xps->fLink);

      // Propagate the type as unsolicited
      if (xps->fProofSrv.Send(kXR_attn, kXPD_interrupt, type) != 0) {
         fResponse.Send(kXP_ServerError,
                        "Interrupt: could not propagate interrupt code to proofsrv");
         return rc;
      }

      TRACEP(REQ, "Interrupt: xps: "<<xps<<", proofsrv ID: "<<xps->SrvID());

      // Notify to user
      fResponse.Send();
      TRACEP(REQ, "Interrupt: interrupt propagated to proofsrv");

   } else {
      TRACEP(REQ, "Interrupt: session ID not found");
   }

   // Over
   return rc;
}

//___________________________________________________________________________
int XrdProofdProtocol::Ping()
{
   // Handle a ping request

   int rc = 1;

   // Unmarshall the data
   int psid = ntohl(fRequest.sendrcv.sid);
   int opt = ntohl(fRequest.sendrcv.opt);

   TRACEP(REQ, "Ping: psid: "<<psid<<"; opt= "<<opt);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (INRANGE(psid,fPClient->fProofServs)) {
      xps = fPClient->fProofServs.at(psid);
   } else {
      TRACEP(REQ, "Ping: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
   }

   kXR_int32 pingres = 0;
   if (xps) {
      TRACEP(REQ, "Ping: xps: "<<xps<<", status: "<<xps->Status());

      // Type of connection
      bool external = !(opt & kXPD_internal);

      if (external) {
         TRACEP(REQ, "Ping: EXTERNAL; psid: "<<psid);

         // Create semaphore
         xps->fPingSem = new XrdOucSemWait(0);

         // Propagate the ping request
         if (xps->fProofSrv.Send(kXR_attn, kXPD_ping, 0, 0) != 0) {
            TRACEP(REQ, "Ping: could not propagate ping to proofsrv");
            SafeDelete(xps->fPingSem);
            fResponse.Send(kXR_ok, pingres);
            return rc;
         }

         if (xps->fPingSem->Wait(fgInternalWait) != 0) {
            SafeDelete(xps->fPingSem);
            TRACEP(REQ, "Ping: did not receive reply after ping");
            fResponse.Send(kXR_ok, pingres);
            return rc;
         }
         // Cleanup
         SafeDelete(xps->fPingSem);

         // ok
         pingres = 1;

         // Notify the client
         TRACEP(REQ, "Ping: external: ping notified to client");
         fResponse.Send(kXR_ok, pingres);
         return rc;

      } else {
         TRACEP(REQ, "Ping: INTERNAL; psid: "<<psid);

         // If a semaphore is waiting, post it
         if (xps->fPingSem)
            xps->fPingSem->Post();

         // Just notify to user
         pingres = 1;
         fResponse.Send(kXR_ok, pingres);
         return rc;
      }
   }

   // Failure
   TRACEP(REQ, "Ping: session ID not found");
   fResponse.Send(kXR_ok, pingres);
   return rc;
}
//___________________________________________________________________________
int XrdProofdProtocol::SetUserEnvironment(const char *usr, const char *dir)
{
   // Set user environment: set effective user and group ID of the process
   // to the ones specified by 'usr', change working dir to subdir 'dir'
   // of 'usr' $HOME.
   // Return 0 on success, -1 if enything goes wrong.

   // Get user info
   struct passwd *pw = getpwnam(usr);
   if (!pw) {
      TRACEP(REQ,"SetUserEnvironment: user '"<<usr<<"' does not exist locally");
      return -1;
   }

   // Change to user's home dir
   XrdOucString home = pw->pw_dir;
   if (dir) {
      home += '/';
      home += dir;
      struct stat st;
      if (stat(home.c_str(), &st) || !S_ISDIR(st.st_mode)) {
         // Specified path does not exist or is not a dir
         TRACEP(REQ,"SetUserEnvironment: subpath "<<dir<<
                    " does not exist or is not a dir");
         home = pw->pw_dir;
      }
   }
   if (chdir(home.c_str()) == -1) {
      TRACEP(REQ,"SetUserEnvironment: can't change directory to "<<home);
      return -1;
   }

   // set HOME env
   char *h = new char[8+home.length()];
   sprintf(h, "HOME=%s", home.c_str());
   putenv(h);

   // Set access control list from /etc/initgroup
   // (super-user privileges required)
   {  XrdSysPrivGuard(0);
      initgroups(usr, pw->pw_gid);
   }

   // acquire permanently target user privileges
   if (XrdSysPriv::ChangePerm(pw->pw_uid) != 0) {
      TRACEP(REQ,"SetUserEnvironment: can't acquire "<< usr <<" identity");
      return -1;
   }

   // We are done
   return 0;
}

//--------------------------------------------------------------------------
//
// XrdProofClient
//
//--------------------------------------------------------------------------

//__________________________________________________________________________
int XrdProofClient::GetClientID(XrdProofdProtocol *p)
{
   // Get next free client ID. If none is found, increase the vector size
   // and get the first new one

   XrdOucMutexHelper mh(fMutex);

   int ic = 0;
   // Search for free places in the existing vector
   for (ic = 0; ic < (int)fClients.size() ; ic++) {
      if (!fClients[ic]) {
         fClients[ic] = p;
         return ic;
      }
   }

   // We need to resize (double it)
   if (ic >= (int)fClients.capacity())
      fClients.reserve(2*fClients.capacity());

   // Fill in new element
   fClients.push_back(p);

   TRACE(REQ, "XrdProofClient::GetClientID: size = "<<fClients.size());

   // We are done
   return ic;
}

//--------------------------------------------------------------------------
//
// XrdProofWorker
//
//--------------------------------------------------------------------------

//__________________________________________________________________________
XrdProofWorker::XrdProofWorker(const char *str)
               : fActive (0), fSuspended(0),
                 fExport(256), fType('W'), fPort(-1), fPerfIdx(100)
{
   // Constructor from a config file-like string 

   // Make sure we got something to parse
   if (!str || strlen(str) <= 0)
      return;

   // The actual work is done by Reset()
   Reset(str);
}

//__________________________________________________________________________
void XrdProofWorker::Reset(const char *str)
{
   // Set content from a config file-like string 

   // Reinit vars
   fActive = 0;
   fSuspended = 0;
   fExport = "";
   fType = 'W';
   fHost = "";
   fPort = -1;
   fPerfIdx = 100;
   fImage = "";
   fWorkDir = "";
   fMsd = "";
   fId = "";

   // Make sure we got something to parse
   if (!str || strlen(str) <= 0)
      return;

   // Tokenize the string
   XrdOucString s(str);

   // First token is the type
   XrdOucString tok;
   XrdOucString typestr = "mastersubmasterworkerslave";
   int from = s.tokenize(tok, 0, ' ');
   if (from == STR_NPOS || typestr.find(tok) == STR_NPOS)
      return;
   if (tok == "submaster")
      fType = 'S';
   else if (tok == "master")
      fType = 'M';

   // Next token is the user@host string
   if ((from = s.tokenize(tok, from, ' ')) == STR_NPOS)
      return;
   fHost = tok;

   // and then the remaining options
   while ((from = s.tokenize(tok, from, ' ')) != STR_NPOS) {
      if (tok.beginswith("workdir=")) {
         // Working dir
         tok.replace("workdir=","");
         fWorkDir = tok;
      } else if (tok.beginswith("image=")) {
         // Image
         tok.replace("image=","");
         fImage = tok;
      } else if (tok.beginswith("msd=")) {
         // Mass storage domain
         tok.replace("msd=","");
         fMsd = tok;
      } else if (tok.beginswith("port=")) {
         // Port
         tok.replace("port=","");
         fPort = strtol(tok.c_str(), (char **)0, 10);
      } else if (tok.beginswith("perf=")) {
         // Performance index
         tok.replace("perf=","");
         fPerfIdx = strtol(tok.c_str(), (char **)0, 10);
      } else {
         // Unknown
         TRACE(REQ, "XrdProofWorker::Set: unknown option "<<tok);
      }
   }
}

//__________________________________________________________________________
const char *XrdProofWorker::Export()
{
   // Export current content in a form understood by parsing algorithms
   // inside the PROOF session, i.e.
   // <type>|<host@user>|<port>|-|-|<perfidx>|<img>|<workdir>|<msd>

   fExport = fType;

   // Add user@host
   fExport += '|' ; fExport += fHost;

   // Add port
   if (fPort > 0) {
      fExport += '|' ; fExport += fPort;
   } else
      fExport += "|-";

   // No ordinal and ID at this level
   fExport += "|-|-";

   // Add performance index
   fExport += '|' ; fExport += fPerfIdx;

   // Add image
   if (fImage.length() > 0) {
      fExport += '|' ; fExport += fImage;
   } else
      fExport += "|-";

   // Add workdir
   if (fWorkDir.length() > 0) {
      fExport += '|' ; fExport += fWorkDir;
   } else
      fExport += "|-";

   // Add mass storage domain
   if (fMsd.length() > 0) {
      fExport += '|' ; fExport += fMsd;
   } else
      fExport += "|-";

   // We are done
   TRACE(REQ, "XrdProofWorker::Export: sending: "<<fExport);
   return fExport.c_str();
}
