// @(#)root/proofd:$Name:  $:$Id: XrdProofdProtocol.cxx,v 1.22 2006/09/28 23:23:45 rdm Exp $
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
#if (defined(SUNCC) || defined(__sun))
#include <sys/isa_defs.h>
#if defined(_ILP32) && (_FILE_OFFSET_BITS != 32)
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 32
#undef  _LARGEFILE_SOURCE
#endif
#endif

// System info on Solaris
#if (defined(SUNCC) || defined(__sun)) && !defined(__KCC)
#   define XPD__SUNCC
#   include <sys/systeminfo.h>
#   include <sys/filio.h>
#   include <sys/sockio.h>
#   define HASNOT_INETATON
#   define INADDR_NONE (UInt_t)-1
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

// Poll
#include <unistd.h>
#include <sys/poll.h>

#include "XrdVersion.hh"
#include "XrdClient/XrdClientMessage.hh"
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

#include "XrdProofConn.h"
#include "XrdProofdProtocol.h"

#include "config.h"

// Tracing utils
#include "XrdProofdTrace.h"
XrdOucTrace          *XrdProofdTrace = 0;
static const char    *gTraceID = " ";

// Static variables
static XrdOucReqID   *XrdProofdReqID = 0;
XrdOucRecMutex        gSysPrivMutex;

//
// Static area: general protocol managing section
int                   XrdProofdProtocol::fgCount    = 0;
XrdOucRecMutex        XrdProofdProtocol::fgXPDMutex;
XrdObjectQ<XrdProofdProtocol>
                      XrdProofdProtocol::fgProtStack("ProtStack",
                                                     "xproofd protocol anchor");
XrdBuffManager       *XrdProofdProtocol::fgBPool    = 0;
int                   XrdProofdProtocol::fgMaxBuffsz= 0;
XrdSecService        *XrdProofdProtocol::fgCIA      = 0;
XrdScheduler         *XrdProofdProtocol::fgSched    = 0;
XrdOucError           XrdProofdProtocol::fgEDest(0, "Proofd");

//
// Static area: protocol configuration section
bool                  XrdProofdProtocol::fgConfigDone = 0;
kXR_int32             XrdProofdProtocol::fgSrvType  = kXPD_AnyServer;
char                 *XrdProofdProtocol::fgROOTsys  = 0;
char                 *XrdProofdProtocol::fgTMPdir   = 0;
char                 *XrdProofdProtocol::fgImage    = 0;
char                 *XrdProofdProtocol::fgWorkDir  = 0;
int                   XrdProofdProtocol::fgPort     = 0;
char                 *XrdProofdProtocol::fgSecLib   = 0;
//
XrdOucString          XrdProofdProtocol::fgEffectiveUser;
XrdOucString          XrdProofdProtocol::fgLocalHost;
char                 *XrdProofdProtocol::fgPoolURL = 0;
char                 *XrdProofdProtocol::fgNamespace = strdup("/proofpool");
//
char                 *XrdProofdProtocol::fgPrgmSrv  = 0;
int                   XrdProofdProtocol::fgSrvProtVers = -1;
XrdOucSemWait         XrdProofdProtocol::fgForkSem;   // To serialize fork requests
//
EResourceType         XrdProofdProtocol::fgResourceType = kRTStatic;
int                   XrdProofdProtocol::fgMaxSessions = -1;
int                   XrdProofdProtocol::fgWorkerMax = -1; // max number or workers per user
EStaticSelOpt         XrdProofdProtocol::fgWorkerSel = kSSORoundRobin; // selection option
//
std::vector<XrdProofWorker *> XrdProofdProtocol::fgWorkers;  // list of possible workers
std::list<XrdOucString *> XrdProofdProtocol::fgMastersAllowed;
std::list<XrdProofdPriority *> XrdProofdProtocol::fgPriorities;
char                 *XrdProofdProtocol::fgSuperUsers = 0; // ':' separated list of privileged users
//
char                 *XrdProofdProtocol::fgPROOFcfg = 0; // PROOF static configuration
bool                  XrdProofdProtocol::fgWorkerUsrCfg = 0; // user cfg files enabled / disabled
//
int                   XrdProofdProtocol::fgReadWait = 0;
int                   XrdProofdProtocol::fgInternalWait = 5; // seconds
// Shutdown options
int                   XrdProofdProtocol::fgShutdownOpt = 1;
int                   XrdProofdProtocol::fgShutdownDelay = 0; // minimum

//
// Static area: client section
std::list<XrdProofClient *> XrdProofdProtocol::fgProofClients;  // keeps track of all users
std::list<int *>      XrdProofdProtocol::fgTerminatedProcess; // List of pids of processes terminating

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

#undef  TRACELINK
#define TRACELINK fLink
#undef  RESPONSE
#define RESPONSE fResponse

typedef struct {
   kXR_int32 ptyp;  // must be always 0 !
   kXR_int32 rlen;
   kXR_int32 pval;
   kXR_int32 styp;
} hs_response_t;

// Should be the same as in proofx/inc/TXSocket.h
enum EAdminMsgType {
   kQuerySessions = 1000,
   kSessionTag,
   kSessionAlias,
   kGetWorkers,
   kQueryWorkers,
   kCleanupSessions
};

// Security handle
typedef XrdSecService *(*XrdSecServLoader_t)(XrdOucLogger *, const char *cfn);

#ifndef LONG_MAX
#define LONG_MAX 2147483647
#endif
//__________________________________________________________________________
static long int GetLong(char *str)
{
   // Extract first integer from string at 'str', if any

   // Reposition on first digit
   char *p = str;
   while ((*p < 48 || *p > 57) && (*p) != '\0')
      p++;
   if (*p == '\0')
      return LONG_MAX;

   // Find the last digit
   int j = 0;
   while (*(p+j) >= 48 && *(p+j) <= 57)
      j++;
   *(p+j) = '\0';

   // Convert now
   return strtol(p, 0, 10);
}

//__________________________________________________________________________
static int GetUserInfo(const char *usr, XrdProofUI &ui)
{
   // Get information about user 'usr' in a thread safe way.
   // Retur 0 on success, -errno on error

   // Make sure input is defined
   if (!usr || strlen(usr) <= 0)
      return -EINVAL;

   // Call getpwnam_r ...
   struct passwd pw;
   struct passwd *ppw = 0;
   char buf[2048];
#if defined(__sun)
   ppw = getpwnam_r(usr, &pw, buf, sizeof(buf));
#else
   getpwnam_r(usr, &pw, buf, sizeof(buf), &ppw);
#endif
   if (ppw) {
      // Fill output
      ui.fUid = (int) pw.pw_uid;
      ui.fGid = (int) pw.pw_gid;
      ui.fHomeDir = pw.pw_dir;
      ui.fUser = usr;
      // Done
      return 0;
   }

   // Failure
   return ((int) -errno);
}

//__________________________________________________________________________
static int GetUserInfo(int uid, XrdProofUI &ui)
{
   // Get information about user with 'uid' in a thread safe way.
   // Retur 0 on success, -errno on error

   // Make sure input make sense
   if (uid <= 0)
      return -EINVAL;

   // Call getpwuid_r ...
   struct passwd pw;
   struct passwd *ppw = 0;
   char buf[2048];
#if defined(__sun)
   ppw = getpwuid_r((uid_t)uid, &pw, buf, sizeof(buf));
#else
   getpwuid_r((uid_t)uid, &pw, buf, sizeof(buf), &ppw);
#endif
   if (ppw) {
      // Fill output
      ui.fUid = uid;
      ui.fGid = (int) pw.pw_gid;
      ui.fHomeDir = pw.pw_dir;
      ui.fUser = pw.pw_name;
      // Done
      return 0;
   }

   // Failure
   return ((int) -errno);
}

//__________________________________________________________________________
int XrdProofdProtocol::Broadcast(int type, const char *msg)
{
   // Broadcast request to known potential sub-nodes.
   // Return 0 on success, -1 on error
   int rc = 0;

   // We try only once
   int maxtry_save = -1;
   int timewait_save = -1;
   XrdProofConn::GetRetryParam(maxtry_save, timewait_save);
   XrdProofConn::SetRetryParam(1, 1);

   // Loop over worker nodes
   int iw = 0;
   XrdProofWorker *w = 0;
   XrdClientMessage *xrsp = 0;
   while (iw < (int)fgWorkers.size()) {
      if ((w = fgWorkers[iw]) && w->fType != 'M') {
         // Do not send it to ourselves
         bool us = (((w->fHost.find("localhost") != STR_NPOS ||
                     fgLocalHost.find(w->fHost.c_str()) != STR_NPOS)) &&
                    (w->fPort == -1 || w->fPort == fgPort)) ? 1 : 0;
         if (!us) {
            // Create 'url'
            XrdOucString u = fgEffectiveUser;
            u += '@';
            u += w->fHost;
            if (w->fPort != -1) {
               u += ':';
               u += w->fPort;
            }
            // Type of server
            int srvtype = (w->fType != 'W') ? (kXR_int32) kXPD_MasterServer
                                            : (kXR_int32) kXPD_WorkerServer;
            TRACEP(REQ,"Broadcast: sending request to "<<u);
            // Send request
            if (!(xrsp = SendCoordinator(u.c_str(), type, msg, srvtype))) {
               TRACEP(REQ,"Broadcast: problems sending request to "<<u);
            }
            // Cleanup answer
            SafeDelete(xrsp);
         }
      }
      // Next worker
      iw++;
   }

   // Restore original retry parameters
   XrdProofConn::SetRetryParam(maxtry_save, timewait_save);

   // Done
   return rc;
}

//__________________________________________________________________________
XrdClientMessage *XrdProofdProtocol::SendCoordinator(const char *url,
                                                     int type,
                                                     const char *msg,
                                                     int srvtype)
{
   // Broadcast request to known potential sub-nodes.
   // Return 0 on success, -1 on error
   XrdClientMessage *xrsp = 0;

   if (!url || strlen(url) <= 0)
      return xrsp;

   // Open the connection
   XrdOucString buf = "session-cleanup-from-";
   buf += fgLocalHost;
   buf += "|ord:000";
   char m = (srvtype == kXPD_MasterServer) ? 'm' : 's';
   XrdProofConn *conn = new XrdProofConn(url, m, -1, -1, 0, buf.c_str());

   bool ok = 1;
   if (conn && conn->IsValid()) {
      // Prepare request
      XPClientRequest reqhdr;
      const void *buf = 0;
      void **vout = 0;
      memset(&reqhdr, 0, sizeof(reqhdr));
      conn->SetSID(reqhdr.header.streamid);
      reqhdr.header.requestid = kXP_admin;
      reqhdr.proof.int1 = type;
      switch (type) {
         case kCleanupSessions:
            reqhdr.proof.int2 = (kXR_int32) srvtype;
            reqhdr.proof.sid = -1;
            reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
            buf = (msg) ? (const void *)msg : buf;
            break;
         default:
            ok = 0;
            TRACEP(REQ,"SendCoordinator: invalid request type "<<type);
            break;
      }

      // Send over
      if (ok)
         xrsp = conn->SendReq(&reqhdr, buf, vout, "XrdProofdProtocol::SendCoordinator");

      // Close physically the connection
      conn->Close("S");

      // Delete it
      SafeDelete(conn);

   } else {
      TRACEP(REQ,"SendCoordinator: could not open connection to "<<url);
      XrdOucString cmsg = "failure attempting connection to ";
      cmsg += url;
      fResponse.Send(kXR_attn, kXPD_srvmsg, (char *) cmsg.c_str(), cmsg.length());
   }

   // Done
   return xrsp;
}

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

   // Ignore childs when they terminate, so they do not become zombies
   SetIgnoreZombieChild();

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

      // Set Open socket
      char *ev = new char[25];
      sprintf(ev, "ROOTOPENSOCK=%d", fp[1]);
      putenv(ev);

      // Prepare for execution: we need to acquire the identity of
      // a normal user
      if (!getuid()) {
         XrdProofUI ui;
         if (GetUserInfo(geteuid(), ui) != 0) {
            PRINT("SetSrvProtVers: could not get info for user-id: "<<geteuid());
            exit(1);
         }

         // acquire permanently target user privileges
         if (XrdSysPriv::ChangePerm((uid_t)ui.fUid, (gid_t)ui.fGid) != 0) {
            PRINT("SetSrvProtVers: can't acquire "<< ui.fUser <<" identity");
            exit(1);
         }

      }

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
static int AssertDir(const char *path, XrdProofUI ui)
{
   // Make sure that 'path' exists and is owned by the entity 
   // described by 'ui'
   // Return 0 in case of success, -1 in case of error

   if (!path || strlen(path) <= 0)
      return -1;

   struct stat st;
   if (stat(path,&st) != 0) {
      if (errno == ENOENT) {
         if (mkdir(path, 0755) != 0) {
            PRINT("AssertDir: unable to create dir: "<<path<<" (errno: "<<errno<<")");
            return -1;
         }
      } else {
         // Failure: stop
         PRINT("AssertDir: unable to stat dir: "<<path<<" (errno: "<<errno<<")");
         return -1;
      }
   } else {
      // Check ownership
      if ((int) st.st_uid != ui.fUid || (int) st.st_gid != ui.fGid) {
         PRINT("MkDir: dir "<<path<<
            " exists but is owned by another entity: "<<
            "target uid: "<<ui.fUid<<", gid: "<<ui.fGid<<
            ": dir uid: "<<st.st_uid<<", gid: "<<st.st_gid<<")");
         return -1;
      }
   }

   // We are done
   return 0;
}

//_____________________________________________________________________________
static int SymLink(const char *path, const char *link)
{
   // Create a symlink 'link' to 'path'
   // Return 0 in case of success, -1 in case of error

   if (!path || strlen(path) <= 0 || !link || strlen(link) <= 0)
      return -1;

   // Remove existing link, if any
   if (unlink(link) != 0 && errno != ENOENT) { 
      PRINT("SymLink: problems unlinking existing symlink "<< link<<
            " (errno: "<<errno<<")");
      return -1;
   }
   if (symlink(path, link) != 0) {
      PRINT("SymLink: problems creating symlink " << link<< 
                " (errno: "<<errno<<")");
      return -1;
   }

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
   fLink = 0;
   fArgp = 0;
   fClientID = 0;
   fPClient = 0;
   fClient = 0;
   fAuthProt = 0;
   fBuff = 0;

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
      XrdProofUI ui;
      int rc = 0;
      if (strlen(pu) > 0) {
         rc = GetUserInfo(pu, ui);
      } else {
         rc = GetUserInfo(getuid(), ui);
      }
      if (rc == 0) {
         int ldir = ui.fHomeDir.length();
         int lpd = strlen(pd);
         po = (char *) malloc(lpd + ldir + 2);
         if (po) {
            memcpy(po, ui.fHomeDir.c_str(), ldir);
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
   xp->fResponse.Set(lp);
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
int XrdProofdProtocol::Configure(char *parms, XrdProtocol_Config *pi)
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
   fgEDest.logger(pi->eDest->logger());
   XrdProofdTrace = new XrdOucTrace(&fgEDest);
   fgSched        = pi->Sched;
   fgBPool        = pi->BPool;
   fgReadWait     = pi->readWait;
   fgPort         = pi->Port;

   // Debug flag
   if (pi->DebugON)
      XrdProofdTrace->What = TRACE_ALL;

   // Effective user
   XrdProofUI ui;
   if (GetUserInfo(geteuid(), ui) == 0) {
      fgEffectiveUser += ui.fUser;
   } else {
      mp = "Configure: could not resolve effective user (getpwuid, errno: ";
      mp += errno;
      mp += ")";
      fgEDest.Say(0, mp.c_str());
      return 0;
   }

   // Local FQDN
   char *host = XrdNetDNS::getHostName();
   fgLocalHost = host ? host : "";
   SafeFree(host);
   // Default pool entry point is this host
   int pulen = strlen("root://") + fgLocalHost.length();
   fgPoolURL = (char *) malloc(pulen + 1);
   if (!fgPoolURL)
      return 0;
   sprintf(fgPoolURL,"root://%s", fgLocalHost.c_str());
   fgPoolURL[pulen] = 0;

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
   char *pto = pe ? (char *)strstr(pe+1, "intwait:") : 0;
   if (pto) {
      pe = (char *)strstr(pto, " ");
      if (pe) *pe = 0;
      fgInternalWait = strtol(pto+8, 0, 10);
      fgEDest.Say(0, "Configure: setting internal timeout to (secs): ", pto+8);
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

   // Work directory, if specified
   if (fgWorkDir) {

      // Make sure it exists
      struct stat st;
      if (stat(fgWorkDir,&st) != 0) {
         if (errno == ENOENT) {
            // Create it
            if (mkdir(fgWorkDir, 0755) != 0) {
               fgEDest.Say(0, "Configure: unable to create work dir: ", fgWorkDir);
               return 0;
            }
         } else {
            // Failure: stop
            fgEDest.Say(0, "Configure: unable to stat work dir: ", fgWorkDir);
            return 0;
         }
      }
      fgEDest.Say(0, "Configure: PROOF work directories under: ", fgWorkDir);
   }

   if (fgSrvType != kXPD_WorkerServer || fgSrvType == kXPD_AnyServer) {
      // Pool and namespace
      fgEDest.Say(0, "Configure: PROOF pool: ", fgPoolURL);
      fgEDest.Say(0, "Configure: PROOF pool namespace: ", fgNamespace);

      if (fgResourceType == kRTStatic) {
         // Initialize the list of workers if a static config has been required
         // Default file path, if none specified
         if (!fgPROOFcfg) {
            const char *cfg = "/etc/proof/proof.conf";
            fgPROOFcfg = new char[strlen(fgROOTsys)+strlen(cfg)+1];
            sprintf(fgPROOFcfg, "%s%s", fgROOTsys, cfg);
            // Check if the file exists and is readable
            if (access(fgPROOFcfg, R_OK)) {
               fgEDest.Say(0, "Configure: PROOF config file cannot be read: ", fgPROOFcfg);
               SafeFree(fgPROOFcfg);
               // Enable user config files
               fgWorkerUsrCfg = 1;
            }
         }
         fgEDest.Say(0, "Configure: PROOF config file: ", fgPROOFcfg);
         // Load file content in memory
         if (fgPROOFcfg && ReadPROOFcfg() != 0) {
            fgEDest.Say(0, "Configure: unable to find valid information"
                           "in PROOF config file ", fgPROOFcfg);
            SafeFree(fgPROOFcfg);
            return 0;
         }
         const char *st[] = { "disabled", "enabled" };
         fgEDest.Say(0, "Configure: user config files are ", st[fgWorkerUsrCfg]);
     }
   }

   // Shutdown options
   mp = "Configure: client sessions shutdown after disconnection";
   if (fgShutdownOpt > 0) {
      if (fgShutdownOpt == 1)
         mp = "Configure: client sessions kept idle for ";
      else if (fgShutdownOpt == 2)
         mp = "Configure: client sessions kept for ";
      mp += fgShutdownDelay;
      mp += " secs after disconnection";
   }
   fgEDest.Say(0, mp.c_str());

   // Superusers: add default
   if (fgSuperUsers) {
      int l = strlen(fgSuperUsers);
      char *su = (char *) malloc(l + fgEffectiveUser.length() + 2);
      if (su) {
         sprintf(su, "%s,%s", fgEffectiveUser.c_str(), fgSuperUsers);
         free(fgSuperUsers);
         fgSuperUsers = su;
      } else {
         // Failure: stop
         fgEDest.Say(0, "Configure: no memory for superuser list - stop");
         return 0;
      }
   } else {
      fgSuperUsers = strdup(fgEffectiveUser.c_str());
   }
   mp = "Configure: list of superusers: ";
   mp += fgSuperUsers;
   fgEDest.Say(0, mp.c_str());

   // Set base environment common to all
   SetProofServEnv();

   // Test forking and get PROOF server protocol version
   if (SetSrvProtVers() < 0) {
      fgEDest.Say(0, "Configure: forking test failed");
      return 0;
   }
   mp = "Configure: PROOF server protocol number: ";
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
       nmPoolUrl = -1, nmNamespace = -1, nmSuperUsers = -1;

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
            } else if (!strcmp("shutdown",var)) {
               // Shutdown option
               int dp = strtol(val,0,10);
               if (dp >= 0 && dp <= 2)
                  fgShutdownOpt = dp;
               // Shutdown delay
               if ((val = Config.GetToken())) {
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
                        fgShutdownDelay = de * f;
                  }
               }
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
               } else if (!strcmp("superusers",var)) {
                  // Local pool entry point
                  XPDSETSTRING(nm, nmSuperUsers, fgSuperUsers, tval);
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
   case kXP_urgent:
      return Urgent();
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

   // If we have a buffer, release it
   if (fArgp) {
      fgBPool->Release(fArgp);
      fArgp = 0;
   }

   // Flag for internal connections: those deserve a different treatment
   bool proofsrv = (fSrvType == kXPD_Internal) ? 1 : 0;

   // Locate the client instance
   XrdProofClient *pmgr = 0;

   // This part may be not thread safe
   {  XrdOucMutexHelper mtxh(&fgXPDMutex);
      if (fgProofClients.size() > 0) {
         std::list<XrdProofClient *>::iterator i;
         for (i = fgProofClients.begin(); i != fgProofClients.end(); ++i) {
            if ((pmgr = *i) && pmgr->Match(fClientID))
               break;
            pmgr = 0;
         }
      }
   }

   if (pmgr) {

      if (!proofsrv) {

         // Reset the corresponding client slot in the list of this client
         // Count the remaining top clients
         int nc = 0;
         int ic = 0;
         for (ic = 0; ic < (int) pmgr->fClients.size(); ic++) {
            if (this == pmgr->fClients.at(ic))
               pmgr->fClients[ic] = 0;
            else if (pmgr->fClients.at(ic) && pmgr->fClients.at(ic)->fTopClient)
               nc++;
         }

         // If top master ...
         if (fSrvType == kXPD_TopMaster) {
            // Loop over servers sessions associated to this client and update
            // their attached client vectors
            if (pmgr->fProofServs.size() > 0) {
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

            // If no more clients schedule a shutdown at the PROOF session
            // by the sending the appropriate information
            if (nc <= 0 && pmgr->fProofServs.size() > 0) {
               XrdProofServProxy *psrv = 0;
               int is = 0;
               for (is = 0; is < (int) pmgr->fProofServs.size(); is++) {
                  if ((psrv = pmgr->fProofServs.at(is)) && psrv->SrvType() == kXPD_TopMaster) {
                     if (SetShutdownTimer(psrv) != 0) {
                        // Just notify locally: link is closed!
                        XrdOucString msg("Recycle: could not send shutdown info to proofsrv");
                        TRACEP(REQ, msg.c_str());
                     }
                     // Set in shutdown state
                     psrv->SetStatus(kXPD_shutdown);
                  }
               }
            }

         } else {

            // We cannot continue if the top master went away: we cleanup the session
            if (pmgr->fProofServs.size() > 0) {
               XrdProofServProxy *psrv = 0;
               int is = 0;
               for (is = 0; is < (int) pmgr->fProofServs.size(); is++) {
                  if ((psrv = pmgr->fProofServs.at(is))) {

                     TRACEP(REQ, "Recycle: found: " << psrv << " (v:" << psrv->IsValid() <<
                                 ",t:"<<psrv->fSrvType << ",nc:"<<psrv->fClients.size()<<")");

                     XrdOucMutexHelper xpmh(psrv->Mutex());

                     // Send a terminate signal to the proofserv
                     if (TerminateProofServ(psrv) != 0)
                        // Try hard kill
                        KillProofServ(psrv, 1);

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
         if (pmgr->fProofServs.size() > 0) {
            XrdProofServProxy *psrv = 0;
            int is = 0;
            for (is = 0; is < (int) pmgr->fProofServs.size(); is++) {
               if ((psrv = pmgr->fProofServs.at(is)) && (psrv->fLink == fLink)) {

               TRACEP(REQ, "Recycle: found: " << psrv << " (v:" << psrv->IsValid() <<
                           ",t:"<<psrv->fSrvType << ",nc:"<<psrv->fClients.size()<<")");

                  XrdOucMutexHelper xpmh(psrv->Mutex());

                  // Tell other attached clients, if any, that this session is gone
                  if (psrv->fClients.size() > 0) {
                     char msg[512] = {0};
                     snprintf(msg, 512, "Recycle: session: %s terminated by peer",
                                         psrv->Tag());
                     int len = strlen(msg);
                     int ic = 0;
                     XrdProofdProtocol *p = 0;
                     for (ic = 0; ic < (int) psrv->fClients.size(); ic++) {
                        // Send message
                        if ((p = psrv->fClients.at(ic)->fP)) {
                           unsigned short sid;
                           p->fResponse.GetSID(sid);
                           p->fResponse.Set(psrv->fClients.at(ic)->fSid);
                           p->fResponse.Send(kXR_attn, kXPD_errmsg, msg, len);
                           p->fResponse.Set(sid);
                        }
                     }
                  }

                  // Send a terminate signal to the proofserv
                  KillProofServ(psrv);

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

   // If not defined, set max sessions to worker list size
   if (fgMaxSessions < 0)
      fgMaxSessions = fgWorkers.size() - 1;

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

   // Make sure that something has been found
   if (xps->GetNWorkers() <= 0) {
      fgEDest.Emsg("GetWorkers", "No worker found: do nothing");
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
      // Check if this is a priviliged client
      char *p = 0;
      if ((p = (char *) strstr(fgSuperUsers, fClientID))) {
         if (p == fgSuperUsers || (p > fgSuperUsers && *(p-1) == ',')) {
            if (!(strncmp(p, fClientID, strlen(fClientID)))) {
               fSuperUser = 1;
               TRACEP(REQ,"Login: privileged user ");
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
   char uname[9];

   // Unmarshall the data
   pid = (int)ntohl(fRequest.login.pid);
   for (i = 0; i < (int)sizeof(uname)-1; i++) {
      if (fRequest.login.username[i] == '\0' || fRequest.login.username[i] == ' ')
         break;
      uname[i] = fRequest.login.username[i];
   }
   uname[i] = '\0';

   // No 'root' logins
   if (!strncmp(uname, "root", 4)) {
      TRACEP(REQ,"Login: 'root' logins not accepted ");
      fResponse.Send(kXR_InvalidRequest,"Login: 'root' logins not accepted");
      return rc;
   }

   // Here we check if the user is known locally.
   // If not, we fail for now.
   // In the future we may try to get a temporary account
   if (GetUserInfo(uname, fUI) != 0) {
      TRACEP(REQ,"Login: unknown ClientID: "<<uname);
      fResponse.Send(kXR_InvalidRequest,"Login: unknown ClientID");
      return rc;
   }

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
      fResponse.Send(kXR_InvalidRequest,"Login: unknown ClientID");
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
      if ((p = (char *) strstr(fgSuperUsers, fClientID))) {
         if (p == fgSuperUsers || (p > fgSuperUsers && *(p-1) == ',')) {
            if (!(strncmp(p, fClientID, strlen(fClientID)))) {
               fSuperUser = 1;
               TRACEP(REQ,"Login: privileged user ");
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

   // Flag for internal connections
   bool proofsrv = ((fSrvType == kXPD_Internal) && all) ? 1 : 0;

   // If call back from proofsrv, find out the target session
   short int psid = -1;
   char protver = -1;
   short int clientvers = -1;
   if (proofsrv) {
      memcpy(&psid, (const void *)&(fRequest.login.reserved[0]), 2);
      if (psid < 0) {
         fResponse.Send(kXR_InvalidRequest,
                        "MapClient: proofsrv callback: sent invalid session id");
         return rc;
      }
      protver = fRequest.login.capver[0];
      TRACEP(REQ,"MapClient: proofsrv callback for session: " <<psid);
   } else {
      // Get PROOF version run by client
      memcpy(&clientvers, (const void *)&(fRequest.login.reserved[0]), 2);
      TRACEP(REQ,"MapClient: PROOF version run by client: " <<clientvers);
   }

   // On workers and submasters get associated login buffer:
   // it contains the main session tag;
   XrdOucString stag;
   XrdOucString ord = "0";
   if (fSrvType == kXPD_WorkerServer || fSrvType == kXPD_MasterServer) {
      char *buf = fArgp->buff;
      int   len = fRequest.proof.dlen;
      stag.assign(buf,0,len-1);
      int iord = stag.find("|ord:");
      if (iord != STR_NPOS) {
         ord.assign(stag,iord+5);
         int icf = ord.find("|cf:");
         if (icf != STR_NPOS)
            ord.erase(icf);
         stag.erase(iord);
      }
      TRACEP(REQ, "MapClient: session tag: "<<stag);
      TRACEP(REQ, "MapClient: ordinal:     "<<ord);
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
            TRACEP(REQ,"MapClient: proofsrv callback:"
                       " link assigned to target session "<<psid);
         }
      } else {

         // The index of the next free slot will be the unique ID
         fCID = pmgr->GetClientID(this);

         // Set session tag, if required
         if (stag.length() > 0)
            pmgr->SetSessionTag(stag.c_str());
         if (ord.length() > 0)
            pmgr->SetOrdinal(ord.c_str());

         // If any PROOF session in shutdown state exists, stop the related
         // shutdown timers
         if (pmgr->fProofServs.size() > 0) {
            XrdProofServProxy *psrv = 0;
            int is = 0;
            for (is = 0; is < (int) pmgr->fProofServs.size(); is++) {
               if ((psrv = pmgr->fProofServs.at(is)) &&
                    psrv->IsValid() && psrv->SrvType() == kXPD_TopMaster) {
                  if (SetShutdownTimer(psrv, 0) != 0) {
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
         fResponse.Send(kXP_nomanager,
                        "MapClient: proofsrv callback:"
                        " no manager to attach to: protocol error");
         return rc;
      }

      // No existing session: create a new one
      pmgr = new XrdProofClient(this, clientvers, stag.c_str(), ord.c_str());
      if (pmgr) {
         TRACEP(REQ,"MapClient: NEW client: "<<pmgr<<", "<<pmgr->ID());

         // The index of the next free slot will be the unique ID
         fCID = pmgr->GetClientID(this);

         // Add to the list
         fgProofClients.push_back(pmgr);

         // Save as reference proof mgr
         fPClient = pmgr;

         // Reference Stream ID
         unsigned short sid;
         memcpy((void *)&sid, (const void *)&(fRequest.header.streamid[0]), 2);
         pmgr->SetRefSid(sid);

         TRACEP(REQ,"MapClient: client "<<pmgr<<" added to the list (ref sid: "<< sid<<")");
      }
   }

   if (!proofsrv) {
      TRACEP(REQ,"MapClient: fCID = "<<fCID<<"; size = "<<fPClient->fClients.size()<<
              "; capacity = "<<fPClient->fClients.capacity());
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
int XrdProofdProtocol::GetBuff(int quantum)
{
   // Allocate a buffer to handle quantum bytes

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
   if (!fPClient || !INRANGE(psid, fPClient->fProofServs) ||
       !(xps = fPClient->fProofServs.at(psid))) {
      TRACEP(REQ, "Attach: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
      return rc;
   }

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
   if (fSrvType == kXPD_TopMaster) {
      // Send also back the data pool url
      XrdOucString dpu = fgPoolURL;
      if (!dpu.endswith('/'))
         dpu += '/';
      dpu += fgNamespace;
      fResponse.Send(psid, fgSrvProtVers, (void *) dpu.c_str(), dpu.length());
   } else
      fResponse.Send(psid, fgSrvProtVers);

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
   if (!fPClient || !INRANGE(psid,fPClient->fProofServs) ||
       !(xps = fPClient->fProofServs.at(psid))) {
      TRACEP(REQ, "Detach: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
      return rc;
   }

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
      if (!fPClient || !INRANGE(psid,fPClient->fProofServs) ||
          !(xpsref = fPClient->fProofServs.at(psid))) {
         TRACEP(REQ, "Destroy: reference session ID not found");
         fResponse.Send(kXR_InvalidRequest,"reference session ID not found");
         return rc;
      }
   }

   // Loop over servers
   XrdProofServProxy *xps = 0;
   int is = 0;
   for (is = 0; is < (int) fPClient->fProofServs.size(); is++) {

      if ((xps = fPClient->fProofServs.at(is)) && (xpsref == 0 || xps == xpsref)) {

         TRACEP(REQ, "Destroy: xps: "<<xps<<", status: "<< xps->Status()<<", pid: "<<xps->SrvID());

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
                         (p != this) && p->fTopClient) {
                        unsigned short sid;
                        p->fResponse.GetSID(sid);
                        p->fResponse.Set(xps->fClients.at(ic)->fSid);
                        p->fResponse.Send(kXR_attn, kXPD_srvmsg, msg, len);
                        p->fResponse.Set(sid);
                     }
                  }
               }
            }

            // Send a terminate signal to the proofserv
            if (TerminateProofServ(xps) != 0)
               if (KillProofServ(xps,1) != 0) {
                  TRACEP(REQ, "Destroy: problems terminating request to proofsrv");
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
int XrdProofdProtocol::SetProofServEnv(XrdProofdProtocol *p,
                                       int psid, int loglevel, const char *cfg)
{
   // Set environment for proofserv

   char *ev = 0;

   TRACE(REQ,"SetProofServEnv: enter: psid: "<<psid<<", log: "<<loglevel);

   if (!p) {
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

      // Port (really needed?)
      ev = new char[25];
      sprintf(ev, "ROOTXPDPORT=%d", fgPort);
      putenv(ev);

      // Done
      return 0;
   }

   // The rest only if a full session

   // Make sure the principla client is defined
   if (!(p->fPClient)) {
      PRINT("SetProofServEnv: principal client undefined - cannot continue");
      return -1;
   }

   // Work directory
   TRACE(REQ,"SetProofServEnv: setting working dir for: "<<p->fClientID);
   XrdOucString udir = p->fUI.fHomeDir;
   if (fgWorkDir) {

      // Make sure that the user directory exists
      udir = fgWorkDir;
      if (!udir.endswith('/'))
         udir += "/";
      udir += p->fClientID;
   } else {
      // Default
      if (!udir.endswith('/'))
         udir += "/";
      udir += "proof";
   }
   // Make sure the directory exists
   if (AssertDir(udir.c_str(), p->fUI) == -1) {
      PRINT("SetProofServEnv: unable to create work dir: "<<udir);
      return -1;
   }

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
   if (p->fSrvType == kXPD_TopMaster) {
      logdir += "/session-";
      logdir += stag;
   } else {
      logdir += "/";
      logdir += p->fPClient->SessionTag();
   }
   TRACE(REQ,"SetProofServEnv: log dir "<<logdir);
   // Make sure the directory exists
   if (AssertDir(logdir.c_str(), p->fUI) == -1) {
      PRINT("SetProofServEnv: unable to create log dir: "<<logdir);
      return -1;
   }
   // The session dir (sandbox) depends on the role
   XrdOucString sessdir = logdir;
   if (p->fSrvType == kXPD_WorkerServer)
      sessdir += "/worker-";
   else
      sessdir += "/master-";
   sessdir += p->fPClient->Ordinal();
   sessdir += "-";
   sessdir += stag;
   ev = new char[strlen("ROOTPROOFSESSDIR=")+sessdir.length()+2];
   sprintf(ev, "ROOTPROOFSESSDIR=%s", sessdir.c_str());
   putenv(ev);
   TRACE(REQ,"SetProofServEnv: "<<ev);

   // Log level
   ev = new char[strlen("ROOTPROOFLOGLEVEL=")+5];
   sprintf(ev, "ROOTPROOFLOGLEVEL=%d", loglevel);
   putenv(ev);
   TRACE(REQ,"SetProofServEnv: "<<ev);

   // Ordinal number
   ev = new char[strlen("ROOTPROOFORDINAL=")+strlen(p->fPClient->Ordinal())+2];
   sprintf(ev, "ROOTPROOFORDINAL=%s", p->fPClient->Ordinal());
   putenv(ev);
   TRACE(REQ,"SetProofServEnv: "<<ev);

   // Create the env file
   TRACE(REQ,"SetProofServEnv: creating env file");
   XrdOucString envfile = sessdir;
   envfile += ".env";
   FILE *fenv = fopen(envfile.c_str(), "w");
   if (!fenv) {
      PRINT("SetProofServEnv: unable to open env file: "<<envfile);
      return -1;
   }
   TRACE(REQ,"SetProofServEnv: environment file: "<< envfile);

   // Set ROOTSYS
   fprintf(fenv, "ROOTSYS=%s\n", fgROOTsys);

   // Set conf dir
   fprintf(fenv, "ROOTCONFDIR=%s\n", fgROOTsys);

   // Set TMPDIR
   fprintf(fenv, "ROOTTMPDIR=%s\n", fgTMPdir);

   // Port (really needed?)
   fprintf(fenv, "ROOTXPDPORT=%d\n", fgPort);

   // Work dir
   fprintf(fenv, "ROOTPROOFWORKDIR=%s\n", udir.c_str());

   // Session tag
   fprintf(fenv, "ROOTPROOFSESSIONTAG=%s\n", stag);

   // Whether user specific config files are enabled
   if (fgWorkerUsrCfg)
      fprintf(fenv, "ROOTUSEUSERCFG=1\n");

   // Set Open socket
   XrdProofServProxy *xps = p->fPClient->fProofServs.at(psid);
   if (!xps) {
      PRINT("SetProofServEnv: unable to get instance of proofserv proxy");
      return -1;
   }
   fprintf(fenv, "ROOTOPENSOCK=%s\n", xps->UNIXSockPath());

   // Entity
   fprintf(fenv, "ROOTENTITY=%s@%s\n", p->fClientID, p->fLink->Host());

   // Session ID
   fprintf(fenv, "ROOTSESSIONID=%d\n", psid);

   // Client ID
   fprintf(fenv, "ROOTCLIENTID=%d\n", p->fCID);

   // Client Protocol
   fprintf(fenv, "ROOTPROOFCLNTVERS=%d\n", p->fPClient->Version());

   // Ordinal number
   fprintf(fenv, "ROOTPROOFORDINAL=%s\n", p->fPClient->Ordinal());

   // Config file
   if (cfg)
      fprintf(fenv, "ROOTPROOFCFGFILE=%s\n", cfg);

   // Default number of workers
   fprintf(fenv, "ROOTPROOFMAXSESSIONS=%d\n", fgMaxSessions);

   // Log file in the log dir
   XrdOucString logfile = sessdir;
   logfile += ".log";
   fprintf(fenv, "ROOTPROOFLOGFILE=%s\n", logfile.c_str());
   xps->SetFileout(logfile.c_str());

   // Close file
   fclose(fenv);

   // Create or Update symlink to last session
   TRACE(REQ,"SetProofServEnv: creating symlink");
   XrdOucString syml = udir;
   if (p->fSrvType == kXPD_WorkerServer)
      syml += "/last-worker-session";
   else
      syml += "/last-master-session";
   if (SymLink(logdir.c_str(), syml.c_str()) != 0) {
      TRACE(REQ,"SetProofServEnv: problems creating symlink to "
                " last session (errno: "<<errno<<")");
   }

   // We are done
   TRACE(REQ,"SetProofServEnv: done");
   return 0;
}

//_________________________________________________________________________________
int XrdProofdProtocol::Create()
{
   // Handle a request to create a new session

   int psid = -1, rc = 1;

   // Allocate next free server ID and fill in the basic stuff
   psid = GetFreeServID();
   XrdProofServProxy *xps = fPClient->fProofServs.at(psid);
   xps->SetClient((const char *)fClientID);
   xps->SetID(psid);
   xps->SetSrvType(fSrvType);

   // Unmarshall log level
   int loglevel = ntohl(fRequest.proof.int1);

   // Parse buffer
   char *buf = fArgp->buff;
   int   len = fRequest.proof.dlen;
   // Extract ordinal number
   XrdOucString ord = "0";
   if ((fSrvType == kXPD_WorkerServer) || (fSrvType == kXPD_MasterServer)) {
      ord.assign(buf,0,len-1);
      int iord = ord.find("|ord:");
      ord.erase(0,iord+5);
      ord.erase(ord.find("|cf:"));
   }
   TRACEP(REQ, "Create: ordinal: "<<ord);
   fPClient->SetOrdinal(ord.c_str());
   // Extract config file, if any (for backward compatibility)
   XrdOucString cffile;
   cffile.assign(buf,0,len-1);
   int icf = cffile.find("|cf:");
   cffile.erase(0,icf+4);
   TRACEP(REQ, "Create: cfg file: "<<cffile);

   // Notify
   TRACEP(REQ, "Create: new psid: "<<psid<<"; client ID: "<<fCID<<"; loglev= "<<loglevel);

   // UNIX socket for internal communications (to/from proofsrv)
   if (xps->CreateUNIXSock(&fgEDest, fgTMPdir) != 0) {
      xps->Reset();
      // Failure creating UNIX socket
      fResponse.Send(kXR_ServerError,
                     "could not create UNIX socket for internal communications");
      return rc;
   }

   // Here we start the fork attempts: for some weird problem on SMP
   // machines there is a non-zero probability for a deadlock situation
   // in system mutexes. For that reasone we are ready to retry a few times.
   // The semaphore seems to have solved the problem. We leave the retry
   // structure in place in case of need.
   int ntry = 1;
   bool notdone = 1;
   int fp[2];
   int setupOK = 0;
   int pollRet = 0;
   int pid = -1;
   while (ntry-- && notdone) {

      // This must be serialized
      if (fgForkSem.Wait(5) != 0)
         // Timed-out: retry if required, or quit
         continue;

      // Pipe to communicate status of setup
      if (pipe(fp) != 0) {
         xps->Reset();
         // Failure creating UNIX socket
         fResponse.Send(kXR_ServerError,
                        "unable to create pipe for status-of-setup communication");
         return rc;
      }

      // Fork an agent process to handle this session
      pid = -1;
      if (!(pid = fgSched->Fork("proofsrv"))) {

         int setupOK = 0;

         // We set to the user environment
         if (SetUserEnvironment(xps, fClientID) != 0) {
            PRINT("Create: SetUserEnvironment did not return OK - EXIT");
            write(fp[1], &setupOK, sizeof(setupOK));
            close(fp[0]);
            close(fp[1]);
            exit(1);
         }

         char *argvv[5] = {0};

         // We add our PID to be able to identify processes coming from us
         char cpid[10] = {0};
         sprintf(cpid, "%d", getppid());

         // start server
         argvv[0] = (char *)fgPrgmSrv;
         argvv[1] = (char *)((fSrvType == kXPD_WorkerServer) ? "proofslave"
                          : "proofserv");
         argvv[2] = (char *)"xpd";
         argvv[3] = (char *)cpid;
         argvv[4] = 0;

         // Set environment for proofserv
         if (SetProofServEnv(this, psid, loglevel, cffile.c_str()) != 0) {
            PRINT("Create: SetProofServEnv did not return OK - EXIT");
            write(fp[1], &setupOK, sizeof(setupOK));
            close(fp[0]);
            close(fp[1]);
            exit(1);
         }

         // Setup OK: now we go
         setupOK = 1;
         write(fp[1], &setupOK, sizeof(setupOK));

         // Cleanup
         close(fp[0]);
         close(fp[1]);

         TRACE(REQ,"Create: fClientID: "<<fClientID<<
                   ", uid: "<<getuid()<<", euid:"<<geteuid());

         // Run the program
         execv(fgPrgmSrv, argvv);

         // We should not be here!!!
         TRACEP(REQ, "Create: returned from execv: bad, bad sign !!!");
         exit(1);
      }

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
      setupOK = 0;
      struct pollfd fds_r;
      fds_r.fd = fp[0];
      fds_r.events = POLLIN;
      pollRet = 0;
      while ((pollRet = poll(&fds_r, 1, 2000)) < 0 &&
             (errno == EINTR)) { }
      if (pollRet > 0) {
         if (read(fp[0], &setupOK, sizeof(setupOK)) != sizeof(setupOK)) {
            xps->Reset();
            fResponse.Send(kXR_ServerError, "problems receiving status-of-setup after forking");
            close(fp[0]);
            close(fp[1]);
            return rc;
         }
         // We are done
         notdone = 0;
      } else if (pollRet == 0) {
         // Got timeout: kill the process and retry, if not done too many times
         close(fp[0]);
         close(fp[1]);
         if (KillProofServ(pid, 1) != 0) {
            // Failed killing process: something starnge is going on: stop
            // trying
            xps->Reset();
            fResponse.Send(kXR_ServerError,
                           "time out after forking: process cannot be killed: stop retrying");
            return rc;
         }
      }

   } // retry loop

   // Cleanup
   close(fp[0]);
   close(fp[1]);

   // If we are not done we quit
   if (notdone) {
      xps->Reset();
      fResponse.Send(kXR_ServerError,
                     "could not fork the proofserv process: quitting");
      return rc;
   }

   // Notify to user
   if (setupOK == 1) {
      if (fSrvType == kXPD_TopMaster) {
         // Send also back the data pool url
         XrdOucString dpu = fgPoolURL;
         if (!dpu.endswith('/'))
            dpu += '/';
         dpu += fgNamespace;
         fResponse.Send(psid, fgSrvProtVers, (void *) dpu.c_str(), dpu.length());
      } else
         fResponse.Send(psid, fgSrvProtVers);
   } else {
      // Failure
      xps->Reset();
      if (pollRet != 0) {
         fResponse.Send(kXR_ServerError, "failure setting up proofserv");
      } else {
         fResponse.Send(kXR_ServerError, "failure setting up proofserv - timeout reached");
      }
      return rc;
   }

   // now we wait for the callback to be (successfully) established
   TRACEP(REQ, "Create: server launched: wait for callback ");

   // We will get back a peer to initialize a link
   XrdNetPeer peerpsrv;
   XrdLink   *linkpsrv = 0;
   int lnkopts = 0;

   // Perform regular accept
   if (!(xps->UNIXSock()->Accept(peerpsrv, XRDNET_NODNTRIM, fgInternalWait))) {

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
      TRACE(REQ, "Accepted connection from " << peerpsrv.InetName);

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
            TRACEP(REQ, "Create: priority of the child process changed by "
                        << dp << " units");
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

//______________________________________________________________________________
int XrdProofdProtocol::SendData(XrdProofdResponse *resp,
                                kXR_int32 sid, XrdSrvBuffer **buf)
{
   // Send data over the open link. Segmentation is done here, if required.

   int rc = 1;

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
      for (ic = 0; ic < (int) xps->fClients.size(); ic++) {
         if ((csid = xps->fClients.at(ic)) && csid->fP) {
            XrdProofdResponse& resp = csid->fP->fResponse;
            int rs = 0;
            {  XrdOucMutexHelper mhp(resp.fMutex);
               unsigned short sid;
               resp.GetSID(sid);
               TRACEP(REQ, "SendDataN: INTERNAL: this sid: "<<sid<<
                           "; client sid:"<<csid->fSid);
               resp.Set(csid->fSid);
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

   XrdOucMutexHelper mh(fResponse.fMutex);

   // Unmarshall the data
   int psid = ntohl(fRequest.sendrcv.sid);
   int opt = ntohl(fRequest.sendrcv.opt);
   TRACEP(REQ, "SendMsg: psid: "<<psid<<"; opt: "<< opt);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (!fPClient || !INRANGE(psid,fPClient->fProofServs) ||
       !(xps = fPClient->fProofServs.at(psid))) {
      TRACEP(REQ, "SendMsg: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
      return rc;
   }

   TRACEP(REQ, "SendMsg: xps: "<<xps<<", status: "<<xps->Status());

   // Type of connection
   bool external = !(opt & kXPD_internal);

   // Forward message as unsolicited
   int   len = fRequest.header.dlen;

   if (external) {
      TRACEP(REQ, "SendMsg: EXTERNAL: psid: "<<psid);

      // Send to proofsrv our client ID
      if (fCID == -1) {
         fResponse.Send(kXR_ServerError,"external: getting clientSID");
         return rc;
      }
      TRACEP(REQ, "SendMsg: EXTERNAL: fCID: " << fCID);
      if (SendData(&(xps->fProofSrv), fCID)) {
         fResponse.Send(kXR_ServerError,"external: sending message to proofserv");
         return rc;
      }
      // Notify to user
      fResponse.Send();
      TRACEP(REQ, "SendMsg: EXTERNAL: message sent to proofserv ("<<len<<" bytes)");

   } else {
      TRACEP(REQ, "SendMsg: INTERNAL: psid: "<<psid);

      XrdSrvBuffer **savedBuf = 0;
      // Additional info about the message
      if (opt & kXPD_setidle) {
         TRACEP(REQ, "SendMsg: INTERNAL: setting proofserv in 'idle' state");
         if (xps->fStatus != kXPD_shutdown)
            xps->fStatus = kXPD_idle;
         // Clean start processing message, if any
         if (xps->fStartMsg) {
            delete xps->fStartMsg;
            xps->fStartMsg = 0;
         }
      } else if (opt & kXPD_querynum) {
         TRACEP(REQ, "SendMsg: INTERNAL: got message with query number");
         // Save query num message for later clients
         SafeDelete(xps->fQueryNum);
         savedBuf = &(xps->fQueryNum);
      } else if (opt & kXPD_startprocess) {
         TRACEP(REQ, "SendMsg: INTERNAL: setting proofserv in 'running' state");
         xps->fStatus = kXPD_running;
         // Save start processing message for later clients
         SafeDelete(xps->fStartMsg);
         savedBuf = &(xps->fStartMsg);
      } else if (opt & kXPD_logmsg) {
         // We broadcast log messages only not idle to catch the
         // result from processing
         if (xps->fStatus == kXPD_running) {
            TRACEP(REQ, "SendMsg: INTERNAL: broadcasting log message");
            opt |= kXPD_fb_prog;
         }
      }
      bool fbprog = (opt & kXPD_fb_prog);

      if (!fbprog) {
         // Get ID of the client
         int cid = ntohl(fRequest.sendrcv.cid);
         TRACEP(REQ, "SendMsg: INTERNAL: cid: "<<cid);

         // Get corresponding instance
         XrdClientID *csid = 0;
         if (!xps || !INRANGE(cid, xps->fClients) ||
             !(csid = xps->fClients.at(cid))) {
            TRACEP(REQ, "SendMsg: client ID not found (cid = "<<cid<<
                        "; size = "<<xps->fClients.size()<<")");
            fResponse.Send(kXR_InvalidRequest,"Client ID not found");
            return rc;
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
            TRACEP(REQ, "SendMsg: INTERNAL: this sid: "<<sid<<
                        "; client sid:"<<csid->fSid);
            csid->fP->fResponse.Set(csid->fSid);
            rs = SendData(&(csid->fP->fResponse), -1, savedBuf);
            csid->fP->fResponse.Set(sid);
         }
         if (rs) {
            fResponse.Send(kXR_ServerError,
                           "SendMsg: INTERNAL: sending message to client"
                           " or master proofserv");
            return rc;
         }
      } else {
         // Send to all connected clients
         if (SendDataN(xps, savedBuf)) {
            fResponse.Send(kXR_ServerError,
                           "SendMsg: INTERNAL: sending message to client"
                           " or master proofserv");
            return rc;
         }
      }
      TRACEP(REQ, "SendMsg: INTERNAL: message sent to "<<crecv[xps->fSrvType]<<
                  " ("<<len<<" bytes)");
      // Notify to proofsrv
      fResponse.Send();
   }

   // Over
   return rc;
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

   TRACEP(REQ, "Urgent: psid: "<<psid<<" type: "<< type);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (!fPClient || !INRANGE(psid, fPClient->fProofServs) ||
       !(xps = fPClient->fProofServs.at(psid))) {
      TRACEP(REQ, "Urgent: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"Urgent: session ID not found");
      return rc;
   }

   TRACEP(REQ, "Urgent: xps: "<<xps<<", status: "<<xps->Status());

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
   if (xps->fProofSrv.Send(kXR_attn, kXPD_urgent, buf, len) != 0) {
      fResponse.Send(kXP_ServerError,
                     "Urgent: could not propagate request to proofsrv");
      return rc;
   }

   // Notify to user
   fResponse.Send();
   TRACEP(REQ, "Urgent: request propagated to proofsrv");

   // Over
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::Admin()
{
   // Handle generic request of administrative type

   int rc = 1;

   // Unmarshall the data
   //
   int psid = ntohl(fRequest.proof.sid);
   int type = ntohl(fRequest.proof.int1);

   TRACEP(REQ, "Admin: type: "<<type<<", psid: "<<psid);

   if (type == kQuerySessions) {

      XrdProofServProxy *xps = 0;
      int ns = 0;
      std::vector<XrdProofServProxy *>::iterator ip;
      for (ip = fPClient->fProofServs.begin(); ip != fPClient->fProofServs.end(); ++ip)
         if ((xps = *ip) && xps->IsValid() && (xps->fSrvType == kXPD_TopMaster)) {
            ns++;
            TRACEP(REQ, "Admin: found: " << xps << "(" << xps->IsValid() <<")");
         }

      // Generic info about all known sessions
      int len = (kXPROOFSRVTAGMAX+kXPROOFSRVALIASMAX+30)* (ns+1);
      char *buf = new char[len];
      if (!buf) {
         TRACEP(REQ, "Admin: no resources for results");
         fResponse.Send(kXR_NoMemory, "Admin: out-of-resources for results");
         return rc;
      }
      sprintf(buf, "%d", ns);

      xps = 0;
      for (ip = fPClient->fProofServs.begin(); ip != fPClient->fProofServs.end(); ++ip) {
         if ((xps = *ip) && xps->IsValid() && (xps->fSrvType == kXPD_TopMaster)) {
            sprintf(buf,"%s | %d %s %s %d %d",
                    buf, xps->fID, xps->fTag, xps->fAlias,
                    xps->Status(), xps->GetNClients());
         }
      }
      TRACEP(REQ, "Admin: sending: "<<buf);

      // Send back to user
      fResponse.Send(buf,strlen(buf)+1);
      if (buf) delete[] buf;

   } else if (type == kCleanupSessions) {

      // This part may be not thread safe
      XrdOucMutexHelper mtxh(&fgXPDMutex);

      // Target client (default us)
      XrdProofClient * tgtclnt = fPClient;

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
               // Find the client instance
               XrdProofClient *c = 0;
               std::list<XrdProofClient *>::iterator i;
               for (i = fgProofClients.begin(); i != fgProofClients.end(); ++i) {
                  if ((c = *i) && c->Match(usr)) {
                     tgtclnt = c;
                     clntfound = 1;
                     break;
                  }
               }
               TRACEP(REQ, "Admin: CleanupSessions: superuser, cleaning usr: "<< usr);
            }
         } else {
            TRACEP(REQ, "Admin: CleanupSessions: superuser, all sessions cleaned");
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
         TRACEP(REQ, "Admin: specified client has no sessions - do nothing");
      }

      if (clntfound) {

         // The clinets to cleaned
         std::list<XrdProofClient *> *clnts;
         if (all) {
            // The full list
            clnts = &fgProofClients;
         } else {
            clnts = new std::list<XrdProofClient *>;
            clnts->push_back(tgtclnt);
         }

         // List of process IDs asked to terminate
         std::list<int *> signalledpid;

         // Loop over them
         XrdProofClient *c = 0;
         std::list<XrdProofClient *>::iterator i;
         for (i = clnts->begin(); i != clnts->end(); ++i) {
            if ((c = *i)) {

               // Notify the attached clients that we are going to cleanup
               XrdOucString msg = "Admin: CleanupSessions: cleaning up client: requested by: ";
               msg += fLink->ID;
               int ic = 0;
               XrdProofdProtocol *p = 0;
               for (ic = 0; ic < (int) c->fClients.size(); ic++) {
                  if ((p = c->fClients.at(ic)) && (p != this) && p->fTopClient) {
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

               // Loop over client sessions and terminated them
               int is = 0;
               XrdProofServProxy *s = 0;
               for (is = 0; is < (int) c->fProofServs.size(); is++) {
                  if ((s = c->fProofServs.at(is)) && s->IsValid() &&
                     s->SrvType() == srvtype) {
                     int *pid = new int;
                     *pid = s->SrvID();
                     TRACEP(REQ, "Admin: CleanupSessions: terminating " << *pid);
                     if (TerminateProofServ(s, 0) != 0) {
                        if (KillProofServ(*pid, 0, 0) != 0) {
                           XrdOucString msg = "Admin: CleanupSessions: WARNING: process ";
                           msg += *pid;
                           msg += " could not be signalled for termination";
                           TRACEP(REQ, msg.c_str());
                        } else
                           signalledpid.push_back(pid);
                     } else
                        signalledpid.push_back(pid);
                     // Reset session proxy
                     s->Reset();
                  }
               }
            }
         }

         // Now we give sometime to sessions to terminate (10 sec).
         // We check the status every second
         int nw = 10;
         int nleft = signalledpid.size();
         while (nw-- && nleft > 0) {

            // Loop over the list of processes requested to terminate
            std::list<int *>::iterator ii;
            for (ii = signalledpid.begin(); ii != signalledpid.end(); )
               if (VerifyProcessByID(*(*ii)) == 0) {
                  nleft--;
                  delete (*ii);
                  ii = signalledpid.erase(ii);
               } else
                  ++ii;

            // Wait a bit before retrying
            sleep(1);
         }
      }

      // Now we cleanup what left (any zombies or super resistent processes)
      CleanupProofServ(all, usr);

      // Cleanup all possible sessions around
      Broadcast(type, usr);

      // Cleanup usr
      SafeDelete(usr);

      // Acknowledge user
      fResponse.Send();

   } else if (type == kSessionTag) {

      //
      // Specific info about a session
      XrdProofServProxy *xps = 0;
      if (!fPClient || !INRANGE(psid, fPClient->fProofServs) ||
          !(xps = fPClient->fProofServs.at(psid))) {
         TRACEP(REQ, "Admin: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"Admin: session ID not found");
         return rc;
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
      if (!fPClient || !INRANGE(psid, fPClient->fProofServs) ||
          !(xps = fPClient->fProofServs.at(psid))) {
         TRACEP(REQ, "Admin: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"Admin: session ID not found");
         return rc;
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
      if (!fPClient || !INRANGE(psid, fPClient->fProofServs) ||
          !(xps = fPClient->fProofServs.at(psid))) {
         TRACEP(REQ, "Admin: session ID not found");
         fResponse.Send(kXR_InvalidRequest,"session ID not found");
         return rc;
      }

      // We should query the chosen resource provider
      XrdOucString wrks;
      if (GetWorkers(wrks, xps) !=0 ) {
         // Something wrong
         fResponse.Send(kXR_InvalidRequest,"Admin: GetWorkers failed");
         return rc;
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
   if (!fPClient || !INRANGE(psid, fPClient->fProofServs) ||
       !(xps = fPClient->fProofServs.at(psid))) {
      TRACEP(REQ, "Interrupt: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"nterrupt: session ID not found");
      return rc;
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
   if (!fPClient || !INRANGE(psid,fPClient->fProofServs) ||
       !(xps = fPClient->fProofServs.at(psid))) {
      TRACEP(REQ, "Ping: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
      return rc;
   }

   kXR_int32 pingres = 0;
   if (xps) {
      TRACEP(REQ, "Ping: xps: "<<xps<<", status: "<<xps->Status());

      // Type of connection
      bool external = !(opt & kXPD_internal);

      if (external) {
         TRACEP(REQ, "Ping: EXTERNAL; psid: "<<psid);

         // Send the request
         if ((pingres = (kXR_int32) VerifyProofServ(xps)) == -1) {
            TRACEP(REQ, "Ping: could not verify proofsrv");
            fResponse.Send(kXR_ServerError, "could not verify proofsrv");
            return rc;
         }

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
int XrdProofdProtocol::SetUserEnvironment(XrdProofServProxy *xps,
                                          const char *usr, const char *dir)
{
   // Set user environment: set effective user and group ID of the process
   // to the ones specified by 'usr', change working dir to subdir 'dir'
   // of 'usr' $HOME.
   // Return 0 on success, -1 if enything goes wrong.

   TRACEP(REQ,"SetUserEnvironment: enter: user: "<<usr);

   // Change to user's home dir
   XrdOucString home = fUI.fHomeDir;
   if (dir) {
      home += '/';
      home += dir;
      struct stat st;
      if (stat(home.c_str(), &st) || !S_ISDIR(st.st_mode)) {
         // Specified path does not exist or is not a dir
         TRACEP(REQ,"SetUserEnvironment: subpath "<<dir<<
                    " does not exist or is not a dir");
         home = fUI.fHomeDir;
      }
   }
   TRACEP(REQ,"SetUserEnvironment: changing dir to : "<<home);
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
   TRACEP(REQ,"SetUserEnvironment: setting ACLs");
   if ((int) geteuid() != fUI.fUid) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (!pGuard.Valid()) {
         TRACEP(REQ,"SetUserEnvironment: could not get privileges");
         return -1;
      }

      initgroups(usr, fUI.fGid);

      // Set ownership of the socket file to the client
      if (chown(xps->UNIXSockPath(), fUI.fUid, fUI.fGid) == -1) {
         TRACEP(REQ,"SetUserEnvironment: cannot set user ownership"
               " on UNIX socket (errno: "<<errno<<")");
         return -1;
      }
   }

   // acquire permanently target user privileges
   TRACEP(REQ,"SetUserEnvironment: acquire target user identity");
   if (XrdSysPriv::ChangePerm((uid_t)fUI.fUid, (gid_t)fUI.fGid) != 0) {
      TRACEP(REQ,"SetUserEnvironment: can't acquire "<< usr <<" identity");
      return -1;
   }

   // We are done
   TRACEP(REQ,"SetUserEnvironment: done");
   return 0;
}

//______________________________________________________________________________
bool XrdProofdProtocol::CanDoThis(const char *client)
{
   // Check if we are allowed to do what foreseen in the procedure
   // calling us

   if (fSuperUser)
      // Always allowed
      return 1;
   else
      // We are allowed to act only on what we own
      if (!strncmp(fClientID, client, strlen(fClientID)))
         return 1;

   // Not allowed by default
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::VerifyProofServ(XrdProofServProxy *xps)
{
   // Check if proofserv process associated with 'xps' is alive.
   // A ping message is sent and the reply waited for the internal timeout.
   // Return 1 if successful, 0 if reply was not received within the
   // internal timeout, -1 in case of error.
   int rc = -1;

   if (!xps || !CanDoThis(xps->Client()))
      return rc;

   // Create semaphore
   xps->fPingSem = new XrdOucSemWait(0);

   // Propagate the ping request
   if (xps->fProofSrv.Send(kXR_attn, kXPD_ping, 0, 0) != 0) {
      TRACEP(REQ, "VerifyProofServ: could not propagate ping to proofsrv");
      SafeDelete(xps->fPingSem);
      return rc;
   }

   // Wait for reply
   rc = 1;
   if (xps->fPingSem->Wait(fgInternalWait) != 0) {
      XrdOucString msg = "VerifyProofServ: did not receive ping reply after ";
      msg += fgInternalWait;
      msg += " secs";
      TRACEP(REQ, msg.c_str());
      rc = 0;
   }

   // Cleanup
   SafeDelete(xps->fPingSem);

   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::SetShutdownTimer(XrdProofServProxy *xps, bool on)
{
   // Start (on = TRUE) or stop (on = FALSE) the shutdown timer for the
   // associated with 'xps'.
   // Return 0 on success, -1 in case of error.
   int rc = -1;

   if (!xps || !CanDoThis(xps->Client()))
      return rc;

   // Prepare buffer
   int len = 2 *sizeof(kXR_int32);
   char *buf = new char[len];
   // Option
   kXR_int32 itmp = (on) ? fgShutdownOpt : -1;
   itmp = static_cast<kXR_int32>(htonl(itmp));
   memcpy(buf, &itmp, sizeof(kXR_int32));
   // Delay
   itmp = (on) ? fgShutdownDelay : 0;
   itmp = static_cast<kXR_int32>(htonl(itmp));
   memcpy(buf + sizeof(kXR_int32), &itmp, sizeof(kXR_int32));
   // Send over
   if (xps->fProofSrv.Send(kXR_attn, kXPD_timer, buf, len) != 0) {
      TRACEP(REQ,"SetShutdownTimer: could not send shutdown info to proofsrv");
   } else {
      rc = 0;
      XrdOucString msg = "SetShutdownTimer: ";
      if (on) {
         if (fgShutdownDelay > 0) {
            msg += "delayed (";
            msg += fgShutdownDelay;
            msg += " secs) ";
         }
         msg += "shutdown notified to process ";
         msg += xps->SrvID();
         if (fgShutdownOpt == 1)
            msg += "; action: when idle";
         else if (fgShutdownOpt == 2)
            msg += "; action: immediate";
      } else {
         msg += "cancellation of shutdown action notified to process ";
         msg += xps->SrvID();
      }
      TRACEP(REQ, msg.c_str());
   }
   // Cleanup
   delete[] buf;

   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::VerifyProcessByID(int pid, const char *pname)
{
   // Check if 'proofserv' (or a process named 'pname') process 'pid' is still
   // in the process table.
   // It invokes the command shell 'ps ax' via popen.
   // Return 1 if running, 0 if not running, -1 if the check could not be run.

   int rc = 0;

   // Build command
   XrdOucString cmd = "ps ax | grep proofserv 2>/dev/null";
   if (pname && strlen(pname))
      cmd.replace("proofserv", pname);

   // Run it ...
   XrdOucString pids = ":";
   FILE *fp = popen(cmd.c_str(), "r");
   if (fp != 0) {
      char line[2048] = { 0 };
      while (fgets(line, sizeof(line), fp)) {
         // Add to the list
         pids += (int) GetLong(line);
         pids += ":";
      }
      fclose(fp);
   } else {
      // Error executing the command
      return -1;
   }

   // Check the list now
   if (pid > -1) {
      XrdOucString spid = ":";
      spid += pid;
      spid += ":";
      if (pids.find(spid) != STR_NPOS)
         rc = 1;
   }

   // Cleanup the list of terminated or killed processes
   if (fgTerminatedProcess.size() > 0) {
      std::list<int *>::iterator i;
      for (i = fgTerminatedProcess.begin(); i != fgTerminatedProcess.end(); ) {
         int xi = *(*i);
         XrdOucString spid = ":";
         spid += xi;
         spid += ":";
         if (pids.find(spid) == STR_NPOS) {
            TRACEP(REQ,"VerifyProcessByID: freeing: "<<(*i)<<", "<<*(*i));
            // Cleanup the integer
            delete *i;
            // Process has terminated: remove it from the list
            i = fgTerminatedProcess.erase(i);
         } else
            ++i;
      }
   }

   // Not found
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::CleanupProofServ(bool all, const char *usr)
{
   // Cleanup (kill) all 'proofserv' processes from the process table.
   // Only the processes associated with the logged client are killed,
   // unless 'all' is TRUE, in which case all 'proofserv' instances are
   // terminated (this requires superuser privileges).
   // Super users can also terminated all processes fo another user (specified
   // via usr).
   // Return number of process notified for termination on success, -1 otherwise

   // Check if 'all' makes sense
   if (all && !fSuperUser) {
      all = 0;
      TRACEP(REQ, "CleanupProofServ: request for all without privileges: setting all = FALSE");
   }

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
   int nk = 0;
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
            int ppid = (int) GetLong(pi);
            TRACEP(REQ, "CleanupProofServ: found alternative parent ID: "<< ppid);
            // If still running then skip
            if (VerifyProcessByID(ppid, "xrootd"))
               continue;
         }
         // Get pid now
         int from = 0;
         if (busr)
            from += strlen(cusr);
         int pid = (int) GetLong(&line[from]);
         // Kill it
         if (KillProofServ(pid, 1) == 0)
            nk++;
      }
      fclose(fp);
   } else {
      // Error executing the command
      return -1;
   }

   // Done
   return nk;
}

//______________________________________________________________________________
int XrdProofdProtocol::KillProofServ(int pid, bool forcekill, bool add)
{
   // Kill the process 'pid'.
   // A SIGTERM is sent, unless 'kill' is TRUE, in which case a SIGKILL is used.
   // If add is TRUE (default) the pid is added to the list of processes
   // requested to terminate.
   // Return 0 on success, -1 if not allowed or other errors occured.

   if (pid > -1) {
      // We need the right privileges to do this
      XrdOucMutexHelper mtxh(&gSysPrivMutex);
      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (pGuard.Valid()) {
         bool signalled = 1;
         if (forcekill)
            // Hard shutdown via SIGKILL 
            if (kill(pid, SIGKILL) != 0) {
               if (errno != ESRCH) {
                  XrdOucString msg = "KillProofServ: could not send SIGKILL to process: ";
                  msg += pid;
                  TRACEP(REQ, msg.c_str());
                  return -1;
               }
               signalled = 0;
            }
         else
            // Softer shutdown via SIGTERM
            if (kill(pid, SIGTERM) != 0) {
               if (errno != ESRCH) {
                  XrdOucString msg = "KillProofServ: could not send SIGTERM to process: ";
                  msg += pid;
                  TRACEP(REQ, msg.c_str());
                  return -1;
               }
               signalled = 0;
            }
         // Add to the list of termination attempts
         if (signalled) {
            if (add) {
               int *ii = new int;
               *ii = pid;
               // This part may be not thread safe
               XrdOucMutexHelper mtxh(&fgXPDMutex);
               fgTerminatedProcess.push_back(ii);
               TRACEP(REQ, "KillProofServ: process ID "<<pid<<" signalled and pushed back");
            } else {
               TRACEP(REQ, "KillProofServ: "<<pid<<" signalled");
            }
         } else {
            TRACEP(REQ, "KillProofServ: process ID "<<pid<<" not found in the process table");
         }
      } else {
        XrdOucString msg = "KillProofServ: could not get privileges";
        TRACEP(REQ, msg.c_str());
        return -1;
      }
   } else {
      return -1;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::KillProofServ(XrdProofServProxy *xps,
                                     bool forcekill, bool add)
{
   // Kill the process associated with the session described by 'xps'.
   // A SIGTERM is sent, unless 'kill' is TRUE, in which case a SIGKILL is used.
   // If add is TRUE (default) the pid is added to the list of processes
   // requested to terminate.
   // Return 0 on success, -1 if not allowed or other errors occured.

   if (!xps || !CanDoThis(xps->Client()))
      return -1;

   int pid = -1;
   {  XrdOucMutexHelper mtxh(&(xps->fMutex));
      pid = xps->SrvID();
   }
   if (pid > -1) {
      // We need the right privileges to do this
      XrdOucMutexHelper mtxh(&gSysPrivMutex);
      XrdSysPrivGuard pGuard(xps->Client());
      if (pGuard.Valid()) {
         bool signalled = 1;
         if (forcekill)
            // Hard shutdown via SIGKILL 
            if (kill(pid, SIGKILL) != 0) {
               if (errno != ESRCH) {
                  XrdOucString msg = "KillProofServ: could not send SIGKILL to process: ";
                  msg += pid;
                  TRACEP(REQ, msg.c_str());
                  return -1;
               }
               signalled = 0;
            }
         else
            // Softer shutdown via SIGTERM
            if (kill(pid, SIGTERM) != 0) {
               if (errno != ESRCH) {
                  XrdOucString msg = "KillProofServ: could not send SIGTERM to process: ";
                  msg += pid;
                  TRACEP(REQ, msg.c_str());
                  return -1;
               }
               signalled = 0;
            }
         if (signalled) {
            if (add) {
               // Add to the list of termination attempts
               int *ii = new int;
               *ii = pid;
               // This part may be not thread safe
               XrdOucMutexHelper mtxh(&fgXPDMutex);
               fgTerminatedProcess.push_back(ii);
               TRACEP(REQ, "KillProofServ: "<<pid<<" signalled and pushed back");
            } else {
               TRACEP(REQ, "KillProofServ: "<<pid<<" signalled");
            }
         } else {
            TRACEP(REQ, "KillProofServ: process ID "<<pid<<" not found in the process table");
         }
      } else {
        XrdOucString msg = "KillProofServ: could not get privileges for: ";
        msg += xps->Client();
        TRACEP(REQ, msg.c_str());
        return -1;
      }
   } else {
      TRACEP(REQ, "KillProofServ: invalid session process ID ("<<pid<<")");
      return -1;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::TerminateProofServ(XrdProofServProxy *xps, bool add)
{
   // Terminate the process associated with the session described by 'xps'.
   // A shutdown interrupt message is forwarded.
   // If add is TRUE (default) the pid is added to the list of processes
   // requested to terminate.
   // Return 0 on success, 1 if the attempt failed, -1 if not allowed
   // or other errors occured.

   if (!xps || !CanDoThis(xps->Client()))
      return -1;

   // Send a terminate signal to the proofserv
   int pid = -1;
   {  XrdOucMutexHelper mtxh(&(xps->fMutex));
      pid = xps->SrvID();
   }
   if (pid > -1) {

      int type = 3;
      if (xps->fProofSrv.Send(kXR_attn, kXPD_interrupt, type) != 0) {
         // Could not send: try termination by signal
         return KillProofServ(xps);
      }
      if (add) {
         // Add to the list of termination attempts
         int *ii = new int;
         *ii = pid;

         // This part may be not thread safe
         XrdOucMutexHelper mtxh(&fgXPDMutex);
         fgTerminatedProcess.push_back(ii);
         TRACEP(REQ, "TerminateProofServ: "<<*ii<<" pushed back");
      }
   }

   // Done
   return 0;
}

//--------------------------------------------------------------------------
//
// XrdProofClient
//
//--------------------------------------------------------------------------

//__________________________________________________________________________
XrdProofClient::XrdProofClient(XrdProofdProtocol *p, short int clientvers,
                               const char *tag, const char *ord)
{
   // Constructor

   fClientID = (p && p->GetID()) ? strdup(p->GetID()) : 0;
   fSessionTag = (tag) ? strdup(tag) : 0;
   fOrdinal = (ord) ? strdup(ord) : 0;
   fClientVers = clientvers;
   fProofServs.reserve(10);
   fClients.reserve(10);
}

//__________________________________________________________________________
XrdProofClient::~XrdProofClient()
{
   // Destructor

   SafeFree(fClientID);
   SafeFree(fSessionTag);
   SafeFree(fOrdinal);
}

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
