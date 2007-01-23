// @(#)root/proofd:$Name:  $:$Id: XrdProofdProtocol.cxx,v 1.39 2007/01/22 11:36:41 rdm Exp $
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

#ifdef __APPLE__
#   ifndef __macos__
#      define __macos__
#   endif
#endif
#ifdef __sun
#   ifndef __solaris__
#      define __solaris__
#   endif
#endif

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
#include <sys/file.h>
#include <dirent.h>

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

// For process info
#if defined(__sun)
#include <procfs.h>
#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// Poll
#include <unistd.h>
#include <sys/poll.h>

#include "XrdVersion.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdSys/XrdSysPriv.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdOuc/XrdOucError.hh"
#include "XrdOuc/XrdOucLogger.hh"
#include "XrdOuc/XrdOucPthread.hh"
#include "XrdOuc/XrdOucReqID.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdOuc/XrdOucTimer.hh"
#include "XrdSut/XrdSutAux.hh"
#include "XrdNet/XrdNet.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "Xrd/XrdLink.hh"
#include "Xrd/XrdPoll.hh"
#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdScheduler.hh"

#include "XrdProofConn.h"
#include "XrdProofdProtocol.h"

#ifdef R__HAVE_CONFIG
#include "RConfigure.h"
#endif

// Tracing utils
#include "XrdProofdTrace.h"
XrdOucTrace          *XrdProofdTrace = 0;
static const char    *gTraceID = " ";

// Static variables
static XrdOucReqID   *XrdProofdReqID = 0;
XrdOucRecMutex        gSysPrivMutex;

// Loggers: we need two to avoid deadlocks
static XrdOucLogger   gMainLogger;
static XrdOucLogger   gForkLogger;

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
kXR_int16             XrdProofdProtocol::fgSrvProtVers = -1;
XrdOucSemWait         XrdProofdProtocol::fgForkSem;   // To serialize fork requests
//
EResourceType         XrdProofdProtocol::fgResourceType = kRTStatic;
int                   XrdProofdProtocol::fgMaxSessions = -1;
int                   XrdProofdProtocol::fgMaxOldLogs = 10;
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
// Cron options
int                   XrdProofdProtocol::fgCron = 1; // Default: start cron thread
int                   XrdProofdProtocol::fgCronFrequency = 60; // Default: run checks every minute
// Access control
int                   XrdProofdProtocol::fgOperationMode = kXPD_OpModeOpen; // Operation mode
XrdOucString          XrdProofdProtocol::fgAllowedUsers; // Users allowed in controlled mode
// Proofserv configuration
XrdOucString          XrdProofdProtocol::fgProofServEnvs; // Additional envs to be exported before proofserv
// Number of workers for local sessions
int                   XrdProofdProtocol::fgNumLocalWrks = -1;
//
// Static area: client section
std::list<XrdProofClient *> XrdProofdProtocol::fgProofClients;  // keeps track of all users
std::list<XrdProofdPInfo *> XrdProofdProtocol::fgTerminatedProcess; // List of pids of processes terminating

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
#define INRANGE(x,y) ((x >= 0) && (x < (int)y->size()))
#endif
#ifndef DIGIT
#define DIGIT(x) (x >= 48 && x <= 57)
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

#ifndef XPDSWAP
#define XPDSWAP(a,b,t) { t = a ; a = b; b = t; }
#endif

#undef  TRACELINK
#define TRACELINK fLink
#undef  RESPONSE
#define RESPONSE fResponse

#undef MHEAD
#define MHEAD "--- Proofd: "

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
   kCleanupSessions,
   kQueryLogPaths,
   kReadBuffer
};

// Security handle
typedef XrdSecService *(*XrdSecServLoader_t)(XrdOucLogger *, const char *cfn);

#ifdef XPD_LONG_MAX
#undefine XPD_LONG_MAX
#endif
#define XPD_LONG_MAX 2147483647
//__________________________________________________________________________
static long int GetLong(char *str)
{
   // Extract first integer from string at 'str', if any

   // Reposition on first digit
   char *p = str;
   while ((*p < 48 || *p > 57) && (*p) != '\0')
      p++;
   if (*p == '\0')
      return XPD_LONG_MAX;

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
   // Return 0 on success, -errno on error

   // Make sure input is defined
   if (!usr || strlen(usr) <= 0)
      return -EINVAL;

   // Call getpwnam_r ...
   struct passwd pw;
   struct passwd *ppw = 0;
   char buf[2048];
#if defined(__sun) && !defined(__GNUC__)
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
   if (errno != 0)
      return ((int) -errno);
   else
      return -ENOENT;
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
#if defined(__sun) && !defined(__GNUC__)
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
   if (errno != 0)
      return ((int) -errno);
   else
      return -ENOENT;
}

//__________________________________________________________________________
static bool SessionTagComp(XrdOucString *&lhs, XrdOucString *&rhs)
{
   // Compare times from session tag strings

   if (!lhs || !rhs)
      return 1;

   // Left hand side
   XrdOucString ll(*lhs);
   ll.erase(ll.rfind('-'));
   ll.erase(0, ll.rfind('-')+1);
   int tl = strtol(ll.c_str(), 0, 10);

   // Right hand side
   XrdOucString rr(*rhs);
   rr.erase(rr.rfind('-'));
   rr.erase(0, rr.rfind('-')+1);
   int tr = strtol(rr.c_str(), 0, 10);

   // Done
   return ((tl < tr) ? 0 : 1);
}

#if defined(__sun)
//__________________________________________________________________________
static void Sort(std::list<XrdOucString *> *lst)
{
   // Sort ascendingly the list.
   // Function used on Solaris where std::list::sort() does not support an
   // alternative comparison algorithm.

   // Check argument
   if (!lst)
      return;

   // If empty or just one element, nothing to do
   if (lst->size() < 2)
      return;

   // Fill a temp array with the current status
   XrdOucString **ta = new XrdOucString *[lst->size()];
   std::list<XrdOucString *>::iterator i;
   int n = 0;
   for (i = lst->begin(); i != lst->end(); ++i)
      ta[n++] = *i;

   // Now start the loops
   XrdOucString *tmp = 0;
   bool notyet = 1;
   int jold = 0;
   while (notyet) {
      int j = jold;
      while (j < n - 1) {
         if (SessionTagComp(ta[j], ta[j+1]))
            break;
         j++;
      }
      if (j >= n - 1) {
         notyet = 0;
      } else {
         jold = j + 1;
         XPDSWAP(ta[j], ta[j+1], tmp);
         int k = j;
         while (k > 0) {
            if (!SessionTagComp(ta[k], ta[k-1])) {
               XPDSWAP(ta[k], ta[k-1], tmp);
            } else {
               break;
            }
            k--;
         }
      }
   }

   // Empty the original list
   lst->clear();

   // Fill it again
   while (n--)
      lst->push_back(ta[n]);

   // Clean up
   delete[] ta;
}
#endif

#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)

typedef struct kinfo_proc kinfo_proc;

//__________________________________________________________________________
static int GetMacProcList(kinfo_proc **plist, int &nproc)
{
   // Returns a list of all processes on the system.  This routine
   // allocates the list and puts it in *plist and counts the
   // number of entries in 'nproc'. Caller is responsible for 'freeing'
   // the list.
   // On success, the function returns 0.
   // On error, the function returns an errno value.
   //
   // Adapted from: reply to Technical Q&A 1123,
   //               http://developer.apple.com/qa/qa2001/qa1123.html
   //

   int rc = 0;
   kinfo_proc *res;
   bool done = 0;
   static const int name[] = {CTL_KERN, KERN_PROC, KERN_PROC_ALL, 0};

   TRACE(ACT, "GetMacProcList: enter ");

   // Declaring name as const requires us to cast it when passing it to
   // sysctl because the prototype doesn't include the const modifier.
   size_t len = 0;

   if (!plist || (*plist))
      return EINVAL;
   nproc = 0;

   // We start by calling sysctl with res == 0 and len == 0.
   // That will succeed, and set len to the appropriate length.
   // We then allocate a buffer of that size and call sysctl again
   // with that buffer.  If that succeeds, we're done.  If that fails
   // with ENOMEM, we have to throw away our buffer and loop.  Note
   // that the loop causes use to call sysctl with 0 again; this
   // is necessary because the ENOMEM failure case sets length to
   // the amount of data returned, not the amount of data that
   // could have been returned.

   res = 0;
   do {
      // Call sysctl with a 0 buffer.
      len = 0;
      if ((rc = sysctl((int *)name, (sizeof(name)/sizeof(*name)) - 1,
                       0, &len, 0, 0)) == -1) {
         rc = errno;
      }

      // Allocate an appropriately sized buffer based on the results
      // from the previous call.
      if (rc == 0) {
         res = (kinfo_proc *) malloc(len);
         if (!res)
            rc = ENOMEM;
      }

      // Call sysctl again with the new buffer.  If we get an ENOMEM
      // error, toss away our buffer and start again.
      if (rc == 0) {
         if ((rc = sysctl((int *)name, (sizeof(name)/sizeof(*name)) - 1,
                          res, &len, 0, 0)) == -1) {
            rc = errno;
         }
         if (rc == 0) {
            done = 1;
         } else if (rc == ENOMEM) {
            if (res)
               free(res);
            res = 0;
            rc = 0;
         }
      }
   } while (rc == 0 && !done);

   // Clean up and establish post conditions.
   if (rc != 0 && !res) {
      free(res);
      res = 0;
   }
   *plist = res;
   if (rc == 0)
      nproc = len / sizeof(kinfo_proc);

   // Done
   return rc;
}
#endif

//__________________________________________________________________________
static int Write(int fd, const void *buf, size_t nb)
{
   // Write nb bytes at buf to descriptor 'fd' ignoring interrupts
   // Return the number of bytes written or -1 in case of error

   if (fd < 0)
      return -1;

   const char *pw = (const char *)buf;
   int lw = nb;
   int nw = 0, written = 0;
   while (lw) {
      if ((nw = write(fd, pw + written, lw)) < 0) {
         if (errno == EINTR) {
            errno = 0;
            continue;
         } else {
            break;
         }
      }
      // Count
      written += nw;
      lw -= nw;
   }

   // Done
   return written;
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

   int freq = *((int *)p);

   while(1) {
      // Wait a while
      XrdOucTimer::Wait(freq*1000);
      // Do something here
      TRACE(REQ, "XrdProofdCron: running periodical checks");
      // Trim the list of processes asked for termination
      XrdProofdProtocol::TrimTerminatedProcesses();
   }

   // Should never come here
   return (void *)0;
}

//__________________________________________________________________________
int XrdProofdProtocol::Broadcast(int type, const char *msg)
{
   // Broadcast request to known potential sub-nodes.
   // Return 0 on success, -1 on error
   int rc = 0;

   TRACEP(ACT, "SendCoordinator: enter: type: "<<type);

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
            TRACEP(HDBG,"Broadcast: sending request to "<<u);
            // Send request
            if (!(xrsp = SendCoordinator(u.c_str(), type, msg, srvtype))) {
               TRACEP(XERR,"Broadcast: problems sending request to "<<u);
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

   TRACEP(ACT, "SendCoordinator: enter: type: "<<type);

   if (!url || strlen(url) <= 0)
      return xrsp;

   // Open the connection
   XrdOucString buf = "session-cleanup-from-";
   buf += fgLocalHost;
   buf += "|ord:000";
   char m = 'A'; // log as admin
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
            TRACEP(XERR,"SendCoordinator: invalid request type "<<type);
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
      TRACEP(XERR,"SendCoordinator: could not open connection to "<<url);
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

   TRACE(ACT, "ChangeProcessPriority: enter: pid: " << pid << ", dp: " << dp);

   // No action requested
   if (dp == 0)
      return 0;

   // Get current priority; errno needs to be reset here, as -1
   // could be a legitimate priority value
   errno = 0;
   int priority = getpriority(PRIO_PROCESS, pid);
   if (priority == -1 && errno != 0) {
      TRACE(XERR, "ChangeProcessPriority:"
                 " getpriority: errno: " << errno);
      return -errno;
   }

   // Reference priority
   int refpriority = priority + dp;

   // Chaneg the priority
   if (setpriority(PRIO_PROCESS, pid, refpriority) != 0) {
      TRACE(XERR, "ChangeProcessPriority:"
                 " setpriority: errno: " << errno);
      return ((errno != 0) ? -errno : -1);
   }

   // Check that it worked out
   errno = 0;
   if ((priority = getpriority(PRIO_PROCESS, pid)) == -1 && errno != 0) {
      TRACE(XERR, "ChangeProcessPriority:"
                 " getpriority: errno: " << errno);
      return -errno;
   }
   if (priority != refpriority) {
      TRACE(XERR, "ChangeProcessPriority:"
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

   TRACE(ACT,"SetSrvProtVers: forking test and protocol retrieval");

   // Make sure the application path has been defined
   if (!fgPrgmSrv|| strlen(fgPrgmSrv) <= 0) {
      XPDERR("SetSrvProtVers: "
            " path to PROOF server application undefined - exit");
      return -1;
   }

   // Make sure the temporary directory has been defined
   if (!fgTMPdir || strlen(fgTMPdir) <= 0) {
      XPDERR("SetSrvProtVers:"
            " path to temporary directory undefined - exit");
      return -1;
   }

   // Make sure the temporary directory has been defined
   if (!fgROOTsys || strlen(fgROOTsys) <= 0) {
      XPDERR("SetSrvProtVers: ROOTSYS undefined - exit");
      return -1;
   }

   // Pipe to communicate the protocol number
   int fp[2];
   if (pipe(fp) != 0) {
      XPDERR("SetSrvProtVers: unable to generate pipe for"
            " PROOT protocol number communication");
      return -1;
   }

   // Fork a test agent process to handle this session
   TRACE(FORK,"Forking external proofsrv");
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
            MERROR(MHEAD, "SetSrvProtVers: could not get info for user-id: "<<geteuid());
            exit(1);
         }

         // acquire permanently target user privileges
         if (XrdSysPriv::ChangePerm((uid_t)ui.fUid, (gid_t)ui.fGid) != 0) {
            MERROR(MHEAD, "SetSrvProtVers: can't acquire "<< ui.fUser <<" identity");
            exit(1);
         }

      }

      // Run the program
      execv(fgPrgmSrv, argvv);

      // We should not be here!!!
      MERROR(MHEAD, "SetSrvProtVers: returned from execv: bad, bad sign !!!");
      exit(1);
   }

   // parent process
   if (pid < 0) {
      XPDERR("SetSrvProtVers: forking failed - exit");
      close(fp[0]);
      close(fp[1]);
      return -1;
   }

   // now we wait for the callback to be (successfully) established
   TRACE(FORK, "SetSrvProtVers: test server launched: wait for protocol ");


   // Read protocol
   int proto = -1;
   struct pollfd fds_r;
   fds_r.fd = fp[0];
   fds_r.events = POLLIN;
   int pollRet = 0;
   int ntry = (fgInternalWait < 2) ? 1 : (int) (fgInternalWait / 2 + 1);
   while (pollRet == 0 && ntry--) {
      while ((pollRet = poll(&fds_r, 1, 2000)) < 0 &&
             (errno == EINTR)) { }
      if (pollRet == 0)
         TRACE(DBG,"SetSrvProtVers: "
                   "receiving PROOF server protocol number: waiting 2 s ...");
   }
   if (pollRet > 0) {
      if (read(fp[0], &proto, sizeof(proto)) != sizeof(proto)) {
         XPDERR("SetSrvProtVers: "
               " problems receiving PROOF server protocol number");
         return -1;
      }
   } else {
      if (pollRet == 0) {
         XPDERR("SetSrvProtVers: "
               " timed-out receiving PROOF server protocol number");
      } else {
         XPDERR("SetSrvProtVers: "
               " failed to receive PROOF server protocol number");
      }
      return -1;
   }

   // Record protocol
   fgSrvProtVers = (kXR_int16) ntohl(proto);

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

   MTRACE(ACT, MHEAD, "AssertDir: enter");

   if (!path || strlen(path) <= 0)
      return -1;

   struct stat st;
   if (stat(path,&st) != 0) {
      if (errno == ENOENT) {
         if (mkdir(path, 0755) != 0) {
            MERROR(MHEAD, "AssertDir: unable to create dir: "<<path<<
                          " (errno: "<<errno<<")");
            return -1;
         }
         if (stat(path,&st) != 0) {
            MERROR(MHEAD, "AssertDir: unable to stat dir: "<<path<<
                          " (errno: "<<errno<<")");
            return -1;
         }
      } else {
         // Failure: stop
         MERROR(MHEAD, "AssertDir: unable to stat dir: "<<path<<
                       " (errno: "<<errno<<")");
         return -1;
      }
   }

   // Make sure the ownership is right
   if ((int) st.st_uid != ui.fUid || (int) st.st_gid != ui.fGid) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (!pGuard.Valid()) {
         MERROR(MHEAD, "AsserDir: could not get privileges");
         return -1;
      }

      // Set ownership of the path to the client
      if (chown(path, ui.fUid, ui.fGid) == -1) {
         MERROR(MHEAD, "AssertDir: cannot set user ownership"
                       " on path (errno: "<<errno<<")");
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

   MTRACE(ACT, MHEAD, "SymLink: enter");

   if (!path || strlen(path) <= 0 || !link || strlen(link) <= 0)
      return -1;

   // Remove existing link, if any
   if (unlink(link) != 0 && errno != ENOENT) {
      MERROR(MHEAD, "SymLink: problems unlinking existing symlink "<< link<<
                    " (errno: "<<errno<<")");
      return -1;
   }
   if (symlink(path, link) != 0) {
      MERROR(MHEAD, "SymLink: problems creating symlink " << link<<
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

   MTRACE(ACT, MHEAD, "LoadSecurity: enter");

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

      // Default 1093
      int port = (pi && pi->Port > 0) ? pi->Port : 1093;

      // Done
      MPRINT(MHEAD, "XrdgetProtocolPort: listening on port: "<< port <<
                    " ("<<pi<<", "<<(pi ? pi->Port : -1)<<")");
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
   //   getenv(<ENVVAR>)      if it begins with $<ENVVAR>)
   // The returned array of chars is the result of reallocation
   // of the input one.
   // If something is inconsistent, for example <ENVVAR> does not
   // exists, the original string is untouched

   // Make sure there soething to expand
   if (!p || strlen(p) <= 0 || p[0] == '/')
      return p;

   char *po = p;

   // Relative to the environment variable
   if (p[0] == '$') {
      // Resolve env
      XrdOucString env(&p[1]);
      int isl = env.find('/');
      env.erase(isl);
      char *p1 = (isl > 0) ? (char *)(p + isl + 2) : 0;
      if (getenv(env.c_str())) {
         int lenv = strlen(getenv(env.c_str()));
         int lp1 = p1 ? strlen(p1) : 0;
         po = (char *) malloc(lp1 + lenv + 2);
         if (po) {
            memcpy(po, getenv(env.c_str()), lenv);
            if (p1) {
               memcpy(po+lenv+1, p1, lp1);
               po[lenv] = '/';
            }
            po[lp1 + lenv + 1] = 0;
            free(p);
         } else
            po = p;
      }
      return po;
   }

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
int XrdProofdProtocol::ResolveKeywords(XrdOucString &s, XrdProofClient *pcl)
{
   // Resolve special keywords in 's' for client 'pcl'. Recognized keywords
   //     <workdir>          fgWorkDir;
   //     <user>             username
   // Return the number of keywords resolved.

   int nk = 0;

   XPDPRT("ResolveKeywords: enter: "<<s<<" - fgWorkDir: "<<fgWorkDir);

   // Parse <workdir>
   if (s.replace("<workdir>",(const char *)fgWorkDir))
      nk++;

   XPDPRT("ResolveKeywords: after <workdir>: "<<s);

   // Parse <user>
   if (pcl)
      if (s.replace("<user>", pcl->ID()))
         nk++;

   XPDPRT("ResolveKeywords: exit: "<<s);

   // We are done
   return nk;
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
   fgEDest.logger(&gMainLogger);
   XrdProofdTrace = new XrdOucTrace(&fgEDest);
   fgSched        = pi->Sched;
   fgBPool        = pi->BPool;
   fgReadWait     = pi->readWait;
   fgPort         = pi->Port;

   // Debug flag
   TRACESET(XERR, 1);
   if (pi->DebugON)
      XrdProofdTrace->What |= (TRACE_REQ | TRACE_LOGIN | TRACE_FORK);

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

   // Number of CPUs
   if (fgNumLocalWrks < 0 && (fgNumLocalWrks = XrdProofdProtocol::GetNumCPUs()) <= 0)
      fgEDest.Say(0, "Configure:"
                     " problems resolving the number of CPUs in the local machine");
   mp = fgNumLocalWrks;
   fgEDest.Say(0, "Configure: default number of workers for local sessions: ", mp.c_str());

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
      if (!(fgCIA = XrdProofdProtocol::LoadSecurity(fgSecLib, pi->ConfigFN))) {
         fgEDest.Emsg(0, "Configure: unable to load security system.");
         return 0;
      }
      fgEDest.Emsg(0, "Configure: security library loaded");
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

   // Notify image
   if (!fgImage)
      // Use the local host name
      fgImage = strdup(fgLocalHost.c_str());
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
               SafeDelArray(fgPROOFcfg);
               // Enable user config files
               fgWorkerUsrCfg = 1;
               // Fill in a default file based on the number of CPUs or on the
               // number of local sessions defined via xpd.localwrks
               if (fgNumLocalWrks > 0)
                  if (CreateDefaultPROOFcfg() != 0)
                     fgEDest.Say(0, "Configure: unable to create the default worker list");
            }
         }
         fgEDest.Say(0, "Configure: PROOF config file: ",
                         (fgPROOFcfg ? (const char *)fgPROOFcfg : "none"));
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

   // Notify controlled mode, if such
   if (fgOperationMode == kXPD_OpModeControlled) {
      fgAllowedUsers += ',';
      fgAllowedUsers += fgSuperUsers;
      mp = "Configure: running in controlled access mode: users allowed: ";
      mp += fgAllowedUsers;
      fgEDest.Say(0, mp.c_str());
   }

   // Set base environment common to all
   SetProofServEnv();

   // Test forking and get PROOF server protocol version
   if (SetSrvProtVers() < 0) {
      fgEDest.Say(0, "Configure: forking test failed");
      return 0;
   }
   mp = "Configure: PROOF server protocol number: ";
   mp += (int) fgSrvProtVers;
   fgEDest.Say(0, mp.c_str());

   // Schedule protocol object cleanup
   fgProtStack.Set(pi->Sched, XrdProofdTrace, TRACE_MEM);
   fgProtStack.Set(pi->ConnOptn, pi->ConnLife);

   // Initialize the request ID generation object
   XrdProofdReqID = new XrdOucReqID((int)fgPort, pi->myName,
                                    XrdNetDNS::IPAddr(pi->myAddr));

   // Start cron thread, if required
   if (fgCron == 1) {
      pthread_t tid;
      if (XrdOucThread::Run(&tid, XrdProofdCron, (void *)&fgCronFrequency, 0,
                                    "Proof cron thread") != 0) {
         fgEDest.Say(0, "Configure: could not start cron thread");
         return 0;
      }
      fgEDest.Say(0, "Configure: cron thread started");
   }

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
   TRACE(DBG, "CheckIf: <pattern>: " <<val);

   // Return number of chars matching
   return fgLocalHost.matches((const char *)val);
}

//______________________________________________________________________________
int XrdProofdProtocol::Config(const char *cfn)
{
   // Scan the config file

   TRACE(ACT, "Config: enter: file: " <<cfn);

   XrdOucStream Config(&fgEDest, getenv("XRDINSTANCE"));
   char *var;
   int cfgFD, NoGo = 0;
   int nmRole = -1, nmRootSys = -1, nmTmp = -1, nmInternalWait = -1,
       nmMaxSessions = -1, nmMaxOldLogs = -1, nmImage = -1, nmWorkDir = -1,
       nmPoolUrl = -1, nmNamespace = -1, nmSuperUsers = -1, nmLocalWrks = -1;

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
            TRACE(DBG, "Config: " <<mess);
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

            } else if (!strcmp("trace",var)) {

               // Specifies tracing options. Valid keywords are:
               //   req            trace protocol requests             [on]*
               //   login          trace details about login requests  [on]*
               //   act            trace internal actions              [off]
               //   rsp            trace server replies                [off]
               //   fork           trace proofserv forks               [on]*
               //   dbg            trace details about actions         [off]
               //   hdbg           trace more details about actions    [off]
               //   err            trace errors                        [on]
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
                  } else if (!strcmp(val,"all")) {
                     // Everything
                     XrdProofdTrace->What = TRACE_ALL;
                  }
                  // Next
                  val = Config.GetToken();
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
               } else if (!strcmp("localwrks",var)) {
                  // Number of workers for local sessions
                  XPDSETINT(nm, nmLocalWrks, fgNumLocalWrks, tval);
               } else if (!strcmp("maxsessions",var)) {
                  // Max number of sessions per user
                  XPDSETINT(nm, nmMaxSessions, fgMaxSessions, tval);
               } else if (!strcmp("maxoldlogs",var)) {
                  // Max number of sessions per user
                  XPDSETINT(nm, nmMaxOldLogs, fgMaxOldLogs, tval);
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
                  // Local namespace
                  XPDSETSTRING(nm, nmNamespace, fgNamespace, tval);
               } else if (!strcmp("superusers",var)) {
                  // Superusers
                  XPDSETSTRING(nm, nmSuperUsers, fgSuperUsers, tval);
               } else if (!strcmp("allowedusers",var)) {
                  // Users allowed to use the cluster
                  fgAllowedUsers = tval;
                  fgOperationMode = kXPD_OpModeControlled;
               } else if (!strcmp("putenv",var)) {
                  // Env variable to exported to 'proofserv'
                  if (fgProofServEnvs.length() > 0)
                     fgProofServEnvs += ',';
                  fgProofServEnvs += tval;
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
   TRACEP(REQ, "Process: enter: instance: " << this);

   // Read the next request header
   if ((rc = GetData("request", (char *)&fRequest, sizeof(fRequest))) != 0)
      return rc;
   TRACEP(DBG, "Process: after GetData: rc: " << rc);

   // Deserialize the data
   fRequest.header.requestid = ntohs(fRequest.header.requestid);
   fRequest.header.dlen      = ntohl(fRequest.header.dlen);

   // The stream ID for the reply
   { XrdOucMutexHelper mh(fResponse.fMutex);
      fResponse.Set(fRequest.header.streamid);
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

   XPDPRT("Process2: this: "<<this<<", auth: "<<fAuthProt);

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
   TRACEP(REQ,"Recycle: enter: instance: " <<this<<", type: "<<srvtype[fSrvType+1]);

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
         for (ic = 0; ic < (int) pmgr->Clients()->size(); ic++) {
            if (this == pmgr->Clients()->at(ic))
               pmgr->ResetClient(ic);
            else if (pmgr->Clients()->at(ic) && pmgr->Clients()->at(ic)->fTopClient)
               nc++;
         }

         // If top master ...
         if (fSrvType == kXPD_TopMaster) {
            // Loop over servers sessions associated to this client and update
            // their attached client vectors
            if (pmgr->ProofServs()->size() > 0) {
               XrdProofServProxy *psrv = 0;
               int is = 0;
               for (is = 0; is < (int) pmgr->ProofServs()->size(); is++) {
                  if ((psrv = pmgr->ProofServs()->at(is))) {
                     // Release CIDs in attached sessions: loop over attached clients
                     XrdClientID *cid = 0;
                     int ic = 0;
                     for (ic = 0; ic < (int) psrv->Clients()->size(); ic++) {
                        if ((cid = psrv->Clients()->at(ic))) {
                           if (cid->fP == this)
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
                     if (SetShutdownTimer(psrv) != 0) {
                        // Just notify locally: link is closed!
                        XrdOucString msg("Recycle: could not send shutdown info to proofsrv");
                        TRACEP(XERR, msg.c_str());
                     }
                     // Set in shutdown state
                     psrv->SetStatus(kXPD_shutdown);
                  }
               }
            }

         } else {

            // We cannot continue if the top master went away: we cleanup the session
            if (pmgr->ProofServs()->size() > 0) {
               XrdProofServProxy *psrv = 0;
               int is = 0;
               for (is = 0; is < (int) pmgr->ProofServs()->size(); is++) {
                  if ((psrv = pmgr->ProofServs()->at(is)) && psrv->IsValid()
                      && psrv->SrvType() != kXPD_TopMaster) {

                     TRACEP(HDBG, "Recycle: found: " << psrv << " (t:"<<psrv->SrvType() <<
                                  ",nc:"<<psrv->Clients()->size()<<")");

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
         if (pmgr->ProofServs()->size() > 0) {
            XrdProofServProxy *psrv = 0;
            int is = 0;
            for (is = 0; is < (int) pmgr->ProofServs()->size(); is++) {
               if ((psrv = pmgr->ProofServs()->at(is)) && (psrv->Link() == fLink)) {

               TRACEP(HDBG, "Recycle: found: " << psrv << " (v:" << psrv->IsValid() <<
                            ",t:"<<psrv->SrvType() << ",nc:"<<psrv->Clients()->size()<<")");

                  XrdOucMutexHelper xpmh(psrv->Mutex());

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
                        if ((p = psrv->Clients()->at(ic)->fP)) {
                           unsigned short sid;
                           p->fResponse.GetSID(sid);
                           p->fResponse.Set(psrv->Clients()->at(ic)->fSid);
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

   TRACE(ACT, "FilterSecConfig: enter");

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
   static time_t lastMod = 0;

   TRACE(ACT, "ReadPROOFcfg: enter: saved time of last modification: " << lastMod);

   // Check inputs
   if (!fgPROOFcfg)
      return -1;

   // Get the modification time
   struct stat st;
   if (stat(fgPROOFcfg, &st) != 0)
      return -1;
   TRACE(DBG, "ReadPROOFcfg: enter: time of last modification: " << st.st_mtime);

   // File should be loaded only once
   if (st.st_mtime <= lastMod)
      return 0;

   // Reserve some space or clear the list
   int allocsz = 50;
   if (lastMod <= 0) {
      fgWorkers.reserve(allocsz);
   } else {
      fgWorkers.clear();
   }

   // Save the modification time
   lastMod = st.st_mtime;

   // Open the defined path.
   FILE *fin = 0;
   if (!(fin = fopen(fgPROOFcfg, "r")))
      return -1;

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

      TRACE(DBG, "ReadPROOFcfg: found line: " << lin);

      const char *pfx[2] = { "master", "node" };
      if (!strncmp(lin, pfx[0], strlen(pfx[0])) ||
          !strncmp(lin, pfx[1], strlen(pfx[1]))) {
         // Init a master instance
         XrdProofWorker *pw = new XrdProofWorker(lin);
         if (pw->fHost == "localhost" ||
             pw->Matches(fgLocalHost.c_str())) {
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
int XrdProofdProtocol::CreateDefaultPROOFcfg()
{
   // Fill-in fgWorkerList for a localhost based on the number of
   // workers fgNumLocalWrks.

   TRACE(ACT, "CreateDefaultPROOFcfg: enter");

   // Reserve some space or clear the list
   int allocsz = 50;
   if (fgWorkers.size() <= 0) {
      fgWorkers.reserve(allocsz);
   } else {
      fgWorkers.clear();
   }

   // Create a default master line
   XrdOucString mm("master ",128);
   mm += fgImage; mm += " image="; mm += fgImage;
   fgWorkers.push_back(new XrdProofWorker(mm.c_str()));
   TRACE(DBG, "CreateDefaultPROOFcfg: added line: " << mm);

   // Create 'localhost' lines for each worker
   int nw = 0;
   int nwrk = fgNumLocalWrks;
   while (nwrk--) {
      mm = "worker localhost port=";
      mm += fgPort;
      fgWorkers.push_back(new XrdProofWorker(mm.c_str()));
      nw++;
      TRACE(DBG, "CreateDefaultPROOFcfg: added line: " << mm);
   }

   // If not defined, set max sessions to worker list size
   if (fgMaxSessions < 0)
      fgMaxSessions = fgWorkers.size() - 1;

   TRACE(ACT, "CreateDefaultPROOFcfg: done ("<<nw<<")");

   // We are done
   return ((nw == 0) ? -1 : 0);
}

//__________________________________________________________________________
int XrdProofdProtocol::GetWorkers(XrdOucString &lw, XrdProofServProxy *xps)
{
   // Get a list of workers from the available resource broker
   int rc = 0;

   TRACE(ACT, "GetWorkers: enter");

   // Static
   if (fgResourceType == kRTStatic) {

      // Read the configuration file
      if (fgPROOFcfg && ReadPROOFcfg() != 0) {
         TRACE(XERR, "GetWorkers: unable to read the configuration file");
         return -1;
      }

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
int XrdProofClient::GetFreeServID()
{
   // Get next free server ID. If none is found, increase the vector size
   // and get the first new one

   TRACE(ACT,"GetFreeServID: enter");

   XrdOucMutexHelper mh(fMutex);

   TRACE(DBG,"GetFreeServID: size = "<<fProofServs.size()<<
              "; capacity = "<<fProofServs.capacity());
   int ic = 0;
   // Search for free places in the existing vector
   for (ic = 0; ic < (int)fProofServs.size() ; ic++) {
      if (fProofServs[ic] && !(fProofServs[ic]->IsValid())) {
         fProofServs[ic]->SetValid();
         return ic;
      }
   }

   // We may need to resize (double it)
   if (ic >= (int)fProofServs.capacity()) {
      int newsz = 2 * fProofServs.capacity();
      fProofServs.reserve(newsz);
   }

   // Allocate new element
   fProofServs.push_back(new XrdProofServProxy());

   TRACE(DBG,"GetFreeServID: size = "<<fProofServs.size()<<
              "; new capacity = "<<fProofServs.capacity()<<"; ic = "<<ic);

   // We are done
   return ic;
}

//______________________________________________________________________________
void XrdProofClient::EraseServer(int psid)
{
   // Erase server with id psid from the list

   TRACE(ACT,"EraseServer: enter: psid: " << psid);

   XrdOucMutexHelper mh(fMutex);

   XrdProofServProxy *xps = 0;
   std::vector<XrdProofServProxy *>::iterator ip;
   for (ip = fProofServs.begin(); ip != fProofServs.end(); ++ip) {
      xps = *ip;
      if (xps && xps->Match(psid)) {
         fProofServs.erase(ip);
         break;
      }
   }
}

//______________________________________________________________________________
int XrdProofdProtocol::Login()
{
   // Process a login request

   int rc = 1;

   TRACEP(REQ, "Login: enter");

   // If this server is explicitely required to be a worker node or a
   // submaster, check whether the requesting host is allowed to connect
   if (fRequest.login.role[0] != 'i' &&
       fgSrvType == kXPD_WorkerServer || fgSrvType == kXPD_MasterServer) {
      if (!CheckMaster(fLink->Host())) {
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
      if ((p = (char *) strstr(fgSuperUsers, fClientID))) {
         if (p == fgSuperUsers || (p > fgSuperUsers && *(p-1) == ',')) {
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
   XrdOucString uname;

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

   // No 'root' logins
   if (uname.length() == 4 && uname == "root") {
      TRACEP(XERR,"Login: 'root' logins not accepted ");
      fResponse.Send(kXR_InvalidRequest,"Login: 'root' logins not accepted");
      return rc;
   }

   // Here we check if the user is known locally.
   // If not, we fail for now.
   // In the future we may try to get a temporary account
   if (GetUserInfo(uname.c_str(), fUI) != 0) {
      XrdOucString emsg("Login: unknown ClientID: ");
      emsg += uname;
      TRACEP(XERR, emsg.c_str());
      fResponse.Send(kXR_InvalidRequest, emsg.c_str());
      return rc;
   }

   // If we are in controlled mode we have to check if the user in the
   // authorized list; otherwise we fail. Privileged users are always
   // allowed to connect.
   if (fgOperationMode == kXPD_OpModeControlled) {
      bool notok = 1;
      XrdOucString us;
      int from = 0;
      while ((from = fgAllowedUsers.tokenize(us, from, ',')) != -1) {
         if (us == uname) {
            notok = 0;
            break;
         }
      }
      if (notok) {
         XrdOucString emsg("Login: controlled operations:"
                           " user not currently authorized to log in: ");
         emsg += uname;
         TRACEP(XERR, emsg.c_str());
         fResponse.Send(kXR_InvalidRequest, emsg.c_str());
         return rc;
      }
   }

   // Establish the ID for this link
   fLink->setID(uname.c_str(), pid);
   fCapVer = fRequest.login.capver[0];

   // Establish the ID for this client
   fClientID = new char[uname.length()+4];
   strcpy(fClientID, uname.c_str());
   TRACEP(LOGIN,"Login: ClientID =" << fClientID);

   // Assert the workdir directory ...
   fUI.fWorkDir = fUI.fHomeDir;
   if (fgWorkDir) {
      // The user directory path will be <workdir>/<user>
      fUI.fWorkDir = fgWorkDir;
      if (!fUI.fWorkDir.endswith('/'))
         fUI.fWorkDir += "/";
      fUI.fWorkDir += fClientID;
   } else {
      // Default: $HOME/proof
      if (!fUI.fWorkDir.endswith('/'))
         fUI.fWorkDir += "/";
      fUI.fWorkDir += "proof";
   }
   // Make sure the directory exists
   if (AssertDir(fUI.fWorkDir.c_str(), fUI) == -1) {
      XrdOucString emsg("Login: unable to create work dir: ");
      emsg += fUI.fWorkDir;
      TRACEP(XERR, emsg);
      fResponse.Send(kXP_ServerError, emsg.c_str());
      return rc;
   }

   // If strong authentication is required ...
   if (fgCIA) {
      // ... make sure that the directory for credentials exists in the sandbox ...
      XrdOucString credsdir = fUI.fWorkDir;
      credsdir += "/.creds";
      if (AssertDir(credsdir.c_str(), fUI) == -1) {
         XrdOucString emsg("Login: unable to create credential dir: ");
         emsg += credsdir;
         TRACEP(XERR, emsg);
         fResponse.Send(kXP_ServerError, emsg.c_str());
         return rc;
      }
   }

   // Find out the server type: 'i', internal, means this is a proofsrv calling back.
   // For the time being authentication is required for clients only.
   bool needauth = 0;
   switch (fRequest.login.role[0]) {
   case 'A':
      fSrvType = kXPD_Admin;
      fResponse.Set(" : admin ");
      break;
   case 'i':
      fSrvType = kXPD_Internal;
      fResponse.Set(" : internal ");
      break;
   case 'M':
      if (fgSrvType == kXPD_AnyServer || fgSrvType == kXPD_TopMaster) {
         fTopClient = 1;
         fSrvType = kXPD_TopMaster;
         needauth = 1;
         fResponse.Set(" : mst->clnt ");
      } else {
         TRACEP(XERR,"Login: top master mode not allowed - ignoring request");
         fResponse.Send(kXR_InvalidRequest,
                        "Server not allowed to be top master - ignoring request");
         return rc;
      }
      break;
   case 'm':
      if (fgSrvType == kXPD_AnyServer || fgSrvType == kXPD_MasterServer) {
         fSrvType = kXPD_MasterServer;
         needauth = 1;
         fResponse.Set(" : mst->mst ");
      } else {
         TRACEP(XERR,"Login: submaster mode not allowed - ignoring request");
         fResponse.Send(kXR_InvalidRequest,
                        "Server not allowed to be submaster - ignoring request");
         return rc;
      }
      break;
   case 's':
      if (fgSrvType == kXPD_AnyServer || fgSrvType == kXPD_WorkerServer) {
         fSrvType = kXPD_WorkerServer;
         needauth = 1;
         fResponse.Set(" : wrk->mst ");
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
               TRACEP(LOGIN,"Login: privileged user ");
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

   TRACEP(REQ,"MapClient: enter");

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
      TRACEP(DBG,"MapClient: proofsrv callback for session: " <<psid);
   } else {
      // Get PROOF version run by client
      memcpy(&clientvers, (const void *)&(fRequest.login.reserved[0]), 2);
      TRACEP(DBG,"MapClient: PROOF version run by client: " <<clientvers);
   }

   // Now search for an existing manager session for this ClientID
   XrdProofClient *pmgr = 0;
   TRACEP(DBG,"MapClient: # of clients: "<<fgProofClients.size());
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

   // Map the existing session, if found
   if (pmgr) {
      // Save as reference proof mgr
      fPClient = pmgr;
      TRACEP(DBG,"MapClient: matching client: "<<pmgr);

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
            return rc;
         } else {
            // Set the protocol version
            psrv->SetProtVer(protver);
            // Assign this link to it
            psrv->SetLink(fLink);
            psrv->ProofSrv()->Set(fLink);
            psrv->ProofSrv()->Set(fRequest.header.streamid);
            // Set Trace ID
            XrdOucString tid(" : xrd->");
            tid += psrv->Ordinal();
            tid += " ";
            psrv->ProofSrv()->Set(tid.c_str());
            TRACEP(DBG,"MapClient: proofsrv callback:"
                       " link assigned to target session "<<psid);
         }
      } else {

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
                    psrv->Status() == kXPD_shutdown) {
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
         TRACEP(XERR, "MapClient: proofsrv callback:"
                     " no manager to attach to: protocol error");
         return -1;
      }

      // This part may be not thread safe
      {  XrdOucMutexHelper mtxh(&fgXPDMutex);

         // Make sure that no zombie proofserv is around
         CleanupProofServ(0, fClientID);
         // No existing session: create a new one
         pmgr = new XrdProofClient(this, clientvers, fUI.fWorkDir.c_str());
      }

      // No existing session: create a new one
      if (pmgr && (pmgr->CreateUNIXSock(&fgEDest, fgTMPdir) == 0)) {

         TRACEP(DBG,"MapClient: NEW client: "<<pmgr<<", "<<pmgr->ID());

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

         // Check if old sessions are still flagged as active
         XrdOucString tobemv;

         // Get list of session working dirs flagged as active,
         // and check if they have to be deactivated
         std::list<XrdOucString *> sactlst;
         if (GetSessionDirs(pmgr, 1, &sactlst) == 0) {
            std::list<XrdOucString *>::iterator i;
            for (i = sactlst.begin(); i != sactlst.end(); ++i) {
               char *p = (char *) strrchr((*i)->c_str(), '-');
               if (p) {
                  int pid = strtol(p+1, 0, 10);
                  if (!VerifyProcessByID(pid)) {
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
                  if (!VerifyProcessByID(pid)) {
                     tobemv += ln;
                     tobemv += '|';
                  }
               }
            }
            fclose(f);
         }

         TRACEP(DBG,"MapClient: client "<<pmgr<<" added to the list (ref sid: "<< sid<<")");

         XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
         if (!pGuard.Valid()) {
            TRACEP(XERR, "MapClient: could not get privileges");
            return -1;
         }

         // Mv inactive sessions, if needed
         if (tobemv.length() > 0) {
            char del = '|';
            XrdOucString tag;
            int from = 0;
            while ((from = tobemv.tokenize(tag, from, del)) != -1) {
               if (XrdProofdProtocol::MvOldSession(fPClient, tag.c_str(), fgMaxOldLogs) == -1)
                  TRACEP(REQ, "MapClient: problems recording session as old in sandbox");
            }
         }

         // Set ownership of the socket file to the client
         if (chown(pmgr->UNIXSockPath(), fUI.fUid, fUI.fGid) == -1) {
            TRACEP(XERR, "MapClient: cannot set user ownership"
                               " on UNIX socket (errno: "<<errno<<")");
            return -1;
         }

      } else {
         SafeDelete(pmgr);
         TRACEP(DBG,"MapClient: cannot instantiate XrdProofClient");
         fResponse.Send(kXP_ServerError,
                        "MapClient: cannot instantiate XrdProofClient");
         return rc;
      }
   }

   if (!proofsrv) {
      TRACEP(DBG,"MapClient: fCID: "<<fCID<<", size: "<<fPClient->Clients()->size()<<
                 ", capacity: "<<fPClient->Clients()->capacity());
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

   TRACEP(REQ,"Auth: enter");

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
         TRACEP(XERR,"Auth: user authentication failed; "<<eText);
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
   eText = eMsg.getErrText(rc);
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
   TRACEP(ACT, "GetData: dtype: "<<(dtype ? dtype : " - ")<<", blen: "<<blen);

   rlen = fLink->Recv(buff, blen, fgReadWait);

   if (rlen  < 0)
      if (rlen != -ENOMSG) {
         TRACEP(XERR, "GetData: link read error");
         return fLink->setEtext("link read error");
      } else {
         TRACEP(DBG, "GetData: connection closed by peer");
         return -1;
      }
   if (rlen < blen) {
      fBuff = buff+rlen; fBlen = blen-rlen;
      TRACEP(XERR, "GetData: " << dtype <<
                  " timeout; read " <<rlen <<" of " <<blen <<" bytes");
      return 1;
   }
   TRACEP(DBG, "GetData: rlen: "<<rlen);

   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::Attach()
{
   // Handle a request to attach to an existing session

   int psid = -1, rc = 1;

   // Unmarshall the data
   psid = ntohl(fRequest.proof.sid);
   TRACEP(REQ, "Attach: psid: "<<psid<<", fCID = "<<fCID);

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
   csid->fP = this;
   csid->fSid = sid;

   // Take parentship, if orphalin
   if (!(xps->Parent()))
      xps->SetParent(csid);

   // Notify to user
   if (fSrvType == kXPD_TopMaster) {
      // Send also back the data pool url
      XrdOucString dpu = fgPoolURL;
      if (!dpu.endswith('/'))
         dpu += '/';
      dpu += fgNamespace;
      fResponse.Send(psid, fgSrvProtVers, (kXR_int16)XPROOFD_VERSBIN,
                     (void *) dpu.c_str(), dpu.length());
   } else
      fResponse.Send(psid, fgSrvProtVers, (kXR_int16)XPROOFD_VERSBIN);

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
   if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
       !(xps = fPClient->ProofServs()->at(psid))) {
      TRACEP(XERR, "Detach: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
      return rc;
   }
   TRACEP(DBG, "Detach: xps: "<<xps<<", status: "<< xps->Status()<<
               ", # clients: "<< xps->Clients()->size());

   XrdOucMutexHelper xpmh(xps->Mutex());

   // Remove this from the list of clients
   std::vector<XrdClientID *>::iterator i;
   for (i = xps->Clients()->begin(); i != xps->Clients()->end(); ++i) {
      if (*i) {
         if ((*i)->fP == this) {
            delete (*i);
            xps->Clients()->erase(i);
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

   XrdOucMutexHelper mh(fPClient->Mutex());

   // Unmarshall the data
   psid = ntohl(fRequest.proof.sid);
   TRACEP(REQ, "Destroy: psid: "<<psid);

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

         TRACEP(DBG, "Destroy: xps: "<<xps<<", status: "<< xps->Status()<<", pid: "<<xps->SrvID());

         {  XrdOucMutexHelper xpmh(xps->Mutex());

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
                     if ((p = xps->Clients()->at(ic)->fP) &&
                         (p != this) && p->fTopClient) {
                        unsigned short sid;
                        p->fResponse.GetSID(sid);
                        p->fResponse.Set(xps->Clients()->at(ic)->fSid);
                        p->fResponse.Send(kXR_attn, kXPD_srvmsg, msg, len);
                        p->fResponse.Set(sid);
                     }
                  }
               }
            }

            // Send a terminate signal to the proofserv
            if (TerminateProofServ(xps) != 0)
               if (KillProofServ(xps,1) != 0) {
                  TRACEP(XERR, "Destroy: problems terminating request to proofsrv");
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
int XrdProofdProtocol::SaveAFSkey(XrdSecCredentials *c, const char *dir)
{
   // Save the AFS key, if any, for usage in proofserv in file 'dir'/.afs .
   // Return 0 on success, -1 on error.

   // Check file name
   if (!dir || strlen(dir) <= 0) {
      MTRACE(XERR, MHEAD, "SaveAFSkey: dir name undefined");
      return -1;
   }

   // Check credentials
   if (!c) {
      MTRACE(XERR, MHEAD, "SaveAFSkey: credentials undefined");
      return -1;
   }

   // Decode credentials
   int lout = 0;
   char *out = new char[c->size];
   if (XrdSutFromHex(c->buffer, out, lout) != 0) {
      MTRACE(XERR, MHEAD, "SaveAFSkey: problems unparsing hex string");
      delete [] out;
      return -1;
   }

   // Locate the key
   char *key = out + 5;
   if (strncmp(key, "afs:", 4)) {
      MTRACE(DBG, MHEAD, "SaveAFSkey: string does not contain an AFS key");
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
      MTRACE(XERR, MHEAD, "SaveAFSkey: problems creating file - errno: " << errno);
      delete [] out;
      return -1;
   }
   // Make sure it is protected
   if (fchmod(fd, 0600) != 0) {
      MTRACE(XERR, MHEAD, "SaveAFSkey: problems setting file permissions to 0600 - errno: " << errno);
      delete [] out;
      close(fd);
      return -1;
   }
   // Write out the key
   int rc = 0;
   int lkey = lout - 9;
   if (Write(fd, key, lkey) != lkey) {
      MTRACE(XERR, MHEAD, "SaveAFSkey: problems writing to file - errno: " << errno);
      rc = -1;
   }

   // Cleanup
   delete [] out;
   close(fd);
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::SetProofServEnv(XrdProofdProtocol *p,
                                       int psid, int loglevel, const char *cfg)
{
   // Set environment for proofserv

   char *ev = 0;

   MTRACE(REQ, MHEAD, "SetProofServEnv: enter: psid: "<<psid<<
                      ", log: "<<loglevel);

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

   // Make sure the principal client is defined
   if (!(p->fPClient)) {
      MERROR(MHEAD, "SetProofServEnv: principal client undefined - cannot continue");
      return -1;
   }

   // Session proxy
   XrdProofServProxy *xps = p->fPClient->ProofServs()->at(psid);
   if (!xps) {
      MERROR(MHEAD, "SetProofServEnv: unable to get instance of proofserv proxy");
      return -1;
   }

   // Work directory
   XrdOucString udir = p->fPClient->Workdir();
   MTRACE(DBG, MHEAD, "SetProofServEnv: working dir for "<<p->fClientID<<" is: "<<udir);

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
      xps->SetTag(stag);
   } else {
      logdir += "/";
      logdir += xps->Tag();
   }
   MTRACE(DBG, MHEAD, "SetProofServEnv: log dir "<<logdir);
   // Make sure the directory exists
   if (AssertDir(logdir.c_str(), p->fUI) == -1) {
      MERROR(MHEAD, "SetProofServEnv: unable to create log dir: "<<logdir);
      return -1;
   }
   // The session dir (sandbox) depends on the role
   XrdOucString sessdir = logdir;
   if (p->fSrvType == kXPD_WorkerServer)
      sessdir += "/worker-";
   else
      sessdir += "/master-";
   sessdir += xps->Ordinal();
   sessdir += "-";
   sessdir += stag;
   ev = new char[strlen("ROOTPROOFSESSDIR=")+sessdir.length()+2];
   sprintf(ev, "ROOTPROOFSESSDIR=%s", sessdir.c_str());
   putenv(ev);
   MTRACE(DBG, MHEAD, "SetProofServEnv: "<<ev);

   // Log level
   ev = new char[strlen("ROOTPROOFLOGLEVEL=")+5];
   sprintf(ev, "ROOTPROOFLOGLEVEL=%d", loglevel);
   putenv(ev);
   MTRACE(DBG, MHEAD, "SetProofServEnv: "<<ev);

   // Ordinal number
   ev = new char[strlen("ROOTPROOFORDINAL=")+strlen(xps->Ordinal())+2];
   sprintf(ev, "ROOTPROOFORDINAL=%s", xps->Ordinal());
   putenv(ev);
   MTRACE(DBG, MHEAD, "SetProofServEnv: "<<ev);

   // Create the env file
   MTRACE(DBG, MHEAD, "SetProofServEnv: creating env file");
   XrdOucString envfile = sessdir;
   envfile += ".env";
   FILE *fenv = fopen(envfile.c_str(), "w");
   if (!fenv) {
      MERROR(MHEAD, "SetProofServEnv: unable to open env file: "<<envfile);
      return -1;
   }
   MTRACE(DBG, MHEAD, "SetProofServEnv: environment file: "<< envfile);

   // Forwarded sec credentials, if any
   if (p->fAuthProt) {

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
               MTRACE(DBG, MHEAD, "SetProofServEnv: "<<ev);
            }
         }
      }

      // The credential buffer, if any
      XrdSecCredentials *creds = p->fAuthProt->getCredentials();
      if (creds) {
         int lev = strlen("XrdSecCREDS=")+creds->size;
         ev = new char[lev+1];
         strcpy(ev, "XrdSecCREDS=");
         memcpy(ev+strlen("XrdSecCREDS="), creds->buffer, creds->size);
         ev[lev] = 0;
         putenv(ev);
         MTRACE(DBG, MHEAD, "SetProofServEnv: XrdSecCREDS set");

         // If 'pwd', save AFS key, if any
         if (!strncmp(p->fAuthProt->Entity.prot, "pwd", 3)) {
            XrdOucString credsdir = udir;
            credsdir += "/.creds";
            // Make sure the directory exists
            if (!AssertDir(credsdir.c_str(), p->fUI)) {
               if (SaveAFSkey(creds, credsdir.c_str()) == 0) {
                  ev = new char[strlen("ROOTPROOFAFSCREDS=")+credsdir.length()+strlen("/.afs")+2];
                  sprintf(ev, "ROOTPROOFAFSCREDS=%s/.afs", credsdir.c_str());
                  putenv(ev);
                  fprintf(fenv, "ROOTPROOFAFSCREDS has been set\n");
                  MTRACE(DBG, MHEAD, "SetProofServEnv: " << ev);
               } else {
                  MTRACE(DBG, MHEAD, "SetProofServEnv: problems in saving AFS key");
               }
            } else {
               MERROR(MHEAD, "SetProofServEnv: unable to create creds dir: "<<credsdir);
               return -1;
            }
         }
      }
   }

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
   fprintf(fenv, "ROOTOPENSOCK=%s\n", p->fPClient->UNIXSockPath());

   // Entity
   fprintf(fenv, "ROOTENTITY=%s@%s\n", p->fClientID, p->fLink->Host());

   // Session ID
   fprintf(fenv, "ROOTSESSIONID=%d\n", psid);

   // Client ID
   fprintf(fenv, "ROOTCLIENTID=%d\n", p->fCID);

   // Client Protocol
   fprintf(fenv, "ROOTPROOFCLNTVERS=%d\n", p->fPClient->Version());

   // Ordinal number
   fprintf(fenv, "ROOTPROOFORDINAL=%s\n", xps->Ordinal());

   // Config file
   if (cfg && strlen(cfg) > 0)
      fprintf(fenv, "ROOTPROOFCFGFILE=%s\n", cfg);

   // Default number of workers
   fprintf(fenv, "ROOTPROOFMAXSESSIONS=%d\n", fgMaxSessions);

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
            ResolveKeywords(env, p->fPClient);
            // Set the env now
            ev = new char[env.length()+1];
            strncpy(ev, env.c_str(), env.length());
            ev[env.length()] = 0;
            putenv(ev);
            fprintf(fenv, "%s\n", ev);
            MTRACE(DBG, MHEAD, "SetProofServEnv: "<<ev);
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
            MTRACE(DBG, MHEAD, "SetProofServEnv: "<<ev);
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
      MTRACE(DBG, MHEAD, "SetProofServEnv: "<<ev);
   }

   // Close file
   fclose(fenv);

   // Create or Update symlink to last session
   MTRACE(DBG, MHEAD, "SetProofServEnv: creating symlink");
   XrdOucString syml = udir;
   if (p->fSrvType == kXPD_WorkerServer)
      syml += "/last-worker-session";
   else
      syml += "/last-master-session";
   if (SymLink(logdir.c_str(), syml.c_str()) != 0) {
      MERROR(MHEAD, "SetProofServEnv: problems creating symlink to "
                    " last session (errno: "<<errno<<")");
   }

   // We are done
   MTRACE(DBG, MHEAD, "SetProofServEnv: done");
   return 0;
}

//_________________________________________________________________________________
int XrdProofdProtocol::Create()
{
   // Handle a request to create a new session

   int psid = -1, rc = 1;

   TRACEP(REQ, "Create: enter");
   XrdOucMutexHelper mh(fPClient->Mutex());

   // Allocate next free server ID and fill in the basic stuff
   psid = fPClient->GetFreeServID();
   XrdProofServProxy *xps = fPClient->ProofServs()->at(psid);
   xps->SetClient((const char *)fClientID);
   xps->SetID(psid);
   xps->SetSrvType(fSrvType);

   // Unmarshall log level
   int loglevel = ntohl(fRequest.proof.int1);

   // Parse buffer
   char *buf = fArgp->buff;
   int   len = fRequest.proof.dlen;

   // Extract session tag
   XrdOucString tag(buf,len);
   tag.erase(tag.find('|'));
   xps->SetTag(tag.c_str());
   TRACEP(DBG, "Create: tag: "<<tag);

   // Extract ordinal number
   XrdOucString ord = "0";
   if ((fSrvType == kXPD_WorkerServer) || (fSrvType == kXPD_MasterServer)) {
      ord.assign(buf,0,len-1);
      int iord = ord.find("|ord:");
      ord.erase(0,iord+5);
      ord.erase(ord.find("|"));
   }
   xps->SetOrdinal(ord.c_str());

   // Extract config file, if any (for backward compatibility)
   XrdOucString cffile;
   cffile.assign(buf,0,len-1);
   int icf = cffile.find("|cf:");
   cffile.erase(0,icf+4);
   cffile.erase(cffile.find("|"));

   // Extract user envs, if any
   XrdOucString uenvs;
   uenvs.assign(buf,0,len-1);
   int ienv = uenvs.find("|envs:");
   uenvs.erase(0,ienv+6);
   uenvs.erase(uenvs.find("|"));
   xps->SetUserEnvs(uenvs.c_str());

   // Notify
   TRACEP(DBG, "Create: {ord,cfg,psid,cid,log}: {"<<ord<<","<<cffile<<","<<psid
                                                  <<","<<fCID<<","<<loglevel<<"}");
   if (uenvs.length() > 0)
      TRACEP(DBG, "Create: user envs: "<<uenvs);

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
   TRACE(FORK,"Forking external proofsrv: UNIX sock: "<<fPClient->UNIXSockPath());
   if (!(pid = fgSched->Fork("proofsrv"))) {

      int setupOK = 0;

      // Change logger instance to avoid deadlocks
      fgEDest.logger(&gForkLogger);

      // We set to the user environment
      if (SetUserEnvironment() != 0) {
         MERROR(MHEAD, "child::Create: SetUserEnvironment did not return OK - EXIT");
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
         MERROR(MHEAD, "child::Create: SetProofServEnv did not return OK - EXIT");
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
         char *buf = (char *) xps->Fileout();
         for (n = 0; n < lfout; n += ns) {
            if ((ns = write(fp[1], buf + n, lfout - n)) <= 0) {
               XPDPRT("Create: SetProofServEnv did not return OK - EXIT");
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

      MTRACE(DBG, MHEAD, "child::Create: fClientID: "<<fClientID<<
                         ", uid: "<<getuid()<<", euid:"<<geteuid());

      // Run the program
      execv(fgPrgmSrv, argvv);

      // We should not be here!!!
      MTRACE(XERR, MHEAD, "child::Create: returned from execv: bad, bad sign !!!");
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
   XrdOucString emsg;
   int setupOK = 0;
   if (read(fp[0], &setupOK, sizeof(setupOK)) == sizeof(setupOK)) {
   // now we wait for the callback to be (successfully) established

      if (setupOK > 0) {
         // Receive path of the log file
         int lfout = setupOK;
         char *buf = new char[lfout + 1];
         int n, nr = 0;
         for (n = 0; n < lfout; n += nr) {
            while ((nr = read(fp[0], buf + n, lfout - n)) == -1 && errno == EINTR)
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
            buf[lfout] = 0;
            xps->SetFileout(buf);
            // Set also the session tag
            XrdOucString stag(buf);
            stag.erase(stag.rfind('/'));
            stag.erase(0, stag.find("session-") + strlen("session-"));
            xps->SetTag(stag.c_str());
         }
         delete[] buf;
      } else {
         emsg += ": proofserv startup failed";
      }
   } else {
      emsg += ": problems receiving status-of-setup after forking";
   }

   // Cleanup
   close(fp[0]);
   close(fp[1]);

   // Notify to user
   if (setupOK > 0) {
      if (fSrvType == kXPD_TopMaster) {
         // Send also back the data pool url
         XrdOucString dpu = fgPoolURL;
         if (!dpu.endswith('/'))
            dpu += '/';
         dpu += fgNamespace;
         fResponse.Send(psid, fgSrvProtVers, (kXR_int16)XPROOFD_VERSBIN,
                       (void *) dpu.c_str(), dpu.length());
      } else
         fResponse.Send(psid, fgSrvProtVers, (kXR_int16)XPROOFD_VERSBIN);
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
            TRACEP(XERR, "Create: problems changing child process priority");
         } else {
            TRACEP(DBG, "Create: priority of the child process changed by "
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
   xps->SetParent(csid);

   TRACEP(DBG, "Create: ClientID: "<<(int *)(xps->Parent())<<" (sid: "<<sid<<")");

   // Record this session in the sandbox
   if (fSrvType != kXPD_Internal) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (pGuard.Valid()) {
         if (XrdProofdProtocol::AddNewSession(fPClient, xps->Tag()) == -1)
            TRACEP(REQ, "Create: problems recording session in sandbox");
      } else {
         TRACEP(REQ, "Create: could not get privileges to run AddNewSession");
      }
   }


   // Over
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::SendData(XrdProofdResponse *resp,
                                kXR_int32 sid, XrdSrvBuffer **buf)
{
   // Send data over the open link. Segmentation is done here, if required.

   int rc = 1;

   TRACEP(ACT, "SendData: enter: length: "<<fRequest.header.dlen<<" bytes ");

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

   TRACEP(ACT, "SendDataN: enter: length: "<<fRequest.header.dlen<<" bytes ");

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
         if ((csid = xps->Clients()->at(ic)) && csid->fP) {
            XrdProofdResponse& resp = csid->fP->fResponse;
            int rs = 0;
            {  XrdOucMutexHelper mhp(resp.fMutex);
               unsigned short sid;
               resp.GetSID(sid);
               TRACEP(HDBG, "SendDataN: INTERNAL: this sid: "<<sid<<
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
   static const char *copt[2] = {"INT", "EXT" };

   int rc = 1;

   XrdOucMutexHelper mh(fResponse.fMutex);

   // Unmarshall the data
   int psid = ntohl(fRequest.sendrcv.sid);
   int opt = ntohl(fRequest.sendrcv.opt);
   bool external = !(opt & kXPD_internal);

   TRACEP(REQ, "SendMsg: enter: "<< copt[(int)external] <<", psid: "<<psid);

   // Find server session
   XrdProofServProxy *xps = 0;
   if (!fPClient || !INRANGE(psid, fPClient->ProofServs()) ||
       !(xps = fPClient->ProofServs()->at(psid))) {
      TRACEP(XERR, "SendMsg: session ID not found");
      fResponse.Send(kXR_InvalidRequest,"session ID not found");
      return rc;
   }

   // Forward message as unsolicited
   int   len = fRequest.header.dlen;

   // Notify
   TRACEP(DBG, "SendMsg: xps: "<<xps<<", status: "<<xps->Status()<<
               ", cid: "<<fCID);

   if (external) {

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

      XrdSrvBuffer *savedBuf = 0;
      // Additional info about the message
      if (opt & kXPD_setidle) {
         TRACEP(DBG, "SendMsg: INT: setting proofserv in 'idle' state");
         if (xps->Status() != kXPD_shutdown)
            xps->SetStatus(kXPD_idle);
         // Clean start processing message, if any
         xps->DeleteStartMsg();
      } else if (opt & kXPD_querynum) {
         TRACEP(DBG, "SendMsg: INT: got message with query number");
         // Save query num message for later clients
         savedBuf = xps->QueryNum();
      } else if (opt & kXPD_startprocess) {
         TRACEP(DBG, "SendMsg: INT: setting proofserv in 'running' state");
         xps->SetStatus(kXPD_running);
         // Save start processing message for later clients
         savedBuf = xps->StartMsg();
      } else if (opt & kXPD_logmsg) {
         // We broadcast log messages only not idle to catch the
         // result from processing
         if (xps->Status() == kXPD_running) {
            TRACEP(DBG, "SendMsg: INT: broadcasting log message");
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
         if (!csid || !(csid->fP)) {
            TRACEP(XERR, "SendMsg: INT: client not connected: csid: "<<csid<<
                        ", cid: "<<cid<<", fSid: " << csid->fSid);
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
            TRACEP(DBG, "SendMsg: INT: this sid: "<<sid<<
                        ", client sid: "<<csid->fSid);
            csid->fP->fResponse.Set(csid->fSid);
            rs = SendData(&(csid->fP->fResponse), -1, &savedBuf);
            csid->fP->fResponse.Set(sid);
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
      TRACEP(DBG, "SendMsg: INT: message sent to "<<crecv[xps->SrvType()]<<
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

   TRACEP(REQ, "Admin: enter: type: "<<type<<", psid: "<<psid);

   if (type == kQuerySessions) {

      XrdProofServProxy *xps = 0;
      int ns = 0;
      std::vector<XrdProofServProxy *>::iterator ip;
      for (ip = fPClient->ProofServs()->begin(); ip != fPClient->ProofServs()->end(); ++ip)
         if ((xps = *ip) && xps->IsValid() && (xps->SrvType() == kXPD_TopMaster)) {
            ns++;
            TRACEP(XERR, "Admin: found: " << xps << "(" << xps->IsValid() <<")");
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
      char *stag = 0;
      int len = fRequest.header.dlen; 
      if (len > 0) {
         char *buf = fArgp->buff;
         if (buf[0] != '*') {
            stag = new char[len+1];
            memcpy(stag, buf, len);
            stag[len] = 0;
         }
      }

      XrdOucString tag = (!stag && ridx >= 0) ? "last" : stag;
      if (!stag && XrdProofdProtocol::GuessTag(fPClient, tag, ridx) != 0) {
         TRACEP(XERR, "Admin: query sess logs: session tag not found");
         fResponse.Send(kXR_InvalidRequest,"Admin: query log: session tag not found");
         return rc;
      }

      // Return message
      XrdOucString rmsg;

      // The session tag first
      rmsg += tag; rmsg += "|";

      // The pool URL second
      rmsg += fgPoolURL; rmsg += "|";

      // Locate the local log file
      XrdOucString sdir(fPClient->Workdir());
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
      // Scan the directory
      bool found = 0;
      struct dirent *ent = 0;
      while ((ent = (struct dirent *)readdir(dir))) {
         if (!strncmp(ent->d_name, "master-", 7) &&
              strstr(ent->d_name, ".log")) {
            rmsg += "|0 proof://"; rmsg += fgLocalHost; rmsg += ':';
            rmsg += fgPort; rmsg += '/';
            rmsg += sdir; rmsg += '/'; rmsg += ent->d_name;
            found = 1;
            break;
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
                     rmsg += "|"; rmsg += po;
                     rmsg += " "; rmsg += ln; rmsg += '/';
                     rmsg += pp;
                  }
               }
            }
         }
         fclose(f);
      }

      // Send back to user
      fResponse.Send((void *) rmsg.c_str(), rmsg.length()+1);

   } else if (type == kCleanupSessions) {

      // Target client (default us)
      XrdProofClient *tgtclnt = fPClient;

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
               TRACEP(DBG, "Admin: CleanupSessions: superuser, cleaning usr: "<< usr);
            }
         } else {
            TRACEP(DBG, "Admin: CleanupSessions: superuser, all sessions cleaned");
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
         TRACEP(DBG, "Admin: specified client has no sessions - do nothing");
      }

      if (clntfound) {

         // The clients to cleaned
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

               // This part may be not thread safe
               XrdOucMutexHelper mh(c->Mutex());

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

               // Loop over client sessions and terminated them
               int is = 0;
               XrdProofServProxy *s = 0;
               for (is = 0; is < (int) c->ProofServs()->size(); is++) {
                  if ((s = c->ProofServs()->at(is)) && s->IsValid() &&
                     s->SrvType() == srvtype) {
                     int *pid = new int;
                     *pid = s->SrvID();
                     TRACEP(HDBG, "Admin: CleanupSessions: terminating " << *pid);
                     if (TerminateProofServ(s, 0) != 0) {
                        if (KillProofServ(*pid, 0, 0) != 0) {
                           XrdOucString msg = "Admin: CleanupSessions: WARNING: process ";
                           msg += *pid;
                           msg += " could not be signalled for termination";
                           TRACEP(XERR, msg.c_str());
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
               if (XrdProofdProtocol::VerifyProcessByID(*(*ii)) == 0) {
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
      SafeDelArray(usr);

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
         TRACEP(DBG, "Admin: session tag set to: "<<xps->Tag());
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
         TRACEP(DBG, "Admin: session alias set to: "<<xps->Alias());
      }

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
      if (GetWorkers(wrks, xps) !=0 ) {
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
      TRACEP(DBG, "Admin: QueryWorkers: sending: "<<buf);

      // Send back to user
      fResponse.Send(buf, len);

   } else {
      TRACEP(XERR, "Admin: unknown request type");
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
   TRACEP(REQ, "Interrupt: psid: "<<psid<<", type:"<<type);

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

   TRACEP(REQ, "Ping: psid: "<<psid<<", opt: "<<opt);

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
      TRACEP(DBG, "Ping: xps: "<<xps<<", status: "<<xps->Status());

      // Type of connection
      bool external = !(opt & kXPD_internal);

      if (external) {
         TRACEP(DBG, "Ping: EXT: psid: "<<psid);

         // Send the request
         if ((pingres = (kXR_int32) VerifyProofServ(xps)) == -1) {
            TRACEP(XERR, "Ping: EXT: could not verify proofsrv");
            fResponse.Send(kXR_ServerError, "EXT: could not verify proofsrv");
            return rc;
         }

         // Notify the client
         TRACEP(DBG, "Ping: EXT: ping notified to client");
         fResponse.Send(kXR_ok, pingres);
         return rc;

      } else {
         TRACEP(DBG, "Ping: INT: psid: "<<psid);

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
   return rc;
}

//___________________________________________________________________________
int XrdProofdProtocol::SetUserEnvironment()
{
   // Set user environment: set effective user and group ID of the process
   // to the ones of the owner of this protocol instnace and change working
   // dir to the sandbox.
   // Return 0 on success, -1 if enything goes wrong.

   MTRACE(ACT, MHEAD, "SetUserEnvironment: enter");

   if (fPClient->Workdir() && strlen(fPClient->Workdir())) {
      MTRACE(DBG, MHEAD, "SetUserEnvironment: changing dir to : "<<fPClient->Workdir());
      if ((int) geteuid() != fUI.fUid) {

         XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
         if (!pGuard.Valid()) {
            MERROR(MHEAD, "SetUserEnvironment: could not get privileges");
            return -1;
         }

         if (chdir(fPClient->Workdir()) == -1) {
            MERROR(MHEAD, "SetUserEnvironment: can't change directory to "<<
                          fPClient->Workdir());
            return -1;
         }
      } else {
         if (chdir(fPClient->Workdir()) == -1) {
            MERROR(MHEAD, "SetUserEnvironment: can't change directory to "<<
                          fPClient->Workdir());
            return -1;
         }
      }

      // set HOME env
      char *h = new char[8 + strlen(fPClient->Workdir())];
      sprintf(h, "HOME=%s", fPClient->Workdir());
      putenv(h);
      MTRACE(XERR, MHEAD, "SetUserEnvironment: set "<<h);

   } else {
      MTRACE(XERR, MHEAD, "SetUserEnvironment: working directory undefined!");
   }

   // Set access control list from /etc/initgroup
   // (super-user privileges required)
   MTRACE(DBG, MHEAD, "SetUserEnvironment: setting ACLs");
   if ((int) geteuid() != fUI.fUid) {

      XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
      if (!pGuard.Valid()) {
         MTRACE(XERR, MHEAD, "SetUserEnvironment: could not get privileges");
         return -1;
      }

      initgroups(fUI.fUser.c_str(), fUI.fGid);
   }

   // acquire permanently target user privileges
   MTRACE(DBG, MHEAD, "SetUserEnvironment: acquire target user identity");
   if (XrdSysPriv::ChangePerm((uid_t)fUI.fUid, (gid_t)fUI.fGid) != 0) {
      MERROR(MHEAD, "SetUserEnvironment: can't acquire "<< fUI.fUser <<" identity");
      return -1;
   }

   // Save UNIX path in the sandbox for later cleaning
   // (it must be done after sandbox login)
   fPClient->SaveUNIXPath();

   // We are done
   MTRACE(DBG, MHEAD, "SetUserEnvironment: done");
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

   TRACEP(ACT, "VerifyProofServ: enter");

   if (!xps || !CanDoThis(xps->Client()))
      return rc;

   // Create semaphore
   xps->CreatePingSem();

   // Propagate the ping request
   if (xps->ProofSrv()->Send(kXR_attn, kXPD_ping, 0, 0) != 0) {
      TRACEP(XERR, "VerifyProofServ: could not propagate ping to proofsrv");
      xps->DeletePingSem();
      return rc;
   }

   // Wait for reply
   rc = 1;
   if (xps->PingSem()->Wait(fgInternalWait) != 0) {
      XrdOucString msg = "VerifyProofServ: did not receive ping reply after ";
      msg += fgInternalWait;
      msg += " secs";
      TRACEP(XERR, msg.c_str());
      rc = 0;
   }

   // Cleanup
   xps->DeletePingSem();

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

   TRACEP(ACT, "SetShutdownTimer: enter: on/off: "<<on);

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
   if (xps->ProofSrv()->Send(kXR_attn, kXPD_timer, buf, len) != 0) {
      TRACEP(XERR, "SetShutdownTimer: could not send shutdown info to proofsrv");
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
      TRACEP(DBG, msg.c_str());
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
   // For {linux, sun, macosx} it uses the system info; for other systems it
   // invokes the command shell 'ps ax' via popen.
   // Return 1 if running, 0 if not running, -1 if the check could not be run.

   int rc = 0;

   TRACE(ACT, "VerifyProcessByID: enter: pid: "<<pid);

   // Check input consistency
   if (pid < 0) {
      TRACE(XERR, "VerifyProcessByID: invalid pid");
      return -1;
   }

   // Name
   const char *pn = (pname && strlen(pname) > 0) ? pname : "proofserv";

#if defined(linux)
   // Look for the relevant /proc dir
   XrdOucString fn("/proc/");
   fn += pid;
   fn += "/stat";
   FILE *ffn = fopen(fn.c_str(), "r");
   if (!ffn) {
      if (errno == ENOENT) {
         TRACE(DBG, "VerifyProcessByID: process does not exists anymore");
         return 0;
      } else {
         XrdOucString emsg("VerifyProcessByID: cannot open ");
         emsg += fn;
         emsg += ": errno: ";
         emsg += errno;
         TRACE(XERR, emsg.c_str());
         return -1;
      }
   }
   // Read status line
   char line[2048] = { 0 };
   if (fgets(line, sizeof(line), ffn)) {
      if (strstr(line, pn))
         // Still there
         rc = 1;
   } else {
      XrdOucString emsg("VerifyProcessByID: cannot read ");
      emsg += fn;
      emsg += ": errno: ";
      emsg += errno;
      TRACE(XERR, emsg.c_str());
      fclose(ffn);
      return -1;
   }
   // Close the file
   fclose(ffn);

#elif defined(__sun)

   // Look for the relevant /proc dir
   XrdOucString fn("/proc/");
   fn += pid;
   fn += "/psinfo";
   int ffd = open(fn.c_str(), O_RDONLY);
   if (ffd <= 0) {
      if (errno == ENOENT) {
         TRACE(DBG, "VerifyProcessByID: process does not exists anymore");
         return 0;
      } else {
         XrdOucString emsg("VerifyProcessByID: cannot open ");
         emsg += fn;
         emsg += ": errno: ";
         emsg += errno;
         TRACE(XERR, emsg.c_str());
         return -1;
      }
   }
   // Get the information
   psinfo_t psi;
   if (read(ffd, &psi, sizeof(psinfo_t)) != sizeof(psinfo_t)) {
      XrdOucString emsg("VerifyProcessByID: cannot read ");
      emsg += fn;
      emsg += ": errno: ";
      emsg += errno;
      TRACE(XERR, emsg.c_str());
      close(ffd);
      return -1;
   }

   // Verify now
   if (strstr(psi.pr_fname, pn))
      // The process is still there
      rc = 1;

   // Close the file
   close(ffd);

#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)

   // Get the proclist
   kinfo_proc *pl = 0;
   int np;
   int ern = 0;
   if ((ern = GetMacProcList(&pl, np)) != 0) {
      XrdOucString emsg("VerifyProcessByID: cannot get the process list: errno: ");
      emsg += ern;
      TRACE(XERR, emsg.c_str());
      return -1;
   }

   // Loop over the list
   while (np--) {
      if (pl[np].kp_proc.p_pid == pid &&
          strstr(pl[np].kp_proc.p_comm, pn)) {
         // Process still exists
         rc = 1;
         break;
      }
   }
   // Cleanup
   free(pl);
#else
   // Use the output of 'ps ax' as a backup solution
   XrdOucString cmd = "ps ax | grep proofserv 2>/dev/null";
   if (pname && strlen(pname))
      cmd.replace("proofserv", pname);
   FILE *fp = popen(cmd.c_str(), "r");
   if (fp != 0) {
      char line[2048] = { 0 };
      while (fgets(line, sizeof(line), fp)) {
         if (pid == GetLong(line)) {
            // Process still running
            rc = 1;
            break;
         }
      }
      pclose(fp);
   } else {
      // Error executing the command
      return -1;
   }
#endif
   // Done
   return rc;
}

//______________________________________________________________________________
int XrdProofdProtocol::TrimTerminatedProcesses()
{
   // Check if the terminated processed have really exited the process
   // table; return number of processes still being terminated

   int np = 0;

   // Cleanup the list of terminated or killed processes
   if (fgTerminatedProcess.size() > 0) {
      std::list<XrdProofdPInfo *>::iterator i;
      for (i = fgTerminatedProcess.begin(); i != fgTerminatedProcess.end();) {
         XrdProofdPInfo *xi = (*i);
         if (VerifyProcessByID(xi->pid, xi->pname.c_str()) == 0) {
            TRACE(HDBG,"VerifyProcessByID: freeing: "<<xi<<" ("<<xi->pid<<", "<<xi->pname<<")");
            // Cleanup the integer
            delete *i;
            // Process has terminated: remove it from the list
            i = fgTerminatedProcess.erase(i);
         } else {
            // Count
            np++;
            // Goto next
            i++;
         }
      }
   }

   // Done
   return np;
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

   TRACEP(ACT, "CleanupProofServ: enter: all: "<<all<<
               ", usr: " << (usr ? usr : (const char *)fClientID));
   int nk = 0;

   // Check if 'all' makes sense
   if (all && !fSuperUser) {
      all = 0;
      TRACEP(DBG, "CleanupProofServ: request for all without privileges: setting all = FALSE");
   }

   // Name
   const char *pn = "proofserv";

   // Uid
   int refuid = -1;
   if (!all) {
      if (!usr) {
         TRACEP(DBG, "CleanupProofServ: usr must be defined for all = FALSE");
         return -1;
      }
      XrdProofUI ui;
      if (GetUserInfo(usr, ui) != 0) {
         TRACEP(DBG, "CleanupProofServ: problems getting info for user " << usr);
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
      TRACEP(DBG, emsg.c_str());
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
            TRACEP(HDBG, emsg.c_str());
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
               pid = (int) GetLong(&line[strlen("Pid:")]);
               xpid = 0;
            }
            if (xppid && strstr(line, "PPid:")) {
               ppid = (int) GetLong(&line[strlen("PPid:")]);
               // Parent process must be us or be dead
               if (ppid != getpid() &&
                   XrdProofdProtocol::VerifyProcessByID(ppid, "xrootd"))
                  // Process created by another running xrootd
                  break;
               xppid = 0;
            }
            if (xuid && strstr(line, "Uid:")) {
               int uid = (int) GetLong(&line[strlen("Uid:")]);
               if (refuid == uid)
                  xuid = 0;
            }
         }
         // Close the file
         fclose(ffn);
         // If this is a good candidate, kill it
         if (!xname && !xpid && !xppid && !xuid) {
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
      TRACEP(DBG, emsg.c_str());
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
            TRACEP(HDBG, emsg.c_str());
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
             XrdProofdProtocol::VerifyProcessByID(ppid, "xrootd")) {
             // Process created by another running xrootd
             continue;
             xppid = 0;
         }

         // If this is a good candidate, kill it
         if (!xname && !xppid && !xuid) {
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
   if ((ern = GetMacProcList(&pl, np)) != 0) {
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
            if (!xppid)
               // Good candidate to be shot
               if (KillProofServ(pl[np].kp_proc.p_pid, 1))
                  nk++;
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
            int ppid = (int) GetLong(pi);
            TRACEP(HDBG, "CleanupProofServ: found alternative parent ID: "<< ppid);
            // If still running then skip
            if (XrdProofdProtocol::VerifyProcessByID(ppid, "xrootd"))
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
int XrdProofdProtocol::KillProofServ(int pid, bool forcekill, bool add)
{
   // Kill the process 'pid'.
   // A SIGTERM is sent, unless 'kill' is TRUE, in which case a SIGKILL is used.
   // If add is TRUE (default) the pid is added to the list of processes
   // requested to terminate.
   // Return 0 on success, -1 if not allowed or other errors occured.

   TRACEP(ACT, "KillProofServ: enter: pid: "<<pid<< ", forcekill: "<< forcekill);

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
                  TRACEP(XERR, msg.c_str());
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
                  TRACEP(XERR, msg.c_str());
                  return -1;
               }
               signalled = 0;
            }
         // Add to the list of termination attempts
         if (signalled) {
            if (add) {
               // This part may be not thread safe
               XrdOucMutexHelper mtxh(&fgXPDMutex);
               fgTerminatedProcess.push_back(new XrdProofdPInfo(pid, "proofserv"));
               TRACEP(DBG, "KillProofServ: process ID "<<pid<<" signalled and pushed back");
            } else {
               TRACEP(DBG, "KillProofServ: "<<pid<<" signalled");
            }
            // Record this session in the sandbox as old session
            XrdOucString tag = "-";
            tag += pid;
            if (XrdProofdProtocol::GuessTag(fPClient, tag) == 0) {
               if (XrdProofdProtocol::MvOldSession(fPClient, tag.c_str(), fgMaxOldLogs) == -1)
                  TRACEP(XERR, "KillProofServ: problems recording session as old in sandbox");
            } else {
                  TRACEP(DBG, "KillProofServ: problems guessing tag");
            }
         } else {
            TRACEP(DBG, "KillProofServ: process ID "<<pid<<" not found in the process table");
         }
      } else {
        XrdOucString msg = "KillProofServ: could not get privileges";
        TRACEP(XERR, msg.c_str());
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

   TRACEP(ACT, "KillProofServ: enter: forcekill: "<< forcekill);

   if (!xps || !CanDoThis(xps->Client()))
      return -1;

   int pid = xps->SrvID();
   if (pid > -1) {
      // Kill by ID
      if (KillProofServ(pid, forcekill, add) != 0) {
         TRACEP(XERR, "KillProofServ: problems killing process by ID ("<<pid<<")");
         return -1;
      }
   } else {
      TRACEP(XERR, "KillProofServ: invalid session process ID ("<<pid<<")");
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

   TRACEP(ACT, "TerminateProofServ: enter: " << (xps ? xps->Ordinal() : "-"));

   if (!xps || !CanDoThis(xps->Client()))
      return -1;

   // Send a terminate signal to the proofserv
   int pid = xps->SrvID();
   if (pid > -1) {

      int type = 3;
      if (xps->ProofSrv()->Send(kXR_attn, kXPD_interrupt, type) != 0) {
         // Could not send: try termination by signal
         return KillProofServ(xps);
      }
      if (add) {
         // Add to the list of termination attempts
         // This part may be not thread safe
         XrdOucMutexHelper mtxh(&fgXPDMutex);
         fgTerminatedProcess.push_back(new XrdProofdPInfo(pid, "proofserv"));
         TRACEP(DBG, "TerminateProofServ: "<<pid<<" pushed back");
      }
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
   char *file = 0;
   int dlen = fRequest.header.dlen; 
   if (dlen > 0 && fArgp->buff) {
      file = new char[dlen+1];
      memcpy(file, fArgp->buff, dlen);
      file[dlen] = 0;
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
   TRACEP(REQ, "ReadBuffer: file: "<<file<<", ofs: "<<ofs<<", len: "<<len);

   // Check if local
   bool local = 0;
   XrdClientUrlInfo ui(file);
   if (ui.Host.length() > 0) {
      // Fully qualified name
      char *fqn = XrdNetDNS::getHostName(ui.Host.c_str());
      if (fqn && (strstr(fqn, "localhost") ||
                 !strcmp(fqn, "127.0.0.1") ||
                  fgLocalHost == (const char *)fqn)) {
         memcpy(file, ui.File.c_str(), ui.File.length());
         file[ui.File.length()] = 0;
         local = 1;
         TRACEP(DBG, "ReadBuffer: file is LOCAL");
      }
      SafeFree(fqn);
   }

   // Get the buffer
   int lout = len;
   char *buf = (local) ? ReadBufferLocal(file, ofs, lout)
                       : ReadBufferRemote(file, ofs, lout);
   if (!buf) {
      emsg = "ReadBuffer: could not read buffer from ";
      emsg += (local) ? "local file " : "remote file ";
      emsg += file;
      TRACEP(XERR, emsg);
      fResponse.Send(kXR_InvalidRequest, emsg.c_str());
      return rc;
   }

   // Send back to user
   fResponse.Send(buf, lout);

   // Cleanup
   SafeFree(buf);

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
   TRACEP(ACT, "ReadBufferLocal: file: "<<file<<", ofs: "<<ofs<<", len: "<<len);

   // Check input
   if (!file || strlen(file) <= 0) {
      TRACEP(XERR, "ReadBufferLocal: file path undefined!");
      return (char *)0;
   }

   // Open the file in read mode
   int fd = open(file, O_RDONLY);
   if (fd < 0) {
      emsg = "ReadBufferLocal: could not open ";
      emsg += file;
      TRACEP(XERR, emsg);
      return (char *)0;
   }

   // Size of the output
   struct stat st;
   if (fstat(fd, &st) != 0) {
      emsg = "ReadBufferLocal: could not get size of file with stat: errno: ";
      emsg += (int)errno;
      TRACEP(XERR, emsg);
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
   off_t lst = (end >= ltot) ? ltot : ((end > fst) ? end  : fst);
   TRACEP(DBG, "ReadBufferLocal: file size: "<<ltot<<
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
      TRACEP(HDBG, "ReadBufferLocal: read "<<nr<<" bytes: "<< buf);
      if (nr < 0) {
         TRACEP(XERR, "ReadBufferLocal: error reading from file: errno: "<< errno);
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
char *XrdProofdProtocol::ReadBufferRemote(const char *url,
                                          kXR_int64 ofs, int &len)
{
   // Send a read buffer request of length 'len' at offset 'ofs' for remote file
   // defined by 'url'; the returned buffer must be freed by the caller.
   // Returns 0 in case of error.

   TRACEP(ACT, "ReadBufferRemote: url: "<<(url ? url : "undef")<<
                                       ", ofs: "<<ofs<<", len: "<<len);

   // Check input
   if (!url || strlen(url) <= 0) {
      TRACEP(XERR, "ReadBufferRemote: url undefined!");
      return (char *)0;
   }

   // We try only once
   int maxtry_save = -1;
   int timewait_save = -1;
   XrdProofConn::GetRetryParam(maxtry_save, timewait_save);
   XrdProofConn::SetRetryParam(1, 1);

   // Open the connection
   XrdOucString msg = "readbuffer request from ";
   msg += fgLocalHost;
   char m = 'A'; // log as admin
   XrdProofConn *conn = new XrdProofConn(url, m, -1, -1, 0, msg.c_str());

   char *buf = 0;
   if (conn && conn->IsValid()) {
      // Prepare request
      XPClientRequest reqhdr;
      memset(&reqhdr, 0, sizeof(reqhdr));
      conn->SetSID(reqhdr.header.streamid);
      reqhdr.header.requestid = kXP_readbuf;
      reqhdr.readbuf.ofs = ofs;
      reqhdr.readbuf.len = len;
      reqhdr.header.dlen = strlen(url);
      const void *btmp = (const void *) url;
      void **vout = (void **)&buf;
      // Send over
      XrdClientMessage *xrsp =
         conn->SendReq(&reqhdr, btmp, vout, "XrdProofdProtocol::ReadBufferRemote");

      // If positive answer
      if (xrsp && buf && (xrsp->DataLen() > 0)) {
         len = xrsp->DataLen();
      } else {
         SafeFree(buf);
      }

      // Clean the message
      SafeDelete(xrsp);

      // Close physically the connection
      conn->Close("S");
      // Delete it
      SafeDelete(conn);
   }

   // Restore original retry parameters
   XrdProofConn::SetRetryParam(maxtry_save, timewait_save);

   // Done
   return buf;
}

//______________________________________________________________________________
int XrdProofdProtocol::GuessTag(XrdProofClient *pcl, XrdOucString &tag, int ridx)
{
   // Guess session tag completing 'tag' (typically "-<pid>") by scanning the
   // active session file or the session dir.
   // In case of success, tag is filled with the full tag and 0 is returned.
   // In case of failure, -1 is returned.

   TRACE(ACT, "GuessTag: enter: "<< (pcl ? pcl->ID() : "-") <<", tag: "<<tag);

   // Check inputs
   if (!pcl) {
      TRACE(XERR, "GuessTag: client undefined");
      return -1;
   }
   bool found = 0;
   bool last = (tag == "last") ? 1 : 0;

   if (!last && tag.length() > 0) {
      // Scan the sessions file
      XrdOucString fn = pcl->Workdir();
      fn += "/.sessions";

      // Open the file for reading
      FILE *fact = fopen(fn.c_str(), "a+");
      if (fact) {
         // Lock the file
         if (lockf(fileno(fact), F_LOCK, 0) == 0) {
            // Read content, if already existing
            char ln[1024];
            while (fgets(ln, sizeof(ln), fact)) {
               // Get rid of '\n'
               if (ln[strlen(ln)-1] == '\n')
                  ln[strlen(ln)-1] = '\0';
               // Skip empty or comment lines
               if (strlen(ln) <= 0 || ln[0] == '#')
                  continue;
               // Count if not the one we want to remove
               if (!strstr(ln, tag.c_str())) {
                  tag = ln;
                  found = 1;
                  break;
               }
            }
            // Unlock the file
            lseek(fileno(fact), 0, SEEK_SET);
            if (lockf(fileno(fact), F_ULOCK, 0) == -1)
               TRACE(DBG, "GuessTag: cannot unlock file "<<fn<<" ; fact: "<<fact<<
                          ", fd: "<< fileno(fact) << " (errno: "<<errno<<")");

         } else {
            TRACE(DBG, "GuessTag: cannot lock file: "<<fn<<" ; fact: "<<fact<<
                       ", fd: "<< fileno(fact) << " (errno: "<<errno<<")");
         }
         // Close the file
         fclose(fact);

      } else {
         TRACE(DBG, "GuessTag: cannot open file "<<fn<<
                    " for reading (errno: "<<errno<<")");
      }
   }

   if (!found) {

      // Search the tag in the dirs
      std::list<XrdOucString *> staglst;
      int rc = GetSessionDirs(pcl, 3, &staglst, &tag);
      if (rc < 0) {
         TRACE(XERR, "GuessTag: cannot scan dir "<<pcl->Workdir());
         return -1;
      }
      found = (rc == 1) ? 1 : 0;

      if (!found) {
         // Take last one, if required
         if (last) {
            tag = staglst.front()->c_str();
            found = 1;
         } else {
            if (ridx < 0) {
               int itag = ridx;
               // Reiterate back
               std::list<XrdOucString *>::iterator i;
               for (i = staglst.end(); i != staglst.begin(); --i) {
                  if (itag == 0) {
                     tag = (*i)->c_str();
                     found = 1;
                     break;
                  }
                  itag++;
               }
            }
         }
      }
      // Cleanup
      staglst.clear();
      // Correct the tag
      if (found) {
         tag.replace("session-", "");
      } else {
         TRACE(DBG, "GuessTag: tag "<<tag<<" not found in dir");
      }
   }

   // We are done
   return ((found) ? 0 : -1);
}

//______________________________________________________________________________
int XrdProofdProtocol::AddNewSession(XrdProofClient *pcl, const char *tag)
{
   // Record entry for client's new proofserv session tagged 'tag' in the active
   // sessions file (<SandBox>/.sessions). The file is created if needed.
   // Return 0 on success, -1 on error. 


   // Check inputs
   if (!pcl || !tag) {
      XPDPRT("XrdProofdProtocol::AddNewSession: invalid inputs");
      return -1;
   }
   TRACE(ACT, "AddNewSession: enter: client: "<< pcl->ID()<<", tag:"<<tag);

   // File name
   XrdOucString fn = pcl->Workdir();
   fn += "/.sessions";

   // Open the file for appending
   FILE *fact = fopen(fn.c_str(), "a+");
   if (!fact) {
      TRACE(XERR, "AddNewSession: cannot open file "<<fn<<
                 " for appending (errno: "<<errno<<")");
      return -1;
   }

   // Lock the file
   lseek(fileno(fact), 0, SEEK_SET);
   if (lockf(fileno(fact), F_LOCK, 0) == -1) {
      TRACE(XERR, "AddNewSession: cannot lock file "<<fn<<
                 " (errno: "<<errno<<")");
      fclose(fact);
      return -1;
   }

   bool writeout = 1;

   // Check if already there
   std::list<XrdOucString *> actln;
   char ln[1024];
   while (fgets(ln, sizeof(ln), fact)) {
      // Get rid of '\n'
      if (ln[strlen(ln)-1] == '\n')
         ln[strlen(ln)-1] = '\0';
      // Skip empty or comment lines
      if (strlen(ln) <= 0 || ln[0] == '#')
         continue;
      // Count if not the one we want to remove
      if (strstr(ln, tag))
         writeout = 0;
   }

   // Append the session unique tag
   if (writeout) {
      lseek(fileno(fact), 0, SEEK_END);
      fprintf(fact, "%s\n", tag);
   }

   // Unlock the file
   lseek(fileno(fact), 0, SEEK_SET);
   if (lockf(fileno(fact), F_ULOCK, 0) == -1)
      TRACE(XERR, "AddNewSession: cannot unlock file "<<fn<<
                 " (errno: "<<errno<<")");

   // Close the file
   fclose(fact);

   // We are done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::MvOldSession(XrdProofClient *pcl,
                                    const char *tag, int maxold)
{
   // Move record for tag from the active sessions file to the old 
   // sessions file (<SandBox>/.sessions). The active file is removed if
   // empty after the operation. The old sessions file is created if needed.
   // If maxold > 0, logs for a maxold number of sessions are kept in the
   // sandbox; working dirs for sessions in excess are removed.
   // Return 0 on success, -1 on error. 

   char ln[1024];

   // Check inputs
   if (!pcl || !tag) {
      TRACE(XERR, "MvOldSession: invalid inputs");
      return -1;
   }
   TRACE(ACT, "MvOldSession: enter: client: "<< pcl->ID()<<
              ", tag:"<<tag<<", maxold:"<<maxold);

   // Update of the active file
   XrdOucString fna = pcl->Workdir();
   fna += "/.sessions";

   // Open the file
   FILE *fact = fopen(fna.c_str(), "a+");
   if (!fact) {
      TRACE(XERR, "MvOldSession: cannot open file "<<fna<<
                 " (errno: "<<errno<<")");
      return -1;
   }

   // Lock the file
   if (lockf(fileno(fact), F_LOCK, 0) == -1) {
      TRACE(XERR, "MvOldSession: cannot lock file "<<fna<<
                 " (errno: "<<errno<<")");
      fclose(fact);
      return -1;
   }

   // Read content, if already existing
   std::list<XrdOucString *> actln;
   while (fgets(ln, sizeof(ln), fact)) {
      // Get rid of '\n'
      if (ln[strlen(ln)-1] == '\n')
         ln[strlen(ln)-1] = '\0';
      // Skip empty or comment lines
      if (strlen(ln) <= 0 || ln[0] == '#')
         continue;
      // Count if not the one we want to remove
      if (!strstr(ln, tag))
         actln.push_back(new XrdOucString(ln));
   }

   // Truncate the file
   if (ftruncate(fileno(fact), 0) == -1) {
      TRACE(XERR, "MvOldSession: cannot truncate file "<<fna<<
                 " (errno: "<<errno<<")");
      lseek(fileno(fact), 0, SEEK_SET);
      lockf(fileno(fact), F_ULOCK, 0);
      fclose(fact);
      return -1;
   }

   // If active sessions still exist, write out new composition
   bool unlk = 1;
   if (actln.size() > 0) {
      unlk = 0;
      std::list<XrdOucString *>::iterator i;
      for (i = actln.begin(); i != actln.end(); ++i) {
         fprintf(fact, "%s\n", (*i)->c_str());
         delete (*i);
      }
   }

   // Unlock the file
   lseek(fileno(fact), 0, SEEK_SET);
   if (lockf(fileno(fact), F_ULOCK, 0) == -1)
      TRACE(XERR, "MvOldSession: cannot unlock file "<<fna<<
                  " (errno: "<<errno<<")");

   // Close the file
   fclose(fact);

   // Unlink the file if empty
   if (unlk)
      if (unlink(fna.c_str()) == -1) 
         TRACE(XERR, "MvOldSession: cannot unlink file "<<fna<<
                    " (errno: "<<errno<<")");

   // Flag the session as closed
   XrdOucString fterm = pcl->Workdir();
   fterm += (strstr(tag,"session-")) ? "/" : "/session-";
   fterm += tag;
   fterm += "/.terminated";
   // Create the file
   FILE *ft = fopen(fterm.c_str(), "w");
   if (!ft) {
      TRACE(XERR, "MvOldSession: cannot open file "<<fterm<<
                 " (errno: "<<errno<<")");
      return -1;
   }
   fclose(ft);

   // If a limit on the number of sessions dirs is set, apply it
   if (maxold > 0) {

      // Get list of terminated session working dirs
      std::list<XrdOucString *> staglst;
      if (GetSessionDirs(pcl, 2, &staglst) != 0) {
         TRACE(XERR, "MvOldSession: cannot get list of dirs ");
         return -1;
      }
      TRACE(DBG, "MvOldSession: number of working dirs: "<<staglst.size());

      std::list<XrdOucString *>::iterator i;
      for (i = staglst.begin(); i != staglst.end(); ++i) {
         TRACE(HDBG, "MvOldSession: found "<<(*i)->c_str());
      }

      // Remove the oldest, if needed
      while ((int)staglst.size() > maxold) {
         XrdOucString *s = staglst.back();
         if (s) {
            TRACE(HDBG, "MvOldSession: removing "<<s->c_str());
            // Remove associated workdir
            XrdOucString rmcmd = "/bin/rm -rf ";
            rmcmd += pcl->Workdir();
            rmcmd += '/';
            rmcmd += s->c_str();
            system(rmcmd.c_str());
            // Delete the string
            delete s;
         }
         // Remove the last element
         staglst.pop_back();
      }

      // Clear the list
      staglst.clear();
   }

   // Done
   return 0;
}

//______________________________________________________________________________
int XrdProofdProtocol::GetSessionDirs(XrdProofClient *pcl, int opt,
                                      std::list<XrdOucString *> *sdirs,
                                      XrdOucString *tag)
{
   // Scan the pcl's sandbox for sessions working dirs and return their
   // sorted (according to creation time, first is the newest) list
   // in 'sdirs'.
   // The option 'opt' may have 3 values:
   //    0        all working dirs are kept
   //    1        active sessions only
   //    2        terminated sessions only
   //    3        search entry containing 'tag' and fill tag with
   //             the full entry name; if defined, sdirs is filled
   // Returns -1 otherwise in case of failure.
   // In case of success returns 0 for opt < 3, 1 if found or 0 if not
   // found for opt == 3.

   // If unknown take all
   opt = (opt >= 0 && opt <= 3) ? opt : 0;

   // Check inputs
   if (!pcl || (opt < 3 && !sdirs) || (opt == 3 && !tag)) {
      TRACE(XERR, "GetSessionDirs: invalid inputs");
      return -1;
   }

   TRACE(ACT, "GetSessionDirs: enter: opt: "<<opt<<", dir: "<<pcl->Workdir());

   // Open dir
   DIR *dir = opendir(pcl->Workdir());
   if (!dir) {
      TRACE(XERR, "GetSessionDirs: cannot open dir "<<pcl->Workdir()<<
            " (errno: "<<errno<<")");
      return -1;
   }

   // Scan the directory, and save the paths for terminated sessions
   // into a list
   bool found = 0;
   struct dirent *ent = 0;
   while ((ent = (struct dirent *)readdir(dir))) {
      if (!strncmp(ent->d_name, "session-", 8)) {
         bool keep = 1;
         if (opt == 3 && tag->length() > 0) {
            if (strstr(ent->d_name, tag->c_str())) {
               *tag = ent->d_name;
               found = 1;
            }
         } else {
            if (opt > 0) {
               XrdOucString fterm(pcl->Workdir());
               fterm += '/';
               fterm += ent->d_name;
               fterm += "/.terminated";
               int rc = access(fterm.c_str(), F_OK);
               if ((opt == 1 && rc == 0) || (opt == 2 && rc != 0))
                  keep = 0;
            }
         }
         TRACE(HDBG, "GetSessionDirs: found entry "<<ent->d_name<<", keep: "<<keep);
         if (sdirs && keep)
            sdirs->push_back(new XrdOucString(ent->d_name));
      }
   }

   // Close the directory
   closedir(dir);

   // Sort the list
   if (sdirs)
#if !defined(__sun)
      sdirs->sort(&SessionTagComp);
#else
      Sort(sdirs);
#endif

   // Done
   return ((opt == 3 && found) ? 1 : 0);
}

//______________________________________________________________________________
int XrdProofdProtocol::GetNumCPUs()
{
   // Find out and return the number of CPUs in the local machine.
   // Return -1 in case of failure.

   int ncpu = 0;

#if defined(linux)
   // Look for in the /proc/cpuinfo file
   XrdOucString fcpu("/proc/cpuinfo");
   FILE *fc = fopen(fcpu.c_str(), "r");
   if (!fc) {
      if (errno == ENOENT) {
         TRACE(DBG, "GetNumCPUs: /proc/cpuinfo missing!!! Something very bad going on");
      } else {
         XrdOucString emsg("VerifyProcessByID: cannot open ");
         emsg += fcpu;
         emsg += ": errno: ";
         emsg += errno;
         TRACE(XERR, emsg.c_str());
      }
      return -1;
   }
   // Read lines and count those starting with "processor"
   char line[2048] = { 0 };
   while (fgets(line, sizeof(line), fc)) {
      if (!strncmp(line, "processor", strlen("processor")))
         ncpu++;
   }
   // Close the file
   fclose(fc);

#elif defined(__sun)

   // Run "psrinfo" in popen and count lines
   FILE *fp = popen("psrinfo", "r");
   if (fp != 0) {
      char line[2048] = { 0 };
      while (fgets(line, sizeof(line), fp))
         ncpu++;
      pclose(fp);
   }

#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)

   // Run "sysctl -n hw.ncpu" in popen and decode the output
   FILE *fp = popen("sysctl -n hw.ncpu", "r");
   if (fp != 0) {
      char line[2048] = { 0 };
      while (fgets(line, sizeof(line), fp))
         ncpu = GetLong(&line[0]);
      pclose(fp);
   }
#endif

   // Done
   return (ncpu <= 0) ? (int)(-1) : ncpu ;
}

//--------------------------------------------------------------------------
//
// XrdProofClient
//
//--------------------------------------------------------------------------

//__________________________________________________________________________
XrdProofClient::XrdProofClient(XrdProofdProtocol *p,
                               short int clientvers, const char *wrkdir)
{
   // Constructor

   fClientID = (p && p->GetID()) ? strdup(p->GetID()) : 0;
   fClientVers = clientvers;
   fProofServs.reserve(10);
   fClients.reserve(10);
   fWorkdir = (wrkdir) ? strdup(wrkdir) : 0;
   fUNIXSock = 0;
   fUNIXSockPath = 0;
   fUNIXSockSaved = 0;
}

//__________________________________________________________________________
XrdProofClient::~XrdProofClient()
{
   // Destructor

   SafeFree(fClientID);
   SafeFree(fWorkdir);

   // Unix socket
   SafeDelete(fUNIXSock);
   SafeDelArray(fUNIXSockPath);
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

   TRACE(DBG, "XrdProofClient::GetClientID: size: "<<fClients.size());

   // We are done
   return ic;
}

//__________________________________________________________________________
int XrdProofClient::CreateUNIXSock(XrdOucError *edest, char *tmpdir)
{
   // Create UNIX socket for internal connections

   TRACE(ACT,"CreateUNIXSock: enter");

   // Make sure we do not have already a socket
   if (fUNIXSock && fUNIXSockPath) {
       TRACE(DBG,"CreateUNIXSock: UNIX socket exists already! (" <<
             fUNIXSockPath<<")");
       return 0;
   }

   // Make sure we do not have inconsistencies
   if (fUNIXSock || fUNIXSockPath) {
       TRACE(XERR,"CreateUNIXSock: inconsistent values: corruption? (sock: " <<
                 fUNIXSock<<", path: "<< fUNIXSockPath);
       return -1;
   }

   // Inputs must make sense
   if (!edest || !tmpdir) {
       TRACE(XERR,"CreateUNIXSock: invalid inputs: edest: " <<
                 (int *)edest <<", tmpdir: "<< (int *)tmpdir);
       return -1;
   }

   // Create socket
   fUNIXSock = new XrdNet(edest);

   // Create path
   fUNIXSockPath = new char[strlen(tmpdir)+strlen("/xpdsock_XXXXXX")+2];
   sprintf(fUNIXSockPath,"%s/xpdsock_XXXXXX", tmpdir);
   int fd = mkstemp(fUNIXSockPath);
   if (fd > -1) {
      close(fd);
      if (fUNIXSock->Bind(fUNIXSockPath)) {
         TRACE(XERR,"CreateUNIXSock: warning:"
                   " problems binding to UNIX socket; path: " <<fUNIXSockPath);
         return -1;
      } else
         TRACE(DBG, "CreateUNIXSock: path for UNIX for socket is " <<fUNIXSockPath);
   } else {
      TRACE(XERR,"CreateUNIXSock: unable to generate unique"
            " path for UNIX socket; tried path " << fUNIXSockPath);
      return -1;
   }

   // We are done
   return 0;
}

//__________________________________________________________________________
void XrdProofClient::SaveUNIXPath()
{
   // Save UNIX path in <SandBox>/.unixpath

   TRACE(ACT,"SaveUNIXPath: enter: saved? "<<fUNIXSockSaved);

   // Make sure we do not have already a socket
   if (fUNIXSockSaved) {
      TRACE(DBG,"SaveUNIXPath: UNIX path saved already");
      return;
   }

   // Make sure we do not have already a socket
   if (!fUNIXSockPath) {
       TRACE(XERR,"SaveUNIXPath: UNIX path undefined!");
       return;
   }

   // File name
   XrdOucString fn = fWorkdir;
   fn += "/.unixpath";

   // Open the file for appending
   FILE *fup = fopen(fn.c_str(), "a+");
   if (!fup) {
      TRACE(XERR, "SaveUNIXPath: cannot open file "<<fn<<
            " for appending (errno: "<<errno<<")");
      return;
   }

   // Lock the file
   lseek(fileno(fup), 0, SEEK_SET);
   if (lockf(fileno(fup), F_LOCK, 0) == -1) {
      TRACE(XERR, "SaveUNIXPath: cannot lock file "<<fn<<
            " (errno: "<<errno<<")");
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
      if ((vrc = XrdProofdProtocol::VerifyProcessByID(pid, "xrootd")) != 0) {
         // Still there
         actln.push_back(new XrdOucString(ln));
      } else if (vrc == 0) {
         // Not running: remove the socket path
         TRACE(DBG, "SaveUNIXPath: unlinking socket path "<< path);
         if (unlink(path) != 0 && errno != ENOENT) {
            TRACE(XERR, "SaveUNIXPath: problems unlinking socket path "<< path<<
                    " (errno: "<<errno<<")");
         }
      }
   }

   // Truncate the file
   if (ftruncate(fileno(fup), 0) == -1) {
      TRACE(XERR, "SaveUNIXPath: cannot truncate file "<<fn<<
                 " (errno: "<<errno<<")");
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
   fprintf(fup, "%d %s\n", getppid(), fUNIXSockPath);

   // Unlock the file
   lseek(fileno(fup), 0, SEEK_SET);
   if (lockf(fileno(fup), F_ULOCK, 0) == -1)
      TRACE(XERR, "SaveUNIXPath: cannot unlock file "<<fn<<
                 " (errno: "<<errno<<")");

   // Close the file
   fclose(fup);

   // Path saved
   fUNIXSockSaved = 1;
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
         TRACE(XERR, "XrdProofWorker::Reset: unknown option "<<tok);
      }
   }

   // Default image is the host name
   if (fImage.length() <= 0)
      fImage.assign(fHost, fHost.find('@'));
}

//__________________________________________________________________________
bool XrdProofWorker::Matches(const char *host)
{
   // Check compatibility of host with this instance.
   // return 1 if compatible.

   XrdOucString thishost;
   thishost.assign(fHost, fHost.find('@'));
   char *h = XrdNetDNS::getHostName(thishost.c_str());
   thishost = (h ? h : "");
   SafeFree(h);

   return ((thishost.matches(host)) ? 1 : 0);
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
   TRACE(DBG, "XrdProofWorker::Export: sending: "<<fExport);
   return fExport.c_str();
}
