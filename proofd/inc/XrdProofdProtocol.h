// @(#)root/proofd:$Name:  $:$Id: XrdProofdProtocol.h,v 1.24 2007/06/12 13:51:03 ganis Exp $
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdProtocol
#define ROOT_XrdProofdProtocol

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdProtocol                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// XrdProtocol implementation to coordinate 'proofserv' applications.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdOuc/XrdOucError.hh"
#include "XrdOuc/XrdOucPthread.hh"
#include "XrdOuc/XrdOucSemWait.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSec/XrdSecInterface.hh"

#include "Xrd/XrdProtocol.hh"
#include "Xrd/XrdObject.hh"

#include "XProofProtocol.h"
#include "XrdProofdManager.h"
#include "XrdProofdResponse.h"
#include "XrdProofGroup.h"
#include "XrdProofServProxy.h"
#include "XrdProofdAux.h"

#include <list>
#include <vector>

// Version index: start from 1001 (0x3E9) to distinguish from 'proofd'
// To be increment when non-backward compatible changes are introduced
//  1001 (0x3E9) -> 1002 (0x3EA) : support for flexible env setting
#define XPROOFD_VERSBIN 0x000003EA
#define XPROOFD_VERSION "0.2"

#define XPD_LOGGEDIN       1
#define XPD_NEED_AUTH      2
#define XPD_ADMINUSER      4
#define XPD_NEED_MAP       8

class XrdROOT;
class XrdBuffer;
class XrdClientMessage;
class XrdLink;
class XrdOucError;
class XrdOucTrace;
class XrdProofdClient;
class XrdProofdPInfo;
class XrdProofdPriority;
class XrdProofSched;
class XrdProofWorker;
class XrdScheduler;
class XrdSrvBuffer;

class XrdProofdProtocol : XrdProtocol {

friend class XrdProofdClient;
friend class XrdROOT;

public:
   XrdProofdProtocol();
   virtual ~XrdProofdProtocol() {} // Never gets destroyed

   static int    Configure(char *parms, XrdProtocol_Config *pi);
   void          DoIt() {}
   XrdProtocol  *Match(XrdLink *lp);
   int           Process(XrdLink *lp);
   void          Recycle(XrdLink *lp, int x, const char *y);
   int           Stats(char *buff, int blen, int do_sync);

   const char   *GetID() const { return (const char *)fClientID; }

   static int    Reconfig();
   static int    TraceConfig(const char *cfn);
   static int    TrimTerminatedProcesses();

 private:

   int           Admin();
   int           Attach();
   int           Auth();
   int           Create();
   int           Destroy();
   int           Detach();
   int           GetBuff(int quantum);
   int           GetData(const char *dtype, char *buff, int blen);
   XrdProofServProxy *GetServer(int psid);
   int           Interrupt();
   int           Login();
   int           MapClient(bool all = 1);
   int           Ping();
   int           Process2();
   int           ReadBuffer();
   char         *ReadBufferLocal(const char *file, kXR_int64 ofs, int &len);
   char         *ReadBufferRemote(const char *file, kXR_int64 ofs, int &len);
   void          Reset();
   int           SendData(XrdProofdResponse *resp, kXR_int32 sid = -1, XrdSrvBuffer **buf = 0);
   int           SendDataN(XrdProofServProxy *xps, XrdSrvBuffer **buf = 0);
   int           SendMsg();
   int           SetUserEnvironment();
   int           Urgent();

   int           CleanupProofServ(bool all = 0, const char *usr = 0);
   int           KillProofServ(int pid, bool forcekill = 0);
   int           SetProofServEnv(int psid = -1, int loglevel = -1, const char *cfg = 0);
   int           SetProofServEnvOld(int psid = -1, int loglevel = -1, const char *cfg = 0);
   //
   // Local area
   //
   XrdObject<XrdProofdProtocol>  fProtLink;
   XrdLink                      *fLink;
   XrdBuffer                    *fArgp;
   char                          fStatus;
   char                         *fClientID;    // login username
   char                         *fGroupID;     // login group name
   XrdProofUI                    fUI;           // user info
   unsigned char                 fCapVer;
   kXR_int32                     fSrvType;      // Master or Worker
   bool                          fTopClient;    // External client (not ProofServ)
   bool                          fSuperUser;    // TRUE for privileged clients (admins)
   //
   XrdProofdClient              *fPClient;    // Our reference XrdProofdClient
   kXR_int32                     fCID;        // Reference ID of this client
   //
   XrdSecEntity                 *fClient;
   XrdSecProtocol               *fAuthProt;
   XrdSecEntity                  fEntity;
   //
   char                         *fBuff;
   int                           fBlen;
   int                           fBlast;
   //
   int                           fhcPrev;
   int                           fhcMax;
   int                           fhcNext;
   int                           fhcNow;
   int                           fhalfBSize;
   //
   XPClientRequest               fRequest;  // handle client requests
   XrdProofdResponse             fResponse; // Response to incoming request
   XrdOucRecMutex                fMutex;    // Local mutex

   //
   // Static area: general protocol managing section
   //
   static XrdOucRecMutex         fgXPDMutex;  // Mutex for static area
   static int                    fgCount;
   static XrdObjectQ<XrdProofdProtocol> fgProtStack;
   static XrdBuffManager        *fgBPool;     // Buffer manager
   static int                    fgMaxBuffsz;    // Maximum buffer size we can have
   static XrdSecService         *fgCIA;       // Authentication Server
   static XrdScheduler          *fgSched;     // System scheduler
   static XrdOucError            fgEDest;     // Error message handler
   static XrdOucLogger           fgMainLogger; // Error logger

   //
   // Static area: protocol configuration section
   //
   static XrdProofdFile          fgCfgFile;    // Main config file
   static bool                   fgConfigDone; // Whether configure has been run
   static std::list<XrdROOT *>   fgROOT;     // ROOT versions; the first is the default
   static char                  *fgTMPdir;   // directory for temporary files
   static char                  *fgSecLib;
   // 
   static char                  *fgPoolURL;    // Local pool URL
   static char                  *fgNamespace;  // Local pool namespace
   //
   static XrdOucSemWait          fgForkSem;   // To serialize fork requests
   //
   static int                    fgMaxSessions; // max number of sessions per client
   static int                    fgWorkerMax; // max number or workers per user
   static EStaticSelOpt          fgWorkerSel; // selection option
   static std::list<XrdOucString *> fgMastersAllowed;  // list of master (domains) allowed
   static std::list<XrdProofdPriority *> fgPriorities;  // list of {users, priority change}
   static char                  *fgSuperUsers;  // ':' separated list of privileged users
   //
   static bool                   fgWorkerUsrCfg; // user cfg files enabled / disabled
   //
   static int                    fgReadWait;
   static int                    fgInternalWait; // Timeout on replies from proofsrv
   //
   static int                    fgShutdownOpt; // What to do when a client disconnects
   static int                    fgShutdownDelay; // Delay shutdown by this (if enabled)
   // 
   static int                    fgCron; // Cron thread option [1 ==> start]
   static int                    fgCronFrequency; // Frequency for running cron checks in secs
   //
   static int                    fgOperationMode; // Operation mode
   static XrdOucString           fgAllowedUsers; // Users allowed in controlled mode
   //
   static XrdOucString           fgProofServEnvs; // Additional envs to be exported before proofserv
   static XrdOucString           fgProofServRCs; // Additional rcs to be passed to proofserv
   //
   static XrdProofGroupMgr       fgGroupsMgr; // Groups manager
   //
   static XrdProofdManager       fgMgr;       // Cluster manager
   //
   static XrdProofSched         *fgProofSched;   // Instance of the PROOF scheduler
   //
   // Worker level scheduling control
   static float                  fgOverallInflate; // Overall inflate factor
   static int                    fgSchedOpt;  // Worker sched option
   //
   // Static area: client section
   //
   static std::list<XrdProofdClient *> fgProofdClients;  // keeps track of all users
   static std::list<XrdProofdPInfo *> fgTerminatedProcess; // List of pids of processes terminating
   static std::list<XrdProofServProxy *> fgActiveSessions; // List of active sessions (non-idle)

   //
   // Static area: methods
   //
   static bool   CheckMaster(const char *m);
   static int    CheckUser(const char *usr, XrdProofUI &ui, XrdOucString &e);
   static int    Config(const char *fn);
   static char  *FilterSecConfig(const char *cfn, int &nd);
   static int    GetWorkers(XrdOucString &workers, XrdProofServProxy *);
   static XrdProofSched *XrdProofdProtocol::LoadScheduler(const char *cfn, XrdOucError *edest);
   static XrdSecService *LoadSecurity(const char *seclib, const char *cfn);
   static int    LogTerminatedProc(int pid);
   static int    ResolveKeywords(XrdOucString &s, XrdProofdClient *pcl);
   static int    SetGroupEffectiveFractions();
   static int    SetInflateFactors();
   static int    SetProofServEnv(XrdROOT *r);
   static int    SaveAFSkey(XrdSecCredentials *c, const char *fn);
   static int    VerifyProcessByID(int pid, const char *pname = 0);
};

#endif
