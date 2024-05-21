// @(#)root/proofd:$Id$
// Author: G. Ganis Jan 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdProofServMgr
#define ROOT_XrdProofdProofServMgr

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdProofServMgr                                                  //
//                                                                      //
// Author: G. Ganis, CERN, 2008                                         //
//                                                                      //
// Class managing proofserv sessions manager.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <list>
#include <map>

#include "XpdSysPthread.h"

#include "XrdProofdXrdVers.h"
#ifndef ROOT_XrdFour
#  include <sys/types.h>
#  include <sys/socket.h>
#  include "XrdNet/XrdNetPeer.hh"
#else
#  include "XrdNet/XrdNetAddr.hh"
#endif
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucString.hh"

#include "XrdProofdConfig.h"
#include "XrdProofdProofServ.h"

class XrdOucStream;
class XrdProtocol_Config;
class XrdProofdManager;
class XrdROOTMgr;
class XrdSysLogger;

#define PSMMAXCNTS  3
#define PSMCNTOK(t) (t >= 0 && t < PSMMAXCNTS)

// Security handle
typedef int (*XrdSecCredsSaver_t)(XrdSecCredentials *, const char *fn, const XrdProofUI &ui);

// Aux structure for session set env inputs
typedef struct {
   XrdProofdProofServ *fPS;
   int          fLogLevel;
   XrdOucString fCfg;
   XrdOucString fLogFile;
   XrdOucString fSessionTag;
   XrdOucString fTopSessionTag;
   XrdOucString fSessionDir;
   XrdOucString fWrkDir;
   bool         fOld;
} ProofServEnv_t;

class XpdClientSessions {
public:
   XrdSysRecMutex   fMutex;
   XrdProofdClient *fClient;
   std::list<XrdProofdProofServ *> fProofServs;
   XpdClientSessions(XrdProofdClient *c = 0) : fClient(c) { }
   int operator==(const XpdClientSessions c) { return (c.fClient == fClient) ? 1 : 0; }
};

class XrdProofSessionInfo {
public:
   time_t         fLastAccess;
   int            fPid;
   int            fID;
   int            fSrvType;
   int            fPLiteNWrks;
   int            fStatus;
   XrdOucString   fUser;
   XrdOucString   fGroup;
   XrdOucString   fUnixPath;
   XrdOucString   fTag;
   XrdOucString   fAlias;
   XrdOucString   fLogFile;
   XrdOucString   fOrdinal;
   XrdOucString   fUserEnvs;
   XrdOucString   fROOTTag;
   XrdOucString   fAdminPath;
   int            fSrvProtVers;

   XrdProofSessionInfo(XrdProofdClient *c, XrdProofdProofServ *s);
   XrdProofSessionInfo(const char *file) { ReadFromFile(file); }

   void FillProofServ(XrdProofdProofServ &s, XrdROOTMgr *rmgr);
   int ReadFromFile(const char *file);
   void Reset();
   int SaveToFile(const char *file);
};

class XpdEnv {
public:
   XrdOucString   fName;
   XrdOucString   fEnv;
   XrdOucString   fUsers;
   XrdOucString   fGroups;
   int            fSvnMin;
   int            fSvnMax;
   int            fVerMin;
   int            fVerMax;
   XpdEnv(const char *n, const char *env, const char *usr = 0, const char *grp = 0,
          int smi = -1, int smx = -1, int vmi = -1, int vmx = -1) :
          fName(n), fEnv(env), fUsers(usr), fGroups(grp),
          fSvnMin(smi), fSvnMax(smx), fVerMin(vmi), fVerMax(vmx) { }
   void Reset(const char *n, const char *env, const char *usr = 0, const char *grp = 0,
              int smi = -1, int smx = -1, int vmi = -1, int vmx = -1) {
              fName = n; fEnv = env; fUsers = usr; fGroups = grp;
              fSvnMin = smi; fSvnMax = smx; fVerMin = vmi; fVerMax = vmx; }
   int Matches(const char *usr, const char *grp, int ver = -1);
   void Print(const char *what);
   static int     ToVersCode(int ver, bool hex = 0);
};

class XrdProofdProofServMgr : public XrdProofdConfig {

   XrdProofdManager  *fMgr;
   XrdSysRecMutex     fMutex;
   XrdSysRecMutex     fRecoverMutex;
   XrdSysRecMutex     fEnvsMutex;    // Protect usage of envs lists
   XrdSysSemWait      fForkSem;   // To serialize fork requests
   XrdSysSemWait      fProcessSem;   // To serialize process requests
   XrdSysLogger      *fLogger;    // Error logger
   int                fInternalWait;   // Timeout on replies from proofsrv
   XrdOucString       fProofPlugin;    // String identifying the plug-in to be loaded, e.g. "condor:"
   std::list<XpdEnv>  fProofServEnvs;  // Additional envs to be exported before proofserv
   std::list<XpdEnv>  fProofServRCs;   // Additional rcs to be passed to proofserv

   int                fShutdownOpt;    // What to do when a client disconnects
   int                fShutdownDelay;  // Delay shutdown by this (if enabled)

   XrdProofdPipe      fPipe;

   int                fCheckFrequency;
   int                fTerminationTimeOut;
   int                fVerifyTimeOut;
   int                fReconnectTime;
   int                fReconnectTimeOut;
   int                fRecoverTimeOut;
   int                fRecoverDeadline;
   bool               fCheckLost;
   bool               fUseFork;       // If true, use fork to start proofserv
   XrdOucString       fParentExecs;   // List of possible 'proofserv' parent names

   int                fCounters[PSMMAXCNTS];  // Internal counters (see enum PSMCounters)
   int                fCurrentSessions;       // Number of sessions (top masters)

   unsigned int       fSeqSessionN;   // Sequential number for sessions created by this instance

   int                fNextSessionsCheck; // Time of next sessions check

   XrdOucString       fActiAdminPath; // Active sessions admin area
   XrdOucString       fTermAdminPath; // Terminated sessions admin area

   XrdOucHash<XrdProofdProofServ> fSessions; // List of sessions
   std::list<XrdProofdProofServ *> fActiveSessions;     // List of active sessions (non-idle)
   std::list<XpdClientSessions *> *fRecoverClients; // List of client potentially recovering

   XrdSecCredsSaver_t fCredsSaver; // If defined, function to be used to save the credentials

   std::map<XrdProofdProtocol*,int> fDestroyTimes; // Tracks destroyed sessions

   int                DoDirectiveProofServMgr(char *, XrdOucStream *, bool);
   int                DoDirectivePutEnv(char *, XrdOucStream *, bool);
   int                DoDirectivePutRc(char *, XrdOucStream *, bool);
   int                DoDirectiveShutdown(char *, XrdOucStream *, bool);
   void               ExtractEnv(char *, XrdOucStream *,
                                 XrdOucString &users, XrdOucString &groups,
                                 XrdOucString &rcval, XrdOucString &rcnam,
                                 int &smi, int &smx, int &vmi, int &vmx, bool &hex);
   void               FillEnvList(std::list<XpdEnv> *el, const char *nam, const char *val,
                                  const char *usrs = 0, const char *grps = 0,
                                  int smi = -1, int smx = -1, int vmi = -1, int vmx = -1, bool hex = 0);
   unsigned int       GetSeqSessionN() { XrdSysMutexHelper mhp(fMutex); return ++fSeqSessionN; }

   int                CreateAdminPath(XrdProofdProofServ *xps,
                                      XrdProofdProtocol *p, int pid, XrdOucString &emsg);
   int                CreateSockPath(XrdProofdProofServ *xps, XrdProofdProtocol *p,
                                     unsigned int seq, XrdOucString &emsg);
//   int                CreateFork(XrdProofdProtocol *p);
   int                CreateProofServEnvFile(XrdProofdProtocol *p,
                                            void *input, const char *envfn, const char *rcfn);
   int                CreateProofServRootRc(XrdProofdProtocol *p,
                                            void *input, const char *rcfn);
#ifndef ROOT_XrdFour
   int                SetupProtocol(XrdNetPeer &peerpsrv,
#else
   int                SetupProtocol(XrdNetAddr &netaddr,
#endif
                                    XrdProofdProofServ *xps, XrdOucString &e);
   void               ParseCreateBuffer(XrdProofdProtocol *p,  XrdProofdProofServ *xps,
                                        XrdOucString &tag, XrdOucString &ord,
                                        XrdOucString &cffile, XrdOucString &uenvs,
                                        int &intwait);
   XrdProofdProofServ *PrepareProofServ(XrdProofdProtocol *p,
                                        XrdProofdResponse *r, unsigned short &sid);
   int                PrepareSessionRecovering();
   int                ResolveSession(const char *fpid);

   void               SendErrLog(const char *errlog, XrdProofdResponse *r);

   // Session Admin path management
   int                AddSession(XrdProofdProtocol *p, XrdProofdProofServ *s);
   bool               IsSessionSocket(const char *fpid);
   int                RmSession(const char *fpid);
   int                TouchSession(const char *fpid, const char *path = 0);
   int                VerifySession(const char *fpid, int to = -1, const char *path = 0);

   void               ResolveKeywords(XrdOucString &s, ProofServEnv_t *in);
   int                SetUserOwnerships(XrdProofdProtocol *p, const char *ord, const char *stag);

public:
   XrdProofdProofServMgr(XrdProofdManager *mgr, XrdProtocol_Config *pi, XrdSysError *e);
   virtual ~XrdProofdProofServMgr() { }

   enum PSMProtocol { kSessionRemoval = 0, kClientDisconnect = 1, kCleanSessions = 2, kProcessReq = 3, kChgSessionSt = 4} ;
   enum PSMCounters { kCreateCnt = 0, kCleanSessionsCnt = 1, kProcessCnt = 2} ;

   XrdSysRecMutex   *Mutex() { return &fMutex; }

   int               Config(bool rcf = 0);
   int               DoDirective(XrdProofdDirective *d,
                                 char *val, XrdOucStream *cfg, bool rcf);
   void              RegisterDirectives();

   int               CheckFrequency() const { return fCheckFrequency; }
   int               InternalWait() const { return fInternalWait; }
   int               VerifyTimeOut() const { return fVerifyTimeOut; }

   inline int        NextSessionsCheck()
                        { XrdSysMutexHelper mhp(fMutex); return fNextSessionsCheck; }
   inline void       SetNextSessionsCheck(int t)
                        { XrdSysMutexHelper mhp(fMutex); fNextSessionsCheck = t; }

   bool              IsReconnecting();
   bool              IsClientRecovering(const char *usr, const char *grp, int &deadline);
   void              SetReconnectTime(bool on = 1);

   bool              Alive(XrdProofdProtocol *p);

   int               Process(XrdProofdProtocol *p);
   XrdSysSemWait    *ProcessSem() { return &fProcessSem; }

   int               AcceptPeer(XrdProofdProofServ *xps, int to, XrdOucString &e);
   int               Attach(XrdProofdProtocol *p);
   int               Create(XrdProofdProtocol *p);
   int               Destroy(XrdProofdProtocol *p);
   int               Detach(XrdProofdProtocol *p);
   int               Recover(XpdClientSessions *cl);

   void              UpdateCounter(int t, int n) { if (PSMCNTOK(t)) {
                                 XrdSysMutexHelper mhp(fMutex); fCounters[t] += n;
                                          if (fCounters[t] < 0) fCounters[t] = 0;} }
   int               CheckCounter(int t) { int cnt = -1; if (PSMCNTOK(t)) {
                                 XrdSysMutexHelper mhp(fMutex); cnt = fCounters[t];}
                                 return cnt; }

   void              BroadcastClusterInfo();
   int               BroadcastPriorities();
   int               CurrentSessions(bool recalculate = 0);
   void              DisconnectFromProofServ(int pid);

   std::list<XrdProofdProofServ *> *ActiveSessions() { return &fActiveSessions; }
   XrdProofdProofServ *GetActiveSession(int pid);

   int               CleanupProofServ(bool all = 0, const char *usr = 0);

   void              FormFileNameInSessionDir(XrdProofdProtocol *p,
                                              XrdProofdProofServ *xps,
                                              const char *sessiondir,
                                              const char *extension,
                                              XrdOucString &outfn);

   void              GetTagDirs(int opt, XrdProofdProtocol *p, XrdProofdProofServ *xps,
                                XrdOucString &sesstag, XrdOucString &topsesstag,
                                XrdOucString &sessiondir, XrdOucString &sesswrkdir);

   int               SetProofServEnv(XrdProofdProtocol *p, void *in);
   int               SetProofServEnvOld(XrdProofdProtocol *p, void *in);
   int               SetUserEnvironment(XrdProofdProtocol *p);

   static int        SetProofServEnv(XrdProofdManager *m, XrdROOT *r);

   inline XrdProofdPipe *Pipe() { return &fPipe; }

   // Checks run periodically by the cron job
   int               DeleteFromSessions(const char *pid);
   int               MvSession(const char *fpid);
   int               CheckActiveSessions(bool verify = 1);
   int               CheckTerminatedSessions();
   int               CleanClientSessions(const char *usr, int srvtype);
   int               CleanupLostProofServ();
   int               RecoverActiveSessions();
};

class XpdSrvMgrCreateCnt {
public:
   int                    fType;
   XrdProofdProofServMgr *fMgr;
   XpdSrvMgrCreateCnt(XrdProofdProofServMgr *m, int t) : fType(t), fMgr(m)
                                        { if (m && PSMCNTOK(t)) m->UpdateCounter(t,1); }
   ~XpdSrvMgrCreateCnt() { if (fMgr && PSMCNTOK(fType)) fMgr->UpdateCounter(fType,-1); }
};

class XpdSrvMgrCreateGuard {
public:
   int *fCnt;
   XpdSrvMgrCreateGuard(int *c = 0) { Set(c); }
   ~XpdSrvMgrCreateGuard() { if (fCnt) (*fCnt)--; }
   void Set(int *c) { fCnt = c; if (fCnt) (*fCnt)++;}
};

#endif
