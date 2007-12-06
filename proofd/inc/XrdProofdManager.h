// @(#)root/proofd:$Id$
// Author: G. Ganis June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdManager
#define ROOT_XrdProofdManager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdManager                                                     //
//                                                                      //
// Author: G. Ganis, CERN, 2007                                         //
//                                                                      //
// Class mapping manager fonctionality.                                 //
// On masters it keeps info about the available worker nodes and allows //
// communication with them. In particular, it reads the proof.conf file //
// when working with static resources.                                  //
// On workers it handles the communication with the master              //
// (to be implemented).                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <list>

#ifdef OLDXRDOUC
#  include "XrdSysToOuc.h"
#  include "XrdOuc/XrdOucPthread.hh"
#else
#  include "XrdSys/XrdSysPthread.hh"
#endif
#include "XrdProofdAux.h"
#include "XrdProofConn.h"
#include "XrdProofGroup.h"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSec/XrdSecInterface.hh"

class XrdClientMessage;
class XrdProofdClient;
class XrdProofWorker;
class XrdProofdResponse;
class XrdProofServProxy;
class XrdROOT;
class XrdProofSched;

class XrdProofdManager {

 public:
   XrdProofdManager();
   virtual ~XrdProofdManager();

   XrdSysRecMutex   *Mutex() { return &fMutex; }

   // Config
   int               Config(const char *fn, bool rcf = 0, XrdSysError *e = 0);
   int               ProcessDirective(XrdProofdDirective *d,
                                      char *val, XrdOucStream *cfg, bool rcf);
   int               ResolveKeywords(XrdOucString &s, XrdProofdClient *pcl);

   // List of available and unique workers (on master only)
   std::list<XrdProofWorker *> *GetActiveWorkers();
   std::list<XrdProofWorker *> *GetNodes();
   int               GetWorkers(XrdOucString &workers, XrdProofServProxy *);

   // Keeping track of active sessions
   void              AddActiveSession(XrdProofServProxy *p) { XrdSysMutexHelper mhp(&fMutex);
                                                              fActiveSessions.push_back(p); }
   XrdProofServProxy *GetActiveSession(int pid);
   std::list<XrdProofServProxy *> *GetActiveSessions() { XrdSysMutexHelper mhp(&fMutex);
                                                         return &fActiveSessions; }
   std::list<XrdProofdClient *> *ProofdClients() { return &fProofdClients; }
   void              RemoveActiveSession(XrdProofServProxy *p) { XrdSysMutexHelper mhp(&fMutex);
                                                                 fActiveSessions.remove(p); }

   // Running
   int               Cron() const { return fCron; }
   int               CronFrequency() const { return fCronFrequency; }
   int               LogTerminatedProc(int pid);
   int               OperationMode() const { return fOperationMode; }
   int               ShutdownDelay() const { return fShutdownDelay; }
   int               ShutdownOpt() const { return fShutdownOpt; }
   int               TrimTerminatedProcesses();
   int               VerifyProcessByID(int pid, const char *pname = 0);

   // Authorization control
   const char       *AllowedUsers() const { return fAllowedUsers.c_str(); }
   bool              ChangeOwn() const { return fChangeOwn; }
   bool              CheckMaster(const char *m);
   int               CheckUser(const char *usr, XrdProofUI &ui, XrdOucString &e);
   std::list<XrdOucString *> *MastersAllowed() { XrdSysMutexHelper mhp(&fMutex); return &fMastersAllowed; }
   bool              MultiUser() const { return fMultiUser; }
   const char       *SuperUsers() { XrdSysMutexHelper mhp(&fMutex); return fSuperUsers.c_str(); }
   bool              WorkerUsrCfg() const { return fWorkerUsrCfg; }

   // Node properties
   const char       *BareLibPath() const { return fBareLibPath.c_str(); }
   const char       *DataSetDir() const { return fDataSetDir.c_str(); }
   const char       *EffectiveUser() const { return fEffectiveUser.c_str(); }
   const char       *Host() const { return fHost.c_str(); }
   const char       *Image() const { return fImage.c_str(); }
   bool              IsSuperMst() const { return fSuperMst; }
   const char       *LocalROOT() const { return fLocalroot.c_str(); }
   const char       *NameSpace() const { return fNamespace.c_str(); }
   const char       *PoolURL() const { return fPoolURL.c_str(); }
   int               Port() const { return fPort; }
   const char       *PROOFcfg() const { return fPROOFcfg.fName.c_str(); }
   int               ResourceType() const { return fResourceType; }
   std::list<XrdROOT *> *ROOT() { return &fROOT; }
   const char       *SecLib() const { return fSecLib.c_str(); }
   int               SrvType() const { return fSrvType; }
   const char       *TMPdir() const { return fTMPdir.c_str(); }
   const char       *WorkDir() const { return fWorkDir.c_str(); }

   // Services
   XrdSecService    *CIA() const { return fCIA; }
   XrdProofGroupMgr *GroupsMgr() const { return fGroupsMgr; }
   XrdProofSched    *ProofSched() const { return fProofSched; }

   // Scheduling
   bool              IsSchedOn() { XrdSysMutexHelper mhp(&fMutex);
                                   return ((fSchedOpt != kXPD_sched_off) ? 1 : 0); }
   float             OverallInflate() { XrdSysMutexHelper mhp(&fMutex); return fOverallInflate; }
   std::list<XrdProofdPriority *> *Priorities() { XrdSysMutexHelper mhp(&fMutex); return &fPriorities; }
   int               PriorityMax() { XrdSysMutexHelper mhp(&fMutex); return fPriorityMax; }
   int               PriorityMin() { XrdSysMutexHelper mhp(&fMutex); return fPriorityMin; }
   int               SchedOpt() { XrdSysMutexHelper mhp(&fMutex); return fSchedOpt; }
   int               SetGroupEffectiveFractions();
   int               SetInflateFactors();
   int               SetNiceValues(int opt = 0);
   void              SetSchedOpt(int opt) { XrdSysMutexHelper mhp(&fMutex);
                                            fSchedOpt = opt; }
   int               UpdatePriorities(bool forceupdate = 0);

   // This part may evolve in the future due to better understanding of
   // how resource brokering will work; for the time being we just move in
   // here the functionality we have now
   int               Broadcast(int type, const char *msg, XrdProofdResponse *r, bool notify = 0);
   XrdProofConn     *GetProofConn(const char *url);
   XrdClientMessage *Send(const char *url, int type,
                          const char *msg, int srvtype, XrdProofdResponse *r, bool notify = 0);

 private:
   XrdSysRecMutex    fMutex;          // Atomize this instance
   XrdSysError      *fEDest;          // Error message handler

   XrdProofdFile     fCfgFile;        // Configuration file
   bool              fSuperMst;       // true if this node is a SuperMst

   int               fSrvType;        // Master, Submaster, Worker or any
   XrdOucString      fEffectiveUser;  // Effective user
   XrdOucString      fHost;           // local host name
   int               fPort;           // Port for client-like connections
   XrdOucString      fImage;          // image name for these servers
   XrdOucString      fWorkDir;        // working dir for these servers
   XrdOucString      fDataSetDir;     // dataset dir for this master server
   int               fNumLocalWrks;   // Number of workers to be started locally
   int               fResourceType;   // resource type
   XrdProofdFile     fPROOFcfg;       // PROOF static configuration

   XrdOucString      fBareLibPath;    // LIBPATH cleaned from ROOT dists
   XrdOucString      fTMPdir;         // directory for temporary files
   XrdOucString      fSecLib;
   XrdOucString      fPoolURL;        // Local pool URL
   XrdOucString      fNamespace;      // Local pool namespace
   XrdOucString      fLocalroot;      // Local root prefix (directive oss.localroot)

   int               fRequestTO;      // Timeout on broadcast request

   // Services
   XrdSecService    *fCIA;            // Authentication Server
   XrdProofGroupMgr *fGroupsMgr;      // Groups manager
   XrdProofSched    *fProofSched;     // Instance of the PROOF scheduler

   XrdOucString      fSuperUsers;     // ':' separated list of privileged users
   //
   bool              fWorkerUsrCfg;   // user cfg files enabled / disabled

   int               fShutdownOpt;    // What to do when a client disconnects
   int               fShutdownDelay;  // Delay shutdown by this (if enabled)
   //
   int               fCron;           // Cron thread option [1 ==> start]
   int               fCronFrequency;  // Frequency for running cron checks in secs
   //
   int               fOperationMode;  // Operation mode
   XrdOucString      fAllowedUsers;   // Users allowed in controlled mode
   bool              fMultiUser;      // Allow/disallow multi-user mode
   bool              fChangeOwn;      // TRUE is ownership has to be changed
   //
   // Scheduling related
   float             fOverallInflate; // Overall inflate factor
   int               fSchedOpt;       // Worker sched option
   int               fPriorityMax;    // Max session priority [1,40], [20]
   int               fPriorityMin;    // Min session priority [1,40], [1]

   //
   // Lists
   std::list<XrdProofdClient *> fProofdClients;        // keeps track of all users
   std::list<XrdProofdPInfo *> fTerminatedProcess;     // List of pids of processes terminating
   std::list<XrdProofWorker *> fWorkers;               // List of possible workers
   std::list<XrdProofWorker *> fNodes;                 // List of worker unique nodes
   std::list<XrdProofServProxy *> fActiveSessions;     // List of active sessions (non-idle)
   std::list<XrdOucString *> fMastersAllowed;          // list of master (domains) allowed
   std::list<XrdProofdPriority *> fPriorities;         // list of {users, priority change}
   std::list<XrdROOT *> fROOT;                         // ROOT versions; the first is the default
   XrdOucHash<XrdProofConn> fProofConnHash;            // Available connections
   XrdOucHash<XrdProofdDirective> fConfigDirectives;   // Config directives
   XrdOucHash<XrdProofdDirective> fReConfigDirectives; // Re-configurable directives

   void              CreateDefaultPROOFcfg();
   int               ReadPROOFcfg();

   // Config methods
   int               ParseConfig(XrdProofUI ui, bool rcf = 0);
   void              RegisterConfigDirectives();
   void              RegisterReConfigDirectives();

   int               DoDirectiveAdminReqTO(char *, XrdOucStream *, bool);
   int               DoDirectiveAllow(char *, XrdOucStream *, bool);
   int               DoDirectiveAllowedUsers(char *, XrdOucStream *, bool);
   int               DoDirectiveCron(char *, XrdOucStream *, bool);
   int               DoDirectiveGroupfile(char *, XrdOucStream *, bool);
   int               DoDirectiveMaxOldLogs(char *, XrdOucStream *, bool);
   int               DoDirectiveMultiUser(char *, XrdOucStream *, bool);
   int               DoDirectivePort(char *, XrdOucStream *, bool);
   int               DoDirectivePriority(char *, XrdOucStream *, bool);
   int               DoDirectiveResource(char *, XrdOucStream *, bool);
   int               DoDirectiveRole(char *, XrdOucStream *, bool);
   int               DoDirectiveRootSys(char *, XrdOucStream *, bool);
   int               DoDirectiveSchedOpt(char *, XrdOucStream *, bool);
   int               DoDirectiveShutdown(char *, XrdOucStream *, bool);

   // Security service
   XrdSecService    *LoadSecurity();
   char             *FilterSecConfig(int &nd);
   // Scheduling service
   XrdProofSched    *LoadScheduler();

};

#endif
