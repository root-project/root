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
// Class mapping manager functionality.                                 //
// On masters it keeps info about the available worker nodes and allows //
// communication with them. In particular, it reads the proof.conf file //
// when working with static resources.                                  //
// On workers it handles the communication with the master              //
// (to be implemented).                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <list>

#include "XpdSysPthread.h"

#include "XrdOuc/XrdOucString.hh"

#include "XrdProofdAux.h"
#include "XrdProofdConfig.h"

class rpdunixsrv;
class XrdProofdAdmin;
class XrdProofdClient;
class XrdProofdClientMgr;
class XrdProofdNetMgr;
class XrdProofdPriorityMgr;
class XrdProofdProofServMgr;
class XrdProofdProtocol;
class XrdProofGroupMgr;
class XrdProofSched;
class XrdProofdProofServ;
class XrdProofWorker;
class XrdProtocol;
class XrdROOT;
class XrdROOTMgr;

class XrdSysPlugin;

class XrdProofdManager : public XrdProofdConfig {

 public:
   XrdProofdManager(char *parms, XrdProtocol_Config *pi, XrdSysError *edest);
   virtual ~XrdProofdManager();

   XrdSysRecMutex   *Mutex() { return &fMutex; }

   // Config
   int               Config(bool rcf = 0);
   int               DoDirective(XrdProofdDirective *d,
                                 char *val, XrdOucStream *cfg, bool rcf);
   void              RegisterDirectives();

   int               ResolveKeywords(XrdOucString &s, XrdProofdClient *pcl);

   int               GetWorkers(XrdOucString &workers, XrdProofdProofServ *, const char *);

   const char       *AdminPath() const { return fAdminPath.c_str(); }
   const char       *BareLibPath() const { return fBareLibPath.c_str(); }
   bool              ChangeOwn() const { return fChangeOwn; }
   void              CheckLogFileOwnership();
   bool              CheckMaster(const char *m);
   int               CheckUser(const char *usr, const char *grp, XrdProofUI &ui, XrdOucString &e, bool &su);
   int               CronFrequency() { return fCronFrequency; }
   const char       *Host() const { return fHost.c_str(); }
   const char       *Image() const { return fImage.c_str(); }
   bool              IsSuperMst() const { return fSuperMst; }
   const char       *LocalROOT() const { return fLocalroot.c_str(); }
   bool              MultiUser() const { return fMultiUser; }
   const char       *NameSpace() const { return fNamespace.c_str(); }
   const char       *PoolURL() const { return fPoolURL.c_str(); }
   int               Port() const { return fPort; }
   int               SrvType() const { return fSrvType; }
   const char       *SockPathDir() const { return fSockPathDir.c_str(); }
   const char       *TMPdir() const { return fTMPdir.c_str(); }
   const char       *WorkDir() const { return fWorkDir.c_str(); }
   const char       *DataDir() const { return fDataDir.c_str(); }
   const char       *DataDirOpts() const { return fDataDirOpts.c_str(); }
   const char       *DataDirUrlOpts() const { return fDataDirUrlOpts.c_str(); }
   const char       *DataSetExp() const { return fDataSetExp.c_str(); }
   const char       *StageReqRepo() const { return fStageReqRepo.c_str(); }

   bool              RemotePLite() const { return fRemotePLite; }

   std::list<XrdProofdDSInfo *> *DataSetSrcs() { return &fDataSetSrcs; }

   // Services
   XrdProofdClientMgr *ClientMgr() const { return fClientMgr; }
   const char       *EffectiveUser() const { return fEffectiveUser.c_str(); }
   XrdProofGroupMgr *GroupsMgr() const { return fGroupsMgr; }
   XrdProofSched    *ProofSched() const { return fProofSched; }
   XrdProofdProofServMgr *SessionMgr() const { return fSessionMgr; }
   XrdProofdNetMgr  *NetMgr() const { return fNetMgr; }
   XrdProofdAdmin   *Admin() const { return fAdmin; }
   XrdROOTMgr       *ROOTMgr() const { return fROOTMgr; }
   XrdProofdPriorityMgr *PriorityMgr() const { return fPriorityMgr; }
   XrdScheduler     *Sched() const { return fSched; }

   XrdProtocol      *Xrootd() const { return fXrootd; }

   // Request processor
   int               Process(XrdProofdProtocol *p);

 private:
   XrdSysRecMutex    fMutex;          // Atomize this instance

   bool              fSuperMst;       // true if this node is a SuperMst
   bool              fRemotePLite;    // true if remote PLite mode is allowed

   XrdOucString      fAdminPath;      // Path to the PROOF admin area

   int               fSrvType;        // Master, Submaster, Worker or any
   XrdOucString      fEffectiveUser;  // Effective user
   XrdOucString      fHost;           // local host name
   int               fPort;           // Port for client-like connections
   XrdOucString      fImage;          // image name for these servers
   XrdOucString      fWorkDir;        // working dir for these servers
   XrdOucString      fMUWorkDir;      // template working dir in multi-user mode
   int               fCronFrequency;  // Frequency of cron checks

   XrdOucString      fBareLibPath;    // LIBPATH cleaned from ROOT dists
   XrdOucString      fSockPathDir;    // directory for Unix sockets
   XrdOucString      fTMPdir;         // directory for temporary files
   XrdOucString      fPoolURL;        // Local pool URL
   XrdOucString      fNamespace;      // Local pool namespace
   XrdOucString      fLocalroot;      // Local root prefix (directive oss.localroot)
   XrdOucString      fDataDir;        // Directory under which to create the sub-dirs for users data
   XrdOucString      fDataDirOpts;    // String specifying options for fDataDir handling
   XrdOucString      fDataDirUrlOpts; // String specifying URL type options for fDataDir
   XrdOucString      fDataSetExp;     // List of local dataset repositories to be asserted
   XrdOucString      fStageReqRepo;   // Directive for staging requests

   XrdProtocol      *fXrootd;         // Reference instance of XrdXrootdProtocol 
   XrdOucString      fXrootdLibPath;  // Path to 'xrootd' plug-in
   XrdSysPlugin     *fXrootdPlugin;   // 'xrootd' plug-in handler

   // Services
   XrdProofdClientMgr    *fClientMgr;  // Client manager
   XrdProofGroupMgr      *fGroupsMgr;  // Groups manager
   XrdProofSched         *fProofSched; // Instance of the PROOF scheduler
   XrdProofdProofServMgr *fSessionMgr; // Proof session manager
   XrdProofdNetMgr       *fNetMgr;     // Proof network manager
   XrdProofdAdmin        *fAdmin;      // Admin services
   XrdROOTMgr            *fROOTMgr;    // ROOT versions manager
   XrdProofdPriorityMgr  *fPriorityMgr;// Priority manager

   XrdScheduler          *fSched;      // System scheduler

   XrdOucString      fSuperUsers;     // ':' separated list of privileged users
   //
   int               fOperationMode;  // Operation mode
   XrdOucHash<int>   fAllowedUsers;   // UNIX users allowed in controlled mode
   XrdOucHash<int>   fAllowedGroups;  // UNIX groups allowed in controlled mode
   bool              fMultiUser;      // Allow/disallow multi-user mode
   bool              fChangeOwn;      // TRUE is ownership has to be changed

   // Lib paths for proofserv
   bool              fRemoveROOTLibPaths; // If true the existing ROOT lib paths are removed
   XrdOucHash<XrdOucString> fLibPathsToRemove;  // Additional paths to be removed

   //
   // Lists
   std::list<XrdOucString *> fMastersAllowed; // list of master (domains) allowed
   std::list<XrdProofdDSInfo *> fDataSetSrcs; // sources of dataset info

   // Temporary storage: not to be trusted after construction
   char               *fParms; 
   XrdProtocol_Config *fPi;


   int               DoDirectiveAllow(char *, XrdOucStream *, bool);
   int               DoDirectiveAllowedGroups(char *, XrdOucStream *, bool);
   int               DoDirectiveAllowedUsers(char *, XrdOucStream *, bool);
   int               DoDirectiveDataDir(char *, XrdOucStream *, bool);
   int               DoDirectiveDataSetSrc(char *, XrdOucStream *, bool);
   int               DoDirectiveDataSetReqRepo(char *, XrdOucStream *, bool);
   int               DoDirectiveFilterLibPaths(char *, XrdOucStream *, bool);
   int               DoDirectiveGroupfile(char *, XrdOucStream *, bool);
   int               DoDirectiveMaxOldLogs(char *, XrdOucStream *, bool);
   int               DoDirectiveMultiUser(char *, XrdOucStream *, bool);
   int               DoDirectivePort(char *, XrdOucStream *, bool);
   int               DoDirectiveRole(char *, XrdOucStream *, bool);
   int               DoDirectiveRootd(char *, XrdOucStream *, bool);
   int               DoDirectiveRootdAllow(char *, XrdOucStream *, bool);
   int               DoDirectiveTrace(char *, XrdOucStream *, bool);
   int               DoDirectiveXrootd(char *, XrdOucStream *, bool);

   bool              ValidateLocalDataSetSrc(XrdOucString &url, bool &local);

   // Load services
   XrdProofSched    *LoadScheduler();
   XrdProtocol      *LoadXrootd(char *parms, XrdProtocol_Config *pi, XrdSysError *edest);
};

// Aux structures
typedef struct {
   XrdProofdClientMgr    *fClientMgr;
   XrdProofdProofServMgr *fSessionMgr;
   XrdProofSched         *fProofSched;
} XpdManagerCron_t;

#endif
