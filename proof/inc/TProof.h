// @(#)root/proof:$Name:  $:$Id: TProof.h,v 1.116 2007/06/22 17:16:35 ganis Exp $
// Author: Fons Rademakers   13/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProof
#define ROOT_TProof


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProof                                                               //
//                                                                      //
// This class controls a Parallel ROOT Facility, PROOF, cluster.        //
// It fires the slave servers, it keeps track of how many slaves are    //
// running, it keeps track of the slaves running status, it broadcasts  //
// messages to all slaves, it collects results, etc.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TProof
#include "TProof.h"
#endif
#ifndef ROOT_TProofMgr
#include "TProofMgr.h"
#endif
#ifndef ROOT_TProofDebug
#include "TProofDebug.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_MessageTypes
#include "MessageTypes.h"
#endif
#ifndef ROOT_TMD5
#include "TMD5.h"
#endif
#ifndef ROOT_TSysEvtHandler
#include "TSysEvtHandler.h"
#endif
#ifndef ROOT_TThread
#include "TThread.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif

#include <map>

#ifdef R__GLOBALSTL
namespace std { using ::map; }
#endif

#define CANNOTUSE(x) Info(x,"Not manager: cannot use this method")

class TChain;
class TCondor;
class TCondorSlave;
class TDrawFeedback;
class TDSet;
class TEventList;
class THashList;
class TList;
class TCollection;
class TMessage;
class TMonitor;
class TPluginHandler;
class TProof;
class TProofInputHandler;
class TProofInterruptHandler;
class TProofLockPath;
class TVirtualProofPlayer;
class TProofPlayer;
class TProofPlayerRemote;
class TProofProgressDialog;
class TProofServ;
class TQueryResult;
class TSignalHandler;
class TSlave;
class TSemaphore;
class TSocket;
class TTree;
class TVirtualMutex;

// protocol changes:
// 1 -> 2: new arguments for Process() command, option added
// 2 -> 3: package manager enabling protocol changed
// 3 -> 4: introduction of multi-level-master support
// 4 -> 5: added friends support
// 5 -> 6: drop TFTP, support for asynchronous queries
// 6 -> 7: support for multisessions, archieve, retrieve, ...
// 7 -> 8: return number of entries in GetNextPacket
// 8 -> 9: support for stateless connection via xproofd
// 9 -> 10: new features requested, tested at CAF
// 10 -> 11: new merging strategy
// 11 -> 12: new progress message
// 12 -> 13: exchange version/architecture/compiler info
// 13 -> 14: new proofserv environment setting

// PROOF magic constants
const Int_t       kPROOF_Protocol        = 14;            // protocol version number
const Int_t       kPROOF_Port            = 1093;          // IANA registered PROOF port
const char* const kPROOF_ConfFile        = "proof.conf";  // default config file
const char* const kPROOF_ConfDir         = "/usr/local/root";  // default config dir
const char* const kPROOF_WorkDir         = "~/proof";     // default working directory
const char* const kPROOF_CacheDir        = "cache";       // file cache dir, under WorkDir
const char* const kPROOF_PackDir         = "packages";    // package dir, under WorkDir
const char* const kPROOF_QueryDir        = "queries";     // query dir, under WorkDir
const char* const kPROOF_DataSetDir      = "datasets";    // dataset dir, under WorkDir
const char* const kPROOF_CacheLockFile   = "/tmp/proof-cache-lock-";   // cache lock file
const char* const kPROOF_PackageLockFile = "/tmp/proof-package-lock-"; // package lock file
const char* const kPROOF_QueryLockFile   = "/tmp/proof-query-lock-";   // query lock file
const char* const kPROOF_DataSetLockFile = "/tmp/proof-dataset-lock-"; // dataset lock file

#ifndef R__WIN32
const char* const kCP     = "/bin/cp -fp";
const char* const kRM     = "/bin/rm -rf";
const char* const kLS     = "/bin/ls -l";
const char* const kUNTAR  = "%s -c %s/%s | (cd %s; tar xf -)";
const char* const kUNTAR2 = "%s -c %s | (cd %s; tar xf -)";
const char* const kGUNZIP = "gunzip";
#else
const char* const kCP     = "copy";
const char* const kRM     = "delete";
const char* const kLS     = "dir";
const char* const kUNTAR  = "...";
const char* const kUNTAR2 = "...";
const char* const kGUNZIP = "gunzip";
#endif

R__EXTERN TVirtualMutex *gProofMutex;


// Helper classes used for parallel startup
class TProofThreadArg {
public:
   TUrl         *fUrl;
   TString       fOrd;
   Int_t         fPerf;
   TString       fImage;
   TString       fWorkdir;
   TString       fMsd;
   TList        *fSlaves;
   TProof       *fProof;
   TCondorSlave *fCslave;
   TList        *fClaims;
   Int_t         fType;

   TProofThreadArg(const char *h, Int_t po, const char *o, Int_t pe,
                   const char *i, const char *w,
                   TList *s, TProof *prf);

   TProofThreadArg(TCondorSlave *csl, TList *clist,
                   TList *s, TProof *prf);

   TProofThreadArg(const char *h, Int_t po, const char *o,
                   const char *i, const char *w, const char *m,
                   TList *s, TProof *prf);

   virtual ~TProofThreadArg() { if (fUrl) delete fUrl; }
};

// PROOF Thread class for parallel startup
class TProofThread {
public:
   TThread         *fThread;
   TProofThreadArg *fArgs;

   TProofThread(TThread *t, TProofThreadArg *a) { fThread = t; fArgs = a; }
   virtual ~TProofThread() { SafeDelete(fThread); SafeDelete(fArgs); }
};

// PROOF Interrupt signal handler
class TProofInterruptHandler : public TSignalHandler {
private:
   TProof *fProof;
public:
   TProofInterruptHandler(TProof *p)
      : TSignalHandler(kSigInterrupt, kFALSE), fProof(p) { }
   Bool_t Notify();
};

// Input handler for messages from TProofServ
class TProofInputHandler : public TFileHandler {
private:
   TSocket *fSocket;
   TProof  *fProof;
public:
   TProofInputHandler(TProof *p, TSocket *s);
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

// Slaves info class
class TSlaveInfo : public TObject {
public:
   enum ESlaveStatus { kActive, kNotActive, kBad };

   TString      fOrdinal;      //slave ordinal
   TString      fHostName;     //hostname this slave is running on
   TString      fMsd;          //mass storage domain slave is in
   Int_t        fPerfIndex;    //relative performance of this slave
   ESlaveStatus fStatus;       //slave status

   TSlaveInfo(const char *ordinal = "", const char *host = "", Int_t perfidx = 0,
              const char *msd = "") :
              fOrdinal(ordinal), fHostName(host), fMsd(msd),
              fPerfIndex(perfidx), fStatus(kNotActive) { }

   const char *GetMsd() const { return fMsd; }
   const char *GetName() const { return fHostName; }
   const char *GetOrdinal() const { return fOrdinal; }
   void        SetStatus(ESlaveStatus stat) { fStatus = stat; }

   Int_t  Compare(const TObject *obj) const;
   Bool_t IsSortable() const { return kTRUE; }
   void   Print(Option_t *option="") const;

   ClassDef(TSlaveInfo,2) //basic info on slave
};

class TProof : public TNamed, public TQObject {

friend class TPacketizer;
friend class TPacketizerDev;
friend class TAdaptivePacketizer;
friend class TProofServ;
friend class TProofInputHandler;
friend class TProofInterruptHandler;
friend class TProofPlayer;
friend class TProofPlayerRemote;
friend class TProofProgressDialog;
friend class TSlave;
friend class TXSlave;
friend class TXSocket;        // to access kPing
friend class TXSocketHandler; // to access fCurrentMonitor and CollectInputFrom
friend class TXProofMgr;      // to access EUrgent
friend class TXProofServ;     // to access EUrgent

public:
   // PROOF status bits
   enum EStatusBits {
      kUsingSessionGui     = BIT(14)
   };
   enum EQueryMode {
      kSync                = 0,
      kAsync               = 1
   };
   enum EUploadOpt {
      kAppend              = 0x1,
      kOverwriteDataSet    = 0x2,
      kNoOverwriteDataSet  = 0x4,
      kOverwriteAllFiles   = 0x8,
      kOverwriteNoFiles    = 0x10,
      kAskUser             = 0x0
   };
   enum EUploadDataSetAnswer {
      kError               = -1,
      kDataSetExists       = -2
   };
   enum EUploadPackageOpt {
      kUntar               = 0x0,  //Untar over existing dir [default]
      kRemoveOld           = 0x1   //Remove existing dir with same name
   };

private:
   enum EUrgent {
      kLocalInterrupt      = -1,
      kPing                = 0,
      kHardInterrupt       = 1,
      kSoftInterrupt,
      kShutdownInterrupt
   };
   enum EProofCacheCommands {
      kShowCache           = 1,
      kClearCache          = 2,
      kShowPackages        = 3,
      kClearPackages       = 4,
      kClearPackage        = 5,
      kBuildPackage        = 6,
      kLoadPackage         = 7,
      kShowEnabledPackages = 8,
      kShowSubCache        = 9,
      kClearSubCache       = 10,
      kShowSubPackages     = 11,
      kDisableSubPackages  = 12,
      kDisableSubPackage   = 13,
      kBuildSubPackage     = 14,
      kUnloadPackage       = 15,
      kDisablePackage      = 16,
      kUnloadPackages      = 17,
      kDisablePackages     = 18,
      kListPackages        = 19,
      kListEnabledPackages = 20,
      kLoadMacro           = 21
   };
   enum EProofDataSetCommands {
      kUploadDataSet       = 1,  //Upload a dataset
      kCheckDataSetName    = 2,  //Check wheter dataset of this name exists
      kGetDataSets         = 3,  //List datasets saved on  the master node
      kCreateDataSet       = 4,  //Save a TList object as a dataset
      kGetDataSet          = 5,  //Get a TList of TFileInfo objects
      kVerifyDataSet       = 6,  //Try open all files from a dataset and report results
      kRemoveDataSet       = 7,  //Remove a dataset but leave files belonging to it
      kAppendDataSet       = 8   //Add new files to an existing dataset
   };
   enum ESendFileOpt {
      kAscii               = 0x0,
      kBinary              = 0x1,
      kForce               = 0x2,
      kForward             = 0x4
   };
   enum EProofWrkListAction {
      kActivateWorker      = 1,
      kDeactivateWorker    = 2
   };
   enum EBuildPackageOpt {
      kDontBuildOnClient   = -2,
      kBuildOnSlavesNoWait = -1,
      kBuildAll            = 0,
      kCollectBuildResults = 1
   };

   Bool_t          fValid;           //is this a valid proof object
   TString         fMaster;          //master server ("" if a master); used in the browser
   TString         fWorkDir;         //current work directory on remote servers
   Int_t           fLogLevel;        //server debug logging level
   Int_t           fStatus;          //remote return status (part of kPROOF_LOGDONE)
   TList          *fSlaveInfo;       //!list returned by kPROOF_GETSLAVEINFO
   Bool_t          fMasterServ;      //true if we are a master server
   Bool_t          fSendGroupView;   //if true send new group view
   TList          *fActiveSlaves;    //list of active slaves (subset of all slaves)
   TList          *fInactiveSlaves;  //list of inactive slaves (good but not used for processing)
   TList          *fUniqueSlaves;    //list of all active slaves with unique file systems
   TList          *fAllUniqueSlaves;  //list of all active slaves with unique file systems, including all submasters
   TList          *fNonUniqueMasters; //list of all active masters with a nonunique file system
   TMonitor       *fActiveMonitor;   //monitor activity on all active slave sockets
   TMonitor       *fUniqueMonitor;   //monitor activity on all unique slave sockets
   TMonitor       *fAllUniqueMonitor; //monitor activity on all unique slave sockets, including all submasters
   TMonitor       *fCurrentMonitor;  //currently active monitor
   Long64_t        fBytesRead;       //bytes read by all slaves during the session
   Float_t         fRealTime;        //realtime spent by all slaves during the session
   Float_t         fCpuTime;         //CPU time spent by all slaves during the session
   TSignalHandler *fIntHandler;      //interrupt signal handler (ctrl-c)
   TPluginHandler *fProgressDialog;  //progress dialog plugin
   Bool_t          fProgressDialogStarted; //indicates if the progress dialog is up
   TVirtualProofPlayer *fPlayer;     //current player
   TList          *fFeedback;        //list of names to be returned as feedback
   TList          *fChains;          //chains with this proof set
   struct MD5Mod_t {
      TMD5   fMD5;                   //file's md5
      Long_t fModtime;               //file's modification time
   };
   typedef std::map<TString, MD5Mod_t> FileMap_t;
   FileMap_t       fFileMap;         //map keeping track of a file's md5 and mod time
   TDSet          *fDSet;            //current TDSet being validated

   Bool_t          fIdle;            //on clients, true if no PROOF jobs running
   Bool_t          fSync;            //true if type of currently processed query is sync

   Bool_t          fRedirLog;        //redirect received log info
   TString         fLogFileName;     //name of the temp file for redirected logs
   FILE           *fLogFileW;        //temp file to redirect logs
   FILE           *fLogFileR;        //temp file to read redirected logs
   Bool_t          fLogToWindowOnly; //send log to window only

   TList          *fWaitingSlaves;   //stores a TPair of the slaves's TSocket and TMessage
   TList          *fQueries;         //list of TProofQuery objects
   Int_t           fOtherQueries;    //number of queries in list from previous sessions
   Int_t           fDrawQueries;     //number of draw queries during this sessions
   Int_t           fMaxDrawQueries;  //max number of draw queries kept
   Int_t           fSeqNum;          //Remote sequential # of the last query submitted

   Int_t           fSessionID;       //remote ID of the session

   Bool_t          fEndMaster;       //true for a master in direct contact only with workers

   TString         fPackageDir;      //package directory (used on client)
   THashList      *fGlobalPackageDirList;//list of directories containing global packages libs
   TProofLockPath *fPackageLock;     //package lock
   TList          *fEnabledPackagesOnClient; //list of packages enabled on client

   static TList   *fgProofEnvList;  // List of TNameds defining environment
                                    // variables to pass to proofserv
protected:
   enum ESlaves { kAll, kActive, kUnique, kAllUnique };

   TUrl            fUrl;            //Url of the master
   TString         fConfFile;       //file containing config information
   TString         fConfDir;        //directory containing cluster config information
   TString         fImage;          //master's image name
   Int_t           fProtocol;       //remote PROOF server protocol version number
   TList          *fSlaves;         //list of all slave servers as in config file
   TList          *fBadSlaves;      //dead slaves (subset of all slaves)
   TMonitor       *fAllMonitor;     //monitor activity on all valid slave sockets
   Bool_t          fDataReady;      //true if data is ready to be analyzed
   Long64_t        fBytesReady;     //number of bytes staged
   Long64_t        fTotalBytes;     //number of bytes to be analyzed
   TList          *fAvailablePackages; //list of available packages
   TList          *fEnabledPackages;   //list of enabled packages

   TString         fDataPoolUrl;    // default data pool entry point URL
   TProofMgr::EServType fServType;  // type of server: proofd, XrdProofd
   TProofMgr      *fManager;        // manager to which this session belongs (if any)
   EQueryMode      fQueryMode;      // default query mode

   static TSemaphore *fgSemaphore;   //semaphore to control no of parallel startup threads

private:
   TProof(const TProof &);           // not implemented
   void operator=(const TProof &);   // idem

   void     CleanGDirectory(TList *ol);

   Int_t    Exec(const char *cmd, ESlaves list, Bool_t plusMaster);
   Int_t    SendCommand(const char *cmd, ESlaves list = kActive);
   Int_t    SendCurrentState(ESlaves list = kActive);
   Bool_t   CheckFile(const char *file, TSlave *sl, Long_t modtime);
   Int_t    SendFile(const char *file, Int_t opt = (kBinary | kForward),
                     const char *rfile = 0, TSlave *sl = 0);
   Int_t    SendObject(const TObject *obj, ESlaves list = kActive);
   Int_t    SendGroupView();
   Int_t    SendInitialState();
   Int_t    SendPrint(Option_t *option="");
   Int_t    Ping(ESlaves list);
   void     Interrupt(EUrgent type, ESlaves list = kActive);
   void     AskStatistics();
   void     AskParallel();
   Int_t    GoParallel(Int_t nodes, Bool_t accept = kFALSE, Bool_t random = kFALSE);
   Int_t    SetParallelSilent(Int_t nodes, Bool_t random = kFALSE);
   void     RecvLogFile(TSocket *s, Int_t size);
   void     NotifyLogMsg(const char *msg, const char *sfx = "\n");
   Int_t    BuildPackage(const char *package, EBuildPackageOpt opt = kBuildAll);
   Int_t    BuildPackageOnClient(const TString &package);
   Int_t    LoadPackage(const char *package, Bool_t notOnClient = kFALSE);
   Int_t    LoadPackageOnClient(const TString &package);
   Int_t    UnloadPackage(const char *package);
   Int_t    UnloadPackageOnClient(const char *package);
   Int_t    UnloadPackages();
   Int_t    UploadPackageOnClient(const TString &package, EUploadPackageOpt opt, TMD5 *md5);
   Int_t    DisablePackage(const char *package);
   Int_t    DisablePackageOnClient(const char *package);
   Int_t    DisablePackages();

   void     Activate(TList *slaves = 0);
   Int_t    Broadcast(const TMessage &mess, TList *slaves);
   Int_t    Broadcast(const TMessage &mess, ESlaves list = kActive);
   Int_t    Broadcast(const char *mess, Int_t kind, TList *slaves);
   Int_t    Broadcast(const char *mess, Int_t kind = kMESS_STRING, ESlaves list = kActive);
   Int_t    Broadcast(Int_t kind, TList *slaves) { return Broadcast(0, kind, slaves); }
   Int_t    Broadcast(Int_t kind, ESlaves list = kActive) { return Broadcast(0, kind, list); }
   Int_t    BroadcastGroupPriority(const char *grp, Int_t priority, ESlaves list = kActive);
   Int_t    BroadcastGroupPriority(const char *grp, Int_t priority, TList *workers);
   Int_t    BroadcastObject(const TObject *obj, Int_t kind, TList *slaves);
   Int_t    BroadcastObject(const TObject *obj, Int_t kind = kMESS_OBJECT, ESlaves list = kActive);
   Int_t    BroadcastRaw(const void *buffer, Int_t length, TList *slaves);
   Int_t    BroadcastRaw(const void *buffer, Int_t length, ESlaves list = kActive);
   Int_t    Collect(const TSlave *sl, Long_t timeout = -1);
   Int_t    Collect(TMonitor *mon, Long_t timeout = -1);
   Int_t    CollectInputFrom(TSocket *s);

   void     FindUniqueSlaves();
   TSlave  *FindSlave(TSocket *s) const;
   TList   *GetListOfSlaves() const { return fSlaves; }
   TList   *GetListOfInactiveSlaves() const { return fInactiveSlaves; }
   TList   *GetListOfUniqueSlaves() const { return fUniqueSlaves; }
   TList   *GetListOfBadSlaves() const { return fBadSlaves; }
   Int_t    GetNumberOfSlaves() const;
   Int_t    GetNumberOfActiveSlaves() const;
   Int_t    GetNumberOfInactiveSlaves() const;
   Int_t    GetNumberOfUniqueSlaves() const;
   Int_t    GetNumberOfBadSlaves() const;

   Bool_t   IsEndMaster() const { return fEndMaster; }
   void     ModifyWorkerLists(const char *ord, Bool_t add);

   Bool_t   IsSync() const { return fSync; }
   void     InterruptCurrentMonitor();

   void     MarkBad(TSlave *sl);
   void     MarkBad(TSocket *s);

   void     ActivateAsyncInput();
   void     DeActivateAsyncInput();
   void     HandleAsyncInput(TSocket *s);
   Int_t    GetQueryReference(Int_t qry, TString &ref);

   void     PrintProgress(Long64_t total, Long64_t processed, Float_t procTime = -1.);

protected:
   TProof(); // For derived classes to use
   Int_t           Init(const char *masterurl, const char *conffile,
                        const char *confdir, Int_t loglevel,
                        const char *alias = 0);
   virtual Bool_t  StartSlaves(Bool_t parallel, Bool_t attach = kFALSE);

   void                         SetPlayer(TVirtualProofPlayer *player);
   TVirtualProofPlayer         *GetPlayer() const { return fPlayer; }
   virtual TVirtualProofPlayer *MakePlayer(const char *player = 0, TSocket *s = 0);

   TList  *GetListOfActiveSlaves() const { return fActiveSlaves; }
   TSlave *CreateSlave(const char *url, const char *ord,
                       Int_t perf, const char *image, const char *workdir);
   TSlave *CreateSubmaster(const char *url, const char *ord,
                           const char *image, const char *msd);

   virtual void SaveWorkerInfo();

   Int_t    Collect(ESlaves list = kActive, Long_t timeout = -1);
   Int_t    Collect(TList *slaves, Long_t timeout = -1);

   void         SetDSet(TDSet *dset) { fDSet = dset; }
   virtual void ValidateDSet(TDSet *dset);

   TPluginHandler *GetProgressDialog() const { return fProgressDialog; }

   static void *SlaveStartupThread(void *arg);

public:
   TProof(const char *masterurl, const char *conffile = kPROOF_ConfFile,
          const char *confdir = kPROOF_ConfDir, Int_t loglevel = 0,
          const char *alias = 0, TProofMgr *mgr = 0);
   virtual ~TProof();

   void        cd(Int_t id = -1);

   Int_t       Ping();
   Int_t       Exec(const char *cmd, Bool_t plusMaster = kFALSE);
   Long64_t    Process(TDSet *dset, const char *selector,
                       Option_t *option = "", Long64_t nentries = -1,
                       Long64_t firstentry = 0, TEventList *evl = 0);
   Long64_t    Process(const char *dsetname, const char *selector,
                       Option_t *option = "", Long64_t nentries = -1,
                       Long64_t firstentry = 0, TEventList *evl = 0);
   Long64_t    DrawSelect(TDSet *dset, const char *varexp,
                          const char *selection = "",
                          Option_t *option = "", Long64_t nentries = -1,
                          Long64_t firstentry = 0);
   Int_t       Archive(Int_t query, const char *url);
   Int_t       Archive(const char *queryref, const char *url = 0);
   Int_t       CleanupSession(const char *sessiontag);
   Long64_t    Finalize(Int_t query = -1, Bool_t force = kFALSE);
   Long64_t    Finalize(const char *queryref, Bool_t force = kFALSE);
   Int_t       Remove(Int_t query, Bool_t all = kFALSE);
   Int_t       Remove(const char *queryref, Bool_t all = kFALSE);
   Int_t       Retrieve(Int_t query, const char *path = 0);
   Int_t       Retrieve(const char *queryref, const char *path = 0);

   void        StopProcess(Bool_t abort, Int_t timeout = -1);
   void        Browse(TBrowser *b);

   Int_t       SetParallel(Int_t nodes = 9999, Bool_t random = kFALSE);
   void        SetLogLevel(Int_t level, UInt_t mask = TProofDebug::kAll);

   void        Close(Option_t *option="");
   void        Print(Option_t *option="") const;

   //-- cache and package management
   void        ShowCache(Bool_t all = kFALSE);
   void        ClearCache();
   TList      *GetListOfPackages();
   TList      *GetListOfEnabledPackages();
   void        ShowPackages(Bool_t all = kFALSE);
   void        ShowEnabledPackages(Bool_t all = kFALSE);
   Int_t       ClearPackages();
   Int_t       ClearPackage(const char *package);
   Int_t       EnablePackage(const char *package, Bool_t notOnClient = kFALSE);
   Int_t       UploadPackage(const char *par, EUploadPackageOpt opt = kUntar);
   Int_t       Load(const char *macro, Bool_t notOnClient = kFALSE);

   Int_t       AddDynamicPath(const char *libpath);
   Int_t       AddIncludePath(const char *incpath);
   Int_t       RemoveDynamicPath(const char *libpath);
   Int_t       RemoveIncludePath(const char *incpath);

   //-- dataset management
   Int_t       UploadDataSet(const char *dataset,
                             TList *files,
                             const char *dest = 0,
                             Int_t opt = kAskUser,
                             TList *skippedFiles = 0);
   Int_t       UploadDataSet(const char *dataset,
                             const char *files,
                             const char *dest = 0,
                             Int_t opt = kAskUser,
                             TList *skippedFiles = 0);
   Int_t       UploadDataSetFromFile(const char *dataset,
                                     const char *file,
                                     const char *dest = 0,
                                     Int_t opt = kAskUser);
   Int_t       CreateDataSet(const char *dataset,
                             TList *files,
                             Int_t opt = kAskUser);
   TList      *GetDataSets(const char *dir = 0);
   void        ShowDataSets(const char *dir = 0);

   void        ShowDataSet(const char *dataset);
   Int_t       RemoveDataSet(const char *dateset);
   Int_t       VerifyDataSet(const char *dataset);
   TList      *GetDataSet(const char *dataset);

   const char *GetMaster() const { return fMaster; }
   const char *GetConfDir() const { return fConfDir; }
   const char *GetConfFile() const { return fConfFile; }
   const char *GetUser() const { return fUrl.GetUser(); }
   const char *GetWorkDir() const { return fWorkDir; }
   const char *GetSessionTag() const { return GetName(); }
   const char *GetImage() const { return fImage; }
   const char *GetUrl() { return fUrl.GetUrl(); }
   Int_t       GetPort() const { return fUrl.GetPort(); }
   Int_t       GetRemoteProtocol() const { return fProtocol; }
   Int_t       GetClientProtocol() const { return kPROOF_Protocol; }
   Int_t       GetStatus() const { return fStatus; }
   Int_t       GetLogLevel() const { return fLogLevel; }
   Int_t       GetParallel() const;
   Int_t       GetSessionID() const { return fSessionID; }
   TList      *GetListOfSlaveInfos();

   EQueryMode  GetQueryMode(Option_t *mode = 0) const;
   void        SetQueryMode(EQueryMode mode);

   void        SetRealTimeLog(Bool_t on = kTRUE);

   Long64_t    GetBytesRead() const { return fBytesRead; }
   Float_t     GetRealTime() const { return fRealTime; }
   Float_t     GetCpuTime() const { return fCpuTime; }

   Bool_t      IsProofd() const { return (fServType == TProofMgr::kProofd); }
   Bool_t      IsFolder() const { return kTRUE; }
   Bool_t      IsMaster() const { return fMasterServ; }
   Bool_t      IsValid() const { return fValid; }
   Bool_t      IsParallel() const { return GetParallel() > 0 ? kTRUE : kFALSE; }
   Bool_t      IsIdle() const { return fIdle; }

   //-- input list parameter handling
   void        SetParameter(const char *par, const char *value);
   void        SetParameter(const char *par, Long_t value);
   void        SetParameter(const char *par, Long64_t value);
   void        SetParameter(const char *par, Double_t value);
   TObject    *GetParameter(const char *par) const;
   void        DeleteParameters(const char *wildcard);
   void        ShowParameters(const char *wildcard = "PROOF_*") const;

   void        AddInput(TObject *obj);
   void        ClearInput();
   TObject    *GetOutput(const char *name);
   TList      *GetOutputList();

   void        AddFeedback(const char *name);
   void        RemoveFeedback(const char *name);
   void        ClearFeedback();
   void        ShowFeedback() const;
   TList      *GetFeedbackList() const;

   TList      *GetListOfQueries(Option_t *opt = "");
   Int_t       GetNumberOfQueries();
   Int_t       GetNumberOfDrawQueries() { return fDrawQueries; }
   TList      *GetQueryResults();
   TQueryResult *GetQueryResult(const char *ref = 0);
   void        GetMaxQueries();
   void        SetMaxDrawQueries(Int_t max);
   void        ShowQueries(Option_t *opt = "");

   Bool_t      IsDataReady(Long64_t &totalbytes, Long64_t &bytesready);

   void        SetActive(Bool_t /*active*/ = kTRUE) { }

   void        LogMessage(const char *msg, Bool_t all); //*SIGNAL*
   void        Progress(Long64_t total, Long64_t processed); //*SIGNAL*
   void        Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                        Float_t initTime, Float_t procTime,
                        Float_t evtrti, Float_t mbrti); // *SIGNAL*
   void        Feedback(TList *objs); //*SIGNAL*
   void        QueryResultReady(const char *ref); //*SIGNAL*
   void        CloseProgressDialog(); //*SIGNAL*
   void        ResetProgressDialog(const char *sel, Int_t sz,
                                   Long64_t fst, Long64_t ent); //*SIGNAL*
   void        StartupMessage(const char *msg, Bool_t status, Int_t done,
                              Int_t total); //*SIGNAL*
   void        DataSetStatus(const char *msg, Bool_t status,
                             Int_t done, Int_t total); //*SIGNAL*

   void        SendDataSetStatus(const char *msg, UInt_t n, UInt_t tot, Bool_t st);

   void        GetLog(Int_t start = -1, Int_t end = -1);
   void        PutLog(TQueryResult *qr);
   void        ShowLog(Int_t qry = -1);
   void        ShowLog(const char *queryref);
   Bool_t      SendingLogToWindow() const { return fLogToWindowOnly; }
   void        SendLogToWindow(Bool_t mode) { fLogToWindowOnly = mode; }

   void        ResetProgressDialogStatus() { fProgressDialogStarted = kFALSE; }

   TTree      *GetTreeHeader(TDSet *tdset);
   TList      *GetOutputNames();

   void        AddChain(TChain *chain);
   void        RemoveChain(TChain *chain);

   TDrawFeedback *CreateDrawFeedback();
   void           SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt);
   void           DeleteDrawFeedback(TDrawFeedback *f);

   void        Detach(Option_t *opt = "");

   virtual void SetAlias(const char *alias="");

   TProofMgr  *GetManager() { return fManager; }
   void        SetManager(TProofMgr *mgr);

   void        ActivateWorker(const char *ord);
   void        DeactivateWorker(const char *ord);

   const char *GetDataPoolUrl() const { return fDataPoolUrl; }
   void        SetDataPoolUrl(const char *url) { fDataPoolUrl = url; }

   // Opening and managing PROOF connections
   static TProof       *Open(const char *url = 0, const char *conffile = 0,
                             const char *confdir = 0, Int_t loglevel = 0);
   static TProofMgr    *Mgr(const char *url);
   static void          Reset(const char *url);

   static void          AddEnvVar(const char *name, const char *value);
   static void          DelEnvVar(const char *name);
   static const TList  *GetEnvVars();
   static void          ResetEnvVars();

   // Input/output list utilities
   static Int_t         GetParameter(TCollection *c, const char *par, TString &value);
   static Int_t         GetParameter(TCollection *c, const char *par, Long_t &value);
   static Int_t         GetParameter(TCollection *c, const char *par, Long64_t &value);
   static Int_t         GetParameter(TCollection *c, const char *par, Double_t &value);

   ClassDef(TProof,0)  //PROOF control class
};

// Global object with default PROOF session
R__EXTERN TProof *gProof;

#endif
