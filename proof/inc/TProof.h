// @(#)root/proof:$Name:  $:$Id: TProof.h,v 1.65 2005/09/17 13:52:54 rdm Exp $
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

#ifndef ROOT_TVirtualProof
#include "TVirtualProof.h"
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
#ifndef ROOT_TSocket
#include "TSocket.h"
#endif
#ifndef ROOT_TSysEvtHandler
#include "TSysEvtHandler.h"
#endif
#ifndef ROOT_TThread
#include "TThread.h"
#endif

#include <map>

#ifdef R__GLOBALSTL
namespace std { using ::map; }
#endif


class TMessage;
class TMonitor;
class TSignalHandler;
class TPluginHandler;
class TSlave;
class TProofServ;
class TProofInputHandler;
class TProofInterruptHandler;
class TProofPlayer;
class TProofPlayerRemote;
class TProofProgressDialog;
class TPacketizer2;
class TCondor;
class TTree;
class TDrawFeedback;
class TDSet;
class TSemaphore;
class TCondorSlave;
class TList;
class TProof;
class TVirtualMutex;

// protocol changes:
// 1 -> 2: new arguments for Process() command, option added
// 2 -> 3: package manager enabling protocol changed
// 3 -> 4: introduction of multi-level-master support
// 4 -> 5: added friends support
// 5 -> 6: drop TFTP, support for asynchronous queries
// 6 -> 7: support for multisessions, archieve, retrieve, ...
// 7 -> 8: return number of entries in GetNextPacket

// PROOF magic constants
const Int_t       kPROOF_Protocol = 8;             // protocol version number
const Int_t       kPROOF_Port     = 1093;          // IANA registered PROOF port
const char* const kPROOF_ConfFile = "proof.conf";  // default config file
const char* const kPROOF_ConfDir  = "/usr/local/root";  // default config dir
const char* const kPROOF_WorkDir  = "~/proof";     // default working directory
const char* const kPROOF_CacheDir = "cache";       // file cache dir, under WorkDir
const char* const kPROOF_PackDir  = "packages";    // package dir, under WorkDir
const char* const kPROOF_QueryDir = "queries";     // query dir, under WorkDir
const char* const kPROOF_CacheLockFile   = "/tmp/proof-cache-lock-";   // cache lock file
const char* const kPROOF_PackageLockFile = "/tmp/proof-package-lock-"; // package lock file
const char* const kPROOF_QueryLockFile = "/tmp/proof-query-lock-"; // package lock file

R__EXTERN TVirtualMutex *gProofMutex;


// Helper classes used for parallel startup
class TProofThreadArg {
public:
   TString       fHost;
   Int_t         fPort;
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

   virtual ~TProofThreadArg() { }
};

// PROOF Thread class for parallel startup
class TProofThread {
public:
   TThread       *thread;
   TProofThreadArg *args;
   TProofThread(TThread *t, TProofThreadArg *a) { thread = t; args = a; }
   virtual ~TProofThread() { SafeDelete(thread); SafeDelete(args); }
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
   TProofInputHandler(TProof *p, TSocket *s)
      : TFileHandler(s->GetDescriptor(), 1) { fProof = p; fSocket = s; }
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

   TSlaveInfo(const char *ordinal = "", const char *host = "", Int_t perfidx = 0)
      : fOrdinal(ordinal), fHostName(host), fPerfIndex(perfidx),
        fStatus(kNotActive) { }

   const char *GetName() const { return fHostName; }
   const char *GetOrdinal() const { return fOrdinal; }
   void        SetStatus(ESlaveStatus stat) { fStatus = stat; }

   Int_t  Compare(const TObject *obj) const;
   Bool_t IsSortable() const { return kTRUE; }
   void   Print(Option_t *option="") const;

   ClassDef(TSlaveInfo,2) //basic info on slave
};


class TProof : public TVirtualProof {

friend class TPacketizer;
friend class TPacketizer2;
friend class TProofServ;
friend class TProofInputHandler;
friend class TProofInterruptHandler;
friend class TProofPlayer;
friend class TProofPlayerRemote;
friend class TProofProgressDialog;
friend class TSlave;

private:
   enum EUrgent {
      kHardInterrupt = 1,
      kSoftInterrupt,
      kShutdownInterrupt
   };
   enum EProofCacheCommands {
      kShowCache = 1,
      kClearCache = 2,
      kShowPackages = 3,
      kClearPackages = 4,
      kClearPackage = 5,
      kBuildPackage = 6,
      kLoadPackage = 7,
      kShowEnabledPackages = 8,
      kShowSubCache = 9,
      kClearSubCache = 10,
      kShowSubPackages = 11,
      kDisableSubPackages = 12,
      kDisableSubPackage = 13,
      kBuildSubPackage = 14,
      kUnloadPackage = 15,
      kDisablePackage = 16,
      kUnloadPackages = 17,
      kDisablePackages = 18
   };
   enum ESendFileOpt {
      kAscii         = 0x0,
      kBinary        = 0x1,
      kForce         = 0x2,
      kForward       = 0x4
   };

   Bool_t          fValid;          //is this a valid proof object
   TString         fMaster;         //name of master server (use "" if this is a master)
   TString         fWorkDir;        //current work directory on remote servers
   TString         fSessionTag;     //unique tag of the remote session
   TString         fUser;           //user under which to run
   TString         fUrlProtocol;    //net protocol name
   Int_t           fLogLevel;       //server debug logging level
   Int_t           fStatus;         //remote return status (part of kPROOF_LOGDONE)
   TList          *fSlaveInfo;      //!list returned by kPROOF_GETSLAVEINFO
   Bool_t          fMasterServ;     //true if we are a master server
   Bool_t          fSendGroupView;  //if true send new group view
   TList          *fActiveSlaves;   //list of active slaves (subset of all slaves)
   TList          *fUniqueSlaves;   //list of all active slaves with unique file systems
   TList          *fNonUniqueMasters; //list of all active masters with a nonunique file system
   TMonitor       *fActiveMonitor;  //monitor activity on all active slave sockets
   TMonitor       *fUniqueMonitor;  //monitor activity on all unique slave sockets
   Long64_t        fBytesRead;      //bytes read by all slaves during the session
   Float_t         fRealTime;       //realtime spent by all slaves during the session
   Float_t         fCpuTime;        //CPU time spent by all slaves during the session
   TSignalHandler *fIntHandler;     //interrupt signal handler (ctrl-c)
   TPluginHandler *fProgressDialog; //progress dialog plugin
   Bool_t          fProgressDialogStarted; //indicates if the progress dialog is up
   TProofPlayer   *fPlayer;         //current player
   TList          *fFeedback;       //list of names to be returned as feedback
   TList          *fChains;         //chains with this proof set
   struct MD5Mod_t {
      TMD5   fMD5;                  //file's md5
      Long_t fModtime;              //file's modification time
   };
   typedef std::map<TString, MD5Mod_t> FileMap_t;
   FileMap_t       fFileMap;        //map keeping track of a file's md5 and mod time
   TDSet          *fDSet;           //current TDSet being validated

   Bool_t          fIdle;           //on clients, true if no PROOF jobs running
   Bool_t          fSync;           //true if type of currently processed query is sync

   Bool_t          fRedirLog;       //redirect received log info
   TString         fLogFileName;    //name of the temp file for redirected logs
   FILE           *fLogFileW;       //temp file to redirect logs
   FILE           *fLogFileR;       //temp file to read redirected logs
   Bool_t          fLogToWindowOnly; //send log to window only

   TList          *fWaitingSlaves;  //stores a TPair of the slaves's TSocket and TMessage
   TList          *fQueries;        //list of TProofQuery objects
   Int_t           fOtherQueries;   //number of queries in list from previous sessions
   Int_t           fDrawQueries;    //number of draw queries during this sessions
   Int_t           fMaxDrawQueries; //max number of draw queries kept
   Int_t           fSeqNum;         //Remote sequential # of the last query submitted

protected:
   enum ESlaves { kAll, kActive, kUnique };

   TString         fConfFile;       //file containing config information
   TString         fConfDir;        //directory containing cluster config information
   TString         fImage;          //master's image name
   Int_t           fPort;           //port we are connected to (proofd = 1093)
   Int_t           fProtocol;       //remote PROOF server protocol version number
   TList          *fSlaves;         //list of all slave servers as in config file
   TList          *fBadSlaves;      //dead slaves (subset of all slaves)
   TMonitor       *fAllMonitor;     //monitor activity on all valid slave sockets
   Bool_t          fDataReady;      //true if data is ready to be analyzed
   Long64_t        fBytesReady;     //number of bytes staged
   Long64_t        fTotalBytes;     //number of bytes to be analyzed

   static TSemaphore *fgSemaphore;  //semaphore to control no of parallel startup threads

private:
   TProof(const TProof &);           // not implemented
   void operator=(const TProof &);   // idem

   Int_t    Exec(const char *cmd, ESlaves list);
   Int_t    SendCommand(const char *cmd, ESlaves list = kActive);
   Int_t    SendCurrentState(ESlaves list = kActive);
   Bool_t   CheckFile(const char *file, TSlave *sl, Long_t modtime);
   Int_t    SendFile(const char *file, Int_t opt = (kBinary | kForward),
                     const char *rfile = 0);
   Int_t    SendObject(const TObject *obj, ESlaves list = kActive);
   Int_t    SendGroupView();
   Int_t    SendInitialState();
   Int_t    SendPrint(Option_t *option="");
   Int_t    Ping(ESlaves list);
   void     Interrupt(EUrgent type, ESlaves list = kActive);
   void     AskStatistics();
   void     AskParallel();
   Int_t    GoParallel(Int_t nodes);
   void     RecvLogFile(TSocket *s, Int_t size);
   Int_t    BuildPackage(const char *package);
   Int_t    LoadPackage(const char *package);
   Int_t    UnloadPackage(const char *package);
   Int_t    UnloadPackages();
   Int_t    DisablePackage(const char *package);
   Int_t    DisablePackages();

   void     Activate(TList *slaves = 0);
   Int_t    Broadcast(const TMessage &mess, TList *slaves);
   Int_t    Broadcast(const TMessage &mess, ESlaves list = kActive);
   Int_t    Broadcast(const char *mess, Int_t kind, TList *slaves);
   Int_t    Broadcast(const char *mess, Int_t kind = kMESS_STRING, ESlaves list = kActive);
   Int_t    Broadcast(Int_t kind, TList *slaves) { return Broadcast(0, kind, slaves); }
   Int_t    Broadcast(Int_t kind, ESlaves list = kActive) { return Broadcast(0, kind, list); }
   Int_t    BroadcastObject(const TObject *obj, Int_t kind, TList *slaves);
   Int_t    BroadcastObject(const TObject *obj, Int_t kind = kMESS_OBJECT, ESlaves list = kActive);
   Int_t    BroadcastRaw(const void *buffer, Int_t length, TList *slaves);
   Int_t    BroadcastRaw(const void *buffer, Int_t length, ESlaves list = kActive);
   Int_t    Collect(const TSlave *sl);
   Int_t    Collect(TMonitor *mon);
   Int_t    CollectInputFrom(TSocket *s);

   void     FindUniqueSlaves();
   TSlave  *FindSlave(TSocket *s) const;
   TList   *GetListOfSlaves() const { return fSlaves; }
   TList   *GetListOfUniqueSlaves() const { return fUniqueSlaves; }
   TList   *GetListOfBadSlaves() const { return fBadSlaves; }
   Int_t    GetNumberOfSlaves() const;
   Int_t    GetNumberOfActiveSlaves() const;
   Int_t    GetNumberOfUniqueSlaves() const;
   Int_t    GetNumberOfBadSlaves() const;

   Bool_t   IsSync() const { return fSync; }

   void     MarkBad(TSlave *sl);
   void     MarkBad(TSocket *s);

   void     ActivateAsyncInput();
   void     DeActivateAsyncInput();
   void     HandleAsyncInput(TSocket *s);

   void     RedirectLog(Bool_t on = kTRUE);  // redirect log msgs

   Int_t    GetQueryReference(Int_t qry, TString &ref);

protected:
   TProof(); // For derived classes to use
   Int_t           Init(const char *masterurl, const char *conffile,
                        const char *confdir, Int_t loglevel);
   virtual Bool_t  StartSlaves(Bool_t parallel);

   void                  SetPlayer(TProofPlayer *player) { fPlayer = player; };
   TProofPlayer         *GetPlayer() const { return fPlayer; };
   virtual TProofPlayer *MakePlayer();

   TList  *GetListOfActiveSlaves() const { return fActiveSlaves; }
   TSlave *CreateSlave(const char *host, Int_t port, const char *ord,
                       Int_t perf, const char *image, const char *workdir);
   TSlave *CreateSubmaster(const char *host, Int_t port, const char *ord,
                           const char *image, const char *msd);

   Int_t    Collect(ESlaves list = kActive);
   Int_t    Collect(TList *slaves);

   void         SetDSet(TDSet *dset) { fDSet = dset; }
   virtual void ValidateDSet(TDSet *dset);

   TPluginHandler *GetProgressDialog() const { return fProgressDialog; };

   static void *SlaveStartupThread(void *arg);

public:
   TProof(const char *masterurl, const char *conffile = kPROOF_ConfFile,
          const char *confdir = kPROOF_ConfDir, Int_t loglevel = 0);
   virtual ~TProof();

   Int_t       Ping();
   Int_t       Exec(const char *cmd);
   Int_t       Process(TDSet *set, const char *selector,
                       Option_t *option = "", Long64_t nentries = -1,
                       Long64_t firstentry = 0, TEventList *evl = 0);
   Int_t       DrawSelect(TDSet *set, const char *varexp,
                          const char *selection = "",
                          Option_t *option = "", Long64_t nentries = -1,
                          Long64_t firstentry = 0);
   Int_t       Archive(Int_t query, const char *url);
   Int_t       Archive(const char *queryref, const char *url = 0);
   Int_t       CleanupSession(const char *sessiontag);
   Int_t       Finalize(Int_t query = -1, Bool_t force = kFALSE);
   Int_t       Finalize(const char *queryref, Bool_t force = kFALSE);
   Int_t       Remove(Int_t query);
   Int_t       Remove(const char *queryref, Bool_t all = kFALSE);
   Int_t       Retrieve(Int_t query, const char *path = 0);
   Int_t       Retrieve(const char *queryref, const char *path = 0);

   void        StopProcess(Bool_t abort);
   void        AddInput(TObject *obj);
   void        Browse(TBrowser *b);
   void        ClearInput();
   TObject    *GetOutput(const char *name);
   TList      *GetOutputList();

   Int_t       SetParallel(Int_t nodes = 9999);
   void        SetLogLevel(Int_t level, UInt_t mask = TProofDebug::kAll);

   void        Close(Option_t *option="");
   void        Print(Option_t *option="") const;

   void        ShowCache(Bool_t all = kFALSE);
   void        ClearCache();
   void        ShowPackages(Bool_t all = kFALSE);
   void        ShowEnabledPackages(Bool_t all = kFALSE);
   Int_t       ClearPackages();
   Int_t       ClearPackage(const char *package);
   Int_t       EnablePackage(const char *package);
   Int_t       UploadPackage(const char *par);

   const char *GetMaster() const { return fMaster; }
   const char *GetConfDir() const { return fConfDir; }
   const char *GetConfFile() const { return fConfFile; }
   const char *GetUser() const { return fUser; }
   const char *GetWorkDir() const { return fWorkDir; }
   const char *GetSessionTag() const { return fSessionTag; }
   const char *GetImage() const { return fImage; }
   const char *GetUrlProtocol() const { return fUrlProtocol; }
   Int_t       GetPort() const { return fPort; }
   Int_t       GetRemoteProtocol() const { return fProtocol; }
   Int_t       GetClientProtocol() const { return kPROOF_Protocol; }
   Int_t       GetStatus() const { return fStatus; }
   Int_t       GetLogLevel() const { return fLogLevel; }
   Int_t       GetParallel() const;
   TList      *GetSlaveInfo();

   EQueryMode  GetQueryMode() const;
   EQueryMode  GetQueryMode(Option_t *mode) const;
   void        SetQueryMode(EQueryMode mode);

   Long64_t    GetBytesRead() const { return fBytesRead; }
   Float_t     GetRealTime() const { return fRealTime; }
   Float_t     GetCpuTime() const { return fCpuTime; }

   Bool_t      IsFolder() const { return kTRUE; }
   Bool_t      IsMaster() const { return fMasterServ; }
   Bool_t      IsValid() const { return fValid; }
   Bool_t      IsParallel() const { return GetParallel() > 0 ? kTRUE : kFALSE; }
   Bool_t      IsIdle() const { return fIdle; }

   void        AddFeedback(const char *name);
   void        RemoveFeedback(const char *name);
   void        ClearFeedback();
   void        ShowFeedback() const;
   TList      *GetFeedbackList() const;

   TList      *GetListOfQueries(Option_t *opt = "");
   Int_t       GetNumberOfQueries();
   Int_t       GetNumberOfDrawQueries() { return fDrawQueries; }
   TList      *GetQueryResults();
   TQueryResult *GetQueryResult(const char *ref);
   void        GetMaxQueries();
   void        SetMaxDrawQueries(Int_t max);
   void        ShowQueries(Option_t *opt = "");

   Bool_t      IsDataReady(Long64_t &totalbytes, Long64_t &bytesready);

   void        SetActive(Bool_t /*active*/ = kTRUE) { }

   void        LogMessage(const char *msg, Bool_t all); //*SIGNAL*
   void        Progress(Long64_t total, Long64_t processed); //*SIGNAL*
   void        Feedback(TList *objs); //*SIGNAL*
   void        QueryResultReady(const char *ref); //*SIGNAL*
   void        ResetProgressDialog(const char *sel, Int_t sz,
                                   Long64_t fst, Long64_t ent); //*SIGNAL*

   void        GetLog(Int_t start = -1, Int_t end = -1);
   void        PutLog(TQueryResult *qr);
   void        ShowLog(Int_t qry = -1);
   void        ShowLog(const char *queryref);
   Bool_t      GetLogToWindow() const { return fLogToWindowOnly; }
   void        SetLogToWindow(Bool_t mode) { fLogToWindowOnly = mode; }

   void        ResetProgressDialogStatus() { fProgressDialogStarted = kFALSE; }

   TTree      *GetTreeHeader(TDSet *tdset);
   TList      *GetOutputNames();

   void        AddChain(TChain *chain);
   void        RemoveChain(TChain *chain);

   TDrawFeedback *CreateDrawFeedback();
   void           SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt);
   void           DeleteDrawFeedback(TDrawFeedback *f);

   ClassDef(TProof,0)  //PROOF control class
};

#endif
