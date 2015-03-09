// @(#)root/proof:$Id$
// Author: Fons Rademakers   16/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TProofServ
#define ROOT_TProofServ

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofServ                                                           //
//                                                                      //
// TProofServ is the PROOF server. It can act either as the master      //
// server or as a slave server, depending on its startup arguments. It  //
// receives and handles message coming from the client or from the      //
// master server.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TApplication
#include "TApplication.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TSysEvtHandler
#include "TSysEvtHandler.h"
#endif
#ifndef ROOT_TStopwatch
#include "TStopwatch.h"
#endif
#ifndef ROOT_TTimer
#include "TTimer.h"
#endif
#ifndef ROOT_TProofQueryResult
#include "TProofQueryResult.h"
#endif

class TDataSetManager;
class TDataSetManagerFile;
class TDSet;
class TDSetElement;
class TFileCollection;
class TFileHandler;
class THashList;
class TIdleTOTimer;
class TList;
class TMap;
class TMessage;
class TMonitor;
class TMutex;
class TProof;
class TProofLockPath;
class TQueryResultManager;
class TReaperTimer;
class TServerSocket;
class TShutdownTimer;
class TSocket;
class TVirtualProofPlayer;

// Hook to external function setting up authentication related stuff
// for old versions.
// For backward compatibility
typedef Int_t (*OldProofServAuthSetup_t)(TSocket *, Bool_t, Int_t,
                                         TString &, TString &, TString &);


class TProofServ : public TApplication {

friend class TProofServLite;
friend class TXProofServ;

public:
   enum EStatusBits { kHighMemory = BIT(16) };
   enum EQueryAction { kQueryOK, kQueryModify, kQueryStop, kQueryEnqueued };

private:
   TString       fService;          //service we are running, either "proofserv" or "proofslave"
   TString       fUser;             //user as which we run
   TString       fGroup;            //group the user belongs to
   TString       fConfDir;          //directory containing cluster config information
   TString       fConfFile;         //file containing config information
   TString       fWorkDir;          //directory containing all proof related info
   TString       fImage;            //image name of the session
   TString       fSessionTag;       //tag for the server session
   TString       fTopSessionTag;    //tag for the global session
   TString       fSessionDir;       //directory containing session dependent files
   TString       fPackageDir;       //directory containing packages and user libs
   THashList    *fGlobalPackageDirList;  //list of directories containing global packages libs
   TString       fCacheDir;         //directory containing cache of user files
   TString       fQueryDir;         //directory containing query results and status
   TString       fDataSetDir;       //directory containing info about known data sets
   TString       fDataDir;          //directory containing data files produced during queries
   TString       fDataDirOpts;      //Url type options for fDataDir
   TString       fAdminPath;        //admin path for this session
   TString       fOutputFile;       //path with the temporary results of the current or last query
   TProofLockPath *fPackageLock;    //package dir locker
   TProofLockPath *fCacheLock;      //cache dir locker
   TProofLockPath *fQueryLock;      //query dir locker
   TString       fArchivePath;      //default archive path
   TSocket      *fSocket;           //socket connection to client
   TProof       *fProof;            //PROOF talking to slave servers
   TVirtualProofPlayer *fPlayer;    //actual player
   FILE         *fLogFile;          //log file
   Int_t         fLogFileDes;       //log file descriptor
   Long64_t      fLogFileMaxSize;   //max size for log files (enabled if > 0)
   TList        *fEnabledPackages;  //list of enabled packages
   Int_t         fProtocol;         //protocol version number
   TString       fOrdinal;          //slave ordinal number
   Int_t         fGroupId;          //slave unique id in the active slave group
   Int_t         fGroupSize;        //size of the active slave group
   Int_t         fLogLevel;         //debug logging level
   Int_t         fNcmd;             //command history number
   Int_t         fGroupPriority;    //priority of group the user belongs to (0 - 100)
   Bool_t        fEndMaster;        //true for a master in direct contact only with workers
   Bool_t        fMasterServ;       //true if we are a master server
   Bool_t        fInterrupt;        //if true macro execution will be stopped
   Float_t       fRealTime;         //real time spent executing commands
   Float_t       fCpuTime;          //CPU time spent executing commands
   TStopwatch    fLatency;          //measures latency of packet requests
   TStopwatch    fCompute;          //measures time spent processing a packet
   TStopwatch    fSaveOutput;       //measures time spent saving the partial result
   Int_t         fQuerySeqNum;      //sequential number of the current or last query

   Int_t         fTotSessions;      //Total number of PROOF sessions on the cluster
   Int_t         fActSessions;      //Total number of active PROOF sessions on the cluster
   Float_t       fEffSessions;      //Effective Number of PROOF sessions on the assigned machines

   TFileHandler *fInputHandler;     //Input socket handler

   TQueryResultManager *fQMgr;      //Query-result manager

   TList        *fWaitingQueries;   //list of TProofQueryResult waiting to be processed
   Bool_t        fIdle;             //TRUE if idle
   TMutex       *fQMtx;             // To protect async msg queue

   TList        *fQueuedMsg;        //list of messages waiting to be processed

   TString       fPrefix;           //Prefix identifying the node

   Bool_t        fRealTimeLog;      //TRUE if log messages should be send back in real-time

   TShutdownTimer *fShutdownTimer;  // Timer used to shutdown out-of-control sessions
   TReaperTimer   *fReaperTimer;    // Timer used to control children state
   TIdleTOTimer   *fIdleTOTimer;    // Timer used to control children state

   Int_t         fCompressMsg;     // Compression level for messages

   TDataSetManager* fDataSetManager; // dataset manager
   TDataSetManagerFile *fDataSetStgRepo;  // repository for staging requests

   Bool_t        fSendLogToMaster; // On workers, controls logs sending to master

   TServerSocket *fMergingSocket;  // Socket used for merging outputs if submerger
   TMonitor      *fMergingMonitor; // Monitor for merging sockets
   Int_t          fMergedWorkers;  // Number of workers merged

   // Quotas (-1 to disable)
   Int_t         fMaxQueries;       //Max number of queries fully kept
   Long64_t      fMaxBoxSize;       //Max size of the sandbox
   Long64_t      fHWMBoxSize;       //High-Water-Mark on the sandbox size

   // Memory limits (-1 to disable) set by envs ROOTPROFOASHARD, PROOF_VIRTMEMMAX, PROOF_RESMEMMAX
   static Long_t fgVirtMemMax;       //Hard limit enforced by the system (in kB)
   static Long_t fgResMemMax;        //Hard limit on the resident memory checked
                                     //in TProofPlayer::Process (in kB)
   static Float_t fgMemHWM;          // Threshold fraction of max for warning and finer monitoring
   static Float_t fgMemStop;         // Fraction of max for stop processing

   // In bytes; default is 1MB
   Long64_t      fMsgSizeHWM;       //High-Water-Mark on the size of messages with results

   static FILE  *fgErrorHandlerFile; // File where to log
   static Int_t  fgRecursive;       // Keep track of recursive inputs during processing

   // Control sending information to syslog
   static Int_t  fgLogToSysLog;      // >0 sent to syslog too
   static TString fgSysLogService;   // name of the syslog service (eg: proofm-0, proofw-0.67)
   static TString fgSysLogEntity;   // logging entity (<user>:<group>)

   Int_t         GetCompressionLevel() const;

   void          RedirectOutput(const char *dir = 0, const char *mode = "w");
   Int_t         CatMotd();
   Int_t         UnloadPackage(const char *package);
   Int_t         UnloadPackages();
   Int_t         OldAuthSetup(TString &wconf);
   Int_t         GetPriority();

   // Query handlers
   TProofQueryResult *MakeQueryResult(Long64_t nentries, const char *opt,
                                      TList *inl, Long64_t first, TDSet *dset,
                                      const char *selec, TObject *elist);
   void          SetQueryRunning(TProofQueryResult *pq);

   // Results handling
   Int_t         SendResults(TSocket *sock, TList *outlist = 0, TQueryResult *pq = 0);
   Bool_t        AcceptResults(Int_t connections, TVirtualProofPlayer *mergerPlayer);

   // Waiting queries handlers
   void          SetIdle(Bool_t st = kTRUE);
   Bool_t        IsWaiting();
   Int_t         WaitingQueries();
   Int_t         QueueQuery(TProofQueryResult *pq);
   TProofQueryResult *NextQuery();
   Int_t         CleanupWaitingQueries(Bool_t del = kTRUE, TList *qls = 0);

protected:
   virtual void  HandleArchive(TMessage *mess, TString *slb = 0);
   virtual Int_t HandleCache(TMessage *mess, TString *slb = 0);
   virtual void  HandleCheckFile(TMessage *mess, TString *slb = 0);
   virtual Int_t HandleDataSets(TMessage *mess, TString *slb = 0);
   virtual void  HandleSubmerger(TMessage *mess);
   virtual void  HandleFork(TMessage *mess);
   virtual Int_t HandleLibIncPath(TMessage *mess);
   virtual void  HandleProcess(TMessage *mess, TString *slb = 0);
   virtual void  HandleQueryList(TMessage *mess);
   virtual void  HandleRemove(TMessage *mess, TString *slb = 0);
   virtual void  HandleRetrieve(TMessage *mess, TString *slb = 0);
   virtual Int_t HandleWorkerLists(TMessage *mess);

   virtual void  ProcessNext(TString *slb = 0);
   virtual Int_t Setup();
   Int_t         SetupCommon();
   virtual void  MakePlayer();
   virtual void  DeletePlayer();

   virtual Int_t Fork();
   Int_t         GetSessionStatus();
   Bool_t        IsIdle();
   Bool_t        UnlinkDataDir(const char *path);

   static TString fgLastMsg;    // Message about status before exception
   static Long64_t fgLastEntry;  // Last entry before exception

public:
   TProofServ(Int_t *argc, char **argv, FILE *flog = 0);
   virtual ~TProofServ();

   virtual Int_t  CreateServer();

   TProof        *GetProof()      const { return fProof; }
   const char    *GetService()    const { return fService; }
   const char    *GetConfDir()    const { return fConfDir; }
   const char    *GetConfFile()   const { return fConfFile; }
   const char    *GetUser()       const { return fUser; }
   const char    *GetGroup()      const { return fGroup; }
   const char    *GetWorkDir()    const { return fWorkDir; }
   const char    *GetImage()      const { return fImage; }
   const char    *GetSessionTag() const { return fSessionTag; }
   const char    *GetTopSessionTag() const { return fTopSessionTag; }
   const char    *GetSessionDir() const { return fSessionDir; }
   const char    *GetPackageDir() const { return fPackageDir; }
   const char    *GetDataDir()    const { return fDataDir; }
   const char    *GetDataDirOpts() const { return fDataDirOpts; }
   Int_t          GetProtocol()   const { return fProtocol; }
   const char    *GetOrdinal()    const { return fOrdinal; }
   Int_t          GetGroupId()    const { return fGroupId; }
   Int_t          GetGroupSize()  const { return fGroupSize; }
   Int_t          GetLogLevel()   const { return fLogLevel; }
   TSocket       *GetSocket()     const { return fSocket; }
   Float_t        GetRealTime()   const { return fRealTime; }
   Float_t        GetCpuTime()    const { return fCpuTime; }
   Int_t          GetQuerySeqNum() const { return fQuerySeqNum; }

   Int_t          GetTotSessions() const { return fTotSessions; }
   Int_t          GetActSessions() const { return fActSessions; }
   Float_t        GetEffSessions() const { return fEffSessions; }

   void           GetOptions(Int_t *argc, char **argv);
   TList         *GetEnabledPackages() const { return fEnabledPackages; }

   static Long_t  GetVirtMemMax();
   static Long_t  GetResMemMax();
   static Float_t GetMemHWM();
   static Float_t GetMemStop();

   Long64_t       GetMsgSizeHWM() const { return fMsgSizeHWM; }

   const char    *GetPrefix()     const { return fPrefix; }

   void           FlushLogFile();
   void           TruncateLogFile();  // Called also by TDSetProxy::Next()

   TProofLockPath *GetCacheLock() { return fCacheLock; }      //cache dir locker; used by TProofPlayer
   Int_t          CopyFromCache(const char *name, Bool_t cpbin);
   Int_t          CopyToCache(const char *name, Int_t opt = 0);

   virtual EQueryAction GetWorkers(TList *workers, Int_t &prioritychange,
                                   Bool_t resume = kFALSE);
   virtual void   HandleException(Int_t sig);
   virtual Int_t  HandleSocketInput(TMessage *mess, Bool_t all);
   virtual void   HandleSocketInput();
   virtual void   HandleUrgentData();
   virtual void   HandleSigPipe();
   virtual void   HandleTermination() { Terminate(0); }
   void           Interrupt() { fInterrupt = kTRUE; }
   Bool_t         IsEndMaster() const { return fEndMaster; }
   Bool_t         IsMaster() const { return fMasterServ; }
   Bool_t         IsParallel() const;
   Bool_t         IsTopMaster() const { return fOrdinal == "0"; }

   void           Run(Bool_t retrn = kFALSE);

   void           Print(Option_t *option="") const;

   void           RestartComputeTime();

   TObject       *Get(const char *namecycle);
   TDSetElement  *GetNextPacket(Long64_t totalEntries = -1);
   virtual void   ReleaseWorker(const char *) { }
   void           Reset(const char *dir);
   Int_t          ReceiveFile(const char *file, Bool_t bin, Long64_t size);
   virtual Int_t  SendAsynMessage(const char *msg, Bool_t lf = kTRUE);
   virtual void   SendLogFile(Int_t status = 0, Int_t start = -1, Int_t end = -1);
   void           SendStatistics();
   void           SendParallel(Bool_t async = kFALSE);

   Int_t          UpdateSessionStatus(Int_t xst = -1);

   // Disable / Enable read timeout
   virtual void   DisableTimeout() { }
   virtual void   EnableTimeout() { }

   virtual void   Terminate(Int_t status);

   // Log control
   void           LogToMaster(Bool_t on = kTRUE) { fSendLogToMaster = on; }

   static FILE   *SetErrorHandlerFile(FILE *ferr);
   static void    ErrorHandler(Int_t level, Bool_t abort, const char *location,
                               const char *msg);

   static void    ResolveKeywords(TString &fname, const char *path = 0);

   static void    SetLastMsg(const char *lastmsg);
   static void    SetLastEntry(Long64_t lastentry);

   // To handle local data server related paths
   static void    FilterLocalroot(TString &path, const char *url = "root://dum/");
   static void    GetLocalServer(TString &dsrv);

   // To prepara ethe map of files to process
   static TMap   *GetDataSetNodeMap(TFileCollection *fc, TString &emsg);
   static Int_t   RegisterDataSets(TList *in, TList *out, TDataSetManager *dsm, TString &e);

   static Bool_t      IsActive();
   static TProofServ *This();

   ClassDef(TProofServ,0)  //PROOF Server Application Interface
};

R__EXTERN TProofServ *gProofServ;

class TProofLockPath : public TNamed {
private:
   Int_t         fLockId;        //file id of dir lock

public:
   TProofLockPath(const char *path) : TNamed(path,path), fLockId(-1) { }
   ~TProofLockPath() { if (IsLocked()) Unlock(); }

   Int_t         Lock();
   Int_t         Unlock();

   Bool_t        IsLocked() const { return (fLockId > -1); }
};

class TProofLockPathGuard {
private:
   TProofLockPath  *fLocker; //locker instance

public:
   TProofLockPathGuard(TProofLockPath *l) { fLocker = l; if (fLocker) fLocker->Lock(); }
   ~TProofLockPathGuard() { if (fLocker) fLocker->Unlock(); }
};

//----- Handles output from commands executed externally via a pipe. ---------//
//----- The output is redirected one level up (i.e., to master or client). ---//
//______________________________________________________________________________
class TProofServLogHandler : public TFileHandler {
private:
   TSocket     *fSocket; // Socket where to redirect the message
   FILE        *fFile;   // File connected with the open pipe
   TString      fPfx;    // Prefix to be prepended to messages

   static TString fgPfx; // Default prefix to be prepended to messages
   static Int_t   fgCmdRtn; // Return code of the command execution (available only
                            // after closing the pipe)
public:
   enum EStatusBits { kFileIsPipe = BIT(23) };
   TProofServLogHandler(const char *cmd, TSocket *s, const char *pfx = "");
   TProofServLogHandler(FILE *f, TSocket *s, const char *pfx = "");
   virtual ~TProofServLogHandler();

   Bool_t IsValid() { return ((fFile && fSocket) ? kTRUE : kFALSE); }

   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }

   static void SetDefaultPrefix(const char *pfx);
   static Int_t GetCmdRtn();
};

//--- Guard class: close pipe, deactivatethe related descriptor --------------//
//______________________________________________________________________________
class TProofServLogHandlerGuard {

private:
   TProofServLogHandler   *fExecHandler;

public:
   TProofServLogHandlerGuard(const char *cmd, TSocket *s,
                             const char *pfx = "", Bool_t on = kTRUE);
   TProofServLogHandlerGuard(FILE *f, TSocket *s,
                             const char *pfx = "", Bool_t on = kTRUE);
   virtual ~TProofServLogHandlerGuard();
};

//--- Special timer to control delayed shutdowns
//______________________________________________________________________________
class TShutdownTimer : public TTimer {
private:
   TProofServ    *fProofServ;
   Int_t          fTimeout;

public:
   TShutdownTimer(TProofServ *p, Int_t delay);

   Bool_t Notify();
};

//--- Synchronous timer used to reap children processes change of state
//______________________________________________________________________________
class TReaperTimer : public TTimer {
private:
   TList  *fChildren;   // List of children (forked) processes

public:
   TReaperTimer(Long_t frequency = 1000) : TTimer(frequency, kTRUE), fChildren(0) { }
   virtual ~TReaperTimer();

   void AddPid(Int_t pid);
   Bool_t Notify();
};

//--- Special timer to terminate idle sessions
//______________________________________________________________________________
class TIdleTOTimer : public TTimer {
private:
   TProofServ    *fProofServ;

public:
   TIdleTOTimer(TProofServ *p, Int_t delay) : TTimer(delay, kTRUE), fProofServ(p) { }

   Bool_t Notify();
};
//______________________________________________________________________________
class TIdleTOTimerGuard {

private:
   TIdleTOTimer *fIdleTOTimer;

public:
   TIdleTOTimerGuard(TIdleTOTimer *t) : fIdleTOTimer(t) { if (fIdleTOTimer) fIdleTOTimer->Stop(); }
   virtual ~TIdleTOTimerGuard() { if (fIdleTOTimer) fIdleTOTimer->Start(-1, kTRUE); }
};

//______________________________________________________________________________
inline Int_t TProofServ::GetCompressionLevel() const
{
   return (fCompressMsg < 0) ? -1 : fCompressMsg % 100;
}

#endif
