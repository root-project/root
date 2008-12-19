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

class TDSet;
class TProof;
class TVirtualProofPlayer;
class TProofLockPath;
class TQueryResultManager;
class TSocket;
class THashList;
class TList;
class TDSetElement;
class TMessage;
class TTimer;
class TReaperTimer;
class TMutex;
class TFileCollection;
class TProofDataSetManager;
class TFileHandler;

// Hook to external function setting up authentication related stuff
// for old versions.
// For backward compatibility
typedef Int_t (*OldProofServAuthSetup_t)(TSocket *, Bool_t, Int_t,
                                         TString &, TString &, TString &);


class TProofServ : public TApplication {

friend class TProofServLite;
friend class TXProofServ;

public:
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
   TString       fAdminPath;        //admin path for this session
   TProofLockPath *fPackageLock;    //package dir locker
   TProofLockPath *fCacheLock;      //cache dir locker
   TProofLockPath *fQueryLock;      //query dir locker
   TString       fArchivePath;      //default archive path
   TSocket      *fSocket;           //socket connection to client
   TProof       *fProof;            //PROOF talking to slave servers
   TVirtualProofPlayer *fPlayer;    //actual player
   FILE         *fLogFile;          //log file
   Int_t         fLogFileDes;       //log file descriptor
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
   TStopwatch    fCompute;          //measures time spend processing a packet

   TFileHandler *fInputHandler;     //Input socket handler

   TQueryResultManager *fQMgr;      //Query-result manager

   TList        *fWaitingQueries;   //list of TProofQueryResult waiting to be processed
   Bool_t        fIdle;             //TRUE if idle

   TList        *fQueuedMsg;        //list of messages waiting to be processed

   TString       fPrefix;           //Prefix identifying the node

   Bool_t        fRealTimeLog;      //TRUE if log messages should be send back in real-time

   TTimer       *fShutdownTimer;    // Timer used to shutdown out-of-control sessions
   TReaperTimer *fReaperTimer;      // Timer used to control children state

   Int_t         fInflateFactor;    // Factor in 1/1000 to inflate the CPU time

   TProofDataSetManager* fDataSetManager; // dataset manager

   Bool_t        fLogToSysLog;     //true if logs should be sent to syslog too
   Bool_t        fSendLogToMaster; // On workers, controls logs sending to master

   // Quotas (-1 to disable)
   Int_t         fMaxQueries;       //Max number of queries fully kept
   Long64_t      fMaxBoxSize;       //Max size of the sandbox
   Long64_t      fHWMBoxSize;       //High-Water-Mark on the sandbox size

   // Memory limits (-1 to disable) set by envs ROOTPROOFASSOFT and RPPTPROFOASHARD
   Long_t        fVirtMemHWM;       //Above this we terminate gently (in kB)
   Long_t        fVirtMemMax;       //Hard limit enforced by the system (in kB)

   static FILE  *fgErrorHandlerFile; // File where to log
   static Int_t  fgRecursive;       // Keep track of recursive inputs during processing

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

protected:
   virtual void  HandleArchive(TMessage *mess);
   virtual Int_t HandleCache(TMessage *mess);
   virtual void  HandleCheckFile(TMessage *mess);
   virtual Int_t HandleDataSets(TMessage *mess);
   virtual void  HandleFork(TMessage *mess);
   virtual void  HandleLibIncPath(TMessage *mess);
   virtual void  HandleProcess(TMessage *mess);
   virtual void  HandleQueryList(TMessage *mess);
   virtual void  HandleRemove(TMessage *mess);
   virtual void  HandleRetrieve(TMessage *mess);
   virtual void  HandleWorkerLists(TMessage *mess);

   virtual void  ProcessNext();
   virtual Int_t Setup();
   Int_t         SetupCommon();
   virtual void  MakePlayer();
   virtual void  DeletePlayer();

   virtual Int_t Fork();

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
   const char    *GetSessionTag() const { return fTopSessionTag; }
   const char    *GetSessionDir() const { return fSessionDir; }
   const char    *GetPackageDir() const { return fPackageDir; }
   Int_t          GetProtocol()   const { return fProtocol; }
   const char    *GetOrdinal()    const { return fOrdinal; }
   Int_t          GetGroupId()    const { return fGroupId; }
   Int_t          GetGroupSize()  const { return fGroupSize; }
   Int_t          GetLogLevel()   const { return fLogLevel; }
   TSocket       *GetSocket()     const { return fSocket; }
   Float_t        GetRealTime()   const { return fRealTime; }
   Float_t        GetCpuTime()    const { return fCpuTime; }
   void           GetOptions(Int_t *argc, char **argv);
   TList         *GetEnabledPackages() const { return fEnabledPackages; }

   Int_t          GetInflateFactor() const { return fInflateFactor; }

   Long_t         GetVirtMemHWM() const { return fVirtMemHWM; }

   const char    *GetPrefix()     const { return fPrefix; }

   void           FlushLogFile();

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

   TObject       *Get(const char *namecycle);
   TDSetElement  *GetNextPacket(Long64_t totalEntries = -1);
   virtual void   ReleaseWorker(const char *) { }
   void           Reset(const char *dir);
   Int_t          ReceiveFile(const char *file, Bool_t bin, Long64_t size);
   virtual Int_t  SendAsynMessage(const char *msg, Bool_t lf = kTRUE);
   virtual void   SendLogFile(Int_t status = 0, Int_t start = -1, Int_t end = -1);
   void           SendStatistics();
   void           SendParallel(Bool_t async = kFALSE);

   // Disable / Enable read timeout
   virtual void   DisableTimeout() { }
   virtual void   EnableTimeout() { }

   virtual void   Terminate(Int_t status);

   // Log control
   Bool_t         LogToSysLog() { return fLogToSysLog; }
   void           LogToMaster(Bool_t on = kTRUE) { fSendLogToMaster = on; }

   static FILE   *SetErrorHandlerFile(FILE *ferr);
   static void    ErrorHandler(Int_t level, Bool_t abort, const char *location,
                               const char *msg);

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
public:
   enum EStatusBits { kFileIsPipe = BIT(23) };
   TProofServLogHandler(const char *cmd, TSocket *s, const char *pfx = "");
   TProofServLogHandler(FILE *f, TSocket *s, const char *pfx = "");
   virtual ~TProofServLogHandler();

   Bool_t IsValid() { return ((fFile && fSocket) ? kTRUE : kFALSE); }

   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }

   static void SetDefaultPrefix(const char *pfx);
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

public:
   TShutdownTimer(TProofServ *p, Int_t delay) : TTimer(delay, kFALSE), fProofServ(p) { }

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

#endif
