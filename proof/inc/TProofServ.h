// @(#)root/proof:$Name:  $:$Id: TProofServ.h,v 1.36 2006/04/19 10:57:44 rdm Exp $
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
#ifndef ROOT_Htypes
#include "Htypes.h"
#endif
#ifndef ROOT_TStopwatch
#include "TStopwatch.h"
#endif
#ifndef ROOT_TProofQueryResult
#include "TProofQueryResult.h"
#endif

class TProof;
class TProofPlayer;
class TSocket;
class TList;
class TDSetElement;
class TMessage;

// Hook to external function setting up authentication related stuff
// for old versions.
// For backward compatibility
typedef Int_t (*OldProofServAuthSetup_t)(TSocket *, Bool_t, Int_t,
                                         TString &, TString &, TString &);
class TList;
class TProofLockPath;

class TProofServ : public TApplication {

friend class TXProofServ;

public:
   enum EQueryAction { kQueryOK, kQueryModify, kQueryStop };

private:
   TString       fService;          //service we are running, either "proofserv" or "proofslave"
   TString       fUser;             //user as which we run
   TString       fConfDir;          //directory containing cluster config information
   TString       fConfFile;         //file containing config information
   TString       fWorkDir;          //directory containing all proof related info
   TString       fImage;            //image name of the session
   TString       fSessionTag;       //tag for the session
   TString       fSessionDir;       //directory containing session dependent files
   TString       fPackageDir;       //directory containing packages and user libs
   TString       fCacheDir;         //directory containing cache of user files
   TString       fQueryDir;         //directory containing query results and status
   TString       fDataSetDir;       //directory containing info about known data sets
   TProofLockPath *fPackageLock;     //package dir locker
   TProofLockPath *fCacheLock;       //cache dir locker
   TProofLockPath *fQueryLock;       //query dir locker
   TProofLockPath *fDataSetLock;     //dataset dir locker
   TString       fArchivePath;      //default archive path
   TSocket      *fSocket;           //socket connection to client
   TProof       *fProof;            //PROOF talking to slave servers
   TProofPlayer *fPlayer;           //actual player
   FILE         *fLogFile;          //log file
   Int_t         fLogFileDes;       //log file descriptor
   TList        *fEnabledPackages;  //list of enabled packages
   Int_t         fProtocol;         //protocol version number
   TString       fOrdinal;          //slave ordinal number
   Int_t         fGroupId;          //slave unique id in the active slave group
   Int_t         fGroupSize;        //size of the active slave group
   Int_t         fLogLevel;         //debug logging level
   Int_t         fNcmd;             //command history number
   Bool_t        fMasterServ;       //true if we are a master server
   Bool_t        fInterrupt;        //if true macro execution will be stopped
   Float_t       fRealTime;         //real time spent executing commands
   Float_t       fCpuTime;          //CPU time spent executing commands
   TStopwatch    fLatency;          //measures latency of packet requests
   TStopwatch    fCompute;          //measures time spend processing a packet

   Int_t         fSeqNum;           //sequential number of last processed query
   Int_t         fDrawQueries;      //number of draw queries processed
   Int_t         fKeptQueries;      //number of queries fully in memory and in dir
   TList        *fQueries;          //list of TProofQueryResult objects
   TList        *fPreviousQueries;  //list of TProofQueryResult objects from previous sections
   TList        *fWaitingQueries;   //list of TProofQueryResult wating to be processed
   Bool_t        fIdle;             //TRUE if idle

   TString       fAFSUser;          //AFS username
   TString       fAFSPasswd;        //AFS password for token initialization

   static Int_t  fgMaxQueries;      //Max number of queries fully kept

   virtual void  RedirectOutput();
   Int_t         CatMotd();
   Int_t         UnloadPackage(const char *package);
   Int_t         UnloadPackages();
   Int_t         OldAuthSetup(TString &wconf);

   // Query handlers
   void          AddLogFile(TProofQueryResult *pq);
   void          FinalizeQuery(TProofPlayer *p, TProofQueryResult *pq);
   TList        *GetDataSet(const char *name);
   Int_t         GetAFSToken();
   TProofQueryResult *MakeQueryResult(Long64_t nentries, const char *opt,
                                      TList *inl, Long64_t first, TDSet *dset,
                                      const char *selec, TEventList *evl);
   TProofQueryResult *LocateQuery(TString queryref, Int_t &qry, TString &qdir);
   void          RemoveQuery(TQueryResult *qr, Bool_t soft = kFALSE);
   void          RemoveQuery(const char *queryref);
   void          SaveQuery(TQueryResult *qr, const char *fout = 0);
   void          SetQueryRunning(TProofQueryResult *pq);

   Int_t         LockSession(const char *sessiontag, TProofLockPath **lck);
   Int_t         CleanupSession(const char *sessiontag);
   void          ScanPreviousQueries(const char *dir);

protected:
   virtual void  HandleArchive(TMessage *mess);
   virtual Int_t HandleCache(TMessage *mess);
   virtual void  HandleCheckFile(TMessage *mess);
   virtual void  HandleLibIncPath(TMessage *mess);
   virtual void  HandleProcess(TMessage *mess);
   virtual void  HandleQueryList(TMessage *mess);
   virtual void  HandleRemove(TMessage *mess);
   virtual void  HandleRetrieve(TMessage *mess);

   virtual void  HandleSocketInputDuringProcess();
   virtual void  Setup();

public:
   TProofServ(Int_t *argc, char **argv);
   virtual ~TProofServ();

   virtual void   CreateServer();

   TProof        *GetProof()      const { return fProof; }
   const char    *GetService()    const { return fService; }
   const char    *GetConfDir()    const { return fConfDir; }
   const char    *GetConfFile()   const { return fConfFile; }
   const char    *GetUser()       const { return fUser; }
   const char    *GetWorkDir()    const { return fWorkDir; }
   const char    *GetImage()      const { return fImage; }
   const char    *GetSessionDir() const { return fSessionDir; }
   Int_t          GetProtocol()   const { return fProtocol; }
   const char    *GetOrdinal()    const { return fOrdinal; }
   Int_t          GetGroupId()    const { return fGroupId; }
   Int_t          GetGroupSize()  const { return fGroupSize; }
   Int_t          GetLogLevel()   const { return fLogLevel; }
   TSocket       *GetSocket()     const { return fSocket; }
   Float_t        GetRealTime()   const { return fRealTime; }
   Float_t        GetCpuTime()    const { return fCpuTime; }
   void           GetOptions(Int_t *argc, char **argv);

   virtual EQueryAction GetWorkers(TList *workers, Int_t &prioritychange);

   virtual void   HandleSocketInput();
   virtual void   HandleUrgentData();
   virtual void   HandleSigPipe();
   void           Interrupt() { fInterrupt = kTRUE; }
   Bool_t         IsMaster() const { return fMasterServ; }
   Bool_t         IsParallel() const;
   Bool_t         IsTopMaster() const { return fOrdinal == "0"; }

   void           Run(Bool_t retrn = kFALSE);

   void           Print(Option_t *option="") const;

   TObject       *Get(const char *namecycle);
   TDSetElement  *GetNextPacket(Long64_t totalEntries = -1);
   void           Reset(const char *dir);
   Int_t          ReceiveFile(const char *file, Bool_t bin, Long64_t size);
   virtual void   SendLogFile(Int_t status = 0, Int_t start = -1, Int_t end = -1);
   void           SendStatistics();
   void           SendParallel();

   // Disable / Enable read timeout
   virtual void   DisableTimeout() { }
   virtual void   EnableTimeout() { }

   virtual void   Terminate(Int_t status);

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

#endif
