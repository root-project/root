// @(#)root/proof:$Name:  $:$Id: TProof.h,v 1.53 2005/03/17 00:31:17 rdm Exp $
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
class TPacketizer2;
class TCondor;
class TTree;
class TDrawFeedback;
class TDSet;

// protocol changes:
// 1 -> 2: new arguments for Process() command, option added
// 2 -> 3: package manager enabling protocol changed
// 3 -> 4: introduction of multi-level-master support

// PROOF magic constants
const Int_t       kPROOF_Protocol = 4;             // protocol version number
const Int_t       kPROOF_Port     = 1093;          // IANA registered PROOF port
const char* const kPROOF_ConfFile = "proof.conf";  // default config file
const char* const kPROOF_ConfDir  = "/usr/local/root";  // default config dir
const char* const kPROOF_WorkDir  = "~/proof";     // default working directory
const char* const kPROOF_CacheDir = "cache";       // file cache dir, under WorkDir
const char* const kPROOF_PackDir  = "packages";    // package dir, under WorkDir
const char* const kPROOF_CacheLockFile   = "/tmp/proof-cache-lock-";   // cache lock file
const char* const kPROOF_PackageLockFile = "/tmp/proof-package-lock-"; // package lock file


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

friend class TProofServ;
friend class TProofInputHandler;
friend class TProofInterruptHandler;
friend class TProofPlayer;
friend class TProofPlayerRemote;
friend class TSlave;
friend class TPacketizer;
friend class TPacketizer2;

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

   Bool_t          fValid;          //is this a valid proof object
   TString         fMaster;         //name of master server (use "" if this is a master)
   TString         fWorkDir;        //current work directory on remote servers
   TString         fUser;           //user under which to run
   TString         fUrlProtocol;    //net protocol name
   TSecContext    *fSecContext;     //SecContext of the related authentication
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
   TProofPlayer   *fPlayer;         //current player
   TList          *fFeedback;       //List of names to be returned as feedback
   TList          *fChains;         //chains with this proof set
   struct MD5Mod_t {
      TMD5   fMD5;                  //file's md5
      Long_t fModtime;              //file's modification time
   };
   typedef std::map<TString, MD5Mod_t> FileMap_t;
   FileMap_t       fFileMap;        //map keeping track of a file's md5 and mod time
   TDSet          *fDSet;           //current TDSet being validated

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

private:
   TProof(const TProof &);           // not implemented
   void operator=(const TProof &);   // idem

   Int_t    Exec(const char *cmd, ESlaves list);
   Int_t    SendCommand(const char *cmd, ESlaves list = kActive);
   Int_t    SendCurrentState(ESlaves list = kActive);
   Long_t   CheckFile(const char *file, TSlave *sl);
   Int_t    SendFile(const char *file, Bool_t bin = kTRUE);
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

   void     FindUniqueSlaves();
   TSlave  *FindSlave(TSocket *s) const;
   TList   *GetListOfSlaves() const { return fSlaves; }
   TList   *GetListOfUniqueSlaves() const { return fUniqueSlaves; }
   TList   *GetListOfBadSlaves() const { return fBadSlaves; }
   Int_t    GetNumberOfSlaves() const;
   Int_t    GetNumberOfActiveSlaves() const;
   Int_t    GetNumberOfUniqueSlaves() const;
   Int_t    GetNumberOfBadSlaves() const;
   void     MarkBad(TSlave *sl);
   void     MarkBad(TSocket *s);

   void     ActivateAsyncInput();
   void     DeActivateAsyncInput();
   void     HandleAsyncInput(TSocket *s);

protected:
   TProof(); // For derived classes to use
   Int_t    Init(const char *masterurl, const char *conffile,
                 const char *confdir, Int_t loglevel);
   virtual Bool_t  StartSlaves();
   void            SetPlayer(TProofPlayer *player) { fPlayer = player; };
   TProofPlayer   *GetPlayer() const { return fPlayer; };
   virtual TProofPlayer *MakePlayer();
   TPluginHandler *GetProgressDialog() const { return fProgressDialog; };
   TList  *GetListOfActiveSlaves() const { return fActiveSlaves; }
   TSlave *CreateSlave(const char *host, Int_t port, const char *ord,
                       Int_t perf, const char *image, const char *workdir);
   TSlave *CreateSubmaster(const char *host, Int_t port,
                           const char *ord, const char *image,
                           const char *conffile, const char *msd);
   Int_t    Collect(ESlaves list = kActive);
   Int_t    Collect(TList *slaves);
   void     SetDSet(TDSet *dset) { fDSet = dset; }
   virtual void ValidateDSet(TDSet *dset);

public:
   TProof(const char *masterurl, const char *conffile = kPROOF_ConfFile,
          const char *confdir = kPROOF_ConfDir, Int_t loglevel = 0);
   virtual ~TProof();

   Int_t       Ping();
   Int_t       Exec(const char *cmd);
   Int_t       Process(TDSet *set, const char *selector,
                       Option_t *option = "", Long64_t nentries = -1,
                       Long64_t firstentry = 0, TEventList *evl = 0);
   Int_t       DrawSelect(TDSet *set, const char *varexp, const char *selection,
                          Option_t *option = "", Long64_t nentries = -1,
                          Long64_t firstentry = 0);

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
   Int_t       UploadPackage(const char *par, Int_t parallel = 1);

   const char *GetMaster() const { return fMaster; }
   const char *GetConfDir() const { return fConfDir; }
   const char *GetConfFile() const { return fConfFile; }
   const char *GetUser() const { return fUser; }
   const char *GetWorkDir() const { return fWorkDir; }
   const char *GetImage() const { return fImage; }
   const char *GetUrlProtocol() const { return fUrlProtocol; }
   Int_t       GetPort() const { return fPort; }
   Int_t       GetSecurity() const { return fSecContext->GetMethod(); }
   Int_t       GetRemoteProtocol() const { return fProtocol; }
   Int_t       GetClientProtocol() const { return kPROOF_Protocol; }
   Int_t       GetStatus() const { return fStatus; }
   Int_t       GetLogLevel() const { return fLogLevel; }
   Int_t       GetParallel() const;
   TList      *GetSlaveInfo();

   Long64_t    GetBytesRead() const { return fBytesRead; }
   Float_t     GetRealTime() const { return fRealTime; }
   Float_t     GetCpuTime() const { return fCpuTime; }

   Bool_t      IsFolder() const { return kTRUE; }
   Bool_t      IsMaster() const { return fMasterServ; }
   Bool_t      IsValid() const { return fValid; }
   Bool_t      IsParallel() const { return GetParallel() > 0 ? kTRUE : kFALSE; }

   void        AddFeedback(const char *name);
   void        RemoveFeedback(const char *name);
   void        ClearFeedback();
   void        ShowFeedback() const;
   TList      *GetFeedbackList() const;

   Bool_t      IsDataReady(Long64_t &totalbytes, Long64_t &bytesready);

   void        SetActive(Bool_t /*active*/ = kTRUE) { }

   void        Progress(Long64_t total, Long64_t processed); //*SIGNAL*
   void        Feedback(TList *objs); //*SIGNAL*

   TTree      *GetTreeHeader(TDSet *tdset);
   TList      *GetOutputNames();

   void        AddChain(TChain *chain);
   void        RemoveChain(TChain *chain);

   TDrawFeedback *CreateDrawFeedback();
   void           SetDrawFeedbackOption(TDrawFeedback *f, Option_t *opt);
   void           DeleteDrawFeedback(TDrawFeedback *f);

   ClassDef(TProof,0)  //PROOF control class
};


class TProofCondor : public TProof {

friend class TCondor;

private:
   TCondor *fCondor; //proxy for our Condor pool
   TTimer  *fTimer;  //timer for delayed Condor COD suspend

protected:
   Bool_t   StartSlaves();
   TString  GetJobAd();

public:
   TProofCondor(const char *masterurl, const char *conffile = kPROOF_ConfFile,
                const char *confdir = kPROOF_ConfDir, Int_t loglevel = 0);
   virtual ~TProofCondor();
   virtual void SetActive() { TProof::SetActive(); }
   virtual void SetActive(Bool_t active);

   ClassDef(TProofCondor,0) //PROOF control class for slaves allocated by condor
};


class TProofSuperMaster : public TProof {

friend class TProofPlayerSuperMaster;

protected:
   Bool_t StartSlaves();
   Int_t  Process(TDSet *set, const char *selector,
                  Option_t *option = "", Long64_t nentries = -1,
                  Long64_t firstentry = 0, TEventList *evl = 0);
   void   ValidateDSet(TDSet *dset);
   virtual TProofPlayer *MakePlayer();

public:
   TProofSuperMaster(const char *masterurl, const char *conffile = kPROOF_ConfFile,
                    const char *confdir = kPROOF_ConfDir, Int_t loglevel = 0);
   virtual ~TProofSuperMaster() { }

   ClassDef(TProofSuperMaster,0) //PROOF control class for making submasters
};

#endif
