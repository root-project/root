// @(#)root/proof:$Name:  $:$Id: TProof.h,v 1.13 2002/02/07 18:06:47 rdm Exp $
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

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_MessageTypes
#include "MessageTypes.h"
#endif


class TList;
class TMessage;
class TSocket;
class TMonitor;
class TFile;
class TSignalHandler;
class TSlave;
class TProofServ;
class TProofInputHandler;
class TProofInterruptHandler;
class TProofPlayer;
class TProofPlayerRemote;
class TDSet;
class TEventList;
class TTree;  // obsolete


// PROOF magic constants
const Int_t       kPROOF_Protocol = 1;            // protocol version number
const Int_t       kPROOF_Port     = 1093;         // IANA registered PROOF port
const char* const kPROOF_ConfFile = "proof.conf"; // default config file
const char* const kPROOF_ConfDir  = "/usr/local/root";  // default config dir
const char* const kPROOF_WorkDir  = "~/proof";    // default working directory


class TProof : public TObject {

friend class TProofServ;
friend class TProofInputHandler;
friend class TProofInterruptHandler;
friend class TProofPlayer;
friend class TProofPlayerRemote;
friend class TSlave;

private:
   TString   fMaster;        //name of master server (use "" if this is a master)
   TString   fConfDir;       //directory containing cluster config information
   TString   fConfFile;      //file containing config information
   TString   fWorkDir;       //current work directory on remote servers
   TString   fUser;          //user under which to run
   TString   fPasswd;        //user password
   TString   fImage;         //master's image name
   Int_t     fPort;          //port we are connected to (proofd = 1093)
   Int_t     fSecurity;      //security level used to connect to master server
   Int_t     fProtocol;      //protocol version number
   Int_t     fLogLevel;      //server debug logging level
   Int_t     fStatus;        //remote return status (part of kPROOF_LOGDONE)
   Int_t     fParallel;      //number of active slaves (only set on client, on server use fActiveSlaves)
   Bool_t    fMasterServ;    //true if we are a master server
   Bool_t    fSendGroupView; //if true send new group view
   TList    *fSlaves;        //list of all slave servers as in config file
   TList    *fActiveSlaves;  //list of active slaves (subset of all slaves)
   TList    *fUniqueSlaves;  //list of all active slaves with unique file systems
   TList    *fBadSlaves;     //dead slaves (subset of all slaves)
   TMonitor *fAllMonitor;    //monitor activity on all valid slave sockets
   TMonitor *fActiveMonitor; //monitor activity on all active slave sockets
   TMonitor *fUniqueMonitor; //monitor activity on all unique slave sockets
   Double_t  fBytesRead;     //bytes read by all slaves during the session
   Float_t   fRealTime;      //realtime spent by all slaves during the session
   Float_t   fCpuTime;       //CPU time spent by all slaves during the session
   Int_t     fLimits;        //used by Limits()
   TSignalHandler *fIntHandler; //interrupt signal handler (ctrl-c)
   TProofPlayer   *fPlayer;     //current player

   enum ESlaves { kAll, kActive, kUnique };
   enum EUrgent { kHardInterrupt = 1, kSoftInterrupt, kShutdownInterrupt };

   TProof() { fSlaves = fActiveSlaves = fBadSlaves = 0; }
   TProof(const TProof &);           // not implemented
   void operator=(const TProof &);   // idem

   Int_t    Init(const char *masterurl, const char *conffile,
                 const char *confdir, Int_t loglevel);

   Int_t    Exec(const char *cmd, ESlaves list);
   Int_t    SendCommand(const char *cmd, ESlaves list = kActive);
   Int_t    SendCurrentState(ESlaves list = kActive);
   Int_t    SendFile(const char *file, Bool_t bin = kTRUE, ESlaves list = kUnique);
   Int_t    SendObject(const TObject *obj, ESlaves list = kActive);
   Int_t    SendGroupView();
   Int_t    SendInitialState();
   Int_t    SendPrint();
   Int_t    Ping(ESlaves list);
   void     Interrupt(EUrgent type, ESlaves list = kActive);
   void     ConnectFiles();
   Int_t    ConnectFile(const TFile *file);
   Int_t    DisConnectFile(const TFile *file);
   void     AskStatus();
   Int_t    GoParallel(Int_t nodes);
   void     Limits(TSocket *s, TMessage &mess);
   void     RecvLogFile(TSocket *s, Int_t size);

   Int_t    Broadcast(const TMessage &mess, ESlaves list = kActive);
   Int_t    Broadcast(const char *mess, Int_t kind = kMESS_STRING, ESlaves list = kActive);
   Int_t    Broadcast(Int_t kind, ESlaves list = kActive) { return Broadcast(0, kind, list); }
   Int_t    BroadcastObject(const TObject *obj, Int_t kind = kMESS_OBJECT, ESlaves list = kActive);
   Int_t    BroadcastRaw(const void *buffer, Int_t length, ESlaves list = kActive);
   Int_t    Collect(ESlaves list = kActive);
   Int_t    Collect(const TSlave *sl);
   Int_t    Collect(TMonitor *mon);

   void     FindUniqueSlaves();
   TSlave  *FindSlave(TSocket *s) const;
   TList   *GetListOfSlaves() const { return fSlaves; }
   TList   *GetListOfActiveSlaves() const { return fActiveSlaves; }
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

   void           SetPlayer(TProofPlayer *player) { fPlayer = player; };
   TProofPlayer  *GetPlayer() const { return fPlayer; };

public:
   TProof(const char *masterurl, const char *conffile = kPROOF_ConfFile,
          const char *confdir = kPROOF_ConfDir, Int_t loglevel = 1);
   virtual ~TProof();

   Int_t       Ping();
   void        Loop(TTree * /*tree*/) { }  // obsolete
   Int_t       Exec(const char *cmd);
   Int_t       Process(TDSet *set, const char *selector, Int_t nentries = -1,
                       Int_t first = 0, TEventList *evl = 0);
   void        AddInput(TObject *obj);
   void        ClearInput();
   TObject    *GetOutput(const char *name);
   TList      *GetOutputList();
   Int_t       SetParallel(Int_t nodes = 9999);
   void        SetLogLevel(Int_t level);

   void        Close(Option_t *option="");
   void        Print(Option_t *option="") const;

   const char *GetMaster() const { return fMaster; }
   const char *GetConfDir() const { return fConfDir; }
   const char *GetConfFile() const { return fConfFile; }
   const char *GetUser() const { return fUser; }
   const char *GetWorkDir() const { return fWorkDir; }
   const char *GetImage() const { return fImage; }
   Int_t       GetPort() const { return fPort; }
   Int_t       GetProtocol() const { return fProtocol; }
   Int_t       GetStatus() const { return fStatus; }
   Int_t       GetLogLevel() const { return fLogLevel; }
   Int_t       GetParallel() const;

   Double_t    GetBytesRead() const { return fBytesRead; }
   Float_t     GetRealTime() const { return fRealTime; }
   Float_t     GetCpuTime() const { return fCpuTime; }

   Bool_t      IsMaster() const { return fMasterServ; }
   Bool_t      IsValid() const { return GetNumberOfActiveSlaves() > 0 ? kTRUE : kFALSE; }
   Bool_t      IsParallel() const { return GetParallel() > 1 ? kTRUE : kFALSE; }

   static Bool_t  IsActive();
   static TProof *This();

   ClassDef(TProof,0)  //PROOF control class
};

R__EXTERN TProof *gProof;

#endif
