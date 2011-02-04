// @(#)root/proof:$Id$
// Author: Fons Rademakers   14/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSlave
#define ROOT_TSlave


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSlave                                                               //
//                                                                      //
// This class describes a PROOF slave server.                           //
// It contains information like the slaves host name, ordinal number,   //
// performance index, socket, etc. Objects of this class can only be    //
// created via TProof member functions.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TFileHandler;
class TObjString;
class TProof;
class TSlave;
class TSocket;

// Special type for the hook to external function setting up authentication
// related stuff for old versions. For backward compatibility.
typedef Int_t (*OldSlaveAuthSetup_t)(TSocket *, Bool_t, TString, TString);

// Special type for the hook to the TSlave constructor, needed to avoid
// using the plugin manager
typedef TSlave *(*TSlave_t)(const char *url, const char *ord, Int_t perf,
                            const char *image, TProof *proof, Int_t stype,
                            const char *workdir, const char *msd);

class TSlave : public TObject {

friend class TProof;
friend class TProofLite;
friend class TSlaveLite;
friend class TXSlave;

public:

   enum ESlaveType { kMaster, kSlave };
   enum ESlaveStatus { kInvalid, kActive, kInactive };

private:

   static TSlave_t fgTXSlaveHook;

   TSlave(const TSlave &s) : TObject(s) { }
   TSlave(const char *host, const char *ord, Int_t perf,
          const char *image, TProof *proof, Int_t stype,
          const char *workdir, const char *msd);

   Int_t  OldAuthSetup(Bool_t master, TString wconf);
   void   Init(const char *host, Int_t port, Int_t stype);
   void   operator=(const TSlave &) { }

   static TSlave *Create(const char *url, const char *ord, Int_t perf,
                         const char *image, TProof *proof, Int_t stype,
                         const char *workdir, const char *msd);

protected:
   TString       fName;      //slave's hostname
   TString       fImage;     //slave's image name
   TString       fProofWorkDir; //base proofserv working directory (info obtained from slave)
   TString       fWorkDir;   //slave's working directory (info obtained from slave)
   TString       fUser;      //slave's user id
   TString       fGroup;     //slave's group id
   Int_t         fPort;      //slave's port number
   TString       fOrdinal;   //slave's ordinal number
   Int_t         fPerfIdx;   //relative CPU performance index
   Int_t         fProtocol;  //slave's protocol level
   TSocket      *fSocket;    //socket to slave
   TProof       *fProof;     //proof cluster to which slave belongs
   TFileHandler *fInput;     //input handler related to this slave
   Long64_t      fBytesRead; //bytes read by slave (info is obtained from slave)
   Float_t       fRealTime;  //real time spent executing commands (info obtained from slave)
   Float_t       fCpuTime;   //CPU time spent executing commands (info obtained from slave)
   ESlaveType    fSlaveType; //type of slave (either kMaster or kSlave)
   Int_t         fStatus;    //remote return status
   Int_t         fParallel;  //number of active slaves
   TString       fMsd;       //mass storage domain of slave
   TString       fSessionTag;//unique tag for ths worker process

   TString       fROOTVers;  //ROOT version run by worker
   TString       fArchComp;  //Build architecture, compiler on worker (e.g. linux-gcc345)

   TSlave();
   virtual void  FlushSocket() { }
   void          Init(TSocket *s, Int_t stype);
   virtual void  Interrupt(Int_t type);
   virtual Int_t Ping();
   virtual TObjString *SendCoordinator(Int_t kind, const char *msg = 0, Int_t int2 = 0);
   virtual Int_t SendGroupPriority(const char * /*grp*/, Int_t /*priority*/) { return 0; }
   virtual void  SetAlias(const char *alias);
   void          SetSocket(TSocket *s) { fSocket = s; }
   virtual void  SetStatus(Int_t st) { fStatus = st; }
   virtual void  StopProcess(Bool_t abort, Int_t timeout);

public:
   virtual ~TSlave();

   virtual void   Close(Option_t *opt = "");

   Int_t          Compare(const TObject *obj) const;
   Bool_t         IsSortable() const { return kTRUE; }

   const char    *GetName() const { return fName; }
   const char    *GetImage() const { return fImage; }
   const char    *GetProofWorkDir() const { return fProofWorkDir; }
   const char    *GetWorkDir() const { return fWorkDir; }
   const char    *GetUser() const { return fUser; }
   const char    *GetGroup() const { return fGroup; }
   Int_t          GetPort() const { return fPort; }
   const char    *GetOrdinal() const { return fOrdinal; }
   Int_t          GetPerfIdx() const { return fPerfIdx; }
   Int_t          GetProtocol() const { return fProtocol; }
   TSocket       *GetSocket() const { return fSocket; }
   TProof        *GetProof() const { return fProof; }
   Long64_t       GetBytesRead() const { return fBytesRead; }
   Float_t        GetRealTime() const { return fRealTime; }
   Float_t        GetCpuTime() const { return fCpuTime; }
   Int_t          GetSlaveType() const { return (Int_t)fSlaveType; }
   Int_t          GetStatus() const { return fStatus; }
   Int_t          GetParallel() const { return fParallel; }
   const char    *GetMsd() const { return fMsd; }
   const char    *GetSessionTag() const { return fSessionTag; }
   TFileHandler  *GetInputHandler() const { return fInput; }
   void           SetInputHandler(TFileHandler *ih);

   const char    *GetArchCompiler() const { return fArchComp; }
   const char    *GetROOTVersion() const { return fROOTVers; }

   virtual Bool_t IsValid() const { return fSocket ? kTRUE : kFALSE; }

   virtual void   Print(Option_t *option="") const;

   virtual Int_t  SetupServ(Int_t stype, const char *conffile);

   virtual void   SetInterruptHandler(Bool_t /* on */) { }

   void           SetArchCompiler(const char *ac) { fArchComp = ac; }
   void           SetROOTVersion(const char *rv) { fROOTVers = rv; }

   void           SetSessionTag(const char *st) { fSessionTag = st; }

   static void    SetTXSlaveHook(TSlave_t xslavehook);

   virtual void   Touch() { }

   ClassDef(TSlave,0)  //PROOF slave server
};

#endif
