// @(#)root/proof:$Name:  $:$Id: TSlave.h,v 1.17 2005/08/15 15:57:18 rdm Exp $
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
#ifndef ROOT_TSocket
#include "TSocket.h"
#endif

class TFileHandler;
class TProof;

// Hook to external function setting up authentication related stuff
// for old versions. For backward compatibility.
typedef Int_t (*OldSlaveAuthSetup_t)(TSocket *, Bool_t, TString, TString);


class TSlave : public TObject {

friend class TProof;

public:
   enum ESlaveType {
      kMaster,
      kSlave
   };

private:
   TString       fName;      //slave's hostname
   TString       fImage;     //slave's image name
   TString       fProofWorkDir; //base proofserv working directory (info obtained from slave)
   TString       fWorkDir;   //slave's working directory (info obtained from slave)
   TString       fUser;      //slave's user id
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

   TSlave(const TSlave &s) : TObject(s) { }
   void operator=(const TSlave &) { }

   TSlave(const char *host, Int_t port, const char *ord, Int_t perf,
          const char *image, TProof *proof, ESlaveType stype,
          const char *workdir, const char *msd);

   void  Init(TSocket *s, ESlaveType stype);
   Int_t OldAuthSetup(Bool_t master, TString wconf);

   static TSlave *Create(const char *host, Int_t port, const char *ord, Int_t perf,
                         const char *image, TProof *proof, ESlaveType stype,
                         const char *workdir, const char *msd);

protected:
   TSlave();
   virtual void  Init(const char *host, Int_t port, ESlaveType stype);
   virtual void  Interrupt(Int_t type);
   virtual Int_t Ping();

public:
   virtual ~TSlave();

   void           Close(Option_t *opt = "");

   Int_t          Compare(const TObject *obj) const;
   Bool_t         IsSortable() const { return kTRUE; }

   const char    *GetName() const { return fName; }
   const char    *GetImage() const { return fImage; }
   const char    *GetProofWorkDir() const { return fProofWorkDir; }
   const char    *GetWorkDir() const { return fWorkDir; }
   const char    *GetUser() const { return fUser; }
   Int_t          GetPort() const { return fPort; }
   const char    *GetOrdinal() const { return fOrdinal; }
   Int_t          GetPerfIdx() const { return fPerfIdx; }
   Int_t          GetProtocol() const { return fProtocol; }
   TSocket       *GetSocket() const { return fSocket; }
   TProof        *GetProof() const { return fProof; }
   Long64_t       GetBytesRead() const { return fBytesRead; }
   Float_t        GetRealTime() const { return fRealTime; }
   Float_t        GetCpuTime() const { return fCpuTime; }
   ESlaveType     GetSlaveType() const { return fSlaveType; }
   Int_t          GetStatus() const { return fStatus; }
   Int_t          GetParallel() const { return fParallel; }
   TString        GetMsd() const { return fMsd; }
   TFileHandler  *GetInputHandler() const { return fInput; }
   void           SetInputHandler(TFileHandler *ih);

   Bool_t         IsValid() const { return fSocket ? kTRUE : kFALSE; }

   void           Print(Option_t *option="") const;

   virtual void   SetupServ(ESlaveType stype, const char *conffile);

   ClassDef(TSlave,0)  //PROOF slave server
};

#endif
