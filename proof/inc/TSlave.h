// @(#)root/proof:$Name:  $:$Id: TSlave.h,v 1.12 2004/10/15 17:10:13 rdm Exp $
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

class TSlave : public TObject {

friend class TProof;

private:
   enum ESlaveType {
      kMaster,
      kSlave
   };

   TString       fName;      //slave's hostname
   TString       fImage;     //slave's image name
   TString       fProofWorkDir; //base proofserv working directory (info obtained from slave)
   TString       fWorkDir;   //slave's working directory (info obtained from slave)
   TString       fUser;      //slave's user id
   Int_t         fPort;      //slave's port number
   TString       fOrdinal;   //slave's ordinal number
   Int_t         fPerfIdx;   //relative CPU performance index
   TSecContext  *fSecContext;//security context of the related authentication
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

   TSlave() { fOrdinal = "-1"; fSocket = 0; fProof = 0; }
   TSlave(const TSlave &s) : TObject(s) { }
   void operator=(const TSlave &) { }

   TSlave(const char *host, Int_t port, const char *ord, Int_t perf,
          const char *image, TProof *proof, ESlaveType stype,
          const char *workdir, const char *conffile, const char *msd);

public:
   virtual ~TSlave();

   void           Close(Option_t *opt = "");

   Int_t          Compare(const TObject *obj) const;
   Bool_t         IsSortable() const { return kTRUE; }

   const char    *GetName() const { return fName.Data(); }
   const char    *GetImage() const { return fImage.Data(); }
   const char    *GetProofWorkDir() const { return fProofWorkDir.Data(); }
   const char    *GetWorkDir() const { return fWorkDir.Data(); }
   const char    *GetUser() const { return fUser.Data(); }
   Int_t          GetPort() const { return fPort; }
   const TString &GetOrdinal() const { return fOrdinal; }
   Int_t          GetPerfIdx() const { return fPerfIdx; }
   Int_t          GetSecurity() const { return fSecContext->GetMethod(); }
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

   ClassDef(TSlave,0)  //PROOF slave server
};

#endif
