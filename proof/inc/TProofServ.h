// @(#)root/proof:$Name:  $:$Id: TProofServ.h,v 1.4 2000/11/24 18:11:32 rdm Exp $
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


class TSocket;


class TProofServ : public TApplication {

private:
   TString     fService;          //service we are running, either "proofserv" or "proofslave"
   TString     fUser;             //user as which we run
   TString     fUserPass;         //encoded user and passwd info for slaves
   TString     fConfDir;          //directory containing cluster config information
   TString     fConfFile;         //file containing config information
   TString     fLogDir;           //directory containing log files
   TSocket    *fSocket;           //socket connection to client
   FILE       *fLogFile;          //log file
   Int_t       fProtocol;         //protocol version number
   Int_t       fOrdinal;          //slave ordinal number, -1 for master
   Int_t       fGroupId;          //slave unique id in the active slave group
   Int_t       fGroupSize;        //size of the active slave group
   Int_t       fLogLevel;         //debug logging level
   Bool_t      fMasterServ;       //true if we are a master server
   Int_t       fNcmd;             //command history number
   Bool_t      fInterrupt;        //if true macro execution will be stopped
   Float_t     fRealTime;         //real time spent executing commands
   Float_t     fCpuTime;          //CPU time spent executing commands
   Stat_t      fEntriesProcessed; //total number of entries processed (obtained via GetNextPacket)

   void        Setup();
   void        RedirectOutput();
   Int_t       CatMotd();
   void        Info(const char *method, const char *msgfmt, ...) const; // should be in TObject

public:
   TProofServ(int *argc, char **argv);
   virtual ~TProofServ();

   const char *GetService() const { return fService.Data(); }
   const char *GetConfDir() const { return fConfDir.Data(); }
   const char *GetConfFile() const { return fConfFile.Data(); }
   const char *GetUser() const { return fUser.Data(); }
   Int_t       GetProtocol() const { return fProtocol; }
   Int_t       GetOrdinal() const { return fOrdinal; }
   Int_t       GetGroupId() const { return fGroupId; }
   Int_t       GetGroupSize() const { return fGroupSize; }
   Int_t       GetLogLevel() const { return fLogLevel; }
   TSocket    *GetSocket() const { return fSocket; }
   Float_t     GetRealTime() const { return fRealTime; }
   Float_t     GetCpuTime() const { return fCpuTime; }
   void        GetOptions(int *argc, char **argv);

   void        HandleSocketInput();
   void        HandleUrgentData();
   void        HandleSigPipe();
   void        Interrupt() { fInterrupt = kTRUE; }
   Bool_t      IsMaster() const { return fMasterServ; }
   Bool_t      IsParallel() const;

   void        Run(Bool_t retrn = kFALSE);

   void        Print(Option_t *option="");

   TObject    *Get(const char *namecycle);
   Stat_t      GetEntriesProcessed() const { return fEntriesProcessed; }
   void        GetLimits(Int_t dim, Int_t nentries, Int_t *nbins, Double_t *vmin, Double_t *vmax);
   Bool_t      GetNextPacket(Int_t &nentries, Stat_t &firstentry);
   void        Reset(const char *dir);
   Int_t       ReceiveFile(const char *file, Bool_t bin, Long_t size);
   void        SendLogFile(Int_t status = 0);
   void        SendStatus();

   void        Terminate(int status);

   static Bool_t      IsActive();
   static TProofServ *This();

   ClassDef(TProofServ,0)  //PROOF Server Application Interface
};

R__EXTERN TProofServ *gProofServ;

#endif
