// @(#)root/proof:$Id$
// Author: G. Ganis March 2008

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofLite
#define ROOT_TProofLite


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofLite                                                           //
//                                                                      //
// This class starts a PROOF session on the local machine: no daemons,  //
// client and master merged, communications via UNIX-like sockets.      //
// By default the number of workers started is NumberOfCores+1; a       //
// different number can be forced on construction.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TProof
#include "TProof.h"
#endif

class TDSet;
class TList;
class TQueryResultManager;
class TDataSetManager;
class TProofLockPath;
class TProofMgr;
class TProofQueryResult;
class TServerSocket;
class TSelector;

class TProofLite : public TProof {

friend class TProofPlayerLite;

private:
   Int_t    fNWorkers;    // Number of workers
   TString  fSandbox;     // PROOF sandbox root dir
   TString  fCacheDir;    // Directory containing cache of user files
   TString  fQueryDir;    // Directory containing query results and status
   TString  fDataSetDir;  // Directory containing info about known data sets
   TString  fSockPath;    // UNIX socket path for communication with workers
   TServerSocket *fServSock; // Server socket to accept call backs
   Bool_t   fForkStartup; // Startup N-1 workers forking the first worker

   TProofLockPath *fCacheLock; // Cache dir locker
   TProofLockPath *fQueryLock; // Query dir locker
   TQueryResultManager *fQMgr; // Query-result manager

   TDataSetManager *fDataSetManager; // Dataset manager

   static Int_t fgWrksMax; // Max number of workers

   TProofLite(const TProofLite &);        // not implemented
   void operator=(const TProofLite &);    // idem

   Int_t CleanupSandbox();
   Int_t CreateSandbox();
   void FindUniqueSlaves();
   void  NotifyStartUp(const char *action, Int_t done, Int_t tot);
   Int_t SetProofServEnv(const char *ord);
   Int_t InitDataSetManager();
   Int_t RegisterDataSets(TList *in, TList *out);
   void  ResolveKeywords(TString &s, const char *logfile);

   void SendInputDataFile();

protected:
   TProofLite() : TProof() { } // For derived classes to use

   Int_t CreateSymLinks(TList *files);
   Int_t Init(const char *masterurl, const char *conffile,
               const char *confdir, Int_t loglevel,
               const char *alias = 0);
   TProofQueryResult *MakeQueryResult(Long64_t nent, const char *opt,
                                      Long64_t fst, TDSet *dset,
                                      const char *selec);
   void SetQueryRunning(TProofQueryResult *pq);
   Int_t SetupWorkers(Int_t opt = 0, TList *wrks = 0);
   Int_t CopyMacroToCache(const char *macro, Int_t headerRequired = 0,
                          TSelector **selector = 0, Int_t opt = 0);

public:
   TProofLite(const char *masterurl, const char *conffile = kPROOF_ConfFile,
              const char *confdir = kPROOF_ConfDir, Int_t loglevel = 0,
              const char *alias = 0, TProofMgr *mgr = 0);
   virtual ~TProofLite();

   void Print(Option_t *option="") const;

   Long64_t DrawSelect(TDSet *dset, const char *varexp,
                       const char *selection = "",
                       Option_t *option = "", Long64_t nentries = -1,
                       Long64_t firstentry = 0);
   Long64_t Process(TDSet *dset, const char *sel, Option_t *o = "",
                    Long64_t nent = -1, Long64_t fst = 0);
   Long64_t Process(TFileCollection *fc, const char *sel, Option_t *o = "",
                    Long64_t nent = -1, Long64_t fst = 0)
                    { return TProof::Process(fc, sel, o, nent, fst); }
   Long64_t Process(const char *dsname, const char *sel, Option_t *o = "",
                    Long64_t nent = -1, Long64_t fst = 0, TObject *enl = 0)
                    { return TProof::Process(dsname, sel, o, nent, fst, enl); }
   Long64_t Process(const char *sel, Long64_t nent, Option_t *o = "")
                    { return TProof::Process(sel, nent, o); }

   // Cache management
   void  ShowCache(Bool_t all = kFALSE);
   void  ClearCache(const char *file = 0);
   Int_t Load(const char *macro, Bool_t notOnClient = kFALSE, Bool_t uniqueOnly = kTRUE,
              TList *wrks = 0);

   // Query management
   TList *GetListOfQueries(Option_t *opt = "");
   Int_t Remove(const char *ref, Bool_t all);

   // Dataset handling
   Bool_t   RegisterDataSet(const char *dsName, TFileCollection *ds, const char *opt = "");
   Bool_t   ExistsDataSet(const char *uri);
   TMap    *GetDataSets(const char *uri = "", const char * = 0);
   void     ShowDataSets(const char *uri = "", const char * = 0);
   TFileCollection *GetDataSet(const char *uri, const char * = 0);
   Int_t    RemoveDataSet(const char *uri, const char * = 0);
   Int_t    VerifyDataSet(const char *uri, const char * = 0);
   Int_t    SetDataSetTreeName( const char *dataset, const char *treename);
   void     ShowDataSetCache(const char *dataset = 0);
   void     ClearDataSetCache(const char *dataset = 0);

   // Browsing
   TTree *GetTreeHeader(TDSet *tdset);

   static Int_t GetNumberOfWorkers(const char *url = 0);

   ClassDef(TProofLite,0)  //PROOF-Lite control class
};

#endif
