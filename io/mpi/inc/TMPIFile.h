// @(#)root/io:$Id$
// Author: Amit Bashyal, August 2018

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMPIFile
#define ROOT_TMPIFile

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMPIFile                                                             //
//                                                                      //
// File Object derived from TMemFile.                                   //
//                                                                      //
// The TMPIFile class provides the ability to aggregate data across     //
// many MPI ranks on a cluster into a single file. This can be useful   //
// on HPCs or large clusters. The user must control the syncronization  //
// of the data across multiple ranks via the Sync() function.           //
// When Sync() is called, this triggers objects in the TFile space to   //
// be communicated over MPI to a master writer which combines the data  //
// before writing it to file.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMPIClientInfo.h"
#include "TBits.h"
#include "TFileMerger.h"
#include "TMemFile.h"

#include <mpi.h>

#include <vector>

class TMPIFile : public TMemFile {

private:
   Int_t fEndProcess = 0; // collector tracks number of exited processes
   Int_t fSplitLevel;     // number of collectors to use
   Int_t fMPIColor;       // used by MPI ranks to track which collector to use

   Int_t fMPIGlobalRank; // global rank number
   Int_t fMPIGlobalSize; // total ranks
   Int_t fMPILocalRank;  // rank number in sub communicator
   Int_t fMPILocalSize;  // number of ranks in sub communicator

   MPI_Comm fSubComm;       // sub communicator handle
   MPI_Request fMPIRequest; // request place holder

   TString fMPIFilename; // output filename, only used by collector

   char *fSendBuf = 0; // message buffer, only used by worker

   struct ParallelFileMerger : public TObject {
   private:
      using ClientColl_t = std::vector<TMPIClientInfo>;

      TString fFilename;
      TBits fClientsContact;
      UInt_t fNClientsContact;
      ClientColl_t fClients;
      TTimeStamp fLastMerge;
      TFileMerger fMerger;

      static void DeleteObject(TDirectory *dir, Bool_t withReset);
      
   public:
      ParallelFileMerger(const char *filename, Int_t compression_settings, Bool_t writeCache = kFALSE);
      virtual ~ParallelFileMerger();

      ULong_t Hash() const { return fFilename.Hash(); };
      const char *GetName() const { return fFilename; };

      static Bool_t NeedInitialMerge(TDirectory *dir);

      Bool_t InitialMerge(TFile *input);
      Bool_t Merge();
      Bool_t NeedMerge(Float_t clientThreshold);
      Bool_t NeedFinalMerge() { return fClientsContact.CountBits() > 0; };
      void RegisterClient(UInt_t clientID, TFile *file);
   };

   void SetOutputName();
   void CheckSplitLevel();
   void SplitMPIComm();
   void UpdateEndProcess();

   Bool_t IsReceived();

public:
   TMPIFile(const char *name, char *buffer, Long64_t size = 0, Option_t *option = "", Int_t split = 1,
            const char *ftitle = "", Int_t compress = 4);
   TMPIFile(const char *name, Option_t *option = "", Int_t split = 1, const char *ftitle = "",
            Int_t compress = 4); // no complete implementation
   virtual ~TMPIFile();

   // some functions on MPI information
   Int_t GetMPIGlobalSize() const { return fMPIGlobalSize; };
   Int_t GetMPILocalSize() const { return fMPILocalSize; };
   Int_t GetMPIGlobalRank() const { return fMPIGlobalRank; };
   Int_t GetMPILocalRank() const { return fMPILocalRank; };
   Int_t GetMPIColor() const { return fMPIColor; };
   Int_t GetSplitLevel() const { return fSplitLevel; };

   TString GetMPIFilename() const { return fMPIFilename; };

   // Collector Functions
   void RunCollector(Bool_t cache = kFALSE);
   Bool_t IsCollector();

   // Sender Functions
   void CreateBufferAndSend();
   // Empty Buffer to signal the end of job...
   void CreateEmptyBufferAndSend();
   void Sync();

   // Finalize work and save output in disk.
   void Close(Option_t *option = "") final;

   ClassDef(TMPIFile, 0)
};
#endif
