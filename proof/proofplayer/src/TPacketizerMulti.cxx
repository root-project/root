// @(#)root/proofplayer:$Id$
// Author: G. Ganis Jan 2010

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizerMulti                                                     //
//                                                                      //
// This class allows to do multiple runs in the same query; each run    //
// can be a, for example, different dataset or the same dataset with    //
// entry list.                                                          //
// The multiple packetizer conatins a list of packetizers which are     //
// processed in turn.                                                   //
// The bit TSelector::kNewRun is set in the TSelector object when a new //
// packetizer is used.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TPacketizerMulti.h"

#include "TClass.h"
#include "TDSet.h"
#include "TError.h"
#include "TFileInfo.h"
#include "TList.h"
#include "TMap.h"
#include "TMethodCall.h"
#include "TProof.h"
#include "TProofDebug.h"

ClassImp(TPacketizerMulti)

//______________________________________________________________________________
TPacketizerMulti::TPacketizerMulti(TDSet *dset, TList *wrks,
                                   Long64_t first, Long64_t num,
                                   TList *input, TProofProgressStatus *st)
                 : TVirtualPacketizer(input, st)
{
   // Constructor

   PDB(kPacketizer,1) Info("TPacketizerMulti",
                           "enter (first %lld, num %lld)", first, num);
   fValid = kFALSE;
   fPacketizersIter = 0;
   fCurrent = 0;
   fAssignedPack = 0;

   // Check inputs
   if (!dset || !wrks || !input || !st) {
      Error("TPacketizerMulti", "invalid inputs: dset:%p wrks:%p input:%p st:%p",
                                dset, wrks, input, st);
      return;
   }
   // Create the list
   fPacketizers = new TList;

   // We do not want progress timers from the single packetizers
   TNamed *progTimerFlag = new TNamed("PROOF_StartProgressTimer", "no");
   input->Add(progTimerFlag);

   fTotalEntries = 0;
   TVirtualPacketizer *packetizer = 0;
   // Simple or multi?
   if (!(dset->TestBit(TDSet::kMultiDSet))) {
      if ((packetizer = CreatePacketizer(dset, wrks, first, num, input, st))) {
         fPacketizers->Add(packetizer);
         fTotalEntries = packetizer->GetTotalEntries();
      } else {
         Error("TPacketizerMulti", "problems initializing packetizer for single dataset");
         input->Remove(progTimerFlag);
         delete progTimerFlag;
         return;
      }
   } else {
      // Iterate on the datasets
      TIter nxds(dset->GetListOfElements());
      TDSet *ds = 0;
      while ((ds = (TDSet *)nxds())) {
         if ((packetizer = CreatePacketizer(ds, wrks, first, num, input, st))) {
            fPacketizers->Add(packetizer);
            fTotalEntries += packetizer->GetTotalEntries();
         } else {
            Error("TPacketizerMulti", "problems initializing packetizer for dataset '%s'", ds->GetName());
         }
      }
   }
   // Cleanup temporary additions to the input list
   input->Remove(progTimerFlag);
   delete progTimerFlag;

   // If no valid packetizer could be initialized we fail
   if (fPacketizers->GetSize() <= 0) {
      Error("TPacketizerMulti", "no valid packetizer could be initialized - aborting");
      SafeDelete(fPacketizers);
      return;
   } else {
      Info("TPacketizerMulti", "%d packetizer(s) have been successfully initialized (%lld events in total)",
                               fPacketizers->GetSize(), fTotalEntries);
      // To avoid problems with counters we must set the total entries in each packetizer
      TIter nxp(fPacketizers);
      while ((packetizer = (TVirtualPacketizer *) nxp()))
         packetizer->SetTotalEntries(fTotalEntries);
   }

   // Create the interator
   fPacketizersIter = new TIter(fPacketizers);

   // Set the current the first
   if (!(fCurrent = (TVirtualPacketizer *) fPacketizersIter->Next())) {
      // Weird
      Error("TPacketizerMulti", "could not point to the first valid packetizer");
      fPacketizers->SetOwner(kTRUE);
      SafeDelete(fPacketizers);
      SafeDelete(fPacketizersIter);
      return;
   }

   // Create map
   fAssignedPack = new TMap;

   // Ok, everything went fine
   fValid = kTRUE;

   PDB(kPacketizer,1) Info("TPacketizerMulti", "done");
}

//______________________________________________________________________________
TPacketizerMulti::~TPacketizerMulti()
{
   // Destructor.

   if (fPacketizers) {
      fPacketizers->SetOwner(kTRUE);
      SafeDelete(fPacketizers);
   }
   SafeDelete(fPacketizers);
   fCurrent = 0;
   if (fAssignedPack) {
      fAssignedPack->SetOwner(kFALSE);
      SafeDelete(fAssignedPack);
   }
   SafeDelete(fPacketizersIter);
}

//______________________________________________________________________________
TDSetElement *TPacketizerMulti::GetNextPacket(TSlave *wrk, TMessage *r)
{
   // Get next packet from the current packetizer.
   // If the current packetizer is done, move to next.
   // Retun null when all packetizers are done.

   TDSetElement *elem = 0;

   // Must be valid
   if (!fValid) return elem;

   // Point to the packetizer last used for thsi worker
   TVirtualPacketizer *lastPacketizer = dynamic_cast<TVirtualPacketizer *>(fAssignedPack->GetValue(wrk));
   if (lastPacketizer && lastPacketizer != fCurrent) {
      PDB(kPacketizer,2)
         Info("GetNextPacket", "%s: asking old packetizer %p ... ", wrk->GetOrdinal(), lastPacketizer);
      if ((elem = lastPacketizer->GetNextPacket(wrk, r))) return elem;
      if (fCurrent) {
         // Transfer the status info
         TVirtualSlaveStat *oldstat = dynamic_cast<TVirtualSlaveStat *>(lastPacketizer->GetSlaveStats()->GetValue(wrk));
         TVirtualSlaveStat *curstat = dynamic_cast<TVirtualSlaveStat *>(fCurrent->GetSlaveStats()->GetValue(wrk));
         if (oldstat && curstat)
            *(curstat->GetProgressStatus()) += *(oldstat->GetProgressStatus());
      }
   }

   // Need something to be processed
   if (!fCurrent) {
      HandleTimer(0);   // Send last timer message
      return elem;
   }

   // Get the next packet from the current packetizer
   PDB(kPacketizer,2)
      Info("GetNextPacket", "%s: asking current packetizer %p ... ", wrk->GetOrdinal(), fCurrent);
   if (!(elem = fCurrent->GetNextPacket(wrk, r))) {
      // We need to transfer the status info if we change packetizer now
      TMap *oldStats = (lastPacketizer && lastPacketizer == fCurrent) ? lastPacketizer->GetSlaveStats() : 0;
      // If the packetizer is done, move to next
      fCurrent = (TVirtualPacketizer *) fPacketizersIter->Next();
      if (fCurrent) {
         // Transfer the status info
         if (oldStats) {
            TVirtualSlaveStat *oldstat = dynamic_cast<TVirtualSlaveStat *>(oldStats->GetValue(wrk));
            TVirtualSlaveStat *curstat = dynamic_cast<TVirtualSlaveStat *>(fCurrent->GetSlaveStats()->GetValue(wrk));
            if (oldstat && curstat)
               *(curstat->GetProgressStatus()) += *(oldstat->GetProgressStatus());
         }
         PDB(kPacketizer,2)
            Info("GetNextPacket", "%s: asking new packetizer %p ... ", wrk->GetOrdinal(), fCurrent);
         elem = fCurrent->GetNextPacket(wrk, r);
      }
   }
   if (fCurrent) {
      // Save the packetizer
      TPair *pair = dynamic_cast<TPair *>(fAssignedPack->FindObject(wrk));
      if (pair) {
         pair->SetValue(fCurrent);
      } else {
         fAssignedPack->Add(wrk, fCurrent);
      }
      PDB(kPacketizer,2)
         Info("GetNextPacket", "assigned packetizer %p to %s (check: %p)",
                               fCurrent,  wrk->GetOrdinal(), fAssignedPack->GetValue(wrk));
   }

   // Check the total number of entries
   if (fProgressStatus->GetEntries() >= fTotalEntries) {
      if (fProgressStatus->GetEntries() > fTotalEntries)
         Error("GetNextPacket", "Processed too many entries!");
      HandleTimer(0);   // Send last timer message
      SafeDelete(fProgress);
   }

   // Done
   return elem;
}

//______________________________________________________________________________
TVirtualPacketizer *TPacketizerMulti::CreatePacketizer(TDSet *dset, TList *wrks,
                                                       Long64_t first, Long64_t num,
                                                       TList *input, TProofProgressStatus *st)
{
   // Create a packetizer for dataset 'dset'
   // Return null on failure.

   TVirtualPacketizer *packetizer = 0;

   // Check inputs
   if (!dset || !wrks || !input || !st) {
      Error("CreatePacketizer", "invalid inputs: dset:%p wrks:%p input:%p st:%p",
                                 dset, wrks, input, st);
      return packetizer;
   }

   // This is for data-driven runs
   if (dset->TestBit(TDSet::kEmpty)) {
      Error("CreatePacketizer", "dataset is empty: protocol error?");
      return packetizer;
   }

   TString packetizername;
   TList *listOfMissingFiles = 0;

   TMethodCall callEnv;
   TClass *cl;

   // Lookup - resolve the end-point urls to optmize the distribution.
   // The lookup was previously called in the packetizer's constructor.
   // A list for the missing files may already have been added to the
   // output list; otherwise, if needed it will be created inside
   if (!(listOfMissingFiles = (TList *) input->FindObject("MissingFiles"))) {
      // Create it
      listOfMissingFiles = new TList;
      // and add it to the input list; it will be later moved to the output list
      input->Add(listOfMissingFiles);
   }
   dset->Lookup(kTRUE, &listOfMissingFiles);

   if (!(dset->GetListOfElements()) ||
       !(dset->GetListOfElements()->GetSize())) {
      Error("CreatePacketizer", "no files from the data set were found - skipping");
      return packetizer;
   }

   if (TProof::GetParameter(input, "PROOF_Packetizer", packetizername) != 0) {
      // Using standard packetizer TAdaptivePacketizer
      packetizername = "TPacketizerAdaptive";
   } else {
      Info("CreatePacketizer", "using alternate packetizer: %s", packetizername.Data());
   }

   // Get linked to the related class
   cl = TClass::GetClass(packetizername);
   if (cl == 0) {
      Error("CreatePacketizer", "class '%s' not found", packetizername.Data());
      return packetizer;
   }

   // Init the constructor
   callEnv.InitWithPrototype(cl, cl->GetName(),"TDSet*,TList*,Long64_t,Long64_t,TList*,TProofProgressStatus*");
   if (!callEnv.IsValid()) {
      Error("CreatePacketizer", "cannot find correct constructor for '%s'", cl->GetName());
      return packetizer;
   }
   callEnv.ResetParam();
   callEnv.SetParam((Long_t) dset);
   callEnv.SetParam((Long_t) wrks);
   callEnv.SetParam((Long64_t) first);
   callEnv.SetParam((Long64_t) num);
   callEnv.SetParam((Long_t) input);
   callEnv.SetParam((Long_t) st);

   // We are going to test validity during the packetizer initialization
   dset->SetBit(TDSet::kValidityChecked);
   dset->ResetBit(TDSet::kSomeInvalid);

   // Get an instance of the packetizer
   Long_t ret = 0;
   callEnv.Execute(ret);
   if ((packetizer = (TVirtualPacketizer *)ret) == 0) {
      Error("CreatePacketizer", "cannot construct '%s'", cl->GetName());
      return packetizer;
   }

   if (!packetizer->IsValid()) {
      Error("CreatePacketizer",
            "instantiated packetizer object '%s' is invalid", cl->GetName());
      SafeDelete(packetizer);
   }

   // Add invalid elements to the list of missing elements
   TDSetElement *elem = 0;
   if (dset->TestBit(TDSet::kSomeInvalid)) {
      TIter nxe(dset->GetListOfElements());
      while ((elem = (TDSetElement *)nxe())) {
         if (!elem->GetValid()) {
            listOfMissingFiles->Add(elem->GetFileInfo(dset->GetType()));
            dset->Remove(elem, kFALSE);
         }
      }
      // The invalid elements have been removed
      dset->ResetBit(TDSet::kSomeInvalid);
   }

   // Done
   return packetizer;
}
