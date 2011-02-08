// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofBenchDataSet                                                   //
//                                                                      //
// Handle operations on datasets used by ProofBench                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofBenchDataSet.h"
#include "TClass.h"
#include "TFileCollection.h"
#include "TList.h"
#include "TMap.h"
#include "TProof.h"
#include "TProofBenchTypes.h"


ClassImp(TProofBenchDataSet)

//______________________________________________________________________________
TProofBenchDataSet::TProofBenchDataSet(TProof *proof)
{
   // Constructor
   
   fProof = proof ? proof : gProof;
}


//______________________________________________________________________________
Int_t TProofBenchDataSet::ReleaseCache(const char *dset)
{
   // Release memory cache for dataset 'dset'
   // Return 0 on success, -1 on error

   // Clear the cache
   TPBHandleDSType type(TPBHandleDSType::kReleaseCache);
   if (Handle(dset, &type) != 0) {
      Error("ReleaseCache", "problems clearing cache for '%s'", dset);
      return -1;
   }
   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TProofBenchDataSet::RemoveFiles(const char *dset)
{
   // Physically remove the dataset 'dset', i.e. remove the dataset and the files
   // it describes
   // Return 0 on success, -1 on error

   // Phyically remove the files
   TPBHandleDSType type(TPBHandleDSType::kRemoveFiles);
   if (Handle(dset, &type) != 0) {
      Error("RemoveFiles", "problems removing files for '%s'", dset);
      return -1;
   }
   // Remove the meta information
   if (!fProof || (fProof && fProof->RemoveDataSet(dset) != 0)) {
      Error("RemoveFiles", "problems removing meta-information for dataset '%s'", dset);
      return -1;
   }
   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TProofBenchDataSet::CopyFiles(const char *dset, const char *destdir)
{
   // Copy the files of dataset 'dset' to another directory
   // Return 0 on success, -1 on error

   // Check input
   if (!destdir || (destdir && strlen(destdir) <= 0)) {
      Error("CopyFiles", "specifying a destination dir is mandatory!");
      return -1;
   }      

   // Set the destination dir
   if (fProof) fProof->SetParameter("PROOF_Benchmark_DestDir", destdir);

   // Copy the files
   TPBHandleDSType type(TPBHandleDSType::kCopyFiles);
   if (Handle(dset, &type) != 0) {
      Error("CopyFiles", "problems copying files for '%s'", dset);
      return -1;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TProofBenchDataSet::Handle(const char *dset, TObject *type)
{
   // Physically remove the dataset 'dset', i.e. remove the dataset and the files
   // it describes
   // Return 0 on success, -1 on error

   // Check input
   if (!dset || (dset && strlen(dset) <= 0)) {
      Error("Handle", "a valid dataset name is mandatory");
      return -1;
   }

   // The dataset must exist
   if (!fProof || (fProof && !fProof->ExistsDataSet(dset))) {
      Error("Handle", "dataset '%s' does not exist", dset);
      return -1;
   }

   // Get the dataset
   TFileCollection *fc = fProof->GetDataSet(dset);
   if (!fc) {
      Error("Handle", "TFileCollection object for dataset '%s' could not be retrieved", dset);
      return -1;
   }

   // Get information per server
   TMap *fcmap = fc->GetFilesPerServer();
   if (!fcmap) {
      Error("Handle", "could not create map with per-server info for dataset '%s'", dset);
      return -1;
   }
   fcmap->Print();
   
   // Load the selector, if needed
   TString selName("TSelHandleDataSet");
   if (!TClass::GetClass(selName)) {
      // Load the parfile
      TString par = TString::Format("%s%s.par", kPROOF_BenchSrcDir, kPROOF_BenchDataSelPar);
      Info("Handle", "Uploading '%s' ...", par.Data());
      if (fProof->UploadPackage(par) != 0) {
         Error("Handle", "problems uploading '%s' - cannot continue", par.Data());
         return -1;
      }
      Info("Handle", "Enabling '%s' ...", kPROOF_BenchDataSelPar);
      if (fProof->EnablePackage(kPROOF_BenchDataSelPar) != 0) {
         Error("Handle", "problems enabling '%s' - cannot continue", kPROOF_BenchDataSelPar);
         return -1;
      }
      // Check
      if (!TClass::GetClass(selName)) {
         Error("Handle", "failed to load '%s'", selName.Data());
         return -1;
      }
   }

   // Add map in the input list
   fcmap->SetName("PROOF_FilesToProcess");
   fProof->AddInput(fcmap);

   // Set parameters for processing
   TString oldpack;
   if (TProof::GetParameter(fProof->GetInputList(), "PROOF_Packetizer", oldpack) != 0) oldpack = "";
   fProof->SetParameter("PROOF_Packetizer", "TPacketizerFile");

   // Process
   fProof->AddInput(type);
   fProof->Process(selName, (Long64_t) fc->GetNFiles());
   if (fProof->GetInputList()) fProof->GetInputList()->Remove(type);

   // Restore parameters
   if (!oldpack.IsNull())
      fProof->SetParameter("PROOF_Packetizer", oldpack);
   else
      fProof->DeleteParameters("PROOF_Packetizer");

   // Cleanup
   fProof->GetInputList()->Remove(fcmap);
   delete fcmap;
   delete fc;
   
   // Done
   return 0;
}
