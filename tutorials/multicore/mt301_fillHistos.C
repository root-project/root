/// \file
/// \ingroup tutorial_multicore
/// Fill histograms w/o in parallel with automatic binning.
/// Illustrates range synchronization
///
/// \macro_code
///
/// \author Gerardo Ganis
/// \date August 2017
#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"
#include "TList.h"
#include "TRandom3.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "TCanvas.h"
#include "TString.h"
#include "TStyle.h"
#include "ROOT/TSeq.hxx"
#include "ROOT/TThreadedObject.hxx"

#include <thread>
#include <iostream>


// The number of workers
const UInt_t nWorkers = 8U;

Int_t mt301_fillHistos(UInt_t nNumbers = 1001, Bool_t ranges = kFALSE)
{

   // The first, fundamental operation to be performed in order to make ROOT
   // thread-aware.
   ROOT::EnableThreadSafety();

   Double_t mi = 0., ma = -1.;
   if (ranges) { mi = -4.; ma = 4.; }

   // Histograms to be filled in parallel
   ROOT::TThreadedObject<TH1D> h1d("h1d", "1D test histogram", 64, mi, ma);

   // We define our work item
   auto workItem = [&](UInt_t workerID) {
      // One generator, file and ntuple per worker
      TRandom3 workerRndm(workerID); // Change the seed

      auto wh1d = h1d.Get();
      wh1d->SetBit(TH1::kAutoBinPTwo);

      Double_t x, y;

      for (UInt_t i = 0; i < nNumbers; ++i) {
         wh1d->Fill(workerRndm.Gaus(3.));
      }
   };

   // Create the collection which will hold the threads, our "pool"
   std::vector<std::thread> workers;

   // Fill the "pool" with workers
   for (auto workerID : ROOT::TSeqI(nWorkers)) {
      workers.emplace_back(workItem, workerID);
   }

   // Now join them
   for (auto && worker : workers) worker.join();

   // Merge
   auto fh1d = h1d.Merge();

   // Make the canvas
   TCanvas *c = new TCanvas("c", "c", 800, 800);
   c->Divide(1,2);

   c->cd(1);
   fh1d->DrawClone();
   c->cd(2);
   gStyle->SetOptStat(111110);
   fh1d->Adjust(-1., 3);
   fh1d->SetTitle(TString::Format("%s - adjusted", fh1d->GetTitle()));
   fh1d->DrawClone();

   gROOTMutex = 0;

   return 0;
}
