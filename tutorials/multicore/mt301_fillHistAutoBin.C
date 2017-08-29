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
#include <thread>
#include <iostream>
#include "ROOT/TSeq.hxx"
#include "ROOT/TThreadedObject.hxx"

// The number of workers
const UInt_t nWorkers = 8U;

Int_t mt301_fillHistos(UInt_t nNumbers = 1001, Bool_t ranges = kFALSE)
{

   // The first, fundamental operation to be performed in order to make ROOT
   // thread-aware.
   ROOT::EnableThreadSafety();

   Double_t mi = 0., ma = -1.;
   if (ranges) {
      mi = -4.;
      ma = 4.;
   }

   // Histograms to be filled in parallel
   ROOT::TThreadedObject<TH1D> h1d("h1d", "1D test histogram", 64, mi, ma);
   ROOT::TThreadedObject<TH2D> h2d("h2d", "2D test histogram", 64, mi, ma, 64, mi, ma);
   ROOT::TThreadedObject<TH3D> h3d("h3d", "3D test histogram", 64, mi, ma, 64, mi, ma, 64, mi, ma);

   // We define our work item
   auto workItem = [&](UInt_t workerID) {
      // One generator, file and ntuple per worker
      TRandom3 workerRndm(workerID); // Change the seed

      auto wh1d = h1d.Get();
      auto wh2d = h2d.Get();
      auto wh3d = h3d.Get();

      Double_t x, y;

      for (UInt_t i = 0; i < nNumbers; ++i) {
         wh1d->Fill(workerRndm.Gaus());
         workerRndm.Rannor(x, y);
         wh2d->Fill(x, y);
         workerRndm.Rannor(x, y);
         wh3d->Fill(x, y, workerRndm.Gaus());
      }
   };

   // Create the collection which will hold the threads, our "pool"
   std::vector<std::thread> workers;

   // Fill the "pool" with workers
   for (auto workerID : ROOT::TSeqI(nWorkers)) {
      workers.emplace_back(workItem, workerID);
   }

   // Now join them
   for (auto &&worker : workers)
      worker.join();

   // Merge
   auto fh1d = h1d.Merge();
   auto fh2d = h2d.Merge();
   auto fh3d = h3d.Merge();

   // Make the canvas
   TCanvas *c = new TCanvas("c", "c", 800, 800);
   c->Divide(2, 2);

   c->cd(1);
   fh1d->DrawClone();
   c->cd(2);
   fh2d->DrawClone();
   c->cd(3);
   fh3d->DrawClone();

   gROOTMutex = 0;

   return 0;
}
