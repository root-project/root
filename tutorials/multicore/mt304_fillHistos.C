/// \file
/// \ingroup tutorial_multicore
/// \notebook -draw
/// Fill histograms in parallel with automatic binning.
/// Illustrates use of power-of-two autobin algorithm
///
/// \macro_code
/// \macro_image
///
/// \date November 2017
/// \author Gerardo Ganis

// The number of workers
const UInt_t nWorkers = 8U;

// Reference boundaries
const Double_t xmiref = -1.;
const Double_t xmaref = 7.;

Int_t mt304_fillHistos(UInt_t nNumbers = 1001)
{

   // The first, fundamental operation to be performed in order to make ROOT
   // thread-aware.
   ROOT::EnableThreadSafety();

   // Histograms to be filled in parallel
   ROOT::TThreadedObject<TH1D> h1d("h1d", "1D test histogram", 64, 0., -1.);
   ROOT::TThreadedObject<TH1D> h1dr("h1dr", "1D test histogram w/ ref boundaries", 64, xmiref, xmaref);

   // We define our work item
   auto workItem = [&](UInt_t workerID) {
      // One generator, file and ntuple per worker
      TRandom3 workerRndm(workerID); // Change the seed

      auto wh1d = h1d.Get();
      wh1d->SetBit(TH1::kAutoBinPTwo);
      auto wh1dr = h1dr.Get();

      Double_t x;
      for (UInt_t i = 0; i < nNumbers; ++i) {
         x = workerRndm.Gaus(3.);
         wh1d->Fill(x);
         wh1dr->Fill(x);
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
   auto fh1dr = h1dr.Merge();

   // Make the canvas
   auto c = new TCanvas("c", "c", 800, 800);
   c->Divide(1, 2);

   gStyle->SetOptStat(111110);
   c->cd(1);
   fh1d->DrawCopy();
   c->cd(2);
   fh1dr->DrawCopy();

   c->Update();

   return 0;
}
