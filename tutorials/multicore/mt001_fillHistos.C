/// \file
/// \ingroup multicore
/// Fill histograms in parallel and write them on file.
/// The simplest meaningful possible example which shows ROOT thread awareness.
///
/// \macro_code
///
/// \author Danilo Piparo

Int_t mt001_fillHistos(UInt_t nWorkers = 4)
{

   // Total amount of numbers
   const UInt_t nNumbers = 20000000U;

   // The first, fundamental operation to be performed in order to make ROOT
   // thread-aware.
   ROOT::EnableThreadSafety();

   // We define our work item
   auto workItem = [nNumbers](UInt_t workerID) {
      // One generator, file and ntuple per worker
      TRandom3 workerRndm(workerID); // Change the seed
      TFile f(Form("myFile_%u.root", workerID), "RECREATE");
      TH1F h(Form("myHisto_%u", workerID), "The Histogram", 64, -4, 4);
      for (UInt_t i = 0; i < nNumbers; ++i) {
         h.Fill(workerRndm.Gaus());
      }
      h.Write();
   };

   // Create the collection which will hold the threads, our "pool"
   std::vector<std::thread> workers;

   // Fill the "pool" with workers
   for (UInt_t workerID = 0; workerID < nWorkers; ++workerID) {
      workers.emplace_back(workItem, workerID);
   }

   // Now join them
   for (auto && worker : workers) worker.join();

   return 0;

}
