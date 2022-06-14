/// \file
/// \ingroup tutorial_multicore
/// \notebook
/// Read n-tuples in distinct workers, fill histograms, merge them and fit.
/// Knowing that other facilities like TProcessExecutor might be more adequate for
/// this operation, this tutorial complements mc101, reading and merging.
/// We convey another message with this tutorial: the synergy of ROOT and
/// STL algorithms is possible.
///
/// \macro_output
/// \macro_code
///
/// \date January 2016
/// \author Danilo Piparo

Int_t mt102_readNtuplesFillHistosAndFit()
{

   // No nuisance for batch execution
   gROOT->SetBatch();

   // Perform the operation sequentially ---------------------------------------
   TChain inputChain("multiCore");
   inputChain.Add("mt101_multiCore_*.root");
   TH1F outHisto("outHisto", "Random Numbers", 128, -4, 4);
   inputChain.Draw("r >> outHisto");
   outHisto.Fit("gaus");

   // We now go MT! ------------------------------------------------------------

   // The first, fundamental operation to be performed in order to make ROOT
   // thread-aware.
   ROOT::EnableThreadSafety();

   // We adapt our parallelisation to the number of input files
   const auto nFiles = inputChain.GetListOfFiles()->GetEntries();

   // We define the histograms we'll fill
   std::vector<TH1F> histograms;
   auto workerIDs = ROOT::TSeqI(nFiles);
   histograms.reserve(nFiles);
   for (auto workerID : workerIDs) {
      histograms.emplace_back(TH1F(Form("outHisto_%u", workerID), "Random Numbers", 128, -4, 4));
   }

   // We define our work item
   auto workItem = [&histograms](UInt_t workerID) {
      TFile f(Form("mt101_multiCore_%u.root", workerID));
      auto ntuple = f.Get<TNtuple>("multiCore");
      auto &histo = histograms.at(workerID);
      for (auto index : ROOT::TSeqL(ntuple->GetEntriesFast())) {
         ntuple->GetEntry(index);
         histo.Fill(ntuple->GetArgs()[0]);
      }
   };

   TH1F sumHistogram("SumHisto", "Random Numbers", 128, -4, 4);

   // Create the collection which will hold the threads, our "pool"
   std::vector<std::thread> workers;

   // Spawn workers
   // Fill the "pool" with workers
   for (auto workerID : workerIDs) {
      workers.emplace_back(workItem, workerID);
   }

   // Now join them
   for (auto &&worker : workers)
      worker.join();

   // And reduce with a simple lambda
   std::for_each(std::begin(histograms), std::end(histograms),
                 [&sumHistogram](const TH1F &h) { sumHistogram.Add(&h); });

   sumHistogram.Fit("gaus", 0);

   return 0;
}
