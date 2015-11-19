/// \file
/// \ingroup multicore
/// Fill n-tuples in distinct workers.
/// This tutorial illustrates the basics of how it's possible with ROOT to
/// offload heavy operations on multiple processes and how it's possible to write
/// simultaneously multiple files. The operation performed in this case is the
/// creation of random gaussian numbers.
///
/// \macro_code
///
/// \author Danilo Piparo

// Measure time in a scope
class TimerRAII {
   TStopwatch fTimer;
   std::string fMeta;
public:
   TimerRAII(const char *meta): fMeta(meta) {
      fTimer.Start();
   }
   ~TimerRAII() {
      fTimer.Stop();
      std::cout << fMeta << " - real time elapsed " << fTimer.RealTime() << "s" << std::endl;
   }
};

Int_t mp101_fillNtuples(UInt_t nWorkers = 4)
{

   // No nuisance for batch execution
   gROOT->SetBatch();

   // Total amount of numbers
   const UInt_t nNumbers = 20000000U;

   // A simple function to fill ntuples randomly

   auto fillRandom = [](TNtuple & ntuple, TRandom3 & rndm, UInt_t n) {
      for (UInt_t i = 0; i < n; ++i) ntuple.Fill(rndm.Gaus());
   };

   // Perform the operation sequentially ---------------------------------------

   // Create a random generator and and Ntuple to hold the numbers
   TRandom3 rndm(1);
   TFile ofile("mp101_singleCore.root", "RECREATE");
   TNtuple randomNumbers("singleCore", "Random Numbers", "r");

   // Now let's measure how much time we need to fill it up
   {
      TimerRAII t("Sequential execution");
      fillRandom(randomNumbers, rndm, nNumbers);
      randomNumbers.Write();
   }


   // We now go MP! ------------------------------------------------------------

   // We define our work item
   auto workItem = [&fillRandom](UInt_t workerID, UInt_t workSize) {
      // One generator, file and ntuple per worker
      TRandom3 workerRndm(workerID); // Change the seed
      TFile ofile(Form("mp101_multiCore_%u.root", workerID), "RECREATE");
      TNtuple workerRandomNumbers("multiCore", "Random Numbers", "r");
      fillRandom(workerRandomNumbers, workerRndm, workSize);
      workerRandomNumbers.Write();
      return 0;
   };

   // Create the pool of workers
   TProcPool workers(nWorkers);

   // We measure time here as well
   {
      TimerRAII t("Parallel execution");

      // We split the work in equal parts
      const auto workSize = nNumbers / nWorkers;

      // The work item requires two arguments, the map infrastructure offer
      // an interface to use only one. A standard solution is to use std::bind
      using namespace std::placeholders;
      auto workItemOneArg = std::bind(workItem, _1, workSize);

      // Fill the pool with work
      std::forward_list<UInt_t> workerIDs(nWorkers);
      std::iota(std::begin(workerIDs), std::end(workerIDs), 0);
      workers.Map(workItemOneArg, workerIDs);
   }

   return 0;

}
