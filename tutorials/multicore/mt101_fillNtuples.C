/// \file
/// \ingroup multicore
/// Fill n-tuples in distinct workers.
/// This tutorial illustrates the basics of how it's possible with ROOT to
/// offload heavy operations on multiple threads and how it's possible to write
/// simultaneously multiple files. The operation performed in this case is the
/// creation of random gaussian numbers.
/// NOTE: this code can be executed in a macro, ACLiC'ed or not, but not yet at
/// the command line prompt.
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

Int_t mt101_fillNtuples(UInt_t nWorkers = 4)
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
   TFile ofile("mt101_singleCore.root", "RECREATE");
   TNtuple randomNumbers("singleCore", "Random Numbers", "r");

   // Now let's measure how much time we need to fill it up
   {
      TimerRAII t("Sequential execution");
      fillRandom(randomNumbers, rndm, nNumbers);
      randomNumbers.Write();
   }


   // We now go MT! ------------------------------------------------------------

   // The first, fundamental operation to be performed in order to make ROOT
   // thread-aware.
   ROOT::EnableThreadSafety();

   // We define our work item
   auto workItem = [&fillRandom](UInt_t workerID, UInt_t workSize) {
      // One generator, file and ntuple per worker
      TRandom3 workerRndm(workerID); // Change the seed
      TFile ofile(Form("mt101_multiCore_%u.root", workerID), "RECREATE");
      TNtuple workerRandomNumbers("multiCore", "Random Numbers", "r");
      fillRandom(workerRandomNumbers, workerRndm, workSize);
      workerRandomNumbers.Write();
   };

   // Create the collection which will hold the threads, our "pool"
   std::vector<std::thread> workers;

   // We measure time here as well
   {
      TimerRAII t("Parallel execution");

      // We split the work in equal parts
      const auto workSize = nNumbers / nWorkers;

      // Fill the "pool" with workers
      for (UInt_t workerID = 0; workerID < nWorkers; ++workerID) {
         workers.emplace_back(workItem, workerID, workSize);
      }

      // Now join them
      for (auto && worker : workers) worker.join();
   }

   return 0;

}
