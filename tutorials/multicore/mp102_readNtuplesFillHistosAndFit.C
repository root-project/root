/// \file
/// \ingroup multicore
/// Read n-tuples in distinct workers, fill histograms, merge them and fit.
/// We express parallelism with multiprocessing as it is done with multithreading
/// in mt102_readNtuplesFillHistosAndFit.
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

Int_t mp102_readNtuplesFillHistosAndFit()
{

   // No nuisance for batch execution
   gROOT->SetBatch();

   // Perform the operation sequentially ---------------------------------------
   TChain inputChain("multiCore");
   inputChain.Add("mp101_multiCore_*.root");
   TH1F outHisto("outHisto", "Random Numbers", 128, -4, 4);
   {
      TimerRAII t("Sequential read and fit");
      inputChain.Draw("r >> outHisto");
      outHisto.Fit("gaus");
   }

   // We now go MP! ------------------------------------------------------------
   // TProcPool offers an interface to directly process trees and chains without
   // the need for the user to go through the low level implementation of a
   // map-reduce.

   // We adapt our parallelisation to the number of input files
   const auto nFiles = inputChain.GetListOfFiles()->GetEntries();


   // This is the function invoked during the processing of the trees.
   auto workItem = [](TTreeReader & reader) {
      TTreeReaderValue<Float_t> randomRV(reader, "r");
      auto partialHisto = new TH1F("outHistoMP", "Random Numbers", 128, -4, 4);
      while (reader.Next()) {
         partialHisto->Fill(*randomRV);
      }
      return partialHisto;
   };

   // Create the pool of processes
   TProcPool workers(nFiles);

   // Process the TChain
   {
      TimerRAII t("Parallel execution");
      TH1F *sumHistogram = workers.ProcTree(inputChain, workItem, "multiCore");
      sumHistogram->Fit("gaus", 0);
   }

   return 0;

}
