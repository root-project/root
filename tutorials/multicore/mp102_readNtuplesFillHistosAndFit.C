/// \file
/// \ingroup tutorial_multicore
/// \notebook -js
/// Read n-tuples in distinct workers, fill histograms, merge them and fit.
/// We express parallelism with multiprocessing as it is done with multithreading
/// in mt102_readNtuplesFillHistosAndFit.
///
/// \macro_code
///
/// \date January 2016
/// \author Danilo Piparo

Int_t mp102_readNtuplesFillHistosAndFit()
{

   // No nuisance for batch execution
   gROOT->SetBatch();

   //---------------------------------------
   // Perform the operation sequentially
   TChain inputChain("multiCore");
   inputChain.Add("mp101_multiCore_*.root");
   if (inputChain.GetNtrees() <= 0) {
      Printf(" No files in the TChain: did you run mp101_fillNtuples.C before?");
      return 1;
   }
   TH1F outHisto("outHisto", "Random Numbers", 128, -4, 4);
   inputChain.Draw("r >> outHisto");
   outHisto.Fit("gaus");

   //---------------------------------------
   // We now go MP!
   // TProcessExecutor offers an interface to directly process trees and chains without
   // the need for the user to go through the low level implementation of a
   // map-reduce.

   // We adapt our parallelisation to the number of input files
   const auto nFiles = inputChain.GetListOfFiles()->GetEntries();

   // This is the function invoked during the processing of the trees.
   auto workItem = [](TTreeReader &reader) {
      TTreeReaderValue<Float_t> randomRV(reader, "r");
      auto partialHisto = new TH1F("outHistoMP", "Random Numbers", 128, -4, 4);
      while (reader.Next()) {
         partialHisto->Fill(*randomRV);
      }
      return partialHisto;
   };

   // Create the pool of processes
   ROOT::TTreeProcessorMP workers(nFiles);

   // Process the TChain
   auto sumHistogram = workers.Process(inputChain, workItem, "multiCore");
   sumHistogram->Fit("gaus", 0);

   return 0;
}
