// Test that Snapshot yields correct results when each worker thread processes multiple tasks
#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"


int main() {
   const auto nSlots = 4u;
   ROOT::EnableImplicitMT(nSlots);

   // easiest way to be sure reading requires spawning of several tasks: create several input files
   const std::string inputFilePrefix = "snapshot_manytasks_";
   const auto nInputFiles = nSlots * 8;
   ROOT::RDataFrame d(1);
   auto dd = d.Define("x", []() { return 42; });
   for (auto i = 0u; i < nInputFiles; ++i)
      dd.Snapshot("t", inputFilePrefix + std::to_string(i) + ".root", {"x"});

   // test multi-thread Snapshotting from many tasks per worker thread
   const auto outputFile = "out_snapshot_manytasks.root";
   ROOT::RDataFrame tdf("t", (inputFilePrefix + "*.root").c_str());
   tdf.Snapshot("t", outputFile, {"x"});

   // check output contents
   ROOT::RDataFrame checkTdf("t", outputFile);
   auto c = checkTdf.Count();
   auto t = checkTdf.Take<int>("x");
   int retCode = 0;
   for (auto v : t) {
      if (v != 42) {
         retCode += 1;
         break;
      }
   }

   if (*c != nInputFiles)
      retCode += 2;

   // clean-up input files
   for (auto i = 0u; i < nInputFiles; ++i)
      gSystem->Unlink((inputFilePrefix + std::to_string(i) + ".root").c_str());
   gSystem->Unlink(outputFile);
   return retCode;
}
