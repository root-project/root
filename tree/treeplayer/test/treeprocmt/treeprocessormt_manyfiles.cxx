#include <atomic>
#include <chrono>
#include <random>
#include <string>
#include <thread>

#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <ROOT/TTreeProcessorMT.hxx>

#include "gtest/gtest.h"

void WriteFiles(const std::string &treename, const std::vector<std::string> &filenames)
{
   int v = 0;
   for (const auto &f : filenames) {
      TFile file(f.c_str(), "recreate");
      TTree t(treename.c_str(), treename.c_str());
      t.Branch("v", &v);
      for (auto i = 0; i < 10; ++i) {
         ++v;
         t.Fill();
      }
      t.Write();
   }
}

void DeleteFiles(const std::vector<std::string> &filenames)
{
   for (const auto &f : filenames)
      gSystem->Unlink(f.c_str());
}

TEST(TreeProcessorMT, ManyFiles)
{
   const auto nFiles = 100u;
   const std::string treename = "t";
   std::vector<std::string> filenames;
   for (auto i = 0u; i < nFiles; ++i)
      filenames.emplace_back("treeprocmt_" + std::to_string(i) + ".root");

   WriteFiles(treename, filenames);

   std::atomic_int sum(0);
   std::atomic_int count(0);
   auto sumValues = [&sum, &count](TTreeReader &r) {
      TTreeReaderValue<int> v(r, "v");
      std::random_device seed;
      std::default_random_engine eng(seed());
      std::uniform_int_distribution<> rand(1, 100);
      while (r.Next()) {
         std::this_thread::sleep_for(std::chrono::nanoseconds(rand(eng)));
         sum += *v;
         ++count;
      }
   };

   // TTreeProcMT requires a vector<string_view>
   std::vector<std::string_view> fnames;
   for (const auto &f : filenames)
      fnames.emplace_back(f);

   ROOT::TTreeProcessorMT proc(fnames, treename);
   proc.Process(sumValues);
   
   EXPECT_EQ(count.load(), int(nFiles*10)); // 10 entries per file
   EXPECT_EQ(sum.load(), 500500); // sum 1..nFiles*10

   DeleteFiles(filenames);
}
