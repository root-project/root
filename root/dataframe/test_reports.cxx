#include "TFile.h"
#include "ROOT/RDataFrame.hxx"
#include "TROOT.h"

#include <cmath> // sqrt
#include <iostream>
#include <numeric> // accumulate

void FillTree(const char* filename, const char* treeName) {
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   double b;
   t.Branch("b", &b);
   for(int i = 0; i < 100000; ++i) {
      b = static_cast<double>(i) * 1.0e-5;
      t.Fill();
   }
   t.Write();
   f.Close();
}

int main() {
   auto fileName = "test_reports.root";
   auto treeName = "reportsTree";
   FillTree(fileName, treeName);

   auto cut1 = [](double b) { return b < 0.001; };
   auto cut2 = [](double b) { return b > 0.05; };
   auto noopb = [](double) { return true; };

   // multi-thread cutflow reports with default branch, multiple runs
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif
   ROOT::RDataFrame df("reportsTree", fileName, {"b"});
   auto f1 = df.Filter(cut1, {}, "mtf");

   auto ac1 = df.Define("foo", []() { return 42; });
   auto f2 = ac1.Filter(noopb);
   auto f3 = f2.Filter(cut2, "mtf2");

   // Report on the original dataframe
   // "mtf", "mtf2" will be listed, in this order
   df.Report()->Print();
   std::cout << "--\n";
   // Report on a named filter
   // only "mtf" listed
   f1.Report()->Print();
   std::cout << "--\n";
   // Report on nodes with no upstream named filters (new column, unnamed filter)
   // no output
   ac1.Report()->Print();
   f2.Report()->Print();
   std::cout << "--\n";
   // Report on a named filter with upstream unnamed filters
   // only "mtf2" listed
   f3.Report()->Print();

   // Consecutive reports on the same RDataFrame
   ROOT::RDataFrame df2(10);
   auto dwx = df2.DefineSlotEntry("x", [](unsigned int, ULong64_t e) { return static_cast<int>(e); });
   auto dwF = dwx.Filter([](int x) { return x > 3; }, {"x"}, "fx1").Filter([](int x) { return x > 5; }, {"x"}, "fx2");
   std::cout << "--\n";
   dwF.Report()->Print();
   auto dwFF = dwF.Filter([](int x) { return x > 8; }, {"x"}, "fx3");
   std::cout << "--\n";
   dwFF.Report()->Print();

   return 0;
}

void test_reports() {
   main();
}
