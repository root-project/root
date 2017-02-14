#include "TFile.h"
#include "ROOT/TDataFrame.hxx"
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
      b = i / 100000.;
      t.Fill();
   }
   t.Write();
   f.Close();
}

int main(int argc, char** argv) {
   auto fileName = "test_reports.root";
   auto treeName = "reportsTree";
   FillTree(fileName, treeName);

   auto cut1 = [](double b) { return b < 0.001; };
   auto cut2 = [](double b) { return b > 0.05; };
   auto noopb = [](double b) { return true; };

   // multi-thread cutflow reports with default branch, multiple runs
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif
   TFile f(fileName);
   ROOT::Experimental::TDataFrame df("reportsTree", &f, {"b"});
   auto f1 = df.Filter(cut1, {}, "mtf");
   auto m1 = f1.Filter(noopb, {}, "mtnoop")
               .Mean<double>();
   f1.Report(); // warning, filters have not run
   *m1;
   df.Report(); // cutflow reports for "mtf", "mtnoop"
   auto f2 = df.AddBranch("foo", []() { return 42; })
               .Filter(cut2, {}, "mtf2");
   auto m2 = f2.Min<double>();
   *m2;
   df.Report(); // report all filters, only mtf2 prints non-zero values
   f2.Report(); // report only mtf2

   return 0;
}

void test_reports(int argc = 1, char** argv = nullptr) {
   main(argc, argv);
}
