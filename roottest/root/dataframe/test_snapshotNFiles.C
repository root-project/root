#include "TTree.h"
#include "ROOT/RDataFrame.hxx"

#include <iostream>

void fill_file(const char *filename, const int n) {
   TFile f(filename, "recreate");
   TTree t("t","t");
   int b;
   t.Branch("b", &b);
   for(b = 0; b < n; ++b)
      t.Fill();
   t.Write();
}

int test_snapshotNFiles() {
   fill_file("file_snapshot2Files_1.root",1);
   fill_file("file_snapshot2Files_2.root",10);
   // Single threaded
   {
      ROOT::RDataFrame d("t", "file_snapshot2Files_*[1,2].root");
      auto d2 = d.Snapshot("t", "outfile.root", {"b"});
      d2->Foreach([](int b){std::cout << "b = " << b << std::endl; }, {"b"});

   }
   // Multithreaded
#ifdef R__USE_IMT
   {
      ROOT::EnableImplicitMT(3);
      std::cout << "Now going MT\n";
      fill_file("file_snapshot2Files_3.root",5);
      fill_file("file_snapshot2Files_4.root",5);
      fill_file("file_snapshot2Files_5.root",5);
      fill_file("file_snapshot2Files_6.root",5);
      fill_file("file_snapshot2Files_7.root",5);
      fill_file("file_snapshot2Files_8.root",5);
      std::cout << "Additional files written\n";
      ROOT::RDataFrame d("t", "file_snapshot2Files_*.root");
      auto d2 = d.Snapshot("t", "outfile.root", {"b"});
      d2->Foreach([](int b){std::cout << "b = " << b << std::endl; }, {"b"});
   }
   std::cout << "Temporary branch\n";
   ROOT::RDataFrame d("t", "file_snapshot2Files_*.root");
   auto d2 = d.Define("x", [] { return 3; });
   auto d3 = d2.Snapshot("t", "outfile.root", {"x", "b"});
   d3->Foreach([](int b){std::cout << "b = " << b << std::endl; }, {"b"});
#endif
   return 0;
}

int main() {
   return test_snapshotNFiles();
}
