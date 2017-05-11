#include "ROOT/TTreeProcessorMT.hxx" // for TTreeView
#include "ROOT/TThreadedObject.hxx"
#include "ROOT/TThreadExecutor.hxx"
#include "TFile.h"
#include "TTree.h"
#include <iostream>
#include <string>

using namespace std;

void FillTree(const char *filename, const char *treeName)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   double b1;
   t.Branch("b1", &b1);
   for (int i = 0; i < 10; ++i) {
      b1 = i;
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}

int main()
{
   auto treeName = "t";
   std::vector<std::string_view> fnames = {"tthrobj_ttreeview_regression_1.root",
                                           "tthrobj_ttreeview_regression_2.root"};
   for (auto &fname : fnames) FillTree(fname.data(), treeName);
   std::vector<size_t> args = {0, 1};

   ROOT::EnableImplicitMT(2);
   //ROOT::EnableThreadSafety();
   ROOT::TThreadExecutor pool;

   {
      ROOT::TThreadedObject<ROOT::Internal::TTreeView> treeView(fnames, treeName);
      auto lambda = [&](size_t s) {
         treeView->SetCurrent(s);
      };
      pool.Foreach(lambda, args);
      std::cout << "finished 0" << std::endl;
   }

   ROOT::TThreadedObject<ROOT::Internal::TTreeView> treeView(fnames, treeName);
   auto lambda = [&](size_t s) {
      treeView->SetCurrent(s);
   };
   pool.Foreach(lambda, args);
   std::cout << "finished 1" << std::endl;

   return 0;
}

int repro_stripped() {return main();}
