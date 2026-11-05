#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "ROOT/RDataFrame.hxx"
#include "Sentinel.h"

unsigned int Sentinel::fCtorCalls{};
unsigned int Sentinel::fDtorCalls{};
unsigned int Sentinel::fCopyCtorCalls{};

// test objects read from tree are constructed once per entry
void FillTree(const char *filename, const char *treeName)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   Sentinel o;
   t.Branch("obj", &o);
   o.set(1);
   t.Fill();
   o.set(2);
   t.Fill();
   t.Write();
   f.Close();
}

int test_ctors()
{
   // Prepare an input tree to run on
   auto fileName = "test_ctors.root";
   auto treeName = "myTree";
   FillTree(fileName, treeName);

   ROOT::RDataFrame d(treeName, fileName);
   auto dd = d.Filter([](const Sentinel &o) { return o.get() > 0; }, {"obj"})
                .Define("ox", [](const Sentinel &o) { return o.get(); }, {"obj"});
   auto r = dd.Count();
   dd.Foreach([](const Sentinel &) {}, {"obj"});
   auto c = *r;

   auto ctor_calls = Sentinel::GetCtorCalls();
   auto copy_ctor_calls = Sentinel::GetCopyCtorCalls();
   auto dtor_calls = Sentinel::GetDtorCalls();

   // Exactly 3 Sentinel objects should have been constructed/destroyed: one in the FillTree function and one per each
   // of the two entries read by RDataFrame.
   if (c == 2 && ctor_calls == 3 && dtor_calls == 3 && copy_ctor_calls == 0)
      return 0;

   return 1;
}
