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
   auto f = std::make_unique<TFile>(filename, "RECREATE");
   auto t = std::make_unique<TTree>(treeName, treeName);
   Sentinel o;
   t->Branch("obj", &o);
   o.set(1);
   t->Fill();
   o.set(2);
   t->Fill();
   f->Write();
}

int test_ctors()
{
   // Run the test in a scope and inspect results afterwards
   std::uint64_t entries{};
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
      entries = *r;
   }

   auto ctor_calls = Sentinel::GetCtorCalls();
   auto copy_ctor_calls = Sentinel::GetCopyCtorCalls();
   auto dtor_calls = Sentinel::GetDtorCalls();

   // Exactly 2 Sentinel objects should have been constructed/destroyed: one in the FillTree function and one by the
   // TTreeReaderValue associated with the column reader for "obj", since the readers keeps the same TTreeReaderValue
   // alive for the duration of the computation graph run, the same Sentinel object can be re-used.
   if (entries == 2 && ctor_calls == 2 && dtor_calls == 2 && copy_ctor_calls == 0)
      return 0;

   return 1;
}
