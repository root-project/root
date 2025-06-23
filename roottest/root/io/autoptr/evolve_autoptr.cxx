#include <TFile.h>
#include <TTree.h>

#include <memory>

#include "TestAutoPtr_v3.hxx"

int main()
{
   auto f = std::unique_ptr<TFile>(TFile::Open("root_test_autoptr.root"));
   auto t = (TTree *)f->Get("t");

   int i = 1;
   for (const auto branchname : {"e_nodot", "e.", "e_split0_nodot", "e_split0."}) {
      TestAutoPtr *entry = nullptr;
      t->SetBranchAddress(branchname, &entry);

      t->GetEntry(0);
      if (!entry->fTrack || entry->fTrack->fFoo != 1)
         return i * 1;

      t->GetEntry(1);
      if (entry->fTrack)
         return i * 2;

      t->GetEntry(2);
      if (!entry->fTrack || entry->fTrack->fFoo != 3)
         return i * 3;

      i += 10;
   }

   return 0;
}
