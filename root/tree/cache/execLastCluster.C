#include <TFile.h>
#include <TTree.h>
#include <TError.h>
#include <iostream>

using std::cout;
using std::endl;

int execLastCluster() {
   TFile *file = TFile::Open("lastcluster.root");
   if (!file || file->IsZombie())
      return 1;
   TTree *t; file->GetObject("t", t);
   if (!t) {
      Error("execLastCluster","Can't find the TTree named 't' in 'lastcluster.root'");
      return 2;
   }
   t->SetAutoFlush(-2000000);
   t->GetEntry(0);
   t->SetCacheSize(100);
   cout << "Estimate cluster size: " << t->GetEntries()*t->GetCacheSize()/ t->GetZipBytes() << '\n';
//   t->GetEntry(951)
//   t->GetEntry(990)
   t->Print("clusters");
   auto iter = t->GetClusterIterator(985);
   cout << "Start entry: " << iter.GetStartEntry() << '\n';
   cout << "Next entry 1: " << iter.GetNextEntry() << '\n';
   cout << "Next entry 2: " << iter.Next() << '\n';
   cout << "Next entry 2: " << iter.Next() << '\n';
   cout << "Cache size:" << t->GetCacheSize() << '\n';
   t->GetEntry(951);
   t->GetEntry(990);
   t->Draw("values");

   return 0;
}
