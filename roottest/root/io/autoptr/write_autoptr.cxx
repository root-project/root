#include <TFile.h>
#include <TTree.h>

#include <memory>

#include "TestAutoPtr_v2.hxx"

int main()
{
   auto f = std::unique_ptr<TFile>(TFile::Open("root_test_autoptr.root", "RECREATE"));
   auto t = new TTree("t", "");

   TestAutoPtr entry;
   t->Branch("e_nodot", &entry);
   t->Branch("e.", &entry);
   t->Branch("e_split0_nodot", &entry, 32000, 0);
   t->Branch("e_split0.", &entry, 32000, 0);

   entry.fTrack = new Track{1};
   t->Fill();
   delete entry.fTrack.fRawPtr;

   entry.fTrack = nullptr;
   t->Fill();
   delete entry.fTrack.fRawPtr;

   entry.fTrack = new Track{3};
   t->Fill();
   delete entry.fTrack.fRawPtr;

   t->Write();
   f->Close();

   return 0;
}
