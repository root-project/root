{
#ifndef ClingWorkAroundUnnamedInclude
#include <vector>
#endif
   TFile *f = TFile::Open("emulvector.root","RECREATE");
   
   TTree * tr = new TTree("tr","testtree");
   
   Int_t evtNr(0);
   Int_t runNr(1);
   std::vector<UShort_t> * tag = new std::vector<UShort_t>;
   std::vector<UInt_t> * itag = new std::vector<UInt_t>;
   
   tr->Branch("evtNr", &evtNr, "evtNr/I");
   tr->Branch("runNr", &runNr, "runNr/I");
   tr->Branch("tag", "std::vector<UShort_t>", &tag, 32000,0);
   tr->Branch("itag", "std::vector<UInt_t>", &itag, 32000,0);
   
   // evt 1: run 1
   evtNr++; 
   tag->clear(); tag->push_back(1); tag->push_back(2);  tag->push_back(3); tag->push_back(4); 
   itag->clear(); itag->push_back(1); itag->push_back(2);  itag->push_back(3); itag->push_back(4); 
   tr->Fill();
   // evt 2: run 1
   evtNr++; 
   tag->clear(); tag->push_back(1); tag->push_back(3); tag->push_back(4); 
   itag->clear(); itag->push_back(1); itag->push_back(3); itag->push_back(4); 
   tr->Fill();
   tr->Write();
   f->Close();
   
   f = TFile::Open("emulvector.root","READ");
   TTree *out; f->GetObject("tr",out);
   out->Scan();
   
#ifndef ClingWorkAroundBrokenUnnamedReturn
   return 0;
#else
   // Make exit code 0;
   out = 0;
#endif
}
