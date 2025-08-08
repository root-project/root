{
   // usage: root[0] .x testrd.C

#ifndef ClingWorkAroundMissingDynamicScope
   // gSystem->Load("sueloader_C");
   // gSystem->Load("ConfigRecord_cxx");
#endif

   ConfigRecord* record = 0;
   TFile* f = new TFile("configtest.root", "READ");
   TTree* t = (TTree*) f->Get("Config");
   t->SetBranchAddress("ConfigRecord", &record);

   for (Int_t i = 0; i < 5; ++i) {
      Int_t nbytes = t->GetEntry(i);
      cout << "Retrieved entry " << i << " with " << nbytes << " bytes." << endl;
      record->Print();
   }

   f->Close();
}

