{
   TFile *f = TFile::Open("MyTree.root", "recreate");
   TTree *t = new TTree("t", "t");
   TParticle p;
   t->Branch("p1", &p, 32000, 0); // NO splitting
   t->Branch("p2.", &p, 32000, 0); // NO splitting
   t->Fill();
   t->Write();
   // t->Print();

   t->MakeProxy("MyTree99", "MyTree99_script.cxx", 0, 0, 99);
   t->MakeProxy("MyTree0", "MyTree0_script.cxx", 0, 0, 0);
   t->Process("MyTree99.h","goff");
   t->Process("MyTree0.h","goff");
   delete f;
}
