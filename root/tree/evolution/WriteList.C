{
   gROOT->ProcessLine(".L MyClassList.cxx+");
   TFile *f = new TFile("listfile.root","RECREATE");
   TTree *t = new TTree("tree","test tree");
   TopLevel *obj = new TopLevel;
   t->Branch("Top","TopLevel",&obj,32000,0);
   t->Branch("TopSplit99.","TopLevel",&obj,32000,99);
   obj->AddTrack(33);
   obj->AddTrack(77);
   t->Fill();
   f->Write();
   f->Close();
}