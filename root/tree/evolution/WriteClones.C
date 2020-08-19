{
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L MyClassClones.cxx+");
#endif
   TFile *f = new TFile("clonesfile.root","RECREATE");
   TTree *t = new TTree("tree","test tree");
   TopLevel *obj = new TopLevel;
   TopLevelCl *objcl = new TopLevelCl;
   t->Branch("Top","TopLevel",&obj,32000,0);
   t->Branch("TopSplit99.","TopLevel",&obj,32000,99);
   t->Branch("TopCl","TopLevelCl",&objcl,32000,0);
   t->Branch("TopClSplit99.","TopLevelCl",&objcl,32000,99);
   obj->AddTrack(33);
   obj->AddTrack(77);
   objcl->AddTrack(11);
   objcl->AddTrack(66);
   t->Fill();
   f->Write();
   f->Close();
}
