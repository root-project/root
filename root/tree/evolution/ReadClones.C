{
   gROOT->ProcessLine(".L MyClassClones.cxx+");
   TFile *f = new TFile("clonesfile.root","READ");
   TTree *t; f->GetObject("tree",t);
   //TopLevel *obj = 0;
   //t->SetBranchAddress("Top",&obj);
   t->Scan("fTracks.fEnergy");
   t->Scan("TopSplit99.fTracks.fEnergy");
   f->Close();
}