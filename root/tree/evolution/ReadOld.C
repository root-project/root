{
   gROOT->ProcessLine(".L MyClassOld.cxx+");
   TFile *f = new TFile("oldfile.root","READ");
   TTree *t; f->GetObject("tree",t);
   //TopLevel *obj = 0;
   //t->SetBranchAddress("Top",&obj);
   t->Scan("fTracks.fEnergy");
   t->Scan("TopSplit99.fTracks.fEnergy");
   f->Close();
}