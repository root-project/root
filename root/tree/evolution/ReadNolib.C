{
   TFile *f;
   TTree *t;

    f = new TFile("vectorfile.root","READ");
   f->GetObject("tree",t);
   //TopLevel *obj = 0;
   //t->SetBranchAddress("Top",&obj);
   t->Scan("fTracks.fEnergy");
   t->Scan("TopSplit99.fTracks.fEnergy");
   f->Close();
   delete f;

   f = new TFile("oldfile.root","READ");
   f->GetObject("tree",t);
   //TopLevel *obj = 0;
   //t->SetBranchAddress("Top",&obj);
   //t->Scan("fTracks.fEnergy");
   t->Scan("TopSplit99.fTracks.fEnergy");
   f->Close();
   delete f;

   f = new TFile("listfile.root","READ");
   f->GetObject("tree",t);
   //TopLevel *obj = 0;
   //t->SetBranchAddress("Top",&obj);
   //t->Scan("fTracks.fEnergy");
   t->Scan("TopSplit99.fTracks.fEnergy");
   f->Close();
   delete f;
}