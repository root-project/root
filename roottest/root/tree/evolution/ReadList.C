{
   gROOT->ProcessLine(".L MyClassList.cxx+");
   TFile *f = new TFile("clonesfile.root","READ");
   TTree *t; f->GetObject("tree",t);
   //TopLevel *obj = 0;
   //t->SetBranchAddress("Top",&obj);
   t->Scan("fTracks.fEnergy","","colsize=35");
   t->Scan("TopSplit99.fTracks.fEnergy","","colsize=35");
   t->Scan("fTracksPtr.fEnergy","","colsize=35");
   t->Scan("TopSplit99.fTracksPtr.fEnergy","","colsize=35");
   t->Scan("TopCl.fTracks.fEnergy","","colsize=35");
   t->Scan("TopClSplit99.fTracks.fEnergy","","colsize=35");
   t->Scan("TopCl.fTracksPtr.fEnergy","","colsize=35");
   t->Scan("TopClSplit99.fTracksPtr.fEnergy","","colsize=35");
   f->Close();
   delete f;

   f = new TFile("vectorfile.root","READ");
   f->GetObject("tree",t);
   //TopLevel *obj = 0;
   //t->SetBranchAddress("Top",&obj);
   t->Scan("fTracks.fEnergy","","colsize=35");
   t->Scan("TopSplit99.fTracks.fEnergy","","colsize=35");
   t->Scan("fTracksPtr.fEnergy","","colsize=35");
   t->Scan("TopSplit99.fTracksPtr.fEnergy","","colsize=35");
   t->Scan("TopCl.fTracks.fEnergy","","colsize=35");
   t->Scan("TopClSplit99.fTracks.fEnergy","","colsize=35");
   t->Scan("TopCl.fTracksPtr.fEnergy","","colsize=35");
   t->Scan("TopClSplit99.fTracksPtr.fEnergy","","colsize=35");
   f->Close();
   delete f;

   f = new TFile("listfile.root","READ");
   f->GetObject("tree",t);
   //TopLevel *obj = 0;
   //t->SetBranchAddress("Top",&obj);
   t->Scan("fTracks.fEnergy","","colsize=35");
   t->Scan("TopSplit99.fTracks.fEnergy","","colsize=35");
   t->Scan("fTracksPtr.fEnergy","","colsize=35");
   t->Scan("TopSplit99.fTracksPtr.fEnergy","","colsize=35");
   t->Scan("TopCl.fTracks.fEnergy","","colsize=35");
   t->Scan("TopClSplit99.fTracks.fEnergy","","colsize=35");
   t->Scan("TopCl.fTracksPtr.fEnergy","","colsize=35");
   t->Scan("TopClSplit99.fTracksPtr.fEnergy","","colsize=35");
   f->Close();
   delete f;
}
