{
   gROOT->ProcessLine(".L MyClassClones.cxx+");
   TFile *f;
   TTree *t;

   if (1) {
      f = new TFile("clonesfile.root","READ");
      f->GetObject("tree",t);
      //TopLevel *obj = 0;
      //t->SetBranchAddress("Top",&obj);
      cout << "Processing clonesfile.root\n";
      t->Scan("fTracks.fEnergy","","colsize=35"); cout << flush;
      t->Scan("TopSplit99.fTracks.fEnergy","","colsize=35"); cout << flush;
      t->Scan("fTracksPtr.fEnergy","","colsize=35"); cout << flush;
      t->Scan("TopSplit99.fTracksPtr.fEnergy","","colsize=35"); cout << flush;
      t->Scan("TopCl.fTracks.fEnergy","","colsize=35"); cout << flush;
      t->Scan("TopClSplit99.fTracks.fEnergy","","colsize=35"); cout << flush;
      t->Scan("TopCl.fTracksPtr.fEnergy","","colsize=35"); cout << flush;
      t->Scan("TopClSplit99.fTracksPtr.fEnergy","","colsize=35"); cout << flush;
      f->Close();
      delete f;
   }

   f = new TFile("listfile.root","READ");
   f->GetObject("tree",t);
   //TopLevel *obj = 0;
   //t->SetBranchAddress("Top",&obj);
   cerr << endl << flush;
   cout << "Processing listsfile.root\n";
   t->Scan("fTracks.fEnergy","","colsize=35");
   t->Scan("TopSplit99.fTracks.fEnergy","","colsize=35");
   cout << flush;
   t->Scan("fTracksPtr.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopSplit99.fTracksPtr.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopCl.fTracks.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopClSplit99.fTracks.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopCl.fTracksPtr.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopClSplit99.fTracksPtr.fEnergy","","colsize=35"); cout << flush;
   f->Close();
   delete f;

   f = new TFile("vectorfile.root","READ");
   f->GetObject("tree",t);
   //TopLevel *obj = 0;
   //t->SetBranchAddress("Top",&obj);
   cout << "Processing vectorfile.root\n";
   t->Scan("fTracks.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopSplit99.fTracks.fEnergy","","colsize=35"); cout << flush;
   t->Scan("fTracksPtr.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopSplit99.fTracksPtr.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopCl.fTracks.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopClSplit99.fTracks.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopCl.fTracksPtr.fEnergy","","colsize=35"); cout << flush;
   t->Scan("TopClSplit99.fTracksPtr.fEnergy","","colsize=35"); cout << flush;
   f->Close();
   delete f;
      
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif

}
