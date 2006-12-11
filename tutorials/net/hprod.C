{
   // Histogram producer script. This script creates a memory mapped
   // file and stores three histogram objects in it (a TH1F, a TH2F and a
   // TProfile). It then fills, in an infinite loop (so use ctrl-c to
   // stop this script), the three histogram objects with random numbers.
   // Every 10 fills the objects are updated in shared memory.
   // Use the hcons.C script to map this file and display the histograms.
   //Author: Fons Rademakers
   
   gROOT->Reset();

   // Create a new memory mapped file. The memory mapped file can be
   // opened in an other process on the same machine and the objects
   // stored in it can be accessed.

   TMapFile::SetMapAddress(0xb46a5000);
   mfile = TMapFile::Create("hsimple.map","RECREATE", 1000000,
                            "Demo memory mapped file with histograms");

   // Create a 1d, a 2d and a profile histogram. These objects will
   // be automatically added to the current directory, i.e. mfile.
   hpx    = new TH1F("hpx","This is the px distribution",100,-4,4);
   hpxpy  = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
   hprof  = new TProfile("hprof","Profile of pz versus px",100,-4,4,0,20);

   // Set a fill color for the TH1F
   hpx->SetFillColor(48);

   // Print status of mapped file
   mfile->Print();

   // Endless loop filling histograms with random numbers
   Float_t px, py, pz;
   int ii = 0;
   while (1) {
      gRandom->Rannor(px,py);
      pz = px*px + py*py;
      hpx->Fill(px);
      hpxpy->Fill(px,py);
      hprof->Fill(px,pz);
      if (!(ii % 10)) {
         mfile->Update();       // updates all objects in shared memory
         if (!ii) mfile->ls();  // print contents of mapped file after
      }                         // first update
      ii++;
   }
}
