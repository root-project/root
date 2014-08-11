{
   // Histogram consumer script. Create a canvas and 3 pads. Connect
   // to memory mapped file "hsimple.map", that was created by hprod.C.
   // It reads the histograms from shared memory and displays them
   // in the pads (sleeping for 0.1 seconds before starting a new read-out
   // cycle). This script runs in an infinite loop, so use ctrl-c to stop it.
   //Author: Fons Rademakers

   gROOT->Reset();

   // Create a new canvas and 3 pads
   TCanvas *c1;
   TPad *pad1, *pad2, *pad3;
   if (!gROOT->IsBatch()) {
      c1 = new TCanvas("c1","Shared Memory Consumer Example",200,10,700,780);
      pad1 = new TPad("pad1","This is pad1",0.02,0.52,0.98,0.98,21);
      pad2 = new TPad("pad2","This is pad2",0.02,0.02,0.48,0.48,21);
      pad3 = new TPad("pad3","This is pad3",0.52,0.02,0.98,0.48,21);
      pad1->Draw();
      pad2->Draw();
      pad3->Draw();
   }

   // Open the memory mapped file "hsimple.map" in "READ" (default) mode.
   mfile = TMapFile::Create("hsimple.map");

   // Print status of mapped file and list its contents
   mfile->Print();
   mfile->ls();

   // Create pointers to the objects in shared memory.
   TH1F     *hpx    = 0;
   TH2F     *hpxpy  = 0;
   TProfile *hprof  = 0;

   // Loop displaying the histograms. Once the producer stops this
   // script will break out of the loop.
   Double_t oldentries = 0;
   while (1) {
      hpx    = (TH1F *) mfile->Get("hpx", hpx);
      hpxpy  = (TH2F *) mfile->Get("hpxpy", hpxpy);
      hprof  = (TProfile *) mfile->Get("hprof", hprof);
      if (hpx->GetEntries() == oldentries) break;
      oldentries = hpx->GetEntries();
      if (!gROOT->IsBatch()) {
         pad1->cd();
         hpx->Draw();
         pad2->cd();
         hprof->Draw();
         pad3->cd();
         hpxpy->Draw("cont");
         c1->Modified();
         c1->Update();
      } else {
         printf("Entries, hpx=%d, Mean=%g, RMS=%g\n",hpx->GetEntries(),hpx->GetMean(),hpx->GetRMS());
      }
      gSystem->Sleep(100);   // sleep for 0.1 seconds
      if (gSystem->ProcessEvents())
         break;
   }
}
