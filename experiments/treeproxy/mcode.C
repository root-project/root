   if (fChain == 0) return;

   Int_t nentries = Int_t(fChain->GetEntriesFast());

if (arg==2) {
   fChain->SetBranchStatus("*",kFALSE);
   fChain->SetBranchStatus("px",kTRUE);
}

TCanvas *c1 = (TCanvas*)gROOT->GetListOfCanvases()->FindObject("c1");
if (c1==0) {
   c1 = new TCanvas;
   c1->Divide(2,1);
}
TH1F *h = new TH1F("h","h",100,-4,4);
h->SetBit(kCanDelete);
TH1F *h2 = new TH1F("h2","h2",100,-4,4);
h2->SetBit(kCanDelete);
   Int_t nbytes = 0, nb = 0;
   for (Int_t jentry=0; jentry<nentries;jentry++) {
      Int_t ientry = LoadTree(jentry); //in case of a TChain, ientry is the entry number in the current file
      if (ientry < 0) break;
      
      #ifdef mc02_cxx
        if (arg==1) {
           nb = b_px->GetEntry(jentry);
        } else 
           nb = fChain->GetEntry(jentry);
      #endif
      #ifdef mc01_cxx
         fDirector.fEntry = jentry;
      #endif

      nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      h->Fill(px);
      if (arg==3) {
         float y;
         y = px * px;
         y = px * px;
         y = px * px;
         y = px * px;
      }
      if (px<0 && px>=2) h2->Fill(px);
      if (px>0 && px<=2) h2->Fill(px+2);
#ifdef TEST_WRITE
      // we now prevent writing to the proxy.
      px = 33;
#endif
   }
c1->cd(1);
h->Draw();
c1->cd(2);
h2->Draw();

