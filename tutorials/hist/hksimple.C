/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Illustrates the advantages of a TH1K histogram
///
/// \macro_image
/// \macro_code
///
/// \author Victor Perevovchikov

void padRefresh(TPad *pad,int flag=0)
{
   if (!pad) return;
   pad->Modified();
   pad->Update();
   TList *tl = pad->GetListOfPrimitives();
   if (!tl) return;
   TListIter next(tl);
   TObject *to;
   while ((to=next())) {
     if (to->InheritsFrom(TPad::Class())) padRefresh((TPad*)to,1);}
   if (flag) return;
   gSystem->ProcessEvents();
}

void hksimple()
{
// Create a new canvas.
   TCanvas* c1 = new TCanvas("c1","Dynamic Filling Example",200,10,600,900);

// Create a normal histogram and two TH1K histograms
   TH1 *hpx[3];
   hpx[0]    = new TH1F("hp0","Normal histogram",1000,-4,4);
   hpx[1]    = new TH1K("hk1","Nearest Neighbour of order 3",1000,-4,4);
   hpx[2]    = new TH1K("hk2","Nearest Neighbour of order 16",1000,-4,4,16);
   c1->Divide(1,3);
   Int_t j;
   for (j=0;j<3;j++) {
      c1->cd(j+1);
      hpx[j]->SetFillColor(48);
      hpx[j]->Draw();
   }

// Fill histograms randomly
   gRandom->SetSeed(12345);
   Float_t px, py, pz;
   const Int_t kUPDATE = 10;
   for (Int_t i = 0; i <= 300; i++) {
      gRandom->Rannor(px,py);
      for (j=0;j<3;j++) {hpx[j]->Fill(px);}
      if (i && (i%kUPDATE) == 0) {
            padRefresh(c1);
      }
   }

   for (j=0;j<3;j++) hpx[j]->Fit("gaus");
   padRefresh(c1);
}
