//Example of script drawing many small pads in one canvas
//and zooming automatically one small pad in a large canvas
//when the mouse points to the small pad.
//Author; Rene Brun
      
TCanvas *c1, *c2;
TPad *selold = 0;
void thumbnail() {
   c1 = new TCanvas("c1","thumbnails",10,10,600,800);
   Int_t nx = 10;
   Int_t ny = 15;
   c1->Divide(nx,ny);
   TH1F *h = new TH1F("h","h",100,-3,3);
   for (Int_t i=1;i<=nx*ny;i++) {
      c1->cd(i);
      h->FillRandom("gaus",10);
      h->DrawCopy();
   }
   c1->AddExec("tnail","tnail()");
   c2 = new TCanvas("c2","c2",650,10,800,600);
}
void tnail() {
   TPad *sel = (TPad*)gPad->GetSelectedPad();
   int px = gPad->GetEventX();
   int py = gPad->GetEventY();
   if (sel && sel != c1 && sel != c2) {
      if (selold) delete selold;
      c2->cd();
      TPad *newpad = (TPad*)sel->Clone();
      c2->GetListOfPrimitives()->Add(newpad);
      newpad->SetPad(0,0,1,1);
      selold = newpad;
      c2->Update();
   }
}
