void padRefresh(TPad *pad,int flag=0);
void hksimple()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*  This script illustrates the advantages of a TH1K histogram
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  TH1 *hpx[3];
  int j;

// Create a new canvas.
  c1 = new TCanvas("c1","Dynamic Filling Example",200,10,700,500);
  c1->SetFillColor(42);
  c1->GetFrame()->SetFillColor(21);
  c1->GetFrame()->SetBorderSize(6);
  c1->GetFrame()->SetBorderMode(-1);

// Create a normal histogram and two TH1K histograms
  hpx[0]    = new TH1F("hp0","This is the px distribution",1000,-4,4);
  hpx[1]    = new TH1K("hk1","This is the px distribution",1000,-4,4);
  hpx[2]    = new TH1K("hk2","This is the px distribution",1000,-4,4,16);
  c1->Divide(1,3);
   for (j=0;j<3;j++) {c1->cd(j+1); hpx[j]->Draw();hpx[j]->SetFillColor(48);}

// Fill histograms randomly
  gRandom->SetSeed();
  Float_t px, py, pz;
  const Int_t kUPDATE = 10;
  for (Int_t i = 0; i <= 600; i++) {
     gRandom->Rannor(px,py);
     for (j=0;j<3;j++) {hpx[j]->Fill(px);}
     if (i && (i%kUPDATE) == 0) {
           padRefresh(c1);
     }
  }
}
void padRefresh(TPad *pad,int flag)
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
