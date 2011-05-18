// 1-D histograms with alphanumeric labels
// author; Rene Brun
void hlabels1()
{
   const Int_t nx = 20;
   char *people[nx] = {"Jean","Pierre","Marie","Odile","Sebastien",
      "Fons","Rene","Nicolas","Xavier","Greg","Bjarne","Anton","Otto",
      "Eddy","Peter","Pasha","Philippe","Suzanne","Jeff","Valery"};
   TCanvas *c1 = new TCanvas("c1","demo bin labels",10,10,900,500);
   c1->SetGrid();
   c1->SetTopMargin(0.15);
   TH1F *h = new TH1F("h","test",3,0,3);
   h->SetStats(0);
   h->SetFillColor(38);
   h->SetBit(TH1::kCanRebin);
   for (Int_t i=0;i<5000;i++) {
      Int_t r = gRandom->Rndm()*20;
      h->Fill(people[r],1);
   }
   h->LabelsDeflate();
   h->Draw();
   TPaveText *pt = new TPaveText(0.7,0.85,0.98,0.98,"brNDC");
   pt->SetFillColor(18);
   pt->SetTextAlign(12);
   pt->AddText("Use the axis Context Menu LabelsOption");
   pt->AddText(" \"a\"   to sort by alphabetic order");
   pt->AddText(" \">\"   to sort by decreasing values");
   pt->AddText(" \"<\"   to sort by increasing values");
   pt->Draw();
}
