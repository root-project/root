// 2-D histograms with alphanumeric labels
// author; Rene Brun
TCanvas *hlabels2()
{
   const Int_t nx = 12;
   const Int_t ny = 20;
   const char *month[nx]  = {"January","February","March","April","May",
      "June","July","August","September","October","November",
      "December"};
   const char *people[ny] = {"Jean","Pierre","Marie","Odile","Sebastien",
      "Fons","Rene","Nicolas","Xavier","Greg","Bjarne","Anton",
      "Otto","Eddy","Peter","Pasha","Philippe","Suzanne","Jeff",
      "Valery"};
   TCanvas *c1 = new TCanvas("c1","demo bin labels",10,10,600,600);
   c1->SetGrid();
   c1->SetLeftMargin(0.15);
   c1->SetBottomMargin(0.15);
   TH2F *h = new TH2F("h","test",3,0,3,2,0,2);
   h->SetCanExtend(TH1::kAllAxes);
   h->SetStats(0);
   gRandom->SetSeed();
   for (Int_t i=0;i<15000;i++) {
      Int_t rx = gRandom->Rndm()*nx;
      Int_t ry = gRandom->Rndm()*ny;
      h->Fill(people[ry],month[rx],1);
   }
   h->LabelsDeflate("X");
   h->LabelsDeflate("Y");
   h->LabelsOption("v");
   h->Draw("text");

   TPaveText *pt = new TPaveText(0.6,0.85,0.98,0.98,"brNDC");
   pt->SetFillColor(18);
   pt->SetTextAlign(12);
   pt->AddText("Use the axis Context Menu LabelsOption");
   pt->AddText(" \"a\"   to sort by alphabetic order");
   pt->AddText(" \">\"   to sort by decreasing values");
   pt->AddText(" \"<\"   to sort by increasing values");
   pt->Draw();
   return c1;
}
