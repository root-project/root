//Setting alphanumeric labels 
//Author: Rene Brun
void labels2()
{
   const Int_t nx = 12;
   const Int_t ny = 20;
   char *month[nx]  = {"January","February","March","April",
      "May","June","July","August","September","October",
      "November","December"};
   char *people[ny] = {"Jean","Pierre","Marie","Odile",
      "Sebastien","Fons","Rene","Nicolas","Xavier","Greg",
      "Bjarne","Anton","Otto","Eddy","Peter","Pasha",
      "Philippe","Suzanne","Jeff","Valery"};
   TCanvas *c1 = new TCanvas("c1","demo bin labels",
      10,10,800,800);
   c1->SetGrid();
   c1->SetLeftMargin(0.15);
   c1->SetBottomMargin(0.15);
   TH2F *h = new TH2F("h","test",nx,0,nx,ny,0,ny);
   for (Int_t i=0;i<5000;i++) {
      h->Fill(gRandom->Gaus(0.5*nx,0.2*nx), 
         gRandom->Gaus(0.5*ny,0.2*ny));
   }
   h->SetStats(0);
   for (i=1;i<=nx;i++) h->GetXaxis()->SetBinLabel(i,month[i-1]);
   for (i=1;i<=ny;i++) h->GetYaxis()->SetBinLabel(i,people[i-1]);
   h->Draw("text");
   
   TPaveText *pt = new TPaveText(0.6,0.85,0.98,0.98,"brNDC");
   pt->SetFillColor(18);
   pt->SetTextAlign(12);
   pt->AddText("Use the axis Context Menu LabelsOption");
   pt->AddText(" \"a\"   to sort by alphabetic order");
   pt->AddText(" \">\"   to sort by decreasing vakues");
   pt->AddText(" \"<\"   to sort by increasing vakues");
   pt->Draw();
}
