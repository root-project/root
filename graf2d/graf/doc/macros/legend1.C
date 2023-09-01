{
   TCanvas *c1 = new TCanvas("c1","c1",600,500);
   gStyle->SetOptStat(0);

   TH1F *h1 = new TH1F("h1","TLegend Example",200,-10,10);
   h1->FillRandom("gaus",30000);
   h1->SetFillColor(kGreen);
   h1->SetFillStyle(3003);
   h1->Draw();

   TF1 *f1=new TF1("f1","1000*TMath::Abs(sin(x)/x)",-10,10);
   f1->SetLineColor(kBlue);
   f1->SetLineWidth(4);
   f1->Draw("same");

   const Int_t n = 20;
   Double_t x[n], y[n], ex[n], ey[n];
   for (Int_t i=0;i<n;i++) {
      x[i]  = i*0.1;
      y[i]  = 1000*sin(x[i]+0.2);
      x[i]  = 17.8*x[i]-8.9;
      ex[i] = 1.0;
      ey[i] = 10.*i;
   }
   TGraphErrors *gr = new TGraphErrors(n,x,y,ex,ey);
   gr->SetName("gr");
   gr->SetLineColor(kRed);
   gr->SetLineWidth(2);
   gr->SetMarkerStyle(21);
   gr->SetMarkerSize(1.3);
   gr->SetMarkerColor(7);
   gr->Draw("P");

   leg = new TLegend(0.1,0.7,0.48,0.9);
   leg->SetHeader("The Legend Title");
   leg->AddEntry(h1,"Histogram filled with random numbers","f");
   leg->AddEntry("f1","Function abs(#frac{sin(x)}{x})","l");
   leg->AddEntry("gr","Graph with error bars","lep");
   leg->Draw();

   return c1;
}

