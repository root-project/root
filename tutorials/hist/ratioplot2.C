{
   gStyle->SetOptStat(0);   
   auto c1 = new TCanvas("c1", "fit residual simple"); 
   auto h1 = new TH1D("h1", "h1", 50, -5, 5);
   h1->FillRandom("gaus", 2000);
   h1->Fit("gaus");
   c1->Clear(); // Fit does not draw into correct pad
   auto rp1 = new TRatioPlot((TH1*)h1->Clone(), "rp1", "rp1");
   rp1->Draw();
   rp1->GetLowYaxis()->SetTitle("ratio");
   rp1->GetUpYaxis()->SetTitle("entries");
   c1->Update();
   return c1;
}
