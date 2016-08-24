{
   gStyle->SetOptStat(0);   
   auto c1 = new TCanvas("c1", "A ratio example");
   auto h1 = new TH1D("h1", "h1", 50, 0, 10);
   auto h2 = new TH1D("h2", "h2", 50, 0, 10);
   auto f1 = new TF1("f1", "exp(- x/[0] )");
   f1->SetParameter(0, 3);
   h1->FillRandom("f1", 1900);
   h2->FillRandom("f1", 2000); 
   h1->Sumw2();
   h2->Scale(1.9 / 2.);
   auto rp = new TRatioPlot(h1, h2, "rp", "rp", "pois", "hist", "E", "AP", 1., 1.);
   rp->SetTicks(0, 1);
   rp->Draw();
   c1->Update();
   return c1;
}
