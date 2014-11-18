void testRandomMultiGaus() { 


   TH1D h1 = TH1D("h1","h1",100,-3, 3); 
   h1.FillRandom("gaus"); 
   TFitResultPtr r = h1.Fit("gaus","S");

   TMatrixDSym m = r->GetCovarianceMatrix(); 

   TVectorD par(3, r->GetParams()); 

   TVectorD v(3); 

   par.Print();
   m.Print();

   TH2D *h2 = new TH2D("h1","h1",100,-0.5, 0.5, 100, 0., 1.5); 
   for (int i = 0; i < 1000; ++i) { 
      v.RandomizeGaus(par, m); 
      if (i<5) v.Print();
      h2->Fill(v(1),v(2) ); 
   }
   h2->Draw("COLZ");
}
