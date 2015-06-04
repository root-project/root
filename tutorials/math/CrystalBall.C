// example of CrystalBall Function and its distribution (pdf and cdf)

void CrystalBall()  {

   auto c1 = new TCanvas();
   c1->Divide(1,3);

   // crystal ball function
   c1->cd(1);

   auto f1  = new TF1("f1","crystalball",-5,5);
   f1->SetParameters(1, 0, 1, 2, 0.5);
   f1->SetLineColor(kRed);
   f1->Draw();
   // use directly the functionin ROOT::MATH note that the parameters definition is different is (alpha, n sigma, mu)
   auto f2 = new TF1("f2","ROOT::Math::crystalball_function(x, 2, 1, 1, 0)",-5,5);
   f2->SetLineColor(kGreen); 
   f2->Draw("same");
   auto f3 = new TF1("f3","ROOT::Math::crystalball_function(x, 2, 2, 1, 0)",-5,5);
   f3->SetLineColor(kBlue);
   f3->Draw("same");

   auto legend = new TLegend(0.7,0.6,0.9,1.);
   legend->AddEntry(f1,"N=0.5 alpha=2","L");
   legend->AddEntry(f2,"N=1   alpha=2","L");
   legend->AddEntry(f3,"N=2   alpha=2","L");
   legend->Draw();

   c1->cd(2);
   auto pdf1  = new TF1("pdf","crystalballn",-5,5);
   pdf1->SetParameters(2, 0, 1, 2, 3);
   pdf1->Draw();
   auto pdf2 = new TF1("pdf","ROOT::Math::crystalball_pdf(x, 3, 1.01, 1, 0)",-5,5);
   pdf2->SetLineColor(kBlue); 
   pdf2->Draw("same");
   auto pdf3 = new TF1("pdf","ROOT::Math::crystalball_pdf(x, 2, 2, 1, 0)",-5,5);
   pdf3->SetLineColor(kGreen);
   pdf3->Draw("same");

   legend = new TLegend(0.7,0.6,0.9,1.);
   legend->AddEntry(pdf1,"N=3    alpha=2","L");
   legend->AddEntry(pdf2,"N=1.01 alpha=3","L");
   legend->AddEntry(pdf3,"N=2    alpha=3","L");
   legend->Draw();

   c1->cd(3);
   auto cdf  = new TF1("cdf","ROOT::Math::crystalball_cdf(x, 1.2, 2, 1, 0)",-5,5);
   auto cdfc  = new TF1("cdfc","ROOT::Math::crystalball_cdf_c(x, 1.2, 2, 1, 0)",-5,5);
   cdf->SetLineColor(kRed-3);
   cdf->SetMinimum(0.);
   cdf->SetMaximum(1.);
   cdf->Draw();
   cdfc->SetLineColor(kMagenta);
   cdfc->Draw("Same");

   legend = new TLegend(0.7,0.7,0.9,1.);
   legend->AddEntry(cdf,"N=1.2 alpha=2","L");
   legend->AddEntry(cdfc,"N=1.2 alpha=2","L");
   legend->Draw();
   
}
