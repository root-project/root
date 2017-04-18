void tprofile2poly_tprofile2d_sim2()
{
  // CREATE PLOT STURCTURES
   auto c1 = new TCanvas("c1","Profile histogram example",200,10,700,500);
   c1->Divide(2,1);

   auto TP2D  = new TProfile2D("hprof2d","Profile of pz versus px and py",100,-10,10,100,-10,10,0,100);
   auto TP2P = new TProfile2Poly();

   TP2P->Honeycomb(-10,-10,.1,100,130);
   TP2P->SetName("mine");
   TP2P->SetTitle("mine");

   int value = 1;
   Float_t px, py, pz;
   for ( Int_t i=0; i<30000; i++) {
      gRandom->Rannor(px,py);
      value = px*px + py*py;
      TP2D->Fill(px,py,value);
      TP2P->Fill(px,py,value);
   }
   c1->cd(1);
   TP2D->Draw("COLZ");

   c1->cd(2);
   TP2P->Draw("COLZ");
}
