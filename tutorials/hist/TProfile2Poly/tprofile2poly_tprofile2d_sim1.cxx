// Brief: Simulate a TProfile2D with TProfile2Poly, and see if results are the same

void tprofile2poly_tprofile2d_sim1()
{
   // CREATE PLOT STURCTURES
   auto c1 = new TCanvas("c1", "Profile histogram example", 200, 10, 700, 500);
   auto TP2D = new TProfile2D("", "", 100, -10, 10, 100, -10, 10, 0, 100);
   auto TP2P = new TProfile2Poly();

   TP2P->SetName("mine");
   TP2P->SetTitle("mine");

   // create the grid in such a way that it is the same as hprof2d bins
   float minx = -10;
   float maxx = 10;
   float miny = -10;
   float maxy = 10;
   float binsz = 0.2;

   for (float i = minx; i < maxx; i += binsz) {
      for (float j = miny; j < maxy; j += binsz) {
         TP2P->AddBin(i, j, i + binsz, j + binsz);
      }
   }

   c1->Divide(2, 1);
   int z;

   Float_t px, py;
   for (Int_t i = 0; i < 100000; i++) {
      gRandom->Rannor(px, py);
      z = px * px + py * py;
      TP2D->Fill(px, py, z);
      TP2P->Fill(px, py, z);
   }
   c1->cd(1);
   TP2D->Draw("COLZ");

   c1->cd(2);
   TP2P->Draw("COLZ");
}
