void earth() {
  //this tutorial illustrate the special contour options
  //    "AITOFF"     : Draw a contour via an AITOFF projection
  //    "MERCATOR"   : Draw a contour via an Mercator projection
  //    "SINUSOIDAL" : Draw a contour via an Sinusoidal projection
  //    "PARABOLIC"  : Draw a contour via an Parabolic projection
  // from an original macro sent by Ernst-Jan Buis
   
   gStyle->SetPalette(1);
   gStyle->SetOptTitle(1);
   gStyle->SetOptStat(0);

   TCanvas *c1 = new TCanvas("c1","earth_projections",1000,800);
   c1->Divide(2,2);

   TH2F *h1 = new TH2F("h1","Aitoff",    180, -180, 180, 179, -89.5, 89.5);
   TH2F *h2 = new TH2F("h2","Mercator",  180, -180, 180, 161, -80.5, 80.5);
   TH2F *h3 = new TH2F("h3","Sinusoidal",180, -180, 180, 181, -90.5, 90.5);
   TH2F *h4 = new TH2F("h4","Parabolic", 180, -180, 180, 181, -90.5, 90.5);

   ifstream in;
   in.open("earth.dat");
   Float_t x,y;
   while (1) {
     in >> x >> y;
     if (!in.good()) break;
     h1->Fill(x,y, 1);
     h2->Fill(x,y, 1);
     h3->Fill(x,y, 1);
     h4->Fill(x,y, 1);
   }
   in.close();

   c1->cd(1);
   h1->Draw("z aitoff");

   c1->cd(2);
   h2->Draw("z mercator");
   
   c1->cd(3);
   h3->Draw("z sinusoidal");
   
   c1->cd(4);
   h4->Draw("z parabolic");
}
