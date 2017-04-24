// Brief: Different charges depending on region

#include <iostream>
#include <fstream>
using namespace std;

void tprofile2poly_realistic(Int_t numEvents=100000)
{
   TCanvas *c1 = new TCanvas("c1", "2 areas w/ unusual charge", 900, 700);
   c1->Divide(3, 2);

   // -------------------- Construct Reference plot bins ------------------------
   auto new_avg = new TProfile2Poly();
   auto ref_abs = new TH2Poly();
   auto ref_avg = new TProfile2D("", "", 60, -15, 15, 60, -15, 15, 0, 200);

   float minx = -15;
   float maxx = 15;
   float miny = -15;
   float maxy = 15;
   float binsz = 0.5;

   for (float i = minx; i < maxx; i += binsz) {
      for (float j = miny; j < maxy; j += binsz) {
         ref_abs->AddBin(i, j, i + binsz, j + binsz);
         new_avg->AddBin(i, j, i + binsz, j + binsz);
      }
   }

   // -------------------- Construct detector bins ------------------------
   auto h2p = new TH2Poly();
   auto tp2p = new TProfile2Poly();
   ifstream infile;
   infile.open("./tutorials/hist/TProfile2Poly/test_data/cms_forward3");

   vector<pair<Double_t, Double_t>> allCoords;
   Double_t a, b;
   while (infile >> a >> b) {
      pair<Double_t, Double_t> coord(a, b);
      allCoords.push_back(coord);
   }

   if (allCoords.size() % 3 != 0) {
      cout << "[ERROR] Bad file" << endl;
      return;
   }

   Double_t x[3], y[3];
   for (Int_t i = 0; i < allCoords.size(); i += 3) {
      x[0] = allCoords[i + 0].first;
      y[0] = allCoords[i + 0].second;
      x[1] = allCoords[i + 1].first;
      y[1] = allCoords[i + 1].second;
      x[2] = allCoords[i + 2].first;
      y[2] = allCoords[i + 2].second;
      h2p->AddBin(3, x, y);
      tp2p->AddBin(3, x, y);
   }

   // -------------------- Simulate particles ------------------------
   TRandom ran;
   int NUM_LS = 3;

   for (int i = 0; i <= NUM_LS - 1; ++i) { // LumiSection
      for (int j = 0; j < numEvents; ++j) {   // Events
         Double_t r1 = ran.Gaus(0, 10);
         Double_t r2 = ran.Gaus(0, 8);
         Double_t rok = ran.Gaus(20, 2);
         Double_t rbad1 = ran.Gaus(2, 5);

         Double_t val = rok;

         if (r2 > 3 && r2 < 8 && r1 > 1 && r1 < 5) val = rok - rbad1;
         if (r2 > -10 && r2 < -2 && r1 > -1 && r1 < 4) val = rok + rbad1;

         ref_abs->Fill(r1, r2, val);
         ref_avg->Fill(r1, r2, val);
         new_avg->Fill(r1, r2, val);

         h2p->Fill(r1, r2, val);
         tp2p->Fill(r1, r2, val);
      }
   }

   // -------------------- Display end state ------------------------
   c1->cd(1);
   ref_abs->SetStats(false);
   ref_abs->SetTitle("TH2Poly: total hits");
   ref_abs->Draw("COLZ");

   c1->cd(2);
   ref_avg->SetStats(false);
   ref_avg->SetTitle("TProfile2D: average charge");
   ref_avg->Draw("COLZ");

   c1->cd(3);
   new_avg->SetStats(false);
   new_avg->SetTitle("TProfile2Poly: average charge");
   new_avg->Draw("COLZ");

   c1->cd(4);
   h2p->SetStats(false);
   h2p->SetTitle("TH2Poly: total hits on detector");
   h2p->Draw("COLZ");

   c1->cd(6);
   tp2p->SetStats(false);
   tp2p->SetTitle("TProfile2Ploly: average charge on detector");
   tp2p->Draw("COLZ");
}
