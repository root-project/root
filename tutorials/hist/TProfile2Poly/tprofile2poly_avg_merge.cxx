// Brief: Individual merge of charge per lumisection, and running total charge

#include <iostream>
using namespace std;

void tprofile2poly_avg_merge()
{
   TCanvas *c1 = new TCanvas("c1", "multipads", 900, 700);

   int NUM_LS = 3;

   auto *whole = new TH2Poly();
   auto *whole_avg = new TProfile2Poly();
   auto *merged = new TProfile2Poly();

   auto *abso = new TH2Poly[NUM_LS];
   auto *avgs = new TProfile2Poly[NUM_LS];

   float minx = -4;
   float maxx = 4;
   float miny = -4;
   float maxy = 4;
   float binsz = 0.5;

   for (float i = minx; i < maxx; i += binsz) {
      for (float j = miny; j < maxy; j += binsz) {
         whole_avg->AddBin(i, j, i + binsz, j + binsz);
         whole->AddBin(i, j, i + binsz, j + binsz);
         merged->AddBin(i, j, i + binsz, j + binsz);
         for (int kk = 0; kk <= NUM_LS - 1; ++kk) {
            avgs[kk].AddBin(i, j, i + binsz, j + binsz);
            abso[kk].AddBin(i, j, i + binsz, j + binsz);
         }
      }
   }

   TRandom ran;

   c1->Divide(3, 3);
   Double_t ii = 0;

   for (int i = 0; i <= NUM_LS - 1; ++i) {
      for (int j = 0; j < 100000; ++j) {
         Double_t r1 = ran.Gaus(0, 2);
         Double_t r2 = ran.Gaus(0, 4);

         Double_t rok = ran.Gaus(20, 2);
         Double_t rbad1 = ran.Gaus(-10, 5);

         Double_t val = rok;
         ii = double(i) * 0.5;

         if (r2 > 0.5 && r2 < 1 && r1 > 1 + ii && r1 < 1.5 + ii) val = rok - rbad1;

         whole->Fill(r1, r2);
         abso[i].Fill(r1, r2);
         whole_avg->Fill(r1, r2, val);
         avgs[i].Fill(r1, r2, val);

         if (j % 50000 == 0) {
            c1->cd(8);
            whole_avg->SetStats(0);
            whole_avg->SetTitle("Running Average");
            whole_avg->Draw("COLZ TEXT");
            c1->Update();
         }
      }

      string title;

      c1->cd(i + 1);
      title = " avg charge in LumiSec " + to_string(i);
      avgs[i].SetStats(0);
      avgs[i].SetTitle(title.c_str());
      avgs[i].Draw("COLZ TEXT");
      c1->Update();

      c1->cd(i + 3 + 1);
      title = " abs hits in LumiSec " + to_string(i);
      abso[i].SetStats(0);
      abso[i].SetTitle(title.c_str());
      abso[i].Draw("COLZ TEXT");
      c1->Update();
   }

   c1->cd(9);
   whole->SetStats(0);
   whole->SetTitle("total hits");
   whole->Draw("COLZ TEXT");

   c1->cd(7);
   vector<TProfile2Poly *> tomerge;
   tomerge.push_back(&avgs[0]);
   tomerge.push_back(&avgs[1]);
   tomerge.push_back(&avgs[2]);

   merged->Merge(tomerge);

   merged->SetStats(0);
   merged->SetTitle("individually merged");
   merged->Draw("COLZ TEXT");
}
