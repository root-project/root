// Brief: Particle charges changing per Lumisection

#include <iostream>
using namespace std;

void tprofile2poly_avg()
{
   TCanvas *c1 = new TCanvas("c1", "multipads", 900, 700);

   int NUM_LS = 3;

   auto *whole = new TH2Poly();
   auto *tmp = new TProfile2Poly[NUM_LS];
   auto *t1 = new TProfile2Poly();

   t1->Honeycomb(0, 0, .1, 30, 30);

   TRandom ran;
   whole->Honeycomb(0, 0, .1, 30, 30);
   for (int i = 0; i <= NUM_LS - 1; ++i) {
      tmp[i].Honeycomb(0, 0, .1, 30, 30);
   }

   c1->Divide(3, 2);
   Double_t ii = 0;

   // Fill histograms events
   for (int i = 0; i <= NUM_LS - 1; ++i) {
      for (int j = 0; j < 1000000; ++j) {

         Double_t r1 = ran.Gaus(2.7, 1.1);
         Double_t r2 = ran.Gaus(2.25, 2);

         Double_t rok = ran.Gaus(20, 2);
         Double_t rbad1 = ran.Gaus(2, 10);
         Double_t rbad2 = ran.Gaus(-3, 10);
         Double_t rbad3 = ran.Gaus(-2, 5);

         Double_t val = rok;
         ii = double(i) * 0.5;

         if (r2 > 0.5 && r2 < 1 && r1 > 1 && r1 < 1.5 + ii) val = rok - rbad1;
         if (r2 > 1 && r2 < 1.5 && r1 > 3.2 && r1 < 3.6) val = rok - rbad2;
         if (r2 > 3.5 - ii && r2 < 4 - ii && r1 > 3 && r1 < 4) val = rok - rbad3;

         tmp[i].Fill(r1, r2, val);
         whole->Fill(r1, r2, val);
         t1->Fill(r1, r2, val);

         if (j % 80000 == 0) { // so that your computer doesn't die a horrible death by update()
            c1->cd(6);
            t1->SetStats(0);
            t1->SetTitle("running average");
            t1->Draw("COLZ");
            c1->Update();
         }
      }

      c1->cd(i + 1);
      string title = " average charge in LumiSec " + to_string(i);
      tmp[i].SetStats(0);
      tmp[i].SetTitle(title.c_str());
      tmp[i].Draw("COLZ");

      c1->Update();
   }

   c1->cd(5);
   whole->SetStats(0);
   whole->SetTitle("Hitmap");
   whole->Draw("COLZ");
}
