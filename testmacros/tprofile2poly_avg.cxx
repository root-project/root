#include <iostream>
                        // x,     , y      , weight
using Event = std::tuple<Double_t, Double_t, Double_t>;

void tprofile2poly_avg() {
  TCanvas* c1 = new TCanvas("c1","multipads",900,700);

  int NUM_LS = 3;

  TH2Poly* whole = new TH2Poly();
  TProfile2Poly* tmp = new TProfile2Poly[NUM_LS];
  TProfile2Poly* t1 = new TProfile2Poly();

  t1->Honeycomb(0,0,.1,30,30);

  TRandom ran;
  whole->Honeycomb(0,0,.1,30,30);
  for (int i = 0; i <= NUM_LS-1; ++i) {
    tmp[i].Honeycomb(0,0,.1,30,30);
  }

  std::vector<Event>                           events;
  std::vector<std::vector<Event>>              lumisection;
  std::vector<std::vector<std::vector<Event>>> run;

  c1->Divide(3,2);
  Double_t ii = 0;

  // Fill histograms events
  for(int i = 0; i <= NUM_LS-1; ++i) {
    lumisection.clear();
    for(int k = 0; k < 110; ++k) {
      events.clear();
      for(int j = 0; j < 1000; ++j) {
        // coordinates of hit
        Double_t r1 = ran.Gaus(2.7,1.1);
        Double_t r2 = ran.Gaus(2.25,2);

        // charge of particle
        Double_t rok   = ran.Gaus(20,2);
        // simulating faulty charge detector values
        // these
        Double_t rbad1 = ran.Gaus(2,10);
        Double_t rbad2 = ran.Gaus(-3,10);
        Double_t rbad3 = ran.Gaus(-2,5);

        Event e = std::make_tuple(r1, r2, rok);           // Normal case

        ii = double(i) * 0.5;

        // simulating faulty detector
        if(r2>0.5  && r2<1  && r1>1 && r1<1.5 + ii){    // increase area vertically
          e = std::make_tuple(r1, r2, rok - rbad1);     //   every iteration
        } else if(r2>1&& r2<1.5 && r1>3.2 && r1<3.6){
          e = std::make_tuple(r1, r2, rok - rbad2);
        } else if(r2>3.5-ii&& r2<4-ii&& r1>3 && r1<4){  // move to the right
          e = std::make_tuple(r1, r2, rok - rbad3);     // every iteration
        }

        tmp[i].Fill(std::get<0>(e), std::get<1>(e), std::get<2>(e)); // weight/charge
        whole->Fill(std::get<0>(e), std::get<1>(e)); // actual number of hits per bin
        t1->Fill(std::get<0>(e), std::get<1>(e), std::get<2>(e)); // actual number of hits per bin
        events.push_back(e);
      }
      lumisection.push_back(events);

      c1->cd(6);
      t1->SetStats(0);
      t1->SetTitle("Average w/ TProfile2Poly::FillWeighted()");
      t1->Draw("COLZ");
      c1->Update();

    }

    c1->cd(i+1);
    std::string title = " AVG.CHARGE in LumiSec " + std::to_string(i);
    tmp[i].SetStats(0);
    tmp[i].SetTitle(title.c_str());
    tmp[i].Draw("COLZ");

    c1->Update();
    run.push_back(lumisection);
  }

  c1->cd(5);
  whole->SetStats(0);
  whole->SetTitle("Hitmap");
  whole->Draw("COLZ");
}
