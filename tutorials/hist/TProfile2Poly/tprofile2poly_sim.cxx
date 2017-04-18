#include <iostream>
                        // x,     , y      , weight
using Event = std::tuple<Double_t, Double_t, Double_t>;

void tprofile2poly_sim() {
  TCanvas* c1 = new TCanvas("c1","multipads",900,700);

  int NUM_LS = 3;

  TH2Poly* whole = new TH2Poly();
  TProfile2Poly* whole_avg = new TProfile2Poly();

  TH2Poly*       abso = new TH2Poly[NUM_LS];
  TProfile2Poly* avgs = new TProfile2Poly[NUM_LS];

  float minx = -4; float maxx = 4;
  float miny = -4; float maxy = 4;
  float binsz = 0.5;

  for(float i=minx; i<maxx; i+=binsz){
    for(float j=miny; j<maxy; j+=binsz){
      whole_avg->AddBin(i, j, i+binsz, j+binsz);
      whole->AddBin(i, j, i+binsz, j+binsz);
      for (int kk = 0; kk <= NUM_LS-1; ++kk) {
        avgs[kk].AddBin(i, j, i+binsz, j+binsz);
        abso[kk].AddBin(i, j, i+binsz, j+binsz);
      }
    }
  }

  TRandom ran;
  std::vector<Event>                           events;
  std::vector<std::vector<Event>>              lumisection;
  std::vector<std::vector<std::vector<Event>>> run;

  c1->Divide(3,3);
  Double_t ii = 0;

  // Fill histograms events
  for(int i = 0; i <= NUM_LS-1; ++i) {
    lumisection.clear();
    for(int k = 0; k < 50; ++k) {
      events.clear();
      for(int j = 0; j < 100; ++j) {
        // coordinates of hit
        Double_t r1 = ran.Gaus(0,2);
        Double_t r2 = ran.Gaus(0,4);

        // charge of particle
        Double_t rok   = ran.Gaus(20,2);
        Double_t rbad1 = ran.Gaus(-10,5);

        Event e = std::make_tuple(r1, r2, rok);           // Normal case
        ii = double(i) * 0.5;
        // simulating faulty detector
        if(r2>0.5  && r2<1  && r1>1 + ii && r1<1.5 + ii){    // increase area vertically
          e = std::make_tuple(r1, r2, rok - rbad1);     //   every iteration
        }

        whole->Fill(std::get<0>(e), std::get<1>(e)); // actual number of hits per bin
        whole_avg->Fill(std::get<0>(e), std::get<1>(e), std::get<2>(e)); // actual number of hits per bin
        abso[i].Fill(std::get<0>(e), std::get<1>(e)); // weight/charge
        avgs[i].Fill(std::get<0>(e), std::get<1>(e), std::get<2>(e)); // weight/charge

        events.push_back(e);
      }
      lumisection.push_back(events);

      c1->cd(8);
      whole_avg->SetStats(0);
      whole_avg->SetTitle("Running Average");
      whole_avg->Draw("COLZ TEXT");
      c1->Update();
    }

    std::string title;

    c1->cd(i+1);
    title = " avg charge in LumiSec " + std::to_string(i);
    avgs[i].SetStats(0);
    avgs[i].SetTitle(title.c_str());
    avgs[i].Draw("COLZ TEXT");
    c1->Update();

    c1->cd(i+3+1);
    title = " abs hits in LumiSec " + std::to_string(i);
    abso[i].SetStats(0);
    abso[i].SetTitle(title.c_str());
    abso[i].Draw("COLZ TEXT");
    c1->Update();

    run.push_back(lumisection);
  }

  c1->cd(9);
  whole->SetStats(0);
  whole->SetTitle("total hits");
  whole->Draw("COLZ TEXT");
}
