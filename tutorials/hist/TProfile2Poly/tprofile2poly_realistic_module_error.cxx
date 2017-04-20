/*
TProfile2Poly can do all the things that TProfile2D does
in addition to handling averages over polygonal shapes, demostrated
by displaying it on top of one of CMS' pixel forward barrels.

The use case of such a plot would be to be able to diagnose if
a detector part is broken and is measuring a significantly
different value on average then the others.
*/

#include <iostream>
#include <fstream>

using namespace std;
void tprofile2poly_realistic_module_error(Int_t numEvents=1000000){

  TCanvas* c1 = new TCanvas("c1","multipads",900,700);
  c1->Divide(2,1);

  // -------------------- Construct detector bins ------------------------
  auto h2p  = new TH2Poly();
  auto tp2p = new TProfile2Poly();
  ifstream infile;
  infile.open("./tutorials/hist/TProfile2Poly/test_data/cms_forward3");

  vector< pair<Double_t,Double_t>> allCoords;
  Double_t a, b;
  while (infile >> a >> b){
     pair<Double_t, Double_t> coord(a,b);
     allCoords.push_back(coord);
  }

  if(allCoords.size() % 3 != 0){
     cout << "[ERROR] Bad file" <<  endl;
    return;
  }

  Double_t x[3], y[3];
  for(Int_t i=0; i<allCoords.size(); i+=3){
    x[0] = allCoords[i+0].first;  y[0] = allCoords[i+0].second;
    x[1] = allCoords[i+1].first;  y[1] = allCoords[i+1].second;
    x[2] = allCoords[i+2].first;  y[2] = allCoords[i+2].second;
    h2p->AddBin(3, x, y);
    tp2p->AddBin(3, x, y);
  }

  // -------------------- Generate particles ------------------------
  TRandom ran;
  for(int j = 0; j < numEvents; ++j) {
    Double_t r1 = ran.Gaus(0,10);
    Double_t r2 = ran.Gaus(0,8);
    Double_t rok   = ran.Gaus(20,2);
    Double_t rbad1 = ran.Gaus(1,2);
    Double_t rbad2 = ran.Gaus(2,1);

    Double_t val = rok;
    // --------------------  Malfunctioning panels -------------------
    if(h2p->IsInsideBin(4,r1,r2))
      val = rok - rbad1;
    if(h2p->IsInsideBin(20,r1,r2))
      val = rok - rbad2;
    if(h2p->IsInsideBin(13,r1,r2))
      val = rok + rbad1;
    if(h2p->IsInsideBin(37,r1,r2))
      val = rok + rbad2;

    // -------------------- Fill histograms ------------------------
    h2p->Fill(r1,r2,val);
    tp2p->Fill(r1,r2,val);
  }

  // -------------------- Display end state ------------------------
  c1->cd(1);
  h2p->SetStats(0);
  h2p->SetTitle("total hits");
  h2p->Draw("COLZ");

  c1->cd(2);
  tp2p->SetStats(0);
  tp2p->SetTitle("average charge - 4 malfunctioning panels");
  tp2p->Draw("COLZ");
}
