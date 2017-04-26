// Brief: Simulate a TProfile2D with TProfile2Poly, and see if results are the same

#include <algorithm>

void tprofile2polyCorrectness()
{

  bool ret = true;

  auto TP2D = new TProfile2D("NoName", "NoTitle", 10, -10, 10, 10, -10, 10, 0, 0);
  auto TP2P = new TProfile2Poly();

  float minx = -10;
  float maxx = 10;
  float miny = -10;
  float maxy = 10;
  float binsz = 2;

  for (float i = minx; i < maxx; i += binsz) {
     for (float j = miny; j < maxy; j += binsz) {
        TP2P->AddBin(i, j, i + binsz, j + binsz);
     }
  }

  Double_t value, weight;

  Float_t px, py;
  TRandom ran;
  for (Int_t i = 0; i < 1000000; i++) {
     gRandom->Rannor(px, py);
     value = ran.Gaus(20, 5);
     weight = ran.Gaus(17,20);

     TP2D->Fill(px, py, value, weight);
     TP2P->Fill(px, py, value, weight);
  }

  // ------------ COMPARE IF THEY YIELD THE SAME RESULTS ---------------
  std::set<Double_t> v1;
  std::set<Double_t> v2;

  Int_t EPSILON = 1000;
  for (Int_t i=0; i<TP2P->GetNumberOfBins(); ++i){
     v1.insert((Int_t)(TP2P->GetBinContent(i)*EPSILON));
  }
  for (Int_t i=0; i<TP2D->GetNumBins(); ++i){
     v2.insert((Int_t)(TP2D->GetBinContent(i)*EPSILON));
  }

  if (v1==v2){
     std::cout << "[ OK ]";
  } else {
    std::cout << "[FAIL]";
  }

  std::cout << "\t Average with Value, Weight" << std::endl;

  ret &= v1==v2;
  // --------------- MERGE TEST
  auto TP2D_2 = new TProfile2D("NoName2", "NoTitle2", 10, -10, 10, 10, -10, 10, 0, 0);
  auto TP2P_2 = new TProfile2Poly();

  for (float i = minx; i < maxx; i += binsz) {
     for (float j = miny; j < maxy; j += binsz) {
        TP2P_2->AddBin(i, j, i + binsz, j + binsz);
     }
  }

  for (Int_t i = 0; i < 1000000; i++) {
     gRandom->Rannor(px, py);
     value = ran.Gaus(20, 5);
     weight = ran.Gaus(17,20);

     TP2D_2->Fill(px+4, py+1, value, weight);
     TP2P_2->Fill(px+4, py+1, value, weight);
  }

  TList DMerge;
  TList PMerge;

  DMerge.Add(TP2D);
  PMerge.Add(TP2P);

  TP2D_2->Merge(&DMerge);
  TP2P_2->Merge(&PMerge);

  // ------------ COMPARE IF THEY YIELD THE SAME RESULTS ---------------
  v1.clear();
  v2.clear();

  for (Int_t i=0; i<TP2P_2->GetNumberOfBins(); ++i){
     v1.insert((Int_t)(TP2P_2->GetBinContent(i)*EPSILON));
  }
  for (Int_t i=0; i<TP2D_2->GetNumBins(); ++i){
     v2.insert((Int_t)(TP2D_2->GetBinContent(i)*EPSILON));
  }

  if (v1==v2){
     std::cout << "[ OK ]";
  } else {
    std::cout << "[FAIL]";
  }

  std::cout << "\t Merge Identical" << std::endl;
  ret &= v1==v2;

  return ret;
}
