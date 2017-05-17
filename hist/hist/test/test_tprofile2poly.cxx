
// #include "gtest/gtest.h"
#include <algorithm>

void compareStats2DtoPoly(std::string s, TProfile2D* tp2d, TProfile2Poly* tpp) {
  cout << "----- " << s << endl;
  cout << "Mean | MeanError | StdDev" << endl << endl;
  for(Int_t c=1; c<=3; ++c) {
    cout << "Along Axis " << c << endl;
    cout << "2D:\t"   << tp2d->GetMean(c) << "\t" << tp2d->GetMeanError(c) << "\t" << tp2d->GetStdDev(c) << endl;
    cout << "Poly:\t" << tpp->GetMean(c) << "\t" << tpp->GetMeanError(c) << "\t" << tpp->GetStdDev(c) << endl << endl;
    EXPECT_DOUBLE_EQ(tp2d->GetMean(c), tpp->GetMean(c));
    EXPECT_DOUBLE_EQ(tp2d->GetMeanError(c), tpp->GetMeanError(c));
    EXPECT_DOUBLE_EQ(tp2d->GetStdDev(c), tpp->GetStdDev(c));
  }
}

void FillForTest(TProfile2D* tp2d, TProfile2Poly* tpp, TRandom& ran) {
  Double_t value, weight;
  Float_t px, py;

  for (Int_t i = 0; i < 1000000; i++) {
     gRandom->Rannor(px, py);
     value = ran.Gaus(20, 5);
     weight = ran.Gaus(17,20);
     tp2d->Fill(px, py, value, weight);
     tpp->Fill(px, py, value, weight);
  }
}

void test_tprofile2poly()
{
  auto TP2D = new TProfile2D("1", "1", 10, -10, 10, 10, -10, 10, 0, 0);
  auto TP2D_2 = new TProfile2D("2", "2", 10, -10, 10, 10, -10, 10, 0, 0);

  auto TP2P = new TProfile2Poly();
  auto TP2P_2 = new TProfile2Poly();

  float minx = -10;
  float maxx = 10;
  float miny = -10;
  float maxy = 10;
  float binsz = 2;

  for (float i = minx; i < maxx; i += binsz) {
     for (float j = miny; j < maxy; j += binsz) {
        TP2P->AddBin(i, j, i + binsz, j + binsz);
        TP2P_2->AddBin(i, j, i + binsz, j + binsz);
     }
  }

  TRandom ran;

  // ----- first  comparison
  FillForTest(TP2D, TP2P, ran);
  compareStats2DtoPoly("first  comparison", TP2D, TP2P);

  // ----- second  comparison
  FillForTest(TP2D_2, TP2P_2, ran);
  compareStats2DtoPoly("second  comparison", TP2D_2, TP2P_2);

  // ----- Merging first and second one and then comparing
  TList DMerge;
  TList PMerge;

  DMerge.Add(TP2D_2);
  PMerge.Add(TP2P_2);

  TP2D->Merge(&DMerge);
  TP2P->Merge(&PMerge);

  compareStats2DtoPoly("Merge 1+2, then compare", TP2D, TP2P);
}
