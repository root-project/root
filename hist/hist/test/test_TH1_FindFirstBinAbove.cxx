// test TH1::FindFirstBinAbove abd TH1::FindLastBinAbove

#include "gtest/gtest.h"

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TRandomGen.h"

TH1 * h1 = nullptr;
TH1 * h2 = nullptr;
TH3 * h3 = nullptr;

int nentries = 1000000;
double th_value = 0.1*nentries; 

void CreateHist() {

   h3 = new TH3D("h3","h3",20,-5,5,20,-5,5,20,-5,5);

   TRandomMixMax r;
   for (int i = 0; i < nentries; ++i)  {
      double x = r.Gaus(0.,2);
      double y = r.Gaus(-3,1.5);
      double z = r.Gaus(2.,2.5);
      h3->Fill(x,y,z);
   }
   h3->SetBinContent(4,5,6, 2.*th_value);
   h3->SetBinContent(14,15,16, 1.5*th_value);

   h2 = h3->Project3D("ZY");  // Y will be the x axis of h2
   h1 = h3->Project3D("X");

}

TEST(TH3, FindFirstBinAbove )
{
   if (!h3) CreateHist();

   EXPECT_EQ( 4, h3->FindFirstBinAbove(th_value, 1, 1,10) );
   EXPECT_EQ( 14, h3->FindFirstBinAbove(th_value, 1, 11,20) );
   EXPECT_EQ( 5, h3->FindFirstBinAbove(th_value, 2) );
   EXPECT_EQ( 16, h3->FindFirstBinAbove(th_value, 3, 11,20) );

}

TEST(TH3, FindLastBinAbove )
{
   if (!h3) CreateHist();

   EXPECT_EQ( 14, h3->FindLastBinAbove(th_value, 1) );
   EXPECT_EQ( 5, h3->FindLastBinAbove(th_value, 2,1,10) );
   EXPECT_EQ( 16, h3->FindLastBinAbove(th_value, 3) );
   EXPECT_EQ( 6, h3->FindLastBinAbove(th_value, 3, 1,10) );

}

TEST(TH2, FindFirstBinAbove )
{
   if (!h2) CreateHist();

   EXPECT_EQ( 5, h2->FindFirstBinAbove(th_value, 1) );
   EXPECT_EQ( 15, h2->FindFirstBinAbove(th_value, 1,13,18) );
   EXPECT_EQ( 6, h2->FindFirstBinAbove(th_value, 2, 1,10) );
   EXPECT_EQ( 16, h2->FindFirstBinAbove(th_value, 2, 11,16) );
}

TEST(TH2, FindLastBinAbove )
{
   if (!h2) CreateHist();

   EXPECT_EQ( 5, h2->FindLastBinAbove(th_value, 1,3,8) );
   EXPECT_EQ( 15, h2->FindLastBinAbove(th_value, 1) );
   EXPECT_EQ( 6, h2->FindLastBinAbove(th_value, 2, 1,10) );
   EXPECT_EQ( 16, h2->FindLastBinAbove(th_value, 2) );
}

TEST(TH1, FindFirstBinAbove )
{
   if (!h1) CreateHist();

   EXPECT_EQ( 4, h1->FindFirstBinAbove(th_value, 1, 1, 20) );
   EXPECT_EQ( 14, h1->FindFirstBinAbove(th_value, 1,14,16) );
}

TEST(TH1, FindLastBinAbove )
{
   if (!h1) CreateHist();

   EXPECT_EQ( 4, h1->FindLastBinAbove(th_value, 1,3,8) );
   EXPECT_EQ( 14, h1->FindLastBinAbove(th_value, 1, 0, 30) );
}

