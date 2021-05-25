// test TH2Poly setting and retrieving bin error 

#include "gtest/gtest.h"

#include "TH2Poly.h"
#include "TRandom.h"

TH2Poly * CreateHist() {

   TH2Poly *h2p = new TH2Poly();
   Double_t x1[] = {0, 5, 6};
   Double_t y1[] = {0, 0, 5};
   Double_t x2[] = {0, -1, -1, 0};
   Double_t y2[] = {0, 0, -1, 3};
   Double_t x3[] = {4, 3, 0, 1, 2.4};
   Double_t y3[] = {4, 3.7, 1, 3.7, 2.5};
   h2p->AddBin(3, x1, y1);
   h2p->AddBin(4, x2, y2);
   h2p->AddBin(5, x3, y3);

   return h2p;
}


TEST(TH2Poly, BinErrorUnweighted)
{

   auto h2p = CreateHist();
   // fill bin1 
   for (int i = 0; i < 9; ++i) h2p->Fill(0.1,0.01); 
   EXPECT_EQ( 3, h2p->GetBinError(1) );

   // fill sea bins
   h2p->Fill(-0.5,-0.5,1); 
   h2p->Fill(2,-0.5,1); 
   EXPECT_EQ( sqrt(2), h2p->GetBinError(-5) );

   // fill bin 3
   h2p->Fill(1,3);
   h2p->Fill(1.5,3);
   h2p->Fill(2.5,3);
   EXPECT_EQ( sqrt(3), h2p->GetBinError(3) );

   // fill overflow bin
   h2p->Fill(7,3); 
   EXPECT_EQ(1, h2p->GetBinError(-6) );
   h2p->Fill(-2,10); 
   EXPECT_EQ( 1, h2p->GetBinError(-1) );

   h2p->SetBinContent(1,10); 
   EXPECT_EQ( sqrt(10), h2p->GetBinError(1) );

   h2p->SetBinContent(-3,4); 
   EXPECT_EQ( 2, h2p->GetBinError(-3) );

   EXPECT_EQ( 0, h2p->GetBinError(2) );
   EXPECT_EQ( 0, h2p->GetBinError(-2) );
   EXPECT_EQ( 0, h2p->GetBinError(-4) );
   EXPECT_EQ( 0, h2p->GetBinError(-7) );
   EXPECT_EQ( 0, h2p->GetBinError(-8) );
   EXPECT_EQ( 0, h2p->GetBinError(-9) );
}

TEST(TH2Poly,BinErrorWeighted)
{
   auto h2p = CreateHist();

   // fill bins
   double w2 = 0; 
   for (int i = 0; i < 10; ++i) {
      double w = gRandom->Uniform(0,10); 
      h2p->Fill(0.1,0.01,w);
      w2 += w*w;
   }
   EXPECT_EQ(sqrt(w2), h2p->GetBinError(1) );

   // fill sea bins
   h2p->Fill(-0.5,-0.5,4); 
   h2p->Fill(2,-0.5,3); 
   EXPECT_EQ( 5 , h2p->GetBinError(-5) );

   // fill bin 3
   h2p->Fill(1,3,2);
   h2p->Fill(1.5,3,2);
   h2p->Fill(2.5,3,1);
   EXPECT_EQ( 3, h2p->GetBinError(3) );

   // fill overflow bin
   h2p->Fill(7,3,2); 
   EXPECT_EQ(2, h2p->GetBinError(-6) );
   h2p->Fill(-2,10,3); 
   EXPECT_EQ( 3, h2p->GetBinError(-1) );

   EXPECT_EQ( 0, h2p->GetBinError(2) );
   EXPECT_EQ( 0, h2p->GetBinError(-2) );
   EXPECT_EQ( 0, h2p->GetBinError(-3) );
   EXPECT_EQ( 0, h2p->GetBinError(-4) );
   EXPECT_EQ( 0, h2p->GetBinError(-7) );
   EXPECT_EQ( 0, h2p->GetBinError(-8) );
   EXPECT_EQ( 0, h2p->GetBinError(-9) );

}

TEST(TH2, SetBinError)
{
   auto h2p = CreateHist();

   h2p->SetBinContent(1, 1.0);
   h2p->SetBinContent(2, 2.0);
   h2p->SetBinContent(3, 3.0);
   h2p->SetBinContent(-9, 4.0);

   EXPECT_EQ( 1, h2p->GetBinError(1) );
   EXPECT_EQ( sqrt(2), h2p->GetBinError(2) );
   EXPECT_EQ( sqrt(3), h2p->GetBinError(3) );
   EXPECT_EQ( 2, h2p->GetBinError(-9) );

   h2p->SetBinError(1, 1.5);
   h2p->SetBinError(2, 2.5);
   h2p->SetBinError(3, 3.5);
   h2p->SetBinError(-8, 1);


   EXPECT_EQ( 1.5, h2p->GetBinError(1) );
   EXPECT_EQ( 2.5, h2p->GetBinError(2) );
   EXPECT_EQ( 3.5, h2p->GetBinError(3) );
   EXPECT_EQ( 1, h2p->GetBinError(-8) );

   // setting a new content does not set bin error
   h2p->SetBinContent(-1,3); 
   EXPECT_EQ( 0, h2p->GetBinError(-1) );


}


