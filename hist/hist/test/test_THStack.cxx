#include "gtest/gtest.h"

#include "TH1F.h"
#include "THStack.h"

// StatOverflows TH1
TEST(THStack, GetMinimumMaximum)
{
   THStack hs("hs", "stack");
   TH1F h1("h1", "h1", 5, 0., 5.);
   TH1F h2("h2", "h2", 5, 0., 5.);
   TH1F h3("h3", "h3", 5, 0., 5.);
   for (int n = 1; n <= 5; ++n) {
      Double_t cont = n < 4 ? n : 6 - n;
      h1.SetBinContent(n, cont);
      h2.SetBinContent(n, cont);
      h3.SetBinContent(n, cont);
   }

   hs.Add(&h1);
   hs.Add(&h2);
   hs.Add(&h3);

   // default with stack building
   EXPECT_EQ(hs.GetMinimum(), 3.);
   EXPECT_EQ(hs.GetMaximum(), 9.);

   // without stack
   EXPECT_EQ(hs.GetMinimum("nostack"), 1.);
   EXPECT_EQ(hs.GetMaximum("nostack"), 3.);

   // stack with errors
   EXPECT_NEAR(hs.GetMinimum("e"), 1.267949, 1e-6);
   EXPECT_NEAR(hs.GetMaximum("e"), 12., 1e-6);

   // nostack with errors
   EXPECT_NEAR(hs.GetMinimum("nostack e"), 0, 1e-6);
   EXPECT_NEAR(hs.GetMaximum("nostack e"), 4.732051, 1e-6);


   // significant error at maximum bin
   h3.SetBinError(4, 16);
   // important - mark hstack as modified
   hs.Modified();

   // stack with errors
   EXPECT_NEAR(hs.GetMinimum("e"), -10.124515, 1e-6);
   EXPECT_NEAR(hs.GetMaximum("e"), 22.124515, 1e-6);

   // nostack with errors
   EXPECT_NEAR(hs.GetMinimum("nostack e"), -14., 1e-6);
   EXPECT_NEAR(hs.GetMaximum("nostack e"), 18., 1e-6);
}
