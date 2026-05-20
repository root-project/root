#include "gtest/gtest.h"

#include "TProfile.h"
#include "TProfile2D.h"
#include "TProfile3D.h"
#include "TRandom.h"

#include "ROOT/TestSupport.hxx"

template <typename T>
void runTest(T const &reference, T const &sameDistr, T const &differentDistr)
{
   EXPECT_EQ(reference.Chi2Test(&sameDistr), reference.Chi2Test(&sameDistr, "WW"));
   EXPECT_EQ(reference.Chi2Test(&sameDistr),
             reference.Chi2Test(&sameDistr, "P WW UW")); // Need more than just the default option
   EXPECT_EQ(reference.Chi2Test(&sameDistr), reference.Chi2Test(&sameDistr, "UW"));
   EXPECT_EQ(reference.Chi2Test(&sameDistr), reference.Chi2Test(&sameDistr, "UU"));
   EXPECT_EQ(reference.Chi2Test(&sameDistr), reference.Chi2Test(&sameDistr, "P UU"));

   const double probSuccess = reference.Chi2Test(&sameDistr, "P");
   EXPECT_GT(probSuccess, 0.1);
   EXPECT_LE(probSuccess, 1.);
   const double probFail = reference.Chi2Test(&differentDistr, "P");
   EXPECT_LT(probFail, 0.05);
}

TEST(TProfile, Chi2Test)
{
   TProfile reference("reference", "reference", 10, 0, 10);
   TProfile sameDistr("sameDistr", "sameDistr", 10, 0, 10);
   TProfile differentDistr("differentDistr", "differentDistr", 10, 0, 10);

   gRandom->SetSeed(1);
   for (unsigned int i = 0; i < 100000; i++) {
      const double x = gRandom->Uniform(10.);
      reference.Fill(x, gRandom->Gaus(5 + x / 2, 5.));
      sameDistr.Fill(x, gRandom->Gaus(5 + x / 2, 5.));
      differentDistr.Fill(x, gRandom->Gaus(20, 1.));
   }

   runTest(reference, sameDistr, differentDistr);
}

TEST(TProfile, Chi2TestWithWrongErrors)
{
   TProfile reference("reference", "reference", 10, 0, 10);
   reference.Fill(1, 2);
   reference.Fill(1, 3);

   for (auto err : {"s", "i", "g"}) {
      ROOT::TestSupport::CheckDiagsRAII checkDiag(kError, "TProfile::Chi2Test", "error of mean", false);

      TProfile sameDistr("sameDistr", "sameDistr", 10, 0, 10, err);
      sameDistr.Fill(1, 2);
      sameDistr.Fill(1, 3);

      reference.Chi2Test(&sameDistr);
   }
}

TEST(TProfile2D, Chi2Test)
{
   TProfile2D reference("reference", "reference", 10, 0, 10, 10, 0, 10);
   TProfile2D sameDistr("sameDistr", "sameDistr", 10, 0, 10, 10, 0, 10);
   TProfile2D differentDistr("differentDistr", "differentDistr", 10, 0, 10, 10, 0, 10);

   gRandom->SetSeed(1);
   for (unsigned int i = 0; i < 50000; i++) {
      const double x = gRandom->Uniform(10.);
      reference.Fill(x, x + 1., gRandom->Gaus(5 + x / 2, 5.));
      sameDistr.Fill(x, x + 1., gRandom->Gaus(5 + x / 2, 5.));
      differentDistr.Fill(x, x + 1., gRandom->Gaus(20, 1.));
   }

   runTest(reference, sameDistr, differentDistr);
}

TEST(TProfile3D, Chi2Test)
{
   TProfile3D reference("reference", "reference", 10, 0, 10, 11, 0, 11, 12, 0, 12);
   TProfile3D sameDistr("sameDistr", "sameDistr", 10, 0, 10, 11, 0, 11, 12, 0, 12);
   TProfile3D differentDistr("differentDistr", "differentDistr", 10, 0, 10, 11, 0, 11, 12, 0, 12);

   gRandom->SetSeed(1);
   for (unsigned int i = 0; i < 50000; i++) {
      const double x = gRandom->Uniform(10.);
      reference.Fill(x, x + 1., x + 2., gRandom->Gaus(5 + x / 2, 5.));
      sameDistr.Fill(x, x + 1., x + 2., gRandom->Gaus(5 + x / 2, 5.));
      differentDistr.Fill(x, x + 1., x + 2., gRandom->Gaus(20, 1.));
   }

   runTest(reference, sameDistr, differentDistr);
}