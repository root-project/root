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

TEST(TProfile3D, BufferEmptyFindsAutomaticZLimits)
{
   TProfile3D profile("profile", "", 4, 0, 4, 5, 0, 10, 3, 0, 0);
   for (double z : {101., 115., 139.})
      profile.Fill(0.5, 1., z, 2.);
   profile.BufferEmpty();

   EXPECT_EQ(4, profile.GetNbinsX());
   EXPECT_DOUBLE_EQ(0., profile.GetXaxis()->GetXmin());
   EXPECT_DOUBLE_EQ(4., profile.GetXaxis()->GetXmax());
   EXPECT_EQ(5, profile.GetNbinsY());
   EXPECT_DOUBLE_EQ(0., profile.GetYaxis()->GetXmin());
   EXPECT_DOUBLE_EQ(10., profile.GetYaxis()->GetXmax());

   const auto *zaxis = profile.GetZaxis();
   EXPECT_LT(zaxis->GetXmin(), zaxis->GetXmax());
   for (double z : {101., 139.}) {
      EXPECT_GE(zaxis->FindFixBin(z), 1);
      EXPECT_LE(zaxis->FindFixBin(z), zaxis->GetNbins());
   }
   for (int zbin : {0, zaxis->GetNbins() + 1})
      for (int xbin = 0; xbin <= profile.GetNbinsX() + 1; ++xbin)
         for (int ybin = 0; ybin <= profile.GetNbinsY() + 1; ++ybin)
            EXPECT_DOUBLE_EQ(0., profile.GetBinEntries(profile.GetBin(xbin, ybin, zbin)));
   EXPECT_NEAR((101. + 115. + 139.) / 3., profile.GetMean(3), 1e-9);
}

TEST(TProfile3D, BufferEmptyPreservesFixedYAndZLimits)
{
   TProfile3D profile("profile", "", 4, 0, 0, 5, 0, 10, 3, 0, 30);
   for (double x : {101., 115., 139.})
      profile.Fill(x, 1., 5., 2.);
   profile.BufferEmpty();

   EXPECT_LT(profile.GetXaxis()->GetXmin(), profile.GetXaxis()->GetXmax());
   EXPECT_EQ(5, profile.GetNbinsY());
   EXPECT_DOUBLE_EQ(0., profile.GetYaxis()->GetXmin());
   EXPECT_DOUBLE_EQ(10., profile.GetYaxis()->GetXmax());
   EXPECT_EQ(3, profile.GetNbinsZ());
   EXPECT_DOUBLE_EQ(0., profile.GetZaxis()->GetXmin());
   EXPECT_DOUBLE_EQ(30., profile.GetZaxis()->GetXmax());
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