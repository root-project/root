
#include "gtest/gtest.h"
#include "ROOT/RHist.hxx"
#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RFile.hxx"

#include "TFile.h"

using namespace ROOT::Experimental;

// Test storing of 1D histogram
TEST(IOTest, OneD)
{
   RAxisConfig xaxis{10, 0., 1.};
   RH1D h(xaxis);
   auto file = RFile::Recreate("IOTestOneD.root");
   file->Write("h", h);
}

// Test storing RCanvas with two RHistDrawable, referencing same histo
TEST(IOTest, OneDOpts)
{
   {
      RAxisConfig xaxis{10, 0., 1.};
      auto h = std::make_shared<RH1D>(xaxis);
      RCanvas canv;
      auto drawable1 = canv.Draw(h);
      drawable1->AttrLine().SetColor(RColor::kRed);
      auto drawable2 = canv.Draw(h);
      drawable2->AttrLine().SetColor(RColor::kBlue);

      EXPECT_EQ(canv.NumPrimitives(), 3u);
      EXPECT_NE(canv.GetPrimitive(0).get(), canv.GetPrimitive(1).get());

      auto pr1 = std::dynamic_pointer_cast<RHist1Drawable>(canv.GetPrimitive(1));
      auto pr2 = std::dynamic_pointer_cast<RHist1Drawable>(canv.GetPrimitive(2));
      ASSERT_NE(pr1, nullptr);
      ASSERT_NE(pr2.get(), nullptr);

      EXPECT_NE(pr1->GetHist().get(), nullptr);
      EXPECT_NE(pr2->GetHist().get(), nullptr);
      EXPECT_EQ(pr1->GetHist().get(), pr2->GetHist().get());

      auto file = RFile::Recreate("IOTestOneDOpts.root");
      file->Write("canv", canv);
      file->Close();
   }

   auto file2 = TFile::Open("IOTestOneDOpts.root");
   ASSERT_NE(file2, nullptr);
   auto canv2 = file2->Get<ROOT::Experimental::RCanvas>("canv");
   ASSERT_NE(canv2, nullptr);

   EXPECT_EQ(canv2->NumPrimitives(), 3u);
   EXPECT_NE(canv2->GetPrimitive(1).get(), canv2->GetPrimitive(2).get());

   auto dr1 = std::dynamic_pointer_cast<RHist1Drawable>(canv2->GetPrimitive(1));
   auto dr2 = std::dynamic_pointer_cast<RHist1Drawable>(canv2->GetPrimitive(2));
   ASSERT_NE(dr1, nullptr);
   ASSERT_NE(dr2, nullptr);

   EXPECT_EQ(dr1->AttrLine().GetColor(), RColor::kRed);
   EXPECT_EQ(dr2->AttrLine().GetColor(), RColor::kBlue);

   EXPECT_NE(dr1->GetHist(), nullptr);
   EXPECT_NE(dr2->GetHist(), nullptr);
   EXPECT_EQ(dr1->GetHist(), dr2->GetHist());

   delete canv2;
   delete file2;
}
