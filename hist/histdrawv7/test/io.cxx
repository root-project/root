#include "gtest/gtest.h"
#include "ROOT/RHist.hxx"
#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RFile.hxx"

#include "TFile.h"



#include <TApplication.h>

using namespace ROOT::Experimental;

int myArgc = 2;
const char* myArgv[] = {"app", "-b", nullptr};
TApplication theApp("iotest", &myArgc, const_cast<char**>(myArgv));

// Test drawing of histograms.
TEST(IOTest, OneD)
{
   TClass::GetClass("ROOT::Experimental::Detail::RHistImpl<ROOT::Experimental::Detail::RHistData<1,double,std::vector<double>,ROOT::Experimental::RHistStatContent,ROOT::Experimental::RHistStatUncertainty>,ROOT::Experimental::RAxisEquidistant>")->GetClassInfo();

   RAxisConfig xaxis{10, 0., 1.};
   RH1D h(xaxis);
   auto file = RFile::Recreate("IOTestOneD.root");
   file->Write("h", h);
}

// Drawing options:
TEST(IOTest, OneDOpts)
{
   RAxisConfig xaxis{10, 0., 1.};
   auto h = std::make_shared<RH1D>(xaxis);
   RCanvas canv;
   auto drawable1 = canv.Draw(h);
   drawable1->AttrLine().SetColor(RColor::kRed);
   auto drawable2 = canv.Draw(h);
   drawable2->AttrLine().SetColor(RColor::kBlue);

   EXPECT_EQ(canv.NumPrimitives(), 2u);
   EXPECT_NE(canv.GetPrimitive(0).get(), canv.GetPrimitive(1).get());

   auto pr1 = std::dynamic_pointer_cast<RHistDrawable<1>>(canv.GetPrimitive(0));
   auto pr2 = std::dynamic_pointer_cast<RHistDrawable<1>>(canv.GetPrimitive(1));
   EXPECT_NE(pr1.get(), nullptr);
   EXPECT_NE(pr2.get(), nullptr);
   if (pr1 && pr2) {
      EXPECT_NE(pr1->GetHist().get(), nullptr);
      EXPECT_NE(pr2->GetHist().get(), nullptr);
      EXPECT_EQ(pr1->GetHist().get(), pr2->GetHist().get());
   }

   {
      auto file = RFile::Recreate("IOTestOneDOpts.root");
      file->Write("canv", canv);
      file->Close();
   }

   {
      auto file2 = TFile::Open("IOTestOneDOpts.root");
      auto canv2 = file2->Get<ROOT::Experimental::RCanvas>("canv");
      EXPECT_NE(canv2, nullptr);

      if(canv2) {
         EXPECT_EQ(canv2->NumPrimitives(), 2u);
         EXPECT_NE(canv2->GetPrimitive(0).get(), canv2->GetPrimitive(1).get());
         auto dr1 = std::dynamic_pointer_cast<RHistDrawable<1>>(canv2->GetPrimitive(0));
         auto dr2 = std::dynamic_pointer_cast<RHistDrawable<1>>(canv2->GetPrimitive(1));
         EXPECT_NE(dr1.get(), nullptr);
         EXPECT_NE(dr2.get(), nullptr);
         if (dr1 && dr2) {
            EXPECT_NE(dr1->GetHist().get(), nullptr);
            EXPECT_NE(dr2->GetHist().get(), nullptr);
            EXPECT_EQ(dr1->GetHist().get(), dr2->GetHist().get());
         }
      }

      delete canv2;
      delete file2;
   }


}
