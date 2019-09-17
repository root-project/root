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
   auto drawable1 = canv.Draw<RHistDrawable<1>>(h);
   drawable1->AttrLine().SetColor(RColor::kRed);
   auto drawable2 = canv.Draw<RHistDrawable<1>>(h);
   drawable2->AttrLine().SetColor(RColor::kBlue);

   std::shared_ptr<RDrawable> shared_1;
   std::shared_ptr<RHistDrawable<1>> shared_2;

   shared_1 = shared_2;
//    shared_2 = shared_1;

   EXPECT_EQ(canv.NumPrimitives(), 2u);
   EXPECT_NE(canv.GetPrimitive(0).get(), canv.GetPrimitive(1).get());
   //EXPECT_NE(canv.GetPrimitive<RHistDrawable<1>>(0).get(), nullptr);
   //EXPECT_NE(canv.GetPrimitive<RHistDrawable<1>>(1).get(), nullptr);

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
         canv2->ResolveSharedPtrs();
         EXPECT_NE(canv2->GetPrimitive(0).get(), canv2->GetPrimitive(1).get());
         //  EXPECT_NE(canv2->GetPrimitive<RHistDrawable<1>>(0).get(), nullptr);
         //  EXPECT_NE(canv2->GetPrimitive<RHistDrawable<1>>(1).get(), nullptr);
      }

      delete canv2;
      delete file2;
   }


}
