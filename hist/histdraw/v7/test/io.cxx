#include "gtest/gtest.h"
#include "ROOT/RHist.hxx"
#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RFile.hxx"

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
   auto drawable = canv.Draw<RHistDrawable<1>>(h);
   drawable->AttrLine().SetColor(RColor::kRed);

   auto file = RFile::Recreate("IOTestOneDOpts.root");
   file->Write("canv", canv);
}
