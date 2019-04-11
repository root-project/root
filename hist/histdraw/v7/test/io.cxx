#include "gtest/gtest.h"
#include "ROOT/RHist.hxx"
#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/TFile.hxx"

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
   auto file = TFile::Recreate("IOTestOneD.root");
   file->Write("h", h);
}

// Drawing options:
TEST(IOTest, OneDOpts)
{
   RAxisConfig xaxis{10, 0., 1.};
   auto h = std::make_unique<RH1D>(xaxis);
   RCanvas canv;
   auto optsPtr = canv.Draw(std::move(h));
   optsPtr->Line().SetColor(RColor::kRed);

   auto file = TFile::Recreate("IOTestOneDOpts.root");
   file->Write("canv", canv);
}
