#include "gtest/gtest.h"
#include "ROOT/THist.hxx"
#include "ROOT/TCanvas.hxx"
#include "ROOT/TColor.hxx"
#include "ROOT/TFile.hxx"

#include <TApplication.h>

using namespace ROOT::Experimental;

TApplication theApp("iotest", 0, 0);

// Test drawing of histograms.
TEST(IOTest, OneD)
{
TClass::GetClass("ROOT::Experimental::Detail::THistImpl<ROOT::Experimental::Detail::THistData<1,double,ROOT::Experimental::Detail::THistDataDefaultStorage,ROOT::Experimental::THistStatContent,ROOT::Experimental::THistStatUncertainty>,ROOT::Experimental::TAxisEquidistant>")->GetClassInfo();

   TAxisConfig xaxis{10, 0., 1.};
   TH1D h(xaxis);
   auto file = TFile::Recreate("IOTestOneD.root");
   file->Write("h", h);
}

// Drawing options:
TEST(IOTest, OneDOpts)
{
   TAxisConfig xaxis{10, 0., 1.};
   auto h = std::make_unique<TH1D>(xaxis);
   TCanvas canv;
   auto &Opts = canv.Draw(std::move(h));
   Opts.SetLineColor(TColor::kRed);

   auto file = TFile::Recreate("IOTestOneDOpts.root");
   file->Write("canv", canv);
}
