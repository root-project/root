// Tests for the HistFactory
// Authors: Stephan Hageboeck, CERN  01/2019

#include "RooStats/HistFactory/Sample.h"
#include "gtest/gtest.h"

using namespace RooStats;
using namespace RooStats::HistFactory;

TEST(Sample, CopyAssignment)
{
  Sample s("s");
  {
    Sample s1("s1");
    auto hist1 = new TH1D("hist1", "hist1", 10, 0, 10);
    s1.SetHisto(hist1);
    s = s1;
    //Now go out of scope. Should delete hist1, that's owned by s1.
  }
  
  auto hist = s.GetHisto();
  ASSERT_EQ(hist->GetNbinsX(), 10);
}

