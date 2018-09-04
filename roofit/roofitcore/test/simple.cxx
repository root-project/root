#include <RooRealVar.h>

#include <sstream>

#include "gtest/gtest.h"


// Tests ROOT-6378
TEST(RooRealVar, PrintDefaultConstructed)
{
   std::stringstream s;
   RooRealVar v;
   v.printStream(s, v.defaultPrintContents(""),v.defaultPrintStyle(""));
   EXPECT_STREQ(s.str().c_str(), "RooRealVar:: = 0 +/- (0,0)  L(0 - 0) B(0) \n");
}
