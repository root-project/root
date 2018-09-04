#include <RooRealVar.h>

#include <sstream>

#include "gtest/gtest.h"


// Tests ROOT-6378
TEST(RooRealVar, PrintDefaultConstructed)
{
   std::stringstream s;
   RooRealVar v;
   v.printStream(s, v.defaultPrintContents(""),v.defaultPrintStyle(""));

   auto resString = s.str();
   auto separatorPos = resString.find("+/-");
   auto resConstChar = resString.c_str();
   auto resCutConstChar = resConstChar + separatorPos;

   EXPECT_TRUE(0 == strncmp(resCutConstChar, "+/- (0,0)  L(", 12));
}
