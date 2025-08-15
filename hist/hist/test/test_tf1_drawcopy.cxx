#include "TF1.h"
#include "TAxis.h"

#include "gtest/gtest.h"

// issue #13122
TEST(TF1, DrawCopy)
{
   auto f = new TF1("f", "x*x", -3, 3);
   f->GetXaxis()->SetTitle("xtitle");
   auto fcopy = f->DrawCopy();
   EXPECT_STREQ("xtitle", fcopy->GetXaxis()->GetTitle());
}