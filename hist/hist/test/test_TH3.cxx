#include "gtest/gtest.h"

#include "TH3D.h"

TEST(TH3L, SetBinContent)
{
   TH3L h("", "", 1, 0, 1, 1, 0, 1, 1, 0, 1);
   // Something that does not fit into Int_t, but is exactly representable in Double_t
   static constexpr long long Large = 1LL << 42;
   h.SetBinContent(1, 1, 1, Large);
   EXPECT_EQ(h.GetBinContent(1, 1, 1), Large);
}
