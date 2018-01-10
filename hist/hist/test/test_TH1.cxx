#include "gtest/gtest.h"

#include "TH1.h"
#include "TH1F.h"

// StatOverflows TH1
TEST(TH1, StatOverflows)
{
   TH1F h0("h0", "h0", 1, 0, 1);
   TH1F h1("h1", "h1", 1, 0, 1);
   TH1F h2("h2", "h2", 1, 0, 1);
   EXPECT_EQ(TH1::EStatOverflows::kNeutral, h0.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kNeutral, h1.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kNeutral, h2.GetStatOverflows());

   h0.SetStatOverflows(TH1::EStatOverflows::kIgnore);
   h1.SetStatOverflows(TH1::EStatOverflows::kConsider);
   h2.SetStatOverflows(TH1::EStatOverflows::kNeutral);
   EXPECT_EQ(TH1::EStatOverflows::kIgnore,   h0.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kConsider, h1.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kNeutral,  h2.GetStatOverflows());

   TH1::StatOverflows(true);
   EXPECT_EQ(TH1::EStatOverflows::kIgnore,   h0.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kConsider, h1.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kNeutral,  h2.GetStatOverflows());

   TH1::StatOverflows(false);
   EXPECT_EQ(TH1::EStatOverflows::kIgnore,   h0.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kConsider, h1.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kNeutral,  h2.GetStatOverflows());
}
