#include "ROOT/TTrivialDS.hxx"

#include "gtest/gtest.h"

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::TDF;

TEST(TTrivialDS, ColTypeNames)
{
   TTrivialDS tds(32);
   tds.SetNSlots(1);

   auto colName = tds.GetColumnNames()[0]; // We know it's one.
   EXPECT_STREQ("col0", colName.c_str());
   EXPECT_STREQ("ULong64_t", tds.GetTypeName("col0").c_str());

   EXPECT_TRUE(tds.HasColumn("col0"));
   EXPECT_FALSE(tds.HasColumn("col1"));
}

TEST(TTrivialDS, EntryRanges)
{
   TTrivialDS tds(32);
   const auto nSlots = 4U;
   tds.SetNSlots(nSlots);

   auto ranges = tds.GetEntryRanges();

   EXPECT_EQ(4U, ranges.size());
   EXPECT_EQ(0U, ranges[0].first);
   EXPECT_EQ(8U, ranges[0].second);
   EXPECT_EQ(8U, ranges[1].first);
   EXPECT_EQ(16U, ranges[1].second);
   EXPECT_EQ(16U, ranges[2].first);
   EXPECT_EQ(24U, ranges[2].second);
   EXPECT_EQ(24U, ranges[3].first);
   EXPECT_EQ(32U, ranges[3].second);
}

TEST(TTrivialDS, ColumnReaders)
{
   TTrivialDS tds(32);
   const auto nSlots = 4U;
   tds.SetNSlots(nSlots);
   auto vals = tds.GetColumnReaders<ULong64_t>("col0");
   auto ranges = tds.GetEntryRanges();
   auto slot = 0U;
   for (auto &&range : ranges) {
      for (auto i : ROOT::TSeq<ULong64_t>(range.first, range.second)) {
         tds.SetEntry(i, slot);
         auto val = **vals[slot];
         EXPECT_EQ(i, val);
      }
      slot++;
   }
}

TEST(TTrivialDS, SetNSlotsTwice)
{
   auto theTest = []() {
      TTrivialDS tds(1);
      tds.SetNSlots(1);
      tds.SetNSlots(1);
   };
   ASSERT_DEATH(theTest(), "Setting the number of slots even if the number of slots is different from zero.");
}

TEST(TTrivialDS, FromATDF)
{
   std::unique_ptr<TDataSource> tds(new TTrivialDS(32));
   TDataFrame tdf(std::move(tds));
   auto max = tdf.Max<ULong64_t>("col0");
   auto min = tdf.Min<ULong64_t>("col0");
   auto c = tdf.Count();

   EXPECT_EQ(32U, *c);
   EXPECT_DOUBLE_EQ(31., *max);
   EXPECT_DOUBLE_EQ(0., *min);
}
