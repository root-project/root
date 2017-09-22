#include <ROOT/TDataFrame.hxx>
#include <ROOT/TTrivialDS.hxx>
#include <ROOT/TSeq.hxx>

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
         tds.SetEntry(slot, i);
         auto val = **vals[slot];
         EXPECT_EQ(i, val);
      }
      slot++;
   }
}

#ifndef NDEBUG
TEST(TTrivialDS, SetNSlotsTwice)
{
   auto theTest = []() {
      TTrivialDS tds(1);
      tds.SetNSlots(1);
      tds.SetNSlots(1);
   };
   ASSERT_DEATH(theTest(), "Setting the number of slots even if the number of slots is different from zero.");
}
#endif

TEST(TTrivialDS, FromATDF)
{
   std::unique_ptr<TDataSource> tds(new TTrivialDS(32));
   TDataFrame tdf(std::move(tds));
   auto max = tdf.Max<ULong64_t>("col0");
   auto min = tdf.Min<ULong64_t>("col0");
   auto c = tdf.Count();
   auto max2 = tdf.Filter([](ULong64_t col0) { return col0 < 10; }, {"col0"}).Max<ULong64_t>("col0");
   auto min2 = tdf.Filter([](ULong64_t col0) { return col0 > 10; }, {"col0"})
                  .Define("j", [](ULong64_t col0) { return col0 * 2; }, {"col0"})
                  .Min<ULong64_t>("j");

   EXPECT_EQ(32U, *c);
   EXPECT_DOUBLE_EQ(31., *max);
   EXPECT_DOUBLE_EQ(0., *min);
   EXPECT_DOUBLE_EQ(9., *max2);
   EXPECT_DOUBLE_EQ(22., *min2);
}

TEST(TTrivialDS, FromATDFWithJitting)
{
   std::unique_ptr<TDataSource> tds(new TTrivialDS(32));
   TDataFrame tdf(std::move(tds));
   auto max = tdf.Filter("col0 < 10").Max("col0");
   auto min = tdf.Filter("col0 > 10").Define("j", "col0*2").Min("j");

   EXPECT_DOUBLE_EQ(9., *max);
   EXPECT_DOUBLE_EQ(22., *min);
}

// NOW MT!-------------

#ifdef R__USE_IMT

TEST(TTrivialDS, DefineSlotCheckMT)
{
   auto nSlots = 4U;
   ROOT::EnableImplicitMT(nSlots);

   std::hash<std::thread::id> hasher;
   using H_t = decltype(hasher(std::this_thread::get_id()));

   std::vector<H_t> ids(nSlots, 0);
   std::unique_ptr<TDataSource> tds(new TTrivialDS(nSlots));
   TDataFrame d(std::move(tds));
   auto m = d.DefineSlot("x", [&](unsigned int slot) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                ids[slot] = hasher(std::this_thread::get_id());
                return 1.;
             }).Max("x");

   EXPECT_EQ(1, *m); // just in case

   std::set<H_t> s(ids.begin(), ids.end());
   EXPECT_EQ(nSlots, s.size());
   EXPECT_TRUE(s.end() == s.find(0));
}

TEST(TTrivialDS, FromATDFMT)
{
   std::unique_ptr<TDataSource> tds(new TTrivialDS(320));
   TDataFrame tdf(std::move(tds));
   auto max = tdf.Max<ULong64_t>("col0");
   auto min = tdf.Min<ULong64_t>("col0");
   auto c = tdf.Count();
   auto max2 = tdf.Filter([](ULong64_t col0) { return col0 < 10; }, {"col0"}).Max<ULong64_t>("col0");
   auto min2 = tdf.Filter([](ULong64_t col0) { return col0 > 10; }, {"col0"})
                  .Define("j", [](ULong64_t col0) { return col0 * 2; }, {"col0"})
                  .Min<ULong64_t>("j");

   EXPECT_EQ(320U, *c);
   EXPECT_DOUBLE_EQ(319., *max);
   EXPECT_DOUBLE_EQ(0., *min);
   EXPECT_DOUBLE_EQ(9., *max2);
   EXPECT_DOUBLE_EQ(22., *min2);
}

TEST(TTrivialDS, FromATDFWithJittingMT)
{
   std::unique_ptr<TDataSource> tds(new TTrivialDS(320));
   TDataFrame tdf(std::move(tds));
   auto max = tdf.Filter("col0 < 10").Max("col0");
   auto min = tdf.Filter("col0 > 10").Define("j", "col0*2").Min("j");

   EXPECT_DOUBLE_EQ(9., *max);
   EXPECT_DOUBLE_EQ(22., *min);
}

#endif

