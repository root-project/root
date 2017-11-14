#include <ROOT/TDataFrame.hxx>
#include <ROOT/TCsvDS.hxx>
#include <ROOT/TSeq.hxx>

#include <gtest/gtest.h>

#include <iostream>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::TDF;

auto fileName0 = "TCsvDS_test_headers.csv";
auto fileName1 = "TCsvDS_test_noheaders.csv";

TEST(TCsvDS, ColTypeNames)
{
   TCsvDS tds(fileName0);
   tds.SetNSlots(1);

   auto colNames = tds.GetColumnNames();

   EXPECT_TRUE(tds.HasColumn("Name"));
   EXPECT_TRUE(tds.HasColumn("Age"));
   EXPECT_FALSE(tds.HasColumn("Address"));

   EXPECT_STREQ("Height", colNames[2].c_str());
   EXPECT_STREQ("Married", colNames[3].c_str());

   EXPECT_STREQ("std::string", tds.GetTypeName("Name").c_str());
   EXPECT_STREQ("Long64_t", tds.GetTypeName("Age").c_str());
   EXPECT_STREQ("double", tds.GetTypeName("Height").c_str());
   EXPECT_STREQ("bool", tds.GetTypeName("Married").c_str());
}

TEST(TCsvDS, ColNamesNoHeaders)
{
   TCsvDS tds(fileName1, false);
   tds.SetNSlots(1);

   auto colNames = tds.GetColumnNames();

   EXPECT_STREQ("Col0", colNames[0].c_str());
   EXPECT_STREQ("Col1", colNames[1].c_str());
   EXPECT_STREQ("Col2", colNames[2].c_str());
   EXPECT_STREQ("Col3", colNames[3].c_str());
}

TEST(TCsvDS, EntryRanges)
{
   TCsvDS tds(fileName0);
   tds.SetNSlots(3U);
   tds.Initialise();

   // Still dividing in equal parts...
   auto ranges = tds.GetEntryRanges();

   EXPECT_EQ(3U, ranges.size());
   EXPECT_EQ(0U, ranges[0].first);
   EXPECT_EQ(2U, ranges[0].second);
   EXPECT_EQ(2U, ranges[1].first);
   EXPECT_EQ(4U, ranges[1].second);
   EXPECT_EQ(4U, ranges[2].first);
   EXPECT_EQ(6U, ranges[2].second);
}

TEST(TCsvDS, ColumnReaders)
{
   TCsvDS tds(fileName0);
   const auto nSlots = 3U;
   tds.SetNSlots(nSlots);
   auto vals = tds.GetColumnReaders<Long64_t>("Age");
   tds.Initialise();
   auto ranges = tds.GetEntryRanges();
   auto slot = 0U;
   std::vector<Long64_t> ages = {60, 50, 40, 30, 1, -1};
   for (auto &&range : ranges) {
      tds.InitSlot(slot, range.first);
      for (auto i : ROOT::TSeq<int>(range.first, range.second)) {
         tds.SetEntry(slot, i);
         auto val = **vals[slot];
         EXPECT_EQ(ages[i], val);
      }
      slot++;
   }
}

TEST(TCsvDS, ColumnReadersString)
{
   TCsvDS tds(fileName0);
   const auto nSlots = 3U;
   tds.SetNSlots(nSlots);
   auto vals = tds.GetColumnReaders<std::string>("Name");
   tds.Initialise();
   auto ranges = tds.GetEntryRanges();
   auto slot = 0U;
   std::vector<std::string> names = {"Harry", "Bob,Bob", "\"Joe\"", "Tom", " John  ", " Mary Ann "};
   for (auto &&range : ranges) {
      tds.InitSlot(slot, range.first);
      for (auto i : ROOT::TSeq<int>(range.first, range.second)) {
         tds.SetEntry(slot, i);
         auto val = *((std::string *)*vals[slot]);
         EXPECT_EQ(names[i], val);
      }
      slot++;
   }
}

#ifndef NDEBUG

TEST(TCsvDS, SetNSlotsTwice)
{
   auto theTest = []() {
      TCsvDS tds(fileName0);
      tds.SetNSlots(1);
      tds.SetNSlots(1);
   };
   ASSERT_DEATH(theTest(), "Setting the number of slots even if the number of slots is different from zero.");
}
#endif

#ifdef R__B64

TEST(TCsvDS, FromATDF)
{
   std::unique_ptr<TDataSource> tds(new TCsvDS(fileName0));
   TDataFrame tdf(std::move(tds));
   auto max = tdf.Max<double>("Height");
   auto min = tdf.Min<double>("Height");
   auto c = tdf.Count();

   EXPECT_EQ(6U, *c);
   EXPECT_DOUBLE_EQ(200.5, *max);
   EXPECT_DOUBLE_EQ(.7, *min);
}

TEST(TCsvDS, FromATDFWithJitting)
{
   std::unique_ptr<TDataSource> tds(new TCsvDS(fileName0));
   TDataFrame tdf(std::move(tds));
   auto max = tdf.Filter("Age<40").Max("Age");
   auto min = tdf.Define("Age2", "Age").Filter("Age2>30").Min("Age2");

   EXPECT_EQ(30, *max);
   EXPECT_EQ(40, *min);
}

// NOW MT!-------------
#ifdef R__USE_IMT

TEST(TCsvDS, DefineSlotCheckMT)
{
   const auto nSlots = 4U;
   ROOT::EnableImplicitMT(nSlots);

   std::vector<unsigned int> ids(nSlots, 0u);
   std::unique_ptr<TDataSource> tds(new TCsvDS(fileName0));
   TDataFrame d(std::move(tds));
   auto m = d.DefineSlot("x", [&](unsigned int slot) {
                ids[slot] = 1u;
                return 1;
             }).Max("x");
   EXPECT_EQ(1, *m); // just in case

   const auto nUsedSlots = std::accumulate(ids.begin(), ids.end(), 0u);
   EXPECT_GT(nUsedSlots, 0u);
   EXPECT_LE(nUsedSlots, nSlots);
}

TEST(TCsvDS, FromATDFMT)
{
   std::unique_ptr<TDataSource> tds(new TCsvDS(fileName0));
   TDataFrame tdf(std::move(tds));
   auto max = tdf.Max<double>("Height");
   auto min = tdf.Min<double>("Height");
   auto c = tdf.Count();

   EXPECT_EQ(6U, *c);
   EXPECT_DOUBLE_EQ(200.5, *max);
   EXPECT_DOUBLE_EQ(.7, *min);
}

TEST(TCsvDS, FromATDFWithJittingMT)
{
   std::unique_ptr<TDataSource> tds(new TCsvDS(fileName0));
   TDataFrame tdf(std::move(tds));
   auto max = tdf.Filter("Age<40").Max("Age");
   auto min = tdf.Define("Age2", "Age").Filter("Age2>30").Min("Age2");

   EXPECT_EQ(30, *max);
   EXPECT_EQ(40, *min);
}

#endif // R__USE_IMT

#endif // R__B64
