#include <ROOT/RDataFrame.hxx>
#include <ROOT/RLazyDS.hxx>
#include <ROOT/TSeq.hxx>

#include <gtest/gtest.h>

#include <iostream>

using namespace ROOT::RDF;

TEST(RLazyDS, Constructor)
{
   ROOT::RDataFrame d(13);
   auto col0Name = "col0";
   auto col1Name = "col1";
   auto col0Init = 0.;
   auto col1Init = 1.f;
   auto col0 = d.Define(col0Name, [&col0Init]() { return col0Init += 1.; }).Take<double>(col0Name);
   auto col1 = d.Define(col1Name, [&col1Init]() { return col1Init += 1.f; }).Take<float>(col1Name);
   RLazyDS<double, float> tds({col0Name, col0}, {col1Name, col1});

   auto colNames = tds.GetColumnNames();
   EXPECT_EQ(2U, colNames.size());
   EXPECT_STREQ(colNames[0].c_str(), col0Name);
   EXPECT_STREQ(colNames[1].c_str(), col1Name);
   EXPECT_TRUE(tds.HasColumn(col0Name));
   EXPECT_STREQ(tds.GetTypeName(col0Name).c_str(), "double");
   EXPECT_TRUE(tds.HasColumn(col1Name));
   EXPECT_STREQ(tds.GetTypeName(col1Name).c_str(), "float");

   tds.SetNSlots(4);
   auto col0Readers = tds.GetColumnReaders<double>(col0Name);
   auto col1Readers = tds.GetColumnReaders<float>(col1Name);
   tds.Initialize();
   auto ranges = tds.GetEntryRanges();

   EXPECT_EQ(0UL, ranges[0].first); // 3 + 1 from the reminder is 4 entries
   EXPECT_EQ(4UL, ranges[0].second);
   EXPECT_EQ(4UL, ranges[1].first); // 3 entries
   EXPECT_EQ(7UL, ranges[1].second);
   EXPECT_EQ(7UL, ranges[2].first); // 3 entries
   EXPECT_EQ(10UL, ranges[2].second);
   EXPECT_EQ(10UL, ranges[3].first);
   EXPECT_EQ(13UL, ranges[3].second); // 3 entries

   auto slot = 0U;

   std::vector<double> col0Vals = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.};
   std::vector<float> col1Vals = {2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f};
   for (auto &&range : ranges) {
      tds.InitSlot(slot, range.first);
      for (auto i : ROOT::TSeqI(range.first, range.second)) {
         tds.SetEntry(slot, i);
         auto val0 = **col0Readers[slot];
         auto val1 = **col1Readers[slot];
         EXPECT_EQ(col0Vals[i], val0);
         EXPECT_EQ(col1Vals[i], val1);
      }
      slot++;
   }
}


TEST(RLazyDS, RangesOneSlot)
{
   ROOT::RDataFrame d(4);
   auto col0Name = "col0";
   auto col0Init = 0.;
   auto col0 = d.Define(col0Name, [&col0Init]() { return col0Init += 1.; }).Take<double>(col0Name);
   RLazyDS<double> tds({col0Name, col0});

   tds.SetNSlots(1);
   auto col0Readers = tds.GetColumnReaders<double>(col0Name);
   tds.Initialize();
   auto ranges = tds.GetEntryRanges();
   EXPECT_EQ(1U, ranges.size());
   EXPECT_EQ(0UL, ranges[0].first);
   EXPECT_EQ(4UL, ranges[0].second);
}

TEST(RLazyDS, ColSizesCheck)
{
   ROOT::RDataFrame d0(1);
   auto colName = "col";
   auto gend = []() { return 0.; };
   auto genf = []() { return 0.f; };
   auto col0 = d0.Define(colName, gend).Take<double>(colName);
   ROOT::RDataFrame d1(2);
   auto col1 = d1.Define(colName, genf).Take<float>(colName);
   RLazyDS<double, float> tds({"zero", col0}, {"one", col1});
   tds.SetNSlots(4);
   EXPECT_ANY_THROW(tds.Initialize());
}

TEST(RLazyDS, RDFSimple)
{
   ROOT::RDataFrame d(4);
   std::string col0Name = "col0";
   std::string col1Name = "col1";
   auto col0Init = 0.;
   auto col1Init = 1.f;
   auto col0 = d.Define(col0Name, [&col0Init]() { return col0Init += 1.; }).Take<double>(col0Name);
   auto col1 = d.Define(col1Name, [&col1Init]() { return col1Init += 1.f; }).Take<float>(col1Name);
   auto tdf = MakeLazyDataFrame(std::make_pair(col0Name, col0), std::make_pair(col1Name, col1));
   auto count = *tdf.Count();
   EXPECT_EQ(count, 4UL);
}

TEST(RLazyDS, FromTwoRDFs)
{
   ROOT::RDataFrame d0(4);
   ROOT::RDataFrame d1(4);
   std::string col0Name = "col0";
   std::string col1Name = "col1";
   auto col0Init = 0.;
   auto col1Init = 1.f;
   auto col0 = d0.Define(col0Name, [&col0Init]() { return col0Init += 1.; }).Take<double>(col0Name);
   auto col1 = d1.Define(col1Name, [&col1Init]() { return col1Init += 1.f; }).Take<float>(col1Name);
   auto tdf = MakeLazyDataFrame(std::make_pair(col0Name, col0), std::make_pair(col1Name, col1));
   auto count = *tdf.Count();
   EXPECT_EQ(count, 4UL);
}
