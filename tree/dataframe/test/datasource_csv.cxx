#include <ROOT/RDataFrame.hxx>
#include <ROOT/RCsvDS.hxx>
#include <ROOT/TSeq.hxx>
#include <ROOT/TestSupport.hxx>
#include <TROOT.h>

#include <gtest/gtest.h>

#include <iostream>

using namespace ROOT::RDF;

auto fileName0 = "RCsvDS_test_headers.csv";
auto fileName1 = "RCsvDS_test_noheaders.csv";
auto fileName2 = "RCsvDS_test_empty.csv";
auto fileName3 = "RCsvDS_test_win.csv";
auto fileName4 = "RCsvDS_test_NaNs.csv";

// must use http: we cannot use https on macOS until we upgrade to the newest Davix
// and turn on the macOS SecureTransport layer.
auto url0 = "http://root.cern/files/dataframe_test_datasource.csv";


TEST(RCsvDS, ColTypeNames)
{
   RCsvDS tds(fileName0);
   tds.SetNSlots(1);

   auto colNames = tds.GetColumnNames();

   EXPECT_TRUE(tds.HasColumn("Name"));
   EXPECT_TRUE(tds.HasColumn("Age"));
   EXPECT_FALSE(tds.HasColumn("Address"));

   EXPECT_STREQ("Height", colNames[2].c_str());
   EXPECT_STREQ("Married", colNames[3].c_str());
   EXPECT_STREQ("Salary", colNames[4].c_str());

   EXPECT_STREQ("std::string", tds.GetTypeName("Name").c_str());
   EXPECT_STREQ("Long64_t", tds.GetTypeName("Age").c_str());
   EXPECT_STREQ("double", tds.GetTypeName("Height").c_str());
   EXPECT_STREQ("bool", tds.GetTypeName("Married").c_str());
   EXPECT_STREQ("double", tds.GetTypeName("Salary").c_str());
}

TEST(RCsvDS, ColNamesNoHeaders)
{
   RCsvDS tds(fileName1, false);
   tds.SetNSlots(1);

   auto colNames = tds.GetColumnNames();

   EXPECT_STREQ("Col0", colNames[0].c_str());
   EXPECT_STREQ("Col1", colNames[1].c_str());
   EXPECT_STREQ("Col2", colNames[2].c_str());
   EXPECT_STREQ("Col3", colNames[3].c_str());
}

TEST(RCsvDS, EmptyFile)
{
   // Cannot read headers
   EXPECT_THROW(RCsvDS{fileName2}, std::runtime_error);
   // Cannot infer column types
   EXPECT_THROW(RCsvDS(fileName2, false), std::runtime_error);
}

TEST(RCsvDS, EntryRanges)
{
   RCsvDS tds(fileName0);
   tds.SetNSlots(3U);
   tds.Initialize();

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

TEST(RCsvDS, ColumnReaders)
{
   RCsvDS tds(fileName0);
   const auto nSlots = 3U;
   tds.SetNSlots(nSlots);
   auto vals = tds.GetColumnReaders<Long64_t>("Age");
   tds.Initialize();
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

TEST(RCsvDS, ColumnReadersWrongType)
{
   RCsvDS tds(fileName0);
   const auto nSlots = 3U;
   tds.SetNSlots(nSlots);
   int res = 1;
   try {
      auto vals = tds.GetColumnReaders<float>("Age");
   } catch (const std::runtime_error &e) {
      EXPECT_STREQ("The type selected for column \"Age\" does not correspond to column type, which is Long64_t",
                   e.what());
      res = 0;
   }
   EXPECT_EQ(0, res);
}

TEST(RCsvDS, Snapshot)
{
   auto tdf = ROOT::RDF::MakeCsvDataFrame(fileName0);
   auto snap = tdf.Snapshot<Long64_t>("data","csv2root.root", {"Age"});
   auto ages = *snap->Take<Long64_t>("Age");
   std::vector<Long64_t> agesRef {60LL, 50LL, 40LL, 30LL, 1LL, -1LL};
   for (auto i : ROOT::TSeqI(agesRef.size())) {
      EXPECT_EQ(ages[i], agesRef[i]);
   }
}

TEST(RCsvDS, ColumnReadersString)
{
   RCsvDS tds(fileName0);
   const auto nSlots = 3U;
   tds.SetNSlots(nSlots);
   auto vals = tds.GetColumnReaders<std::string>("Name");
   tds.Initialize();
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

TEST(RCsvDS, ProgressiveReadingEntryRanges)
{
   auto chunkSize = 3LL;
   RCsvDS tds(fileName0, true, ',', chunkSize);
   const auto nSlots = 3U;
   tds.SetNSlots(nSlots);
   auto vals = tds.GetColumnReaders<std::string>("Name");
   tds.Initialize();

   std::vector<std::string> names = {"Harry", "Bob,Bob", "\"Joe\"", "Tom", " John  ", " Mary Ann "};
   auto ranges = tds.GetEntryRanges();
   auto numIterations = 0U;
   while (!ranges.empty()) {
      EXPECT_EQ(nSlots, ranges.size());

      auto slot = 0U;
      for (auto &&range : ranges) {
         tds.InitSlot(slot, range.first);
         for (auto i : ROOT::TSeq<int>(range.first, range.second)) {
            tds.SetEntry(slot, i);
            auto val = *((std::string *)*vals[slot]);
            EXPECT_EQ(names[i], val);
         }
         slot++;
      }

      ranges = tds.GetEntryRanges();
      numIterations++;
   }

   EXPECT_EQ(2U, numIterations); // we should have processed 2 chunks
}

TEST(RCsvDS, ProgressiveReadingRDF)
{
   // Even chunks
   auto chunkSize = 2LL;
   auto tdf = ROOT::RDF::MakeCsvDataFrame(fileName0, true, ',', chunkSize);
   auto c = tdf.Count();
   EXPECT_EQ(6U, *c);

   // Uneven chunks
   chunkSize = 4LL;
   auto tdf2 = ROOT::RDF::MakeCsvDataFrame(fileName0, true, ',', chunkSize);
   auto c2 = tdf2.Count();
   EXPECT_EQ(6U, *c2);
}

#ifndef NDEBUG

TEST(RCsvDS, SetNSlotsTwice)
{
   auto theTest = []() {
      RCsvDS tds(fileName0);
      tds.SetNSlots(1);
      tds.SetNSlots(1);
   };
   ASSERT_DEATH(theTest(), "Setting the number of slots even if the number of slots is different from zero.");
}
#endif

#ifdef R__B64

TEST(RCsvDS, FromARDF)
{
   std::unique_ptr<RDataSource> tds(new RCsvDS(fileName0));
   ROOT::RDataFrame tdf(std::move(tds));
   auto max = tdf.Max<double>("Height");
   auto min = tdf.Min<double>("Height");
   auto c = tdf.Count();

   EXPECT_EQ(6U, *c);
   EXPECT_DOUBLE_EQ(200.5, *max);
   EXPECT_DOUBLE_EQ(.7, *min);
}

TEST(RCsvDS, FromARDFWithJitting)
{
   std::unique_ptr<RDataSource> tds(new RCsvDS(fileName0));
   ROOT::RDataFrame tdf(std::move(tds));
   auto max = tdf.Filter("Age<40").Max("Age");
   auto min = tdf.Define("Age2", "Age").Filter("Age2>30").Min("Age2");

   EXPECT_EQ(30, *max);
   EXPECT_EQ(40, *min);
}

TEST(RCsvDS, MultipleEventLoops)
{
   auto tdf = ROOT::RDF::MakeCsvDataFrame(fileName0, true, ',', 2LL);
   EXPECT_EQ(6U, *tdf.Count());
   EXPECT_EQ(6U, *tdf.Count());
   EXPECT_EQ(6U, *tdf.Count());
   EXPECT_EQ(6U, *tdf.Count());
}

TEST(RCsvDS, WindowsLinebreaks)
{
   auto tdf = ROOT::RDF::MakeCsvDataFrame(fileName3);
   EXPECT_EQ(6U, *tdf.Count());
}

TEST(RCsvDS, Remote)
{
   (void)url0; // silence -Wunused-const-variable
#ifdef R__HAS_DAVIX
   auto tdf = ROOT::RDF::MakeCsvDataFrame(url0, false);
   EXPECT_EQ(1U, *tdf.Count());
#else
   EXPECT_THROW(ROOT::RDF::MakeCsvDataFrame(url0, false), std::runtime_error);
#endif
}

// NOW MT!-------------
#ifdef R__USE_IMT

TEST(RCsvDS, DefineSlotCheckMT)
{
   const auto nSlots = 4U;
   ROOT::EnableImplicitMT(nSlots);

   std::vector<unsigned int> ids(nSlots, 0u);
   std::unique_ptr<RDataSource> tds(new RCsvDS(fileName0));
   ROOT::RDataFrame d(std::move(tds));
   auto m = d.DefineSlot("x", [&](unsigned int slot) {
                ids[slot] = 1u;
                return 1;
             }).Max("x");
   EXPECT_EQ(1, *m); // just in case

   const auto nUsedSlots = std::accumulate(ids.begin(), ids.end(), 0u);
   EXPECT_GT(nUsedSlots, 0u);
   EXPECT_LE(nUsedSlots, nSlots);
}

TEST(RCsvDS, FromARDFMT)
{
   std::unique_ptr<RDataSource> tds(new RCsvDS(fileName0));
   ROOT::RDataFrame tdf(std::move(tds));
   auto max = tdf.Max<double>("Height");
   auto min = tdf.Min<double>("Height");
   auto c = tdf.Count();

   EXPECT_EQ(6U, *c);
   EXPECT_DOUBLE_EQ(200.5, *max);
   EXPECT_DOUBLE_EQ(.7, *min);
}

TEST(RCsvDS, FromARDFWithJittingMT)
{
   std::unique_ptr<RDataSource> tds(new RCsvDS(fileName0));
   ROOT::RDataFrame tdf(std::move(tds));
   auto max = tdf.Filter("Age<40").Max("Age");
   auto min = tdf.Define("Age2", "Age").Filter("Age2>30").Min("Age2");

   EXPECT_EQ(30, *max);
   EXPECT_EQ(40, *min);
}

TEST(RCsvDS, ProgressiveReadingRDFMT)
{
   // Even chunks
   auto chunkSize = 2LL;
   auto tdf = ROOT::RDF::MakeCsvDataFrame(fileName0, true, ',', chunkSize);
   auto c = tdf.Count();
   EXPECT_EQ(6U, *c);

   // Uneven chunks
   chunkSize = 4LL;
   auto tdf2 = ROOT::RDF::MakeCsvDataFrame(fileName0, true, ',', chunkSize);
   auto c2 = tdf2.Count();
   EXPECT_EQ(6U, *c2);
}

TEST(RCsvDS, NaNTypeIndentification)
{
   RCsvDS tds(fileName4);

   EXPECT_STREQ("double", tds.GetTypeName("col1").c_str());
   EXPECT_STREQ("double", tds.GetTypeName("col2").c_str());
   EXPECT_STREQ("bool", tds.GetTypeName("col3").c_str());
   EXPECT_STREQ("double", tds.GetTypeName("col4").c_str());
   EXPECT_STREQ("Long64_t", tds.GetTypeName("col5").c_str());
   EXPECT_STREQ("Long64_t", tds.GetTypeName("col6").c_str());
   EXPECT_STREQ("std::string", tds.GetTypeName("col7").c_str());
   EXPECT_STREQ("std::string", tds.GetTypeName("col8").c_str());
}

TEST(RCsvDS, NanWarningChecks)
{
   ROOT::DisableImplicitMT(); // to allow usage of display

   auto rdf = ROOT::RDF::MakeCsvDataFrame(fileName4);
   auto d = rdf.Display<double, Long64_t, std::string>({"col2", "col5", "col8"});

   const std::string Warn = "Column \"col3\" of type bool contains empty cell(s).\n"
                            "There is no `nan` equivalent for bool type, hence `false` is stored.\n"
                            "Column \"col5\" of type Long64_t contains empty cell(s).\n"
                            "There is no `nan` equivalent for Long64_t type, hence `0` is stored.\n"
                            "Column \"col6\" of type Long64_t contains empty cell(s).\n"
                            "There is no `nan` equivalent for Long64_t type, hence `0` is stored.\n"
                            "Empty cells of type colT are read as 0/false. "
                            "You can manually set the column type to `double` to read NaN instead.\n";

   ROOT_EXPECT_WARNING(d->AsString(), "RCsvDS", Warn);
}

TEST(RCsvDS, SetCustomColumnTypes)
{
   RCsvDS tds(fileName4, true, ',', -1LL, {{"col5", 'D'}, {"col6", 'D'}});

   EXPECT_STREQ("double", tds.GetTypeName("col5").c_str());
   EXPECT_STREQ("double", tds.GetTypeName("col6").c_str());

   EXPECT_THROW(
      try {
         ROOT::RDF::MakeCsvDataFrame(fileName4, true, ',', -1LL, {{"col5", 'D'}, {"wrong", 'D'}});
      } catch (const std::runtime_error &err) {
         EXPECT_EQ(std::string(err.what()), "There is no column with name \"wrong\".");
         throw;
      },
      std::runtime_error);

   EXPECT_THROW(
      try {
         ROOT::RDF::MakeCsvDataFrame(fileName4, true, ',', -1LL, {{"col5", 'D'}, {"col6", 'W'}});
      } catch (const std::runtime_error &err) {
         std::string msg = "Type alias 'W' is not supported.\n";
         msg += "Supported type aliases are 'O' for boolean, 'D' for double, 'L' for Long64_t, 'T' for std::string.";
         EXPECT_EQ(std::string(err.what()), msg);
         throw;
      },
      std::runtime_error);
}

#endif // R__USE_IMT

#endif // R__B64
