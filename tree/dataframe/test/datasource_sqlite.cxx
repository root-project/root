#include <ROOT/TestSupport.hxx>
#include <ROOT/RConfig.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RSqliteDS.hxx>
#include <ROOT/TSeq.hxx>

#include <TROOT.h>
#include <TSystem.h>

#include <gtest/gtest.h>

#include <algorithm>

using namespace ROOT::RDF;

constexpr auto fileName0 = "RSqliteDS_test.sqlite";
constexpr auto url0 = "http://root.cern/files/RSqliteDS_test.sqlite";
constexpr auto url1 = "http://root.cern/files/RSqliteDS_test.sqlite.404";
constexpr auto query0 = "SELECT * FROM test";
constexpr auto query1 = "SELECT fint + 1, freal/1.0 as fmyreal, NULL, 'X', fblob FROM test";
constexpr auto query2 = "SELECT fint, freal, fint FROM test";
constexpr auto query3 = "SELECT fint, freal, ftext, fblob FROM test";
constexpr auto epsilon = 0.001;

TEST(RSqliteDS, Basics)
{
   auto rdf = ROOT::RDF::FromSqlite(fileName0, query0);
   EXPECT_EQ(1, *rdf.Min("fint"));
   EXPECT_EQ(2, *rdf.Max("fint"));

   EXPECT_THROW(ROOT::RDF::FromSqlite(fileName0, ""), std::runtime_error);
   EXPECT_THROW(ROOT::RDF::FromSqlite("", query0), std::runtime_error);
}

TEST(RSqliteDS, Snapshot)
{
   // Use query 3 to avoid storing a void * in the root file
   auto rdf = ROOT::RDF::FromSqlite(fileName0, query3);

   constexpr auto fname = "datasource_sqlite_snapshot.root";
   auto rdf_root = rdf.Snapshot("tree", fname);

   auto fintVec = rdf_root->Take<Long64_t>("fint");
   auto frealVec = rdf_root->Take<double>("freal");
   auto ftextVec = rdf_root->Take<std::string>("ftext");
   auto fblobVec = rdf_root->Take<std::vector<unsigned char>>("fblob");

   EXPECT_EQ(2U, fintVec->size());
   EXPECT_EQ(2U, frealVec->size());
   EXPECT_EQ(2U, ftextVec->size());
   EXPECT_EQ(2U, fblobVec->size());

   EXPECT_EQ(1, (*fintVec)[0]);
   EXPECT_EQ(2, (*fintVec)[1]);
   EXPECT_NEAR(1.0, (*frealVec)[0], epsilon);
   EXPECT_NEAR(2.0, (*frealVec)[1], epsilon);
   EXPECT_EQ("1", (*ftextVec)[0]);
   EXPECT_EQ("2", (*ftextVec)[1]);
   EXPECT_EQ(1U, (*fblobVec)[0].size());
   EXPECT_EQ('1', (*fblobVec)[0][0]);
   EXPECT_EQ(1U, (*fblobVec)[1].size());
   EXPECT_EQ('2', (*fblobVec)[1][0]);

   gSystem->Unlink(fname);
}

TEST(RSqliteDS, ColTypeNames)
{
   RSqliteDS rds(fileName0, query0);

   auto colNames = rds.GetColumnNames();
   ASSERT_EQ(5U, colNames.size());
   std::sort(colNames.begin(), colNames.end());
   EXPECT_EQ("fblob", colNames[0]);
   EXPECT_EQ("fint", colNames[1]);
   EXPECT_EQ("fnull", colNames[2]);
   EXPECT_EQ("freal", colNames[3]);
   EXPECT_EQ("ftext", colNames[4]);

   EXPECT_TRUE(rds.HasColumn("fint"));
   EXPECT_TRUE(rds.HasColumn("freal"));
   EXPECT_TRUE(rds.HasColumn("ftext"));
   EXPECT_TRUE(rds.HasColumn("fblob"));
   EXPECT_TRUE(rds.HasColumn("fnull"));
   EXPECT_FALSE(rds.HasColumn("foo"));

   EXPECT_EQ("Long64_t", rds.GetTypeName("fint"));
   EXPECT_EQ("double", rds.GetTypeName("freal"));
   EXPECT_EQ("std::string", rds.GetTypeName("ftext"));
   EXPECT_EQ("std::vector<unsigned char>", rds.GetTypeName("fblob"));
   EXPECT_EQ("void *", rds.GetTypeName("fnull"));
   EXPECT_THROW(rds.GetTypeName("foo"), std::runtime_error);
}

TEST(RSqliteDS, ExprTypeNames)
{
   RSqliteDS rds(fileName0, query1);

   EXPECT_EQ("Long64_t", rds.GetTypeName("fint + 1"));
   EXPECT_EQ("double", rds.GetTypeName("fmyreal"));
   EXPECT_EQ("void *", rds.GetTypeName("NULL"));
   EXPECT_EQ("std::string", rds.GetTypeName("'X'"));
   EXPECT_EQ("std::vector<unsigned char>", rds.GetTypeName("fblob"));
   EXPECT_THROW(rds.GetTypeName("foo"), std::runtime_error);
}

TEST(RSqliteDS, DuplicateColumns)
{
   RSqliteDS rds(fileName0, query2);
   rds.SetNSlots(1);

   EXPECT_EQ("Long64_t", rds.GetTypeName("fint"));
   EXPECT_EQ("double", rds.GetTypeName("freal"));
   auto val = rds.GetColumnReaders(/*slot*/ 0, "fint", typeid(Long64_t));
   rds.Initialize();
   auto ranges = rds.GetEntryRanges();
   ASSERT_EQ(1U, ranges.size());
   EXPECT_TRUE(rds.SetEntry(0, ranges[0].first));
   auto trueval = *val->TryGet<Long64_t>(0);
   EXPECT_EQ(1, trueval);
}

TEST(RSqliteDS, ColumnReaders)
{
   RSqliteDS rds(fileName0, query0);
   const auto nSlots = 2U;
   ROOT_EXPECT_WARNING(rds.SetNSlots(nSlots), "SetNSlots",
                       "Currently the SQlite data source faces performance degradation in multi-threaded mode. "
                       "Consider turning off IMT.");
   std::vector<std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>> vals;
   vals.reserve(2);
   for (auto i = 0U; i < 2U; i++) {
      vals.push_back(rds.GetColumnReaders(/*slot*/ i, "fint", typeid(Long64_t)));
   }
   rds.Initialize();
   auto ranges = rds.GetEntryRanges();
   EXPECT_EQ(1U, ranges.size());
   for (auto i : ROOT::TSeq<unsigned>(0, nSlots)) {
      EXPECT_TRUE(rds.SetEntry(i, ranges[0].first));
      auto val = *vals[i]->TryGet<Long64_t>(i);
      EXPECT_EQ(1, val);
   }

   EXPECT_THROW(rds.GetColumnReaders(/*slot*/ 0, "fint", typeid(double)), std::runtime_error);
}

TEST(RSqliteDS, GetEntryRanges)
{
   RSqliteDS rds(fileName0, query0);
   rds.Initialize();
   auto ranges = rds.GetEntryRanges();
   ASSERT_EQ(1U, ranges.size());
   EXPECT_EQ(0U, ranges[0].first);
   EXPECT_EQ(1U, ranges[0].second);
   ranges = rds.GetEntryRanges();
   ASSERT_EQ(1U, ranges.size());
   EXPECT_EQ(1U, ranges[0].first);
   EXPECT_EQ(2U, ranges[0].second);
   ranges = rds.GetEntryRanges();
   EXPECT_EQ(0U, ranges.size());

   // New event loop
   rds.Initialize();
   ranges = rds.GetEntryRanges();
   EXPECT_EQ(1U, ranges.size());
   EXPECT_EQ(0U, ranges[0].first);
   EXPECT_EQ(1U, ranges[0].second);
}

TEST(RSqliteDS, SetEntry)
{
   RSqliteDS rds(fileName0, query0);
   rds.SetNSlots(1);
   auto vint = rds.GetColumnReaders(0, "fint", typeid(Long64_t));
   auto vreal = rds.GetColumnReaders(0, "freal", typeid(double));
   auto vtext = rds.GetColumnReaders(0, "ftext", typeid(std::string));
   auto vblob = rds.GetColumnReaders(0, "fblob", typeid(std::vector<unsigned char>));
   auto vnull = rds.GetColumnReaders(0, "fnull", typeid(void *));

   rds.Initialize();

   rds.GetEntryRanges();
   EXPECT_TRUE(rds.SetEntry(0, 0));
   EXPECT_EQ(1, *vint->TryGet<Long64_t>(0));
   EXPECT_NEAR(1.0, *vreal->TryGet<double>(0), epsilon);
   EXPECT_EQ("1", *vtext->TryGet<std::string>(0));
   EXPECT_EQ(1U, (*vblob->TryGet<std::vector<unsigned char>>(0)).size());
   EXPECT_EQ('1', (*vblob->TryGet<std::vector<unsigned char>>(0))[0]);
   EXPECT_EQ(nullptr, *vnull->TryGet<void *>(0));

   rds.GetEntryRanges();
   EXPECT_TRUE(rds.SetEntry(0, 1));
   EXPECT_EQ(2, *vint->TryGet<Long64_t>(0));
   EXPECT_NEAR(2.0, *vreal->TryGet<double>(0), epsilon);
   EXPECT_EQ("2", *vtext->TryGet<std::string>(0));
   EXPECT_EQ(1U, (*vblob->TryGet<std::vector<unsigned char>>(0)).size());
   EXPECT_EQ('2', (*vblob->TryGet<std::vector<unsigned char>>(0))[0]);
   EXPECT_EQ(nullptr, *vnull->TryGet<void *>(0));
}

#ifdef R__USE_IMT

TEST(RSqliteDS, IMT)
{
   using Blob_t = std::vector<unsigned char>;
   const auto nSlots = 4U;
   ROOT::EnableImplicitMT(nSlots);

   ROOT::TestSupport::CheckDiagsRAII diagRAII{kWarning, "SetNSlots", "Currently the SQlite data source faces performance degradation in multi-threaded mode. Consider turning off IMT."};
   auto rdf = ROOT::RDF::FromSqlite(fileName0, query0);
   EXPECT_EQ(3, *rdf.Sum("fint"));
   EXPECT_NEAR(3.0, *rdf.Sum("freal"), epsilon);
   auto sum_text = *rdf.Reduce([](std::string a, std::string b) { return a + b; }, "ftext");
   std::sort(sum_text.begin(), sum_text.end());
   EXPECT_EQ("12", sum_text);
   auto sum_blob = *rdf.Reduce(
      [](Blob_t a, Blob_t b) {
         a.insert(a.end(), b.begin(), b.end());
         return a;
      },
      "fblob");
   std::sort(sum_blob.begin(), sum_blob.end());

   ROOT::DisableImplicitMT();

   ASSERT_EQ(2U, sum_blob.size());
   EXPECT_EQ('1', sum_blob[0]);
   EXPECT_EQ('2', sum_blob[1]);
}

#endif // R__USE_IMT

TEST(RSqliteDS, Davix)
{
#ifdef R__HAS_DAVIX
   auto rdf = ROOT::RDF::FromSqlite(url0, query0);
   EXPECT_EQ(1, *rdf.Min("fint"));
   EXPECT_EQ(2, *rdf.Max("fint"));

   EXPECT_THROW(ROOT::RDF::FromSqlite(url1, query0), std::runtime_error);
#else
   EXPECT_THROW(ROOT::RDF::FromSqlite(url0, query0), std::runtime_error);
   (void)url1; // silence -Wunused-const-variable
#endif
}
