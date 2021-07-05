#include <ROOT/RDataFrame.hxx>
#include <TROOT.h>
#include <TSystem.h>

#include "RNonCopiableColumnDS.hxx"
#include "RStreamingDS.hxx"
#include "RArraysDS.hxx"

#include "ROOTUnitTestSupport.h"
#include "gtest/gtest.h"

#include <memory>

using namespace ROOT::RDF;

TEST(RNonCopiableColumnDS, UseNonCopiableColumnType)
{
   std::unique_ptr<RDataSource> tds(new RNonCopiableColumnDS());
   ROOT::RDataFrame tdf(std::move(tds));

   auto getNCVal = [](RNonCopiableColumnDS::NonCopiable_t &nc) { return nc.fValue; };
   auto m = *tdf.Define("val", getNCVal, {RNonCopiableColumnDS::fgColumnName}).Min<RNonCopiableColumnDS::NonCopiable_t::type>("val");

   RNonCopiableColumnDS::NonCopiable_t dummy;

   EXPECT_EQ(dummy.fValue, m);
}

TEST(RStreamingDS, MultipleEntryRanges)
{
   ROOT::RDataFrame tdf(std::make_unique<RStreamingDS>());
   auto c = tdf.Count();
   auto ansmin = tdf.Min<int>("ans");
   auto ansmax = tdf.Max("ans");
   EXPECT_EQ(*c, 4ull);
   EXPECT_EQ(*ansmin, *ansmax);
   EXPECT_EQ(*ansmin, 42);
}

TEST(RArraysDS, ShortSyntaxForCollectionSizes)
{
   ROOT::RDataFrame df(std::make_unique<RArraysDS>());
   // GetColumnNames must hide column "__rdf_sizeof_var"...
   EXPECT_EQ(df.GetColumnNames(), std::vector<std::string>{"var"});
   // ...but it must nonetheless be a valid column
   EXPECT_EQ(df.GetColumnType("#var"), "std::size_t");
   EXPECT_EQ(df.GetColumnType("var"), "std::vector<int>");
   EXPECT_EQ(df.Take<std::size_t>("#var").GetValue(), std::vector<std::size_t>{1ull});
   EXPECT_EQ(df.Take<std::vector<int>>("var").GetValue(), std::vector<std::vector<int>>{{42}});
}

TEST(RArraysDS, SnapshotAndShortSyntaxForCollectionSizes)
{
   const auto fname = "snapshotandshortsyntaxforcollectionsizes.root";
   ROOT::RDataFrame df(std::make_unique<RArraysDS>());

   // Snapshot must ignore the #var columns
   df.Snapshot("t", fname);
   TFile f(fname);
   auto *t = f.Get<TTree>("t");
   auto *blist = t->GetListOfBranches();
   EXPECT_EQ(blist->GetEntriesUnsafe(), 1u);
   EXPECT_STREQ(blist->At(0)->GetName(), "var");
   f.Close(); // Windows does not allow deletion/recreation of files that are still in use.

   // Snapshot must throw if #var is passed explicitly
   EXPECT_THROW(df.Snapshot<std::size_t>("t", fname, {"#var"}), std::runtime_error);

   // ...and work if the Snapshot is performed via an Alias
   const auto nvar =
      df.Alias("nvar", "#var").Snapshot<std::size_t>("t", fname, {"nvar"})->Take<std::size_t>("nvar").GetValue();
   EXPECT_EQ(nvar, std::vector<std::size_t>{1});

   gSystem->Unlink(fname);
}

TEST(RArraysDS, CacheAndShortSyntaxForCollectionSizes)
{
   ROOT::RDataFrame df(std::make_unique<RArraysDS>());

   // Cache must ignore the #var columns
   auto cached = df.Cache();
   EXPECT_EQ(cached.GetColumnNames(), std::vector<std::string>{"var"});

   // Cache must throw if #var is passed explicitly...
   EXPECT_THROW(df.Cache<std::size_t>({"#var"}), std::runtime_error);

   // ...and work if the caching is performed via an Alias
   const auto nvar = df.Alias("nvar", "#var").Cache<std::size_t>({"nvar"}).Take<std::size_t>("nvar").GetValue();
   EXPECT_EQ(nvar, std::vector<std::size_t>{1});
}

#ifdef R__USE_IMT
TEST(RStreamingDS, MultipleEntryRangesMT)
{
   ROOT::EnableImplicitMT(2);
   ROOT::RDataFrame tdf(std::make_unique<RStreamingDS>());
   auto c = tdf.Count();
   auto ansmin = tdf.Min<int>("ans");
   auto ansmax = tdf.Max("ans");
   EXPECT_EQ(*c, 8ull); // TStreamingDS provides 4 entries per slot
   EXPECT_EQ(*ansmin, *ansmax);
   EXPECT_EQ(*ansmin, 42);
   ROOT::DisableImplicitMT();
}
#endif
