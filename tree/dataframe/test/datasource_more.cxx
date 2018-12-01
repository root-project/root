#include <ROOT/RDataFrame.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <TROOT.h>

#include "RNonCopiableColumnDS.hxx"
#include "RStreamingDS.hxx"

#include "gtest/gtest.h"

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
