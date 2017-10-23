#include <ROOT/TDataFrame.hxx>
#include <ROOT/RMakeUnique.hxx>

#include "TNonCopiableDS.hxx"
#include "TStreamingDS.hxx"

#include "gtest/gtest.h"

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::TDF;

TEST(TNonCopiableDS, UseNonCopiableColumnType)
{
   std::unique_ptr<TDataSource> tds(new NonCopiableDS());
   TDataFrame tdf(std::move(tds));

   auto getNCVal = [](NonCopiableDS::NonCopiable_t &nc) { return nc.fValue; };
   auto m = *tdf.Define("val", getNCVal, {NonCopiableDS::fgColumnName}).Min<NonCopiableDS::NonCopiable_t::type>("val");

   NonCopiableDS::NonCopiable_t dummy;

   EXPECT_EQ(dummy.fValue, m);
}

TEST(TStreamingDS, MultipleEntryRanges)
{
   TDataFrame tdf(std::make_unique<TStreamingDS>());
   auto c = tdf.Count();
   auto ansmin = tdf.Min<int>("ans");
   auto ansmax = tdf.Max("ans");
   EXPECT_EQ(*c, 4ull);
   EXPECT_EQ(*ansmin, *ansmax);
   EXPECT_EQ(*ansmin, 42);
}

#ifdef R__USE_IMT
TEST(TStreamingDS, MultipleEntryRangesMT)
{
   ROOT::EnableImplicitMT(2);
   TDataFrame tdf(std::make_unique<TStreamingDS>());
   auto c = tdf.Count();
   auto ansmin = tdf.Min<int>("ans");
   auto ansmax = tdf.Max("ans");
   EXPECT_EQ(*c, 8ull); // TStreamingDS provides 4 entries per slot
   EXPECT_EQ(*ansmin, *ansmax);
   EXPECT_EQ(*ansmin, 42);
   ROOT::DisableImplicitMT();
}
#endif