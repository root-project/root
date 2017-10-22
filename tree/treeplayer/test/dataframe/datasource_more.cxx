#include <ROOT/TDataFrame.hxx>

#include "TNonCopiableDS.hxx"

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
