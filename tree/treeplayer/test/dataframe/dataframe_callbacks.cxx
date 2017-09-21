#include "ROOT/TDataFrame.hxx"
#include "TRandom.h"
#include "gtest/gtest.h"
using namespace ROOT::Experimental;
using namespace ROOT::Experimental::TDF;
using namespace ROOT::Detail::TDF;

// fixture that provides a TDF with no data-source and a single column "x" containing normal-distributed doubles
class TDFCallbacks : public ::testing::Test {
protected:
   const ULong64_t nEvents = 8ull; // must be initialized before fLoopManager

private:
   TDataFrame fLoopManager;
   TInterface<TLoopManager> DefineRandomCol()
   {
      TRandom r;
      return fLoopManager.Define("x", [r]() mutable { return r.Gaus(); });
   }

protected:
   TDFCallbacks() : fLoopManager(nEvents), tdf(DefineRandomCol()) {}
   TInterface<TLoopManager> tdf;
};

TEST_F(TDFCallbacks, Histo1DWithFillTOHelper)
{
   auto h = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   ULong64_t everyN = 1ull;
   h.RegisterCallback(everyN, [&i, &everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), everyN * i);
   });
   *h;
}

TEST_F(TDFCallbacks, MultipleCallbacks)
{
   auto h = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t everyN = 1ull;
   ULong64_t i = 0ull;
   h.RegisterCallback(everyN, [&i, everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), i);
   });

   everyN = 2ull;
   ULong64_t i2 = 0ull;
   h.RegisterCallback(everyN, [&i2, everyN](value_t &h_) {
      i2 += everyN;
      EXPECT_EQ(h_.GetEntries(), i2);
   });

   *h;
}

TEST_F(TDFCallbacks, MultipleEventLoops)
{
   auto h = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   h.RegisterCallback(1ull, [&i](value_t &) { ++i; });
   *h;

   auto h2 = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   *h2;

   EXPECT_EQ(i, nEvents);
}
