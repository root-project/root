#include "gtest/gtest.h"

#include "ROOT/THist.hxx"

#include <atomic>
#include <vector>

/** Basic tests for histograms of integral precision using vector<atomic> storage.
   NOTE that floating point precision is not supported - it would need a hand-written
   fetch_add implementation or the arrival of P0020 in C++. */

using namespace ROOT::Experimental;

// Storage using vector<atomic<>>
template <class PRECISION>
using atomicvec_t = std::vector<std::atomic<PRECISION>>;

// A THistData using vector<atomic<>> as storage.
template <int DIM, class PRECISION>
using atomic_histdata_t = Detail::THistData<DIM, PRECISION, atomicvec_t, THistStatContent>;

// THistImpl uses op+= to add to bin content; provide the relevant overload.
template <class INTEGRAL, class = decltype(std::declval<std::atomic<INTEGRAL>>().fetch_add(INTEGRAL()))>
std::atomic<INTEGRAL> &operator+=(std::atomic<INTEGRAL> &lhs, INTEGRAL rhs)
{
   lhs.fetch_add(rhs);
   return lhs;
}

// Test creation of THistImpl with atomic precision.
TEST(HistAtomicPrecisionTest, Create)
{
   Detail::THistImpl<atomic_histdata_t<1, int>, TAxisEquidistant> h1I(TAxisEquidistant{100, 0., 1});
   Detail::THistImpl<atomic_histdata_t<2, char>, TAxisEquidistant, TAxisEquidistant> h2C(TAxisEquidistant{100, 0., 1},
                                                                                         TAxisEquidistant{10, -1., 1});
   Detail::THistImpl<atomic_histdata_t<1, long long>, TAxisIrregular> h1LLIrr(TAxisIrregular{{0., 0.1, 0.5, 1}});
}

// Test filling of THistImpl with atomic precision.
TEST(HistAtomicPrecisionTest, Fill1Int)
{
   Detail::THistImpl<atomic_histdata_t<1, int>, TAxisEquidistant> h1I(TAxisEquidistant{100, 0., 1});
   h1I.Fill({0.2222}, 12);
   EXPECT_EQ(12, h1I.GetBinContent({0.2222}));
}

TEST(HistAtomicPrecisionTest, Fill2LongLong)
{
   Detail::THistImpl<atomic_histdata_t<2, long long>, TAxisEquidistant, TAxisEquidistant> hist(
      TAxisEquidistant{100, 0., 1}, TAxisEquidistant{10, -1., 1});
   hist.Fill({0.1111, -0.2222}, 42ll);
   EXPECT_EQ(42ll, hist.GetBinContent({0.1111, -0.2222}));
}

TEST(HistAtomicPrecisionTest, Fill1CharIrr)
{
   Detail::THistImpl<atomic_histdata_t<1, char>, TAxisIrregular> hist(TAxisIrregular{{0., 0.1, 0.5, 1}});
   hist.Fill({0.1111}, 17);
   EXPECT_EQ(17, hist.GetBinContent({0.1111}));
}
