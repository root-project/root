#include "gtest/gtest.h"

#include <atomic>
#include <vector>

/** Basic tests for histograms of integral precision using vector<atomic> storage.
   */

// Implementation of op+= for integral.
template <class INTEGRAL>
std::atomic<INTEGRAL> &atomic_addeq(std::atomic<INTEGRAL> &lhs, INTEGRAL rhs,
                                    decltype(std::declval<std::atomic<INTEGRAL>>().fetch_add(INTEGRAL())) * = nullptr)
{
   lhs.fetch_add(rhs);
   return lhs;
}

// Implementation of op+= for floats.
template <class FLOAT>
std::atomic<FLOAT> &atomic_addeq(std::atomic<FLOAT> &lhs, FLOAT rhs,
                                 typename std::enable_if<std::is_floating_point<FLOAT>::value>::type * = nullptr)
{
   auto oldval = lhs.load(std::memory_order_relaxed);
   auto newval = oldval + rhs;
   while (lhs.compare_exchange_strong(oldval, newval)) {
      // oldval is changed to what bin holds at the call to compare_exchange_strong
      newval = oldval + rhs;
   }
   return lhs;
}

// THistImpl uses op+= to add to bin content; provide the relevant overload.
template <class PRECISION>
std::atomic<PRECISION> &operator+=(std::atomic<PRECISION> &lhs, PRECISION rhs)
{
   return atomic_addeq(lhs, rhs);
}

#include "ROOT/THist.hxx"

using namespace ROOT::Experimental;

// Storage using vector<atomic<>>
template <class PRECISION>
using atomicvec_t = std::vector<std::atomic<PRECISION>>;

// A THistData using vector<atomic<>> as storage.
template <int DIM, class PRECISION>
using content_t = Detail::THistData<DIM, PRECISION, atomicvec_t, THistStatContent>;

template <int DIM, class PRECISION>
using uncert_t = Detail::THistData<DIM, PRECISION, atomicvec_t, THistStatContent, THistStatUncertainty>;

// Test creation of THistImpl with atomic precision.
TEST(HistAtomicPrecisionTest, Create)
{
   Detail::THistImpl<content_t<1, int>, TAxisEquidistant> h1I(TAxisEquidistant{100, 0., 1});
   Detail::THistImpl<content_t<2, char>, TAxisEquidistant, TAxisEquidistant> h2C(TAxisEquidistant{100, 0., 1},
                                                                                 TAxisEquidistant{10, -1., 1});
   Detail::THistImpl<content_t<1, long long>, TAxisIrregular> h1LLIrr(TAxisIrregular{{0., 0.1, 0.5, 1}});
   Detail::THistImpl<uncert_t<1, float>, TAxisEquidistant> h1F(TAxisEquidistant{100, 0., 1});
   Detail::THistImpl<uncert_t<1, double>, TAxisEquidistant> h1D(TAxisEquidistant{100, 0., 1});
}

// Test filling of THistImpl with atomic precision.
TEST(HistAtomicPrecisionTest, Fill1Int)
{
   Detail::THistImpl<content_t<1, int>, TAxisEquidistant> hist(TAxisEquidistant{100, 0., 1});
   hist.Fill({0.2222}, 12);
   EXPECT_EQ(12, hist.GetBinContent({0.2222}));
}

TEST(HistAtomicPrecisionTest, Fill2LongLong)
{
   Detail::THistImpl<content_t<2, long long>, TAxisEquidistant, TAxisEquidistant> hist(TAxisEquidistant{100, 0., 1},
                                                                                       TAxisEquidistant{10, -1., 1});
   hist.Fill({0.1111, -0.2222}, 42ll);
   EXPECT_EQ(42ll, hist.GetBinContent({0.1111, -0.2222}));
}

TEST(HistAtomicPrecisionTest, Fill1CharIrr)
{
   Detail::THistImpl<content_t<1, char>, TAxisIrregular> hist(TAxisIrregular{{0., 0.1, 0.5, 1}});
   hist.Fill({0.1111}, 17);
   EXPECT_EQ(17, hist.GetBinContent({0.1111}));
}

TEST(HistAtomicPrecisionTest, Fill1Double)
{
   Detail::THistImpl<uncert_t<1, double>, TAxisEquidistant> hist(TAxisEquidistant{100, 0., 1});
   hist.Fill({0.2222}, 19.);
   EXPECT_DOUBLE_EQ(19., hist.GetBinContent({0.2222}));
}

TEST(HistAtomicPrecisionTest, Fill1Float)
{
   Detail::THistImpl<uncert_t<1, float>, TAxisEquidistant> hist(TAxisEquidistant{100, 0., 1});
   hist.Fill({0.9999}, -9.);
   EXPECT_FLOAT_EQ(-9., hist.GetBinContent({0.9999}));
   EXPECT_FLOAT_EQ(9., hist.GetBinUncertainty(hist.GetBinIndex({0.9999})));
}
