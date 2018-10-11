#include "gtest/gtest.h"

#include <atomic>
#include <vector>

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

/// \class WrappedAtomic.
/// \brief Provides copy constructor for `atomic` and `+=` even for floats.
///
/// It provides the operations needed for `Hist` with atomic bin content.
template <class T>
class WrappedAtomic {
private:
   /// The wrapped atomic value.
   std::atomic<T> fVal;
public:
   /// Value-initialize the atomic.
   WrappedAtomic(): fVal(T{}) {}

   /// Copy-construct the atomic.
   WrappedAtomic(const WrappedAtomic &other):
      fVal(other.fVal.load(std::memory_order_relaxed))
   {}

   /// Construct the atomic from the underlying type.
   WrappedAtomic(T other): fVal(other) {}

   /// Increment operator, as needed by histogram filling.
   WrappedAtomic &operator+=(T rhs) {
      atomic_addeq(fVal, rhs);
      return *this;
   }

   /// Implicitly convert to the underlying type.
   operator T() const { return fVal.load(std::memory_order_relaxed); }
};

/** Basic tests for histograms of integral precision using vector<atomic> storage.
   */


#include "ROOT/RHist.hxx"

using namespace ROOT::Experimental;

// Storage using vector<atomic<>>
template <class PRECISION>
using atomicvec_t = std::vector<WrappedAtomic<PRECISION>>;

// A RHistData using vector<atomic<>> as storage.
template <int DIM, class PRECISION>
using content_t = Detail::RHistData<DIM, PRECISION, atomicvec_t<PRECISION>, RHistStatContent>;

template <int DIM, class PRECISION>
using uncert_t = Detail::RHistData<DIM, PRECISION, atomicvec_t<PRECISION>, RHistStatContent, RHistStatUncertainty>;

// Test creation of RHistImpl with atomic precision.
TEST(HistAtomicPrecisionTest, Create)
{
   Detail::RHistImpl<content_t<1, int>, RAxisEquidistant> h1I(RAxisEquidistant{100, 0., 1});
   Detail::RHistImpl<content_t<2, char>, RAxisEquidistant, RAxisEquidistant> h2C(RAxisEquidistant{100, 0., 1},
                                                                                 RAxisEquidistant{10, -1., 1});
   Detail::RHistImpl<content_t<1, long long>, RAxisIrregular> h1LLIrr(RAxisIrregular{{0., 0.1, 0.5, 1}});
   Detail::RHistImpl<uncert_t<1, float>, RAxisEquidistant> h1F(RAxisEquidistant{100, 0., 1});
   Detail::RHistImpl<uncert_t<1, double>, RAxisEquidistant> h1D(RAxisEquidistant{100, 0., 1});
}

// Test filling of RHistImpl with atomic precision.
TEST(HistAtomicPrecisionTest, Fill1Int)
{
   Detail::RHistImpl<content_t<1, int>, RAxisEquidistant> hist(RAxisEquidistant{100, 0., 1});
   hist.Fill({0.2222}, 12);
   EXPECT_EQ(12, hist.GetBinContent({0.2222}));
}

TEST(HistAtomicPrecisionTest, Fill2LongLong)
{
   Detail::RHistImpl<content_t<2, long long>, RAxisEquidistant, RAxisEquidistant> hist(RAxisEquidistant{100, 0., 1},
                                                                                       RAxisEquidistant{10, -1., 1});
   hist.Fill({0.1111, -0.2222}, 42ll);
   EXPECT_EQ(42ll, hist.GetBinContent({0.1111, -0.2222}));
}

TEST(HistAtomicPrecisionTest, Fill1CharIrr)
{
   Detail::RHistImpl<content_t<1, char>, RAxisIrregular> hist(RAxisIrregular{{0., 0.1, 0.5, 1}});
   hist.Fill({0.1111}, 17);
   EXPECT_EQ(17, hist.GetBinContent({0.1111}));
}

TEST(HistAtomicPrecisionTest, Fill1Double)
{
   Detail::RHistImpl<uncert_t<1, double>, RAxisEquidistant> hist(RAxisEquidistant{100, 0., 1});
   hist.Fill({0.2222}, 19.);
   EXPECT_DOUBLE_EQ(19., hist.GetBinContent({0.2222}));
}

TEST(HistAtomicPrecisionTest, Fill1Float)
{
   Detail::RHistImpl<uncert_t<1, float>, RAxisEquidistant> hist(RAxisEquidistant{100, 0., 1});
   hist.Fill({0.9999}, -9.);
   EXPECT_FLOAT_EQ(-9., hist.GetBinContent({0.9999}));
   EXPECT_FLOAT_EQ(9., hist.GetBinUncertainty(hist.GetBinIndex({0.9999})));
}
