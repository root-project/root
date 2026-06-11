#include "hist_test.hxx"

#include <cstdint>
#include <iterator>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

static_assert(std::is_nothrow_move_constructible_v<RProfile>);
static_assert(std::is_nothrow_move_assignable_v<RProfile>);

TEST(RProfile, Constructor)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis regularAxis(Bins, {0, Bins});

   // The most generic constructor takes a vector of axis objects.
   const std::vector<RAxisVariant> axes = {regularAxis, regularAxis};
   RProfile profile(axes);
   EXPECT_EQ(profile.GetNDimensions(), 2);
   const auto &engine = profile.GetEngine();
   EXPECT_EQ(engine.GetNDimensions(), 2);
   EXPECT_EQ(profile.GetAxes().size(), 2);
   // Both axes include underflow and overflow bins.
   EXPECT_EQ(profile.GetTotalNBins(), (Bins + 2) * (Bins + 2));

   // Test other constructors, including move-assignment.
   profile = RProfile(Bins, {0, Bins});
   ASSERT_EQ(profile.GetNDimensions(), 1);
   auto *regular = profile.GetAxes()[0].GetRegularAxis();
   ASSERT_TRUE(regular != nullptr);
   EXPECT_EQ(regular->GetNNormalBins(), Bins);
   EXPECT_EQ(regular->GetLow(), 0);
   EXPECT_EQ(regular->GetHigh(), Bins);
   // std::make_pair will take the types of the arguments, std::size_t in this case.
   profile = RProfile(Bins, std::make_pair(0, Bins));
   EXPECT_EQ(profile.GetNDimensions(), 1);

   // Brace-enclosed initializer list
   profile = RProfile({regularAxis});
   EXPECT_EQ(profile.GetNDimensions(), 1);
   profile = RProfile({regularAxis, regularAxis});
   EXPECT_EQ(profile.GetNDimensions(), 2);

   // Templated constructors
   profile = RProfile(regularAxis);
   EXPECT_EQ(profile.GetNDimensions(), 1);
   profile = RProfile(regularAxis, regularAxis);
   EXPECT_EQ(profile.GetNDimensions(), 2);
   profile = RProfile(regularAxis, regularAxis, regularAxis);
   EXPECT_EQ(profile.GetNDimensions(), 3);
}

TEST(RProfile, Fill)
{
   static constexpr std::size_t Bins = 20;
   RProfile profile(Bins, {0, Bins});

   profile.Fill(std::make_tuple(9.5), 25.0);

   std::array<RBinIndex, 1> indices = {9};
   auto &bin9 = profile.GetBinContent(indices);
   EXPECT_EQ(bin9.fSumValues, 25.0);
   EXPECT_EQ(bin9.fSumValues2, 625.0);
   EXPECT_EQ(bin9.fSum, 1.0);
   EXPECT_EQ(bin9.fSum2, 1.0);
}

TEST(RProfile, FillWeight)
{
   static constexpr std::size_t Bins = 20;
   RProfile profile(Bins, {0, Bins});

   profile.Fill(std::make_tuple(9.5), 25.0, RWeight(0.9));

   std::array<RBinIndex, 1> indices = {9};
   auto &bin9 = profile.GetBinContent(indices);
   EXPECT_FLOAT_EQ(bin9.fSumValues, 22.5);
   EXPECT_FLOAT_EQ(bin9.fSumValues2, 562.5);
   EXPECT_FLOAT_EQ(bin9.fSum, 0.9);
   EXPECT_FLOAT_EQ(bin9.fSum2, 0.81);
}

TEST(RProfile, FillCategorical)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis axis(categories);
   RProfile profile({axis});

   profile.Fill(std::make_tuple("c"), 25.0);

   std::array<RBinIndex, 1> indices = {2};
   auto &bin2 = profile.GetBinContent(indices);
   EXPECT_EQ(bin2.fSumValues, 25.0);
   EXPECT_EQ(bin2.fSumValues2, 625.0);
   EXPECT_EQ(bin2.fSum, 1.0);
   EXPECT_EQ(bin2.fSum2, 1.0);
}

TEST(RProfile, FillCategoricalWeight)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis axis(categories);
   RProfile profile({axis});

   profile.Fill(std::make_tuple("c"), 25.0, RWeight(0.9));

   std::array<RBinIndex, 1> indices = {2};
   auto &bin2 = profile.GetBinContent(indices);
   EXPECT_FLOAT_EQ(bin2.fSumValues, 22.5);
   EXPECT_FLOAT_EQ(bin2.fSumValues2, 562.5);
   EXPECT_FLOAT_EQ(bin2.fSum, 0.9);
   EXPECT_FLOAT_EQ(bin2.fSum2, 0.81);
}
