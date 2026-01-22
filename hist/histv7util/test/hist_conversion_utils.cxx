#include "histutil_test.hxx"

#include <TAxis.h>

#include <string>
#include <vector>

using ROOT::Experimental::Hist::Internal::ConvertAxis;

TEST(ConvertAxis, RegularAxis)
{
   static constexpr std::size_t Bins = 20;
   const RAxisVariant src{RRegularAxis(Bins, {0, Bins})};

   TAxis dst;
   ConvertAxis(dst, src);

   EXPECT_FALSE(dst.IsVariableBinSize());
   EXPECT_EQ(dst.GetNbins(), Bins);
   EXPECT_EQ(dst.GetXmin(), 0.0);
   EXPECT_EQ(dst.GetXmax(), Bins);
   EXPECT_EQ(dst.GetXbins()->size(), 0);
}

TEST(ConvertAxis, VariableBinAxis)
{
   static constexpr std::size_t Bins = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < Bins; i++) {
      bins.push_back(i);
   }
   bins.push_back(Bins);
   const RAxisVariant src{RVariableBinAxis(bins)};

   TAxis dst;
   ConvertAxis(dst, src);

   EXPECT_TRUE(dst.IsVariableBinSize());
   EXPECT_EQ(dst.GetNbins(), Bins);
   EXPECT_EQ(dst.GetXmin(), 0.0);
   EXPECT_EQ(dst.GetXmax(), Bins);

   ASSERT_EQ(dst.GetXbins()->size(), Bins + 1);
   for (std::size_t i = 0; i <= Bins; i++) {
      EXPECT_EQ(dst.GetXbins()->At(i), bins[i]);
   }
}

TEST(ConvertAxis, CategoricalAxis)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RAxisVariant src{RCategoricalAxis(categories)};

   TAxis dst;
   ConvertAxis(dst, src);

   EXPECT_FALSE(dst.IsVariableBinSize());
   EXPECT_EQ(dst.GetNbins(), categories.size());
   for (std::size_t i = 0; i < categories.size(); i++) {
      EXPECT_EQ(dst.GetBinLabel(i + 1), categories[i]);
   }
}
