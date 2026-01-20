#include "hist_test.hxx"

#include <string>
#include <vector>

TEST(RAxisVariant, RegularAxis)
{
   static constexpr std::size_t Bins = 20;
   {
      const RAxisVariant axis{RRegularAxis(Bins, {0, Bins})};
      EXPECT_EQ(axis.GetVariant().index(), 0);
      EXPECT_TRUE(axis.GetRegularAxis() != nullptr);
      EXPECT_TRUE(axis.GetVariableBinAxis() == nullptr);
      EXPECT_TRUE(axis.GetCategoricalAxis() == nullptr);
   }
}

TEST(RAxisVariant, VariableBinAxis)
{
   static constexpr std::size_t Bins = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < Bins; i++) {
      bins.push_back(i);
   }
   bins.push_back(Bins);

   {
      const RAxisVariant axis{RVariableBinAxis(bins)};
      EXPECT_EQ(axis.GetVariant().index(), 1);
      EXPECT_TRUE(axis.GetRegularAxis() == nullptr);
      EXPECT_TRUE(axis.GetVariableBinAxis() != nullptr);
      EXPECT_TRUE(axis.GetCategoricalAxis() == nullptr);
   }
}

TEST(RAxisVariant, CategoricalAxis)
{
   const std::vector<std::string> categories = {"a", "b", "c"};

   {
      const RAxisVariant axis{RCategoricalAxis(categories)};
      EXPECT_EQ(axis.GetVariant().index(), 2);
      EXPECT_TRUE(axis.GetRegularAxis() == nullptr);
      EXPECT_TRUE(axis.GetVariableBinAxis() == nullptr);
      EXPECT_TRUE(axis.GetCategoricalAxis() != nullptr);
   }
}
