#include "histutil_test.hxx"

#include <TAxis.h>

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
}
