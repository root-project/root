#include "gtest/gtest.h"

#include "TASImage.h"

namespace {

constexpr UInt_t kSize = 64;

constexpr UInt_t kPixels = kSize * kSize;

// Index of the four canvas corners.
constexpr UInt_t kCorners[4] = {0, kSize - 1, (kSize - 1) * kSize, kPixels - 1};

// Draw a filled circle of `colour` well inside a kSize x kSize image and check
// that the fill stayed inside it. The corner values are compared against what
// they were before drawing rather than against a constant, so the test does not
// depend on how a fresh TASImage is initialised.
void CheckFilledCircleStaysInside(const char *colour)
{
   TASImage img(kSize, kSize);

   UInt_t *argb = img.GetArgbArray();
   ASSERT_NE(argb, nullptr);

   UInt_t before[4];
   for (int i = 0; i < 4; ++i)
      before[i] = argb[kCorners[i]];
   const UInt_t centre = (kSize / 2) * kSize + kSize / 2;
   const UInt_t centreBefore = argb[centre];

   img.DrawCircle(kSize / 2, kSize / 2, kSize / 4, colour, -1);

   argb = img.GetArgbArray();
   ASSERT_NE(argb, nullptr);

   for (int i = 0; i < 4; ++i)
      EXPECT_EQ(argb[kCorners[i]], before[i]) << "the fill escaped the circle and reached corner " << i;

   EXPECT_NE(argb[centre], centreBefore) << "the circle was not filled at all";
}

} // namespace

// https://github.com/root-project/root/issues/23014
//
// libAfterImage scaled the coverage it wrote into the scratch canvas by the
// brush alpha, which made the flood fill that closes a filled shape depend on
// that alpha. Two symptoms followed, and this geometry shows both: at an alpha
// of 0x8C or below the fill never terminated, and between 0x8D and 0xFE it
// returned but leaked through the anti-aliased outline and covered the whole
// image. Only a fully opaque brush behaved correctly.

TEST(TASImage, FilledCircleOpaque)
{
   CheckFilledCircleStaysInside("#FFFF0000");
}

// Used to leak out of the circle and fill the whole image.
TEST(TASImage, FilledCircleHighAlpha)
{
   CheckFilledCircleStaysInside("#C0FF0000");
}

// Used to hang: the colour from the issue report.
TEST(TASImage, FilledCircleSemiTransparent)
{
   CheckFilledCircleStaysInside("#7FFF0000");
}
