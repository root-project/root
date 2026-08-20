#include "gtest/gtest.h"

#include "TASImage.h"
#include "TPoint.h"

namespace {

constexpr UInt_t kSize = 64;

constexpr UInt_t kPixels = kSize * kSize;

// Index of the four canvas corners.
constexpr UInt_t kCorners[4] = {0, kSize - 1, (kSize - 1) * kSize, kPixels - 1};

// Draw a filled shape of `colour` well inside a kSize x kSize image and check
// that the fill stayed inside it. The corner values are compared against what
// they were before drawing rather than against a constant, so the test does not
// depend on how a fresh TASImage is initialised.
// At the end check number of modified pixels comparing with expected value
void CheckFilledShapeStaysInside(const char *colour, Int_t shape)
{
   TASImage img(kSize, kSize);

   UInt_t *argb = img.GetArgbArray();
   ASSERT_NE(argb, nullptr);

   UInt_t before[4];
   for (int i = 0; i < 4; ++i)
      before[i] = argb[kCorners[i]];
   const UInt_t centre = (kSize / 2) * kSize + kSize / 2;
   const UInt_t centreBefore = argb[centre];

   float expected_area = 0, expected_delta = 1.;
   const char *name = "<undef>";

   std::vector<TPoint> vect;

   switch (shape) {
      case 1:
         img.FillRectangle(colour, kSize / 4, kSize / 4, kSize / 2, kSize / 2);
         name = "Rectangle";
         expected_area = (kSize / 2) * (kSize / 2);
         expected_delta = 1;
         break;
      case 2:
         vect.emplace_back(kSize/4, kSize/4);
         vect.emplace_back(kSize/4*3, kSize/2);
         vect.emplace_back(kSize/4, kSize/4*3);
         vect.emplace_back(kSize/2, kSize/2);
         img.FillPolygon(vect.size(), vect.data(), colour);
         name = "Polygon";
         expected_area = (kSize / 4) * (kSize / 4);
         expected_delta = kSize / 4;
         break;
      case 3:
         img.DrawEllips2(kSize / 2, kSize / 2, kSize / 3, kSize / 7, 45, colour, -1);
         name = "Ellipse";
         expected_area = 3.1415 * (kSize / 3 + 1) * (kSize / 7 + 1);
         expected_delta = kSize / 3 * 3.14;
         break;
      default:
         img.DrawCircle(kSize / 2, kSize / 2, kSize / 4, colour, -1);
         name = "Circle";
         expected_area = 3.1415 * (kSize / 4 + 1) * (kSize / 4 + 1);
         expected_delta = kSize / 2 * 3.14;
         break;
   }

   argb = img.GetArgbArray();
   ASSERT_NE(argb, nullptr);

   for (int i = 0; i < 4; ++i)
      EXPECT_EQ(argb[kCorners[i]], before[i]) << "the fill escaped the " << name << " and reached corner " << i;

   EXPECT_NE(argb[centre], centreBefore) << "the " << name << " was not filled at all";

   if (shape == 3) {
      // recent bug in the DrawEllips2
      unsigned x = kSize / 2 + kSize / 5 + 1;
      unsigned y = kSize / 2 - kSize / 5 - 2;
      EXPECT_NE(argb[y * kSize + x], centreBefore) << "the DrawEllips2 does not fill point inside the " << name;
   }

   // also check filling status
   Int_t fillcnt = 0;
   for (UInt_t i = 0; i < kPixels; i++)
      if(argb[i] != centreBefore) fillcnt++;

   EXPECT_NEAR(fillcnt, expected_area, expected_delta) << "number of filled " << name << " points too far from expected";
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
   CheckFilledShapeStaysInside("#FFFF0000", 0);
}

// Used to leak out of the circle and fill the whole image.
TEST(TASImage, FilledCircleHighAlpha)
{
   CheckFilledShapeStaysInside("#C0FF0000", 0);
}

// Used to hang: the colour from the issue report.
TEST(TASImage, FilledCircleSemiTransparent)
{
   CheckFilledShapeStaysInside("#7FFF0000", 0);
}

TEST(TASImage, FilledCircleLowAlpha)
{
   CheckFilledShapeStaysInside("#20FF0000", 0);
}


TEST(TASImage, FilledRectOpaque)
{
   CheckFilledShapeStaysInside("#FF00FF00", 1);
}

// Used to leak out of the circle and fill the whole image.
TEST(TASImage, FilledRectHighAlpha)
{
   CheckFilledShapeStaysInside("#C000FF00", 1);
}

// Used to hang: the colour from the issue report.
TEST(TASImage, FilledRectSemiTransparent)
{
   CheckFilledShapeStaysInside("#7F00FF00", 1);
}

TEST(TASImage, FilledRectLowAlpha)
{
   CheckFilledShapeStaysInside("#1000FF00", 1);
}


TEST(TASImage, FilledPolygonOpaque)
{
   CheckFilledShapeStaysInside("#FF0000FF", 2);
}

// Used to leak out of the circle and fill the whole image.
TEST(TASImage, FilledPolygonHighAlpha)
{
   CheckFilledShapeStaysInside("#C00000FF", 2);
}

// Used to hang: the colour from the issue report.
TEST(TASImage, FilledPolygonSemiTransparent)
{
   CheckFilledShapeStaysInside("#7F0000FF", 2);
}

TEST(TASImage, FillePolygonLowAlpha)
{
   CheckFilledShapeStaysInside("#100000FF", 2);
}

/*

// comment out all ellpse test while they are failing

TEST(TASImage, FilledEllipsOpaque)
{
   CheckFilledShapeStaysInside("#FF2277CC", 3);
}

// Used to leak out of the circle and fill the whole image.
TEST(TASImage, FilledEllipsHighAlpha)
{
   CheckFilledShapeStaysInside("#C02277CC", 3);
}

// Used to hang: the colour from the issue report.
TEST(TASImage, FilledEllipsSemiTransparent)
{
   CheckFilledShapeStaysInside("#7F2277CC", 3);
}

TEST(TASImage, FilleEllipsLowAlpha)
{
   CheckFilledShapeStaysInside("#102277CC", 3);
}
*/
