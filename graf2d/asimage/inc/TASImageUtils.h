// For the licensing terms see $ROOTSYS/LICENSE.
// For the list of contributors see $ROOTSYS/README/CREDITS.

#ifndef ROOT_TASImageUtils
#define ROOT_TASImageUtils

#include "TASUtils.h"
#include "TVirtualImageUtils.h"

/// Image utility plugin backed by TASImage.
class TASImageUtils : public TVirtualImageUtils {
public:
   TASImageUtils() = default;

   Bool_t PaintImage(TImage *image, Drawable_t window, Int_t x, Int_t y, Bool_t opaque) override
   {
      return TASUtils::PaintImage(image, window, x, y, opaque);
   }

   Bool_t FillImageFromWindow(TImage *image, Window_t window, Int_t width, Int_t height) override
   {
      return TASUtils::FillImageFromWindow(image, window, width, height);
   }

   ClassDefOverride(TASImageUtils, 0) // TASImage implementation of native image operations
};

#endif
