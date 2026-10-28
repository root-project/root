// For the licensing terms see $ROOTSYS/LICENSE.
// For the list of contributors see $ROOTSYS/README/CREDITS.

#ifndef ROOT_TVirtualImageUtils
#define ROOT_TVirtualImageUtils

#include "Rtypes.h"
#include "GuiTypes.h"

class TImage;

/// Interface for image operations on native windows.
class TVirtualImageUtils {
public:
   virtual ~TVirtualImageUtils() = default;

   virtual Bool_t PaintImage(TImage *image, Drawable_t window, Int_t x, Int_t y, Bool_t opaque) = 0;
   virtual Bool_t FillImageFromWindow(TImage *image, Window_t window, Int_t width, Int_t height) = 0;

   static TVirtualImageUtils *Create();

   ClassDef(TVirtualImageUtils, 0) // Image operations for native windows
};

#endif
