#include "gtest/gtest.h"

#include "TCanvas.h"
#include "TImage.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TVirtualPadPainter.h"
#include "TVirtualX.h"

#include <algorithm>
#include <memory>
#include <string>

namespace {

// Supply pixels and record image painting without requiring a window server.
class ImageBackend : public TVirtualX {
public:
   Int_t fCopies = 0;
   Int_t fX = 0;
   Int_t fY = 0;
   UInt_t fWidth = 0;
   UInt_t fHeight = 0;

   Window_t GetWindowID(Int_t) override { return 1; }
   Window_t GetCurrentWindow() const override { return 1; }
   GContext_t CreateGC(Drawable_t, GCValues_t *) override { return 1; }
   Pixmap_t CreatePixmapFromData(unsigned char *, UInt_t, UInt_t) override { return 1; }

   unsigned char *GetColorBits(Drawable_t, Int_t, Int_t, UInt_t width, UInt_t height) override
   {
      auto pixels = new unsigned char[4 * width * height];
      std::fill_n(pixels, 4 * width * height, 255);
      return pixels;
   }

   void CopyArea(Drawable_t, Drawable_t, GContext_t, Int_t, Int_t, UInt_t w, UInt_t h, Int_t x, Int_t y) override
   {
      ++fCopies;
      fX = x;
      fY = y;
      fWidth = w;
      fHeight = h;
   }
};

class BackendGuard {
   TVirtualX *fPrevious = gVirtualX;

public:
   explicit BackendGuard(TVirtualX &backend) { gVirtualX = &backend; }
   ~BackendGuard() { gVirtualX = fPrevious; }
};

} // namespace

TEST(TPadPainterImage, LoadHelperPluginOnDemand)
{
   gROOT->SetBatch(kTRUE);
   ASSERT_EQ(std::string(gSystem->GetLibraries()).find("libASImage"), std::string::npos);
   TCanvas canvas("image_painter", "image painter", 64, 64);
   auto painter = canvas.GetPainter();
   ASSERT_NE(painter, nullptr);
   painter->OnPad(&canvas);

   ASSERT_NE(gPluginMgr->FindHandler("TVirtualImageUtils"), nullptr);
   EXPECT_EQ(std::string(gSystem->GetLibraries()).find("libASImage"), std::string::npos);

   std::unique_ptr<TImage> image(TImage::Create());
   ASSERT_NE(image, nullptr);
   EXPECT_EQ(std::string(gSystem->GetLibraries()).find("libASImageUtils"), std::string::npos);

   ImageBackend backend;
   BackendGuard guard(backend);
   ASSERT_TRUE(painter->FillImageFromPad(image.get(), &canvas));
   EXPECT_TRUE(image->IsValid());
   EXPECT_EQ(image->GetWidth(), canvas.GetPadWidth());
   EXPECT_EQ(image->GetHeight(), canvas.GetPadHeight());
   EXPECT_NE(std::string(gSystem->GetLibraries()).find("libASImageUtils"), std::string::npos);

   painter->DrawImage(image.get(), 7, 9);
   EXPECT_EQ(backend.fCopies, 1);
   EXPECT_EQ(backend.fX, 7);
   EXPECT_EQ(backend.fY, 9);
   EXPECT_EQ(backend.fWidth, image->GetWidth());
   EXPECT_EQ(backend.fHeight, image->GetHeight());

   TImage unsupportedImage(1, 1);
   EXPECT_FALSE(painter->FillImageFromPad(&unsupportedImage, &canvas));
}
