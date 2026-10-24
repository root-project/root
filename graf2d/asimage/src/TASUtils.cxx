// @(#)root/asimage:$Id$
// Author: Sergey Linev 16/10/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TASUtils.h"
#include "RConfigure.h"
#include "TASImage.h"
#include "TVirtualX.h"
#include "TSystem.h"
#include "TVirtualPad.h"
#include "TError.h"

#ifdef WIN32
#define X_DISPLAY_MISSING 1
#include "Windows4root.h"
#else
#ifdef R__HAS_COCOA
#define X_DISPLAY_MISSING 1
#endif
#endif

#include "afterbase.h"
#include "afterimage.h"
#include "bmp.h"

#ifndef X_DISPLAY_MISSING
const unsigned long kAllPlanes = ~0;
#endif

// Holder of ASVisual instance, destroyed when application closed
class ASVisualHolder {
   ASVisual *fVisual = nullptr;

public:

   GContext_t fGC = 0; // context used in PaintImage

   ASVisual *operator()()
   {
#ifndef X_DISPLAY_MISSING
      if (!fVisual) {
         Display *disp = (Display*) gVirtualX->GetDisplay();
         Int_t screen  = gVirtualX->GetScreen();
         Int_t depth   = gVirtualX->GetDepth();
         Visual *vis   = (Visual*) gVirtualX->GetVisual();
         Colormap cmap = (Colormap) gVirtualX->GetColormap();

         if (vis && cmap)
            fVisual = create_asvisual_for_id(disp, screen, depth, XVisualIDFromVisual(vis), cmap, nullptr);
      }
#endif
      if (!fVisual) {
         fVisual = create_asvisual(nullptr, 0, 0, nullptr);
         fVisual->dpy = nullptr; // fake (not used)
      }

      return fVisual;
   }

   ~ASVisualHolder()
   {
      if (fVisual) {
         destroy_asvisual(fVisual, 0);
         fVisual = nullptr;
      }
      if (fGC && gVirtualX)
         gVirtualX->DeleteGC(fGC);
   }
};

thread_local ASVisualHolder holder;


////////////////////////////////////////////////////////////////////////////////
/// Returns image pixmap.
/// The pixmap must deleted by user.
/// Replacement for deprecated TImage::GetPixmap method
/// Special handling for X11-based displays where ASImage has optimized implementation

Pixmap_t TASUtils::GetPixmap(TImage *timg)
{
   if (!timg) {
      ::Error("TASUtils::GetPixmap", "Image not provided");
      return 0;
   }
   static Bool_t x11 = gVirtualX->InheritsFrom("TGX11");

   if (x11) {
#ifndef X_DISPLAY_MISSING
      auto asimg = dynamic_cast<TASImage *>(timg);
      if (!asimg) {
         ::Warning("TASUtils::GetPixmap", "ASImage not provided");
         return 0;
      }

      auto vis = holder();

      if (!vis) {
         ::Warning("TASUtils::GetPixmap", "Visual not initiated");
         return 0;
      }

      ASImage *img = asimg->fScaledImage ? asimg->fScaledImage->fImage : asimg->fImage;

      return (Pixmap_t)asimage2pixmap(vis, gVirtualX->GetDefaultRootWindow(), img, nullptr, kTRUE);
#else
      ::Error("TASUtils::GetPixmap", "Invoke X11-specific methods on unsupported platform");
      return 0;
#endif
   }

   auto argb = timg->GetArgbArray();
   if (!argb) {
      ::Error("TASUtils::GetPixmap", "ARGB array is not exists");
      return 0;
   }

   return gVirtualX->CreatePixmapFromData((unsigned char*) argb, timg->GetScaledWidth(), timg->GetScaledHeight());
}

////////////////////////////////////////////////////////////////////////////////
/// Returns image mask pixmap (alpha channel).
/// The pixmap must deleted by user.
/// Replacement for deprecated TImage::GetMask() method

Pixmap_t TASUtils::GetMask(TImage *img)
{
   Pixmap_t pixmap = 0;

   UInt_t width = 0, height = 0;
   std::vector<char> bits;

   if (!img)
      ::Error("TGWindow::GetImageMask", "Image not provided");
   else if (img->GetMaskBits(width, height, bits))
      pixmap = gVirtualX->CreateBitmap(gVirtualX->GetDefaultRootWindow(), bits.data(), width, height);

   return pixmap;
}

////////////////////////////////////////////////////////////////////////////////
/// Create an image from the given window

Bool_t TASUtils::FillImageFromWindow(TImage *img, Window_t wid, Int_t width, Int_t height)
{
   auto asimg = dynamic_cast<TASImage *> (img);
   if (!asimg)
      return kFALSE;

   asimg->DestroyImage();
   asimg->DestroyScaledImage();

   gVirtualX->Update(1);
   if (!gThreadXAR) {
      gSystem->Sleep(10);
      gSystem->ProcessEvents();
      gSystem->Sleep(10);
      gSystem->ProcessEvents();
   }

   static Bool_t x11 = gVirtualX->InheritsFrom("TGX11");

   if (x11) {
      auto vis = holder();

      if (!vis) {
         ::Error("TASUtils::FillImageFromWindow", "Visual not initiated");
         return kFALSE;
      }

#ifndef X_DISPLAY_MISSING
      asimg->fImage = pixmap2asimage(vis, wid, 0, 0, width, height, kAllPlanes, 0, 0);
#else
      ::Error("TASUtils::FillImageFromWindow", "Invoke X11-specific methods on unsupported platform");
#endif
   } else {
      unsigned char *bits = gVirtualX->GetColorBits(wid, 0, 0, width, height);
      if (!bits) {
        ::Error("TASUtils::FillImageFromWindow", "Fail to extract color bits for specified window");
        return kFALSE;
      }
      asimg->FromBitmap(bits, width, height);
      delete [] bits;
   }

   if (!asimg->IsValid())
      return kFALSE;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Create an image from the given window

std::unique_ptr<TImage> TASUtils::FromWindow(Window_t wid)
{
   Int_t x = 0, y = 0;
   UInt_t w = 0, h = 0;

   gVirtualX->GetWindowSize(wid, x, y, w, h);

   (void) x; // position of window is not used
   (void) y;

   auto asimg = std::make_unique<TASImage>();

   if (!FillImageFromWindow(asimg.get(), wid, w, h))
      return nullptr;

   return asimg;
}

////////////////////////////////////////////////////////////////////////////////
/// Create image from pixmap.

std::unique_ptr<TImage> TASUtils::FromPixmap(Pixmap_t pxm, Pixmap_t mask)
{
   auto asimg = std::make_unique<TASImage>();

   Int_t xy;
   UInt_t w, h;
   gVirtualX->GetWindowSize(pxm, xy, xy, w, h);

   asimg->SetName(TString::Format("img_%dx%d",w, h));

   if (gVirtualX->InheritsFrom("TGX11")) { //use built-in optimized version
      auto vis = holder();

      if (!vis) {
         ::Warning("TASUtils::FromPixmap", "Visual not initiated");
         return nullptr;
      }

#ifndef X_DISPLAY_MISSING
      asimg->fImage = picture2asimage(vis, pxm, mask, 0, 0, w, h, kAllPlanes, 1, 0);
#else
      ::Error("TASUtils::FromPixmap", "Invoke X11-specific methods on unsupported platform");
#endif
   } else {
      unsigned char *bits = gVirtualX->GetColorBits(pxm, 0, 0, w, h);
      if (!bits)    // error
         return nullptr;

      unsigned char *mask_bits = mask ? gVirtualX->GetColorBits(mask, 0, 0, w, h) : nullptr;

      asimg->fImage = bitmap2asimage(bits, w, h, 0, mask_bits);

      if (mask_bits)
         delete [] mask_bits;
      delete [] bits;
   }

   if (!asimg->IsValid())
      return nullptr;

   return asimg;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint image on specified window

Bool_t TASUtils::PaintImage(TImage *timg, Drawable_t wid, Int_t x, Int_t y, Bool_t opaque)
{
   auto asimg = dynamic_cast<TASImage *>(timg);
   if (!asimg) {
      ::Warning("TASUtils::PaintImage", "Image not provided");
      return kFALSE;
   }

   Pixmap_t mask = kNone;

   Bool_t isX11 = gVirtualX->InheritsFrom("TGX11");

   if (isX11) {
      UInt_t mwidth = 0, mheight = 0;
      std::vector<char> bits;
      if (asimg->GetMaskBits(mwidth, mheight, bits))
         mask = gVirtualX->CreateBitmap(gVirtualX->GetDefaultRootWindow(), bits.data(), mwidth, mheight);
   }

   GCValues_t gv;
   gv.fMask = kGCClipMask | kGCClipXOrigin | kGCClipYOrigin;
   gv.fClipMask = mask;
   gv.fClipXOrigin = x;
   gv.fClipYOrigin = y;

   if (!holder.fGC) {
      holder.fGC = gVirtualX->CreateGC(gVirtualX->GetDefaultRootWindow(), &gv);
   } else {
      gVirtualX->ChangeGC(holder.fGC, &gv);
   }

   UInt_t width = asimg->GetScaledWidth();
   UInt_t height = asimg->GetScaledHeight();

   Bool_t res = kTRUE;

   if (isX11) { //use built-in optimized version
      #ifndef X_DISPLAY_MISSING
      ASImage *im = asimg->fScaledImage ? asimg->fScaledImage->fImage : asimg->fImage;
      auto vis = holder();
      if (vis && im)
         asimage2drawable(vis, wid, im, (GC)holder.fGC, 0, 0, x, y, width, height, 1);
      else
         res = kFALSE;
      #else
      ::Error("TASUtils::PaintImage", "Invoke X11-specific methods on unsupported platform");
      res = kFALSE;
      #endif
   } else {
       auto argb =  asimg->GetArgbArray();

       Pixmap_t pic = argb ? gVirtualX->CreatePixmapFromData((unsigned char *) argb, width, height) : 0;

       if (pic) {
          if (!opaque) {
             SETBIT(width, 31);
             SETBIT(height, 31);
          }
          gVirtualX->CopyArea(pic, wid, holder.fGC, 0, 0, width, height, x, y);
          gVirtualX->DeletePixmap(pic);
       } else
          res = kFALSE;
   }

   // free mask pixmap
   if (mask != kNone)
      gVirtualX->DeletePixmap(mask);

   gv.fMask = kGCClipMask;
   gv.fClipMask = kNone;
   if (holder.fGC)
      gVirtualX->ChangeGC(holder.fGC, &gv);

   return res;
}
