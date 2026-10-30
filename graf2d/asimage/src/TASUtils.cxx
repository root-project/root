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
#include "TVirtualX.h"
#include "TSystem.h"
#include "TVirtualPad.h"
#include "TError.h"


#ifdef NO_ASIMAGE

#include "TImage.h"

ASVisual *TASUtils::GetX11Visual()
{
   return nullptr;
}


Pixmap_t TASUtils::GetPixmap(TImage *)
{
   ::Error("TASUtils::GetPixmap", "Not implemented, ASImage is not compiled");
   return 0;
}

Pixmap_t TASUtils::GetMask(TImage *)
{
   ::Error("TASUtils::GetMask", "Not implemented, ASImage is not compiled");
   return 0;
}

Bool_t TASUtils::PaintImage(TImage *, Drawable_t, Int_t, Int_t, Bool_t)
{
   ::Error("TASUtils::PaintImage", "Not implemented, ASImage is not compiled");
   return kFALSE;
}

Bool_t TASUtils::FillImageFromWindow(TImage *, Window_t, Int_t, Int_t)
{
   ::Error("TASUtils::FillImageFromWindow", "Not implemented, ASImage is not compiled");
   return kFALSE;
}

std::unique_ptr<TImage> TASUtils::FromWindow(Window_t)
{
   ::Error("TASUtils::FromWindow", "Not implemented, ASImage is not compiled");
   return nullptr;
}

std::unique_ptr<TImage> TASUtils::FromPixmap(Pixmap_t, Pixmap_t)
{
   ::Error("TASUtils::FromPixmap", "Not implemented, ASImage is not compiled");
   return nullptr;
}

#else // NO_ASIMAGE

#include "TASImage.h"

ASVisual *TASUtils::GetX11Visual()
{
#ifndef WIN32
#ifndef R__HAS_COCOA

   thread_local struct ASHolder {
      ASVisual *fVisual = nullptr;
      ~ASHolder()
      {
         if (fVisual) {
            TASImage::DestroyVisual(fVisual);
            fVisual = nullptr;
         }
      }
   } holder;

   static Bool_t x11 = gVirtualX->InheritsFrom("TGX11");

   if (x11) {
      if (!holder.fVisual)
         holder.fVisual =
            TASImage::CreateX11Visual(gVirtualX->GetDisplay(), gVirtualX->GetScreen(), gVirtualX->GetDepth(),
                                      gVirtualX->GetVisual(), gVirtualX->GetColormap());
      return holder.fVisual;
   }

#endif
#endif
   return nullptr;
}


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

   auto vis = GetX11Visual();

   if (vis) {
      auto asimg = dynamic_cast<TASImage *>(timg);
      if (!asimg) {
         ::Warning("TASUtils::GetPixmap", "ASImage not provided");
         return 0;
      }

      ASImage *img = asimg->fScaledImage ? asimg->fScaledImage->fImage : asimg->fImage;

      return TASImage::ASImage2Pixmap(vis, gVirtualX->GetDefaultRootWindow(), img);
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
   if (!asimg || (width < 0) || (height < 0))
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

   auto vis = GetX11Visual();

   if (vis) {
      asimg->fImage = TASImage::Pixmap2ASImage(vis, wid, 0, 0, width, height);
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

   auto vis = GetX11Visual();

   if (vis) { //use built-in optimized version
      asimg->fImage = TASImage::Picture2ASImage(vis, pxm, mask, w, h);
   } else {
      unsigned char *bits = gVirtualX->GetColorBits(pxm, 0, 0, w, h);
      if (!bits)    // error
         return nullptr;

      unsigned char *mask_bits = mask ? gVirtualX->GetColorBits(mask, 0, 0, w, h) : nullptr;

      asimg->FromBitmap(bits, w, h, mask_bits);

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

   GContext_t fGC = gVirtualX->CreateGC(gVirtualX->GetDefaultRootWindow(), &gv);

   UInt_t width = asimg->GetScaledWidth();
   UInt_t height = asimg->GetScaledHeight();

   Bool_t res = kTRUE;

   auto vis = GetX11Visual();

   if (isX11 && vis) { //use built-in optimized version
      ASImage *im = asimg->fScaledImage ? asimg->fScaledImage->fImage : asimg->fImage;
      if (im)
         TASImage::ASImage2Drawable(vis, im, wid, fGC, 0, 0, x, y, width, height);
      else
         res = kFALSE;
   } else {
       auto argb =  asimg->GetArgbArray();

       Pixmap_t pic = argb ? gVirtualX->CreatePixmapFromData((unsigned char *) argb, width, height) : 0;

       if (pic) {
          if (!opaque) {
             SETBIT(width, 31);
             SETBIT(height, 31);
          }
          gVirtualX->CopyArea(pic, wid, fGC, 0, 0, width, height, x, y);
          gVirtualX->DeletePixmap(pic);
       } else
          res = kFALSE;
   }

   // free mask pixmap
   if (mask != kNone)
      gVirtualX->DeletePixmap(mask);

   gv.fMask = kGCClipMask;
   gv.fClipMask = kNone;
   if (fGC) {
      gVirtualX->ChangeGC(fGC, &gv);
      gVirtualX->DeleteGC(fGC);
   }

   return res;
}

#endif // NO_ASIMAGE
