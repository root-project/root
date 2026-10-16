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

const unsigned long kAllPlanes = ~0;

// Holder of ASVisual instance, destroyed when application closed
class ASVisualHolder {
    ASVisual *fVisual = nullptr;
    public:

    ASVisual *operator()()
    {
        if (!fVisual) {
#ifndef WIN32
#ifndef R__HAS_COCOA
           Display *disp = (Display*) gVirtualX->GetDisplay();
           Int_t screen  = gVirtualX->GetScreen();
           Int_t depth   = gVirtualX->GetDepth();
           Visual *vis   = (Visual*) gVirtualX->GetVisual();
           Colormap cmap = (Colormap) gVirtualX->GetColormap();

           if (vis && cmap)
              fVisual = create_asvisual_for_id(disp, screen, depth, XVisualIDFromVisual(vis), cmap, nullptr);
#endif
#endif
           if (!fVisual) {
              fVisual = create_asvisual(nullptr, 0, 0, nullptr);
              fVisual->dpy = nullptr; // fake (not used)
            }
        }

        return fVisual;
    }

   ~ASVisualHolder()
   {
      if (fVisual) {
         destroy_asvisual(fVisual, 0);
         fVisual = nullptr;
      }
   }
};


thread_local ASVisualHolder holder;

////////////////////////////////////////////////////////////////////////////////
/// Returns image X11 pixmap.
/// Workrs only on the platforms with support of X11
/// The pixmap must deleted by user.

Pixmap_t TASUtils::GetX11Pixmap(TImage *timg)
{
   auto asimg = dynamic_cast<TASImage *>(timg);
   if (!asimg) {
      ::Warning("TASUtils::GetX11Pixmap", "Image not provided");
      return 0;
   }

   auto vis = holder();

   if (!vis) {
      ::Warning("TASUtils::GetX11Pixmap", "Visual not initiated");
      return 0;
   }

   ASImage *img = asimg->fScaledImage ? asimg->fScaledImage->fImage : asimg->fImage;

   return (Pixmap_t)asimage2pixmap(vis, gVirtualX->GetDefaultRootWindow(), img, nullptr, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Create an image from the given X11 window
/// Can only be used on platforms where X11 is supported

Bool_t TASUtils::FromX11Window(TImage *timg, Window_t win, Int_t x, Int_t y, Int_t width, Int_t height)
{
   auto asimg = dynamic_cast<TASImage *>(timg);
   if (!asimg) {
      ::Warning("TASUtils::FromX11Window", "Image not provided");
      return kFALSE;
   }

   auto vis = holder();

   if (!vis) {
      ::Warning("TASUtils::FromX11Window", "Visual not initiated");
      return kFALSE;
   }

   asimg->DestroyImage();
   asimg->DestroyScaledImage();

   asimg->fImage = pixmap2asimage(vis, win, x, y, width, height, kAllPlanes, 0, 0);

   return asimg->fImage != nullptr;
}
