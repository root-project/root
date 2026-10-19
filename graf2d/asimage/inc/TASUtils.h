// @(#)root/asimage:$Id$
// Author: Sergey Linev 16/10/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TASUtils
#define ROOT_TASUtils

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TASUtils                                                             //
//                                                                      //
// Extra functions for TASImage requiring gVirtualX and X11 usage       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"
#include "GuiTypes.h"
#include <memory>

class TImage;

class TASUtils {
public:
   static Pixmap_t GetPixmap(TImage *timg);
   static Pixmap_t GetMask(TImage *timg);

   static Bool_t   PaintImage(TImage *timg, Drawable_t wid, Int_t x, Int_t y, Bool_t opaque = kFALSE);
   static Bool_t   FillImageFromWindow(TImage *img, Window_t wid, Int_t width, Int_t height);

   static std::unique_ptr<TImage> FromWindow(Window_t wid);
   static std::unique_ptr<TImage> FromPixmap(Pixmap_t pxm, Pixmap_t mask);
};

#endif
