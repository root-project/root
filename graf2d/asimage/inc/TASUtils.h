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

class TImage;

class TASUtils {
public:
   static Pixmap_t GetX11Pixmap(TImage *img);
   static Bool_t FromX11Window(TImage *timg, Window_t win, Int_t x, Int_t y, Int_t width, Int_t height);
};

#endif
