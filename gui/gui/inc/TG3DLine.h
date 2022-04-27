// @(#)root/gui:$Id$
// Author: Fons Rademakers   6/09/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TG3DLine
#define ROOT_TG3DLine


#include "TGFrame.h"

class TGHorizontal3DLine : public TGFrame {

public:
   TGHorizontal3DLine(const TGWindow *p = nullptr, UInt_t w = 4, UInt_t h = 2,
                      UInt_t options = kChildFrame,
                      Pixel_t back = GetDefaultFrameBackground());

   void DrawBorder() override;

   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGHorizontal3DLine,0)  //A horizontal 3D separator line
};


class TGVertical3DLine : public TGFrame {

public:
   TGVertical3DLine(const TGWindow *p = nullptr, UInt_t w = 2, UInt_t h = 4,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());

   void DrawBorder() override;

   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGVertical3DLine,0)  //A vertical 3D separator line
};

#endif
