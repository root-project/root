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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGHorizontal3DLine and TGVertical3DLine                              //
//                                                                      //
// A horizontal 3D line is a line that typically separates a toolbar    //
// from the menubar.                                                    //
// A vertical 3D line is a line that can be used to separate groups of  //
// widgets.                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif


class TGHorizontal3DLine : public TGFrame {

public:
   TGHorizontal3DLine(const TGWindow *p = 0, UInt_t w = 4, UInt_t h = 2,
                      UInt_t options = kChildFrame,
                      Pixel_t back = GetDefaultFrameBackground());

   virtual void DrawBorder() {
      gVirtualX->DrawLine(fId, GetShadowGC()(),  0, 0, fWidth-2, 0);
      gVirtualX->DrawLine(fId, GetHilightGC()(), 0, 1, fWidth-1, 1);
      gVirtualX->DrawLine(fId, GetHilightGC()(), fWidth-1, 0, fWidth-1, 1);
   }
   virtual void  SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TGHorizontal3DLine,0)  //A horizontal 3D separator line
};


class TGVertical3DLine : public TGFrame {

public:
   TGVertical3DLine(const TGWindow *p = 0, UInt_t w = 2, UInt_t h = 4,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());

   virtual void DrawBorder() {
      gVirtualX->DrawLine(fId, GetShadowGC()(),  0, 0, 0, fHeight-2);
      gVirtualX->DrawLine(fId, GetHilightGC()(), 1, 0, 1, fHeight-1);
      gVirtualX->DrawLine(fId, GetHilightGC()(), 0, fHeight-1, 1, fHeight-1);
   }
   virtual void  SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TGVertical3DLine,0)  //A vertical 3D separator line
};

#endif
