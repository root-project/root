// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEvePolygonSetProjectedEditor
#define ROOT_TEvePolygonSetProjectedEditor

#include "TGedFrame.h"

class TGNumberEntry;
class TGColorSelect;

class TEvePolygonSetProjected;

class TEvePolygonSetProjectedEditor : public TGedFrame
{
   TEvePolygonSetProjectedEditor(const TEvePolygonSetProjectedEditor&);            // Not implemented
   TEvePolygonSetProjectedEditor& operator=(const TEvePolygonSetProjectedEditor&); // Not implemented

protected:
   TEvePolygonSetProjected *fPS;         // Model object.

   TGNumberEntry           *fLineWidth;  // TEveLine width widget.
   TGColorSelect           *fLineColor;  // TEveLine color widget.

public:
   TEvePolygonSetProjectedEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                                 UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   ~TEvePolygonSetProjectedEditor() {}

   virtual void SetModel(TObject* obj);

   virtual void DoLineWidth();
   virtual void DoLineColor(Pixel_t color);

   ClassDef(TEvePolygonSetProjectedEditor, 0); // Editor for TEvePolygonSetProjected class.
};

#endif
