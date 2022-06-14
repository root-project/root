// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveShapeEditor
#define ROOT_TEveShapeEditor

#include "TGedFrame.h"

class TGButton;
class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveShape;

class TEveShapeEditor : public TGedFrame
{
private:
   TEveShapeEditor(const TEveShapeEditor&);            // Not implemented
   TEveShapeEditor& operator=(const TEveShapeEditor&); // Not implemented

protected:
   TEveShape        *fM; // Model object.

   TGNumberEntry    *fLineWidth;  // Line width widget.
   TGColorSelect    *fLineColor;  // Line color widget.
   TGCheckButton    *fDrawFrame;       // Checkbox for frame rendering.
   TGCheckButton    *fHighlightFrame;  // Checkbox for frame rendering.

public:
   TEveShapeEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
         UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveShapeEditor() {}

   virtual void SetModel(TObject* obj);

   void DoLineWidth();
   void DoLineColor(Pixel_t color);
   void DoDrawFrame();
   void DoHighlightFrame();

   ClassDef(TEveShapeEditor, 0); // GUI editor for TEveShape.
};

#endif
