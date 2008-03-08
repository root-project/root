// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTriangleSetEditor
#define ROOT_TEveTriangleSetEditor

#include "TGedFrame.h"

class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveTransSubEditor;
class TEveTriangleSet;

class TEveTriangleSetEditor : public TGedFrame
{
private:
   TEveTriangleSetEditor(const TEveTriangleSetEditor&);            // Not implemented
   TEveTriangleSetEditor& operator=(const TEveTriangleSetEditor&); // Not implemented

protected:
   TEveTriangleSet    *fM;        // Model object.

   TEveTransSubEditor *fTrans;  // Sub-editor of transforamtion matrix.

public:
   TEveTriangleSetEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30, UInt_t options = kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveTriangleSetEditor() {}

   virtual void SetModel(TObject* obj);

   ClassDef(TEveTriangleSetEditor, 1); // Editor for TEveTriangleSet class.
};

#endif
