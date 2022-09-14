// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveLineEditor
#define ROOT_TEveLineEditor

#include "TGedFrame.h"

class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveLine;

class TEveLineEditor : public TGedFrame
{
private:
   TEveLineEditor(const TEveLineEditor&);            // Not implemented
   TEveLineEditor& operator=(const TEveLineEditor&); // Not implemented

protected:
   TEveLine          *fM;          // Model object.

   TGCheckButton     *fRnrLine;    // Checkbox for line-rendering.
   TGCheckButton     *fRnrPoints;  // Checkbox for point-rendering.
   TGCheckButton     *fSmooth;     // Checkbox for line smoothing.

public:
   TEveLineEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30, UInt_t options = kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveLineEditor() {}

   virtual void SetModel(TObject* obj);

   void DoRnrLine();
   void DoRnrPoints();
   void DoSmooth();

   ClassDef(TEveLineEditor, 0); // Editor for TEveLine class.
};

#endif
