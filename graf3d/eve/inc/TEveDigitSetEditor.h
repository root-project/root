// @(#)root/eve:$Id: 2e075f81994f9a7eca182d23ab52b081eda5c617 $
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveDigitSetEditor
#define ROOT_TEveDigitSetEditor

#include "TGedFrame.h"

class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveDigitSet;

class TEveGValuator;
class TEveGDoubleValuator;
class TEveTransSubEditor;

// It would be also good to have button to change model to the palette
// object itself.
class TEveRGBAPaletteSubEditor;

class TEveDigitSetEditor : public TGedFrame
{
private:
   TEveDigitSetEditor(const TEveDigitSetEditor&);            // Not implemented
   TEveDigitSetEditor& operator=(const TEveDigitSetEditor&); // Not implemented

   void CreateInfoTab();
protected:
   TEveDigitSet             *fM;              // Model object.

   TEveRGBAPaletteSubEditor *fPalette;        // Palette sub-editor.

   TGHorizontalFrame    *fHistoButtFrame;  // Frame holding histogram display buttons.
   TGVerticalFrame      *fInfoFrame;       // Frame displaying basic digit statistics.

public:
   TEveDigitSetEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                      UInt_t options = kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveDigitSetEditor() {}

   virtual void SetModel(TObject* obj);

   // Declare callback/slot methods
   void DoHisto();
   void DoRangeHisto();
   void PlotHisto(Int_t min, Int_t max);

   ClassDef(TEveDigitSetEditor, 0); // Editor for TEveDigitSet class.
};

#endif
