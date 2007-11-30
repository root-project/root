// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveProjectionManagerEditor
#define ROOT_TEveProjectionManagerEditor

#include "TGedFrame.h"

class TGComboBox;
class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveProjectionManager;
class TEveGValuator;

class TEveProjectionManagerEditor : public TGedFrame
{
private:
   TEveProjectionManagerEditor(const TEveProjectionManagerEditor&);            // Not implemented
   TEveProjectionManagerEditor& operator=(const TEveProjectionManagerEditor&); // Not implemented

protected:
   TEveProjectionManager    *fM; // Model object.

   // projection
   TGComboBox      *fType;         // TEveProjection type widget
   TEveGValuator   *fDistortion;   // TEveProjection distortion widget
   TEveGValuator   *fFixedRadius;  // TEveProjection fixed-radius widget
   TEveGValuator   *fCurrentDepth; // TEveProjection z-coordinate widget

   // center
   TGVerticalFrame *fCenterFrame;  // Parent frame for Center tab.
   TGCheckButton   *fDrawCenter;   // draw center widget
   TGCheckButton   *fDrawOrigin;   // draw origin widget
   TEveGValuator   *fCenterX;      // center x value widget
   TEveGValuator   *fCenterY;      // center y value widget
   TEveGValuator   *fCenterZ;      // center z value widget

   // axis
   TGColorSelect   *fAxisColor;  // color of axis widget
   TGComboBox      *fSIMode;     // tick-mark positioning widget 
   TGNumberEntry   *fSILevel;    // tick-mark density widget

public:
   TEveProjectionManagerEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30, UInt_t options = kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveProjectionManagerEditor(){}

   virtual void SetModel(TObject* obj);

   // Declare callback/slot methods

   void DoSplitInfoMode(Int_t type);
   void DoSplitInfoLevel();
   void DoAxisColor(Pixel_t pixel);

   void DoType(Int_t type);
   void DoDistortion();
   void DoFixedRadius();
   void DoCurrentDepth();
   void DoDrawCenter();
   void DoDrawOrigin();
   void DoCenter();

   ClassDef(TEveProjectionManagerEditor, 0); // Editor for TEveProjectionManager class.
};

#endif
