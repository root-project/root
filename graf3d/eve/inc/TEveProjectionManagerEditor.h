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
   TEveProjectionManager    *fM;   // Model object.

   TGComboBox      *fType;         // TEveProjection type widget
   TEveGValuator   *fDistortion;   // TEveProjection distortion widget
   TEveGValuator   *fFixR;         // TEveProjection fixed-radius widget
   TEveGValuator   *fFixZ;         // TEveProjection fixed-z widget
   TEveGValuator   *fPastFixRFac;  // TEveProjection relative scale after FixR
   TEveGValuator   *fPastFixZFac;  // TEveProjection relative scale after FixZ
   TEveGValuator   *fCurrentDepth; // TEveProjection z-coordinate widget
   TEveGValuator   *fMaxTrackStep;  // TEveProjection relative scale after FixZ

   TGVerticalFrame *fCenterFrame;  // parent frame for distortion center
   TEveGValuator   *fCenterX;      // center x value widget
   TEveGValuator   *fCenterY;      // center y value widget
   TEveGValuator   *fCenterZ;      // center z value widget

public:
   TEveProjectionManagerEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30, UInt_t options = kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveProjectionManagerEditor(){}

   virtual void SetModel(TObject* obj);

   // Declare callback/slot methods

   void DoType(Int_t type);
   void DoDistortion();
   void DoFixR();
   void DoFixZ();
   void DoPastFixRFac();
   void DoPastFixZFac();
   void DoCurrentDepth();
   void DoMaxTrackStep();
   void DoCenter();

   ClassDef(TEveProjectionManagerEditor, 0); // Editor for TEveProjectionManager class.
};

#endif
