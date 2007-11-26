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

#include <TGedFrame.h>

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
   TGComboBox      *fType;
   TEveGValuator   *fDistortion;
   TEveGValuator   *fFixedRadius;
   TEveGValuator   *fCurrentDepth;

   // center
   TGVerticalFrame *fCenterFrame;  // Parent frame for projection center interface.
   TGCheckButton   *fDrawCenter;
   TGCheckButton   *fDrawOrigin;
   TEveGValuator   *fCenterX;
   TEveGValuator   *fCenterY;
   TEveGValuator   *fCenterZ;

   // axis
   TGColorSelect   *fAxisColor;
   TGComboBox      *fSIMode;
   TGNumberEntry   *fSILevel;

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
