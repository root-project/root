// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveGridStepperEditor
#define ROOT_TEveGridStepperEditor

#include "TGedFrame.h"

class TGButton;
class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveGridStepper;
class TEveGValuator;

class TEveGridStepperSubEditor : public TGVerticalFrame
{
private:
   TEveGridStepperSubEditor(const TEveGridStepperSubEditor&);            // Not implemented
   TEveGridStepperSubEditor& operator=(const TEveGridStepperSubEditor&); // Not implemented

protected:
   TEveGridStepper  *fM;    // Model object.

   TEveGValuator    *fNx;   // Number of slots along x.
   TEveGValuator    *fNy;   // Number of slots along y.
   TEveGValuator    *fNz;   // Number of slots along z.
   TEveGValuator    *fDx;   // Step in the x direction.
   TEveGValuator    *fDy;   // Step in the y direction.
   TEveGValuator    *fDz;   // Step in the z direction.

public:
   TEveGridStepperSubEditor(const TGWindow* p);
   virtual ~TEveGridStepperSubEditor() {}

   void SetModel(TEveGridStepper* m);

   void Changed(); //*SIGNAL*

   void DoNs();
   void DoDs();

   ClassDef(TEveGridStepperSubEditor, 0); // Sub-editor for TEveGridStepper class.
};


class TEveGridStepperEditor : public TGedFrame
{
private:
   TEveGridStepperEditor(const TEveGridStepperEditor&);            // Not implemented
   TEveGridStepperEditor& operator=(const TEveGridStepperEditor&); // Not implemented

protected:
   TEveGridStepper            *fM;   // Model object.
   TEveGridStepperSubEditor   *fSE;  // Sub-editor containg GUI controls.

public:
   TEveGridStepperEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30, UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveGridStepperEditor() {}

   virtual void SetModel(TObject* obj);

   ClassDef(TEveGridStepperEditor, 0); // Editor for TEveGridStepper class.
};

#endif
