// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveProjectionAxesEditor
#define ROOT_TEveProjectionAxesEditor

#include "TGedFrame.h"

class TGCheckButton;
class TGComboBox;
class TGNumberEntry;

class TEveProjectionAxes;

class TEveProjectionAxesEditor : public TGedFrame
{
private:
   TEveProjectionAxesEditor(const TEveProjectionAxesEditor&);            // Not implemented
   TEveProjectionAxesEditor& operator=(const TEveProjectionAxesEditor&); // Not implemented

protected:
   TEveProjectionAxes   *fM;       // Model object.
  
   TGComboBox      *fSplitMode;    // tick-mark positioning widget 
   TGNumberEntry   *fSplitLevel;   // tick-mark density widget

   TGVerticalFrame *fCenterFrame;  // Parent frame for Center tab.
   TGCheckButton   *fDrawCenter;   // draw center widget
   TGCheckButton   *fDrawOrigin;   // draw origin widget

public:
   TEveProjectionAxesEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                            UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveProjectionAxesEditor() {}

   virtual void SetModel(TObject* obj);

   // Declare callback/slot methods

   void DoSplitMode(Int_t type);
   void DoSplitLevel();
   void DoDrawCenter();
   void DoDrawOrigin();

   ClassDef(TEveProjectionAxesEditor, 0); // GUI editor for TEveProjectionAxes.
};

#endif
