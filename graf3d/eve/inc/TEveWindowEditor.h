// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveWindowEditor
#define ROOT_TEveWindowEditor

#include "TGedFrame.h"

class TGButton;
class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveWindow;

class TEveWindowEditor : public TGedFrame
{
private:
   TEveWindowEditor(const TEveWindowEditor&);            // Not implemented
   TEveWindowEditor& operator=(const TEveWindowEditor&); // Not implemented

protected:
   TEveWindow            *fM; // Model object.

   TGCheckButton         *fShowTitleBar;

public:
   TEveWindowEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30,
         UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveWindowEditor() {}

   virtual void SetModel(TObject* obj);

   void DoShowTitleBar();

   ClassDef(TEveWindowEditor, 0); // GUI editor for TEveWindow.
};

#endif
