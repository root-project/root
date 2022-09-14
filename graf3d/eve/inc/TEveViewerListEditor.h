// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveViewerListEditor
#define ROOT_TEveViewerListEditor

#include "TGedFrame.h"

class TEveGValuator;
class TEveViewerList;

class TEveViewerListEditor : public TGedFrame
{
private:
   TEveViewerListEditor(const TEveViewerListEditor&);            // Not implemented
   TEveViewerListEditor& operator=(const TEveViewerListEditor&); // Not implemented

protected:
   TEveViewerList            *fM; // Model object.

   TEveGValuator             *fBrightness;
   TGTextButton              *fColorSet;

public:
   TEveViewerListEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30,
         UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveViewerListEditor() {}

   virtual void SetModel(TObject* obj);

   // Declare callback/slot methods
   void DoBrightness();
   void SwitchColorSet();

   ClassDef(TEveViewerListEditor, 0); // GUI editor for TEveViewerList.
};

#endif
