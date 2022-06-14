// @(#)root/eve:$Id$
// Author: Matevz Tadel, Jochen Thaeder 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveJetConeEditor
#define ROOT_TEveJetConeEditor

#include "TGedFrame.h"

class TGButton;
class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveJetCone;

class TEveJetConeEditor : public TGedFrame
{
private:
   TEveJetConeEditor(const TEveJetConeEditor&);            // Not implemented
   TEveJetConeEditor& operator=(const TEveJetConeEditor&); // Not implemented

protected:
   TEveJetCone            *fM; // Model object.

   // Declare widgets
   // TGSomeWidget*   fXYZZ;

public:
   TEveJetConeEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                     UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveJetConeEditor() {}

   virtual void SetModel(TObject* obj);

   // Declare callback/slot methods
   // void DoXYZZ();

   ClassDef(TEveJetConeEditor, 0); // GUI editor for TEveJetCone.
};

#endif
