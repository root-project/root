// @(#)root/guibuilder:$Id$
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGuiBldHintsButton
#define ROOT_TGuiBldHintsButton


#include "TGButton.h"


//////////////////////////////////////////////////////////////////////////
class TGuiBldHintsButton : public TGButton {

protected:
   virtual void DrawExpandX();
   virtual void DrawExpandY();
   virtual void DrawCenterX();
   virtual void DrawCenterY();
   virtual void DrawTopLeft();
   virtual void DrawTopRight();
   virtual void DrawBottomLeft();
   virtual void DrawBottomRight();

   void DoRedraw() override;

public:
   TGuiBldHintsButton(const TGWindow *p, Int_t id);
   ~TGuiBldHintsButton() override {}

   ClassDefOverride(TGuiBldHintsButton,0) //Button for editing layout hints in GUI Builder
};

#endif
