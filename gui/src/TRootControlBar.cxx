// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   22/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootControlBar                                                      //
//                                                                      //
// This class provides an interface to the GUI dependent functions of   //
// the TControlBar class. A control bar is a horizontal or vertical bar //
// with a number of buttons (text or picture buttons).                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRootControlBar.h"
#include "TControlBar.h"
#include "TList.h"
#include "TGButton.h"


ClassImp(TRootControlBar)

//______________________________________________________________________________
TRootControlBar::TRootControlBar(TControlBar *c, const char *title, Int_t x, Int_t y)
   : TGMainFrame(gClient->GetRoot(), 10, 10), TControlBarImp(c)
{
   // Create a ROOT native GUI controlbar.

   fWidgets = 0;
   fXpos    = x;
   fYpos    = y;

   // if controlbar orientation is horizontal change layout manager
   if (c->GetOrientation() == TControlBar::kHorizontal) {
      ChangeOptions(kHorizontalFrame);
      fL1 = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 1, 1, 1, 1);
   } else
      fL1 = new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 1, 1, 1, 1);

   SetWindowName(title);
   SetIconName(title);
}

//______________________________________________________________________________
TRootControlBar::~TRootControlBar()
{
   // Delete the control bar implementation.

   if (fWidgets) fWidgets->Delete();
   delete fWidgets;
   delete fL1;
}

//______________________________________________________________________________
void TRootControlBar::Create()
{
   // Create the control bar. Loop over all buttons defined in the
   // TControlBar and create the buttons.

   fWidgets = new TList;

   TControlBarButton *button;
   TIter next(fControlBar->GetListOfButtons());

   while ((button = (TControlBarButton *) next())) {

      switch (button->GetType()) {

         case TControlBarButton::kSeparator:
            Warning("Create", "separators not yet supported");
            break;

         case TControlBarButton::kDrawnButton:
            Warning("Create", "picture buttons not yet supported");
            break;

         case TControlBarButton::kButton:
            {
               TGButton *b = new TGTextButton(this, button->GetName());
               b->SetToolTipText(button->GetTitle());
               b->SetUserData(button);
               AddFrame(b, fL1);
               fWidgets->Add(b);
            }
            break;
      }
   }

   MapSubwindows();
   Resize(GetDefaultSize());

   SetMWMHints(kMWMDecorAll | kMWMDecorResizeH,
               kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize,
               kMWMInputModeless);

   if (fXpos != -999) {
      Move(fXpos, fYpos);
      SetWMPosition(fXpos, fYpos);
   }
}

//______________________________________________________________________________
void TRootControlBar::Show()
{
   // Show controlbar. If not yet created create it first.

   if (!fWidgets) Create();

   MapRaised();
}

//______________________________________________________________________________
void TRootControlBar::Hide()
{
   // Hide controlbar.

   UnmapWindow();
}

//______________________________________________________________________________
Bool_t TRootControlBar::ProcessMessage(Long_t, Long_t, Long_t parm2)
{
   // Handle controlbar button messages.

   TControlBarButton *button = (TControlBarButton *) parm2;

   if (button) button->Action();

   return kTRUE;
}


//______________________________________________________________________________
void TRootControlBar::CloseWindow()
{
   // Called when closed via window manager action.

   delete this;
}
