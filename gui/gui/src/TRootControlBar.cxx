// @(#)root/gui:$Id$
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
   fBwidth  = 0;
   fClicked = 0;
   SetCleanup(kDeepCleanup);

   // if controlbar orientation is horizontal change layout manager
   if (c && c->GetOrientation() == TControlBar::kHorizontal) {
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

   delete fWidgets;
   fWidgets = 0;
}

//______________________________________________________________________________
void TRootControlBar::Create()
{
   // Create the control bar. Loop over all buttons defined in the
   // TControlBar and create the buttons.

   fWidgets = new TList;
   TGButton *b = 0;

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
               b = new TGTextButton(this, button->GetName());
               b->SetToolTipText(button->GetTitle());
               b->SetUserData(button);
               AddFrame(b, fL1);
               fWidgets->Add(b);
               if (fBwidth < b->GetDefaultWidth())
                  fBwidth = b->GetDefaultWidth();  //do not cut the label
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
   if (GetOptions() & kHorizontalFrame)
      SetWMSize(fBwidth*fWidgets->GetSize(), GetHeight());
   else
      SetWMSize(fBwidth, GetHeight());
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

   if (button) {
      fClicked = button;
      button->Action();
   }
   return kTRUE;
}

//______________________________________________________________________________
void TRootControlBar::ReallyDelete()
{
   // Really delete the control bar and the this GUI.

   delete fControlBar;    // will in turn delete this object
}

//______________________________________________________________________________
void TRootControlBar::CloseWindow()
{
   // Called when closed via window manager action.

   DeleteWindow();        // but do it slightly delayed here
}

//______________________________________________________________________________
void TRootControlBar::SetFont(const char *fontName)
{
   // sets new font for control bar buttons

   TIter next(fWidgets);

   TObject *obj;

   while ((obj=next())) {
      if (!obj->InheritsFrom(TGTextButton::Class())) continue;

      ((TGTextButton *)obj)->SetFont(fontName);
   }
   Resize();
}

//______________________________________________________________________________
void TRootControlBar::SetButtonState(const char *label, Int_t state)
{
   // sets new font for control bar buttons

   TIter next(fWidgets);

   TObject *obj;

   while ((obj=next())) {
      if (!obj->InheritsFrom(TGTextButton::Class())) continue;

      if (!strcmp(((TGTextButton *)obj)->GetTitle(), label)) {
         switch (state) {
            case 0: {
               ((TGTextButton *)obj)->SetState(kButtonUp);
               break;
            }
            case 1: {
               ((TGTextButton *)obj)->SetState(kButtonDown);
               break;
            }
            case 2: {
               ((TGTextButton *)obj)->SetState(kButtonEngaged);
               break;
            }
            case 3: {
               ((TGTextButton *)obj)->SetState(kButtonDisabled);
               break;
            }
            default: {
               Error("SetButtonState", "not valid button state (expecting 0, 1, 2 or 3)");
               break;
            }
         }
      }
   }
   Resize();
}

//______________________________________________________________________________
void TRootControlBar::SetTextColor(const char *colorName)
{
   // sets text color for control bar buttons, e.g.:
   // root > .x tutorials/demos.C
   // root > bar->SetTextColor("red")

   Pixel_t color;
   gClient->GetColorByName(colorName, color);

   if (!fWidgets) Create();

   TIter next(fWidgets);

   TObject *obj;

   while ((obj=next())) {
      if (!obj->InheritsFrom(TGTextButton::Class())) continue;

      ((TGTextButton *)obj)->SetTextColor(color);
   }
   Resize();
}

//______________________________________________________________________________
void TRootControlBar::SetButtonWidth(UInt_t width)
{
   // Set button width in pixels.

   fBwidth = width;
}
