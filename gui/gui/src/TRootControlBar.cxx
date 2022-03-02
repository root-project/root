// @(#)root/gui:$Id$
// Author: Fons Rademakers   22/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class  TRootControlBar
    \ingroup guiwidgets

This class provides an interface to the GUI dependent functions of
the TControlBar class. A control bar is a horizontal or vertical bar
with a number of buttons (text or picture buttons).

*/


#include "TRootControlBar.h"
#include "TControlBar.h"
#include "TList.h"
#include "TGButton.h"


ClassImp(TRootControlBar);

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI controlbar.

TRootControlBar::TRootControlBar(TControlBar *c, const char *title, Int_t x, Int_t y)
   : TGMainFrame(gClient->GetRoot(), 10, 10), TControlBarImp(c)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Delete the control bar implementation.

TRootControlBar::~TRootControlBar()
{
   delete fWidgets;
   fWidgets = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Create the control bar. Loop over all buttons defined in the
/// TControlBar and create the buttons.

void TRootControlBar::Create()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Show controlbar. If not yet created create it first.

void TRootControlBar::Show()
{
   if (!fWidgets) Create();

   MapRaised();
}

////////////////////////////////////////////////////////////////////////////////
/// Hide controlbar.

void TRootControlBar::Hide()
{
   UnmapWindow();
}

////////////////////////////////////////////////////////////////////////////////
/// Handle controlbar button messages.

Bool_t TRootControlBar::ProcessMessage(Longptr_t, Longptr_t, Longptr_t parm2)
{
   TControlBarButton *button = (TControlBarButton *) parm2;

   if (button) {
      fClicked = button;
      button->Action();
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Really delete the control bar and the this GUI.

void TRootControlBar::ReallyDelete()
{
   delete fControlBar;    // will in turn delete this object
}

////////////////////////////////////////////////////////////////////////////////
/// Called when closed via window manager action.

void TRootControlBar::CloseWindow()
{
   DeleteWindow();        // but do it slightly delayed here
}

////////////////////////////////////////////////////////////////////////////////
/// sets new font for control bar buttons

void TRootControlBar::SetFont(const char *fontName)
{
   TIter next(fWidgets);

   TObject *obj;

   while ((obj=next())) {
      if (!obj->InheritsFrom(TGTextButton::Class())) continue;

      ((TGTextButton *)obj)->SetFont(fontName);
   }
   Resize();
}

////////////////////////////////////////////////////////////////////////////////
/// sets new font for control bar buttons

void TRootControlBar::SetButtonState(const char *label, Int_t state)
{
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

////////////////////////////////////////////////////////////////////////////////
/// sets text color for control bar buttons, e.g.:
/// root > .x tutorials/demos.C
/// root > bar->SetTextColor("red")

void TRootControlBar::SetTextColor(const char *colorName)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set button width in pixels.

void TRootControlBar::SetButtonWidth(UInt_t width)
{
   fBwidth = width;
}
