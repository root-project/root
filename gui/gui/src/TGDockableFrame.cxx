// @(#)root/gui:$Id$
// Author: Abdelhalim Ssadik   07/07/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TGDockableFrame is a frame with handles that allow it to be        //
// undocked (i.e. put in a transient frame of its own) and to be docked //
// again or hidden and shown again. It uses the TGDockButton, which is  //
// a button with two vertical bars (||) and TGDockHideButton, which is  //
// a button with a small triangle. The TGUndockedFrame is a transient   //
// frame that on closure will put the frame back in the dock.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TColor.h"
#include "TGFrame.h"
#include "TMessage.h"
#include "TGWidget.h"
#include "TGButton.h"
#include "TGDockableFrame.h"
#include "TGWindow.h"
#include "TList.h"
#include "TVirtualX.h"
#include "Riostream.h"


ClassImp(TGDockButton)
ClassImp(TGDockHideButton)
ClassImp(TGUndockedFrame)
ClassImp(TGDockableFrame)

//______________________________________________________________________________
TGDockButton::TGDockButton(const TGCompositeFrame *p, int id) :
   TGButton (p, id, GetDefaultGC()(), kChildFrame)
{
   // Create a dock button (i.e. button with two vertical bars).

   fWidgetFlags = kWidgetIsEnabled;
   fMouseOn = kFALSE;
   Resize(10, GetDefaultHeight());

   fNormBg = fBackground;

   Float_t r, g, b, h, l, s;
   TColor::Pixel2RGB(fNormBg, r, g, b);
   TColor::RGB2HLS(r, g, b, h, l, s);
   l = l + (1. - l) * 45. / 100.;
   TColor::HLS2RGB(h, l, s, r, g, b);
   fHiBg = TColor::RGB2Pixel(r, g, b);

   AddInput(kEnterWindowMask | kLeaveWindowMask);
   SetWindowName();
}

//______________________________________________________________________________
TGDockButton::~TGDockButton()
{
   // Delete dock button.
}

//______________________________________________________________________________
Bool_t TGDockButton::HandleCrossing(Event_t *event)
{
   // Handle dock button crossing events.

   TGButton::HandleCrossing(event);
   if (event->fType == kLeaveNotify) {
      fMouseOn = kFALSE;
   } else if (event->fType == kEnterNotify) {
      fMouseOn = kTRUE;
   }
   if (IsEnabled())
      fClient->NeedRedraw(this);

   return kTRUE;
}

//______________________________________________________________________________
void TGDockButton::DrawBorder()
{
   // Draw borders of dock button.

   int options = GetOptions();

   if (fState == kButtonDown || fState == kButtonEngaged)
      ;
   else if (fMouseOn == kTRUE && IsEnabled()) {
      SetBackgroundColor(fHiBg);
      ChangeOptions(kChildFrame);
   } else {
      SetBackgroundColor(fNormBg);
      ChangeOptions(kChildFrame);
   }
   gVirtualX->ClearWindow(fId);
   TGFrame::DrawBorder();

   ChangeOptions(options);
}

//______________________________________________________________________________
void TGDockButton::DoRedraw()
{
   // Draw the dock button, i.e. two vertical lines.

   int x = 1, y = 0;

   DrawBorder();
   if (fState == kButtonDown || fState == kButtonEngaged) { ++x; ++y; }

   for (int i = 0; i < 5; i +=4) {
      gVirtualX->DrawLine(fId, GetHilightGC()(), i+x,   y+1, i+x,   fHeight-y-3);
      gVirtualX->DrawLine(fId, GetShadowGC()(),  i+x+1, y+1, i+x+1, fHeight-y-3);
   }
}


//______________________________________________________________________________
TGDockHideButton::TGDockHideButton(const TGCompositeFrame *p) :
   TGDockButton (p, 2)
{
   // Create a dock hide button (i.e. button with small triangle).

   Resize(10, 8);
   fAspectRatio = 0;
   SetWindowName();
}

//______________________________________________________________________________
void TGDockHideButton::DoRedraw()
{
   // Draw dock hide button.

   int x = 1, y = 0;

   DrawBorder();
   if (fState == kButtonDown || fState == kButtonEngaged) { ++x; ++y; }

   if (fAspectRatio) {
      gVirtualX->DrawLine(fId, GetBlackGC()(), x+1, y+1, x+5, y+3);
      gVirtualX->DrawLine(fId, GetBlackGC()(), x+1, y+5, x+5, y+3);
      gVirtualX->DrawLine(fId, GetHilightGC()(), x, y+1, x, y+5);
   } else {
      gVirtualX->DrawLine(fId, GetHilightGC()(), x+5, y+1, x+1, y+3);
      gVirtualX->DrawLine(fId, GetHilightGC()(), x+5, y+5, x+1, y+3);
      gVirtualX->DrawLine(fId, GetBlackGC()(), x+6, y+1, x+6, y+5);
   }
}


//______________________________________________________________________________
TGUndockedFrame::TGUndockedFrame(const TGWindow *p, TGDockableFrame *dockable) :
   TGTransientFrame(p, dockable ? dockable->GetMainFrame() : 0, 10, 10)
{
   // Create the undocked (transient) frame.

   SetWindowName("");
   fDockable = dockable;

   SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                              kMWMDecorMinimize | kMWMDecorMenu,
               kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize |
                              kMWMFuncMinimize,
               kMWMInputModeless);
   SetWindowName();
}

//______________________________________________________________________________
TGUndockedFrame::~TGUndockedFrame()
{
   // Delete undocked frame. Puts back dockable frame in its original container.

   if (fDockable && !fDockable->fDeleted) {
      fDockable->DockContainer(kFALSE);
   }
}

//______________________________________________________________________________
void TGUndockedFrame::FixSize()
{
   // Fix the size of the undocked frame so it cannot be changed via the WM.

   ChangeOptions(GetOptions() | kFixedSize);
   SetWMSize(fWidth, fHeight);
   SetWMSizeHints(fWidth, fHeight, fWidth, fHeight, 0, 0);
}

//______________________________________________________________________________
void TGUndockedFrame::CloseWindow()
{
   // Close undocked frame (called via WM close button).

   DeleteWindow();
}


//______________________________________________________________________________
TGDockableFrame::TGDockableFrame(const TGWindow *p, int id, UInt_t /*options*/)
   : TGCompositeFrame(p, 10, 10, kHorizontalFrame), TGWidget(id)
{
   // Create a dockable frame widget.

   fMsgWindow = fParent;

   TGLayoutHints *l1 = new TGLayoutHints(kLHintsTop | kLHintsLeft);
   TGLayoutHints *l2 = new TGLayoutHints(kLHintsExpandY | kLHintsLeft);
   fLb = new TGLayoutHints(kLHintsExpandY | kLHintsLeft, 0, 2, 0, 0);
   fLc = new TGLayoutHints(kLHintsExpandY | kLHintsExpandX);

   fButtons = new TGCompositeFrame(this, 10, 10, kVerticalFrame);
   fButtons->SetCleanup();
   fHideButton = new TGDockHideButton(fButtons);
   fButtons->AddFrame(fHideButton, l1);
   fDockButton = new TGDockButton(fButtons);
   fButtons->AddFrame(fDockButton, l2);

   TGCompositeFrame::AddFrame(fButtons, fLb);

   fContainer = new TGCompositeFrame(this, 10, 10);
   
   TGCompositeFrame::AddFrame(fContainer, fLc);

   fEnableHide   = kTRUE;
   fEnableUndock = kTRUE;
   fHidden       = kFALSE;
   fFrame        = 0;
   fDeleted      = kFALSE;
   fFixedSize    = kTRUE;

   fDockButton->Associate(this);
   fHideButton->Associate(this);

   MapSubwindows();
   Resize(GetDefaultSize());
   TGFrame::SetWindowName();
}

//______________________________________________________________________________
TGDockableFrame::~TGDockableFrame()
{
   // Cleanup dockable frame.

   // Just set the flag and delete fFrame. The other components
   // are deleted in TGCompositeFrame destructor.
   if (fFrame) {
      fDeleted = kTRUE;
      delete fFrame;
   }
}

//______________________________________________________________________________
void TGDockableFrame::AddFrame(TGFrame *f, TGLayoutHints *hints)
{
   // Add frame to dockable frame container. Frame and hints are NOT adopted.

   f->ReparentWindow(fContainer);
   fContainer->AddFrame(f, fHints = hints);
   fContainer->Layout();
}

//______________________________________________________________________________
void TGDockableFrame::UndockContainer()
{
   // Undock container.

   int ax, ay;
   Window_t wdummy;

   if (fFrame || !fEnableUndock) return;

   fFrame = new TGUndockedFrame(fClient->GetDefaultRoot(), this);
   fFrame->SetEditDisabled();

   TGDimension size = fContainer->GetSize();
   RemoveFrame(fContainer);
   fContainer->ReparentWindow(fFrame);
   fFrame->AddFrame(fContainer, new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));

   gVirtualX->TranslateCoordinates(GetId(), fClient->GetDefaultRoot()->GetId(), fX,
                                   fY + fFrame->GetHeight(), ax, ay, wdummy);

   if (fDockName) fFrame->SetWindowName(fDockName);

   fFrame->MapSubwindows();
   fFrame->Resize(size);
   if (fFixedSize)
      fFrame->FixSize();
   fFrame->MapWindow();
   fFrame->Move(ax, ay);

   if (((TGFrame *)fParent)->IsComposite())           // paranoia check
      ((TGCompositeFrame *)fParent)->HideFrame(this);

   Layout();

   SendMessage(fMsgWindow, MK_MSG(kC_DOCK, kDOCK_UNDOCK), fWidgetId, 0);
   Undocked();
}

//______________________________________________________________________________
void TGDockableFrame::DockContainer(Int_t del)
{
   // Dock container back to TGDockableFrame.

   if (!fFrame) return;
   if (del) {
      delete fFrame;  // this will call DockContainer again with del = kFALSE
      return;
   }

   fFrame->RemoveFrame(fContainer);
   fContainer->ReparentWindow(this);
   TGCompositeFrame::AddFrame(fContainer, new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));

   // kludge! (for special case)
   fDockButton->Resize(fDockButton->GetDefaultWidth(), 1);

   Layout();
   if (((TGFrame *)fParent)->IsComposite())           // paranoia check
      ((TGCompositeFrame *)fParent)->ShowFrame(this);

   // fFrame is just being deleted (we're here called by TGUndockedFrame's
   // destructor) so just set it NULL below to avoid eventual problems in
   // TGDockableFrame's destructor.

   fFrame = 0;

   SendMessage(fMsgWindow, MK_MSG(kC_DOCK, kDOCK_DOCK), fWidgetId, 0);
   Docked();
}

//______________________________________________________________________________
void TGDockableFrame::ShowContainer()
{
   // Show dock container.

   if (!fHidden) return;

   ShowFrame(fContainer);
   if (fEnableUndock) fButtons->ShowFrame(fDockButton);
   fHideButton->SetAspectRatio(0);
   if (((TGFrame *)fParent)->IsComposite())           // paranoia check
      ((TGCompositeFrame *)fParent)->Layout();
   fHidden = kFALSE;

   SendMessage(fMsgWindow, MK_MSG(kC_DOCK, kDOCK_SHOW), fWidgetId, 0);
}

//______________________________________________________________________________
void TGDockableFrame::HideContainer()
{
   // Hide dock container.

   if (fHidden || !fEnableHide) return;

   HideFrame(fContainer);
   fButtons->HideFrame(fDockButton);
   fHideButton->SetAspectRatio(1);
   if (((TGFrame *)fParent)->IsComposite())           // paranoia check
      ((TGCompositeFrame *)fParent)->Layout();
   fHidden = kTRUE;

   SendMessage(fMsgWindow, MK_MSG(kC_DOCK, kDOCK_HIDE),fWidgetId, 0);
}

//______________________________________________________________________________
Bool_t TGDockableFrame::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process dockable frame messages.

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch (parm1) {
                  case 1:
                     if (!fHidden) UndockContainer();
                     break;
                  case 2:
                     if (!fHidden)
                        HideContainer();
                     else
                        ShowContainer();
                     break;
               }
               break;
         }
         break;
   }

   return kTRUE;
}

//______________________________________________________________________________
void TGDockableFrame::EnableUndock(Bool_t onoff)
{
   // Enable undocking.

   fEnableUndock = onoff;
   if (onoff)
      fButtons->ShowFrame(fDockButton);
   else
      fButtons->HideFrame(fDockButton);
   Layout();
}

//______________________________________________________________________________
void TGDockableFrame::EnableHide(Bool_t onoff)
{
   // Enable hiding.

   fEnableHide = onoff;
   if (onoff)
      fButtons->ShowFrame(fHideButton);
   else
      fButtons->HideFrame(fHideButton);
   Layout();
}

//______________________________________________________________________________
void TGDockableFrame::SetWindowName(const char *name)
{
   // Set window name so it appear as title of the undock window.

   fDockName = "";
   if (name) {
      fDockName = name;
      if (fFrame) fFrame->SetWindowName(fDockName);
   }
}

//______________________________________________________________________________
void TGDockableFrame::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a dockable frame widget as a C++ statement(s) on output stream out.

   char quote = '"';

   out << endl << "   // dockable frame" << endl;
   out << "   TGDockableFrame *";
   out << GetName()<<" = new TGDockableFrame(" << fParent->GetName();

   if (GetOptions() == kHorizontalFrame) {
      if (fWidgetId == -1) {
         out << ");" << endl;
      } else {
         out << "," << fWidgetId << ");" << endl;
      }
   } else {
      out << "," << fWidgetId << "," << GetOptionString() << ");" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   if (GetContainer()->GetList()->First()) {
      out << "   TGCompositeFrame *" << GetContainer()->GetName() << " = "
          << GetName() << "->GetContainer();" << endl;

      TGFrameElement *el;
      TIter next(GetContainer()->GetList());

      while ((el = (TGFrameElement *) next())) {
         el->fFrame->SavePrimitive(out, option);
         out << "   " << GetName() << "->AddFrame(" << el->fFrame->GetName();
         el->fLayout->SavePrimitive(out, option);
         out << ");"<< endl;
      }
   }
   out << endl << "   // next lines belong to the dockable frame widget" << endl;
   if (EnableUndock())
      out << "   " << GetName() << "->EnableUndock(kTRUE);" << endl;
   else
      out << "   " << GetName() << "->EnableUndock(kFALSE);" << endl;

   if (EnableHide())
      out << "   " << GetName() << "->EnableHide(kTRUE);" << endl;
   else
      out << "   " << GetName() << "->EnableHide(kFALSE);" << endl;

   if (fDockName != "")
      out << "   " << GetName() << "->SetWindowName(" << quote << fDockName
          << quote << ");" << endl;

   if (IsUndocked())
      out << "   " << GetName() << "->UndockContainer();" << endl;
   else
      out << "   " << GetName() << "->DockContainer();" << endl;

   if (IsHidden())
      out << "   " << GetName() << "->HideContainer();" << endl;

   out << endl;
}
