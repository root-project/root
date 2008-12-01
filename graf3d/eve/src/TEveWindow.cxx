// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveWindow.h"
#include "TEveManager.h"
#include "TEveSelection.h"

#include "TContextMenu.h"

#include "TGButton.h"
#include "TContextMenu.h"
#include "TGMenu.h"
#include "TGPack.h"
#include "TGTab.h"

#include <cassert>

//==============================================================================
//==============================================================================
// CompositeFrame classes - slots for TEveWindows
//==============================================================================
//==============================================================================


//==============================================================================
// TEveCompositeFrame
//==============================================================================

//______________________________________________________________________________
// 
// Base-class for EVE window slots.
//

ClassImp(TEveCompositeFrame);

TContextMenu* TEveCompositeFrame::fgCtxMenu = 0;

//______________________________________________________________________________
TEveCompositeFrame::TEveCompositeFrame(TGCompositeFrame* parent,
                                       TEveWindow*   eve_parent) :
   TGCompositeFrame (parent, 0, 0, kVerticalFrame),

   fTopFrame    (0),
   fToggleBar   (0),
   fTitleBar    (0),
   fIconBar     (0),
   fEveWindowLH (0),

   fMiniBar     (0),

   fEveParentWindow (eve_parent),
   fEveWindow       (0)
{
   // Constructor.

   static const UInt_t topH = 14, miniH = 4;

   // --- TopFrame

   fTopFrame = new TGHorizontalFrame(this, 20, topH);

   fToggleBar = new TGTextButton(fTopFrame, "Hide");
   fToggleBar->ChangeOptions(kRaisedFrame);
   fToggleBar->Resize(40, topH);
   fTopFrame->AddFrame(fToggleBar, new TGLayoutHints(kLHintsNormal, 0,0,0,0));//1,1,1,1));

   fTitleBar = new TGTextButton(fTopFrame, "Title Bar");
   fTitleBar->ChangeOptions(kRaisedFrame);
   fTitleBar->Resize(40, topH);
   fTopFrame->AddFrame(fTitleBar, new TGLayoutHints(kLHintsNormal | kLHintsExpandX,  0,0,0,0));//1,1,1,1));

   fIconBar = new TGTextButton(fTopFrame, "Actions");
   fIconBar->ChangeOptions(kRaisedFrame);
   fIconBar->Resize(40, topH);
   fTopFrame->AddFrame(fIconBar, new TGLayoutHints(kLHintsNormal,  0,0,0,0));//1,1,1,1));

   AddFrame(fTopFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX));

   fToggleBar->Connect("Clicked()", "TEveCompositeFrame", this, "FlipTitleBarState()");
   fTitleBar ->Connect("Clicked()", "TEveCompositeFrame", this, "TitleBarClicked()");
   fIconBar  ->Connect("Pressed()", "TEveCompositeFrame", this, "ActionPressed()");

   // --- MiniBar

   fMiniBar = new TGButton(this);
   fMiniBar->ChangeOptions(kRaisedFrame | kFixedHeight);
   fMiniBar->Resize(20, miniH);
   fMiniBar->SetBackgroundColor(TEveWindow::fgMiniBarBackgroundColor);
   AddFrame(fMiniBar, new TGLayoutHints(kLHintsNormal | kLHintsExpandX));

   fMiniBar->Connect("Clicked()", "TEveCompositeFrame", this, "FlipTitleBarState()");

   // --- Common settings.

   SetCleanup(kDeepCleanup);

   MapSubwindows();
   HideFrame(fMiniBar);

   // Layout for embedded windows.
   fEveWindowLH = new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY);
}

//______________________________________________________________________________
TEveCompositeFrame::~TEveCompositeFrame()
{
   // If fEveWindow != 0 we are being deleted from the ROOT GUI side.
   // Relinquishe EveWindow and ref-counting should do the rest.

   if (fEveWindow != 0)
   {
      printf("TEveCompositeFrame::~TEveCompositeFrame - EveWindow not null '%s'.\n",
             fEveWindow->GetElementName());
      fEveWindow->ClearEveFrame();
      RelinquishEveWindow();
   }

   delete fEveWindowLH;
}

//______________________________________________________________________________
void TEveCompositeFrame::Destroy()
{
   // Virtual function called from eve side when the frame should be
   // destroyed. This means we expect that fEveWindow is null.
   //
   // See implementations in TEveCompositeFrameInPack and
   // TEveCompositeFrameInTab.
}

//==============================================================================

void TEveCompositeFrame::ActionPressed()
{
   if (fgCtxMenu == 0) {
      fgCtxMenu = new TContextMenu("", "");
   }

   Int_t    x, y;
   UInt_t   w, h;
   Window_t childdum;
   gVirtualX->GetWindowSize(fIconBar->GetId(), x, y, w, h);
   gVirtualX->TranslateCoordinates(fIconBar->GetId(),
                                   gClient->GetDefaultRoot()->GetId(),
                                   0, 0, x, y, childdum);

   fgCtxMenu->Popup(x - 2, y + h - 2, fEveWindow);
}

//______________________________________________________________________________
void TEveCompositeFrame::FlipTitleBarState()
{
   fEveWindow->FlipShowTitleBar();
}

//______________________________________________________________________________
void TEveCompositeFrame::TitleBarClicked()
{
   fEveWindow->TitleBarClicked();
}

//==============================================================================

//______________________________________________________________________________
void TEveCompositeFrame::AcquireEveWindow(TEveWindow* ew)
{
   // Accept window and increase its deny-destroy count.
   // Window's gui-frame is embedded and mapped.
   // Layout is not called.
   //
   // Throws an exception if a window is already embedded or if 0 is
   // passed as an argument.

   // Replace current eve-window with the given one.
   // Current GUI window is unmapped, removed and reparented to default-root.
   // New GUI window is reparented to this, added and mapped.

   static const TEveException eh("TEveCompositeFrame::AcquireEveWindow ");

   if (fEveWindow)
      throw eh + "Window already set.";

   if (ew == 0)
      throw eh + "Called with 0 argument.";

   fEveWindow = ew;

   fEveWindow->IncDenyDestroy();
   TGFrame* gui_frame = fEveWindow->GetGUIFrame();
   gui_frame->ReparentWindow(this);
   AddFrame(gui_frame, fEveWindowLH);
   gui_frame->MapWindow();

   SetCurrent(fEveWindow->IsCurrent());
   SetShowTitleBar(fEveWindow->GetShowTitleBar());
}

//______________________________________________________________________________
TEveWindow* TEveCompositeFrame::RelinquishEveWindow()
{
   // Remove window and decrease its deny-destroy count.
   // Window's gui-frame is unmapped, removed and reparented to default-root.

   TEveWindow* ex_ew = fEveWindow;

   if (fEveWindow)
   {
      TGFrame* gui_frame = fEveWindow->GetGUIFrame();
      gui_frame->UnmapWindow();
      RemoveFrame(gui_frame);
      gui_frame->ReparentWindow(fClient->GetDefaultRoot());
      fEveWindow->DecDenyDestroy();
      fEveWindow = 0;
   }

   return ex_ew;
}

//______________________________________________________________________________
TEveWindow* TEveCompositeFrame::ChangeEveWindow(TEveWindow* ew)
{
   // Replace current eve-window with the given one.
   // Current GUI window is unmapped, removed and reparented to default-root.
   // New GUI window is reparented to this, added and mapped.

   TEveWindow* ex_ew = RelinquishEveWindow();

   AcquireEveWindow(ew);

   return ex_ew;
}

//______________________________________________________________________________
void TEveCompositeFrame::SetCurrent(Bool_t curr)
{
   if (curr) {
      fTitleBar->SetBackgroundColor(TEveWindow::fgCurrentBackgroundColor);
   } else {
      fTitleBar->SetBackgroundColor(GetDefaultFrameBackground());
   }
   fClient->NeedRedraw(fTitleBar);
}

//______________________________________________________________________________
void TEveCompositeFrame::SetShowTitleBar(Bool_t show)
{
   if (show) {
      HideFrame(fMiniBar);
      ShowFrame(fTopFrame);
   } else {
      HideFrame(fTopFrame);
      ShowFrame(fMiniBar);
   }
}


//==============================================================================
// TEveCompositeFrameInPack
//==============================================================================

//______________________________________________________________________________
//
// An EVE window-slot contained within one tab of a TGTab.

ClassImp(TEveCompositeFrameInPack);

//______________________________________________________________________________
TEveCompositeFrameInPack::TEveCompositeFrameInPack(TGCompositeFrame* parent,
                                                   TEveWindow* eve_parent,
                                                   TGPack*     pack) :
   TEveCompositeFrame(parent, eve_parent),
   fPack (pack)
{
   // Constructor.
}

//______________________________________________________________________________
TEveCompositeFrameInPack::~TEveCompositeFrameInPack()
{
   // Destructor.
}

//______________________________________________________________________________
void TEveCompositeFrameInPack::Destroy()
{
   printf("TEveCompositeFrameInPack::Destroy()\n");

   assert(fEveWindow == 0);

   fPack->RemoveFrame(this);
   delete this;
}

//==============================================================================
// TEveCompositeFrameInTab
//==============================================================================

//______________________________________________________________________________
//
// An EVE window-slot contained within one tab of a TGTab.

ClassImp(TEveCompositeFrameInTab);

//______________________________________________________________________________
TEveCompositeFrameInTab::TEveCompositeFrameInTab(TGCompositeFrame* parent,
                                                 TEveWindow* eve_parent,
                                                 TGTab*      tab) :
   TEveCompositeFrame(parent, eve_parent),
   fTab         (tab),
   fParentInTab (parent)
{
   // Constructor.
}

//______________________________________________________________________________
TEveCompositeFrameInTab::~TEveCompositeFrameInTab()
{
   // Destructor.
}

//______________________________________________________________________________
Int_t TEveCompositeFrameInTab::FindTabIndex()
{
   // Return index of this frame in the tab.
   // Throws an exception if it is not found.

   static const TEveException eh("TEveCompositeFrameInTab::FindTabIndex ");

   Int_t nt = fTab->GetNumberOfTabs();
   for (Int_t t = 0; t < nt; ++t)
   {
      if (fTab->GetTabContainer(t) == fParentInTab)
      {
         return t;
      }
   }

   throw eh + "parent frame not found in tab.";
}

//______________________________________________________________________________
void TEveCompositeFrameInTab::Destroy()
{
   printf("TEveCompositeFrameInTab::Destroy()\n");

   assert (fEveWindow == 0);

   Int_t t = FindTabIndex();

   // disconnect form Removed() if / when connected
   fTab->RemoveTab(t, kFALSE);
   fParentInTab->DestroyWindow();
   fParentInTab->SetCleanup(kNoCleanup);
   delete fParentInTab;
   delete this;
}

//______________________________________________________________________________
void TEveCompositeFrameInTab::AcquireEveWindow(TEveWindow* ew)
{
   TEveCompositeFrame::AcquireEveWindow(ew);

   Int_t t = FindTabIndex();
   fTab->GetTabTab(t)->SetText(new TGString(fEveWindow->GetElementName()));
   fTab->Layout();
}
//______________________________________________________________________________
TEveWindow* TEveCompositeFrameInTab::RelinquishEveWindow()
{
   Int_t t = FindTabIndex();
   fTab->GetTabTab(t)->SetText(new TGString("<relinquished>"));
   fTab->Layout();

   return TEveCompositeFrame::RelinquishEveWindow();
}

//______________________________________________________________________________
void TEveCompositeFrameInTab::SetCurrent(Bool_t curr)
{
   TEveCompositeFrame::SetCurrent(curr);

   Int_t t = FindTabIndex();
   TGTabElement* te = fTab->GetTabTab(t);
   if (curr) {
      te->SetBackgroundColor(TEveWindow::fgCurrentBackgroundColor);
   } else {
      te->SetBackgroundColor(GetDefaultFrameBackground());
   }
   fClient->NeedRedraw(te);
}


//==============================================================================
//==============================================================================
// TEveWindow classes
//==============================================================================
//==============================================================================


//==============================================================================
// TEveWindow
//==============================================================================

//______________________________________________________________________________
// Description of TEveWindow
//

ClassImp(TEveWindow);

TEveWindow* TEveWindow::fgCurrentWindow = 0;
Pixel_t     TEveWindow::fgCurrentBackgroundColor = 0x80A0C0;
Pixel_t     TEveWindow::fgMiniBarBackgroundColor = 0x80C0A0;

//______________________________________________________________________________
TEveWindow::TEveWindow(const Text_t* n, const Text_t* t) :
   TEveElementList(n, t),

   fEveFrame     (0),
   fShowTitleBar (kTRUE)
{
   // Constructor.

   // Override from TEveElementList.
   fChildClass = TEveWindow::Class();
}

//______________________________________________________________________________
TEveWindow::~TEveWindow()
{
   if (this == fgCurrentWindow)
      fgCurrentWindow = 0;

   printf("TEveWindow::~TEveWindow  '%s' '%s', cnt=%d\n", GetElementName(), ClassName(), fDenyDestroy);
}

//==============================================================================

//______________________________________________________________________________
void TEveWindow::SwapWindow(TEveWindow* w)
{
   printf ("Swapping ... yeah, right :)\n");
}

//______________________________________________________________________________
void TEveWindow::SwapWindowWithCurrent()
{
   static const TEveException eh("TEveWindow::SwapWindowWithCurrent ");

   if (fgCurrentWindow == 0)
      throw eh + "Current eve-window is not set.";

   if (fgCurrentWindow == this)
      throw eh + "This is the current window ... nothing changed.";

   SwapWindow(fgCurrentWindow);
}

//______________________________________________________________________________
void TEveWindow::DestroyWindow()
{
   printf("TEveWindow::DestroyWindow '%s' '%s', cnt=%d\n", GetElementName(), ClassName(), fDenyDestroy);

   if (fEveFrame != 0 && fDenyDestroy == 1)
   {
      TEveWindowSlot* ew_slot = TEveWindow::CreateDefaultWindowSlot();
      Bool_t dozrc = fDestroyOnZeroRefCnt;
      fDestroyOnZeroRefCnt = kFALSE;
      ew_slot->PopulateSlot(fEveFrame);
      fDestroyOnZeroRefCnt = dozrc;
      fEveFrame = 0;
   }

   TEveElementList::Destroy();
}

//______________________________________________________________________________
void TEveWindow::DestroyWindowAndSlot()
{
   printf("TEveWindow::DestroyWindowAndSlot '%s' '%s', cnt=%d\n", GetElementName(), ClassName(), fDenyDestroy);

   if (fEveFrame != 0 && fDenyDestroy == 1)
   {
      fEveFrame->RelinquishEveWindow();
      fEveFrame->Destroy();
      fEveFrame = 0;
   }

   TEveElementList::Destroy();
}

//______________________________________________________________________________
void TEveWindow::ClearEveFrame()
{
   // Clears eve-frame associated with this window.
   // This is used in special case when the window is embedded in a foreign
   // GUI container and gets deleted from this side.
   // In particular, this happens when TRootBrowser closes a tab.

   fEveFrame = 0;
}

//______________________________________________________________________________
void TEveWindow::PopulateSlot(TEveCompositeFrame* ef)
{
   TEveWindow* my_ex_parent = fEveFrame ? fEveFrame->fEveParentWindow : 0;

   TEveWindow* ex_win = ef->fEveWindow;

   if (ef->fEveParentWindow)
   {
      if (ex_win) ef->fEveParentWindow->RemoveElement(ex_win);
      ef->fEveParentWindow->AddElement(this);
   }
   else
   {
      if (ex_win) gEve->GetWindows()->RemoveElement(ex_win);
      gEve->GetWindows()->AddElement(this);
   }

   if (my_ex_parent)
   {
      my_ex_parent->RemoveElement(this);
   }

   if (ex_win)
      ex_win->fEveFrame = 0;
   ef->ChangeEveWindow(this); // XXXX
   fEveFrame = ef;

   fEveFrame->fTitleBar->SetText(GetElementName());

   fEveFrame->Layout();
}

//______________________________________________________________________________
void TEveWindow::SetShowTitleBar(Bool_t x)
{
   if (fShowTitleBar == x)
      return;

   fShowTitleBar = x;
   fEveFrame->SetShowTitleBar(fShowTitleBar);
   fEveFrame->Layout();
}

//______________________________________________________________________________
void TEveWindow::TitleBarClicked()
{
   if (fgCurrentWindow == this)
   {
      SetCurrent(kFALSE);
      fgCurrentWindow = 0;
   }
   else
   {
      if (fgCurrentWindow)
      {
         fgCurrentWindow->SetCurrent(kFALSE);
      }
      fgCurrentWindow = this;
      SetCurrent(kTRUE);
      gEve->GetSelection()->UserPickedElement(this, kFALSE);
   }
}

//______________________________________________________________________________
void TEveWindow::SetCurrent(Bool_t curr)
{
   fEveFrame->SetCurrent(curr);
}

//------------------------------------------------------------------------------
// Static helper functions.
//------------------------------------------------------------------------------

//______________________________________________________________________________
TEveWindowSlot* TEveWindow::CreateDefaultWindowSlot()
{
   return new TEveWindowSlot("Free Window Slot", "A free window slot, can become a container or swallow a window.");
}

//______________________________________________________________________________
TEveWindowSlot* TEveWindow::CreateWindowInTab(TGTab* tab, TEveWindow* eve_parent)
{
   TGCompositeFrame *parent = tab->AddTab("<unused>");

   TEveCompositeFrameInTab *slot = new TEveCompositeFrameInTab(parent, eve_parent, tab);

   TEveWindowSlot* ew_slot = TEveWindow::CreateDefaultWindowSlot();
   ew_slot->PopulateSlot(slot);

   parent->AddFrame(slot, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));
   slot->MapWindow();

   tab->Layout();

   return ew_slot;
}


//==============================================================================
// TEveWindowSlot
//==============================================================================

//______________________________________________________________________________
// Description of TEveWindowSlot
//

ClassImp(TEveWindowSlot);

//______________________________________________________________________________
TEveWindowSlot::TEveWindowSlot(const Text_t* n, const Text_t* t) :
   TEveWindow(n, t),
   fEmptyButt(0)
{
   // Constructor.

   fEmptyButt = new TGTextButton(0, "    <empty>\nclick to select");
   fEmptyButt->ChangeOptions(kRaisedFrame);
   fEmptyButt->SetTextJustify(kTextCenterX | kTextCenterY);

   fEmptyButt->Connect("Clicked()", "TEveWindow", this, "TitleBarClicked()");
}

//______________________________________________________________________________
TEveWindowSlot::~TEveWindowSlot()
{
   delete fEmptyButt;
}

//______________________________________________________________________________
TGFrame* TEveWindowSlot::GetGUIFrame()
{
   return fEmptyButt;
}

//______________________________________________________________________________
void TEveWindowSlot::SetCurrent(Bool_t curr)
{
   TEveWindow::SetCurrent(curr);

   if (curr)
      fEmptyButt->SetBackgroundColor(fgCurrentBackgroundColor);
   else
      fEmptyButt->SetBackgroundColor(fEmptyButt->GetDefaultFrameBackground());
   gClient->NeedRedraw(fEmptyButt);
}

//______________________________________________________________________________
TEveWindowPack* TEveWindowSlot::MakePack()
{
   // A pack is created in place of this window-slot.
   // This window-slot will auto-destruct.

   TGPack* pack = new TGPack();
   pack->SetVertical(kFALSE);

   TEveWindowPack* eve_pack = new TEveWindowPack
      (pack, "Pack", "Window container for horizontal and vertical stacking.");

   eve_pack->PopulateSlot(fEveFrame);

   return eve_pack;
}

//______________________________________________________________________________
TEveWindowTab* TEveWindowSlot::MakeTab()
{
   // A tab is created in place of this window-slot.
   // This window-slot will auto-destruct.

   TGTab* tab = new TGTab();

   TEveWindowTab* eve_tab= new TEveWindowTab
      (tab, "Tab", "Window container for horizontal and vertical stacking.");

   eve_tab->PopulateSlot(fEveFrame);

   return eve_tab;
}


//==============================================================================
// TEveWindowMainFrame
//==============================================================================

//______________________________________________________________________________
// Description of TEveWindowMainFrame
//

ClassImp(TEveWindowMainFrame);

//______________________________________________________________________________
TEveWindowMainFrame::TEveWindowMainFrame(const Text_t* n, const Text_t* t) :
   TEveWindow (n, t),
   fMainFrame (0)
{
   // Constructor.
}

//______________________________________________________________________________
TGFrame* TEveWindowMainFrame::GetGUIFrame()
{
   return fMainFrame;
}


//==============================================================================
// TEveWindowPack
//==============================================================================

//______________________________________________________________________________
// Description of TEveWindowPack
//

ClassImp(TEveWindowPack);

//______________________________________________________________________________
TEveWindowPack::TEveWindowPack(TGPack* p, const Text_t* n, const Text_t* t) :
   TEveWindow   (n, t),
   fPack        (p)
{
   // Constructor.
}

//______________________________________________________________________________
TEveWindowPack::~TEveWindowPack()
{
   // Destructor.

   delete fPack;
}

//______________________________________________________________________________
TGFrame* TEveWindowPack::GetGUIFrame()
{
   return fPack;
}

//______________________________________________________________________________
TEveWindowSlot* TEveWindowPack::NewSlot()
{
   TEveCompositeFrame* slot = new TEveCompositeFrameInPack(fPack, this, fPack);

   TEveWindowSlot* ew_slot = TEveWindow::CreateDefaultWindowSlot();
   ew_slot->PopulateSlot(slot);

   fPack->AddFrame(slot);
   slot->MapWindow();

   fPack->Layout();

   return ew_slot;
}

//______________________________________________________________________________
void TEveWindowPack::FlipOrientation()
{
   fPack->SetVertical( ! fPack->GetVertical());
}


//==============================================================================
// TEveWindowTab
//==============================================================================

//______________________________________________________________________________
// Description of TEveWindowTab
//

ClassImp(TEveWindowTab);

//______________________________________________________________________________
TEveWindowTab::TEveWindowTab(TGTab* tab, const Text_t* n, const Text_t* t) :
   TEveWindow(n, t),
   fTab (tab)
{
   // Constructor.
}

//______________________________________________________________________________
TGFrame* TEveWindowTab::GetGUIFrame()
{
   return fTab;
}

//______________________________________________________________________________
TEveWindowSlot* TEveWindowTab::NewSlot()
{
   return TEveWindow::CreateWindowInTab(fTab, this);
}
