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
// Abstract base-class for frame-slots that encompass EVE-windows
// (sub-classes of TEveWindow).
//
// The EVE-frame classes are managed by their embedded EVE-windows and
// mostly serve as an interface to particular ROOT widgets
// (sub-classes of TGCompositeFrame) they are embedded into.
//
// This base-class, a sub-class of a vertical composite-frame, creates
// also the title-bar which can be used to interact with the embedded
// window. Optionally, the title-bar can be replaced with a mini-bar
// (a 4-pixel thin bar at the top). By clicking on the mini-bar, the
// title-bar is restored.
//
// Sub-classes provide for specific behaviour and expectations of
// individual ROOT GUI container frames.
//

ClassImp(TEveCompositeFrame);

TContextMenu* TEveCompositeFrame::fgCtxMenu = 0;
const TString TEveCompositeFrame::fgkEmptyFrameName("<relinquished>");

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

   fEveParent   (eve_parent),
   fEveWindow   (0)
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
   fMiniBar->SetBackgroundColor(TEveWindow::GetMiniBarBackgroundColor());
   AddFrame(fMiniBar, new TGLayoutHints(kLHintsNormal | kLHintsExpandX));

   fMiniBar->Connect("Clicked()", "TEveCompositeFrame", this, "FlipTitleBarState()");

   // --- Common settings.

   SetCleanup(kDeepCleanup);

   MapSubwindows();
   HideFrame(fMiniBar);
   SetMapSubwindows(kFALSE);

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

//==============================================================================

void TEveCompositeFrame::ActionPressed()
{
   // The action-button of the title-bar was pressed.
   // This opens context menu of the eve-window.

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
   // Change display-state of the title-bar / mini-bar.
   // This function is used as a slot and passes the call to eve-window.

   fEveWindow->FlipShowTitleBar();
}

//______________________________________________________________________________
void TEveCompositeFrame::TitleBarClicked()
{
   // Slot for mouse-click on the central part of the title-bar.
   // The call is passed to eve-window.

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
   fTitleBar->SetText(fEveWindow->GetElementName());
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
      fTitleBar->SetText(fgkEmptyFrameName);
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
   // Set current state of this frame.
   // This is called by the management functions in TEveWindow.

   if (curr) {
      fTitleBar->SetBackgroundColor(TEveWindow::GetCurrentBackgroundColor());
   } else {
      fTitleBar->SetBackgroundColor(GetDefaultFrameBackground());
   }
   fClient->NeedRedraw(fTitleBar);
}

//______________________________________________________________________________
void TEveCompositeFrame::SetShowTitleBar(Bool_t show)
{
   // Set state of title-bar. This toggles between the display of the full
   // title-bar and 4-pixel-high mini-bar.

   if (show) {
      HideFrame(fMiniBar);
      ShowFrame(fTopFrame);
   } else {
      HideFrame(fTopFrame);
      ShowFrame(fMiniBar);
   }
}


//==============================================================================
// TEveCompositeFrameInMainFrame
//==============================================================================

//______________________________________________________________________________
//
// An EVE window-slot contained within a TGMainFrame.

ClassImp(TEveCompositeFrameInMainFrame);

//______________________________________________________________________________
TEveCompositeFrameInMainFrame::TEveCompositeFrameInMainFrame(TGCompositeFrame* parent,
                                                             TEveWindow*  eve_parent,
                                                             TGMainFrame* mf) :
   TEveCompositeFrame(parent, eve_parent),
   fMainFrame        (mf)
{
   // Constructor.

   fMainFrame->Connect("CloseWindow()", "TEveCompositeFrameInMainFrame", this, "MainFrameClosed()");
}

//______________________________________________________________________________
TEveCompositeFrameInMainFrame::~TEveCompositeFrameInMainFrame()
{
   // Destructor.

   printf("TEveCompositeFrameInMainFrame::~TEveCompositeFrameInMainFrame\n");
}

//______________________________________________________________________________
void TEveCompositeFrameInMainFrame::Destroy()
{
   // Virtual function called from eve side when the frame should be
   // destroyed. This means we expect that fEveWindow is null.
   //
   // We simply call CloseWindow() on the main-frame which will in
   // turn generate the "CloseWindow()" signal.
   // This is then handled in MainFrameClosed().

   printf("TEveCompositeFrameInMainFrame::Destroy()\n");

   assert (fEveWindow == 0);

   fMainFrame->CloseWindow();
}

//______________________________________________________________________________
void TEveCompositeFrameInMainFrame::AcquireEveWindow(TEveWindow* ew)
{
   // Virtual from TEveCompositeFrame.
   // Set also main-frame-name to window-name.

   TEveCompositeFrame::AcquireEveWindow(ew);

   fMainFrame->SetWindowName(fEveWindow->GetElementName());
}

//______________________________________________________________________________
TEveWindow* TEveCompositeFrameInMainFrame::RelinquishEveWindow()
{
   // Virtual from TEveCompositeFrame.
   // Set also main-frame-name to "<relinquished>".

   fMainFrame->SetWindowName(fgkEmptyFrameName);

   return TEveCompositeFrame::RelinquishEveWindow();
}

//______________________________________________________________________________
void TEveCompositeFrameInMainFrame::MainFrameClosed()
{
   // Slot for main-frame's "CloseWindow()" signal.

   fMainFrame->DontCallClose();
   fEveWindow->DestroyWindowAndSlot();

   printf("TEveCompositeFrameInMainFrame::MainFrameClosed() ... expecting destructor.\n");
}


//==============================================================================
// TEveCompositeFrameInPack
//==============================================================================

//______________________________________________________________________________
//
// An EVE window-slot contained within one frame of a TGPack.

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
   // Virtual function called from eve side when the frame should be
   // destroyed. This means we expect that fEveWindow is null.
   //
   // Remove the frame from pack and delete it.

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
   // Virtual function called from eve side when the frame should be
   // destroyed. This means we expect that fEveWindow is null.
   //
   // Remove the frame from tab and delete it.

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
   // Virtual from TEveCompositeFrame.
   // Set also tab-name to window-name and call tab-layout.

   TEveCompositeFrame::AcquireEveWindow(ew);

   Int_t t = FindTabIndex();
   fTab->GetTabTab(t)->SetText(new TGString(fEveWindow->GetElementName()));
   fTab->Layout();
}

//______________________________________________________________________________
TEveWindow* TEveCompositeFrameInTab::RelinquishEveWindow()
{
   // Virtual from TEveCompositeFrame.
   // Set also tab-name to "<relinquished>" and call tab-layout.

   Int_t t = FindTabIndex();
   fTab->GetTabTab(t)->SetText(new TGString(fgkEmptyFrameName));
   fTab->Layout();

   return TEveCompositeFrame::RelinquishEveWindow();
}

//______________________________________________________________________________
void TEveCompositeFrameInTab::SetCurrent(Bool_t curr)
{
   // Set current state of this frame.
   // Virtual from TEveCompositeFrame.

   TEveCompositeFrame::SetCurrent(curr);

   Int_t t = FindTabIndex();
   TGTabElement* te = fTab->GetTabTab(t);
   if (curr) {
      te->SetBackgroundColor(TEveWindow::GetCurrentBackgroundColor());
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
UInt_t      TEveWindow::fgMainFrameDefWidth  = 640;
UInt_t      TEveWindow::fgMainFrameDefHeight = 480;
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
   // Destructor.

   if (this == fgCurrentWindow)
      fgCurrentWindow = 0;

   printf("TEveWindow::~TEveWindow  '%s' '%s', cnt=%d\n", GetElementName(), ClassName(), fDenyDestroy);
}

//==============================================================================

//______________________________________________________________________________
void TEveWindow::SwapWindow(TEveWindow* w)
{
   // Swap frames with the given window.

   static const TEveException eh("TEveWindow::SwapWindow ");

   if (w == 0)
      throw eh + "Called with null argument.";

   SwapWindows(this, w);
}

//______________________________________________________________________________
void TEveWindow::SwapWindowWithCurrent()
{
   // Swap frames with the current window.

   static const TEveException eh("TEveWindow::SwapWindowWithCurrent ");

   if (fgCurrentWindow == 0)
      throw eh + "Current eve-window is not set.";

   if (fgCurrentWindow == this)
      throw eh + "This is the current window ... nothing changed.";

   SwapWindows(this, fgCurrentWindow);
}

//______________________________________________________________________________
void TEveWindow::DestroyWindow()
{
   // Destroy eve-window - replace it with an empty frame-slot.

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
   // Destroy eve-window and its frame-slot.

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
   // Populate given frame-slot.
   //
   // This function does all the eve-element side management of
   // removing the old eve-window from the eve-window-parent of the
   // frame-slot and adding a new one.
   //
   // This will be replaced with function: SwapWindow(TEveWindow* w).

   TEveElement* my_ex_parent = fEveFrame ? fEveFrame->fEveParent : 0;

   TEveWindow* ex_win = ef->fEveWindow;

   if (ef->fEveParent)
   {
      if (ex_win) ef->fEveParent->RemoveElement(ex_win);
      ef->fEveParent->AddElement(this);
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
   ef->ChangeEveWindow(this);
   fEveFrame = ef;

   fEveFrame->Layout();
}

//______________________________________________________________________________
void TEveWindow::SetShowTitleBar(Bool_t x)
{
   // Set display state of the title-bar.
   // This is forwarded to eve-frame.

   if (fShowTitleBar == x)
      return;

   fShowTitleBar = x;
   fEveFrame->SetShowTitleBar(fShowTitleBar);
   fEveFrame->Layout();
}

//______________________________________________________________________________
void TEveWindow::TitleBarClicked()
{
   // Slot for clicking on the title-bar. This window becomes the current
   // window or, if it was already current, the current is set to zero. 

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
   // Set current state of this eve-window.
   // Should be protected.

   fEveFrame->SetCurrent(curr);
}

//------------------------------------------------------------------------------
// Static helper functions.
//------------------------------------------------------------------------------

//______________________________________________________________________________
TEveWindowSlot* TEveWindow::CreateDefaultWindowSlot()
{
   // Create a default window slot.
   // Static helper.

   return new TEveWindowSlot("Free Window Slot", "A free window slot, can become a container or swallow a window.");
}

//______________________________________________________________________________
TEveWindowSlot* TEveWindow::CreateWindowMainFrame(TEveWindow* eve_parent)
{
   // Create a new main-frame and populate it with a default window-slot.
   // The main-frame is mapped.
   // Static helper.

   TGMainFrame* mf = new TGMainFrame(gClient->GetRoot(), fgMainFrameDefWidth, fgMainFrameDefHeight);
   mf->SetCleanup(kDeepCleanup);

   TEveCompositeFrameInMainFrame *slot = new TEveCompositeFrameInMainFrame
      (mf, eve_parent, mf);

   mf->AddFrame(slot, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));
   slot->MapWindow();

   TEveWindowSlot* ew_slot = TEveWindow::CreateDefaultWindowSlot();
   ew_slot->PopulateSlot(slot);

   mf->Layout();
   mf->MapWindow();

   return ew_slot;
}

//______________________________________________________________________________
TEveWindowSlot* TEveWindow::CreateWindowInTab(TGTab* tab, TEveWindow* eve_parent)
{
   // Create a new tab in a given tab-widget and populate it with a
   // default window-slot.
   // Static helper.

   TGCompositeFrame *parent = tab->AddTab("<unused>");

   TEveCompositeFrameInTab *slot = new TEveCompositeFrameInTab(parent, eve_parent, tab);

   TEveWindowSlot* ew_slot = TEveWindow::CreateDefaultWindowSlot();
   ew_slot->PopulateSlot(slot);

   parent->AddFrame(slot, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));
   slot->MapWindow();

   tab->Layout();

   return ew_slot;
}

//______________________________________________________________________________
void TEveWindow::SwapWindows(TEveWindow* w1, TEveWindow* w2)
{
   // Swap windows w1 and w2. They are properly reparented in the eve
   // hierarch as well.
   // Layout is called on both frames.

   static const TEveException eh("TEveWindow::SwapWindows ");

   if (w1 == 0 || w2 == 0)
      throw eh + "Called with null argument.";

   if (w1 == w2)
      throw eh + "Arguments are equal ... nothing to change.";

   ::Warning("TEveWindow::SwapWindows", "Implementation in progress!"); // !!!!
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
   TEveWindow (n, t),
   fEmptyButt   (0),
   fEmbedBuffer (0)
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
   // Destructor.

   delete fEmptyButt;
}

//______________________________________________________________________________
TGFrame* TEveWindowSlot::GetGUIFrame()
{
   // Return top-frame of this eve-window - the big button to make slot current.

   return fEmptyButt;
}

//______________________________________________________________________________
void TEveWindowSlot::SetCurrent(Bool_t curr)
{
   // Set current state of this window-slot.
   // Virtual from TEveWindow.

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

//______________________________________________________________________________
void TEveWindowSlot::StartEmbedding()
{
   // Start embedding a window that will replace the current slot.
   // It is expected that a main-frame will be created and then
   // StopEmbedding() will be called.

   static const TEveException eh("TEveWindowSlot::StartEmbedding ");

   if (fEmbedBuffer != 0)
      throw eh + "Already embedding.";

   fEmbedBuffer = new TGCompositeFrame(gClient->GetDefaultRoot());
   fEmbedBuffer->SetCleanup(kLocalCleanup);
   fEmbedBuffer->SetEditable(kTRUE);
}

//______________________________________________________________________________
TEveWindowFrame* TEveWindowSlot::StopEmbedding()
{
   // An embedded window is created in place of this window-slot.
   // This window-slot will auto-destruct.

   static const TEveException eh("TEveWindowSlot::StopEmbedding ");

   if (fEmbedBuffer == 0) {
      Warning(eh, "Embedding not in progress.");
      return 0;
   }

   fEmbedBuffer->SetEditable(kFALSE);

   Int_t size = fEmbedBuffer->GetList()->GetSize();

   if (size == 0) {
      Warning(eh, "Frame has not been registered.");
      delete fEmbedBuffer;
      fEmbedBuffer = 0;
      return 0;
   }

   if (size > 1) {
      Warning(eh, "Several frames have been registered (%d). Only the first one will be taken.", size);
   }

   TGFrame *f = ((TGFrameElement*)fEmbedBuffer->GetList()->First())->fFrame;
   fEmbedBuffer->RemoveFrame(f);
   f->UnmapWindow();
   f->ReparentWindow(gClient->GetDefaultRoot());
   delete fEmbedBuffer;
   fEmbedBuffer = 0;

   TGMainFrame *mf = dynamic_cast<TGMainFrame*>(f);
   assert(mf != 0);

   TEveWindowFrame* eve_frame= new TEveWindowFrame
      (f, mf->GetWindowName(), mf->ClassName());

   eve_frame->PopulateSlot(fEveFrame);

   return eve_frame;
}


//==============================================================================
// TEveWindowFrame
//==============================================================================

//______________________________________________________________________________
// Description of TEveWindowFrame
//

ClassImp(TEveWindowFrame);

//______________________________________________________________________________
TEveWindowFrame::TEveWindowFrame(TGFrame* f, const Text_t* n, const Text_t* t) :
   TEveWindow (n, t),
   fGUIFrame  (f)
{
   // Constructor.
}

//______________________________________________________________________________
TEveWindowFrame::~TEveWindowFrame()
{
   // Destructor.

   delete fGUIFrame;
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
   // Return top-frame of this eve-window - the pack.

   return fPack;
}

//______________________________________________________________________________
TEveWindowSlot* TEveWindowPack::NewSlot()
{
   // Create a new frame-slot at the last position of the pack.

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
   // Flip orientation of the pack (horizontal / vertical).

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
TEveWindowTab::~TEveWindowTab()
{
   // Destructor.

   delete fTab;
}

//______________________________________________________________________________
TGFrame* TEveWindowTab::GetGUIFrame()
{
   // Return top-frame of this eve-window - the tab.

   return fTab;
}

//______________________________________________________________________________
TEveWindowSlot* TEveWindowTab::NewSlot()
{
   // Create new frame-slot - a new tab.

   return TEveWindow::CreateWindowInTab(fTab, this);
}
