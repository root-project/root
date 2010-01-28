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
#include "TEveWindowManager.h"
#include "TEveManager.h"
#include "TEveSelection.h"

#include "THashList.h"
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
//
// POSSIBLE EXTENSIONS
//
// No frame is drawn around this composite-frame - frame style could be
// available as a (static) member.
//
// Menus of embedded windows could also be managed - hidden or transposed
// to a top-level menubar.

ClassImp(TEveCompositeFrame);

TContextMenu* TEveCompositeFrame::fgCtxMenu = 0;

const TString TEveCompositeFrame::fgkEmptyFrameName("<relinquished>");
TList*        TEveCompositeFrame::fgFrameList = new THashList;

TEveCompositeFrame::IconBarCreator_foo TEveCompositeFrame::fgIconBarCreator = 0;

UInt_t             TEveCompositeFrame::fgTopFrameHeight        = 14;
UInt_t             TEveCompositeFrame::fgMiniBarHeight         = 4;
Bool_t             TEveCompositeFrame::fgAllowTopFrameCollapse = kTRUE;

//______________________________________________________________________________
void TEveCompositeFrame::SetupFrameMarkup(IconBarCreator_foo creator,
                                          UInt_t top_frame_height,
                                          UInt_t mini_bar_height,
                                          Bool_t allow_top_collapse)
{
   // Set properties of the EVE frame.
   // Should be called before the windows are created.

   fgIconBarCreator        = creator;
   fgTopFrameHeight        = top_frame_height;
   fgMiniBarHeight         = mini_bar_height;
   fgAllowTopFrameCollapse = allow_top_collapse;
}

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
   fEveWindow   (0),

   fShowInSync  (kTRUE)
{
   // Constructor.

   fTopFrame = new TGHorizontalFrame(this, 20, fgTopFrameHeight);

   if (fgAllowTopFrameCollapse)
   {
      fToggleBar = new TGTextButton(fTopFrame, "Hide");
      fToggleBar->ChangeOptions(kRaisedFrame);
      fToggleBar->Resize(40, fgTopFrameHeight);
      fToggleBar->Connect("Clicked()", "TEveCompositeFrame", this, "FlipTitleBarState()");
      fTopFrame->AddFrame(fToggleBar, new TGLayoutHints(kLHintsNormal));
   }

   fTitleBar = new TGTextButton(fTopFrame, "Title Bar");
   fTitleBar->ChangeOptions(kRaisedFrame);
   fTitleBar->Resize(40, fgTopFrameHeight);
   fTitleBar->Connect("Clicked()", "TEveCompositeFrame", this, "TitleBarClicked()");
   fTopFrame->AddFrame(fTitleBar, new TGLayoutHints(kLHintsNormal | kLHintsExpandX));

   if (fgIconBarCreator)
   {
      fIconBar = (fgIconBarCreator)(this, fTopFrame, fgTopFrameHeight);
   }
   else
   {
      TGButton* b = new TGTextButton(fTopFrame, "Actions");
      b->ChangeOptions(kRaisedFrame);
      b->Resize(40, fgTopFrameHeight);
      b->Connect("Pressed()", "TEveCompositeFrame", this, "ActionPressed()");
      fIconBar = b;
   }
   fTopFrame->AddFrame(fIconBar, new TGLayoutHints(kLHintsNormal));

   AddFrame(fTopFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX));

   // --- MiniBar
   if (fgAllowTopFrameCollapse)
   {
      fMiniBar = new TGButton(this);
      fMiniBar->ChangeOptions(kRaisedFrame | kFixedHeight);
      fMiniBar->Resize(20, fgMiniBarHeight);
      fMiniBar->SetBackgroundColor(TEveWindow::GetMiniBarBackgroundColor());
      fMiniBar->Connect("Clicked()", "TEveCompositeFrame", this, "FlipTitleBarState()");
      AddFrame(fMiniBar, new TGLayoutHints(kLHintsNormal | kLHintsExpandX));
   }

   // --- Common settings.

   fTopFrame->SetCleanup(kLocalCleanup);
   SetCleanup(kLocalCleanup);

   MapSubwindows();
   HideFrame(fMiniBar);
   SetMapSubwindows(kFALSE);

   // Layout for embedded windows.
   fEveWindowLH = new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY);

   // !!! The following should actually be done somewhere else, in
   // some not-yet-existing static method of TEveWindow. Right now the
   // eve-frame-creation code is still a little bit everywhere.
   if (fEveParent == 0)
      fEveParent = gEve->GetWindowManager();

   fgFrameList->Add(this);
}

//______________________________________________________________________________
TEveCompositeFrame::~TEveCompositeFrame()
{
   // If fEveWindow != 0 we are being deleted from the ROOT GUI side.
   // Relinquishe EveWindow and ref-counting should do the rest.

   fgFrameList->Remove(this);

   if (fEveWindow != 0)
   {
      if (gDebug > 0)
         Info("TEveCompositeFrame::~TEveCompositeFrame",
              "EveWindow not null '%s', relinquishing it now.",
              fEveWindow->GetElementName());

      fEveWindow->ClearEveFrame();
      RelinquishEveWindow();
   }

   delete fEveWindowLH;
}

//==============================================================================

//______________________________________________________________________________
void TEveCompositeFrame::WindowNameChanged(const TString& name)
{
   // Update widgets using window's name or title.

   fTitleBar->SetText(name);
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
   fEveWindow->PostDock();
   gui_frame->MapWindow();

   SetCurrent(fEveWindow->IsCurrent());
   SetShowTitleBar(fEveWindow->GetShowTitleBar());
   WindowNameChanged(fEveWindow->GetElementName());
}

//______________________________________________________________________________
TEveWindow* TEveCompositeFrame::RelinquishEveWindow(Bool_t reparent)
{
   // Remove window and decrease its deny-destroy count.
   // Window's gui-frame is unmapped, removed and, if reparent flag is
   // true (default), reparented to default-root.

   TEveWindow* ex_ew = fEveWindow;

   if (fEveWindow)
   {
      TGFrame* gui_frame = fEveWindow->GetGUIFrame();
      gui_frame->UnmapWindow();
      fEveWindow->PreUndock();
      RemoveFrame(gui_frame);
      if (reparent)
         gui_frame->ReparentWindow(fClient->GetDefaultRoot());
      fEveWindow->DecDenyDestroy();
      fEveWindow = 0;
      SetCurrent(kFALSE);
      WindowNameChanged(fgkEmptyFrameName);
   }

   return ex_ew;
}

//______________________________________________________________________________
TEveWindow* TEveCompositeFrame::GetEveParentAsWindow() const
{
   // Returns eve-parent dynamic-casted to TEveWindow.

   return dynamic_cast<TEveWindow*>(fEveParent);
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

   fShowInSync = show == fEveWindow->GetShowTitleBar();
}

//______________________________________________________________________________
void TEveCompositeFrame::HideAllDecorations()
{
   // Hide title-bar and mini-bar.

   HideFrame(fTopFrame);
   HideFrame(fMiniBar);

   fShowInSync = kFALSE;
}

//______________________________________________________________________________
void TEveCompositeFrame::ShowNormalDecorations()
{
   // Show title-bar or mini-bar, as dictated by the window.

   SetShowTitleBar(fEveWindow->GetShowTitleBar());
}

//______________________________________________________________________________
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

   if (fShowInSync)
      fEveWindow->FlipShowTitleBar();
   else
      SetShowTitleBar(fEveWindow->GetShowTitleBar());
}

//______________________________________________________________________________
void TEveCompositeFrame::TitleBarClicked()
{
   // Slot for mouse-click on the central part of the title-bar.
   // The call is passed to eve-window.

   fEveWindow->TitleBarClicked();
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
   fMainFrame         (mf),
   fOriginalSlot      (0),
   fOriginalContainer (0)
{
   // Constructor.

   fMainFrame->Connect("CloseWindow()", "TEveCompositeFrameInMainFrame", this, "MainFrameClosed()");
   gEve->GetWindowManager()->Connect("WindowDeleted(TEveWindow*)", "TEveCompositeFrameInMainFrame", this, "SomeWindowClosed(TEveWindow*)");
}

//______________________________________________________________________________
TEveCompositeFrameInMainFrame::~TEveCompositeFrameInMainFrame()
{
   // Destructor.

   if (gDebug > 0)
      Info("~TEveCompositeFrameInMainFrame", "Destructor.");

   // MainFrames get deleted with a time-out. So, during EVE manager
   // shutdown, it might happen that this gets called when gEve is null.
   if (gEve && gEve->GetWindowManager())
   {
      gEve->GetWindowManager()->Disconnect("WindowDeleted(TEveWindow*)", this, "SomeWindowClosed(TEveWindow*)");
   }
   else
   {
      Info("~TEveCompositeFrameInMainFrame", "gEve null - OK if it was terminated.");
   }
}

//______________________________________________________________________________
void TEveCompositeFrameInMainFrame::WindowNameChanged(const TString& name)
{
   // Update widgets using window's name or title.

   fMainFrame->SetWindowName(name);

   TEveCompositeFrame::WindowNameChanged(name);
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

   if (gDebug > 0)
      Info("TEveCompositeFrameInMainFrame::Destroy()",
           "Propagating call to main-frame.");

   assert (fEveWindow == 0);

   fMainFrame->CloseWindow();
}

//______________________________________________________________________________
void TEveCompositeFrameInMainFrame::SetOriginalSlotAndContainer(TEveWindow* slot,
                                                                TEveWindow* container)
{
   // Set the container where to return the contained window on destruction.

   static const TEveException kEH("TEveCompositeFrameInMainFrame::SetOriginalSlotAndContainer ");

   if (container && ! container->CanMakeNewSlots())
      throw kEH + "Given window can not make new slots.";

   fOriginalSlot      = slot;
   fOriginalContainer = container;
}

//______________________________________________________________________________
void TEveCompositeFrameInMainFrame::SomeWindowClosed(TEveWindow* w)
{
   // Slot called when a window is closed ... we check that this was
   // not our original container.

   if (w == fOriginalSlot)
      fOriginalSlot = 0;

   if (w == fOriginalContainer)
      fOriginalContainer = 0;
}

//______________________________________________________________________________
void TEveCompositeFrameInMainFrame::MainFrameClosed()
{
   // Slot for main-frame's "CloseWindow()" signal.
   // If an eve window is still present, it will be put into:
   //   - original-container, if it is set;
   ///  - into window-managers default-container.

   if (fEveWindow != 0)
   {
      TEveWindow* swapCandidate = 0;
      if (fOriginalSlot)
      {
         // if use pack, show hidden slot
         TEveCompositeFrameInPack* packFrame = dynamic_cast<TEveCompositeFrameInPack*>(fOriginalSlot->GetEveFrame());
         if (packFrame) {
            TGPack* pack = (TGPack*)(packFrame->GetParent());
            pack->ShowFrame(packFrame);
         }
         swapCandidate = fOriginalSlot;
      }
      else if (fOriginalContainer)
      {
         swapCandidate = fOriginalContainer->NewSlot();
      }
      else if (gEve->GetWindowManager()->HasDefaultContainer())
      {
         swapCandidate =  gEve->GetWindowManager()->GetDefaultContainer()->NewSlot();
      }

      if (swapCandidate)
      {
         TEveWindow::SwapWindows(fEveWindow, swapCandidate);
         gEve->GetWindowManager()->WindowDocked(fEveWindow );
      }
   }

   fMainFrame->DontCallClose();

   if (fEveWindow != 0)
      fEveWindow->DestroyWindowAndSlot();

   if (gDebug > 0)
      Info("TEveCompositeFrameInMainFrame::MainFrameClosed()",
           "Expecting destructor call soon.");
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

   if (gDebug > 0)
      Info("TEveCompositeFrameInPack::Destroy()", "Removing from pack and deleting.");

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
void TEveCompositeFrameInTab::WindowNameChanged(const TString& name)
{
   // Update widgets using window's name or title.

   Int_t t = FindTabIndex();
   fTab->GetTabTab(t)->SetText(new TGString(name));
   fTab->Layout();

   TEveCompositeFrame::WindowNameChanged(name);
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

   if (gDebug > 0)
      Info("TEveCompositeFrameInTab::Destroy()", "Removing from tab and deleting.");

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
//
// Abstract base-class for representing eve-windows.
// Sub-classes define a particular GUI frame that gets show
// in the window.
//

ClassImp(TEveWindow);

UInt_t      TEveWindow::fgMainFrameDefWidth  = 640;
UInt_t      TEveWindow::fgMainFrameDefHeight = 480;
Pixel_t     TEveWindow::fgCurrentBackgroundColor = 0x80A0C0;
Pixel_t     TEveWindow::fgMiniBarBackgroundColor = 0x80C0A0;

//______________________________________________________________________________
TEveWindow::TEveWindow(const char* n, const char* t) :
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

   if (gDebug > 0)
      Info("~TEveWindow", "name='%s', deny-destroy=%d.",
           GetElementName(), fDenyDestroy);
}

//______________________________________________________________________________
void TEveWindow::PreDeleteElement()
{
   // Called before the element is deleted, thus offering the last chance
   // to detach from acquired resources and from the framework itself.
   // Here the request is just passed to TEveManager.
   // If you override it, make sure to call base-class version.

   gEve->GetWindowManager()->DeleteWindow(this);
   TEveElementList::PreDeleteElement();
}

//==============================================================================

//______________________________________________________________________________
void TEveWindow::PreUndock()
{
   // Virtual function called before a window is undocked.

   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveWindow* w = dynamic_cast<TEveWindow*>(*i);
      if (w)
         w->PreUndock();
   }
}

//______________________________________________________________________________
void TEveWindow::PostDock()
{
   // Virtual function called after a window is docked.

   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveWindow* w = dynamic_cast<TEveWindow*>(*i);
      if (w)
         w->PostDock();
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveWindow::NameTitleChanged()
{
   // Name or title of the window changed - propagate to frames.
   // Virtual from TEveElement.

   fEveFrame->WindowNameChanged(GetElementName());
}

//______________________________________________________________________________
void TEveWindow::PopulateEmptyFrame(TEveCompositeFrame* ef)
{
   // Populate given frame-slot - intended for initial population
   // of a new slot or low-level window-swapping.
   // No layout or window-mapping is done.

   ef->fEveParent->AddElement(this);
   ef->AcquireEveWindow(this);
   fEveFrame = ef;
}

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

   TEveWindow* current = gEve->GetWindowManager()->GetCurrentWindow();

   if (current == 0)
      throw eh + "Current eve-window is not set.";

   if (current == this)
      throw eh + "This is the current window ... nothing changed.";

   SwapWindows(this, current);
}

//______________________________________________________________________________
void TEveWindow::UndockWindow()
{
   // Undock the window - put it into a dedicated main-frame.

   TEveWindow* return_cont = fEveFrame->GetEveParentAsWindow();
   if (return_cont && ! return_cont->CanMakeNewSlots())
      return_cont = 0;

   // hide slot if in pack
   TEveCompositeFrameInPack* packFrame = dynamic_cast<TEveCompositeFrameInPack*>(fEveFrame);
   if (packFrame) {
      TGPack* pack = (TGPack*)(packFrame->GetParent());
      pack->HideFrame(fEveFrame);
   }

   TEveWindowSlot* ew_slot = TEveWindow::CreateWindowMainFrame(0);

   TEveWindow::SwapWindows(ew_slot, this);

   ((TEveCompositeFrameInMainFrame*) fEveFrame)->
      SetOriginalSlotAndContainer(ew_slot, return_cont);

   gEve->GetWindowManager()->WindowUndocked(this );
}

//______________________________________________________________________________
void TEveWindow::UndockWindowDestroySlot()
{
   // Undock the window - put it into a dedicated main-frame.
   // The old window slot is destroyed.

   TEveWindow* return_cont = fEveFrame->GetEveParentAsWindow();
   if (return_cont && ! return_cont->CanMakeNewSlots())
      return_cont = 0;

   TEveWindowSlot* ew_slot = TEveWindow::CreateWindowMainFrame(0);

   TEveWindow::SwapWindows(ew_slot, this);

   ((TEveCompositeFrameInMainFrame*) fEveFrame)->
      SetOriginalSlotAndContainer(0, return_cont);

   ew_slot->DestroyWindowAndSlot();

   gEve->GetWindowManager()->WindowUndocked(this);
}

//______________________________________________________________________________
void TEveWindow::ReplaceWindow(TEveWindow* w)
{
   // Replace this window with the passed one.
   // Eve parentship is properly handled.
   // This will most likely lead to the destruction of this window.
   // Layout is called on the frame.

   fEveFrame->RelinquishEveWindow();

   fEveFrame->fEveParent->AddElement(w);
   fEveFrame->AcquireEveWindow(w);
   w->fEveFrame = fEveFrame;

   fEveFrame->fEveParent->RemoveElement(this);

   w->fEveFrame->Layout();
}

//______________________________________________________________________________
void TEveWindow::DestroyWindow()
{
   // Destroy eve-window - replace it with an empty frame-slot.

   if (gDebug > 0)
      Info("TEveWindow::DestroyWindow()", "name='%s', class='%s', deny-destroy=%d.",
           GetElementName(), ClassName(), fDenyDestroy);

   if (fEveFrame != 0 && fDenyDestroy == 1)
   {
      TEveWindowSlot* ew_slot = TEveWindow::CreateDefaultWindowSlot();

      fEveFrame->UnmapWindow();

      Bool_t dozrc = fDestroyOnZeroRefCnt;
      fDestroyOnZeroRefCnt = kFALSE;

      fEveFrame->RelinquishEveWindow();
      ew_slot->PopulateEmptyFrame(fEveFrame);
      fEveFrame->fEveParent->RemoveElement(this);

      fDestroyOnZeroRefCnt = dozrc;

      fEveFrame->Layout();
      fEveFrame->MapWindow();
      fEveFrame = 0;
   }

   TEveElementList::Destroy();
}

//______________________________________________________________________________
void TEveWindow::DestroyWindowAndSlot()
{
   // Destroy eve-window and its frame-slot.

   if (gDebug > 0)
      Info("TEveWindow::DestroyWindowAndSlot()", "'name=%s', class= '%s', deny-destroy=%d.",
           GetElementName(), ClassName(), fDenyDestroy);

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
Bool_t TEveWindow::IsCurrent() const
{
   // Returns true if this window is the current one.

   return gEve->GetWindowManager()->IsCurrentWindow(this);
}

//______________________________________________________________________________
void TEveWindow::MakeCurrent()
{
   // Make this window current.

   if ( ! gEve->GetWindowManager()->IsCurrentWindow(this))
      gEve->GetWindowManager()->SelectWindow(this);
}

//______________________________________________________________________________
void TEveWindow::SetCurrent(Bool_t curr)
{
   // Set current state of this eve-window.
   // Protected method - called by window-manager.

   fEveFrame->SetCurrent(curr);
}

//______________________________________________________________________________
Bool_t TEveWindow::IsAncestorOf(TEveWindow* win)
{
   // Returns true if this is an ancestor of win.

   TEveWindow* parent = dynamic_cast<TEveWindow*>(win->fEveFrame->fEveParent);
   if (parent)
   {
      if (parent == this)
         return kTRUE;
      else
         return IsAncestorOf(parent);
   }
   else
   {
      return kFALSE;
   }
}

//______________________________________________________________________________
void TEveWindow::TitleBarClicked()
{
   // Slot for clicking on the title-bar.
   // The wish that this window becomes the current one is sent to
   // the window-manager.

   gEve->GetWindowManager()->SelectWindow(this);
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
   mf->SetCleanup(kLocalCleanup);

   TEveCompositeFrameInMainFrame *slot = new TEveCompositeFrameInMainFrame
      (mf, eve_parent, mf);

   TEveWindowSlot* ew_slot = TEveWindow::CreateDefaultWindowSlot();
   ew_slot->PopulateEmptyFrame(slot);

   mf->AddFrame(slot, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));
   slot->MapWindow();

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
   parent->SetCleanup(kLocalCleanup);

   TEveCompositeFrameInTab *slot = new TEveCompositeFrameInTab(parent, eve_parent, tab);

   TEveWindowSlot* ew_slot = TEveWindow::CreateDefaultWindowSlot();

   ew_slot->PopulateEmptyFrame(slot);

   parent->AddFrame(slot, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));

   tab->Layout();

   slot->MapWindow();

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
      throw eh + "Called with null window.";

   if (w1 == w2)
      throw eh + "Windows are equal ... nothing to change.";

   if (w1->IsAncestorOf(w2) || w2->IsAncestorOf(w1))
      throw eh + "Windows are in direct ancestry.";

   TEveCompositeFrame *f1 = w1->fEveFrame,  *f2 = w2->fEveFrame;
   TEveElement        *p1 = f1->fEveParent, *p2 = f2->fEveParent;

   if (p1 != p2)
   {
      p1->AddElement(w2);
      p2->AddElement(w1);
   }

   f1->RelinquishEveWindow(kFALSE);
   f2->RelinquishEveWindow(kFALSE);
   f1->AcquireEveWindow(w2); w2->fEveFrame = f1;
   f2->AcquireEveWindow(w1); w1->fEveFrame = f2;

   if (p1 != p2)
   {
      p1->RemoveElement(w1);
      p2->RemoveElement(w2);
   }

   f1->Layout(); f2->Layout();
}

//==============================================================================

//______________________________________________________________________________
UInt_t TEveWindow::GetMainFrameDefWidth()
{
   // Get default width for new main-frame windows. Static.

   return fgMainFrameDefWidth;
}

//______________________________________________________________________________
UInt_t TEveWindow::GetMainFrameDefHeight()
{
   // Get default heigth for new main-frame windows. Static.

   return fgMainFrameDefHeight;
}

//______________________________________________________________________________
void TEveWindow::SetMainFrameDefWidth (UInt_t x)
{
   // Set default width for new main-frame windows. Static.

   fgMainFrameDefWidth  = x;
}

//______________________________________________________________________________
void TEveWindow::SetMainFrameDefHeight(UInt_t x)
{
   // Set default height for new main-frame windows. Static.

   fgMainFrameDefHeight = x;
}

//______________________________________________________________________________
Pixel_t TEveWindow::GetCurrentBackgroundColor()
{
   // Get background-color for marking the title-bar of current window. Static.

   return fgCurrentBackgroundColor;
}

//______________________________________________________________________________
Pixel_t TEveWindow::GetMiniBarBackgroundColor()
{
   // Get background-color for mini-bar (collapsed title-bar). Static.

   return fgMiniBarBackgroundColor;
}

//______________________________________________________________________________
void TEveWindow::SetCurrentBackgroundColor(Pixel_t p)
{
   // Set background-color for marking the title-bar of current window. Static.

   fgCurrentBackgroundColor = p;
}

//______________________________________________________________________________
void TEveWindow::SetMiniBarBackgroundColor(Pixel_t p)
{
   // Set background-color for mini-bar (collapsed title-bar). Static.

   fgMiniBarBackgroundColor = p;
}


//==============================================================================
// TEveWindowSlot
//==============================================================================

//______________________________________________________________________________
// Description of TEveWindowSlot
//

ClassImp(TEveWindowSlot);

//______________________________________________________________________________
TEveWindowSlot::TEveWindowSlot(const char* n, const char* t) :
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

   fEmptyButt->DeleteWindow();
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

   TEveWindowPack* eve_pack = new TEveWindowPack
      (0, "Pack", "Window container for horizontal and vertical stacking.");

   ReplaceWindow(eve_pack);

   return eve_pack;
}

//______________________________________________________________________________
TEveWindowTab* TEveWindowSlot::MakeTab()
{
   // A tab is created in place of this window-slot.
   // This window-slot will auto-destruct.

   TEveWindowTab* eve_tab = new TEveWindowTab
      (0, "Tab", "Window container for horizontal and vertical stacking.");

   ReplaceWindow(eve_tab);

   return eve_tab;
}

//______________________________________________________________________________
TEveWindowFrame* TEveWindowSlot::MakeFrame(TGFrame* frame)
{
   // An eve-window-frame is created and frame is passed into it.
   // If frame is 0 (the default), a default composite-frame will be created
   // in TEveWindowFrame() constructor.
   // This window-slot will auto-destruct.

   TEveWindowFrame* eve_frame = new TEveWindowFrame
      (frame, "External frame", "");

   ReplaceWindow(eve_frame);

   return eve_frame;
}

//______________________________________________________________________________
TGCompositeFrame* TEveWindowSlot::StartEmbedding()
{
   // Start embedding a window that will replace the current slot.
   // It is expected that a main-frame will be created and then
   // StopEmbedding() will be called.

   static const TEveException eh("TEveWindowSlot::StartEmbedding ");

   if (fEmbedBuffer != 0)
      throw eh + "Already embedding.";

   fEmbedBuffer = new TGCompositeFrame(gClient->GetDefaultRoot());
   fEmbedBuffer->SetEditable(kTRUE);

   return fEmbedBuffer;
}

//______________________________________________________________________________
TEveWindowFrame* TEveWindowSlot::StopEmbedding(const char* name)
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

   if (name) {
      mf->SetWindowName(name);
   }

   TEveWindowFrame* eve_frame = new TEveWindowFrame
      (f, mf->GetWindowName(), mf->ClassName());

   ReplaceWindow(eve_frame);

   return eve_frame;
}


//==============================================================================
// TEveWindowFrame
//==============================================================================

//______________________________________________________________________________
//
// Encapsulates TGFrame into an eve-window.
// The frame is owned by the eve-window.

ClassImp(TEveWindowFrame);

//______________________________________________________________________________
TEveWindowFrame::TEveWindowFrame(TGFrame* frame, const char* n, const char* t) :
   TEveWindow (n, t),
   fGUIFrame  (frame)
{
   // Constructor.
   // If the passed frame is 0, a default TGCompositeFrame frame is instantiated
   // and set to local-cleanup.

   if (fGUIFrame == 0)
   {
      fGUIFrame = new TGCompositeFrame();
      fGUIFrame->SetCleanup(kLocalCleanup);
   }
}

//______________________________________________________________________________
TEveWindowFrame::~TEveWindowFrame()
{
   // Destructor.

   fGUIFrame->DeleteWindow();
}

//______________________________________________________________________________
TGCompositeFrame* TEveWindowFrame::GetGUICompositeFrame()
{
   // Returns the registered top-frame of this eve-window dynamic-casted
   // to composite-frame.
   // Throws an execption if the cast fails.

   static const TEveException kEH("TEveWindowFrame::GetGUICompositeFrame ");

   TGCompositeFrame *cf = dynamic_cast<TGCompositeFrame*>(fGUIFrame);
   if (cf == 0)
      throw kEH + "The registered frame is not a composite-frame.";

   return cf;
}

//==============================================================================
// TEveWindowPack
//==============================================================================

//______________________________________________________________________________
//
// Encapsulates TGPack into an eve-window.
// The pack is owned by the eve-window.

ClassImp(TEveWindowPack);

//______________________________________________________________________________
TEveWindowPack::TEveWindowPack(TGPack* p, const char* n, const char* t) :
   TEveWindow   (n, t),
   fPack        (p ? p : new TGPack())
{
   // Constructor.
   // If passed pack is 0, a defualt one is instantiated.
}

//______________________________________________________________________________
TEveWindowPack::~TEveWindowPack()
{
   // Destructor.

   fPack->DeleteWindow();
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

   return NewSlotWithWeight(1.f);
}

//______________________________________________________________________________
TEveWindowSlot* TEveWindowPack::NewSlotWithWeight(Float_t w)
{
   // Create a new weighted frame-slot at the last position of the pack.

   TEveCompositeFrame* slot = new TEveCompositeFrameInPack(fPack, this, fPack);

   TEveWindowSlot* ew_slot = TEveWindow::CreateDefaultWindowSlot();
   ew_slot->PopulateEmptyFrame(slot);

   fPack->AddFrameWithWeight(slot, 0, w);
   slot->MapWindow();

   fPack->Layout();

   return ew_slot;
}

//______________________________________________________________________________
void TEveWindowPack::FlipOrientation()
{
   // Flip orientation of the pack (vertical / horizontal).

   fPack->SetVertical( ! fPack->GetVertical());
}

//______________________________________________________________________________
void TEveWindowPack::SetVertical(Bool_t x)
{
   // Set orientation of the pack (vertical / horizontal).

   fPack->SetVertical(x);
}

//______________________________________________________________________________
void TEveWindowPack::EqualizeFrames()
{
   // Refit existing frames so that their lengths are equal.

   fPack->EqualizeFrames();
   fPack->Layout();
}

//==============================================================================
// TEveWindowTab
//==============================================================================

//______________________________________________________________________________
//
// Encapsulates TGTab into an eve-window.
// The tab is owned by the eve-window.

ClassImp(TEveWindowTab);

//______________________________________________________________________________
TEveWindowTab::TEveWindowTab(TGTab* tab, const char* n, const char* t) :
   TEveWindow(n, t),
   fTab (tab ? tab : new TGTab())
{
   // Constructor.
   // If passed tab is 0, a defualt one is instantiated.
}

//______________________________________________________________________________
TEveWindowTab::~TEveWindowTab()
{
   // Destructor.

   fTab->DeleteWindow();
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
