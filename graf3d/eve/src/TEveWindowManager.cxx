// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveWindowManager.h"
#include "TEveWindow.h"

//______________________________________________________________________________
//
// Manager for EVE windows.
//
// Provides the concept of the current window and takes care for proper
// destruction of the windows.
//
// It is also the EVE-parent of windows that are not attaced into the
// hierarchy of EVE-windows.
//
// Window-manager is created by the EVE-manager and can be retrieved via:
//   gEve->GetWindowManager.

ClassImp(TEveWindowManager);

//______________________________________________________________________________
TEveWindowManager::TEveWindowManager(const char* n, const char* t) :
   TEveElementList(n, t),
   TQObject       (),
   fCurrentWindow    (0),
   fDefaultContainer (0)
{
   // Constructor.
}

//______________________________________________________________________________
TEveWindowManager::~TEveWindowManager()
{
   // Destructor.
}

//==============================================================================

//______________________________________________________________________________
void TEveWindowManager::SelectWindow(TEveWindow* window)
{
   // Entry-point for communicating the fact that a window was acted
   // upon in such a way that it should become the current window.
   // If the passed window is already the current one, it is deselcted.
   //
   // For example, this is called from title-bar, when creating a new
   // window slot, etc.
   //
   // If the change is accepted (the manager can refuse to make a
   // window current), the state of window is changed accordingly and
   // WindowSelected() signal is emitted.

   if (window == fCurrentWindow)
      window = 0;

   if (fCurrentWindow)
      fCurrentWindow->SetCurrent(kFALSE);

   fCurrentWindow = window;

   if (fCurrentWindow)
      fCurrentWindow->SetCurrent(kTRUE);

   WindowSelected(fCurrentWindow);
}

//______________________________________________________________________________
void TEveWindowManager::DeleteWindow(TEveWindow* window)
{
   // Called by a window before it gets deleted.

   if (window == fCurrentWindow)
   {
      fCurrentWindow = 0;
      WindowSelected(fCurrentWindow);
   }
   WindowDeleted(window);
}

//______________________________________________________________________________
void TEveWindowManager::WindowDocked(TEveWindow* window)
{
   // Emit the "WindowDocked(TEveWindow*)" signal.

   Emit("WindowDocked(TEveWindow*)", (Long_t)window);
}

//______________________________________________________________________________
void TEveWindowManager::WindowUndocked(TEveWindow* window)
{
   // Emit the "WindowUndocked(TEveWindow*)" signal.

   Emit("WindowUndocked(TEveWindow*)", (Long_t)window);
}

//______________________________________________________________________________
void TEveWindowManager::WindowSelected(TEveWindow* window)
{
   // Emit the "WindowSelected(TEveWindow*)" signal.

   Emit("WindowSelected(TEveWindow*)", (Long_t)window);
}

//______________________________________________________________________________
void TEveWindowManager::WindowDeleted(TEveWindow* window)
{
   // Emit the "WindowDeleted(TEveWindow*)" signal.

   Emit("WindowDeleted(TEveWindow*)", (Long_t)window);
}

//==============================================================================

//______________________________________________________________________________
TEveWindowSlot* TEveWindowManager::GetCurrentWindowAsSlot() const
{
   // Return current window dynamic-casted to TEveWindowSlot.

   return dynamic_cast<TEveWindowSlot*>(fCurrentWindow);
}

void TEveWindowManager::SetDefaultContainer(TEveWindow* w)
{
   // Set default container window.
   // It has to be able to create new slots.
   // When main-frames are closed they will place the windows here.

   static const TEveException kEH("TEveWindowManager::SetDefaultContainer ");

   if ( ! w->CanMakeNewSlots())
      throw kEH + "Given window can not make new slots.";

   fDefaultContainer = w;
}

//______________________________________________________________________________
void TEveWindowManager::DestroyWindowRecursively(TEveWindow* window)
{
   // Destroy window's children and then the window itself.
   // Protected method used during shutdown.

   while (window->HasChildren())
   {
      TEveWindow* w = dynamic_cast<TEveWindow*>(window->FirstChild());
      if (w)
         DestroyWindowRecursively(w);
      else
         window->RemoveElement(window->FirstChild());
   }
   window->DestroyWindowAndSlot();
}

//______________________________________________________________________________
void TEveWindowManager::DestroyWindows()
{
   // Wait for all windows to shut-down.

   while (HasChildren())
   {
      TEveWindow* w = dynamic_cast<TEveWindow*>(FirstChild());
      if (w)
         DestroyWindowRecursively(w);
      else
         RemoveElement(FirstChild());
   }

}

//==============================================================================

//______________________________________________________________________________
void TEveWindowManager::HideAllEveDecorations()
{
   // Hide all eve decorations (title-bar and mini-bar) on all frames.

   TEveCompositeFrame *ecf = 0;
   TIter wins(TEveCompositeFrame::fgFrameList);
   while ((ecf = (TEveCompositeFrame*) wins()))
   {
      ecf->HideAllDecorations();
      ecf->Layout();
   }
}

//______________________________________________________________________________
void TEveWindowManager::ShowNormalEveDecorations()
{
   // Show eve decorations (title-bar or mini-bar) as specified for
   // the contained window on all frames.

   TEveCompositeFrame *ecf = 0;
   TIter wins(TEveCompositeFrame::fgFrameList);
   while ((ecf = (TEveCompositeFrame*) wins()))
   {
      ecf->ShowNormalDecorations();
      ecf->Layout();
   }
}

//______________________________________________________________________________
void TEveWindowManager::SetShowTitleBars(Bool_t state)
{
   // Set show title-bar state on all frames.
   // This does not modify the per-window settings - call
   // ShowNormalEveDecorations() to restore them.

   TEveCompositeFrame *ecf = 0;
   TIter wins(TEveCompositeFrame::fgFrameList);
   while ((ecf = (TEveCompositeFrame*) wins()))
   {
      ecf->SetShowTitleBar(state);
      ecf->Layout();
   }
}
