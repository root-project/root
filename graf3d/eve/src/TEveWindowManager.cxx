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

/** \class TEveWindowManager
\ingroup TEve
Manager for EVE windows.

Provides the concept of the current window and takes care for proper
destruction of the windows.

It is also the EVE-parent of windows that are not attached into the
hierarchy of EVE-windows.

Window-manager is created by the EVE-manager and can be retrieved via:
~~~ {.cpp}
   gEve->GetWindowManager.
~~~
*/

ClassImp(TEveWindowManager);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveWindowManager::TEveWindowManager(const char* n, const char* t) :
   TEveElementList(n, t),
   TQObject       (),
   fCurrentWindow    (0),
   fDefaultContainer (0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveWindowManager::~TEveWindowManager()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Entry-point for communicating the fact that a window was acted
/// upon in such a way that it should become the current window.
/// If the passed window is already the current one, it is deselected.
///
/// For example, this is called from title-bar, when creating a new
/// window slot, etc.
///
/// If the change is accepted (the manager can refuse to make a
/// window current), the state of window is changed accordingly and
/// WindowSelected() signal is emitted.

void TEveWindowManager::SelectWindow(TEveWindow* window)
{
   if (window == fCurrentWindow)
      window = 0;

   if (fCurrentWindow)
      fCurrentWindow->SetCurrent(kFALSE);

   fCurrentWindow = window;

   if (fCurrentWindow)
      fCurrentWindow->SetCurrent(kTRUE);

   WindowSelected(fCurrentWindow);
}

////////////////////////////////////////////////////////////////////////////////
/// Called by a window before it gets deleted.

void TEveWindowManager::DeleteWindow(TEveWindow* window)
{
   if (window == fCurrentWindow)
   {
      fCurrentWindow = 0;
      WindowSelected(fCurrentWindow);
   }
   WindowDeleted(window);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit the "WindowDocked(TEveWindow*)" signal.

void TEveWindowManager::WindowDocked(TEveWindow* window)
{
   Emit("WindowDocked(TEveWindow*)", (Longptr_t)window);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit the "WindowUndocked(TEveWindow*)" signal.

void TEveWindowManager::WindowUndocked(TEveWindow* window)
{
   Emit("WindowUndocked(TEveWindow*)", (Longptr_t)window);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit the "WindowSelected(TEveWindow*)" signal.

void TEveWindowManager::WindowSelected(TEveWindow* window)
{
   Emit("WindowSelected(TEveWindow*)", (Longptr_t)window);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit the "WindowDeleted(TEveWindow*)" signal.

void TEveWindowManager::WindowDeleted(TEveWindow* window)
{
   Emit("WindowDeleted(TEveWindow*)", (Longptr_t)window);
}

////////////////////////////////////////////////////////////////////////////////
/// Return current window dynamic-casted to TEveWindowSlot.

TEveWindowSlot* TEveWindowManager::GetCurrentWindowAsSlot() const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Destroy window's children and then the window itself.
/// Protected method used during shutdown.

void TEveWindowManager::DestroyWindowRecursively(TEveWindow* window)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Wait for all windows to shut-down.

void TEveWindowManager::DestroyWindows()
{
   while (HasChildren())
   {
      TEveWindow* w = dynamic_cast<TEveWindow*>(FirstChild());
      if (w)
         DestroyWindowRecursively(w);
      else
         RemoveElement(FirstChild());
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Hide all eve decorations (title-bar and mini-bar) on all frames.

void TEveWindowManager::HideAllEveDecorations()
{
   TEveCompositeFrame *ecf = 0;
   TIter wins(TEveCompositeFrame::fgFrameList);
   while ((ecf = (TEveCompositeFrame*) wins()))
   {
      ecf->HideAllDecorations();
      ecf->Layout();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Show eve decorations (title-bar or mini-bar) as specified for
/// the contained window on all frames.

void TEveWindowManager::ShowNormalEveDecorations()
{
   TEveCompositeFrame *ecf = 0;
   TIter wins(TEveCompositeFrame::fgFrameList);
   while ((ecf = (TEveCompositeFrame*) wins()))
   {
      ecf->ShowNormalDecorations();
      ecf->Layout();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set show title-bar state on all frames.
/// This does not modify the per-window settings - call
/// ShowNormalEveDecorations() to restore them.

void TEveWindowManager::SetShowTitleBars(Bool_t state)
{
   TEveCompositeFrame *ecf = 0;
   TIter wins(TEveCompositeFrame::fgFrameList);
   while ((ecf = (TEveCompositeFrame*) wins()))
   {
      ecf->SetShowTitleBar(state);
      ecf->Layout();
   }
}
