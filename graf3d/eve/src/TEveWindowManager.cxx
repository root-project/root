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
TEveWindowManager::TEveWindowManager(const Text_t* n, const Text_t* t) :
   TEveElementList(n, t),
   TQObject       (),
   fCurrentWindow (0)
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
void TEveWindowManager::WindowDeleted(TEveWindow* window)
{
   // Called by a window before it gets deleted.

   if (window == fCurrentWindow)
   {
      fCurrentWindow = 0;
      CurrentWindowChanged(fCurrentWindow);
   }
}

//______________________________________________________________________________
void TEveWindowManager::WindowSelected(TEveWindow* window)
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
   // CurrentWindowChanged() signal is emitted.

   if (window == fCurrentWindow)
      window = 0;

   if (fCurrentWindow)
      fCurrentWindow->SetCurrent(kFALSE);

   fCurrentWindow = window;

   if (fCurrentWindow)
      fCurrentWindow->SetCurrent(kTRUE);

   CurrentWindowChanged(fCurrentWindow);
}

//______________________________________________________________________________
void TEveWindowManager::CurrentWindowChanged(TEveWindow* window)
{
   // Emit the "CurrentWindowChanged(TEveWindow*)" signal.

   Emit("CurrentWindowChanged(TEveWindow*)", (Long_t)window);
}
