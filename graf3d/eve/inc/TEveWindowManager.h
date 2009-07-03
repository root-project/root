// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveWindowManager
#define ROOT_TEveWindowManager

#include "TEveElement.h"
#include "TQObject.h"

class TEveWindow;
class TEveWindowSlot;

class TEveWindowManager : public TEveElementList,
                          public TQObject
{
private:
   TEveWindowManager(const TEveWindowManager&);            // Not implemented
   TEveWindowManager& operator=(const TEveWindowManager&); // Not implemented

protected:
   TEveWindow   *fCurrentWindow;
   TEveWindow   *fDefaultContainer;

   void DestroyWindowRecursively(TEveWindow* window);

public:
   TEveWindowManager(const char* n="TEveWindowManager", const char* t="");
   virtual ~TEveWindowManager();

   void SelectWindow(TEveWindow* w);
   void DeleteWindow(TEveWindow* w);

   void WindowDocked(TEveWindow* window); // *SIGNAL*
   void WindowUndocked (TEveWindow* window); // *SIGNAL*
   void WindowSelected(TEveWindow* window); // *SIGNAL*
   void WindowDeleted (TEveWindow* window); // *SIGNAL*

   TEveWindow*     GetCurrentWindow() const { return fCurrentWindow; }
   Bool_t          IsCurrentWindow(const TEveWindow* w) const { return w == fCurrentWindow; }
   TEveWindowSlot* GetCurrentWindowAsSlot() const;

   TEveWindow*     GetDefaultContainer() const { return fDefaultContainer; }
   Bool_t          HasDefaultContainer() const { return fDefaultContainer != 0; }
   void            SetDefaultContainer(TEveWindow* w);

   void            DestroyWindows();

   // Global frame decoration control.

   void HideAllEveDecorations();
   void ShowNormalEveDecorations();
   void SetShowTitleBars(Bool_t state);

   ClassDef(TEveWindowManager, 0); // Manager for EVE windows.
};

#endif
