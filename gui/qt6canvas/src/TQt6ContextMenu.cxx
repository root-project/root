// Author: Sergey Linev  2/07/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class QRootContextMenu
    \ingroup guiwidgets

This class provides an interface to context sensitive popup menus.
These menus pop up when the user hits the right mouse button, and
are destroyed when the menu pops downs.
The picture below shows a canvas with a pop-up menu.

*/


#include "QRootContextMenu.h"

#include "TROOT.h"
#include "TContextMenu.h"
#include "TVirtualPad.h"


////////////////////////////////////////////////////////////////////////////////
/// Create context menu.

QRootContextMenu::QRootContextMenu(TContextMenu *c, const char *)
    : TObject(), TContextMenuImp(c)
{
   gROOT->GetListOfCleanups()->Add(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete a context menu.

QRootContextMenu::~QRootContextMenu()
{
   gROOT->GetListOfCleanups()->Remove(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Display context popup menu for currently selected object.

void QRootContextMenu::DisplayPopup(Int_t x, Int_t y)
{
   // add menu items to popup menu
   // CreateMenu(fContextMenu->GetSelectedObject());

   printf("Start menu for %p\n", fContextMenu->GetSelectedObject());
}


////////////////////////////////////////////////////////////////////////////////
/// Create dialog object with OK and Cancel buttons. This dialog
/// prompts for the arguments of "method".

void QRootContextMenu::Dialog(TObject *object, TMethod *method)
{
   Dialog(object, (TFunction *)method);
}

////////////////////////////////////////////////////////////////////////////////
/// Create dialog object with OK and Cancel buttons. This dialog
/// prompts for the arguments of "function".
/// function may be a global function or a method

void QRootContextMenu::Dialog(TObject *object, TFunction *function)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Close the context menu if the object is deleted in the
/// RecursiveRemove() operation.

void QRootContextMenu::RecursiveRemove(TObject *obj)
{
   if (obj == fContextMenu->GetSelectedCanvas())
      fContextMenu->SetCanvas(nullptr);
   if (obj == fContextMenu->GetSelectedPad())
      fContextMenu->SetPad(nullptr);
   if (obj == fContextMenu->GetSelectedObject()) {
      // if the object being deleted is the one selected,
      // ungrab the mouse pointer and terminate (close) the menu
      fContextMenu->SetObject(nullptr);
   }
}

