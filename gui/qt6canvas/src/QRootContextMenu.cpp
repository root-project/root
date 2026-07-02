// Author: Sergey Linev  2/07/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class RRootContextMenu
    \ingroup qt6canvas

This class provides an interface to context sensitive popup menus.
These menus pop up when the user hits the right mouse button, and
are destroyed when the menu pops downs.
The picture below shows a canvas with a pop-up menu.

*/


#include "QRootContextMenu.h"

#include "TROOT.h"
#include "TContextMenu.h"
#include "TCanvas.h"
#include "TMethod.h"


#include "QRootMethodDialog.h"
#include "QPaintWidget.h"
#include "TQt6Canvas.h"

#include <QtCore/QSignalMapper>
#include <QMenu>
#include <QAction>

////////////////////////////////////////////////////////////////////////////////
/// Create context menu.

QRootContextMenu::QRootContextMenu(TContextMenu *c, const char *)
    : QObject(), TObject(), TContextMenuImp(c)
{
   gROOT->GetListOfCleanups()->Add(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete a context menu.

QRootContextMenu::~QRootContextMenu()
{
   gROOT->GetListOfCleanups()->Remove(this);

   fMenuObj = nullptr;
   delete fMenuMethods;
   fMenuMethods = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Display context popup menu for currently selected object.

void QRootContextMenu::DisplayPopup(Int_t x, Int_t y)
{
   // add menu items to popup menu
   // CreateMenu(fContextMenu->GetSelectedObject());

   fMenuObj = nullptr;
   delete fMenuMethods;
   fMenuMethods = nullptr;

   printf("Start menu for %p\n", fContextMenu->GetSelectedObject());

   auto canv = dynamic_cast<TCanvas *>(fContextMenu->GetSelectedCanvas());
   auto canvimp = dynamic_cast<TQt6Canvas *>(canv->GetCanvasImp());
   auto widget = canvimp->GetPaintWidget();

   QPoint screenPos = widget->mapToGlobal(widget->rect().topLeft());

   QMenu menu;
   QSignalMapper map;

   QObject::connect(&map, &QSignalMapper::mappedInt,
                    this, &QRootContextMenu::executeMenu);

   fMousePosX = x;
   fMousePosY = y;

   fMenuObj = fContextMenu->GetSelectedObject();
   fMenuMethods = new TList;
   TClass *cl = fMenuObj->IsA();
   int curId = -1;

   QString buffer = TString::Format("%s::%s", cl->GetName(), fMenuObj->GetName()).Data();
   addMenuAction(&menu, &map, buffer, curId++);

   cl->GetMenuItems(fMenuMethods);
   menu.addSeparator();

   TIter iter(fMenuMethods);
   while (auto method = dynamic_cast<TMethod*>(iter())) {
      buffer = method->GetName();
      addMenuAction(&menu, &map, buffer, curId++);
   }

   if (menu.exec(screenPos + QPoint(x, y)) == nullptr) {
      fMenuObj = nullptr;
      delete fMenuMethods;
      fMenuMethods = nullptr;
   }
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

void QRootContextMenu::Dialog(TObject * /* object */, TFunction * /* function */)
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

QAction* QRootContextMenu::addMenuAction(QMenu* menu, QSignalMapper* map, const QString& text, int id)
{
   bool enabled = true;

   QAction* act = new QAction(text, menu);

   if (!enabled)
      if ((text.compare("DrawClone") == 0) || (text.compare("DrawClass") == 0) || (text.compare("Inspect") == 0) ||
          (text.compare("SetShowProjectionX") == 0) || (text.compare("SetShowProjectionY") == 0) ||
          (text.compare("DrawPanel") == 0) || (text.compare("FitPanel") == 0))
         act->setEnabled(false);

   QObject::connect(act, &QAction::triggered, [id, map]() {
      map->mappedInt(id);
   });

   menu->addAction(act);
   map->setMapping(act, id);

   return act;
}

void QRootContextMenu::executeMenu(int id)
{
   QString text;
   bool ok = false;
   if (id >= 0) {

      // save global to Pad before calling TObject::Execute()

      TVirtualPad *psave = gROOT->GetSelectedPad();

      auto canv = dynamic_cast<TCanvas *>(fContextMenu->GetSelectedCanvas());

      TMethod *method = (TMethod *) fMenuMethods->At(id);

      /// test: do this in any case!
      canv->HandleInput(kButton3Up, gPad->XtoAbsPixel(fMousePosX), gPad->YtoAbsPixel(fMousePosY));

      // change current dir that all new histograms appear here
      gROOT->cd();

      if (method->GetListOfMethodArgs()->First()) {
         QRootMethodDialog dlg;

         dlg.methodDialog(fMenuObj, method);
      } else {
         gROOT->SetFromPopUp(kTRUE);
         fMenuObj->Execute(method->GetName(), "");

         if (fMenuObj->TestBit(TObject::kNotDeleted)) {
            // emit MenuCommandExecuted(fMenuObj, method->GetName());
         } else {
            fMenuObj = nullptr;
         }

      }

      canv->GetPadSave()->Update();
      canv->GetPadSave()->Modified();

      gROOT->SetSelectedPad(psave);

      gROOT->GetSelectedPad()->Update();
      gROOT->GetSelectedPad()->Modified();

      canv->Modified();
      canv->ForceUpdate();
      gROOT->SetFromPopUp(kFALSE);
   }

   fMenuObj = nullptr;
   delete fMenuMethods;
   fMenuMethods = nullptr;
}
