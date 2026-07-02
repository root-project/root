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
#include "TDataMember.h"
#include "TToggle.h"
#include "TClassMenuItem.h"

#include "QRootMethodDialog.h"
#include "QPaintWidget.h"
#include "TQt6Canvas.h"

#include <QtCore/QSignalMapper>
#include <QMenu>
#include <QAction>

enum EContextMenu {
   kToggleStart       = 1000, // first id of toggle menu items
   kToggleListStart   = 2000, // first id of toggle list menu items
   kUserFunctionStart = 3000  // first id of user added functions/methods, etc...
};

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
   fTrash.Delete();
}

////////////////////////////////////////////////////////////////////////////////
/// Display context popup menu for currently selected object.

void QRootContextMenu::DisplayPopup(Int_t x, Int_t y)
{
   // add menu items to popup menu
   // CreateMenu(fContextMenu->GetSelectedObject());

   auto object = fContextMenu->GetSelectedObject();
   if (!object)
      return;

   fCustomArg.clear();
   fTrash.Delete();


   auto canv = dynamic_cast<TCanvas *>(fContextMenu->GetSelectedCanvas());
   auto canvimp = dynamic_cast<TQt6Canvas *>(canv->GetCanvasImp());
   auto widget = canvimp->GetPaintWidget();

   QPoint screenPos = widget->mapToGlobal(widget->rect().topLeft());

   QMenu menu;
   QSignalMapper map;

   QObject::connect(&map, &QSignalMapper::mappedInt,
                    this, &QRootContextMenu::executeMenu);

   // Add a title
   QString buffer = fContextMenu->CreatePopupTitle(object);
   addMenuAction(&menu, &map, buffer, -1, nullptr);
   menu.addSeparator();
   bool last_separ = true;



   // addMenuAction(&menu, &map, buffer, curId++);

   int entry = 0, toggle = kToggleStart, togglelist = kToggleListStart;
   int userfunction = kUserFunctionStart;

   // Get list of menu items from the selected object's class
   TList *menuItemList = object->IsA()->GetMenuList();

   TIter nextItem(menuItemList);

   while (auto menuItem = (TClassMenuItem*) nextItem()) {
      switch (menuItem->GetType()) {
         case TClassMenuItem::kPopupSeparator: {
            if (!last_separ)
               menu.addSeparator();
            last_separ = true;
            break;
         }
         case TClassMenuItem::kPopupStandardList: {
            // Standard list of class methods. Rebuild from scratch.
            // Get linked list of objects menu items (i.e. member functions
            // with the token *MENU in their comment fields.
            TList *methodList = new TList;
            object->IsA()->GetMenuItems(methodList);

            TMethod *method;
            TClass  *classPtr = nullptr;
            TIter next(methodList);
            Bool_t needSep = kFALSE;

            while ((method = (TMethod*) next())) {
               if (classPtr != method->GetClass()) {
                  needSep = kTRUE;
                  classPtr = method->GetClass();
               }

               EMenuItemKind menuKind = method->IsMenuItem();
               TString last_component;

               switch (menuKind) {
                  case kMenuDialog:
                     // search for arguments to the MENU statement
                     if (needSep) {
                        menu.addSeparator();
                        needSep = kFALSE;
                     }
                     addMenuAction(&menu, &map, method->GetName(), entry++, method);
                     break;
                  case kMenuSubMenu:
                     if (auto m = method->FindDataMember()) {
                        if (needSep) {
                           menu.addSeparator();
                           needSep = kFALSE;
                        }

                        if (m->GetterMethod()) {

                           QMenu *r = menu.addMenu(method->GetName());
                           TIter nxt(m->GetOptions());
                           while (auto it = (TOptionListItem*) nxt()) {
                              const char *name = it->fOptName;
                              Long_t val = it->fValue;

                              TToggle *t = new TToggle;
                              t->SetToggledObject(object, method);
                              t->SetOnValue(val);
                              fTrash.Add(t);

                              auto act = addMenuAction(r, &map, name, togglelist++, t);
                              act->setCheckable(true);
                              if (t->GetState())
                                 act->setChecked(true);
                           }
                        } else {
                           addMenuAction(&menu, &map, method->GetName(), entry++, method);
                        }
                     }
                     break;

                  case kMenuToggle: {
                     if (needSep) {
                        menu.addSeparator();
                        needSep = kFALSE;
                     }

                     TToggle *t = new TToggle;
                     t->SetToggledObject(object, method);
                     t->SetOnValue(1);
                     fTrash.Add(t);

                     auto act = addMenuAction(&menu, &map, method->GetName(), toggle++, t);
                     act->setCheckable(true);
                     if (t->GetState())
                        act->setChecked(true);
                     break;
                  }
                  default:
                     break;
               }
            }
            delete methodList;
         }
         break;
         case TClassMenuItem::kPopupUserFunction: {
            const char* menuItemTitle = menuItem->GetTitle();
            if (menuItem->IsToggle()) {
               TMethod* method = object->IsA()->GetMethodWithPrototype(menuItem->GetFunctionName(),menuItem->GetArgs());
               if (method) {
                  TToggle *t = new TToggle;
                  t->SetToggledObject(object, method);
                  t->SetOnValue(1);
                  fTrash.Add(t);

                  if (strlen(menuItemTitle)==0)
                     menuItemTitle = method->GetName();
                  auto act = addMenuAction(&menu, &map, menuItemTitle, toggle++, t);
                  act->setCheckable(true);
                  if (t->GetState())
                     act->setChecked(true);
               }
            } else {
               if (strlen(menuItemTitle)==0)
                  menuItemTitle = menuItem->GetFunctionName();
               addMenuAction(&menu, &map, menuItemTitle, userfunction++, menuItem);
            }
            break;
         }

         default:
            break;
      }
   }

   menu.exec(screenPos + QPoint(x, y));
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

void QRootContextMenu::Dialog(TObject * object, TFunction * func)
{
   QRootMethodDialog dlg;
   dlg.methodDialog(fContextMenu, object, func);
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

QAction* QRootContextMenu::addMenuAction(QMenu* menu, QSignalMapper* map, const QString& text, int id, void *arg)
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

   fCustomArg[id] = arg;

   return act;
}

void QRootContextMenu::executeMenu(int id)
{
   if (id < 0)
      return;
   void *ud = fCustomArg[id];

   if (ud) {
      // retrieve the highlighted function
      TFunction *function = nullptr;
      if (id < kToggleStart) {
         TMethod *m = (TMethod *)ud;
         function = (TFunction *)m;
      } else if (id >= kToggleStart && id < kUserFunctionStart) {
         TToggle *t = (TToggle *)ud;
         TMethodCall *mc = (TMethodCall *)t->GetSetter();
         function = (TFunction *)mc->GetMethod();
      } else {
         TClassMenuItem *mi = (TClassMenuItem *)ud;
         function = gROOT->GetGlobalFunctionWithPrototype(mi->GetFunctionName());
      }
      if (function)
         fContextMenu->SetMethod(function);
   }

   if (id < kToggleStart) {
      TMethod *m = (TMethod *) ud;
      fContextMenu->Action(m);
   } else if (id >= kToggleStart && id < kToggleListStart) {
      TToggle *t = (TToggle *) ud;
      fContextMenu->Action(t);
   } else if (id >= kToggleListStart && id < kUserFunctionStart) {
      TToggle *t = (TToggle *) ud;
      if (t->GetState() == 0)
         t->SetState(1);
   } else {
      TClassMenuItem *mi = (TClassMenuItem*)ud;
      fContextMenu->Action(mi);
   }
}
