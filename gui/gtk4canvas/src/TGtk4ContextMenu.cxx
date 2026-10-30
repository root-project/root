// Author: Sergey Linev  13/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TGtk4ContextMenu
    \ingroup gtk4canvas

This class provides an interface to context-sensitive popup menus.
These menus pop up when the user hits the right mouse button, and
are destroyed when the menu pops downs.
*/


#include "TGtk4ContextMenu.h"

#include "TROOT.h"
#include "TContextMenu.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TMethod.h"
#include "TDataMember.h"
#include "TToggle.h"
#include "TClassMenuItem.h"
#include "TAttText.h"
#include "TAttMarker.h"

#include "TGtk4Canvas.h"
#include "Gtk4DrawArea.h"
#include "Gtk4MethodDialog.h"

#include <iostream>

enum EContextMenu {
   kToggleStart       = 1000, // first id of toggle menu items
   kToggleListStart   = 2000, // first id of toggle list menu items
   kUserFunctionStart = 3000  // first id of user added functions/methods, etc...
};

////////////////////////////////////////////////////////////////////////////////
/// Create context menu.

TGtk4ContextMenu::TGtk4ContextMenu(TContextMenu *c, const char *)
    : TObject(), TContextMenuImp(c)
{
   gROOT->GetListOfCleanups()->Add(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete a context menu.

TGtk4ContextMenu::~TGtk4ContextMenu()
{
   gROOT->GetListOfCleanups()->Remove(this);
   fTrash.Delete();
}

////////////////////////////////////////////////////////////////////////////////
/// Display context popup menu for currently selected object.

void TGtk4ContextMenu::DisplayPopup(Int_t x, Int_t y)
{
   // add menu items to popup menu
   // CreateMenu(fContextMenu->GetSelectedObject());

   auto object = fContextMenu->GetSelectedObject();
   if (!object)
      return;

   fCustomArg.clear();
   fTrash.Delete();


   auto canv = dynamic_cast<TCanvas *>(fContextMenu->GetSelectedCanvas());
   auto canvimp = dynamic_cast<ROOT::Experimental::TGtk4Canvas *>(canv->GetCanvasImp());
   auto widget = canvimp->GetDrawArea();

//   QPoint screenPos = widget->mapToGlobal(widget->rect().topLeft());

//   QMenu menu;
//   QSignalMapper map;

//   QObject::connect(&map, &QSignalMapper::mappedInt,
//                    this, &TGtk4ContextMenu::executeMenu);

   auto action_group = Gio::SimpleActionGroup::create();

   action_group->add_action_with_parameter(
      "action",
      Glib::VariantType("i"),  // Expected parameter type (int32)
      [this](const Glib::VariantBase& parameter) {
         if (auto var = Glib::VariantBase::cast_dynamic<Glib::Variant<int>>(parameter)) {
            int value = var.get();
            std::cout << "Action triggered with integer value: " << value << std::endl;
            executeMenu(value);
         }
      }
   );

   auto menu_model = Gio::Menu::create();

   auto current_section = Gio::Menu::create();


   auto addSeparator = [&menu_model, &current_section]() {
      menu_model->append_section(current_section);
      current_section = Gio::Menu::create();
   };

   auto addMenuAction = [&current_section, &action_group, this](const TString &text, int id, void *arg = nullptr, int checked = 0) {
      bool enabled = true;
      if ((text == "DrawClone") || (text == "DrawClass") || (text == "Inspect") ||
          (text == "SetShowProjectionX") || (text == "SetShowProjectionY") ||
          (text == "DrawPanel") || (text == "FitPanel"))
         enabled = false;


      if (checked != 0) {
         TString action_name = TString::Format("toggle-item-%d", id);
         auto action = Gio::SimpleAction::create_bool(action_name.Data(), checked > 0);
         action->signal_activate().connect([action,this, id](const Glib::VariantBase & parameter) {
            bool current;
            action->get_state(current);
            action->set_state(Glib::Variant<bool>::create(!current));
            std::cout << "Action triggered with integer value: " << id << std::endl;
            executeMenu(id);
         });
         action->set_enabled(enabled);
         action_group->add_action(action);

         action_name.Prepend("context.");
         auto item = Gio::MenuItem::create(text.Data(), action_name.Data());
         current_section->append_item(item);
      } else if (enabled) {
         auto item = Gio::MenuItem::create(text.Data(), "");
         item->set_action_and_target("context.action", Glib::Variant<int>::create(id));
         current_section->append_item(item);
      } else {
         // just refer to non-existing action
         current_section->append(text.Data(), "context.dummy");
      }

      fCustomArg[id] = arg;
   };

   // Add a title
   TString buffer = fContextMenu->CreatePopupTitle(object);
   addMenuAction(buffer, -1, nullptr);
   addSeparator();
   bool last_separ = true;

   int entry = 0, toggle = kToggleStart, togglelist = kToggleListStart;
   int userfunction = kUserFunctionStart;

   // Get list of menu items from the selected object's class
   TList *menuItemList = object->IsA()->GetMenuList();

   TIter nextItem(menuItemList);

   while (auto menuItem = (TClassMenuItem*) nextItem()) {
      switch (menuItem->GetType()) {
         case TClassMenuItem::kPopupSeparator: {
            if (!last_separ)
               addSeparator();
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
                        addSeparator();
                        needSep = kFALSE;
                     }
                     addMenuAction(method->GetName(), entry++, method);
                     break;
                  case kMenuSubMenu:
                     if (auto m = method->FindDataMember()) {
                        if (needSep) {
                           addSeparator();
                           needSep = kFALSE;
                        }

                        // TODO: implement selection menu
                        if (m->GetterMethod()) {
                           // QMenu *r = menu.addMenu(method->GetName());
                           TIter nxt(m->GetOptions());
                           while (auto it = (TOptionListItem*) nxt()) {
                              TToggle *t = new TToggle;
                              t->SetToggledObject(object, method);
                              t->SetOnValue(it->fValue);
                              fTrash.Add(t);

                              // TODO: implement checked state
                              addMenuAction(it->fOptName.Data(), togglelist++, t, t->GetState() ? 1 : -1);
                           }
                        } else {
                           addMenuAction(method->GetName(), entry++, method);
                        }
                     }
                     break;

                  case kMenuToggle: {
                     if (needSep) {
                        addSeparator();
                        needSep = kFALSE;
                     }

                     TToggle *t = new TToggle;
                     t->SetToggledObject(object, method);
                     t->SetOnValue(1);
                     fTrash.Add(t);

                     addMenuAction(method->GetName(), toggle++, t, t->GetState() ? 1 : -1);
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
                  addMenuAction(menuItemTitle, toggle++, t, t->GetState() ? 1 : -1);
               }
            } else {
               if (strlen(menuItemTitle)==0)
                  menuItemTitle = menuItem->GetFunctionName();
               addMenuAction(menuItemTitle, userfunction++, menuItem);
            }
            break;
         }

         default:
            break;
      }
   }

   widget->insert_action_group("context", action_group);

   widget->ShowContextMenu(menu_model, x, y);
}

////////////////////////////////////////////////////////////////////////////////
/// Create dialog object with OK and Cancel buttons. This dialog
/// prompts for the arguments of "method".

void TGtk4ContextMenu::Dialog(TObject *object, TMethod *method)
{
   Dialog(object, (TFunction *)method);
}

////////////////////////////////////////////////////////////////////////////////
/// Create dialog object with OK and Cancel buttons. This dialog
/// prompts for the arguments of "function".
/// function may be a global function or a method

void TGtk4ContextMenu::Dialog(TObject * object, TFunction * func)
{
   auto widget = new Gtk4MethodDialog(300, 300);

   widget->methodDialog(fContextMenu, object, func);
}

////////////////////////////////////////////////////////////////////////////////
/// Handle remove of some ROOT objects

void TGtk4ContextMenu::RecursiveRemove(TObject *obj)
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


////////////////////////////////////////////////////////////////////////////////
/// Execute specified menu item

void TGtk4ContextMenu::executeMenu(int id)
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
      auto m = (TMethod *) ud;

      if (!strcmp(m->GetName(), "SetLineAttributes")) {
         auto attline = dynamic_cast<TAttLine *>(fContextMenu->GetSelectedObject());
         if (attline) {
            auto widget = new Gtk4MethodDialog(300, 300);
            widget->attLineDialog(attline);
         }
      } else if (!strcmp(m->GetName(), "SetFillAttributes")) {
         auto attfill = dynamic_cast<TAttFill *>(fContextMenu->GetSelectedObject());
         if (attfill) {
            auto widget = new Gtk4MethodDialog(300, 300);
            widget->attFillDialog(attfill);
         }
      } else if (!strcmp(m->GetName(), "SetTextAttributes")) {
         auto atttext = dynamic_cast<TAttText *>(fContextMenu->GetSelectedObject());
         if (atttext) {
            auto widget = new Gtk4MethodDialog(300, 300);
            widget->attTextDialog(atttext);
         }
      } else if (!strcmp(m->GetName(), "SetMarkerAttributes")) {
         auto attmark = dynamic_cast<TAttMarker *>(fContextMenu->GetSelectedObject());
         if (attmark) {
            auto widget = new Gtk4MethodDialog(300, 300);
            widget->attMarkerDialog(attmark);
         }
      } else {
         fContextMenu->Action(m);
      }
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
