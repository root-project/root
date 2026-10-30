// Author: Sergey Linev, GSI  14/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_Gtk4MethodDialog
#define ROOT_Gtk4MethodDialog

#include <gtkmm.h>
#include <vector>

class TContextMenu;
class TObject;
class TFunction;

class Gtk4MethodDialog: public Gtk::Window {

   public:
      Gtk4MethodDialog(unsigned width = 400, unsigned height = 200);

      void methodDialog(TContextMenu *menu, TObject *object, TFunction* func);

      void clearAllElements();

      void addLine(const char *txt, Gtk::Widget *widget);

      Gtk::ColorDialogButton *addColorInput(const char *title, int colindx);

      void addArg(const char *argname, const char *value, const char *type);

      Gtk::Button *addOkCancelButtons();

   protected:

      void on_ok_button_clicked();

      Gtk::Box *m_main_box = nullptr;

      std::vector<Gtk::Entry *> fArgs;
      TContextMenu *fMenu = nullptr;
      TObject *fObject = nullptr;
      TFunction *fFunc = nullptr;
};

#endif
