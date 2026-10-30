// Author: Sergey Linev, GSI  14/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Gtk4MethodDialog.h"

#include "TString.h"
#include "TROOT.h"
#include "TVirtualPad.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TMethodCall.h"
#include "TObjString.h"
#include "TContextMenu.h"
#include "TColor.h"

#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttText.h"
#include "TAttMarker.h"


#include <iostream>

/** \class Gtk4MethodDialog
    \ingroup gtk4canvas

Specialized dialog to enter arguments for method exection
*/

////////////////////////////////////////////////////////////////////////////////
/// Create method dialog

Gtk4MethodDialog::Gtk4MethodDialog(unsigned width, unsigned height)
{

   set_title("method dialog");
   set_default_size(width, height);

   m_main_box = Gtk::make_managed<Gtk::Box>(Gtk::Orientation::VERTICAL, 10);
   m_main_box->set_margin(20);
   set_child(*m_main_box);
}

////////////////////////////////////////////////////////////////////////////////
/// add line with color button

void Gtk4MethodDialog::addColorInput(Int_t colindx)
{
   TColor *rootColor = gROOT->GetColor(colindx);
   Gdk::RGBA initial_color;
   initial_color.set_rgba(0.0, 0.0, 0.0, 1.0);

   if (rootColor)
      initial_color.set_rgba(rootColor->GetRed(), rootColor->GetGreen(), rootColor->GetBlue(), rootColor->GetAlpha());

   auto color_dialog = Gtk::ColorDialog::create();
   color_dialog->set_title("Select a Custom Color");
   color_dialog->set_modal(true);

   // Instantiate the specific ColorDialogButton using our dialog rules
   m_color_button = Gtk::make_managed<Gtk::ColorDialogButton>(color_dialog);
   m_color_button->set_halign(Gtk::Align::END);

   // Set an initial default color (e.g., solid Red)
   m_color_button->set_rgba(initial_color);

   addLine("Color", m_color_button);
}

////////////////////////////////////////////////////////////////////////////////
/// Add single dialog line

void Gtk4MethodDialog::addLine(const char *txt, Gtk::Widget *widget)
{
   auto layout_grid = Gtk::make_managed<Gtk::Grid>();
   layout_grid->set_column_spacing(15);
   layout_grid->set_hexpand(true);
   layout_grid->set_halign(Gtk::Align::FILL);
   m_main_box->append(*layout_grid);

   auto label = Gtk::make_managed<Gtk::Label>(txt);
   label->set_halign(Gtk::Align::START);
   label->set_hexpand(true);              // Wants to expand
   label->set_halign(Gtk::Align::FILL);   // Stretch to fill its 50% allocation
   label->set_xalign(0.0);


   widget->set_hexpand(true);             // Wants to expand equally with the label
   widget->set_halign(Gtk::Align::FILL);

   layout_grid->attach(*label,  0, 0, 3, 1);
   layout_grid->attach(*widget, 3, 0, 7, 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Add method argument

void Gtk4MethodDialog::addArg(const char *argname, const char *value, const char *)
{
   auto entry = Gtk::make_managed<Gtk::Entry>();
   // Optional configuration properties
   entry->set_text(value);
   // entry->set_placeholder_text("enter value");
   // entry->set_max_length(50); // Restrict length if needed
   // entry->signal_activate().connect(sigc::mem_fun(*this, &Gtk4MethodDialog::on_entry_submitted));

   addLine(TString::Format("%s:", argname).Data(), entry);

   fArgs.push_back(entry);
}


////////////////////////////////////////////////////////////////////////////////
/// Add Ok and Cancel buttons

void Gtk4MethodDialog::addOkCancelButtons()
{
   auto hbox = Gtk::make_managed<Gtk::Box>();
   hbox->set_hexpand(true);
   hbox->set_halign(Gtk::Align::END);
   m_main_box->append(*hbox);

   auto button1 = Gtk::make_managed<Gtk::Button>("Ok");
   button1->signal_clicked().connect(sigc::mem_fun(*this, &Gtk4MethodDialog::on_ok_button_clicked));

   auto button2 = Gtk::make_managed<Gtk::Button>("Cancel");
   button2->signal_clicked().connect([this]() { close(); });

   hbox->append(*button1);
   hbox->append(*button2);
}


void Gtk4MethodDialog::methodDialog(TContextMenu *menu, TObject *object, TFunction* func)
{
   if (!menu || !object || !func)
      return;

   set_title(menu->CreateDialogTitle(object, func));

   // iterate through all arguments and create appropriate input-data objects:
   // inputlines, option menus...
   TIter next(func->GetListOfMethodArgs());

   while (auto argument = (TMethodArg *) next()) {
      TString argTitle = menu->CreateArgumentTitle(argument);
      TString type = argument->GetTypeName();
      TDataType *datatype = gROOT->GetType(type);
      TString basictype;

      if (datatype) {
         basictype = datatype->GetTypeName();
      } else {
         if (type.CompareTo("enum") != 0)
            std::cout << "*** Warning in Dialog(): data type is not basic type, assuming (int)\n";
         basictype = "int";
      }

      if (TString(argument->GetTitle()).Index("*") != kNPOS) {
         basictype += "*";
         type = "char*";
      }

      TDataMember *m = argument->GetDataMember();
      if (m && m->GetterMethod()) {

         m->GetterMethod()->Init(object->IsA(), m->GetterMethod()->GetMethodName(), "");

         // Get the current value and form it as a text:

         TString val;

         if (basictype == "char*") {
            char *tdefval = nullptr;
            m->GetterMethod()->Execute(object, "", &tdefval);
            if (tdefval) val = tdefval;
         } else
         if ((basictype == "float") ||
             (basictype == "double")) {
            Double_t ddefval = 0.;
            m->GetterMethod()->Execute(object, "", ddefval);
            val = TString::Format("%g", ddefval);
         } else
         if ((basictype == "char") ||
             (basictype == "int")  ||
             (basictype == "bool")  ||
             (basictype == "long") ||
             (basictype == "short")) {
            Longptr_t ldefval = 0;
            m->GetterMethod()->Execute(object, "", ldefval);
            val = TString::Format("%ld", (long) ldefval);
         }

         // Find out whether we have options ...

         TList *opt;
         if ((opt = m->GetOptions()) != nullptr) {
            // should stop dialog
            // workaround JAM: do not stop dialog, use textfield (for time display toggle)
            addArg(argTitle.Data(), val.Data(), type.Data());
            //return;
         } else {
            // we haven't got options - textfield ...
            addArg(argTitle.Data(), val.Data(), type.Data());
         }
      } else {    // if m not found ...
         TString argDflt;
         if (argument->GetDefault())
            argDflt = argument->GetDefault();

         if ((argDflt.Length() > 1) &&
             (argDflt[0]=='\"') && (argDflt[argDflt.Length()-1]=='\"')) {
            // cut "" from the string argument
            argDflt.Remove(0,1);
            argDflt.Remove(argDflt.Length()-1,1);
         }

         addArg(argTitle.Data(), argDflt.Data(), type.Data());
      }
   }

   addOkCancelButtons();

   present();

   fMenu = menu;
   fObject = object;
   fFunc = func;
}


void Gtk4MethodDialog::attLineDialog(TAttLine *att)
{
   addColorInput(att->GetLineColor());

   auto style_list = Gtk::StringList::create({"None", "Style 1", "Style 2", "Style 3", "Style 4", "Style 5", "Style 6", "Style 7", "Style 8", "Style 9"});
   m_line_style = Gtk::make_managed<Gtk::DropDown>(style_list);
   m_line_style->set_selected(att->GetLineStyle());

   addLine("Line style", m_line_style);

   auto adjustment = Gtk::Adjustment::create(/* value */ att->GetLineWidth(), /* lower */ 1, /* upper */ 20,
                                            /* step_increment */ 1, /* page_increment */ 5.0);

   m_line_width = Gtk::make_managed<Gtk::SpinButton>(adjustment);
   m_line_width->set_digits(0); // 0 decimal places -> integer display

   addLine("Line width", m_line_width);

   addOkCancelButtons();

   present();

   fAttLine = att;
}


void Gtk4MethodDialog::attFillDialog(TAttFill *att)
{
   addColorInput(att->GetFillColor());

   addOkCancelButtons();

   present();

   fAttFill = att;
}

void Gtk4MethodDialog::attTextDialog(TAttText *att)
{
   addColorInput(att->GetTextColor());

   addOkCancelButtons();

   present();

   fAttText = att;

}

void Gtk4MethodDialog::attMarkerDialog(TAttMarker *att)
{
   addColorInput(att->GetMarkerColor());

   addOkCancelButtons();

   present();

   fAttMarker = att;
}


void Gtk4MethodDialog::on_ok_button_clicked()
{
   if (fMenu && fObject) {

      TObjArray argslist(fArgs.size());

      for (auto entry : fArgs) {
         Glib::ustring v = entry->get_text();
         argslist.AddLast(new TObjString(v.c_str()));
      }

      close();

      fMenu->Execute(fObject, fFunc, &argslist);

      return;
   }

   Gdk::RGBA selected_color;
   Color_t newColorIdx = 0;
   if (m_color_button) {
      auto selected_color = m_color_button->get_rgba();
      newColorIdx = TColor::GetColor(selected_color.get_red(),
                                     selected_color.get_green(),
                                     selected_color.get_blue(),
                                     selected_color.get_alpha());
   }

   if (fAttLine) {
      fAttLine->SetLineColor(newColorIdx);
      fAttLine->SetLineStyle(m_line_style->get_selected());
      fAttLine->SetLineWidth(m_line_width->get_value());
   } else if (fAttFill) {
      fAttFill->SetFillColor(newColorIdx);
   } else if (fAttText) {
      fAttText->SetTextColor(newColorIdx);
   } else if (fAttMarker) {
      fAttMarker->SetMarkerColor(newColorIdx);
   }

   close();

   if (gPad)
      gPad->ModifiedUpdate();

}
