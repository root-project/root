// Author: Sergey Linev, GSI   12/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/* gtkmm example Copyright (C) 2002 gtkmm development team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see <https://www.gnu.org/licenses/>.
 */

#include "Gtk4CanvasWindow.h"

#include <gtkmm.h>
#include <iostream>

#include "TROOT.h"
#include "TTimer.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"



enum { kFileNewCanvas, kFileOpen, kFileSaveAs, kFilePrint, kOptionInterrupt, kOptionRefresh,
       kInspectRoot, kToolsBrowser, kToolModify, kToolArc, kToolLine, kToolArrow, kToolDiamond,
       kToolEllipse, kToolPad, kToolPave, kToolPLabel, kToolPText, kToolPsText, kToolGraph,
       kToolCurlyLine, kToolCurlyArc, kToolLatex, kToolMarker, kToolCutG };

struct ToolBarData_t {
   const char *fPixmap;
   const char *fTipText;
   Bool_t      fStayDown;
   Int_t       fId;
   void       *fButton;
};

static ToolBarData_t toolBarData[] = {
   // { filename,      tooltip,            staydown,  id,              button}
   { "newcanvas.xpm",  "New",              kFALSE,    kFileNewCanvas,  0 },
   { "open.xpm",       "Open",             kFALSE,    kFileOpen,       0 },
   { "save.xpm",       "Save As",          kFALSE,    kFileSaveAs,     0 },
   { "printer.xpm",    "Print",            kFALSE,    kFilePrint,      0 },
   { "",               "",                 kFALSE,    -1,              0 },
   { "interrupt.xpm",  "Interrupt",        kFALSE,    kOptionInterrupt,0 },
   { "refresh2.xpm",   "Refresh",          kFALSE,    kOptionRefresh,  0 },
   { "",               "",                 kFALSE,    -1,              0 },
   { "inspect.xpm",    "Inspect",          kFALSE,    kInspectRoot,    0 },
//   { "browser.xpm",    "Browser",          kFALSE,    kToolsBrowser,   0 },
   { "",                "",                kFALSE,    -1,              0 },
   { "pointer.xpm",    "Modify",           kFALSE,    kToolModify,     0 },
   { "arc.xpm",        "Arc",              kFALSE,    kToolArc,        0 },
   { "line.xpm",       "Line",             kFALSE,    kToolLine,       0 },
   { "arrow.xpm",      "Arrow",            kFALSE,    kToolArrow,      0 },
   { "diamond.xpm",    "Diamond",          kFALSE,    kToolDiamond,    0 },
   { "ellipse.xpm",    "Ellipse",          kFALSE,    kToolEllipse,    0 },
   { "pad.xpm",        "Pad",              kFALSE,    kToolPad,        0 },
   { "pave.xpm",       "Pave",             kFALSE,    kToolPave,       0 },
   { "pavelabel.xpm",  "Pave Label",       kFALSE,    kToolPLabel,     0 },
   { "pavetext.xpm",   "Pave Text",        kFALSE,    kToolPText,      0 },
   { "pavestext.xpm",  "Paves Text",       kFALSE,    kToolPsText,     0 },
   { "graph.xpm",      "Graph",            kFALSE,    kToolGraph,      0 },
   { "curlyline.xpm",  "Curly Line",       kFALSE,    kToolCurlyLine,  0 },
   { "curlyarc.xpm",   "Curly Arc",        kFALSE,    kToolCurlyArc,   0 },
   { "latex.xpm",      "Text/Latex",       kFALSE,    kToolLatex,      0 },
   { "marker.xpm",     "Marker",           kFALSE,    kToolMarker,     0 },
   { "cut.xpm",        "Graphical Cut",    kFALSE,    kToolCutG,       0 },
   { 0,                0,                  kFALSE,    0,               0 }
};



/** \class Gtk4CanvasWindow
    \ingroup gtk4canvas
*/


Gtk4CanvasWindow::Gtk4CanvasWindow(TCanvas *canv, unsigned width, unsigned height)
: m_Box(Gtk::Orientation::VERTICAL),
  m_ToolBar(Gtk::Orientation::HORIZONTAL, 4)
{
  set_title("canvas widget");
  set_default_size(width, height);

  set_child(m_Box); //We can put a MenuBar at the top of the box and other stuff below it.

  // all actions used with canvas


  fillMenuBar(canv);

  fillToolBar();

  m_ToolBar.set_visible(canv->TestBit(TCanvas::kShowToolBar));
  m_Box.append(m_ToolBar);

  m_DrawArea.set_has_tooltip(canv->TestBit(TCanvas::kShowToolTips));
  m_DrawArea.set_hexpand(true);
  m_DrawArea.set_vexpand(true);
  m_Box.append(m_DrawArea);

  m_DrawArea.status_changed().connect(sigc::mem_fun(*this, &Gtk4CanvasWindow::on_status_changed));

  fContextId = m_StatusBar.get_context_id("main_status");
  m_StatusBar.push("Ready to draw", fContextId);
  m_StatusBar.set_visible(canv->TestBit(TCanvas::kShowEventStatus));
  m_Box.append(m_StatusBar);
}

Gtk4CanvasWindow::~Gtk4CanvasWindow()
{
}

void Gtk4CanvasWindow::fillMenuBar(TCanvas *canv)
{
  m_action_group = Gio::SimpleActionGroup::create();

  m_action_group->add_action("new", []() { std::cout << "New canvas Clicked!\n"; });
  m_action_group->add_action("open", [this]() { on_openRootFile(); });

  m_action_group->add_action("close", [this]() { close(); });
  m_action_group->add_action("quit", []() {
     gROOT->SetInterrupt(kTRUE);

     if (gApplication)
        TTimer::SingleShot(100, "TApplication",  gApplication, "Terminate()");
  });

  bool isShowToolbar = canv->TestBit(TCanvas::kShowToolBar);

  bool isShowEventStatus = canv->TestBit(TCanvas::kShowEventStatus);

  bool isShowToolTips = canv->TestBit(TCanvas::kShowToolTips);

  bool isShowMenuBar = canv->TestBit(TCanvas::kMenuBar);

  auto toggle_toolbar = Gio::SimpleAction::create_bool("show-toolbar", isShowToolbar);
  toggle_toolbar->signal_change_state().connect(sigc::mem_fun(*this, &Gtk4CanvasWindow::on_show_toolbar_changed));
  m_action_group->add_action(toggle_toolbar);

  auto toggle_statusbar = Gio::SimpleAction::create_bool("show-eventstatus", isShowEventStatus);
  toggle_statusbar->signal_change_state().connect(sigc::mem_fun(*this, &Gtk4CanvasWindow::on_show_eventstatus_changed));
  m_action_group->add_action(toggle_statusbar);

  auto toggle_tooltip = Gio::SimpleAction::create_bool("show-tooltip", isShowToolTips);
  toggle_tooltip->signal_change_state().connect(sigc::mem_fun(*this, &Gtk4CanvasWindow::on_show_tooltip_changed));
  m_action_group->add_action(toggle_tooltip);


  // Bind actions to an identifier prefix name string ("win.new", "win.copy", etc.)
  insert_action_group("canvas", m_action_group);

  // 3. Construct the Abstract Menu Tree Model
  auto menu_model = Gio::Menu::create();


  // --- SECTION "File"

  auto sec1 = Gio::Menu::create();
  sec1->append("New Canvas", "canvas.new");
  sec1->append("Open", "canvas.open");
  sec1->append("Close Canvas", "canvas.close");

  auto sec2 = Gio::Menu::create();
  sec2->append("Save", "canvas.save");
  sec2->append("Save as", "canvas.saveas");

  auto sec3 = Gio::Menu::create();
  sec3->append("Quit ROOT", "canvas.quit");

  auto file_menu = Gio::Menu::create();
  file_menu->append_section(sec1);
  file_menu->append_section(sec2);
  file_menu->append_section(sec3);

  // Attach File structure onto root Menu Model Bar
  menu_model->append_submenu("File", file_menu);


  // --- SECTION "Edit"
  auto edit_menu = Gio::Menu::create();
  edit_menu->append("Clear pad", "canvas.clearpad");
  edit_menu->append("Clear canvas", "canvas.clearcanvas");

  // CREATE A SUB-MENU ITEM (Nested Layer)
  auto special_paste_submenu = Gio::Menu::create();
  special_paste_submenu->append("Paste Text Only", "canvas.paste");
  special_paste_submenu->append("Paste with Formatting", "canvas.paste");

  // Nest the submenu inside the parent Edit menu container
  edit_menu->append_submenu("Paste Special...", special_paste_submenu);

  // Attach Edit structure onto root Menu Model Bar
  menu_model->append_submenu("Edit", edit_menu);


  // --- SECTION "View"
  auto view_menu = Gio::Menu::create();
  view_menu->append("Editor", "canvas.show-editor");
  view_menu->append("Toolbar", "canvas.show-toolbar");
  view_menu->append("Event Status", "canvas.show-eventstatus");
  view_menu->append("Tooltip", "canvas.show-tooltip");

  menu_model->append_submenu("View", view_menu);

  // 4. Instantiate the PopoverMenuBar from our structured menu tree
  // Gtk::PopoverMenuBar requires a Glib::RefPtr<Gio::MenuModel>
  auto menu_bar = Gtk::make_managed<Gtk::PopoverMenuBar>(menu_model);

  menu_bar->set_visible(isShowMenuBar);

  m_Box.append(*menu_bar);
}


Gtk::Image* create_resized_xpm_image(const char *filename)
{
    // 2. Initialize a pointer for our final texture image
    Glib::RefPtr<Gdk::Texture> texture;

    try {
        // Load the original XPM file safely into a Pixbuf
        auto original_pixbuf = Gdk::Pixbuf::create_from_file(filename);

        // Force-resize the XPM image to precisely 16x16 using Bilinear interpolation
        auto scaled_pixbuf = original_pixbuf->scale_simple(16, 16, Gdk::InterpType::BILINEAR);

        // Convert the scaled Pixbuf into a GTK4 Paintable Texture
        texture = Gdk::Texture::create_for_pixbuf(scaled_pixbuf);

    } catch (const Glib::FileError& e) {
        std::cerr << "File system error: " << e.what() << std::endl;
        return nullptr; // Returns an empty image
    } catch (const Gdk::PixbufError& e) {
        std::cerr << "Failed to parse XPM image format: " << e.what() << std::endl;
        return nullptr; // Returns an empty image
    }

    // 3. Create the GTK4 Image widget directly from the validated texture
    auto image = Gtk::make_managed<Gtk::Image>(texture);

    return image;
}


void Gtk4CanvasWindow::fillToolBar()
{

   auto css_provider = Gtk::CssProvider::create();

   // Target the button widget structure to minimize interior padding spaces
   css_provider->load_from_data("button { padding: 2px 2px; min-width: 0; min-height: 0; }");

   for (int i = 0; toolBarData[i].fPixmap; i++) {
      auto btn = &(toolBarData[i]);
      if (strlen(btn->fPixmap) == 0) {
         // fToolBar->addSeparator();
         auto separator = Gtk::make_managed<Gtk::Separator>(Gtk::Orientation::VERTICAL);
         m_ToolBar.append(*separator);
         continue;
      }

      auto button = Gtk::make_managed<Gtk::Button>();

      TString iconname = TROOT::GetIconPath();
      iconname.Append("/");
      iconname.Append(btn->fPixmap);

      // 2. Load the icon from the file path
      auto image = create_resized_xpm_image(iconname.Data());

      if (image)
         button->set_child(*image);

      button->get_style_context()->add_provider(css_provider, GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);

      button->set_tooltip_text(btn->fTipText);

      button->signal_clicked().connect([this, btn]() {
         clickToolButton(btn->fId);
      });


      m_ToolBar.append(*button);
   }
}

void Gtk4CanvasWindow::clickToolButton(int id)
{
   auto canv = m_DrawArea.GetCanvas();
   switch (id) {
   case kFileNewCanvas:
      gROOT->MakeDefCanvas();
      break;
   case kFileOpen:
      on_openRootFile();
      break;
   case kFileSaveAs:
      on_saveCanvasAs();
      break;
   case kFilePrint:
      on_printCanvas();
      break;
   case kOptionInterrupt: gROOT->SetInterrupt(); break;
   case kOptionRefresh:
      if (canv) {
         canv->Modified(kTRUE);
         // trigger update of the paint widget, paint will be called
         canv->Update();
      }
      break;
   case kInspectRoot:
      if (canv) {
         canv->cd();
         gROOT->Inspect();
         canv->Update();
      }
      break;

   case kToolsBrowser: break;
   case kToolModify: gROOT->SetEditorMode(); break;
   case kToolArc: gROOT->SetEditorMode("Arc"); break;
   case kToolLine: gROOT->SetEditorMode("Line"); break;
   case kToolArrow: gROOT->SetEditorMode("Arrow"); break;
   case kToolDiamond: gROOT->SetEditorMode("Diamond"); break;
   case kToolEllipse: gROOT->SetEditorMode("Ellipse"); break;
   case kToolPad: gROOT->SetEditorMode("Pad"); break;
   case kToolPave: gROOT->SetEditorMode("Pave"); break;
   case kToolPLabel: gROOT->SetEditorMode("PaveLabel"); break;
   case kToolPText: gROOT->SetEditorMode("PaveText"); break;
   case kToolPsText: gROOT->SetEditorMode("PavesText"); break;
   case kToolGraph: gROOT->SetEditorMode("PolyLine"); break;
   case kToolCurlyLine: gROOT->SetEditorMode("CurlyLine"); break;
   case kToolCurlyArc: gROOT->SetEditorMode("CurlyArc"); break;
   case kToolLatex: gROOT->SetEditorMode("Text"); break;
   case kToolMarker: gROOT->SetEditorMode("Marker"); break;
   case kToolCutG: gROOT->SetEditorMode("CutG"); break;
   }
}

void Gtk4CanvasWindow::on_status_changed(const char *msg)
{
   m_StatusBar.remove_all_messages(fContextId);
   m_StatusBar.push(msg, fContextId);
}

void Gtk4CanvasWindow::on_show_toolbar_changed(const Glib::VariantBase& value)
{
   bool is_checked = Glib::VariantBase::cast_dynamic<Glib::Variant<bool>>(value).get();

   m_ToolBar.set_visible(is_checked);

   // Remember to update the actual action state so the checkmark reflects visually
   auto action = m_action_group->lookup_action("show-toolbar");
   if (action) {
      auto simple_action = std::dynamic_pointer_cast<Gio::SimpleAction>(action);
      simple_action->set_state(value);
   }
}

void Gtk4CanvasWindow::on_show_eventstatus_changed(const Glib::VariantBase &value)
{
   bool is_checked = Glib::VariantBase::cast_dynamic<Glib::Variant<bool>>(value).get();

   m_StatusBar.set_visible(is_checked);

   // Remember to update the actual action state so the checkmark reflects visually
   auto action = m_action_group->lookup_action("show-eventstatus");
   if (action) {
      auto simple_action = std::dynamic_pointer_cast<Gio::SimpleAction>(action);
      simple_action->set_state(value);
   }
}

void Gtk4CanvasWindow::on_show_tooltip_changed(const Glib::VariantBase &value)
{
   bool is_checked = Glib::VariantBase::cast_dynamic<Glib::Variant<bool>>(value).get();

   m_DrawArea.set_has_tooltip(is_checked);

   // Remember to update the actual action state so the checkmark reflects visually
   auto action = m_action_group->lookup_action("show-tooltip");
   if (action) {
      auto simple_action = std::dynamic_pointer_cast<Gio::SimpleAction>(action);
      simple_action->set_state(value);
   }

}

void Gtk4CanvasWindow::on_openRootFile()
{
   if (!m_file_dialog) {
      m_file_dialog = Gtk::FileDialog::create();
      m_file_dialog->set_title("Select a ROOT File");
      m_file_dialog->set_modal(true);

      auto filter_root_files = Gtk::FileFilter::create();
      filter_root_files->set_name("ROOT files (*.root)");
      filter_root_files->add_pattern("*.root");

      auto filter_xml_files = Gtk::FileFilter::create();
      filter_xml_files->set_name("ROOT XML files (*.xml)");
      filter_xml_files->add_pattern("*.xml");

      auto filter_all_files = Gtk::FileFilter::create();
      filter_all_files->set_name("All files (*.*)");
      filter_all_files->add_pattern("*.*");

      auto filters_list = Gio::ListStore<Gtk::FileFilter>::create();
      filters_list->append(filter_root_files);
      filters_list->append(filter_xml_files);
      filters_list->append(filter_all_files);

      m_file_dialog->set_filters(filters_list);

      // Optional: Set which filter is selected by default (e.g., Images)
      m_file_dialog->set_default_filter(filter_root_files);
   }

   m_file_dialog->open(*this, sigc::mem_fun(*this, &Gtk4CanvasWindow::on_file_selected));
}

void Gtk4CanvasWindow::on_file_selected(const Glib::RefPtr<Gio::AsyncResult> &result)
{
   try {
      // Retrieve the selected file reference
      Glib::RefPtr<Gio::File> file = m_file_dialog->open_finish(result);

      std::cout << "Selected File Path: " << file->get_path() << std::endl;

      TFile::Open(file->get_path().c_str(), "update");

   } catch (const Gtk::DialogError &err) {
      // Triggers if the user cancels or closes the dialog window
      // std::cout << "Dialog closed or dismissed: " << err.what() << std::endl;
   }
}

void Gtk4CanvasWindow::on_saveCanvasAs()
{
}

void Gtk4CanvasWindow::on_printCanvas()
{
}
