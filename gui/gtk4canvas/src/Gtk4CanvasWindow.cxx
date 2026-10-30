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
#include "TApplication.h"


/** \class Gtk4CanvasWindow
    \ingroup gtk4canvas
*/


Gtk4CanvasWindow::Gtk4CanvasWindow(unsigned width, unsigned height)
: m_Box(Gtk::Orientation::VERTICAL)
{
  set_title("canvas widget");
  set_default_size(width, height);

  set_child(m_Box); //We can put a MenuBar at the top of the box and other stuff below it.

  m_action_group = Gio::SimpleActionGroup::create();

  m_action_group->add_action("new", []() { std::cout << "New canvas Clicked!\n"; });
  m_action_group->add_action("open", []() { std::cout << "Open File Clicked!\n"; });

  m_action_group->add_action("close", [this]() { close(); });
  m_action_group->add_action("quit", []() {

    printf("Quit ROOT action\n");
     gROOT->SetInterrupt(kTRUE);

     if (gApplication)
        TTimer::SingleShot(100, "TApplication",  gApplication, "Terminate()");
  });

  // Bind actions to an identifier prefix name string ("win.new", "win.copy", etc.)
  insert_action_group("win", m_action_group);

  // 3. Construct the Abstract Menu Tree Model
  auto menu_model = Gio::Menu::create();

  // --- SECTION A: The "File" Top Level Dropdown Menu ---

  auto sec1 = Gio::Menu::create();
  sec1->append("New Canvas", "win.new");
  sec1->append("Open", "win.open");
  sec1->append("Close Canvas", "win.close");

  auto sec2 = Gio::Menu::create();
  sec2->append("Save", "win.save");
  sec2->append("Save as", "win.saveas");

  auto sec3 = Gio::Menu::create();
  sec3->append("Quit ROOT", "win.quit");

  auto file_menu = Gio::Menu::create();
  file_menu->append_section(sec1);
  file_menu->append_section(sec2);
  file_menu->append_section(sec3);


  // Attach File structure onto root Menu Model Bar
  menu_model->append_submenu("File", file_menu);

  // --- SECTION B: The "Edit" Top Level Dropdown Menu ---
  auto edit_menu = Gio::Menu::create();
  edit_menu->append("Clear pad", "win.clearpad");
  edit_menu->append("Clear canvas", "win.clearcanvas");

  // CREATE A SUB-MENU ITEM (Nested Layer)
  auto special_paste_submenu = Gio::Menu::create();
  special_paste_submenu->append("Paste Text Only", "win.paste");
  special_paste_submenu->append("Paste with Formatting", "win.paste");

  // Nest the submenu inside the parent Edit menu container
  edit_menu->append_submenu("Paste Special...", special_paste_submenu);

  // Attach Edit structure onto root Menu Model Bar
  menu_model->append_submenu("Edit", edit_menu);

  // 4. Instantiate the PopoverMenuBar from our structured menu tree
  // Gtk::PopoverMenuBar requires a Glib::RefPtr<Gio::MenuModel>
  auto menu_bar = Gtk::make_managed<Gtk::PopoverMenuBar>(menu_model);

  // 5. Append the layout widget to the window canvas
  m_Box.append(*menu_bar);


  m_DrawArea.set_hexpand(true);
  m_DrawArea.set_vexpand(true);
  m_Box.append(m_DrawArea);


  unsigned int context_id = m_StatusBar.get_context_id("main_status");
  m_StatusBar.push("Ready to draw", context_id);
  m_Box.append(m_StatusBar);
}

Gtk4CanvasWindow::~Gtk4CanvasWindow()
{
}

/*

void Gtk4CanvasWindow::on_action_file_quit()
{
  close();
}

void Gtk4CanvasWindow::on_action_toggle()
{
  std::cout << "The toggle menu item was selected." << std::endl;

  bool active = false;
  m_refActionRain->get_state(active);

  //The toggle action's state does not change automatically:
  active = !active;
  m_refActionRain->change_state(active);

  Glib::ustring message;
  if(active)
    message = "Toggle is active.";
  else
    message = "Toggle is not active";

  std::cout << message << std::endl;

}

*/