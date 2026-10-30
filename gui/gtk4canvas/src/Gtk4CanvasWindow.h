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

#ifndef ROOT_Gtk4CanvasWindow_H
#define ROOT_Gtk4CanvasWindow_H

#include <gtkmm.h>

#include "Gtk4DrawArea.h"

class Gtk4CanvasWindow : public Gtk::Window
{
public:
  Gtk4CanvasWindow(unsigned width = 400, unsigned height = 200);
  virtual ~Gtk4CanvasWindow();

  Gtk4DrawArea *GetDrawArea() { return &m_DrawArea; }

private:
  //Signal handlers:
  //void on_action_file_new();
  //void on_action_file_quit();
  //void on_action_others();
  //void on_action_toggle();


  //Child widgets:
  Gtk::Box m_Box;

  Glib::RefPtr<Gio::SimpleActionGroup> m_action_group;

  Gtk4DrawArea  m_DrawArea;
  Gtk::Statusbar m_StatusBar;

};

#endif //GTKMM_Gtk4CanvasWindow_H
