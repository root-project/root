// Author: Sergey Linev, GSI   12/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Gtk4DrawArea.h"

#include <cmath>

#include "TGtk4PadPainter.h"
#include "TCanvas.h"

#include <gtkmm.h>

/** \class Gtk4DrawArea
    \ingroup gtk4canvas
*/


Gtk4DrawArea::Gtk4DrawArea()
{
   // Bind the member function as the draw function
   set_draw_func(sigc::mem_fun(*this, &Gtk4DrawArea::on_draw));

   set_expand(true);

   m_click_gesture = Gtk::GestureClick::create();
   m_click_gesture->set_button(0);
   m_click_gesture->signal_pressed().connect(
       sigc::mem_fun(*this, &Gtk4DrawArea::on_mouse_click)
   );
   m_click_gesture->signal_released().connect(
       sigc::mem_fun(*this, &Gtk4DrawArea::on_mouse_released)
   );
   add_controller(m_click_gesture);

   // 2. Create and attach the motion event controller
   m_motion_controller = Gtk::EventControllerMotion::create();
   m_motion_controller->signal_motion().connect(
      sigc::mem_fun(*this, &Gtk4DrawArea::on_mouse_move)
   );

   m_motion_controller->signal_enter().connect(
        sigc::mem_fun(*this, &Gtk4DrawArea::on_mouse_enter)
    );

    // 3. Connect the leave signal
    m_motion_controller->signal_leave().connect(
        sigc::mem_fun(*this, &Gtk4DrawArea::on_mouse_leave)
    );
   add_controller(m_motion_controller);

   m_scroll_controller = Gtk::EventControllerScroll::create();
   m_scroll_controller->set_flags(Gtk::EventControllerScroll::Flags::VERTICAL);

   m_scroll_controller->signal_scroll().connect(
      sigc::mem_fun(*this, &Gtk4DrawArea::on_mouse_scroll),
      false // Run during the capture phase? False means bubble phase (standard)
   );

   add_controller(m_scroll_controller);
}

void Gtk4DrawArea::on_draw(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height)
{
    if (!fCanvas) {
       cr->set_source_rgb(0.15, 0.15, 0.15);
       cr->paint();

       // 2. Draw a blue circle in the center that scales with the window
       double radius = std::min(width, height) / 4.0;
       cr->set_source_rgb(0.2, 0.5, 0.8);
       cr->arc(width / 2.0, height / 2.0, radius, 0.0, 2.0 * M_PI);
       cr->fill();
       return;
    }

    if ((width != fCanvas->GetPadWidth()) || (height != fCanvas->GetPadHeight())) {
       fCanvas->Resize();
       // printf("After resize user coordinates pX %d pY %d  relative coordinates pX %d pY %d\n",
       //        fCanvas->XtoAbsPixel(0.5), fCanvas->YtoAbsPixel(0.5),
       //         fCanvas->UtoAbsPixel(0.5), fCanvas->VtoAbsPixel(0.5));
    }


    // clear
    cr->set_source_rgb(1., 1., 1.);
    cr->paint();

    fContext = cr.get();

    // TODO: call canvas->paint instead
    fCanvas->Paint();

    fContext = nullptr;
}

void Gtk4DrawArea::on_mouse_click(int n_press, double x, double y)
{
   if (!fCanvas)
      return;

   auto evnt = kNoEvent;

   switch(m_click_gesture->get_current_button()) {
      case GDK_BUTTON_PRIMARY:   // 1, usually left
         evnt = n_press > 1 ? kButton1Double : kButton1Down;
         break;
      case GDK_BUTTON_MIDDLE:    // 2
         evnt = n_press > 1 ? kButton2Double : kButton2Down;
         break;
      case GDK_BUTTON_SECONDARY: // 3, usually right
         evnt = n_press > 1 ? kButton3Double : kButton3Down;
         break;
      default:
         break;
   }

   if (evnt != kNoEvent)
      fCanvas->HandleInput(evnt, (int) x, (int) y);
}

void Gtk4DrawArea::on_mouse_released(int n_press, double x, double y)
{
   if (!fCanvas)
      return;

   auto evnt = kNoEvent;

   switch(m_click_gesture->get_current_button()) {
      case GDK_BUTTON_PRIMARY:   // 1, usually left
         evnt = kButton1Up;
         break;
      case GDK_BUTTON_MIDDLE:    // 2
         evnt = kButton2Up;
         break;
      case GDK_BUTTON_SECONDARY: // 3, usually right
         evnt = kButton3Up;
         break;
      default:
         break;
   }

   if (evnt != kNoEvent)
      fCanvas->HandleInput(evnt, (int) x, (int) y);
}


void Gtk4DrawArea::on_mouse_enter(double x, double y)
{
   fLastMouseX = x;
   fLastMouseY = y;

   if (fCanvas)
      fCanvas->HandleInput(kMouseEnter, (int) x, (int) y);
}

void Gtk4DrawArea::on_mouse_leave()
{
   if (fCanvas)
      fCanvas->HandleInput(kMouseLeave, 0, 0);
}


// Callback handles mouse movements
void Gtk4DrawArea::on_mouse_move(double x, double y)
{
   if (!fCanvas)
      return;

   fLastMouseX = x;
   fLastMouseY = y;

   Gdk::ModifierType modifiers = m_motion_controller->get_current_event_state();

   auto evnt = kMouseMotion;
   if ((modifiers & Gdk::ModifierType::BUTTON1_MASK) == Gdk::ModifierType::BUTTON1_MASK) {
      evnt = kButton1Motion;
   } else if ((modifiers & Gdk::ModifierType::BUTTON3_MASK) == Gdk::ModifierType::BUTTON3_MASK) {
      evnt = kButton3Motion;
   }

   fCanvas->HandleInput(evnt, (int) x, (int) y);
}

bool Gtk4DrawArea::on_mouse_scroll(double dx, double dy)
{
   if (fCanvas)
      fCanvas->HandleInput(dy < 0 ? kWheelUp : kWheelDown, (int) fLastMouseX, (int) fLastMouseY);

   return true;
}


void Gtk4DrawArea::ShowContextMenu(Glib::RefPtr<Gio::Menu> menu_model, int x, int y)
{
   auto menu = Gtk::make_managed<Gtk::PopoverMenu>(menu_model);
   menu->set_parent(*this);
   menu->set_has_arrow(false); // Removes the little popover triangle point

   Gdk::Rectangle rect(x, y, 1, 1);

   // 2. Snap the context menu anchor to this exact tiny point rectangle
   menu->set_pointing_to(rect);

   // 3. Open the popup menu smoothly on screen
   menu->popup();

}
