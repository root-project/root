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


   signal_query_tooltip().connect(sigc::mem_fun(*this, &Gtk4DrawArea::on_my_query_tooltip), true);
}

void Gtk4DrawArea::on_draw(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height)
{
   if (!fCanvas) {
      // only for debug purposes to see not empty widget
      cr->set_source_rgb(0.15, 0.15, 0.15);
      cr->paint();

      // 2. Draw a blue circle in the center that scales with the window
      double radius = std::min(width, height) / 4.0;
      cr->set_source_rgb(0.2, 0.5, 0.8);
      cr->arc(width / 2.0, height / 2.0, radius, 0.0, 2.0 * 3.1415927);
      cr->fill();
      return;
   }

   if ((width != (int) fCanvas->GetPadWidth()) || (height != (int) fCanvas->GetPadHeight()))
      fCanvas->Resize();

   if (!fDrawingCache || fDrawingCache->get_width() != width || fDrawingCache->get_height() != height)
      fDrawingCache = Cairo::ImageSurface::create(Cairo::Surface::Format::ARGB32, width, height);

   // TODO: extend caching for each pad, making it very similar to double-buffer model of gVirtualX

   auto cache_context = Cairo::Context::create(fDrawingCache);

   cache_context->set_source_rgb(1., 1., 1.);
   cache_context->paint();

   // main place where painting happens - only here fContext pointer is set
   // TODO: really repaint pad only when something changed

   fContext = cache_context.get();

   fCanvas->Paint();

   fContext = nullptr;

   // and now copy image to real context
   cr->set_source(fDrawingCache, 0, 0);
   cr->paint();
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
   (void) n_press;

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

   if (fShowEventStatus) {
      TObject *selected = fCanvas->GetSelected();
      Int_t px = fCanvas->GetEventX();
      Int_t py = fCanvas->GetEventY();
      TString buffer = "";
      if (selected) {
         buffer = selected->GetName();
         buffer.Append("  ");
         buffer.Append(selected->GetObjectInfo(px, py));
      } else {
         buffer.Form("No selected object x = %d y = %d", px, py);
      }

      m_status_changed.emit(buffer.Data());
   }
}

sigc::signal<void(const char *)> Gtk4DrawArea::status_changed()
{
   return m_status_changed;
}


bool Gtk4DrawArea::on_mouse_scroll(double dx, double dy)
{
   (void) dx;
   if (fCanvas)
      fCanvas->HandleInput(dy < 0 ? kWheelUp : kWheelDown, (int) fLastMouseX, (int) fLastMouseY);

   return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Shows context menu with provided items and title at specified position

void Gtk4DrawArea::ShowContextMenu(Glib::RefPtr<Gio::Menu> menu_model, const char *title, int x, int y)
{
   if (title && *title) {
      auto item = Gio::MenuItem::create("", "");
      item->set_attribute_value("custom", Glib::Variant<Glib::ustring>::create("header"));
      menu_model->prepend_item(item);
   }

   auto menu = Gtk::make_managed<Gtk::PopoverMenu>(menu_model);
   menu->set_parent(*this);
   menu->set_has_arrow(false); // Removes the little popover triangle point

   menu->add_css_class("compact-menu");

   static bool has_compact_style = false;
   if (!has_compact_style) {
      auto css_provider = Gtk::CssProvider::create();
      std::string css_data =
         "popover.compact-menu contents modelbutton, "
         "popover.compact-menu contents button.model { "
         "  padding-top: 1px; "
         "  padding-bottom: 0px; "
         "  margin-top: 0px; "
         "  margin-bottom: 0px; "
         "  min-height: 10px; "
         "}";

      css_provider->load_from_data(css_data);

      // 4. Inject into the display context
      Gtk::StyleContext::add_provider_for_display(
         Gdk::Display::get_default(),
         css_provider,
         GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
      );

      has_compact_style = true;
   }

   if (title && *title) {
      auto label = Gtk::make_managed<Gtk::Label>(title);
      label->add_css_class("heading");
      menu->add_child(*label, "header");
   }

   Gdk::Rectangle rect(x, y, 1, 1);

   // 2. Snap the context menu anchor to this exact tiny point rectangle
   menu->set_pointing_to(rect);

   // 3. Open the popup menu smoothly on screen
   menu->popup();
}

bool Gtk4DrawArea::on_my_query_tooltip(int x, int y, bool keyboard_tooltip, const Glib::RefPtr<Gtk::Tooltip>& tooltip)
{
   if (keyboard_tooltip)
      return false; // Skip if triggered by keyboard focus

   auto canv = GetCanvas();
   if (!canv)
      return false;

   TObject *selected = canv->GetSelected();
   Int_t px = canv->GetEventX();
   Int_t py = canv->GetEventY();
   TString customText;
   if (selected)
      customText.Form("%s::%s\n%s\n%d, %d\n%s",selected->ClassName() ,selected->GetName(),
                       selected->GetTitle(), px, py, selected->GetObjectInfo(px, py));
   else
      customText.Form("No selected object\n%d, %d", x, y);

   tooltip->set_text(customText.Data());

   return true; // Return true to instruct GTK to show the tooltip
}
