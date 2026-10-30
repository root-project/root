// Author: Sergey Linev, GSI   12/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Gtk4DrawArea_H
#define ROOT_Gtk4DrawArea_H

#include <gtkmm.h>
#include <cairomm/context.h>

class TCanvas;

class Gtk4DrawArea : public Gtk::DrawingArea {
public:
    Gtk4DrawArea();
    virtual ~Gtk4DrawArea() = default;

    void SetCanvas(TCanvas *c) { fCanvas = c; }

    Cairo::Context *GetContext() const { return fContext; }

    void ShowContextMenu(Glib::RefPtr<Gio::Menu> menu_model, int x, int y);

protected:
    // The render function for gtkmm 4
    void on_draw(const Cairo::RefPtr<Cairo::Context>& cr, int width, int height);

    TCanvas *fCanvas = nullptr;
    Cairo::Context *fContext = nullptr;
    double fLastMouseX = 0;
    double fLastMouseY = 0;

    void on_mouse_click(int n_press, double x, double y);
    void on_mouse_released(int n_press, double x, double y);
    void on_mouse_enter(double x, double y);
    void on_mouse_move(double x, double y);
    void on_mouse_leave();
    bool on_mouse_scroll(double dx, double dy);

    Glib::RefPtr<Gtk::GestureClick> m_click_gesture;
    Glib::RefPtr<Gtk::EventControllerMotion> m_motion_controller;
    Glib::RefPtr<Gtk::EventControllerScroll> m_scroll_controller;
};

#endif
