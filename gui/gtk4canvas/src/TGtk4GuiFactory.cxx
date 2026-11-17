// Author: Sergey Linev   12/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TGtk4GuiFactory
    \ingroup gtk4canvas

This class is a factory for ROOT GUI components.
For Gtk4 it provides specialization for TCanvasImp and TContextMenuImp classes
*/


#include "TGtk4GuiFactory.h"

#include "TGtk4Canvas.h"
#include "TGtk4ContextMenu.h"

using namespace ROOT::Experimental;

////////////////////////////////////////////////////////////////////////////////
/// TGtk4GuiFactory ctor.

TGtk4GuiFactory::TGtk4GuiFactory(const char *name, const char *title)
   : TGuiFactory(name, title)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TApplicationImp

TApplicationImp *TGtk4GuiFactory::CreateApplicationImp(const char *classname,
                      Int_t *argc, char **argv)
{
   return TGuiFactory::CreateApplicationImp(classname, argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TCanvasImp

TCanvasImp *TGtk4GuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
                                             UInt_t width, UInt_t height)
{
   return TGtk4Canvas::NewCanvas(c, title, -1, -1, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TCanvasImp

TCanvasImp *TGtk4GuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
                                  Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   return TGtk4Canvas::NewCanvas(c, title, x, y, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TContextMenuImp

TContextMenuImp *TGtk4GuiFactory::CreateContextMenuImp(TContextMenu *c,
                                             const char *name, const char *)
{
   return new TGtk4ContextMenu(c, name);
}
