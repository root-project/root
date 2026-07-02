// Author: Sergey Linev   2/07/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TQt6GuiFactory
    \ingroup guiwidgets

This class is a factory for ROOT GUI components. It overrides
the member functions of the ABS TGuiFactory.

*/


#include "TQt6GuiFactory.h"
#include "TQt6Canvas.h"

#include <iostream>


////////////////////////////////////////////////////////////////////////////////
/// TQt6GuiFactory ctor.

TQt6GuiFactory::TQt6GuiFactory(const char *name, const char *title)
   : TGuiFactory(name, title)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TApplicationImp

TApplicationImp *TQt6GuiFactory::CreateApplicationImp(const char *classname,
                      Int_t *argc, char **argv)
{
   return TGuiFactory::CreateApplicationImp(classname, argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TCanvasImp

TCanvasImp *TQt6GuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
                                             UInt_t width, UInt_t height)
{
   return TQt6Canvas::NewCanvas(c, title, -1, -1, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TCanvasImp

TCanvasImp *TQt6GuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
                                  Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   return TQt6Canvas::NewCanvas(c, title, x, y, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TContextMenuImp

TContextMenuImp *TQt6GuiFactory::CreateContextMenuImp(TContextMenu *c,
                                             const char *name, const char *title)
{
   return TGuiFactory::CreateContextMenuImp(c, name, title);
}
