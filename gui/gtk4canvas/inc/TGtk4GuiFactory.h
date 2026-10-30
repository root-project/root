// Author: Sergey Linev   12/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGtk4GuiFactory
#define ROOT_TGtk4GuiFactory

#include "TGuiFactory.h"

namespace ROOT {
namespace Experimental {

class TGtk4GuiFactory : public TGuiFactory {

public:
   TGtk4GuiFactory(const char *name = "gtk4", const char *title = "ROOT Gtk4 GUI Factory");
   ~TGtk4GuiFactory() override {}

   Bool_t UseVirtualX() const override { return kFALSE; }

   TApplicationImp *CreateApplicationImp(const char *classname, int *argc, char **argv) override;

   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height) override;
   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height) override;

   TContextMenuImp *CreateContextMenuImp(TContextMenu *c, const char *name, const char *title) override;

   ClassDefOverride(TGtk4GuiFactory,0)  //Factory for Gtk4 GUI components
};

} //  namespace Experimental
} // namespace ROOT


#endif
