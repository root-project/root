// Author: Sergey Linev   2/07/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TQt6GuiFactory
#define ROOT_TQt6GuiFactory


#include "TGuiFactory.h"


class TQt6GuiFactory : public TGuiFactory {

protected:
   void ShowWebCanvasWarning();

public:
   TQt6GuiFactory(const char *name = "RootQt6", const char *title = "ROOT Qt6 GUI Factory");
   ~TQt6GuiFactory() override {}

   TApplicationImp *CreateApplicationImp(const char *classname, int *argc, char **argv) override;

   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height) override;
   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height) override;

   TContextMenuImp *CreateContextMenuImp(TContextMenu *c, const char *name, const char *title) override;

   ClassDefOverride(TQt6GuiFactory,0)  //Factory for Qt6 GUI components
};

#endif
