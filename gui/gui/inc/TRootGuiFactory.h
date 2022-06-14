// @(#)root/gui:$Id$
// Author: Fons Rademakers   15/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TRootGuiFactory
#define ROOT_TRootGuiFactory


#include "TGuiFactory.h"

class TApplicationImp;
class TCanvasImp;
class TBrowserImp;
class TContextMenuImp;
class TContextMenu;
class TControlBarImp;
class TControlBar;

class TRootGuiFactory : public TGuiFactory {

public:
   TRootGuiFactory(const char *name = "Root", const char *title = "ROOT GUI Factory");
   virtual ~TRootGuiFactory() {}

   TApplicationImp *CreateApplicationImp(const char *classname, int *argc, char **argv) override;

   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height) override;
   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height) override;

   TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *opt="") override;
   TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt="") override;

   TContextMenuImp *CreateContextMenuImp(TContextMenu *c, const char *name, const char *title) override;

   TControlBarImp *CreateControlBarImp(TControlBar *c, const char *title) override;
   TControlBarImp *CreateControlBarImp(TControlBar *c, const char *title, Int_t x, Int_t y) override;

   ClassDefOverride(TRootGuiFactory,0)  //Factory for ROOT GUI components
};

#endif
