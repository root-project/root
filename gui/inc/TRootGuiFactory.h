// @(#)root/gui:$Name$:$Id$
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootGuiFactory                                                      //
//                                                                      //
// This class is a factory for ROOT GUI components. It overrides        //
// the member functions of the ABS TGuiFactory.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGuiFactory
#include "TGuiFactory.h"
#endif

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
   ~TRootGuiFactory() { }

   TApplicationImp *CreateApplicationImp(const char *classname, int *argc, char **argv, void *option, Int_t numOptions);

   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height);
   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);

   TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height);
   TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);

   TContextMenuImp *CreateContextMenuImp(TContextMenu *c, const char *name, const char *title);

   TControlBarImp *CreateControlBarImp(TControlBar *c, const char *title);
   TControlBarImp *CreateControlBarImp(TControlBar *c, const char *title, Int_t x, Int_t y);

   ClassDef(TRootGuiFactory,0)  //Factory for ROOT GUI components
};

#endif
