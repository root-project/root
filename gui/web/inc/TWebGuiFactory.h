// Author: Sergey Linev, GSI   7/12/2016

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWebGuiFactory
#define ROOT_TWebGuiFactory

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWebGuiFactory                                                       //
//                                                                      //
// This class is a proxy-factory for web-base ROOT GUI components.      //
// It overrides the member functions of the X11/win32gdk-based          //
// TRootGuiFactory.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGuiFactory
#include "TGuiFactory.h"
#endif

class TWebGuiFactory : public TGuiFactory {

private:
   TGuiFactory *fGuiProxy;

public:
   TWebGuiFactory();
   virtual ~TWebGuiFactory();

   virtual TApplicationImp *CreateApplicationImp(const char *classname, int *argc, char **argv);

   virtual TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height);
   virtual TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);

   virtual TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height);
   virtual TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);

   virtual TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *opt);
   virtual TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt);

   virtual TContextMenuImp *CreateContextMenuImp(TContextMenu *c, const char *name, const char *title);

   virtual TControlBarImp *CreateControlBarImp(TControlBar *c, const char *title);
   virtual TControlBarImp *CreateControlBarImp(TControlBar *c, const char *title, Int_t x, Int_t y);

   virtual TInspectorImp *CreateInspectorImp(const TObject *obj, UInt_t width, UInt_t height);

   ClassDef(TWebGuiFactory,0)  //Factory for web-based ROOT GUI components
};


#endif
