// @(#)root/win32:$Name:  $:$Id: TWin32GuiFactory.h,v 1.1.1.1 2000/05/16 17:00:47 rdm Exp $
// Author: Rene Brun   11/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32GuiFactory
#define ROOT_TWin32GuiFactory

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32GuiFactory                                                     //
//                                                                      //
// This class is a factory for Win32 GUI components. It overrides       //
// the member functions of the ABS TGuiFactory.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGuiFactory
#include "TGuiFactory.h"
#endif


class TWin32GuiFactory : public TGuiFactory {

public:
   TWin32GuiFactory() { }
   TWin32GuiFactory(const char *name, const char *title);
   virtual ~TWin32GuiFactory() { }

   TApplicationImp *CreateApplicationImp(const char *classname, int *argc, char **argv);

   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height);
   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);

   TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height);
   TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);

   TContextMenuImp *CreateContextMenuImp( TContextMenu *c, const char *name, const char *title );

   TControlBarImp *CreateControlBarImp( TControlBar *c, const char *title );
   TControlBarImp *CreateControlBarImp( TControlBar *c, const char *title, Int_t x, Int_t y );

   TInspectorImp *CreateInspectorImp(const TObject *obj, UInt_t width, UInt_t height);

   ClassDef(TWin32GuiFactory,0)  //Factory for Win32 GUI components
};

#endif
