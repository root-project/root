// @(#)root/win32:$Name$:$Id$
// Author: Rene Brun   11/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//________________________________________________________________________
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32GuiFactory                                                     //
//                                                                      //
// This class is a factory for Win32 GUI components. It overrides       //
// the member functions of the ABS TGuiFactory.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TWin32GuiFactory.h"
#include "TWin32Canvas.h"
#include "TWin32Application.h"
#include "TWin32ContextMenuImp.h"
#include "TWin32ControlBarImp.h"
#include "TWin32BrowserImp.h"
#include "TWin32InspectImp.h"

ClassImp(TWin32GuiFactory)

//______________________________________________________________________________
TWin32GuiFactory::TWin32GuiFactory(const char *name, const char *title)
   : TGuiFactory(name, title)
{
   // TWin32GuiFactory ctor.
}

//______________________________________________________________________________
   TApplicationImp *TWin32GuiFactory::CreateApplicationImp(const char *classname, int *argc, char **argv, void *option, Int_t numOptions)
{
   return new TWin32Application(classname,argc,argv, option, numOptions);
}

//______________________________________________________________________________
TCanvasImp *TWin32GuiFactory::CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height)
{
   // Create a Win32 version of TCanvasImp

   return new TWin32Canvas(c, title, width, height);
}

//______________________________________________________________________________
TCanvasImp *TWin32GuiFactory::CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // Create a Win32 version of TCanvasImp

   return new TWin32Canvas(c, title, x, y, width, height);
}

//______________________________________________________________________________
TBrowserImp *TWin32GuiFactory::CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height)
{
   return new TWin32BrowserImp(b,title,width, height);
}

//______________________________________________________________________________
TBrowserImp *TWin32GuiFactory::CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   return new TWin32BrowserImp(b,title,x,y,width, height);
}

//______________________________________________________________________________
TContextMenuImp *TWin32GuiFactory::CreateContextMenuImp(TContextMenu *c, const char *name, const char *title )
{
   return new TWin32ContextMenuImp(c);
}

//______________________________________________________________________________
TControlBarImp *TWin32GuiFactory::CreateControlBarImp(TControlBar *c, const char *title )
{
  return new TWin32ControlBarImp(c);
}

//______________________________________________________________________________
TControlBarImp *TWin32GuiFactory::CreateControlBarImp(TControlBar *c, const char *title, Int_t x, Int_t y )
{
   return new TWin32ControlBarImp(c, x, y);
}

//______________________________________________________________________________
TInspectorImp *TWin32GuiFactory::CreateInspectorImp(const TObject *obj, UInt_t width, UInt_t height)
{
   return new TWin32InspectImp(obj, "Inspector", width, height);
}
