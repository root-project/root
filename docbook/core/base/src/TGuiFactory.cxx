// @(#)root/base:$Id$
// Author: Fons Rademakers   15/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiFactory                                                          //
//                                                                      //
// This ABC is a factory for GUI components. Depending on which         //
// factory is active one gets either ROOT native (X11 based with Win95  //
// look and feel), Win32 or Mac components.                             //
// In case there is no platform dependent implementation on can run in  //
// batch mode directly using an instance of this base class.            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGuiFactory.h"
#include "TApplicationImp.h"
#include "TCanvasImp.h"
#include "TBrowserImp.h"
#include "TContextMenuImp.h"
#include "TControlBarImp.h"
#include "TInspectorImp.h"
#include "TROOT.h"

TGuiFactory *gGuiFactory = 0;
TGuiFactory *gBatchGuiFactory = 0;

ClassImp(TGuiFactory)

//______________________________________________________________________________
TGuiFactory::TGuiFactory(const char *name, const char *title)
    : TNamed(name, title)
{
   // TGuiFactory ctor only called by derived classes.
}

//______________________________________________________________________________
TApplicationImp *TGuiFactory::CreateApplicationImp(const char *classname, int *argc, char **argv)
{
   // Create a batch version of TApplicationImp.

   return new TApplicationImp(classname, argc, argv);
}

//______________________________________________________________________________
TCanvasImp *TGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height)
{
   // Create a batch version of TCanvasImp.

   return new TCanvasImp(c, title, width, height);
}

//______________________________________________________________________________
TCanvasImp *TGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // Create a batch version of TCanvasImp.

   return new TCanvasImp(c, title, x, y, width, height);
}

//______________________________________________________________________________
TBrowserImp *TGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *)
{
   // Create a batch version of TBrowserImp.

   return new TBrowserImp(b, title, width, height);
}

//______________________________________________________________________________
TBrowserImp *TGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *)
{
   // Create a batch version of TBrowserImp.

   return new TBrowserImp(b, title, x, y, width, height);
}

//______________________________________________________________________________
TContextMenuImp *TGuiFactory::CreateContextMenuImp(TContextMenu *c, const char *, const char *)
{
   // Create a batch version of TContextMenuImp.

   return new TContextMenuImp(c);
}

//______________________________________________________________________________
TControlBarImp *TGuiFactory::CreateControlBarImp(TControlBar *c, const char *title)
{
   // Create a batch version of TControlBarImp.

   return new TControlBarImp(c, title);
}

//______________________________________________________________________________
TControlBarImp *TGuiFactory::CreateControlBarImp(TControlBar *c, const char *title, Int_t x, Int_t y)
{
   // Create a batch version of TControlBarImp.

   return new TControlBarImp(c, title, x, y);
}

//______________________________________________________________________________
TInspectorImp *TGuiFactory::CreateInspectorImp(const TObject *obj, UInt_t width, UInt_t height)
{
   // Create a batch version of TInspectorImp.

   if (gROOT->IsBatch()) {
      return new TInspectorImp(obj, width, height);
   }

   gROOT->ProcessLine(Form("TInspectCanvas::Inspector((TObject*)0x%lx);", (ULong_t)obj));
   return 0;
}
