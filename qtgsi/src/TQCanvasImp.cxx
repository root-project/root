// @(#)root/qtgsi:$Id$
// Author: Denis Bertini, M. Al-Turany  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ABC describing GUI independent main window (with menubar, scrollbars //
// and a drawing area).                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TQCanvasImp.h"
#include "TQRootCanvas.h"

ClassImp(TQCanvasImp)

//______________________________________________________________________________
TQCanvasImp::TQCanvasImp(TCanvas *c, const char *name, UInt_t width, UInt_t height)
{
   //  TQCanvasImp ctor

   //  @param c (ptr to ROOT TCanvas)
   //  @param name (title for canvas)
   //  @param width
   //  @param height

   Build(c,name,10,10,width,height);
}

//______________________________________________________________________________
TQCanvasImp::TQCanvasImp(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   //   TQCanvasImp ctor

   //   @param c (ptr to ROOT TCanvas)
   //   @param name (title for canvas)
   //   @param x
   //   @param y
   //   @param width
   //   @param height

   Build(c,name,x,y,width,height);
}

//______________________________________________________________________________
TQCanvasImp::TQCanvasImp(TCanvas* /*v*/) : TCanvasImp()
{
   // TQCanvasImp ctor

}

//______________________________________________________________________________
void TQCanvasImp::Build(TCanvas *c, const char *name, Int_t /*x*/, Int_t /*y*/, UInt_t /*width*/,
                        UInt_t /*height*/)
{
   // Build the TQRootCanvas

   fQCanvas = new TQRootCanvas(0,name,c);
   fCanvas = fQCanvas->GetCanvas();
}

//______________________________________________________________________________
TQCanvasImp::~TQCanvasImp()
{
   // dtor
}
