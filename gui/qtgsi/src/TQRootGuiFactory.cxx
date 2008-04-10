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
// TQRootGuiFactory                                                     //
//                                                                      //
// This class is a factory for Qt GUI components. It overrides          //
// the member functions of the ABS TGuiFactory.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TQRootGuiFactory.h"
#include "TRootCanvas.h"
#include "TQCanvasImp.h"

ClassImp(TQRootGuiFactory)

//______________________________________________________________________________
TQRootGuiFactory::TQRootGuiFactory(const char *name, const char *title)
   : TRootGuiFactory(name, title)
{
   // TQRootGuiFactory ctor.
   // The default implementation is not customized.
   // The ROOT TRootCanvas class is being used.

   fCustom=kFALSE;

}
//______________________________________________________________________________
TQRootGuiFactory::~TQRootGuiFactory()
{
   //destructor
}

//______________________________________________________________________________
TCanvasImp *TQRootGuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
                                             UInt_t width, UInt_t height)
{
   // Create a ROOT native GUI version of TCanvasImp
   //  @param TCanvas *c (ptr to ROOT TCanvas)
   //  @param char* title (title for canvas)
   //  @param width
   //  @param height
   //  @return QCanvasImp*

   if ( fCustom ) {
      TQCanvasImp* cimp= new TQCanvasImp(c,title,width,height);
      return ( cimp );
   }
   else {
      return new TRootCanvas(c, title, width, height);
   }
}

//______________________________________________________________________________
TCanvasImp *TQRootGuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
                                  Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // Create a ROOT native GUI version of TCanvasImp
   //    @param TCanvas *c (ptr to ROOT TCanvas)
   //    @param char* title (title for canvas)
   //    @param x
   //    @param y
   //    @param width
   //    @param height
   //    @return TQCanvasImp*

   if ( fCustom ) {
      TQCanvasImp* cimp= new TQCanvasImp(c,title,x,y,width,height);
      return cimp ;
   }
   else {
      return new TRootCanvas(c, title, x, y, width, height);
   }
}

