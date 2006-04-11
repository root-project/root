// @(#)root/qtgsi:$Name:$:$Id:$
// Author: Denis Bertini  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQRootGuiFactory
#define ROOT_TQRootGuiFactory

////////////////////////////////////////////////////////////////////
//
//  TQRootGuiFactory
//
//  As TRootGuiFactory from the ROOT library, this
//  class uses the services of the general ABC TGuiFactory
//  in order to get Qt Native GUI components instead of
//  the ROOT ones. Basically it will overrides the
//  member functions:
//    @li TGuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
//                                     UInt_t width, UInt_t height);
//    @li TGuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
//                      Int_t x, Int_t y, UInt_t width, UInt_t height)
//
//@short Qt Factory GUI components
//
//Services:
//@li Creates a specific Canvas Implementation QCanvasImp
////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGuiFactory
#include "TGuiFactory.h"
#endif
#ifndef ROOT_TRootGuiFactory
#include "TRootGuiFactory.h"
#endif
#ifndef ROOT_TQCanvasImp
#include "TQCanvasImp.h"
#endif


class TQRootGuiFactory : public TRootGuiFactory {

private:
   Bool_t fCustom;
public:

   TQRootGuiFactory(const char *name = "QRoot", const char *title = "Qt/ROOT GUI Factory");
   ~TQRootGuiFactory();
   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height);
   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);
   void SetCustomFlag(Bool_t custom) { fCustom=custom; }

};

#endif

