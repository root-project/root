// Author: Sergey Linev, GSI   7/12/2016

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWebGuiFactory
#define ROOT_TWebGuiFactory

#include "TGuiFactory.h"

class TWebGuiFactory : public TGuiFactory {

public:
   TWebGuiFactory();
   virtual ~TWebGuiFactory() = default;

   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height) override;
   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height) override;

   TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *opt = "") override;
   TBrowserImp *CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt = "") override;

   ClassDefOverride(TWebGuiFactory, 0)  //Factory for web-based ROOT GUI components
};


#endif
