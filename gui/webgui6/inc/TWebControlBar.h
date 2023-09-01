// Author: Sergey Linev, GSI   15/12/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWebControlBar
#define ROOT_TWebControlBar

#include "TControlBarImp.h"

#include <ROOT/RWebWindow.hxx>

class TWebControlBar : public TControlBarImp {

protected:

   std::shared_ptr<ROOT::RWebWindow> fWindow; ///!< configured display

   void SendInitMsg(unsigned connid);
   Bool_t ProcessData(unsigned connid, const std::string &arg);

public:
   TWebControlBar(TControlBar *bar, const char *title, Int_t x, Int_t y);
   ~TWebControlBar() override = default;

   void Create() override { }
   void Hide() override;
   void Show() override;
   void SetFont(const char * /*fontName*/) override { }
   void SetTextColor(const char * /*colorName*/) override { }
   void SetButtonState(const char * /*label*/, Int_t /*state*/) override { }
   void SetButtonWidth(UInt_t /*width*/) override { }

   static TControlBarImp *NewControlBar(TControlBar *bar, const char *title, Int_t x, Int_t y);

   ClassDefOverride(TWebControlBar, 0) // Web-based implementation for TControlBarImp
};

#endif
