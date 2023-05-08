// @(#)root/gpad:$Id$
// Author: Rene Brun   01/07/96

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGroupButton
#define ROOT_TGroupButton

#include "TButton.h"

class TGroupButton : public TButton {

private:
   TGroupButton(const TGroupButton &) = delete;
   TGroupButton &operator=(const TGroupButton &) = delete;

public:
   TGroupButton();
   TGroupButton(const char *groupname, const char *title, const char *method, Double_t x1, Double_t y1,Double_t x2 ,Double_t y2);
   virtual ~TGroupButton();
   virtual void  DisplayColorTable(const char *action, Double_t x0, Double_t y0, Double_t wc, Double_t hc);
   virtual void  ExecuteAction();
           void  ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
           void  SavePrimitive(std::ostream &out, Option_t *option = "") override;
   ClassDefOverride(TGroupButton,0)  //A user interface button in a group of buttons.
};

#endif

