// @(#)root/graf:$Id$
// Author: Rene Brun   22/06/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDiamond
#define ROOT_TDiamond

#include "TPaveText.h"

class TDiamond :  public TPaveText {

public:
   TDiamond();
   TDiamond(Double_t x1, Double_t y1,Double_t x2, Double_t  y2);
   TDiamond(const TDiamond &diamond);
   virtual ~TDiamond();
   Int_t DistancetoPrimitive(Int_t px, Int_t py) override;
   void  Draw(Option_t *option="") override;
   void  ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   void  Paint(Option_t *option="") override;
   void  SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TDiamond,1)  //Diamond class
};

#endif

