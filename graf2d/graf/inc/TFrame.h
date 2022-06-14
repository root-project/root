// @(#)root/graf:$Id$
// Author: Rene Brun   31/10/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFrame
#define ROOT_TFrame


#include "TWbox.h"


class TFrame : public TWbox {


public:
   TFrame();
   TFrame(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2);
   TFrame(const TFrame &frame);
   virtual ~TFrame();
   void  Copy(TObject &frame) const override;
   void  Draw(Option_t *option="") override;
   void  ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   void  Paint(Option_t *option="") override;
   void  Pop() override;
   void  SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void  UseCurrentStyle() override;  // *MENU*

   ClassDefOverride(TFrame,1)  //Pad graphics frame
};

#endif

