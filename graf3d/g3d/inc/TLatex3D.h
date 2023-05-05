// @(#)root/graf:$Id$
// Author: Olivier Couet   03/05/23

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLatex3D
#define ROOT_TLatex3D


#include "TLatex.h"

class TLatex3D : public TLatex {

protected:
   double fZ{0}; ///< Z position of text

public:

   TLatex3D() {}
   TLatex3D(Double_t x, Double_t y, Double_t z, const char *text);
   virtual ~TLatex3D();
   virtual TLatex3D *DrawText3D(Double_t x, Double_t y, Double_t z, const char *text);
   void ls(Option_t *option="") const override;
   void Paint(Option_t *option="") override;
   void Print(Option_t *option="") const override;

   ClassDefOverride(TLatex3D,1)  //Text 3D
};

#endif
