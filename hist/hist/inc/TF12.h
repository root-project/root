// @(#)root/hist:$Id$
// Author: Rene Brun   05/04/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TF12
#define ROOT_TF12

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TF12                                                                 //
//                                                                      //
// Projection of a TF2 along x or y                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TF2.h"

class TF12 : public TF1 {

protected:
   Double_t    fXY;           ///< Value along Y (if projection X) or X (if projection Y)
   Int_t       fCase;         ///< Projection along X(0), or Y(1)
   TF2        *fF2;           ///< Pointer to the mother TF2

public:
   TF12();
   TF12(const char *name, TF2 *f2, Double_t xy, Option_t *option="x");
   TF12(const TF12 &f12);
     ~TF12() override;
   void     Copy(TObject &f12) const override;
   TF1     *DrawCopy(Option_t *option="") const override;
   Double_t Eval(Double_t x, Double_t y=0, Double_t z=0, Double_t t=0) const override;
   Double_t EvalPar(const Double_t *x, const Double_t *params=0) override;

#ifdef R__HAS_VECCORE
   using TF1::Eval;    // to not hide the vectorized version
   using TF1::EvalPar; // to not hide the vectorized version
#endif

   virtual Double_t GetXY() const {return fXY;}
   void     SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void     SetXY(Double_t xy);  // *MENU*

   ClassDefOverride(TF12,1)  //Projection of a TF2 along x or y
};

#endif
