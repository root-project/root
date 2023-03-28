// @(#)root/hist:$Id$
// Author: Olivier Couet   18/05/2022

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TScatter
#define ROOT_TScatter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TScatter                                                             //
//                                                                      //
// A scatter plot able to draw four variables on a single plot          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGraph.h"
class TH1F;
class TScatter : public TGraph {

protected:
   Double_t  *fColor{nullptr};   ///< [fNpoints] array of colors
   Double_t  *fSize{nullptr};    ///< [fNpoints] array of marker sizes
   Double_t   fScale;            ///< Largest marker size used to paint the markers
   Double_t   fMargin;           ///< Margin around the plot in %
   Bool_t     CtorAllocate();
   void       FillZero(Int_t begin, Int_t end, Bool_t from_ctor = kTRUE) override;

public:
   TScatter();
   TScatter(Int_t n);
   TScatter(Int_t n, const Double_t *x, const Double_t *y, const Double_t *col = nullptr, const Double_t *size = nullptr);
   ~TScatter() override;

   Double_t *GetColor() const {return fColor;}   ///< Get the array of colors
   Double_t *GetSize()  const {return fSize;}    ///< Get the array of marker sizes
   Double_t  GetMargin() const {return fMargin;} ///< Set the margin around the plot in %
   Double_t  GetScale() const {return fScale;}   ///< Get the largest marker size used to paint the markers
   TH1F     *GetHistogram() const override;

   void      SetScale(Double_t scale) {fScale = scale;}     ///< Set the largest marker size used to paint the markers
   void      SetMargin(Double_t);
   void      Print(Option_t *chopt="") const override;
   void      SavePrimitive(std::ostream &out, Option_t *option = "") override;


   ClassDefOverride(TScatter,1)  //A scatter plot
};
#endif

