// @(#)root/graf:$Id$
// Author: Anna Kreshuk 18/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphQQ
#define ROOT_TGraphQQ


#include "TGraph.h"

class TGraphQQ : public TGraph{
protected:
   Int_t     fNy0{0};      ///< size of the fY0 dataset
   Double_t  fXq1{0.};     ///< x1 coordinate of the interquartile line
   Double_t  fXq2{0.};     ///< x2 coordinate of the interquartile line
   Double_t  fYq1{0.};     ///< y1 coordinate of the interquartile line
   Double_t  fYq2{0.};     ///< y2 coordinate of the interquartile line
   Double_t *fY0{nullptr}; ///<! second dataset, if specified
   TF1      *fF{nullptr};  ///< theoretical density function, if specified

   void      Quartiles();
   void      MakeQuantiles();
   void      MakeFunctionQuantiles();

public:
   TGraphQQ();
   TGraphQQ(Int_t n, Double_t *x);
   TGraphQQ(Int_t n, Double_t *x, TF1 *f);
   TGraphQQ(Int_t nx, Double_t *x, Int_t ny, Double_t *y);
   virtual ~TGraphQQ();

   void SetFunction(TF1 *f);
   Double_t  GetXq1() const {return fXq1;}
   Double_t  GetXq2() const {return fXq2;}
   Double_t  GetYq1() const {return fYq1;}
   Double_t  GetYq2() const {return fYq2;}
   TF1      *GetF()   const {return fF;}

   ClassDefOverride(TGraphQQ, 1); // to create and to draw quantile-quantile plots
};

#endif
