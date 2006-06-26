// @(#)root/graf:$Name:  $:$Id: TGraphQQ.h,v 1.2 2005/11/21 08:49:48 couet Exp $
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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphQQ                                                             //
//                                                                      //
// to create and to draw quantile-quantile plots                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGraph
#include "TGraph.h"
#endif

class TGraphQQ : public TGraph{
protected:
   Int_t     fNy0;    //size of the fY0 dataset
   Double_t  fXq1;    //x1 coordinate of the interquartile line
   Double_t  fXq2;    //x2 coordinate of the interquartile line
   Double_t  fYq1;    //y1 coordinate of the interquartile line
   Double_t  fYq2;    //y2 coordinate of the interquartile line
   Double_t *fY0;     //!second dataset, if specified
   TF1      *fF;      //theoretical density function, if specified
   
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
   void Paint(Option_t *opt="");

   ClassDef(TGraphQQ, 1); // to create and to draw quantile-quantile plots
};

#endif
