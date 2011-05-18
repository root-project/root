// @(#)root/hist:$Id$
// Author: Christian Stratowa 30/09/2001

/******************************************************************************
* Copyright(c) 2001-    , Dr. Christian Stratowa, Vienna, Austria.            *
* Author: Christian Stratowa with help from Rene Brun.                                                 *
*                                                                             *
* Algorithms for smooth regression adapted from:                              *
* R: A Computer Language for Statistical Data Analysis                        *
* Copyright (C) 1996 Robert Gentleman and Ross Ihaka                          *
* Copyright (C) 1999-2001 Robert Gentleman, Ross Ihaka and the                *
* R Development Core Team                                                     *
* R is free software, for licensing see the GNU General Public License        *
* http://www.ci.tuwien.ac.at/R-project/welcome.html                           *
*                                                                             *
* Based on: "The ROOT System", All rights reserved.                           *
* Authors: Rene Brun and Fons Rademakers.                                     *
* For the licensing terms of "The ROOT System" see $ROOTSYS/AA_LICENSE.       *
* For the list of contributors to "The ROOT System" see $ROOTSYS/AA_CREDITS.  *
******************************************************************************/

#ifndef ROOT_TGraphSmooth
#define ROOT_TGraphSmooth

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphSmooth                                                         //
//                                                                      //
// Class for different regression smoothers                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGraph
#include "TGraph.h"
#endif


class TGraphSmooth: public TNamed {

private:
   TGraphSmooth(const TGraphSmooth&); // Not implented
   TGraphSmooth& operator=(const TGraphSmooth&); // Not implented

protected:
   Int_t       fNin;        //Number of input points
   Int_t       fNout;       //Number of output points
   TGraph     *fGin;        //Input graph
   TGraph     *fGout;       //Output graph
   Double_t    fMinX;       //Minimum value of array X
   Double_t    fMaxX;       //Maximum value of array X

public :
   TGraphSmooth();
   TGraphSmooth(const char *name);
//      TGraphSmooth(const TGraphSmooth &smoothReg);   //??
   virtual ~TGraphSmooth();

   TGraph         *Approx(TGraph *grin, Option_t *option="linear", Int_t nout=50, Double_t *xout=0,
                          Double_t yleft=0, Double_t yright=0, Int_t rule=0, Double_t f=0, Option_t *ties="mean");
   TGraph         *SmoothKern(TGraph *grin, Option_t *option="normal", Double_t bandwidth=0.5, Int_t nout=100, Double_t *xout=0);
   TGraph         *SmoothLowess(TGraph *grin, Option_t *option="",Double_t span=0.67, Int_t iter = 3, Double_t delta = 0);
   TGraph         *SmoothSuper(TGraph *grin, Option_t *option="", Double_t bass = 0, Double_t span=0, Bool_t isPeriodic = kFALSE, Double_t *w=0);

   void            Approxin(TGraph *grin, Int_t iKind, Double_t &Ylow, Double_t &Yhigh, Int_t rule, Int_t iTies);
   void            Smoothin(TGraph *grin);
   static Double_t Approx1(Double_t v, Double_t f, Double_t *x, Double_t *y, Int_t n, Int_t iKind, Double_t Ylow, Double_t Yhigh);
   void            Lowess(Double_t *x, Double_t *y, Int_t n, Double_t *ys, Double_t span, Int_t iter, Double_t delta);
   static void     Lowest(Double_t *x, Double_t *y, Int_t n, Double_t &xs,
                          Double_t &ys, Int_t nleft, Int_t nright, Double_t *w, Bool_t userw, Double_t *rw, Bool_t &ok);
   static Int_t    Rcmp(Double_t x, Double_t y);
   static void     Psort(Double_t *x, Int_t n, Int_t k);
   static void     Rank(Int_t n, Double_t *a, Int_t *index, Int_t *rank, Bool_t down=kTRUE);
   static void     BDRksmooth(Double_t *x, Double_t *y, Int_t n,
                              Double_t *xp, Double_t *yp, Int_t np, Int_t kernel, Double_t bw);
   static void     BDRsupsmu(Int_t n, Double_t *x, Double_t *y, Double_t *w, Int_t iper,
                             Double_t span, Double_t alpha, Double_t *smo, Double_t *sc);
   static void     BDRsmooth(Int_t n, Double_t *x, Double_t *y, Double_t *w,
                             Double_t span, Int_t iper, Double_t vsmlsq, Double_t *smo, Double_t *acvr);

   ClassDef(TGraphSmooth,1) //Graph Smoother
};

#endif
