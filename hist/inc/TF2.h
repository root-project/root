// @(#)root/hist:$Name$:$Id$
// Author: Rene Brun   23/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// ---------------------------------- F2.h

#ifndef ROOT_TF2
#define ROOT_TF2



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TF2                                                                  //
//                                                                      //
// The Parametric 2-D function                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TF1
#include "TF1.h"
#endif
#ifndef ROOT_TArrayF
#include "TArrayF.h"
#endif

class TF2 : public TF1 {

protected:
   Float_t   fYmin;        //Lower bound for the range in y
   Float_t   fYmax;        //Upper bound for the range in y
   Int_t     fNpy;         //Number of points along y used for the graphical representation
   TArrayF   fContour;     //Array to display contour levels

public:
   TF2();
   TF2(const char *name, const char *formula, Float_t xmin=0, Float_t xmax=1, Float_t ymin=0, Float_t ymax=1);
   TF2(const char *name, void *fcn, Float_t xmin=0, Float_t xmax=1, Float_t ymin=0, Float_t ymax=1, Int_t npar=0);
   TF2(const char *name, Double_t (*fcn)(Double_t *, Double_t *), Float_t xmin=0, Float_t xmax=1, Float_t ymin=0, Float_t ymax=1, Int_t npar=0);
   TF2(const TF2 &f2);
   virtual   ~TF2();
   virtual void     Copy(TObject &f2);
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual void     Draw(Option_t *option="");
   virtual TF1     *DrawCopy(Option_t *option="");
   virtual void     DrawF2(const char *formula, Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax, Option_t *option="");
   virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual Int_t    GetContour(Float_t *levels=0);
   virtual Float_t  GetContourLevel(Int_t level);
          Int_t     GetNpy() {return fNpy;}
   virtual char    *GetObjectInfo(Int_t px, Int_t py);
       Double_t     GetRandom();
   virtual void     GetRandom2(Float_t &xrandom, Float_t &yrandom);
   virtual void     GetRange(Float_t &xmin, Float_t &xmax) { TF1::GetRange(xmin, xmax); }
   virtual void     GetRange(Float_t &xmin, Float_t &ymin, Float_t &xmax, Float_t &ymax);
   virtual void     GetRange(Float_t &xmin, Float_t &ymin, Float_t &zmin, Float_t &xmax, Float_t &ymax, Float_t &zmax);
   virtual Float_t  GetYmin() {return fYmin;}
   virtual Float_t  GetYmax() {return fYmax;}
   virtual Double_t Integral(Double_t a, Double_t b, Double_t *params=0, Double_t epsil=0.000001) {return TF1::Integral(a,b,params,epsil);}
   virtual Double_t Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t epsil=0.000001);
   virtual Double_t Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t az, Double_t bz, Double_t epsil=0.000001)
                            {return TF1::Integral(ax,bx,ay,by,az,bz,epsil);}
   virtual void     Paint(Option_t *option="");
   virtual void     SetNpy(Int_t npy=100); // *MENU*
   virtual void     SetContour(Int_t nlevels=20, Float_t *levels=0);
   virtual void     SetContourLevel(Int_t level, Float_t value);
   virtual void     SetRange(Float_t xmin, Float_t xmax);
   virtual void     SetRange(Float_t xmin, Float_t ymin, Float_t xmax, Float_t ymax); // *MENU*
   virtual void     SetRange(Float_t xmin, Float_t ymin, Float_t zmin, Float_t xmax, Float_t ymax, Float_t zmax);

   ClassDef(TF2,2)  //The Parametric 2-D function
};

inline void TF2::SetRange(Float_t xmin, Float_t xmax)
   { TF1::SetRange(xmin, xmax); }
inline void TF2::SetRange(Float_t xmin, Float_t ymin, Float_t, Float_t xmax, Float_t ymax, Float_t)
   { SetRange(xmin, ymin, xmax, ymax); }

#endif
