// @(#)root/hist:$Name:  $:$Id: TF3.h,v 1.2 2000/06/13 10:38:11 brun Exp $
// Author: Rene Brun   27/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// ---------------------------------- F3.h

#ifndef ROOT_TF3
#define ROOT_TF3



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TF3                                                                  //
//                                                                      //
// The Parametric 2-D function                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TF2
#include "TF2.h"
#endif

class TF3 : public TF2 {

protected:
   Double_t  fZmin;        //Lower bound for the range in z
   Double_t  fZmax;        //Upper bound for the range in z
   Int_t     fNpz;         //Number of points along z used for the graphical representation

public:
   TF3();
   TF3(const char *name, const char *formula, Double_t xmin=0, Double_t xmax=1, Double_t ymin=0,
       Double_t ymax=1, Double_t zmin=0, Double_t zmax=1);
   TF3(const char *name, void *fcn, Double_t xmin=0, Double_t xmax=1, Double_t ymin=0,
       Double_t ymax=1, Double_t zmin=0, Double_t zmax=1, Int_t npar=0);
   TF3(const char *name, Double_t (*fcn)(Double_t *, Double_t *), Double_t xmin=0, Double_t xmax=1, Double_t ymin=0,
       Double_t ymax=1, Double_t zmin=0, Double_t zmax=1, Int_t npar=0);

   TF3(const TF3 &f3);
   virtual   ~TF3();
   virtual void     Copy(TObject &f3);
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual void     Draw(Option_t *option="");
   virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
          Int_t     GetNpz() const {return fNpz;}
   virtual void     GetRandom3(Double_t &xrandom, Double_t &yrandom, Double_t &zrandom);
   virtual void     GetRange(Double_t &xmin, Double_t &xmax);
   virtual void     GetRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax);
   virtual void     GetRange(Double_t &xmin, Double_t &ymin, Double_t &zmin, Double_t &xmax, Double_t &ymax, Double_t &zmax);
   virtual Double_t GetZmin() const {return fZmin;}
   virtual Double_t GetZmax() const {return fZmax;}
   virtual Double_t Integral(Double_t a, Double_t b, Double_t *params=0, Double_t epsilon=0.000001) {return TF1::Integral(a,b,params,epsilon);}
   virtual Double_t Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t epsilon=0.000001) {return TF1::Integral(ax,bx,ay,by,epsilon);}
   virtual Double_t Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t az, Double_t bz, Double_t epsilon=0.000001);
   virtual void     Paint(Option_t *option="");
   virtual void     SetNpz(Int_t npz=30);
   virtual void     SetRange(Double_t xmin, Double_t xmax);
   virtual void     SetRange(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax);
   virtual void     SetRange(Double_t xmin, Double_t ymin, Double_t zmin, Double_t xmax, Double_t ymax, Double_t zmax); // *MENU*

   ClassDef(TF3,2)  //The Parametric 3-D function
};

inline void TF3::GetRange(Double_t &xmin, Double_t &xmax)
   { TF2::GetRange(xmin, xmax); }
inline void TF3::GetRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax)
   { TF2::GetRange(xmin, ymin, xmax, ymax); }
inline void TF3::SetRange(Double_t xmin, Double_t xmax)
   { TF2::SetRange(xmin, xmax); }
inline void TF3::SetRange(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
   { TF2::SetRange(xmin, ymin, xmax, ymax); }

#endif
