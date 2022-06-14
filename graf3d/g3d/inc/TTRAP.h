// @(#)root/g3d:$Id$
// Author: Nenad Buncic   19/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTRAP
#define ROOT_TTRAP


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TTRAP                                                                  //
//                                                                        //
// TRAP is a general trapezoid, i.e. one for which the faces perpendicular//
// to z are trapezia and their centres are not the same x, y. It has 11   //
// parameters: the half length in z, the polar angles from the centre of  //
// the face at low z to that at high z, H1 the half length in y at low z, //
// LB1 the half length in x at low z and y low edge, LB2 the half length  //
// in x at low z and y high edge, TH1 the angle w.r.t. the y axis from the//
// centre of low y edge to the centre of the high y edge, and H2, LB2,    //
// LH2, TH2, the corresponding quantities at high z.                      //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TBRIK.h"

class TTRAP : public TBRIK {
protected:
   Float_t fH1;        // half length in y at low z
   Float_t fBl1;       // half length in x at low z and y low edge
   Float_t fTl1;       // half length in x at low z and y high edge
   Float_t fAlpha1;    // angle w.r.t. the y axis
   Float_t fH2;        // half length in y at high z
   Float_t fBl2;       // half length in x at high z and y low edge
   Float_t fTl2;       // half length in x at high z and y high edge
   Float_t fAlpha2;    // angle w.r.t. the y axis

   virtual void     SetPoints(Double_t *points) const;

public:
   TTRAP();
   TTRAP(const char *name, const char *title, const char *material, Float_t dz, Float_t theta, Float_t phi, Float_t h1,
         Float_t bl1, Float_t tl1, Float_t alpha1, Float_t h2, Float_t bl2, Float_t tl2,
         Float_t alpha2);
   virtual ~TTRAP();

   virtual Float_t  GetH1() const     {return fH1;}
   virtual Float_t  GetBl1() const    {return fBl1;}
   virtual Float_t  GetTl1() const    {return fTl1;}
   virtual Float_t  GetAlpha1() const {return fAlpha1;}
   virtual Float_t  GetH2() const     {return fH2;}
   virtual Float_t  GetBl2() const    {return fBl2;}
   virtual Float_t  GetTl2() const    {return fTl2;}
   virtual Float_t  GetAlpha2() const {return fAlpha2;}

   ClassDef(TTRAP,1)  //TRAP shape
};

#endif
