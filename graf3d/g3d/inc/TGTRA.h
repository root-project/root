// @(#)root/g3d:$Id$
// Author: Nenad Buncic   19/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTRA
#define ROOT_TGTRA


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGTRA                                                                  //
//                                                                        //
// GTRA is general twisted trapezoid. Essentially this is a TRAP shape,   //
// except this it is twisted in the x, y plane as a function z.           //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TBRIK
#include "TBRIK.h"
#endif

class TGTRA : public TBRIK {

protected:
   Float_t fTwist;     // twisting parameter
   Float_t fH1;        // half length in y at low z
   Float_t fBl1;       // half length in x at low z and y low edge
   Float_t fTl1;       // half length in x at low z and y high edge
   Float_t fAlpha1;    // angle w.r.t. the y axis
   Float_t fH2;        // half length in y at high z
   Float_t fBl2;       // half length in x at high z and y low edge
   Float_t fTl2;       // half length in x at high z and y high edge
   Float_t fAlpha2;    // angle w.r.t. the y axis

   virtual void    SetPoints(Double_t *points) const;
public:
   TGTRA();
   TGTRA(const char *name, const char *title, const char *material, Float_t dz, Float_t theta, Float_t phi, Float_t twist, Float_t h1,
         Float_t bl1, Float_t tl1, Float_t alpha1, Float_t h2, Float_t bl2, Float_t tl2,
         Float_t alpha2);
   virtual ~TGTRA();

   Float_t         GetTwist() const  {return fTwist;}
   Float_t         GetH1() const     {return fH1;}
   Float_t         GetBl1() const    {return fBl1;}
   Float_t         GetTl1() const    {return fTl1;}
   Float_t         GetAlpha1() const {return fAlpha1;}
   Float_t         GetH2() const     {return fH2;}
   Float_t         GetBl2() const    {return fBl2;}
   Float_t         GetTl2() const    {return fTl2;}
   Float_t         GetAlpha2() const {return fAlpha2;}

   ClassDef(TGTRA,1)  //GTRA shape
};

#endif
