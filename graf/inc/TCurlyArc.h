// @(#)root/graf:$Name$:$Id$
// Author: Otto Schaile   20/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TCurlyArc
#define ROOT_TCurlyArc

//
// This class implements a curly or wavy arc typically used
// to draw Feynman diagrams.

#include "TCurlyLine.h"

class TCurlyArc : public TCurlyLine {

private:
   Float_t fR1;                  //  Radius of arc
   Float_t fPhimin;              //  start phi (degrees)
   Float_t fPhimax;              //  end phi (degrees)
   Float_t fTheta;               //  used internally

public:
   TCurlyArc(){;}
   TCurlyArc(Float_t x1, Float_t y1, Float_t rad,
            Float_t phimin, Float_t phimax,
            Float_t tl = .05, Float_t trad = .02);
   virtual     ~TCurlyArc(){;}
   virtual void Build();
   Int_t        DistancetoPrimitive(Int_t px, Int_t py);
   void         ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void SetCenter(Float_t x1, Float_t y1);  // *MENU* *ARGS={x1=>fX1,y1=>fY1}
   virtual void SetRadius(Float_t radius);          // *MENU* *ARGS={radius=>fR1}
   virtual void SetPhimin(Float_t phimin);          // *MENU* *ARGS={phimin=>fPhimin}
   virtual void SetPhimax(Float_t phimax);          // *MENU* *ARGS={phimax=>fPhimax}
   Float_t      GetRadius() {return fR1;}
   Float_t      GetPhimin() {return fPhimin;}
   Float_t      GetPhimax() {return fPhimax;}
   virtual void SavePrimitive(ofstream &out, Option_t *);

   ClassDef(TCurlyArc,1) // a curly arc
};

#endif

