// @(#)root/graf:$Name:  $:$Id: TCurlyArc.h,v 1.1.1.1 2000/05/16 17:00:50 rdm Exp $
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
   Double_t fR1;                  //  Radius of arc
   Double_t fPhimin;              //  start phi (degrees)
   Double_t fPhimax;              //  end phi (degrees)
   Double_t fTheta;               //  used internally

public:
   TCurlyArc(){;}
   TCurlyArc(Double_t x1, Double_t y1, Double_t rad,
             Double_t phimin, Double_t phimax,
             Double_t tl = .05, Double_t trad = .02);
   virtual     ~TCurlyArc(){;}
   virtual void Build();
   Int_t        DistancetoPrimitive(Int_t px, Int_t py);
   void         ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void SetCenter(Double_t x1, Double_t y1); // *MENU* *ARGS={x1=>fX1,y1=>fY1}
   virtual void SetRadius(Double_t radius);          // *MENU* *ARGS={radius=>fR1}
   virtual void SetPhimin(Double_t phimin);          // *MENU* *ARGS={phimin=>fPhimin}
   virtual void SetPhimax(Double_t phimax);          // *MENU* *ARGS={phimax=>fPhimax}
   Double_t      GetRadius() {return fR1;}
   Double_t      GetPhimin() {return fPhimin;}
   Double_t      GetPhimax() {return fPhimax;}
   virtual void SavePrimitive(ofstream &out, Option_t *);

   ClassDef(TCurlyArc,2) // a curly arc
};

#endif

