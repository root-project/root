// @(#)root/graf:$Name:  $:$Id: TCurlyLine.h,v 1.2 2000/06/13 10:49:14 brun Exp $
// Author: Otto Schaile   20/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TCurlyLine
#define ROOT_TCurlyLine

//
// This class implements a curly or wavy polyline typically
// used to draw Feynman diagrams.

#ifndef __CINT__
#include "fstream.h"
#else
class ofstream;
#endif

#ifndef ROOT_TPolyLine
#include "TPolyLine.h"
#endif

class TCurlyLine : public TPolyLine {

protected:
   Double_t fX1;             // start x, center for arc
   Double_t fY1;             // start y, center for arc
   Double_t fX2;             // end x
   Double_t fY2;             // end y
   Double_t fWaveLength;     // wavelength of sinusoid in percent of pad height
   Double_t fAmplitude;      // amplitude of sinusoid in percent of pad height
   Int_t    fNsteps;         // used internally (controls precision)
   Bool_t   fIsCurly;        // true: Gluon, false: Gamma

public:
   // TCurlyLine status bits
   enum {
      kTooShort = BIT(11)
   };
   TCurlyLine(){;}
   TCurlyLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2,
              Double_t tl = .05, Double_t rad = .02);
   virtual ~TCurlyLine(){;}
   virtual void Build();
   Int_t        DistancetoPrimitive(Int_t px, Int_t py);
   void         ExecuteEvent(Int_t event, Int_t px, Int_t py);
   Bool_t       GetCurly() const     {return fIsCurly;}
   Double_t     GetWaveLength() const{return fWaveLength;}
   Double_t     GetAmplitude() const {return fAmplitude;}
   Double_t     GetStartX() const    {return fX1;}
   Double_t     GetEndX() const      {return fX2;}
   Double_t     GetStartY() const    {return fY1;}
   Double_t     GetEndY() const      {return fY2;}
   virtual void SetCurly();                             // *MENU*
   virtual void SetWavy();                              // *MENU*
   virtual void SetWaveLength(Double_t WaveLength);     // *MENU* *ARGS={WaveLength=>fWaveLength}
   virtual void SetAmplitude(Double_t x);               // *MENU* *ARGS={x=>fAmplitude}
   virtual void SetStartPoint(Double_t x1, Double_t y1);
   virtual void SetEndPoint  (Double_t x2, Double_t y2);
   virtual void SavePrimitive(ofstream &out, Option_t *);

   ClassDef(TCurlyLine,2) // A curly polyline
};

#endif
