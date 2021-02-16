// @(#)root/gpad:$Id$
// Author: Rene Brun   10/03/2007

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TColorWheel
#define ROOT_TColorWheel

#include "TNamed.h"

class TCanvas;
class TArc;
class TLine;
class TText;
class TGraph;

class TColorWheel : public TNamed {

private:
   Double_t  fRmin;     ///<Minimum radius for rectangles
   Double_t  fRmax;     ///<Maximum radius for rectangles
   Double_t  fR0;       ///<Minimum radius for circles
   Double_t  fDr;       ///<Circles radius
   Double_t  fRgray;    ///<Maximum radius of gray circle
   Double_t  fX[15];    ///<X coordinates of the center of circles
   Double_t  fY[15];    ///<Y coordinates of the center of circles
   TCanvas  *fCanvas;   ///<! Canvas used to draw the Color Wheel
   TArc     *fArc;      ///<! pointer to utility arc
   TLine    *fLine;     ///<! pointer to utility line
   TText    *fText;     ///<! pointer to utility text
   TGraph   *fGraph;    ///<! pointer to utility graph

   TColorWheel(const TColorWheel &) = delete;
   TColorWheel &operator=(const TColorWheel &) = delete;

protected:
   Int_t InCircles(Double_t x, Double_t y, Int_t coffset, Double_t angle) const;
   Int_t InGray(Double_t x, Double_t y) const;
   Int_t InRectangles(Double_t x, Double_t y, Int_t coffset, Double_t angle) const;
   void  PaintCircle(Int_t coffset,Int_t n,Double_t x, Double_t y, Double_t ang) const;
   void  PaintCircles(Int_t coffset, Double_t angle) const ;
   void  PaintGray() const;
   void  PaintRectangles(Int_t coffset, Double_t angle) const;
   void  Rotate(Double_t x, Double_t y, Double_t &u, Double_t &v, Double_t ang) const;

public:
   TColorWheel();
   virtual  ~TColorWheel();
   virtual  Int_t DistancetoPrimitive(Int_t px, Int_t py);
   virtual  void  Draw(Option_t *option="");
   TCanvas *GetCanvas() const {return fCanvas;}
   virtual  Int_t GetColor(Int_t px, Int_t py) const;
   virtual  char *GetObjectInfo(Int_t px, Int_t py) const;
   virtual  void  Paint(Option_t *option="");
   virtual  void  SetCanvas(TCanvas *can) {fCanvas = can;}

   ClassDef(TColorWheel,1)  //The ROOT Color Wheel
};

#endif

