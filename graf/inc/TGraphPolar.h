// @(#)root/graf:$Name:  $:$Id: TGraphPolar.h,v 1.1  Exp $
// Author: Sebastian Boser, 02/02/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphPolar
#define ROOT_TGraphPolar

#include "TGraph.h"
#include "TGraphErrors.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TEllipse.h"
#include "TLine.h"
#include "TLatex.h"
#include "TGaxis.h"
#include "TFrame.h"
#include "Riostream.h"
#include "TString.h"

const Double_t kPi = TMath::Pi();

// TGraphPolargram creates the polar coordinate system

class TGraphPolargram: public TNamed, public TAttText, public TAttLine {

private:
   void Paint(Option_t* options="");
   Double_t fRwrmin;      // Minimal radial value (real world)
   Double_t fRwrmax;      // Maximal radial value (real world)
   Double_t fRwtmin;      // Minimal angular value (real world)
   Double_t fRwtmax;      // Minimal angular value (real world)
   Int_t fNdivRad;        // Number of radial divisions
   Int_t fNdivPol;        // Number of radial divisions
   Double_t fLabelOffset; // Offset for radial and polar labels

public:
   TGraphPolargram(const char* name, Double_t rmin, Double_t rmax,
                                     Double_t tmin, Double_t tmax);
   ~TGraphPolargram();
   void Draw(Option_t* options="");
   Int_t DistancetoPrimitive(Int_t px, Int_t py);
   void ExecuteEvent(Int_t event, Int_t px, Int_t py);
   Double_t GetRMin() { return fRwrmin;};
   Double_t GetRMax() { return fRwrmax;};
   Double_t GetTMin() { return fRwtmin;};
   Double_t GetTMax() { return fRwtmax;};
   void SetRangeRadial(Double_t rmin, Double_t rmax); //*MENU*
   void SetRangePolar(Double_t tmin, Double_t tmax); //*MENU*
   void SetTwoPi() { SetRangePolar(0,2*TMath::Pi()); }; //*MENU*
   void SetNdivRadial(Int_t Ndiv = 502); //*MENU*
   Int_t GetNdivRadial() { return fNdivRad; };
   void SetNdivPolar(Int_t Ndiv = 204); //*MENU*
   Int_t GetNdivPolar() { return fNdivPol; };
   void SetLabelOffset(Double_t LabelOffset=0.03); //*MENU*
   Double_t GetLabelOffset() { return fLabelOffset; };
   void PaintCircle(Double_t x, Double_t y, Double_t r,
                    Double_t phimin, Double_t phimax, Double_t theta);
    
   ClassDef(TGraphPolargram,0); // Polar axis
};


class TGraphPolar: public TGraphErrors {

private:
   void Paint(Option_t* options = "");
   Bool_t fOptionAxis;           // Force drawing of new coord system

protected:
   TGraphPolargram* fPolargram; // The polar coord system
   Double_t* fXpol;             // [fNpoints] points in polar coordinates
   Double_t* fYpol;             // [fNpoints] points in polar coordinates

public:
   TGraphPolar(Int_t n, const Double_t* x=0, const Double_t* y=0,
                        const Double_t* ex=0, const Double_t* ey=0);
   ~TGraphPolar();

   void Draw(Option_t* options = "");
   TGraphPolargram* GetPolargram() { return fPolargram; };
   Int_t DistancetoPrimitive(Int_t px, Int_t py);
   void ExecuteEvent(Int_t event, Int_t px, Int_t py);
   void SetMaxRadial(Double_t maximum = 1); //*MENU*
   void SetMinRadial(Double_t minimum = 0); //*MENU*
   void SetMaximum(Double_t maximum = 1) {SetMaxRadial(maximum);} ; 
   void SetMinimum(Double_t minimum = 0) {SetMinRadial(minimum);} ;
   void SetMaxPolar(Double_t maximum = 6.28318530717958623); //*MENU*
   void SetMinPolar(Double_t minimum = 0); //*MENU*

   ClassDef(TGraphPolar,0); // Polar graph
};

#endif
