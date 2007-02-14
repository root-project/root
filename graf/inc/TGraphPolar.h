// @(#)root/graf:$Name:  $:$Id: TGraphPolar.h,v 1.4 2007/01/15 16:10:10 brun Exp $
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

#include "TGraphErrors.h"
#include "Riostream.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphPolargram                                                      //
//                                                                      //
// Creates the polar coordinate system                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGraphPolargram: public TNamed, public TAttText, public TAttLine {
friend class TGraphPolar;

private:
   Bool_t   fRadian;
   Bool_t   fDegree;
   Bool_t   fGrad;
   
  
   Color_t  fPolarLabelColor;// Set color of the angular labels
   Color_t  fRadialLabelColor; // Set color of the radial labels
     
   Double_t fAxisAngle;        // Set angle of the radial axis
   Double_t fPolarOffset;    // Offset for Polar labels
   Double_t fPolarTextSize;  // Set Polar text size
   Double_t fRadialOffset;     // Offset for radial labels
   Double_t fRadialTextSize;
   Double_t fRwrmin;           // Minimal radial value (real world)
   Double_t fRwrmax;           // Maximal radial value (real world)
   Double_t fRwtmin;           // Minimal angular value (real world)
   Double_t fRwtmax;           // Minimal angular value (real world)
   Double_t fTickpolarSize;        // Set size of Tickmarks
     
   Font_t   fPolarLabelFont; // Set font of angular labels
   Font_t   fRadialLabelFont;  // Set font of radial labels
   
   Int_t    fCutRadial;    // if fCutRadial = 0, circles are cut by radial axis
                           // if fCutRadial = 1, circles are not cut
   Int_t    fNdivRad;      // Number of radial divisions
   Int_t    fNdivPol;      // Number of radial divisions
   
   
   
   void Paint(Option_t* options="");
   void PaintRadialDivisions();
   void PaintPolarDivisions();
   void ReduceFraction(Int_t Num, Int_t Denom, Int_t &rnum, Int_t &rden);

public:
   // TGraphPolarGram status bits
   enum { kLabelOrtho    = BIT(14)
        }; 

   TGraphPolargram(const char* name, Double_t rmin, Double_t rmax,
                                     Double_t tmin, Double_t tmax);
   ~TGraphPolargram();
   
   
   
   Color_t  GetPolarColorLabel (){ return fPolarLabelColor;};
   Color_t  GetRadialColorLabel (){ return fRadialLabelColor;};
      
   Double_t GetAngle() { return fAxisAngle;};
   Double_t GetPolarLabelSize() {return fPolarTextSize;};
   Double_t GetPolarOffset() { return fPolarOffset; };
   Double_t GetRadialLabelSize() {return fRadialTextSize;};
   Double_t GetRadialOffset() { return fRadialOffset; };
   Double_t GetRMin() { return fRwrmin;};
   Double_t GetRMax() { return fRwrmax;};
   Double_t GetTickpolarSize() {return fTickpolarSize;};
   Double_t GetTMin() { return fRwtmin;};
   Double_t GetTMax() { return fRwtmax;};
      
   Font_t   GetPolarLabelFont() { return fPolarLabelFont;};
   Font_t   GetRadialLabelFont() { return fRadialLabelFont;};
   
   Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   Int_t    GetNdivPolar() { return fNdivPol;};
   Int_t    GetNdivRadial() { return fNdivRad;};
   
   void     ChangeRangePolar(Double_t tmin, Double_t tmax);
   void     Draw(Option_t* options="");
   void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   void     PaintCircle(Double_t x, Double_t y, Double_t r,
                        Double_t phimin, Double_t phimax, Double_t theta);
   void     SetAxisAngle(Double_t angle = 0); //*MENU*
   void     SetNdivPolar(Int_t Ndiv = 508); //*MENU*
   void     SetNdivRadial(Int_t Ndiv = 508); //*MENU*
   void     SetPolarLabelSize(Double_t angularsize = 0.04); //*MENU*
   void     SetPolarLabelColor(Color_t tcolorangular = 1); //*MENU*
   void     SetPolarLabelFont(Font_t tfontangular = 62); //*MENU*
   void     SetPolarOffset(Double_t PolarOffset=0.04); //*MENU*
   void     SetRadialOffset(Double_t RadialOffset=0.025); //*MENU*
   void     SetRadialLabelSize (Double_t radialsize = 0.035); //*MENU*
   void     SetRadialLabelColor(Color_t tcolorradial = 1); //*MENU*
   void     SetRadialLabelFont(Font_t tfontradial = 62); //*MENU*
   void     SetRangePolar(Double_t tmin, Double_t tmax); //*MENU*
   void     SetRangeRadial(Double_t rmin, Double_t rmax); //*MENU*
   void     SetTickpolarSize(Double_t tickpolarsize = 0.02); //*MENU*
   void     SetToDegree(); //*MENU*
   void     SetToGrad(); //*MENU*
   void     SetToRadian(); //*MENU*
   void     SetTwoPi();
   
   ClassDef(TGraphPolargram,0); // Polar axis
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphPolar                                                          //
//                                                                      //
// Polar graph graphics class.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGraphPolar: public TGraphErrors {

private:
   Bool_t fOptionAxis;           // Force drawing of new coord system
   void Paint(Option_t* options = "");
   
protected:
   TGraphPolargram* fPolargram; // The polar coord system
   Double_t* fXpol;             // [fNpoints] points in polar coordinates
   Double_t* fYpol;             // [fNpoints] points in polar coordinates

public:
   TGraphPolar();
   TGraphPolar(Int_t n, const Double_t* x=0, const Double_t* y=0,
                        const Double_t* ex=0, const Double_t* ey=0);
   ~TGraphPolar();

   Int_t            DistancetoPrimitive(Int_t px, Int_t py);
   
   TGraphPolargram *GetPolargram() { return fPolargram; };
   
   void             Draw(Option_t* options = "");
   void             ExecuteEvent(Int_t event, Int_t px, Int_t py);
   void             PaintTitle();
   void             SetMaxRadial(Double_t maximum = 1); //*MENU*
   void             SetMinRadial(Double_t minimum = 0); //*MENU*
   void             SetMaximum(Double_t maximum = 1) {SetMaxRadial(maximum);} ; 
   void             SetMinimum(Double_t minimum = 0) {SetMinRadial(minimum);} ;
   void             SetMaxPolar(Double_t maximum = 6.28318530717958623); //*MENU*
   void             SetMinPolar(Double_t minimum = 0); //*MENU*
  
   ClassDef(TGraphPolar,0); // Polar graph
};

#endif
