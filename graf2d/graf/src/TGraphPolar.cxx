// @(#)root/graf:$Id$
// Author: Sebastian Boser, Mathieu Demaret 02/02/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGraphPolar
\ingroup BasicGraphics

To draw a polar graph.

TGraphPolar creates a polar graph (including error bars). A TGraphPolar is
a TGraphErrors represented in polar coordinates.
It uses the class TGraphPolargram to draw the polar axis.

Example:

Begin_Macro(source)
{
   TCanvas * CPol = new TCanvas("CPol","TGraphPolar Example",500,500);

   Double_t theta[8];
   Double_t radius[8];
   Double_t etheta[8];
   Double_t eradius[8];

   for (int i=0; i<8; i++) {
      theta[i]   = (i+1)*(TMath::Pi()/4.);
      radius[i]  = (i+1)*0.05;
      etheta[i]  = TMath::Pi()/8.;
      eradius[i] = 0.05;
   }

   TGraphPolar * grP1 = new TGraphPolar(8, theta, radius, etheta, eradius);
   grP1->SetTitle("TGraphPolar Example");

   grP1->SetMarkerStyle(20);
   grP1->SetMarkerSize(2.);
   grP1->SetMarkerColor(4);
   grP1->SetLineColor(2);
   grP1->SetLineWidth(3);
   grP1->Draw("PE");

   // Update, otherwise GetPolargram returns 0
   CPol->Update();
   grP1->GetPolargram()->SetToRadian();

   return CPol;
}
End_Macro
*/

#include "TGraphPolar.h"
#include "TGraphPolargram.h"

ClassImp(TGraphPolar);

////////////////////////////////////////////////////////////////////////////////
/// TGraphPolar default constructor.

TGraphPolar::TGraphPolar() : TGraphErrors(),
             fOptionAxis(kFALSE),fPolargram(nullptr),fXpol(nullptr),fYpol(nullptr)
{
}

////////////////////////////////////////////////////////////////////////////////
/// TGraphPolar constructor.
///
/// \param[in] n         number of points.
/// \param[in] theta     angular values.
/// \param[in] r         radial values.
/// \param[in] etheta    errors on angular values.
/// \param[in] er        errors on radial values.

TGraphPolar::TGraphPolar(Int_t n, const Double_t* theta, const Double_t* r,
                                  const Double_t *etheta, const Double_t* er)
  : TGraphErrors(n,theta,r,etheta,er),
             fOptionAxis(kFALSE),fPolargram(nullptr),fXpol(nullptr),fYpol(nullptr)
{
   SetEditable(kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// TGraphPolar destructor.

TGraphPolar::~TGraphPolar()
{
   delete [] fXpol;
   delete [] fYpol;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw TGraphPolar.

void TGraphPolar::Draw(Option_t* options)
{
   AppendPad(options);
}

////////////////////////////////////////////////////////////////////////////////
/// Return points in polar coordinates.

Double_t *TGraphPolar::GetXpol()
{
   if (!fXpol) fXpol = new Double_t[fNpoints];
   return fXpol;
}

////////////////////////////////////////////////////////////////////////////////
/// Return points in polar coordinates.

Double_t *TGraphPolar::GetYpol()
{
   if (!fYpol) fYpol = new Double_t[fNpoints];
   return fYpol;
}

////////////////////////////////////////////////////////////////////////////////
/// Set maximum Polar.

void TGraphPolar::SetMaxPolar(Double_t maximum)
{
   if (fPolargram) fPolargram->ChangeRangePolar(fPolargram->GetTMin(),maximum);
}

////////////////////////////////////////////////////////////////////////////////
/// Set maximum radial at the intersection of the positive X axis part and
/// the circle.

void TGraphPolar::SetMaxRadial(Double_t maximum)
{
   if (fPolargram) fPolargram->SetRangeRadial(fPolargram->GetRMin(),maximum);
}

////////////////////////////////////////////////////////////////////////////////
/// Set minimum Polar.

void TGraphPolar::SetMinPolar(Double_t minimum)
{
   if (fPolargram) fPolargram->ChangeRangePolar(minimum, fPolargram->GetTMax());
}

////////////////////////////////////////////////////////////////////////////////
/// Set minimum radial in the center of the circle.

void TGraphPolar::SetMinRadial(Double_t minimum)
{
   if (fPolargram) fPolargram->SetRangeRadial(minimum, fPolargram->GetRMax());
}

////////////////////////////////////////////////////////////////////////////////
/// Create polargram object for given draw options

TGraphPolargram *TGraphPolar::CreatePolargram(const char *opt)
{
   Int_t theNpoints = GetN();

   if (theNpoints < 1)
      return nullptr;

   Double_t *theX = GetX();
   Double_t *theY = GetY();
   Double_t *theEX = GetEX();
   Double_t *theEY = GetEY();

   // Get range, initialize with first/last value
   Double_t rwrmin = theY[0], rwrmax = theY[theNpoints - 1], rwtmin = theX[0], rwtmax = theX[theNpoints - 1];

   for (Int_t ipt = 0; ipt < theNpoints; ipt++) {
      // Check for errors if available
      if (theEX) {
         if (theX[ipt] - theEX[ipt] < rwtmin)
            rwtmin = theX[ipt] - theEX[ipt];
         if (theX[ipt] + theEX[ipt] > rwtmax)
            rwtmax = theX[ipt] + theEX[ipt];
      } else {
         if (theX[ipt] < rwtmin)
            rwtmin = theX[ipt];
         if (theX[ipt] > rwtmax)
            rwtmax = theX[ipt];
      }
      if (theEY) {
         if (theY[ipt] - theEY[ipt] < rwrmin)
            rwrmin = theY[ipt] - theEY[ipt];
         if (theY[ipt] + theEY[ipt] > rwrmax)
            rwrmax = theY[ipt] + theEY[ipt];
      } else {
         if (theY[ipt] < rwrmin)
            rwrmin = theY[ipt];
         if (theY[ipt] > rwrmax)
            rwrmax = theY[ipt];
      }
   }

   // Add radial and Polar margins.
   if (rwrmin == rwrmax)
      rwrmax += 1.;
   if (rwtmin == rwtmax)
      rwtmax += 1.;

   Double_t dr = rwrmax - rwrmin, dt = rwtmax - rwtmin;

   rwrmax += 0.1 * dr;
   rwrmin -= 0.1 * dr;

   // Assume equally spaced points for full 2*Pi.
   rwtmax += dt / theNpoints;

   return new TGraphPolargram("Polargram", rwrmin, rwrmax, rwtmin, rwtmax, opt);
}
