// @(#)root/graf:$Id$
// Author: Sebastian Boser, Mathieu Demaret 02/02/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
/* Begin_Html
<center><h2>TGraphPolar : to draw a polar graph</h2></center>
TGraphPolar creates a polar graph (including error bars). A TGraphPolar is
a TGraphErrors represented in polar coordinates.
It uses the class TGraphPolargram to draw the polar axis.
<p>
Example:
End_Html
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
End_Macro */

#include "TROOT.h"
#include "TGraphPolar.h"
#include "TGraphPolargram.h"
#include "TVirtualPad.h"


ClassImp(TGraphPolar);


//______________________________________________________________________________
TGraphPolar::TGraphPolar() : TGraphErrors(),
             fOptionAxis(kFALSE),fPolargram(0),fXpol(0),fYpol(0)
{
   // TGraphPolar default constructor.

}


//______________________________________________________________________________
TGraphPolar::TGraphPolar(Int_t n, const Double_t* theta, const Double_t* r,
                                  const Double_t *etheta, const Double_t* er)
  : TGraphErrors(n,theta,r,etheta,er),
             fOptionAxis(kFALSE),fPolargram(0),fXpol(0),fYpol(0)
{
   // TGraphPolar constructor.
   //
   // n      : number of points.
   // theta  : angular values.
   // r      : radial values.
   // etheta : errors on angular values.
   // er     : errors on radial values.

   SetEditable(kFALSE);
}


//______________________________________________________________________________
TGraphPolar::~TGraphPolar()
{
   // TGraphPolar destructor.
}


//______________________________________________________________________________
void TGraphPolar::Draw(Option_t* options)
{
   // Draw TGraphPolar.

   // Process options
   TString opt = options;
   opt.ToUpper();

   // Ignore same
   opt.ReplaceAll("SAME","");

   // ReDraw polargram if required by options
   if (opt.Contains("A")) fOptionAxis = kTRUE;
   opt.ReplaceAll("A","");

   AppendPad(opt);
}


//______________________________________________________________________________
Double_t *TGraphPolar::GetXpol() 
{
   // Return points in polar coordinates.

   if (!fXpol) fXpol = new Double_t[fNpoints];
   return fXpol;
}


//______________________________________________________________________________
Double_t *TGraphPolar::GetYpol()
{
   // Return points in polar coordinates.

   if (!fYpol) fYpol = new Double_t[fNpoints];
   return fYpol;
}


//______________________________________________________________________________
void TGraphPolar::SetMaxPolar(Double_t maximum)
{
   // Set maximum Polar.

   if (fPolargram) fPolargram->ChangeRangePolar(fPolargram->GetTMin(),maximum);
}


//______________________________________________________________________________
void TGraphPolar::SetMaxRadial(Double_t maximum)
{
   // Set maximum radial at the intersection of the positive X axis part and
   // the circle.

   if (fPolargram) fPolargram->SetRangeRadial(fPolargram->GetRMin(),maximum);
}


//______________________________________________________________________________
void TGraphPolar::SetMinPolar(Double_t minimum)
{
   // Set minimum Polar.

   if (fPolargram) fPolargram->ChangeRangePolar(minimum, fPolargram->GetTMax());
}


//______________________________________________________________________________
void TGraphPolar::SetMinRadial(Double_t minimum)
{
   // Set minimum radial in the center of the circle.

   if (fPolargram) fPolargram->SetRangeRadial(minimum, fPolargram->GetRMax());
}
