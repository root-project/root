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
   TCanvas * CPol = new TCanvas("CPol","TGraphPolar Examples",500,500);

   Double_t rmin=0;
   Double_t rmax=TMath::Pi()*2;
   Double_t r[1000];
   Double_t theta[1000];

   TF1 * fp1 = new TF1("fplot","cos(x)",rmin,rmax);
   for (Int_t ipt = 0; ipt < 1000; ipt++) {
      r[ipt] = ipt*(rmax-rmin)/1000+rmin;
      theta[ipt] = fp1->Eval(r[ipt]);
   }
   TGraphPolar * grP1 = new TGraphPolar(1000,r,theta);
   grP1->SetLineColor(2);
   grP1->Draw("AOL");

   return CPol;
}
End_Macro */

#include "TROOT.h"
#include "TGraphPolar.h"
#include "TGraphPolargram.h"
#include "TVirtualPad.h"


ClassImp(TGraphPolar);


//______________________________________________________________________________
TGraphPolar::TGraphPolar() : TGraphErrors()
{
   // TGraphPolar default constructor.

   fPolargram  = 0;
   fOptionAxis = kFALSE;
}


//______________________________________________________________________________
TGraphPolar::TGraphPolar(Int_t n, const Double_t* r, const Double_t* theta,
                                  const Double_t *er, const Double_t* etheta)
  : TGraphErrors(n,r,theta,er,etheta)
{
   // TGraphPolar constructor.
   //
   // n      : number of points.
   // r      : radial values.
   // theta  : angular values.
   // er     : errors on radial values.
   // etheta : errors on angular values.

   fPolargram  = 0;
   fOptionAxis = kFALSE;
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
