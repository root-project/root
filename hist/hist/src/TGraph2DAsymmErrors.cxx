// @(#)root/hist:$Id: TGraph2DAsymmErrors.cxx,v 1.00
// Author: Olivier Couet

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>
#include "TBuffer.h"
#include "TGraph2DAsymmErrors.h"
#include "TMath.h"
#include "TH2.h"
#include "TVirtualPad.h"
#include "TVirtualFitter.h"
#include "THLimitsFinder.h"

ClassImp(TGraph2DAsymmErrors);

/** \class TGraph2DAsymmErrors
    \ingroup Graphs
Graph 2D class with errors.

A TGraph2DAsymmErrors is a TGraph2D with asymmetric errors. It behaves like a TGraph2D and has
the same drawing options.

The **"ERR"** drawing option allows to display the error bars. The
following example shows how to use it:

Begin_Macro(source)
{
   auto c = new TCanvas("c","TGraph2DAsymmErrors example",0,0,600,600);
   Double_t P = 6.;
   Int_t np   = 200;

   Double_t *rx=0, *ry=0, *rz=0;
   Double_t *exl=0, *exh=0, *eyl=0, *eyh=0, *ezl=0, *ezh=0;

   rx  = new Double_t[np];
   ry  = new Double_t[np];
   rz  = new Double_t[np];
   exl = new Double_t[np];
   exh = new Double_t[np];
   eyl = new Double_t[np];
   eyh = new Double_t[np];
   ezl = new Double_t[np];
   ezh = new Double_t[np];

   auto r = new TRandom();

   for (Int_t N=0; N<np;N++) {
      rx[N] = 2*P*(r->Rndm(N))-P;
      ry[N] = 2*P*(r->Rndm(N))-P;
      rz[N] = rx[N]*rx[N]-ry[N]*ry[N];
      rx[N] = 10.+rx[N];
      ry[N] = 10.+ry[N];
      rz[N] = 40.+rz[N];
      exl[N] = r->Rndm(N);
      exh[N] = r->Rndm(N);
      eyl[N] = r->Rndm(N);
      eyh[N] = r->Rndm(N);
      ezl[N] = 10*r->Rndm(N);
      ezh[N] = 10*r->Rndm(N);
   }

   auto g = new TGraph2DAsymmErrors(np, rx, ry, rz, exl, exh, eyl, eyh, ezl, ezh);
   g->SetTitle("TGraph2D with asymmetric error bars: option \"ERR\"");
   g->SetFillColor(29);
   g->SetMarkerSize(0.8);
   g->SetMarkerStyle(20);
   g->SetMarkerColor(kRed);
   g->SetLineColor(kBlue-3);
   g->SetLineWidth(2);
   gPad->SetLogy(1);
   g->Draw("err p0");
}
End_Macro
*/


////////////////////////////////////////////////////////////////////////////////
/// TGraph2DAsymmErrors default constructor

TGraph2DAsymmErrors::TGraph2DAsymmErrors(): TGraph2D()
{
   fEXlow  = 0;
   fEXhigh = 0;
   fEYlow  = 0;
   fEYhigh = 0;
   fEZlow  = 0;
   fEZhigh = 0;
}


////////////////////////////////////////////////////////////////////////////////
/// TGraph2DAsymmErrors normal constructor
/// the arrays are preset to zero

TGraph2DAsymmErrors::TGraph2DAsymmErrors(Int_t n)
               : TGraph2D(n)
{
   if (n <= 0) {
      Error("TGraph2DAsymmErrors", "Invalid number of points (%d)", n);
      return;
   }

   fEXlow  = new Double_t[n];
   fEXhigh = new Double_t[n];
   fEYlow  = new Double_t[n];
   fEYhigh = new Double_t[n];
   fEZlow  = new Double_t[n];
   fEZhigh = new Double_t[n];

   for (Int_t i=0;i<n;i++) {
      fEXlow[i]  = 0;
      fEXhigh[i] = 0;
      fEYlow[i]  = 0;
      fEYhigh[i] = 0;
      fEZlow[i]  = 0;
      fEZhigh[i] = 0;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// TGraph2DAsymmErrors constructor with doubles vectors as input.

TGraph2DAsymmErrors::TGraph2DAsymmErrors(Int_t n, Double_t *x, Double_t *y, Double_t *z, Double_t *exl, Double_t *exh, Double_t *eyl, Double_t *eyh, Double_t *ezl, Double_t *ezh,  Option_t *)
               :TGraph2D(n, x, y, z)
{
   if (n <= 0) {
      Error("TGraph2DAsymmErrorsErrors", "Invalid number of points (%d)", n);
      return;
   }

   fEXlow  = new Double_t[n];
   fEXhigh = new Double_t[n];
   fEYlow  = new Double_t[n];
   fEYhigh = new Double_t[n];
   fEZlow  = new Double_t[n];
   fEZhigh = new Double_t[n];

   for (Int_t i=0;i<n;i++) {
      if (exl) fEXlow[i]  = exl[i];
      else     fEXlow[i]  = 0;
      if (exh) fEXhigh[i] = exh[i];
      else     fEXhigh[i] = 0;
      if (eyl) fEYlow[i]  = eyl[i];
      else     fEYlow[i]  = 0;
      if (eyh) fEYhigh[i] = eyh[i];
      else     fEYhigh[i] = 0;
      if (ezl) fEZlow[i]  = ezl[i];
      else     fEZlow[i]  = 0;
      if (ezh) fEZhigh[i] = ezh[i];
      else     fEZhigh[i] = 0;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// TGraph2DAsymmErrors destructor.

TGraph2DAsymmErrors::~TGraph2DAsymmErrors()
{
   delete [] fEXlow;
   delete [] fEXhigh;
   delete [] fEYlow;
   delete [] fEYhigh;
   delete [] fEZlow;
   delete [] fEZhigh;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.
/// Copy everything except list of functions

TGraph2DAsymmErrors::TGraph2DAsymmErrors(const TGraph2DAsymmErrors &g)
: TGraph2D(g), fEXlow(0), fEXhigh(0), fEYlow(0), fEYhigh(0), fEZlow(0), fEZhigh(0)
{
   if (fSize > 0) {
      fEXlow  = new Double_t[fSize];
      fEXhigh = new Double_t[fSize];
      fEYlow  = new Double_t[fSize];
      fEYhigh = new Double_t[fSize];
      fEZlow  = new Double_t[fSize];
      fEZhigh = new Double_t[fSize];
      for (Int_t n = 0; n < fSize; n++) {
         fEXlow[n]  = g.fEXlow[n];
         fEXhigh[n] = g.fEXhigh[n];
         fEYlow[n]  = g.fEYlow[n];
         fEYhigh[n] = g.fEYhigh[n];
         fEZlow[n]  = g.fEZlow[n];
         fEZhigh[n] = g.fEZhigh[n];
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator
/// Copy everything except list of functions

TGraph2DAsymmErrors & TGraph2DAsymmErrors::operator=(const TGraph2DAsymmErrors &g)
{
   if (this == &g) return *this;

   // call operator= on TGraph2D
   this->TGraph2D::operator=(static_cast<const TGraph2D&>(g) );

   // delete before existing contained objects
   if (fEXlow)  delete [] fEXlow;
   if (fEXhigh) delete [] fEXhigh;
   if (fEYlow)  delete [] fEYlow;
   if (fEYhigh) delete [] fEYhigh;
   if (fEZlow)  delete [] fEZlow;
   if (fEZhigh) delete [] fEZhigh;

   fEXlow    = (fSize > 0) ? new Double_t[fSize] : 0;
   fEXhigh   = (fSize > 0) ? new Double_t[fSize] : 0;
   fEYlow    = (fSize > 0) ? new Double_t[fSize] : 0;
   fEYhigh   = (fSize > 0) ? new Double_t[fSize] : 0;
   fEZlow    = (fSize > 0) ? new Double_t[fSize] : 0;
   fEZhigh   = (fSize > 0) ? new Double_t[fSize] : 0;


   // copy error arrays
   for (Int_t n = 0; n < fSize; n++) {
      fEXlow[n]  = g.fEXlow[n];
      fEXhigh[n] = g.fEXhigh[n];
      fEYlow[n]  = g.fEYlow[n];
      fEYhigh[n] = g.fEYhigh[n];
      fEZlow[n]  = g.fEZlow[n];
      fEZhigh[n] = g.fEZhigh[n];
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the combined error along X at point i by computing the average
/// of the lower and upper variance.

Double_t TGraph2DAsymmErrors::GetErrorX(Int_t i) const
{
   if (i < 0 || i >= fNpoints) return -1;
   if (!fEXlow && !fEXhigh) return -1;
   Double_t elow=0, ehigh=0;
   if (fEXlow)  elow  = fEXlow[i];
   if (fEXhigh) ehigh = fEXhigh[i];
   return TMath::Sqrt(0.5*(elow*elow + ehigh*ehigh));
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the low error along X at point i.

Double_t TGraph2DAsymmErrors::GetErrorXlow(Int_t i) const
{
   if (i < 0 || i >= fNpoints) return -1;
   if (fEXlow) return fEXlow[i];
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the high error along X at point i.

Double_t TGraph2DAsymmErrors::GetErrorXhigh(Int_t i) const
{
   if (i < 0 || i >= fNpoints) return -1;
   if (fEXhigh) return fEXhigh[i];
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the combined error along Y at point i by computing the average
/// of the lower and upper variance.

Double_t TGraph2DAsymmErrors::GetErrorY(Int_t i) const
{
   if (i < 0 || i >= fNpoints) return -1;
   if (!fEYlow && !fEYhigh) return -1;
   Double_t elow=0, ehigh=0;
   if (fEYlow)  elow  = fEYlow[i];
   if (fEYhigh) ehigh = fEYhigh[i];
   return TMath::Sqrt(0.5*(elow*elow + ehigh*ehigh));
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the low error along Y at point i.

Double_t TGraph2DAsymmErrors::GetErrorYlow(Int_t i) const
{
   if (i < 0 || i >= fNpoints) return -1;
   if (fEYlow) return fEYlow[i];
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the high error along Y at point i.

Double_t TGraph2DAsymmErrors::GetErrorYhigh(Int_t i) const
{
   if (i < 0 || i >= fNpoints) return -1;
   if (fEYhigh) return fEYhigh[i];
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the combined error along Z at point i by computing the average
/// of the lower and upper variance.

Double_t TGraph2DAsymmErrors::GetErrorZ(Int_t i) const
{
   if (i < 0 || i >= fNpoints) return -1;
   if (!fEZlow && !fEZhigh) return -1;
   Double_t elow=0, ehigh=0;
   if (fEZlow)  elow  = fEZlow[i];
   if (fEZhigh) ehigh = fEZhigh[i];
   return TMath::Sqrt(0.5*(elow*elow + ehigh*ehigh));
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the low error along Z at point i.

Double_t TGraph2DAsymmErrors::GetErrorZlow(Int_t i) const
{
   if (i < 0 || i >= fNpoints) return -1;
   if (fEZlow) return fEZlow[i];
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the high error along Z at point i.

Double_t TGraph2DAsymmErrors::GetErrorZhigh(Int_t i) const
{
   if (i < 0 || i >= fNpoints) return -1;
   if (fEZhigh) return fEZhigh[i];
   return -1;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the X maximum with errors.

Double_t TGraph2DAsymmErrors::GetXmaxE() const
{
   Double_t v = fX[0]+fEXhigh[0];
   for (Int_t i=1; i<fNpoints; i++) if (fX[i]+fEXhigh[i]>v) v=fX[i]+fEXhigh[i];
   return v;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the X minimum with errors.

Double_t TGraph2DAsymmErrors::GetXminE() const
{
   Double_t v = fX[0]-fEXlow[0];
   for (Int_t i=1; i<fNpoints; i++) if (fX[i]-fEXlow[i]<v) v=fX[i]-fEXlow[i];
   return v;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the Y maximum with errors.

Double_t TGraph2DAsymmErrors::GetYmaxE() const
{
   Double_t v = fY[0]+fEYhigh[0];
   for (Int_t i=1; i<fNpoints; i++) if (fY[i]+fEYhigh[i]>v) v=fY[i]+fEYhigh[i];
   return v;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the Y minimum with errors.

Double_t TGraph2DAsymmErrors::GetYminE() const
{
   Double_t v = fY[0]-fEYlow[0];
   for (Int_t i=1; i<fNpoints; i++) if (fY[i]-fEYlow[i]<v) v=fY[i]-fEYlow[i];
   return v;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the Z maximum with errors.

Double_t TGraph2DAsymmErrors::GetZmaxE() const
{
   Double_t v = fZ[0]+fEZhigh[0];
   for (Int_t i=1; i<fNpoints; i++) if (fZ[i]+fEZhigh[i]>v) v=fZ[i]+fEZhigh[i];
   return v;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the Z minimum with errors.

Double_t TGraph2DAsymmErrors::GetZminE() const
{
   Double_t v = fZ[0]-fEZlow[0];
   for (Int_t i=1; i<fNpoints; i++) if (fZ[i]-fEZlow[i]<v) v=fZ[i]-fEZlow[i];
   return v;
}


////////////////////////////////////////////////////////////////////////////////
/// Print 2D graph and errors values.

void TGraph2DAsymmErrors::Print(Option_t *) const
{
   for (Int_t i = 0; i < fNpoints; i++) {
      printf("x[%d]=%g, y[%d]=%g, z[%d]=%g, exl[%d]=%g, exh[%d]=%g, eyl[%d]=%g, eyh[%d]=%g, ezl[%d]=%g, ezh[%d]=%g\n",
            i, fX[i], i, fY[i], i, fZ[i], i, fEXlow[i], i, fEXhigh[i], i, fEYlow[i], i, fEYhigh[i], i, fEZlow[i], i, fEZhigh[i]);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Multiply the values and errors of a TGraph2DAsymmErrors by a constant c1.
///
/// If option contains "x" the x values and errors are scaled
/// If option contains "y" the y values and errors are scaled
/// If option contains "z" the z values and errors are scaled
/// If option contains "xyz" all three x, y and z values and errors are scaled

void TGraph2DAsymmErrors::Scale(Double_t c1, Option_t *option)
{
   TGraph2D::Scale(c1, option);
   TString opt = option; opt.ToLower();
   if (opt.Contains("x") && GetEXlow()) {
      for (Int_t i=0; i<GetN(); i++)
         GetEXlow()[i] *= c1;
   }
   if (opt.Contains("x") && GetEXhigh()) {
      for (Int_t i=0; i<GetN(); i++)
         GetEXhigh()[i] *= c1;
   }
   if (opt.Contains("y") && GetEYlow()) {
      for (Int_t i=0; i<GetN(); i++)
         GetEYlow()[i] *= c1;
   }
   if (opt.Contains("y") && GetEYhigh()) {
      for (Int_t i=0; i<GetN(); i++)
         GetEYhigh()[i] *= c1;
   }
   if (opt.Contains("z") && GetEZlow()) {
      for (Int_t i=0; i<GetN(); i++)
         GetEZlow()[i] *= c1;
   }
   if (opt.Contains("z") && GetEZhigh()) {
      for (Int_t i=0; i<GetN(); i++)
         GetEZhigh()[i] *= c1;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set number of points in the 2D graph.
/// Existing coordinates are preserved.
/// New coordinates above fNpoints are preset to 0.

void TGraph2DAsymmErrors::Set(Int_t n)
{
   if (n < 0) n = 0;
   if (n == fNpoints) return;
   if (n >  fNpoints) SetPointError(n,0,0,0,0,0,0);
   fNpoints = n;
}

////////////////////////////////////////////////////////////////////////////////
/// Deletes point number ipoint

Int_t TGraph2DAsymmErrors::RemovePoint(Int_t ipoint)
{
   if (ipoint < 0) return -1;
   if (ipoint >= fNpoints) return -1;

   fNpoints--;
   Double_t *newX      = new Double_t[fNpoints];
   Double_t *newY      = new Double_t[fNpoints];
   Double_t *newZ      = new Double_t[fNpoints];
   Double_t *newEXlow  = new Double_t[fNpoints];
   Double_t *newEXhigh = new Double_t[fNpoints];
   Double_t *newEYlow  = new Double_t[fNpoints];
   Double_t *newEYhigh = new Double_t[fNpoints];
   Double_t *newEZlow  = new Double_t[fNpoints];
   Double_t *newEZhigh = new Double_t[fNpoints];

   Int_t j = -1;
   for (Int_t i = 0; i < fNpoints + 1; i++) {
      if (i == ipoint) continue;
      j++;
      newX[j]      = fX[i];
      newY[j]      = fY[i];
      newZ[j]      = fZ[i];
      newEXlow[j]  = fEXlow[i];
      newEXhigh[j] = fEXhigh[i];
      newEYlow[j]  = fEYlow[i];
      newEYhigh[j] = fEYhigh[i];
      newEZlow[j]  = fEZlow[i];
      newEZhigh[j] = fEZhigh[i];
   }
   delete [] fX;
   delete [] fY;
   delete [] fZ;
   delete [] fEXlow;
   delete [] fEXhigh;
   delete [] fEYlow;
   delete [] fEYhigh;
   delete [] fEZlow;
   delete [] fEZhigh;
   fX  = newX;
   fY  = newY;
   fZ  = newZ;
   fEXlow  = newEXlow;
   fEXhigh = newEXhigh;
   fEYlow  = newEYlow;
   fEYhigh = newEYhigh;
   fEZlow  = newEZlow;
   fEZhigh = newEZhigh;
   fSize = fNpoints;
   if (fHistogram) {
      delete fHistogram;
      fHistogram = nullptr;
      fDelaunay = nullptr;
   }
   return ipoint;
}

////////////////////////////////////////////////////////////////////////////////
/// Set x, y and z values for point number i

void TGraph2DAsymmErrors::SetPoint(Int_t i, Double_t x, Double_t y, Double_t z)
{
   if (i < 0) return;
   if (i >= fNpoints) {
   // re-allocate the object
      Double_t *savex   = new Double_t[i+1];
      Double_t *savey   = new Double_t[i+1];
      Double_t *savez   = new Double_t[i+1];
      Double_t *saveexl = new Double_t[i+1];
      Double_t *saveexh = new Double_t[i+1];
      Double_t *saveeyl = new Double_t[i+1];
      Double_t *saveeyh = new Double_t[i+1];
      Double_t *saveezl = new Double_t[i+1];
      Double_t *saveezh = new Double_t[i+1];
      if (fNpoints > 0) {
         memcpy(savex, fX,      fNpoints*sizeof(Double_t));
         memcpy(savey, fY,      fNpoints*sizeof(Double_t));
         memcpy(savez, fZ,      fNpoints*sizeof(Double_t));
         memcpy(saveexl,fEXlow, fNpoints*sizeof(Double_t));
         memcpy(saveexh,fEXhigh,fNpoints*sizeof(Double_t));
         memcpy(saveeyl,fEYlow, fNpoints*sizeof(Double_t));
         memcpy(saveeyh,fEYhigh,fNpoints*sizeof(Double_t));
         memcpy(saveezl,fEZlow, fNpoints*sizeof(Double_t));
         memcpy(saveezh,fEZhigh,fNpoints*sizeof(Double_t));
      }
      if (fX)  delete [] fX;
      if (fY)  delete [] fY;
      if (fZ)  delete [] fZ;
      if (fEXlow)  delete [] fEXlow;
      if (fEXhigh) delete [] fEXhigh;
      if (fEYlow)  delete [] fEYlow;
      if (fEYhigh) delete [] fEYhigh;
      if (fEZlow)  delete [] fEZlow;
      if (fEZhigh) delete [] fEZhigh;
      fX       = savex;
      fY       = savey;
      fZ       = savez;
      fEXlow   = saveexl;
      fEXhigh  = saveexh;
      fEYlow   = saveeyl;
      fEYhigh  = saveeyh;
      fEZlow   = saveezl;
      fEZhigh  = saveezh;
      fNpoints = i+1;
   }
   fX[i] = x;
   fY[i] = y;
   fZ[i] = z;
}


////////////////////////////////////////////////////////////////////////////////
/// Set ex, ey and ez values for point number i

void TGraph2DAsymmErrors::SetPointError(Int_t i, Double_t exl, Double_t exh, Double_t eyl, Double_t eyh, Double_t ezl,  Double_t ezh)
{
   if (i < 0) return;
   if (i >= fNpoints) {
      // re-allocate the object
      TGraph2DAsymmErrors::SetPoint(i,0,0,0);
   }
   fEXlow[i]  = exl;
   fEXhigh[i] = exh;
   fEYlow[i]  = eyl;
   fEYhigh[i] = eyh;
   fEZlow[i]  = ezl;
   fEZhigh[i] = ezh;
}


////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class TGraph2DAsymmErrors.

void TGraph2DAsymmErrors::Streamer(TBuffer &b)
{
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      b.ReadClassBuffer(TGraph2DAsymmErrors::Class(), this, R__v, R__s, R__c);
   } else {
      b.WriteClassBuffer(TGraph2DAsymmErrors::Class(),this);
   }
}
