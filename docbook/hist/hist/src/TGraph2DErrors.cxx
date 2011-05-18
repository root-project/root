// @(#)root/hist:$Id: TGraph2DErrors.cxx,v 1.00
// Author: Olivier Couet

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TClass.h"
#include "TGraph2DErrors.h"
#include "TMath.h"
#include "TPolyMarker.h"
#include "TVirtualPad.h"
#include "TVirtualFitter.h"
#include "THLimitsFinder.h"
#include "TStyle.h"

ClassImp(TGraph2DErrors)

//______________________________________________________________________________
/* Begin_Html
<center><h2>Graph 2D class with errors</h2></center>
A TGraph2DErrors is a TGraph2D with errors. It behaves like a TGraph2D and has
the same drawing options.
<p>
The <tt>"ERR"</tt> drawing option allows to display the error bars. The
following example shows how to use it:

End_Html
Begin_Macro(source)
{
   TCanvas *c = new TCanvas("c","Graph2DErrors example",0,0,600,600);
   Double_t P = 6.;
   Int_t np   = 200;

   Double_t *rx=0, *ry=0, *rz=0;
   Double_t *ex=0, *ey=0, *ez=0;

   rx = new Double_t[np];
   ry = new Double_t[np];
   rz = new Double_t[np];
   ex = new Double_t[np];
   ey = new Double_t[np];
   ez = new Double_t[np];

   TRandom *r = new TRandom();

   for (Int_t N=0; N<np;N++) {
      rx[N] = 2*P*(r->Rndm(N))-P;
      ry[N] = 2*P*(r->Rndm(N))-P;
      rz[N] = rx[N]*rx[N]-ry[N]*ry[N];
      rx[N] = 10.+rx[N];
      ry[N] = 10.+ry[N];
      rz[N] = 40.+rz[N];
      ex[N] = r->Rndm(N);
      ey[N] = r->Rndm(N);
      ez[N] = 10*r->Rndm(N);
   }

   TGraph2DErrors *dte = new TGraph2DErrors(np, rx, ry, rz, ex, ey, ez);
   dte->SetTitle("TGraph2D with error bars: option \"ERR\"");
   dte->SetFillColor(29);
   dte->SetMarkerSize(0.8);
   dte->SetMarkerStyle(20);
   dte->SetMarkerColor(kRed);
   dte->SetLineColor(kBlue-3);
   dte->SetLineWidth(2);
   dte->Draw("err p0");
   gPad->SetLogy(1);
   return c;
}
End_Macro
*/


//______________________________________________________________________________
TGraph2DErrors::TGraph2DErrors(): TGraph2D()
{
   // TGraph2DErrors default constructor

   fEX = 0;
   fEY = 0;
   fEZ = 0;
}


//______________________________________________________________________________
TGraph2DErrors::TGraph2DErrors(Int_t n)
               : TGraph2D(n)
{
   // TGraph2DErrors normal constructor
   // the arrays are preset to zero

   if (n <= 0) {
      Error("TGraph2DErrors", "Invalid number of points (%d)", n);
      return;
   }

   fEX = new Double_t[n];
   fEY = new Double_t[n];
   fEZ = new Double_t[n];

   for (Int_t i=0;i<n;i++) {
      fEX[i] = 0;
      fEY[i] = 0;
      fEZ[i] = 0;
   }
}


//______________________________________________________________________________
TGraph2DErrors::TGraph2DErrors(Int_t n, Double_t *x, Double_t *y, Double_t *z,
                               Double_t *ex, Double_t *ey, Double_t *ez, Option_t *)
               :TGraph2D(n, x, y, z)
{
   // TGraph2DErrors constructor with doubles vectors as input.

   if (n <= 0) {
      Error("TGraphErrors", "Invalid number of points (%d)", n);
      return;
   }

   fEX = new Double_t[n];
   fEY = new Double_t[n];
   fEZ = new Double_t[n];

   for (Int_t i=0;i<n;i++) {
      if (ex) fEX[i] = ex[i];
      else    fEX[i] = 0;
      if (ey) fEY[i] = ey[i];
      else    fEY[i] = 0;
      if (ez) fEZ[i] = ez[i];
      else    fEZ[i] = 0;
   }
}


//______________________________________________________________________________
TGraph2DErrors::~TGraph2DErrors()
{
   // TGraph2DErrors destructor.

   delete [] fEX;
   delete [] fEY;
   delete [] fEZ;
}


//______________________________________________________________________________
Double_t TGraph2DErrors::GetErrorX(Int_t i) const
{
   // This function is called by Graph2DFitChisquare.
   // It returns the error along X at point i.

   if (i < 0 || i >= fNpoints) return -1;
   if (fEX) return fEX[i];
   return -1;
}


//______________________________________________________________________________
Double_t TGraph2DErrors::GetErrorY(Int_t i) const
{
   // This function is called by Graph2DFitChisquare.
   // It returns the error along X at point i.

   if (i < 0 || i >= fNpoints) return -1;
   if (fEY) return fEY[i];
   return -1;
}


//______________________________________________________________________________
Double_t TGraph2DErrors::GetErrorZ(Int_t i) const
{
   // This function is called by Graph2DFitChisquare.
   // It returns the error along X at point i.

   if (i < 0 || i >= fNpoints) return -1;
   if (fEZ) return fEZ[i];
   return -1;
}


//______________________________________________________________________________
Double_t TGraph2DErrors::GetXmaxE() const
{
   // Returns the X maximum with errors.

   Double_t v = fX[0]+fEX[0];
   for (Int_t i=1; i<fNpoints; i++) if (fX[i]+fEX[i]>v) v=fX[i]+fEX[i];
   return v;
}


//______________________________________________________________________________
Double_t TGraph2DErrors::GetXminE() const
{
   // Returns the X minimum with errors.

   Double_t v = fX[0]-fEX[0];
   for (Int_t i=1; i<fNpoints; i++) if (fX[i]-fEX[i]<v) v=fX[i]-fEX[i];
   return v;
}


//______________________________________________________________________________
Double_t TGraph2DErrors::GetYmaxE() const
{
   // Returns the Y maximum with errors.

   Double_t v = fY[0]+fEY[0];
   for (Int_t i=1; i<fNpoints; i++) if (fY[i]+fEY[i]>v) v=fY[i]+fEY[i];
   return v;
}


//______________________________________________________________________________
Double_t TGraph2DErrors::GetYminE() const
{
   // Returns the Y minimum with errors.

   Double_t v = fY[0]+fEY[0];
   for (Int_t i=1; i<fNpoints; i++) if (fY[i]-fEY[i]<v) v=fY[i]-fEY[i];
   return v;
}


//______________________________________________________________________________
Double_t TGraph2DErrors::GetZmaxE() const
{
   // Returns the Z maximum with errors.

   Double_t v = fZ[0]+fEZ[0];
   for (Int_t i=1; i<fNpoints; i++) if (fZ[i]+fEZ[i]>v) v=fZ[i]+fEZ[i];
   return v;
}


//______________________________________________________________________________
Double_t TGraph2DErrors::GetZminE() const
{
   // Returns the Z minimum with errors.

   Double_t v = fZ[0]+fEZ[0];
   for (Int_t i=1; i<fNpoints; i++) if (fZ[i]-fEZ[i]<v) v=fZ[i]-fEZ[i];
   return v;
}


//______________________________________________________________________________
void TGraph2DErrors::Set(Int_t n)
{
   // Set number of points in the 2D graph.
   // Existing coordinates are preserved.
   // New coordinates above fNpoints are preset to 0.

   if (n < 0) n = 0;
   if (n == fNpoints) return;
   if (n >  fNpoints) SetPointError(n,0,0,0);
   fNpoints = n;
}


//______________________________________________________________________________
void TGraph2DErrors::SetPoint(Int_t i, Double_t x, Double_t y, Double_t z)
{
   // Set x, y and z values for point number i

   if (i < 0) return;
   if (i >= fNpoints) {
   // re-allocate the object
      Double_t *savex  = new Double_t[i+1];
      Double_t *savey  = new Double_t[i+1];
      Double_t *savez  = new Double_t[i+1];
      Double_t *saveex = new Double_t[i+1];
      Double_t *saveey = new Double_t[i+1];
      Double_t *saveez = new Double_t[i+1];
      if (fNpoints > 0) {
         memcpy(savex, fX, fNpoints*sizeof(Double_t));
         memcpy(savey, fY, fNpoints*sizeof(Double_t));
         memcpy(savez, fZ, fNpoints*sizeof(Double_t));
         memcpy(saveex,fEX,fNpoints*sizeof(Double_t));
         memcpy(saveey,fEY,fNpoints*sizeof(Double_t));
         memcpy(saveez,fEZ,fNpoints*sizeof(Double_t));
      }
      if (fX)  delete [] fX;
      if (fY)  delete [] fY;
      if (fZ)  delete [] fZ;
      if (fEX) delete [] fEX;
      if (fEY) delete [] fEY;
      if (fEZ) delete [] fEZ;
      fX  = savex;
      fY  = savey;
      fZ  = savez;
      fEX = saveex;
      fEY = saveey;
      fEZ = saveez;
      fNpoints = i+1;
   }
   fX[i] = x;
   fY[i] = y;
   fZ[i] = z;
}


//______________________________________________________________________________
void TGraph2DErrors::SetPointError(Int_t i, Double_t ex, Double_t ey, Double_t ez)
{
   // Set ex, ey and ez values for point number i

   if (i < 0) return;
   if (i >= fNpoints) {
      // re-allocate the object
      TGraph2DErrors::SetPoint(i,0,0,0);
   }
   fEX[i] = ex;
   fEY[i] = ey;
   fEZ[i] = ez;
}


//______________________________________________________________________________
void TGraph2DErrors::Streamer(TBuffer &b)
{
   // Stream an object of class TGraphErrors.

   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      b.ReadClassBuffer(TGraph2DErrors::Class(), this, R__v, R__s, R__c);
   } else {
      b.WriteClassBuffer(TGraph2DErrors::Class(),this);
   }
}
