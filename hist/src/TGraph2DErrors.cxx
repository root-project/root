// @(#)root/hist:$Name:  $:$Id: TGraph2DErrors.cxx,v 1.00
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
#include "TGraph2DErrors.h"
#include "TMath.h"
#include "TPolyLine.h"
#include "TPolyMarker.h"
#include "TVirtualPad.h"
#include "TVirtualFitter.h"
#include "TView.h"
#include "THLimitsFinder.h"
#include "TStyle.h"

ClassImp(TGraph2DErrors)

//______________________________________________________________________________
//
//   A TGraph2DErrors is a TGraph2D with errors.
//

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
      TGraph2DErrors::Class()->ReadBuffer(b, this, R__v, R__s, R__c);

      if (!gROOT->ReadingObject()) {
         fDirectory = gDirectory;
         if (!gDirectory->GetList()->FindObject(this)) gDirectory->Append(this);
      }
      ResetBit(kCanDelete);
   } else {
      TGraph2DErrors::Class()->WriteBuffer(b,this);
   }
}
