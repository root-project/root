// @(#)root/base:$Name$:$Id$
// Author: Rene Brun   31/08/99
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TVirtualFitter.h"

TVirtualFitter *TVirtualFitter::fgFitter    = 0;
Int_t           TVirtualFitter::fgMaxpar    = 0;
Int_t           TVirtualFitter::fgMaxiter   = 5000;
Double_t        TVirtualFitter::fgPrecision = 1e-6;

ClassImp(TVirtualFitter)

//______________________________________________________________________________
TVirtualFitter::TVirtualFitter()
{
//*-*-*-*-*-*-*-*-*-*-*default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===================
}

//______________________________________________________________________________
TVirtualFitter::~TVirtualFitter()
{
//*-*-*-*-*-*-*-*-*-*-*default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==================

   fgFitter    = 0;
   fgMaxpar    = 0;
}


//______________________________________________________________________________
TVirtualFitter *TVirtualFitter::Fitter(TObject *obj, Int_t maxpar)
{
   // Static function returning a pointer to the current fitter.
   // If the fitter does not exist, the default TFitter is created

   if (!fgFitter) {
      if (gROOT->LoadClass("TFitter","Minuit")) return 0;
      gROOT->ProcessLineFast(Form("new TFitter(%d);", maxpar));
      fgMaxpar = maxpar;
   }
   if (fgFitter && maxpar > fgMaxpar) {
      delete fgFitter;
      gROOT->ProcessLineFast(Form("new TFitter(%d);", maxpar));
      fgMaxpar = maxpar;
   }

   if (fgFitter)  fgFitter->SetObjectFit(obj);
   return fgFitter;
}

//______________________________________________________________________________
void TVirtualFitter::SetFitter(TVirtualFitter *fitter, Int_t maxpar)
{
   // Static function to set an alternative fitter

   fgFitter    = fitter;
   fgMaxpar    = maxpar;
}

//______________________________________________________________________________
Int_t TVirtualFitter::GetMaxIterations()
{
   // Return the maximum number of iterations

   return fgMaxiter;
}

//______________________________________________________________________________
Double_t TVirtualFitter::GetPrecision()
{
   // Return the fit relative precision

   return fgPrecision;
}

//______________________________________________________________________________
void TVirtualFitter::SetMaxIterations(Int_t niter)
{
   // Set the maximum number of iterations

   fgMaxiter  = niter;
}

//______________________________________________________________________________
void TVirtualFitter::SetPrecision(Double_t prec)
{
   // Set the fit relative precision

   fgPrecision = prec;
}
