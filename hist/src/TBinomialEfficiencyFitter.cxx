// @(#)root/hist:$Name:  $:$Id: TBinomialEfficiencyFitter.h,v 1.88 2007/03/02 15:37:18 couet Exp $
// Author: Frank Fielthaut, Rene Brun   30/05/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBinomialEfficiencyFitter                                            //      
//                                                                      //
// Binomial Fitter for the division of two histograms.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBinomialEfficiencyFitter.h"

#include "TMath.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TH1.h"
#include "TF1.h"
#include "TVirtualFitter.h"

TVirtualFitter *TBinomialEfficiencyFitter::fgFitter = 0;

ClassImp(TBinomialEfficiencyFitter)


//______________________________________________________________________________
TBinomialEfficiencyFitter::TBinomialEfficiencyFitter() {
   // default constructor
  
   fNumerator   = 0;
   fDenominator = 0;
   fFunction    = 0;
   fFitDone     = kFALSE;
   fAverage     = kFALSE;
   fRange       = kFALSE;
}

//______________________________________________________________________________
TBinomialEfficiencyFitter::TBinomialEfficiencyFitter(const TH1 *numerator, const TH1 *denominator)
{
   // Constructor

   Set(numerator,denominator);
}

//______________________________________________________________________________
TBinomialEfficiencyFitter::~TBinomialEfficiencyFitter() {
   // destructor
   
   delete fgFitter; fgFitter = 0;
}

//______________________________________________________________________________
void TBinomialEfficiencyFitter::Set(const TH1 *numerator, const TH1 *denominator)
{
   // Initialize with a new set of inputs.

   fNumerator   = (TH1*)numerator;
   fDenominator = (TH1*)denominator;

   // Note that there is currently NO check that the given histograms are consistent!

   fFitDone     = kFALSE;
   fAverage     = kFALSE;
   fRange       = kFALSE;
}

//______________________________________________________________________________
TVirtualFitter* TBinomialEfficiencyFitter::GetFitter() {
   // static: Provide access to the underlying fitter object

   return fgFitter;
}

//______________________________________________________________________________
Int_t TBinomialEfficiencyFitter::Fit(TF1 *f1, Option_t* option) 
{
   // Carry out the fit of the given function to the given histograms.
   //
   // If option "I" is used, the fit function will be averaged over the
   // bin (the default is to evaluate it simply at the bin center).
   //
   // If option "R" is used, the fit range will be taken from the fit
   // function (the default is to use the entire histogram).
   //
   // Note that all parameter values, limits, and step sizes are copied
   // from the input fit function f1(so they should be set before calling
   // this method).
   // In output f1 contains the fitted parameters and errors

   TString opt = option;
   opt.ToUpper();
   fAverage  = opt.Contains("I");
   fRange    = opt.Contains("R");
   fFunction = (TF1*)f1;
   Int_t i, npar;
   npar = f1->GetNpar();
   if (npar <= 0) {
      Error("Fit", "function %s has illegal number of parameters = %d", f1->GetName(), npar);
      return -3;
   }

   // Check that function has same dimension as histogram
   if (!f1) return -1;
   if (!fNumerator || !fDenominator) {
      Error("Fit","No numerator or denominator histograms set");
      return -5;
   }
   if (f1->GetNdim() != fNumerator->GetDimension()) {
      Error("Fit","function %s dimension, %d, does not match histogram dimension, %d",
            f1->GetName(), f1->GetNdim(), fNumerator->GetDimension());
      return -4;
   }

   // initialize the fitter

   Int_t maxpar = npar;
   if (!fgFitter) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualFitter","Minuit"))) {
         if (h->LoadPlugin() == -1)
            return 0;
         fgFitter = (TVirtualFitter*) h->ExecPlugin(1, maxpar);
      }
      if (!fgFitter) printf("ERROR fgFitter is NULL\n");
   }

   fgFitter->SetObjectFit(this);
   fgFitter->Clear();
   fgFitter->SetFCN(BinomialEfficiencyFitterFCN);
   Int_t nfixed = 0;
   Double_t al,bl,arglist[100];
   for (i = 0; i < npar; i++) {
      f1->GetParLimits(i,al,bl);
      if (al*bl != 0 && al >= bl) {
         al = bl = 0;
         arglist[nfixed] = i+1;
         nfixed++;
      }
      // assign an ARBITRARY starting error to ensure the parameter won't be fixed!
      if (f1->GetParError(i) <= 0) f1->SetParError(i, 0.01);
      fgFitter->SetParameter(i, f1->GetParName(i),
			        f1->GetParameter(i),
			        f1->GetParError(i), al,bl);
   }
   if(nfixed > 0)fgFitter->ExecuteCommand("FIX",arglist,nfixed); // Otto

   Double_t plist[1];
   plist[0] = 0.5;
   fgFitter->ExecuteCommand("SET ERRDEF",plist,1);

   // perform the actual fit

   fFitDone = kTRUE;
   Int_t result = fgFitter->ExecuteCommand("MINIMIZE",0,0);
   
   //Store fit results in fitFunction
   char parName[50];
   Double_t par, we;
   Double_t eplus,eminus,eparab,globcc,werr;
   for (i=0;i<npar;i++) {
      fgFitter->GetParameter(i,parName, par,we,al,bl);
      fgFitter->GetErrors(i,eplus,eminus,eparab,globcc);
      if (eplus > 0 && eminus < 0) werr = 0.5*(eplus-eminus);
      else                         werr = we;
      f1->SetParameter(i,par);
      f1->SetParError(i,werr);
   }
   f1->SetNDF(f1->GetNumberFitPoints()-npar+nfixed);
   return result;
}

//______________________________________________________________________________
void TBinomialEfficiencyFitter::ComputeFCN(Int_t& /*npar*/, Double_t* /* gin */,
					  Double_t& f, Double_t* par, Int_t /*flag*/) {
   // Compute the likelihood.

   int lowbin  = fDenominator->GetXaxis()->GetFirst();
   int highbin = fDenominator->GetXaxis()->GetLast();
   if (fRange) {
      double xmin, xmax;
      fFunction->GetRange(xmin, xmax);
      fFunction->SetParameters(par);

      // Note: this way to ensure that a minimum range chosen exactly at a
      //       bin boundary is far from elegant, but is hopefully adequate.
      lowbin  = fDenominator->GetXaxis()->FindBin(xmin);
      highbin = fDenominator->GetXaxis()->FindBin(xmax);
   }

   f = 0.;

   Int_t npoints = 0;
   for (int bin = lowbin; bin <= highbin; ++bin) {

      // compute the bin edge
      double xlow = fDenominator->GetBinLowEdge(bin);
      double xup  = fDenominator->GetBinLowEdge(bin+1);
      double N    = fDenominator->GetBinContent(bin);
      double n    = fNumerator->GetBinContent(bin);
      if (N <= 0.) continue;
      npoints++;
      
      // mu is the average of the function over the bin OR
      // the function evaluated at the bin centre

      // As yet, there is nothing to prevent mu from being outside the range <0,1> !!
      double mu = (fAverage) ?
        fFunction->Integral(xlow, xup) / (xup-xlow) :
        fFunction->Eval(fDenominator->GetBinCenter(bin));

      // binomial formula (forgetting about the factorials)
      if (n != 0.)
         if (mu > 0.) 
            f -= n * TMath::Log(mu);
         else
            f -= n * -1E30; // crossing our fingers
      if (N - n != 0.)
         if (1. - mu > 0.)
            f -= (N - n) * TMath::Log(1. - mu);
         else 
            f -= (N - n) * -1E30; // crossing our fingers
   }
   fFunction->SetNumberFitPoints(npoints);
   fFunction->SetChisquare(f); //store likelihood instead of chisquare!
}

//______________________________________________________________________________
void BinomialEfficiencyFitterFCN(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t flag) {
   // Function called by the minimisation package. The actual functionality is passed
   // on to the TBinomialEfficiencyFitter::ComputeFCN member function.

   TBinomialEfficiencyFitter* fitter = dynamic_cast<TBinomialEfficiencyFitter*>(TBinomialEfficiencyFitter::GetFitter()->GetObjectFit());
   if (!fitter) {
      fitter->Error("binomialFCN","Invalid fit object encountered!");
      return;
   }
   fitter->ComputeFCN(npar, gin, f, par, flag);
}
