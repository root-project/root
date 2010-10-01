// @(#)root/hist:$Id$
// Author: Frank Filthaut, Rene Brun   30/05/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//
// TBinomialEfficiencyFitter
//
// Binomial fitter for the division of two histograms.
// Use when you need to calculate a selection's efficiency from two histograms,
// one containing all entries, and one containing the subset of these entries
// that pass the selection, and when you have a parametrization available for
// the efficiency as a function of the variable(s) under consideration.
//
// A very common problem when estimating efficiencies is that of error estimation:
// when no other information is available than the total number of events N and
// the selected number n, the best estimate for the selection efficiency p is n/N.
// Standard binomial statistics dictates that the uncertainty (this presupposes
// sufficiently high statistics that an approximation by a normal distribution is
// reasonable) on p, given N, is
//Begin_Latex
//   #sqrt{#frac{p(1-p)}{N}}.
//End_Latex
// However, when p is estimated as n/N, fluctuations from the true p to its
// estimate become important, especially for low numbers of events, and giving
// rise to biased results.
//
// When fitting a parametrized efficiency, these problems can largely be overcome,
// as a hypothesized true efficiency is available by construction. Even so, simply
// using the corresponding uncertainty still presupposes that Gaussian errors
// yields a reasonable approximation. When using, instead of binned efficiency
// histograms, the original numerator and denominator histograms, a binned maximum
// likelihood can be constructed as the product of bin-by-bin binomial probabilities
// to select n out of N events. Assuming that a correct parametrization of the
// efficiency is provided, this construction in general yields less biased results
// (and is much less sensitive to binning details).
//
// A generic use of this method is given below (note that the method works for 2D
// and 3D histograms as well):
//
// {
//   TH1* denominator;              // denominator histogram
//   TH1* numerator;                // corresponding numerator histogram
//   TF1* eff;                      // efficiency parametrization
//   ....                           // set step sizes and initial parameter
//   ....                           //   values for the fit function
//   ....                           // possibly also set ranges, see TF1::SetRange()
//   TBinomialEfficiencyFitter* f = new TBinomialEfficiencyFitter(
//                                      numerator, denominator);
//   Int_t status = f->Fit(eff, "I");
//   if (status == 0) {
//      // if the fit was successful, display bin-by-bin efficiencies
//      // as well as the result of the fit
//      numerator->Sumw2();
//      TH1* hEff = dynamic_cast<TH1*>(numerator->Clone("heff"));
//      hEff->Divide(hEff, denominator, 1.0, 1.0, "B");
//      hEff->Draw("E");
//      eff->Draw("same");
//   }
// }
//
// Note that this method cannot be expected to yield reliable results when using
// weighted histograms (because the likelihood computation will be incorrect).
//////////////////////////////////////////////////////////////////////////

#include "TBinomialEfficiencyFitter.h"

#include "TMath.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TH1.h"
#include "TF1.h"
#include "TF2.h"
#include "TF3.h"
#include "TVirtualFitter.h"
#include "TEnv.h"

#include <limits>

TVirtualFitter *TBinomialEfficiencyFitter::fgFitter = 0;

const Double_t kDefaultEpsilon = 1E-12;

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
   fEpsilon     = kDefaultEpsilon;
}

//______________________________________________________________________________
TBinomialEfficiencyFitter::TBinomialEfficiencyFitter(const TH1 *numerator, const TH1 *denominator) {
   // Constructor.
   //
   // Note that no objects are copied, so it is up to the user to ensure that the
   // histogram pointers remain valid.
   //
   // Both histograms need to be "consistent". This is not checked here, but in
   // TBinomialEfficiencyFitter::Fit().

   fEpsilon  = kDefaultEpsilon;
   fFunction = 0;
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

   fFitDone     = kFALSE;
   fAverage     = kFALSE;
   fRange       = kFALSE;
}

//______________________________________________________________________________
void TBinomialEfficiencyFitter::SetPrecision(Double_t epsilon)
{
   // Set the required integration precision, see TF1::Integral()
   fEpsilon = epsilon;
}

//______________________________________________________________________________
TVirtualFitter* TBinomialEfficiencyFitter::GetFitter()
{
   // static: Provide access to the underlying fitter object.
   // This may be useful e.g. for the retrieval of additional information (such
   // as the output covariance matrix of the fit).

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
   // from the input fit function f1 (so they should be set before calling
   // this method. This is particularly relevant for the step sizes, taken
   // to be the "error" set on input, as a null step size usually fixes the
   // corresponding parameter. That is protected against, but in such cases
   // an arbitrary starting step size will be used, and the reliability of
   // the fit should be questioned). If parameters are to be fixed, this
   // should be done by specifying non-null parameter limits, with lower
   // limits larger than upper limits.
   //
   // On output, f1 contains the fitted parameters and errors, as well as
   // the number of degrees of freedom, and the goodness-of-fit estimator
   // as given by S. Baker and R. Cousins, Nucl. Instr. Meth. A221 (1984) 437.

   TString opt = option;
   opt.ToUpper();
   fAverage  = opt.Contains("I");
   fRange    = opt.Contains("R");
   Bool_t verbose    = opt.Contains("V");
   if (!f1) return -1;
   fFunction = (TF1*)f1;
   Int_t i, npar;
   npar = f1->GetNpar();
   if (npar <= 0) {
      Error("Fit", "function %s has illegal number of parameters = %d", 
            f1->GetName(), npar);
      return -3;
   }

   // Check that function has same dimension as histogram
   if (!fNumerator || !fDenominator) {
      Error("Fit","No numerator or denominator histograms set");
      return -5;
   }
   if (f1->GetNdim() != fNumerator->GetDimension()) {
      Error("Fit","function %s dimension, %d, does not match histogram dimension, %d",
            f1->GetName(), f1->GetNdim(), fNumerator->GetDimension());
      return -4;
   }
   // Check that the numbers of bins for the histograms match
   if (fNumerator->GetNbinsX() != fDenominator->GetNbinsX() ||
       (f1->GetNdim() > 1 && fNumerator->GetNbinsY() != fDenominator->GetNbinsY()) ||
       (f1->GetNdim() > 2 && fNumerator->GetNbinsZ() != fDenominator->GetNbinsZ())) {
      Error("Fit", "numerator and denominator histograms do not have identical numbers of bins");
      return -6;
   }

   // initialize the fitter

   Int_t maxpar = npar;
   if (!fgFitter) {
      TPluginHandler *h;
      TString fitterName = TVirtualFitter::GetDefaultFitter(); 
      if (fitterName == "") 
         fitterName = gEnv->GetValue("Root.Fitter","Minuit");
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualFitter", fitterName ))) {
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
   Double_t al,bl,we,arglist[100];
   for (i = 0; i < npar; i++) {
      f1->GetParLimits(i,al,bl);
      if (al*bl != 0 && al >= bl) {
         al = bl = 0;
         arglist[nfixed] = i+1;
         nfixed++;
      }
      // assign an ARBITRARY starting error to ensure the parameter won't be fixed!
      we = f1->GetParError(i);
      if (we <= 0) we = 0.3*TMath::Abs(f1->GetParameter(i));
      if (we == 0) we = 0.01;
      fgFitter->SetParameter(i, f1->GetParName(i),
                                f1->GetParameter(i),
                                we, al,bl);
   }
   if (nfixed > 0) fgFitter->ExecuteCommand("FIX",arglist,nfixed); // Otto

   Double_t plist[2];
   plist[0] = 0.5;
   fgFitter->ExecuteCommand("SET ERRDEF",plist,1);

   if (verbose)   { 
      plist[0] = 3;
      fgFitter->ExecuteCommand("SET PRINT",plist,1);
   }

   // perform the actual fit

   fFitDone = kTRUE;
   plist[0] = TVirtualFitter::GetMaxIterations();
   plist[1] = TVirtualFitter::GetPrecision();
   Int_t result = fgFitter->ExecuteCommand("MINIMIZE",plist,2);
   
   //Store fit results in fitFunction
   char parName[50];
   Double_t par;
   Double_t eplus,eminus,eparab,globcc,werr;
   for (i = 0; i < npar; ++i) {
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
                                           Double_t& f, Double_t* par, Int_t /*flag*/)
{
   // Compute the likelihood.

   int nDim = fDenominator->GetDimension();

   int xlowbin  = fDenominator->GetXaxis()->GetFirst();
   int xhighbin = fDenominator->GetXaxis()->GetLast();
   int ylowbin = 0, yhighbin = 0, zlowbin = 0, zhighbin = 0;
   if (nDim > 1) {
      ylowbin  = fDenominator->GetYaxis()->GetFirst();
      yhighbin = fDenominator->GetYaxis()->GetLast();
      if (nDim > 2) {
         zlowbin  = fDenominator->GetZaxis()->GetFirst();
         zhighbin = fDenominator->GetZaxis()->GetLast();
      }
   }

   fFunction->SetParameters(par);

   if (fRange) {
      double xmin, xmax, ymin, ymax, zmin, zmax;

      // This way to ensure that a minimum range chosen exactly at a
      // bin boundary is far from elegant, but is hopefully adequate.

      if (nDim == 1) {
         fFunction->GetRange(xmin, xmax);
         xlowbin  = fDenominator->GetXaxis()->FindBin(xmin);
         xhighbin = fDenominator->GetXaxis()->FindBin(xmax);
      } else if (nDim == 2) {
         fFunction->GetRange(xmin, ymin, xmax, ymax);
         xlowbin  = fDenominator->GetXaxis()->FindBin(xmin);
         xhighbin = fDenominator->GetXaxis()->FindBin(xmax);
         ylowbin  = fDenominator->GetYaxis()->FindBin(ymin);
         yhighbin = fDenominator->GetYaxis()->FindBin(ymax);
      } else if (nDim == 3) {
         fFunction->GetRange(xmin, ymin, zmin, xmax, ymax, zmax);
         xlowbin  = fDenominator->GetXaxis()->FindBin(xmin);
         xhighbin = fDenominator->GetXaxis()->FindBin(xmax);
         ylowbin  = fDenominator->GetYaxis()->FindBin(ymin);
         yhighbin = fDenominator->GetYaxis()->FindBin(ymax);
         zlowbin  = fDenominator->GetZaxis()->FindBin(zmin);
         zhighbin = fDenominator->GetZaxis()->FindBin(zmax);
      }
   }

   // The coding below is perhaps somewhat awkward -- but it is done
   // so that 1D, 2D, and 3D cases can be covered in the same loops.

   f = 0.;

   Int_t npoints = 0;
   Double_t nmax = 0;
   for (int xbin = xlowbin; xbin <= xhighbin; ++xbin) {

      // compute the bin edges
      Double_t xlow = fDenominator->GetXaxis()->GetBinLowEdge(xbin);
      Double_t xup  = fDenominator->GetXaxis()->GetBinLowEdge(xbin+1);

      for (int ybin = ylowbin; ybin <= yhighbin; ++ybin) {

         // compute the bin edges (if applicable)
         Double_t ylow  = (nDim > 1) ? fDenominator->GetYaxis()->GetBinLowEdge(ybin) : 0;
         Double_t yup   = (nDim > 1) ? fDenominator->GetYaxis()->GetBinLowEdge(ybin+1) : 0;

         for (int zbin = zlowbin; zbin <= zhighbin; ++zbin) {

            // compute the bin edges (if applicable)
            Double_t zlow  = (nDim > 2) ? fDenominator->GetZaxis()->GetBinLowEdge(zbin) : 0;
            Double_t zup   = (nDim > 2) ? fDenominator->GetZaxis()->GetBinLowEdge(zbin+1) : 0;

            int bin = fDenominator->GetBin(xbin, ybin, zbin);
            Double_t nDen = fDenominator->GetBinContent(bin);
            Double_t nNum = fNumerator->GetBinContent(bin);

            // count maximum value to use in the likelihood for inf
            // i.e. a number much larger than the other terms  
            if (nDen> nmax) nmax = nDen; 
            if (nDen <= 0.) continue;
            npoints++;
      
            // mu is the average of the function over the bin OR
            // the function evaluated at the bin centre
            // As yet, there is nothing to prevent mu from being 
            // outside the range <0,1> !!

            Double_t mu = 0;
            switch (nDim) {
               case 1:
                  mu = (fAverage) ?
                     fFunction->Integral(xlow, xup, (Double_t*)0, fEpsilon) 
                        / (xup-xlow) :
                     fFunction->Eval(fDenominator->GetBinCenter(bin));
                  break;
               case 2:
                  {
                     mu = (fAverage) ?
                     fFunction->Integral(xlow, xup, ylow, yup, fEpsilon) 
                        / ((xup-xlow)*(yup-ylow)) :
                     fFunction->Eval(fDenominator->GetXaxis()->GetBinCenter(xbin),
                     fDenominator->GetYaxis()->GetBinCenter(ybin));
                  }
                  break;
               case 3:
                  {
                     mu = (fAverage) ?
                        fFunction->Integral(xlow, xup, ylow, yup, zlow, zup, fEpsilon)
                           / ((xup-xlow)*(yup-ylow)*(zup-zlow)) :
                        fFunction->Eval(fDenominator->GetXaxis()->GetBinCenter(xbin),
                                 fDenominator->GetYaxis()->GetBinCenter(ybin),
                                 fDenominator->GetZaxis()->GetBinCenter(zbin));
                  }
            }

            // binomial formula (forgetting about the factorials)
            if (nNum != 0.) {
               if (mu > 0.)
                  f -= nNum * TMath::Log(mu*nDen/nNum);
               else
                  f -= nmax * -1E30; // crossing our fingers
            }
            if (nDen - nNum != 0.) {
               if (1. - mu > 0.)
                  f -= (nDen - nNum) * TMath::Log((1. - mu)*nDen/(nDen-nNum));
               else 
                  f -= nmax * -1E30; // crossing our fingers
            }
         }
      }
   }

   fFunction->SetNumberFitPoints(npoints);
   fFunction->SetChisquare(2.*f);    // store goodness of fit (Baker&Cousins)
}

//______________________________________________________________________________
void BinomialEfficiencyFitterFCN(Int_t& npar, Double_t* gin, Double_t& f, 
                                 Double_t* par, Int_t flag)
{
   // Function called by the minimisation package. The actual functionality is 
   // passed on to the TBinomialEfficiencyFitter::ComputeFCN() member function.

   TBinomialEfficiencyFitter* fitter = dynamic_cast<TBinomialEfficiencyFitter*>(TBinomialEfficiencyFitter::GetFitter()->GetObjectFit());
   if (!fitter) {
      Error("binomialFCN","Invalid fit object encountered!");
      return;
   }
   fitter->ComputeFCN(npar, gin, f, par, flag);
}
