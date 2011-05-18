// @(#)root/hist:$Id$
// Author: Frank Filthaut, Rene Brun   30/05/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBinomialEfficiencyFitter
#define ROOT_TBinomialEfficiencyFitter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBinomialEfficiencyFitter                                            //
//                                                                      //
// Binomial Fitter for the division of two histograms.                  //
// Use when you need to calculate a selection's efficiency from two     //
// histograms, one containing all entries, and one containing the subset//
// of these entries that pass the selection                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"

class TH1;
class TF1;
class TVirtualFitter;

class TBinomialEfficiencyFitter: public TObject {

protected:
   TH1             *fDenominator;    //Denominator histogram
   TH1             *fNumerator;      //Numerator histogram
   TF1             *fFunction;       //Function to fit
   Double_t         fEpsilon;        //Precision required for function integration (option "I")
   Bool_t           fFitDone;        //Set to kTRUE when the fit has been done
   Bool_t           fAverage;        //True if the fit function must be averaged over the bin
   Bool_t           fRange;          //True if the fit range must be taken from the function range
   static TVirtualFitter  *fgFitter; //pointer to the real fitter
  
public:
   TBinomialEfficiencyFitter();
   TBinomialEfficiencyFitter(const TH1 *numerator, const TH1 *denominator);
   virtual ~TBinomialEfficiencyFitter();

   void   Set(const TH1 *numerator, const TH1 *denominator);
   void   SetPrecision(Double_t epsilon);
   Int_t  Fit(TF1 *f1, Option_t* option = "");
   static TVirtualFitter* GetFitter();
   void   ComputeFCN(Int_t& npar, Double_t* /* gin */, Double_t& f, Double_t* par, Int_t flag);

   ClassDef(TBinomialEfficiencyFitter, 1) //Binomial Fitter for the division of two histograms

};

void BinomialEfficiencyFitterFCN(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t flag);

#endif
