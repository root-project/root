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

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TFitResultPtr
#include "TFitResultPtr.h"
#endif



class TH1;
class TF1;

namespace ROOT {
   namespace Fit {
      class Fitter;
   }
}

class TBinomialEfficiencyFitter: public TObject {

protected:
   TH1             *fDenominator;    //Denominator histogram
   TH1             *fNumerator;      //Numerator histogram
   TF1             *fFunction;       //Function to fit
   Double_t         fEpsilon;        //Precision required for function integration (option "I")
   Bool_t           fFitDone;        //Set to kTRUE when the fit has been done
   Bool_t           fAverage;        //True if the fit function must be averaged over the bin
   Bool_t           fRange;          //True if the fit range must be taken from the function range
   ROOT::Fit::Fitter *fFitter;       //pointer to the real fitter

private:

   void   ComputeFCN(Double_t& f, const Double_t* par);

public:
   TBinomialEfficiencyFitter();
   TBinomialEfficiencyFitter(const TH1 *numerator, const TH1 *denominator);
   virtual ~TBinomialEfficiencyFitter();

   void   Set(const TH1 *numerator, const TH1 *denominator);
   void   SetPrecision(Double_t epsilon);
   TFitResultPtr  Fit(TF1 *f1, Option_t* option = "");
   ROOT::Fit::Fitter * GetFitter();
   Double_t  EvaluateFCN(const Double_t * par) {
      Double_t f = 0;
      ComputeFCN(f, par);
      return f;
   }

   ClassDef(TBinomialEfficiencyFitter, 1) //Binomial Fitter for the division of two histograms


};


#endif
