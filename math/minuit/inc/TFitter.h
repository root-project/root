// @(#)root/minuit:$Id$
// Author: Rene Brun   31/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TFitter
#define ROOT_TFitter


#include "TVirtualFitter.h"

class TMinuit;

class TFitter : public TVirtualFitter {

private:
   Int_t      fNlog;       //Number of elements in fSunLog
   Double_t  *fCovar;      //Covariance matrix
   Double_t  *fSumLog;     //Sum of logs (array of fNlog elements)
   TMinuit   *fMinuit;     //pointer to the TMinuit object

   TFitter(const TFitter&); // Not implemented
   TFitter& operator=(const TFitter&); // Not implemented

public:
   TFitter(Int_t maxpar = 25);
   ~TFitter() override;
   Double_t   Chisquare(Int_t npar, Double_t *params) const override ;
   void       Clear(Option_t *option="") override;
   Int_t      ExecuteCommand(const char *command, Double_t *args, Int_t nargs) override;
   // virtual void       FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
   // virtual void       FitChisquareI(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
   // virtual void       FitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
   // virtual void       FitLikelihoodI(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
   void       FixParameter(Int_t ipar) override;
   void       GetConfidenceIntervals(Int_t n, Int_t ndim, const Double_t *x, Double_t *ci, Double_t cl=0.95) override;
   void       GetConfidenceIntervals(TObject *obj, Double_t cl=0.95) override;
   Double_t  *GetCovarianceMatrix() const override;
   Double_t   GetCovarianceMatrixElement(Int_t i, Int_t j) const override;
   Int_t      GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) const override;
   TMinuit           *GetMinuit() const {return fMinuit;}
   Int_t      GetNumberTotalParameters() const override;
   Int_t      GetNumberFreeParameters() const override;
   Double_t   GetParError(Int_t ipar) const override;
   Double_t   GetParameter(Int_t ipar) const override;
   Int_t      GetParameter(Int_t ipar,char *name,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) const override;
   const char *GetParName(Int_t ipar) const override;
   Int_t      GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) const override;
   Double_t   GetSumLog(Int_t i) override;
   Bool_t     IsFixed(Int_t ipar) const override;
   void       PrintResults(Int_t level, Double_t amin) const override;
   void       ReleaseParameter(Int_t ipar) override;
   void       SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t)) override;
   void       SetFitMethod(const char *name) override;
   Int_t      SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh) override;

    ClassDefOverride(TFitter,0)  //The ROOT standard fitter based on TMinuit
};

#endif
