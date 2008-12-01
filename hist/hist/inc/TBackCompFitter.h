// @(#)root/minuit2:$Id$
// Author: L. Moneta    08/2008  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2008 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TBackCompFitter_H_
#define ROOT_TBackCompFitter_H_

#ifndef ROOT_TVirtualFitter
#include "TVirtualFitter.h"
#endif

#ifndef ROOT_Fit_Fitter
#include "Fit/Fitter.h"
#endif

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif


#include <vector>

/**
    TVirtualFitter backward compatibility implementation using new ROOT::Fit::Fitter
*/

namespace ROOT { 
   namespace Fit { 
      class FitData; 
   }
}


class TBackCompFitter : public TVirtualFitter {

public:



   TBackCompFitter();

   TBackCompFitter(ROOT::Fit::Fitter & fitter, ROOT::Fit::FitData * ); 

   virtual ~TBackCompFitter();

public:

   // inherited interface
   virtual Double_t  Chisquare(Int_t npar, Double_t *params) const;
   virtual void      Clear(Option_t *option="");
   virtual Int_t     ExecuteCommand(const char *command, Double_t *args, Int_t nargs);
   virtual void      FixParameter(Int_t ipar);

   virtual void      GetConfidenceIntervals(Int_t n, Int_t ndim, const Double_t *x, Double_t *ci, Double_t cl=0.95);
   virtual void      GetConfidenceIntervals(TObject *obj, Double_t cl=0.95);

   virtual Double_t *GetCovarianceMatrix() const;
   virtual Double_t  GetCovarianceMatrixElement(Int_t i, Int_t j) const;
   virtual Int_t     GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) const;
   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
   virtual Int_t     GetNumberTotalParameters() const;
   virtual Int_t     GetNumberFreeParameters() const;

   virtual Double_t  GetParError(Int_t ipar) const;
   virtual Double_t  GetParameter(Int_t ipar) const;
   virtual Int_t     GetParameter(Int_t ipar,char *name,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) const;
   virtual const char *GetParName(Int_t ipar) const;
   virtual Int_t     GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) const;
   virtual Double_t  GetSumLog(Int_t i);

   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
   virtual Bool_t    IsFixed(Int_t ipar) const ;

   virtual void      PrintResults(Int_t level, Double_t amin) const;
   virtual void      ReleaseParameter(Int_t ipar);
   virtual void      SetFitMethod(const char *name);
   virtual Int_t     SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh);

   virtual void      SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t) );
   // this for CINT (interactive functions)
   virtual void      SetFCN(void * );
   // for using interpreted function passed by the user
   virtual void SetMethodCall(TMethodCall * m) { fMethodCall = m; }


   // set FCN using new interface
   virtual void SetObjFunction(  ROOT::Math::IMultiGenFunction * f);

   // recreate minimizer and FCN for TMinuit fits and standard printout 
   void ReCreateMinimizer();
   

protected: 

   // internal methods

   bool ValidParameterIndex(int ipar) const;
  
   void DoSetDimension(); 
   
   
private:


   ROOT::Fit::FitData * fFitData;
   ROOT::Math::Minimizer * fMinimizer;
   ROOT::Fit::Fitter fFitter; 
   ROOT::Math::IMultiGenFunction * fObjFunc; 
   ROOT::Math::IParamMultiFunction * fModelFunc; 
   mutable std::vector<double> fCovar; // cached covariance matrix (NxN)


   ClassDef(TBackCompFitter,1)  // Class providing backward compatibility for fitting by implementing the TVirtualFitter interface

};



#endif //ROOT_TBackCompFitter_H_
