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

#ifndef ROOT_Fit_DataVector
#include "Fit/DataVector.h"
#endif

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif



#include <vector>

/**
    TVirtualFitter backward compatibility implementation using new ROOT::Fit::Fitter
*/

class TGraph; 
class TFitResult; 

namespace ROOT { 
   namespace Fit { 
      class FitData; 
   }
   namespace Math { 
      class Minimizer; 
   }
}


class TBackCompFitter : public TVirtualFitter {

public:



   TBackCompFitter();

   //TBackCompFitter(ROOT::Fit::Fitter & fitter, ROOT::Fit::FitData * ); 
   TBackCompFitter( std::auto_ptr<ROOT::Fit::Fitter>  fitter, std::auto_ptr<ROOT::Fit::FitData> data  ); 

   virtual ~TBackCompFitter();

public:

   enum { 
      kCanDeleteLast = BIT(9)  // object can be deleted before creating a new one
   };

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

   ///!!!! new method (of this interface) 


   // get reference to Fit configuration (NOTE: it will be invalid when class is deleted) 
   ROOT::Fit::FitConfig & GetFitConfig()  { return fFitter->Config(); }

   // get reference to Fit Result object (NOTE: it will be invalid when class is deleted) 
   const ROOT::Fit::FitResult & GetFitResult() const { return fFitter->Result(); }

   // get a copy of the Fit result returning directly a new  TFitResult 
   TFitResult * GetTFitResult() const; 

   // get reference to Fit Data object (NOTE: it will be invalid when class is deleted) 
   const ROOT::Fit::FitData & GetFitData() const { return *fFitData; }

   // return pointer to last used minimizer
   ROOT::Math::Minimizer * GetMinimizer() const; 

   // return pointer to last used objective function
   ROOT::Math::IMultiGenFunction * GetObjFunction() const; 
   
   // scan likelihood value of  parameter and fill the given graph. 
   bool  Scan(unsigned int ipar, TGraph * gr, double xmin = 0, double xmax = 0);

//    // scan likelihood value for two  parameters and fill the given graph. 
//    bool  Scan2D(unsigned int ipar, unsigned int jpar, TGraph2D * gr, 
//                         double xmin = 0, double xmax = 0, double ymin = 0, double ymax = 0);

   // create contour of two parameters around the minimum
   // pass as option confidence level:  default is a value of 0.683 
   bool  Contour(unsigned int ipar, unsigned int jpar, TGraph * gr , double confLevel = 0.683); 
   
   // set FCN using new interface
   virtual void SetObjFunction(  ROOT::Math::IMultiGenFunction * f);

   // recreate minimizer and FCN for TMinuit fits and standard printout 
   void ReCreateMinimizer();
   

protected: 

   // internal methods

   bool ValidParameterIndex(int ipar) const;
  
   void DoSetDimension(); 
   
   
private:


   //ROOT::Fit::FitData * fFitData;          
   std::auto_ptr<ROOT::Fit::FitData>  fFitData;  //! data of the fit (managed by TBackCompFitter)
   std::auto_ptr<ROOT::Fit::Fitter>   fFitter;   //! pointer to fitter object (managed by TBackCompFitter)
   ROOT::Math::Minimizer * fMinimizer;
   ROOT::Math::IMultiGenFunction * fObjFunc; 
   ROOT::Math::IParamMultiFunction * fModelFunc; 
   mutable std::vector<double> fCovar; // cached covariance matrix (NxN)



   ClassDef(TBackCompFitter,1)  // Class providing backward compatibility for fitting by implementing the TVirtualFitter interface

};



#endif //ROOT_TBackCompFitter_H_
