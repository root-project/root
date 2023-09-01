// @(#)root/hist:$Id$
// Author: L. Moneta    08/2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2008 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TBackCompFitter_H_
#define ROOT_TBackCompFitter_H_

#include "TVirtualFitter.h"
#include "Fit/BasicFCN.h"
#include "Fit/FitResult.h"
#include "Fit/Fitter.h"
#include "Math/IFunctionfwd.h"
#include <vector>

/*
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
   TBackCompFitter( const std::shared_ptr<ROOT::Fit::Fitter> & fitter, const std::shared_ptr<ROOT::Fit::FitData> & data  );

   ~TBackCompFitter() override;

public:

   enum EStatusBits {
      kCanDeleteLast = BIT(9)  // object can be deleted before creating a new one
   };

   // inherited interface
   Double_t  Chisquare(Int_t npar, Double_t *params) const override;
   void      Clear(Option_t *option="") override;
   Int_t     ExecuteCommand(const char *command, Double_t *args, Int_t nargs) override;
   void      FixParameter(Int_t ipar) override;

   void      GetConfidenceIntervals(Int_t n, Int_t ndim, const Double_t *x, Double_t *ci, Double_t cl=0.95) override;
   void      GetConfidenceIntervals(TObject *obj, Double_t cl=0.95) override;

   Double_t *GetCovarianceMatrix() const override;
   Double_t  GetCovarianceMatrixElement(Int_t i, Int_t j) const override;
   Int_t     GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) const override;
   Int_t     GetNumberTotalParameters() const override;
   Int_t     GetNumberFreeParameters() const override;

   Double_t  GetParError(Int_t ipar) const override;
   Double_t  GetParameter(Int_t ipar) const override;
   Int_t     GetParameter(Int_t ipar,char *name,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) const override;
   const char *GetParName(Int_t ipar) const override;
   Int_t     GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) const override;
   Double_t  GetSumLog(Int_t i) override;

   Bool_t    IsFixed(Int_t ipar) const override ;

   void      PrintResults(Int_t level, Double_t amin) const override;
   void      ReleaseParameter(Int_t ipar) override;
   void      SetFitMethod(const char *name) override;
   Int_t     SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh) override;

   void      SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t) ) override;

   /// For using interpreted function passed by the user
   virtual void SetMethodCall(TMethodCall * m) { fMethodCall = m; }

   /// Get reference to Fit configuration (NOTE: it will be invalid when class is deleted)
   ROOT::Fit::FitConfig & GetFitConfig()  { return fFitter->Config(); }

   /// Get reference to Fit Result object (NOTE: it will be invalid when class is deleted)
   const ROOT::Fit::FitResult & GetFitResult() const { return fFitter->Result(); }

   /// Get a copy of the Fit result returning directly a new  TFitResult
   TFitResult * GetTFitResult() const;

   /// Get reference to Fit Data object (NOTE: it will be invalid when class is deleted)
   const ROOT::Fit::FitData & GetFitData() const { return *fFitData; }

   // Return pointer to last used minimizer
   ROOT::Math::Minimizer * GetMinimizer() const;

   // Return pointer to last used objective function
   ROOT::Math::IMultiGenFunction * GetObjFunction() const;

   // Scan likelihood value of  parameter and fill the given graph.
   bool  Scan(unsigned int ipar, TGraph * gr, double xmin = 0, double xmax = 0);

   //    scan likelihood value for two  parameters and fill the given graph.
   //    bool  Scan2D(unsigned int ipar, unsigned int jpar, TGraph2D * gr,
   //                         double xmin = 0, double xmax = 0, double ymin = 0, double ymax = 0);

   // Create contour of two parameters around the minimum
   // pass as option confidence level:  default is a value of 0.683
   bool  Contour(unsigned int ipar, unsigned int jpar, TGraph * gr , double confLevel = 0.683);

   // Set FCN using new interface
   virtual void SetObjFunction(  ROOT::Math::IMultiGenFunction * f);

   // Recreate minimizer and FCN for TMinuit fits and standard printout
   void ReCreateMinimizer();


protected:

   bool ValidParameterIndex(int ipar) const;
   void DoSetDimension();

private:

   //ROOT::Fit::FitData * fFitData;
   std::shared_ptr<ROOT::Fit::FitData>  fFitData;  ///<! Data of the fit
   std::shared_ptr<ROOT::Fit::Fitter>   fFitter;   ///<! Pointer to fitter object
   ROOT::Math::Minimizer * fMinimizer;
   ROOT::Math::IMultiGenFunction * fObjFunc;
   ROOT::Math::IParamMultiFunction * fModelFunc;
   mutable std::vector<double> fCovar;             ///< Cached covariance matrix (NxN)



   ClassDefOverride(TBackCompFitter,1)  // Class providing backward compatibility for fitting by implementing the TVirtualFitter interface

};



#endif //ROOT_TBackCompFitter_H_
