 /*****************************************************************************
  * Project: RooFit                                                           *
  * @(#)root/roofit:$Id$   *
  *                                                                           *
  * RooFit NonCentralChisquare PDF                                            *
  *                                                                           *
  * Author: Kyle Cranmer                                                      *
  *                                                                           *
  *****************************************************************************/

#ifndef ROO_NONCENTRALCHISQUARE
#define ROO_NONCENTRALCHISQUARE

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"

class RooNonCentralChiSquare : public RooAbsPdf {
public:
   RooNonCentralChiSquare() {} ;
   RooNonCentralChiSquare(const char *name, const char *title,
                          RooAbsReal& _x,
                          RooAbsReal& _k,
                          RooAbsReal& _lambda);
   RooNonCentralChiSquare(const RooNonCentralChiSquare& other, const char* name=nullptr) ;
   TObject* clone(const char* newname=nullptr) const override { return new RooNonCentralChiSquare(*this,newname); }

   void SetErrorTolerance(double t) {fErrorTol = t;}
   void SetMaxIters(Int_t mi) {fMaxIters = mi;}
   void SetForceSum(bool flag);


   Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
   double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

protected:

   RooRealProxy x ;
   RooRealProxy k ;
   RooRealProxy lambda ;
   double fErrorTol = 0.0;
   Int_t fMaxIters = 0;
   bool fForceSum = false;
   mutable bool fHasIssuedConvWarning = false;
   mutable bool fHasIssuedSumWarning = false;
   double evaluate() const override ;

private:

   ClassDefOverride(RooNonCentralChiSquare,1) // non-central chisquare pdf
};

#endif
