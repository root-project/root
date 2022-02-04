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
   RooNonCentralChiSquare(const RooNonCentralChiSquare& other, const char* name=0) ;
   TObject* clone(const char* newname) const override { return new RooNonCentralChiSquare(*this,newname); }
   inline ~RooNonCentralChiSquare() override { }

   void SetErrorTolerance(Double_t t) {fErrorTol = t;}
   void SetMaxIters(Int_t mi) {fMaxIters = mi;}
   void SetForceSum(Bool_t flag);


   Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
   Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

protected:

   RooRealProxy x ;
   RooRealProxy k ;
   RooRealProxy lambda ;
   Double_t fErrorTol;
   Int_t fMaxIters;
   Bool_t fForceSum;
   mutable Bool_t fHasIssuedConvWarning;
   mutable Bool_t fHasIssuedSumWarning;
   Double_t evaluate() const override ;

private:

   ClassDefOverride(RooNonCentralChiSquare,1) // non-central chisquare pdf
};

#endif
