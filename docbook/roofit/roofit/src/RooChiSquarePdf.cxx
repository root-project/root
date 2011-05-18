/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   Kyle Cranmer
 *                                                                           *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// The PDF of the Chi Square distribution for n degrees of freedom.  
// Oddly, this is hard to find in ROOT (except via relation to GammaDist).
// Here we also implement the analytic integral.
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>
#include "TMath.h"
#include "RooChiSquarePdf.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"

ClassImp(RooChiSquarePdf)
;


//_____________________________________________________________________________
RooChiSquarePdf::RooChiSquarePdf()
{
}


//_____________________________________________________________________________
RooChiSquarePdf::RooChiSquarePdf(const char* name, const char* title, 
                           RooAbsReal& x, RooAbsReal& ndof): 
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _ndof("ndof","ndof", this, ndof)
{
}



//_____________________________________________________________________________
RooChiSquarePdf::RooChiSquarePdf(const RooChiSquarePdf& other, const char* name) :
  RooAbsPdf(other, name), 
  _x("x", this, other._x), 
  _ndof("ndof",this,other._ndof)
{
}


//_____________________________________________________________________________
Double_t RooChiSquarePdf::evaluate() const 
{

  if(_x <= 0) return 0;

  return  pow(_x,(_ndof/2.)-1.) * exp(-_x/2.) / TMath::Gamma(_ndof/2.) / pow(2.,_ndof/2.);


}


//_____________________________________________________________________________
Int_t RooChiSquarePdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const 
{
  // No analytical calculation available (yet) of integrals over subranges
  if (rangeName && strlen(rangeName)) {
    return 0 ;
  }


  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}


//_____________________________________________________________________________
Double_t RooChiSquarePdf::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  assert(code==1) ;
  Double_t xmin = _x.min(rangeName); Double_t xmax = _x.max(rangeName);

  // TMath::Prob needs ndof to be an integer, or it returns 0.
  //  return TMath::Prob(xmin, _ndof) - TMath::Prob(xmax,_ndof);

  // cumulative is known based on lower incomplete gamma function, or regularized gamma function
  // Wikipedia defines lower incomplete gamma function without the normalization 1/Gamma(ndof), 
  // but it is included in the ROOT implementation.  
  Double_t pmin = TMath::Gamma(_ndof/2,xmin/2);
  Double_t pmax = TMath::Gamma(_ndof/2,xmax/2);

  // only use this if range is appropriate
  return pmax-pmin;
}

