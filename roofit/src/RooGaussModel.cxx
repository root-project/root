/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGaussModel.cc,v 1.2 2001/06/19 02:17:19 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// 

#include <iostream.h>
#include <complex.h>
#include "RooFitModels/RooGaussModel.hh"
#include "RooFitCore/RooMath.hh"

ClassImp(RooGaussModel) 
;


RooGaussModel::RooGaussModel(const char *name, const char *title, RooRealVar& x, 
			     RooAbsReal& _mean, RooAbsReal& _sigma) : 
  RooResolutionModel(name,title,x), 
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma)
{  
}


RooGaussModel::RooGaussModel(const RooGaussModel& other, const char* name) : 
  RooResolutionModel(other,name),
  mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma)
{
}


RooGaussModel::~RooGaussModel()
{
  // Destructor
}



Int_t RooGaussModel::basisCode(const char* name) const 
{
  if (!TString("exp(-abs(@0)/@1)").CompareTo(name)) return expBasisPlus ;
  if (!TString("exp(-abs(-@0)/@1)").CompareTo(name)) return expBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisPlus ;
  if (!TString("exp(-abs(-@0)/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisPlus ;
  if (!TString("exp(-abs(-@0)/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisMinus ;
  return 0 ;
} 



Double_t RooGaussModel::evaluate(const RooDataSet* dset) const 
{  
  // Special case: no convolution
  if (_basisCode==noBasis) {
    Double_t xprime = (x-mean)/sigma ;
    return exp(-0.5*xprime*xprime) ;
  }

  // Precalculate intermediate quantities  
  Double_t sign = (_basisCode==expBasisPlus||_basisCode==sinBasisPlus||_basisCode==cosBasisPlus)?-1:1 ;
  Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
  Double_t xprime = sign*(x-mean)/tau ;
  Double_t c = sigma/(sqrt(2)*tau) ; 
  Double_t u = xprime/(2*c) ;

  Double_t result ;
  if (_basisCode==expBasisPlus||_basisCode==expBasisMinus) {
    result = 0.25/tau * exp(xprime+c*c) * erfc(u+c) ;
    return result ;
  }

  Double_t swt = ((RooAbsReal*)basis().getParameter(2))->getVal() * tau * sign ;
  RooComplex evalTerm = evalCerf(swt,u,c) ;    

  if (_basisCode==sinBasisPlus||_basisCode==sinBasisMinus) {
    result = 0.25/tau * evalTerm.im() ;
    return result ;
  }

  if (_basisCode==cosBasisPlus||_basisCode==cosBasisMinus) {
    result = 0.25/tau * evalTerm.re() ;
    return result ;
  }

  return result ;
}



Int_t RooGaussModel::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  switch(_basisCode) {

  // Analytical integration capability of raw PDF
  case noBasis:
    if (matchArgs(allVars,analVars,convVar())) return 1 ;
    break ;

  // Analytical integration capability of convoluted PDF
  case expBasisPlus:
  case expBasisMinus:
  case sinBasisPlus:
  case sinBasisMinus:
  case cosBasisPlus:
  case cosBasisMinus:
    if (matchArgs(allVars,analVars,convVar())) return 1 ;
    break ;
  }
  
  return 0 ;
}



Double_t RooGaussModel::analyticalIntegral(Int_t code) const 
{
  static Double_t root2 = sqrt(2) ;
  static Double_t rootPiBy2 = sqrt(atan2(0.0,-1.0)/2.0);

  // No integration
  if (code==0) return getVal() ;

  // Code must be 0 or 1
  assert(code==1) ;
  
  // Unconvoluted function integral
  if (_basisCode==noBasis) {
    Double_t xscale = root2*sigma;
    return rootPiBy2*sigma*(erf((x.max()-mean)/xscale)-erf((x.min()-mean)/xscale));
  }

  // Calculate intermediate variables
  Double_t sign = (_basisCode==expBasisPlus)?-1:1 ;
  Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
  Double_t c = sigma/(sqrt(2)*tau) ; 
  Double_t xpmin = sign*(x.min()-mean)/tau ;
  Double_t xpmax = sign*(x.max()-mean)/tau ;

  // EXP basis function integrals
  if (_basisCode==expBasisPlus||_basisCode==expBasisMinus) {   
    Double_t result = sign * 0.25 * exp(c*c) * ( erf(xpmax/(2*c)) - erf(xpmin/(2*c)) 
						 + exp(xpmax)*erfc(xpmax/(2*c)+c)
						 - exp(xpmin)*erfc(xpmin/(2*c)+c) ) ;     
    return result ;
  }

  // Calculate additional intermediate results for oscillating terms
  Double_t umin = xpmin/(2*c) ;
  Double_t umax = xpmax/(2*c) ;
  Double_t swt = ((RooAbsReal*)basis().getParameter(2))->getVal() * tau * sign ;
  RooComplex evalDif(evalCerf(swt,umax,c) - evalCerf(swt,umin,c)) ;

  // SIN basis function integrals
  if (_basisCode==sinBasisPlus||_basisCode==sinBasisMinus) {    
    Double_t result = 0.25*sign/(1+swt*swt) * ( evalDif.im() - swt*evalDif.re() + erf(umax) - erf(umin) ) ;
    return result ;
  }

  // COS basis function integrals
  if (_basisCode==cosBasisPlus||_basisCode==cosBasisMinus) {
    Double_t result = 0.25*sign/(1+swt*swt) * ( evalDif.re() + swt*evalDif.im() + erf(umax) - erf(umin) ) ;
    return result ;
  }

}





RooComplex RooGaussModel::evalCerf(Double_t swt, Double_t u, Double_t c) const
//                                 sign*omg_tau, xprime/(2*c), c
{
// Calculate exp(-u^2) cwerf(swt*c + i(u+c)), taking care of
// numerical instabilities

  static Double_t rootpi= sqrt(atan2(0,-1));
  RooComplex z(swt*c,u+c);

  if(z.im() > -4) {
    // ComplexErrFunc actually evaluates the CERNLIB CWERF which
    // is exp(-z*z)*erfc(-i z)
    return RooMath::ComplexErrFunc(z)*exp(-u*u);
  }
  else {
    // use the approximation: erf(z) = exp(-z*z)/(sqrt(pi)*z)
    // to explicitly cancel the divergent exp(y*y) behaviour of
    // CWERF for z = x + i y with large negative y
    RooComplex zc(u+c,-swt*c);
    RooComplex zsq= z*z;
    RooComplex v= -zsq - u*u;
    return v.exp()*(-zsq.exp()/(zc*rootpi) + 1)*2;
  }
}



