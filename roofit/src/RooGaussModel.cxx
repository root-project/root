/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGaussModel.cc,v 1.5 2001/07/31 05:58:11 verkerke Exp $
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
#include "RooFitModels/RooGaussModel.hh"
#include "RooFitCore/RooMath.hh"

ClassImp(RooGaussModel) 
;


RooGaussModel::RooGaussModel(const char *name, const char *title, RooRealVar& x, 
			     RooAbsReal& _mean, RooAbsReal& _sigma, 
			     RooAbsReal& _meanSF, RooAbsReal& _sigmaSF) : 
  RooResolutionModel(name,title,x), 
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma),
  msf("msf","Mean Scale Factor",this,_meanSF),
  ssf("ssf","Sigma Scale Factor",this,_sigmaSF)
{  
}


RooGaussModel::RooGaussModel(const RooGaussModel& other, const char* name) : 
  RooResolutionModel(other,name),
  mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma),
  msf("msf",this,other.msf),
  ssf("ssf",this,other.ssf)
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



Double_t RooGaussModel::evaluate(const RooArgSet* nset) const 
{  
  // *** 1st form: Straight Gaussian, used for unconvoluted PDF or expBasis with 0 lifetime ***
  static Double_t root2(sqrt(2)) ;
  static Double_t root2pi(sqrt(2*atan2(0,-1))) ;

  Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;

  if (_basisCode==noBasis || 
      ((_basisCode==expBasisPlus||_basisCode==expBasisMinus||
	_basisCode==cosBasisPlus||_basisCode==cosBasisMinus)&&tau==0.)) {
    Double_t xprime = (x-(mean*msf))/(sigma*ssf) ;
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 1st form" << endl ;
    return exp(-0.5*xprime*xprime)/(sigma*ssf*root2pi) ;
  }

  // *** 2nd form: 0, used for sinBasis and cosBasis with tau=0 ***
  if (tau==0) {
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 2nd form" << endl ;
    return 0. ;
  }

  // *** 3nd form: Convolution with exp(-t/tau), used for expBasis and cosBasis(omega=0) ***
  Double_t sign = (_basisCode==expBasisPlus||_basisCode==sinBasisPlus||_basisCode==cosBasisPlus)?-1:1 ;
  Double_t omega = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
  Double_t xprime = sign*(x-(mean*msf))/tau ;
  Double_t c = (sigma*ssf)/(root2*tau) ; 
  Double_t u = xprime/(2*c) ;
    
  if ( _basisCode==expBasisPlus || _basisCode==expBasisMinus || 
    ((_basisCode==cosBasisPlus||_basisCode==cosBasisMinus)&&omega==0.)) {      
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 3d form tau=" << tau << endl ;
    return exp(xprime+c*c) * erfc(u+c) ;
  }
  
  // *** 4th form: Convolution with exp(-t/tau)*sin(omega*t), used for sinBasis(omega<>0,tau<>0) ***
  Double_t swt = sign * omega *tau ;
  if (_basisCode==sinBasisPlus||_basisCode==sinBasisMinus) {
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 4th form" << endl ;
    return (swt==0.) ? 0. : evalCerfIm(swt,u,c) ;    
  }

  // *** 5th form: Convolution with exp(-t/tau)*cos(omega*t), used for cosBasis(omega<>0) ***
  if (_basisCode==cosBasisPlus||_basisCode==cosBasisMinus) {
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() 
			     << ") 5th form omega = " << omega << ", tau = " << tau << endl ;
    return evalCerfRe(swt,u,c) ;          
  }
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
  
  // *** 1st form: Straight Gaussian, used for unconvoluted PDF or expBasis with 0 lifetime ***
  Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
  if (_basisCode==noBasis || 
      ((_basisCode==expBasisPlus||_basisCode==expBasisMinus||
	_basisCode==cosBasisPlus||_basisCode==cosBasisMinus)&&tau==0.)) {
    Double_t xscale = root2*(sigma*ssf);
    return 0.5*(erf((x.max()-(mean*msf))/xscale)-erf((x.min()-(mean*msf))/xscale));
  }

  Double_t omega = ((RooAbsReal*)basis().getParameter(2))->getVal() ;

  // *** 2nd form: unity, used for sinBasis and cosBasis with tau=0 (PDF is zero) ***
  if (tau==0&&omega!=0) return 1. ;

  // *** 3rd form: Convolution with exp(-t/tau), used for expBasis and cosBasis(omega=0) ***
  Double_t sign = (_basisCode==expBasisPlus)?-1:1 ;
  Double_t c = (sigma*ssf)/(root2*tau) ; 
  Double_t xpmin = sign*(x.min()-(mean*msf))/tau ;
  Double_t xpmax = sign*(x.max()-(mean*msf))/tau ;
  Double_t umin = xpmin/(2*c) ;
  Double_t umax = xpmax/(2*c) ;
  if (_basisCode==expBasisPlus||_basisCode==expBasisMinus || 
    ((_basisCode==cosBasisPlus||_basisCode==cosBasisPlus)&&omega==0.)) {
    Double_t result = sign * tau * ( erf(umax) - erf(umin) + 
                                      exp(c*c) * ( exp(xpmax)*erfc(umax+c)
						 - exp(xpmin)*erfc(umin+c) )) ;     
    return result ;
  }

  // *** 4th form: Convolution with exp(-t/tau)*sin(omega*t), used for sinBasis(omega<>0,tau<>0) ***
  Double_t swt = omega * tau * sign ;
  RooComplex evalDif(evalCerf(swt,umax,c) - evalCerf(swt,umin,c)) ;
  if (_basisCode==sinBasisPlus||_basisCode==sinBasisMinus) {    
    Double_t result = (swt==0)? 1.0 
                    : (tau*sign/(1+swt*swt) * ( evalDif.im() - swt*evalDif.re() + erf(umax) - erf(umin) )) ;
    return result ;
  }

  // *** 5th form: Convolution with exp(-t/tau)*cos(omega*t), used for cosBasis(omega<>0) ***
  if (_basisCode==cosBasisPlus||_basisCode==cosBasisMinus) {
    Double_t result = tau*sign/(1+swt*swt) * ( evalDif.re() + swt*evalDif.im() + erf(umax) - erf(umin) ) ;
    return result ;
  }

}



RooComplex RooGaussModel::evalCerfApprox(Double_t swt, Double_t u, Double_t c) const
{
  // use the approximation: erf(z) = exp(-z*z)/(sqrt(pi)*z)
  // to explicitly cancel the divergent exp(y*y) behaviour of
  // CWERF for z = x + i y with large negative y

  static Double_t rootpi= sqrt(atan2(0,-1));
  RooComplex z(swt*c,u+c);  
  RooComplex zc(u+c,-swt*c);
  RooComplex zsq= z*z;
  RooComplex v= -zsq - u*u;

  return v.exp()*(-zsq.exp()/(zc*rootpi) + 1)*2 ;
}



