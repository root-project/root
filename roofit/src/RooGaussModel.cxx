/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGaussModel.cc,v 1.19 2002/03/27 08:07:05 mwilson Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// 

#include <iostream.h>
#include "RooFitModels/RooGaussModel.hh"
#include "RooFitCore/RooMath.hh"
#include "RooFitCore/RooRealConstant.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooGaussModel) 
;


RooGaussModel::RooGaussModel(const char *name, const char *title, RooRealVar& x, 
			     RooAbsReal& _mean, RooAbsReal& _sigma) :
  RooResolutionModel(name,title,x), 
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma),
  msf("msf","Mean Scale Factor",this,(RooRealVar&)RooRealConstant::value(1)),
  ssf("ssf","Sigma Scale Factor",this,(RooRealVar&)RooRealConstant::value(1))
{  
}


RooGaussModel::RooGaussModel(const char *name, const char *title, RooRealVar& x, 
			     RooAbsReal& _mean, RooAbsReal& _sigma, 
			     RooAbsReal& _msSF) : 
  RooResolutionModel(name,title,x), 
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma),
  msf("msf","Mean Scale Factor",this,_msSF),
  ssf("ssf","Sigma Scale Factor",this,_msSF)
{  
}


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
  if (!TString("exp(-@0/@1)").CompareTo(name)) return expBasisPlus ;
  if (!TString("exp(@0/@1)").CompareTo(name)) return expBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)").CompareTo(name)) return expBasisSum ;
  if (!TString("exp(-@0/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisPlus ;
  if (!TString("exp(@0/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisSum ;
  if (!TString("exp(-@0/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisPlus ;
  if (!TString("exp(@0/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisSum ;
  if (!TString("(@0/@1)*exp(-@0/@1)").CompareTo(name)) return linBasisPlus ;
  if (!TString("(@0/@1)*(@0/@1)*exp(-@0/@1)").CompareTo(name)) return quadBasisPlus ;
  return 0 ;
} 



Double_t RooGaussModel::evaluate() const 
{  
  //cout << "RooGaussModel::evaluate(" << GetName() << ") basisCode = " << _basisCode << endl ;
  
  // *** 1st form: Straight Gaussian, used for unconvoluted PDF or expBasis with 0 lifetime ***
  static Double_t root2(sqrt(2)) ;
  static Double_t root2pi(sqrt(2*atan2(0,-1))) ;
  static Double_t rootpi(sqrt(atan2(0,-1))) ;

  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;

  Double_t tau = (_basisCode!=noBasis)?((RooAbsReal*)basis().getParameter(1))->getVal():0 ;

  if (basisType==none || ((basisType==expBasis || basisType==cosBasis) && tau==0.)) {
    Double_t xprime = (x-(mean*msf))/(sigma*ssf) ;
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 1st form" << endl ;
    
    Double_t result = exp(-0.5*xprime*xprime)/(sigma*ssf*root2pi) ;
    if (_basisCode!=0 && basisSign==Both) result *= 2 ;
    return result ;
  }

  // *** 2nd form: 0, used for sinBasis, linBasis, and quadBasis with tau=0 ***
  if (tau==0) {
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 2nd form" << endl ;
    return 0. ;
  }

  // *** 3nd form: Convolution with exp(-t/tau), used for expBasis and cosBasis(omega=0) ***
  Double_t omega = (basisType==sinBasis || basisType==cosBasis)
    ? ((RooAbsReal*)basis().getParameter(2))->getVal() : 0 ;
  Double_t xprime = (x-(mean*msf))/tau ;
  Double_t c = (sigma*ssf)/(root2*tau) ; 
  Double_t u = xprime/(2*c) ;

  if (basisType==expBasis || (basisType==cosBasis && omega==0.)) {
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 3d form tau=" << tau << endl ;
    Double_t result(0) ;
    if (basisSign!=Minus) result += exp(-xprime+c*c) * erfc(-u+c) ;
    if (basisSign!=Plus)  result += exp(xprime+c*c) * erfc(u+c) ;
    return result ;
  }
  
  // *** 4th form: Convolution with exp(-t/tau)*sin(omega*t), used for sinBasis(omega<>0,tau<>0) ***
  Double_t wt = omega *tau ;
  if (basisType==sinBasis) {
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 4th form omega = " 
			     << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (wt==0.) return result ;
    if (basisSign!=Minus) result += -1*evalCerfIm(-wt,-u,c) ; 
    if (basisSign!=Plus) result += -1*evalCerfIm(wt,u,c) ; 
    return result ;
  }

  // *** 5th form: Convolution with exp(-t/tau)*cos(omega*t), used for cosBasis(omega<>0) ***
  if (basisType==cosBasis) {
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() 
			     << ") 5th form omega = " << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (basisSign!=Minus) result += evalCerfRe(-wt,-u,c) ; 
    if (basisSign!=Plus) result += evalCerfRe(wt,u,c) ; 
    return result ;  
  }

  // *** 6th form: Convolution with (t/tau)*exp(-t/tau), used for linBasis ***
  if (basisType==linBasis) {
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() 
			     << ") 6th form tau = " << tau << endl ;

    assert(basisSign==Plus);  // This should only be for positive times

    Double_t f0 = exp(-xprime+c*c) * erfc(-u+c);
    Double_t f1 = exp(-u*u);

    return (xprime - 2*c*c)*f0 + (2*c/rootpi)*f1 ; 
  }

  // *** 7th form: Convolution with (t/tau)^2*exp(-t/tau), used for quadBasis ***
  if (basisType==quadBasis) {
    if (_verboseEval>2) cout << "RooGaussModel::evaluate(" << GetName() 
			     << ") 7th form tau = " << tau << endl ;

    assert(basisSign==Plus);  // This should only be for positive times

    Double_t f0 = exp(-xprime+c*c) * erfc(-u+c);
    Double_t f1 = exp(-u*u);

    Double_t x2c2 = xprime - 2*c*c; 

    return ( x2c2*x2c2*f0 + (2*c/rootpi)*x2c2*f1 + 2*c*c*f0 );
  }


  assert(0) ;
  return 0 ;
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
  case expBasisSum:
  case sinBasisPlus:
  case sinBasisMinus:
  case sinBasisSum:
  case cosBasisPlus:
  case cosBasisMinus:
  case cosBasisSum:
  case linBasisPlus:
  case quadBasisPlus:
    if (matchArgs(allVars,analVars,convVar())) return 1 ;
    break ;
  }
  
  return 0 ;
}



Double_t RooGaussModel::analyticalIntegral(Int_t code) const 
{
  static Double_t root2 = sqrt(2) ;
  static Double_t rootPiBy2 = sqrt(atan2(0.0,-1.0)/2.0);
  static Double_t rootpi = sqrt(atan2(0.0,-1.0));

  // Code must be 1
  assert(code==1) ;


  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;

  // *** 1st form: Straight Gaussian, used for unconvoluted PDF or expBasis with 0 lifetime ***
  Double_t tau = (_basisCode!=noBasis)?((RooAbsReal*)basis().getParameter(1))->getVal():0 ;

  if (basisType==none || ((basisType==expBasis || basisType==cosBasis) && tau==0.)) {
    Double_t xscale = root2*(sigma*ssf);
    if (_verboseEval>0) cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 1st form" << endl ;
    
    Double_t xpmin = (x.min()-(mean*msf))/xscale ;
    Double_t xpmax = (x.max()-(mean*msf))/xscale ;
 
    Double_t result ;
    if (xpmin<-6 && xpmax>6) {
      // If integral is over >6 sigma, approximate with full integral
      result = 1.0 ;
    } else {
      result = 0.5*(erf(xpmax)-erf(xpmin)) ;
    }

    if (_basisCode!=0 && basisSign==Both) result *= 2 ;    
    return result ;
  }


  Double_t omega = ((basisType==sinBasis)||(basisType==cosBasis)) ?
    ((RooAbsReal*)basis().getParameter(2))->getVal() : 0 ;

  // *** 2nd form: unity, used for sinBasis and linBasis with tau=0 (PDF is zero) ***
  if (tau==0) {
    if (_verboseEval>0) cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 2nd form" << endl ;
    return 0. ;
  }

  

  // *** 3rd form: Convolution with exp(-t/tau), used for expBasis and cosBasis(omega=0) ***
  Double_t c = (sigma*ssf)/(root2*tau) ; 
  Double_t xpmin = (x.min()-(mean*msf))/tau ;
  Double_t xpmax = (x.max()-(mean*msf))/tau ;
  Double_t umin = xpmin/(2*c) ;
  Double_t umax = xpmax/(2*c) ;

  if (basisType==expBasis || (basisType==cosBasis && omega==0.)) {
    if (_verboseEval>0) cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 3d form tau=" << tau << endl ;

    Double_t result(0) ;
    if (umin<-6 && umax>6) {
      // If integral is over >6 sigma, approximate with full integral
      if (basisSign!=Minus) result += 2 * tau ;
      if (basisSign!=Plus)  result += 2 * tau ;      
    } else {
      if (basisSign!=Minus) result += -1 * tau * ( erf(-umax) - erf(-umin) + 
						   exp(c*c) * ( exp(-xpmax)*erfc(-umax+c)
								- exp(-xpmin)*erfc(-umin+c) )) ;     
      if (basisSign!=Plus)  result +=      tau * ( erf(umax) - erf(umin) + 
						   exp(c*c) * ( exp(xpmax)*erfc(umax+c)
								- exp(xpmin)*erfc(umin+c) )) ;     
    }
    return result ;
  }

  // *** 4th form: Convolution with exp(-t/tau)*sin(omega*t), used for sinBasis(omega<>0,tau<>0) ***
  Double_t wt = omega * tau ;
    
  if (basisType==sinBasis) {    
    if (_verboseEval>0) cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 4th form omega = " 
			     << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (wt==0) return result ;
    if (basisSign!=Minus) {
      RooComplex evalDif(evalCerf(-wt,-umax,c) - evalCerf(-wt,-umin,c)) ;
      result += -tau/(1+wt*wt) * ( -evalDif.im() +   -wt*evalDif.re() -   -wt*(erf(-umax) - erf(-umin)) ) ; 
    }
    if (basisSign!=Plus) {
      RooComplex evalDif(evalCerf(wt,umax,c) - evalCerf(wt,umin,c)) ;
      result +=  tau/(1+wt*wt) * ( -evalDif.im() +    wt*evalDif.re() -    wt*(erf(umax) - erf(umin)) ) ;
    }

    return result ;
  }

  // *** 5th form: Convolution with exp(-t/tau)*cos(omega*t), used for cosBasis(omega<>0) ***
  if (basisType==cosBasis) {
    if (_verboseEval>0) cout << "RooGaussModel::analyticalIntegral(" << GetName() 
			     << ") 5th form omega = " << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;

    if (basisSign!=Minus) {
      RooComplex evalDif(evalCerf(-wt,-umax,c) - evalCerf(-wt,-umin,c)) ;
      result += -tau/(1+wt*wt) * ( evalDif.re() + -wt*evalDif.im() + erf(-umax) - erf(-umin) ) ;
    }
    if (basisSign!=Plus) {
      RooComplex evalDif(evalCerf(wt,umax,c) - evalCerf(wt,umin,c)) ;
      result +=  tau/(1+wt*wt) * ( evalDif.re() +  wt*evalDif.im() + erf(umax) - erf(umin) ) ;
    }

    return result ;
  }

  // *** 6th form: Convolution with (t/tau)*exp(-t/tau), used for linBasis ***
  if (basisType==linBasis) {
    if (_verboseEval>0) cout << "RooGaussModel::analyticalIntegral(" << GetName()
			     << ") 6th form tau=" << tau << endl ;

    Double_t f0 = erf(-umax) - erf(-umin);
    Double_t f1 = exp(-umax*umax) - exp(-umin*umin);

    Double_t tmp1 = exp(-xpmax)*erfc(-umax + c);
    Double_t tmp2 = exp(-xpmin)*erfc(-umin + c);

    Double_t f2 = tmp1 - tmp2;
    Double_t f3 = xpmax*tmp1 - xpmin*tmp2;

    Double_t expc2 = exp(c*c);

    return -tau*(              f0 +
		  (2*c/rootpi)*f1 +
	     (1 - 2*c*c)*expc2*f2 +
			 expc2*f3
		);
  }

  // *** 7th form: Convolution with (t/tau)*(t/tau)*exp(-t/tau), used for quadBasis ***
  if (basisType==quadBasis) {
    if (_verboseEval>0) cout << "RooGaussModel::analyticalIntegral(" << GetName()
			     << ") 7th form tau=" << tau << endl ;

    Double_t f0 = erf(-umax) - erf(-umin);

    Double_t tmpA1 = exp(-umax*umax);
    Double_t tmpA2 = exp(-umin*umin);

    Double_t f1 = tmpA1 - tmpA2;
    Double_t f2 = umax*tmpA1 - umin*tmpA2;

    Double_t tmpB1 = exp(-xpmax)*erfc(-umax + c);
    Double_t tmpB2 = exp(-xpmin)*erfc(-umin + c);

    Double_t f3 = tmpB1 - tmpB2;
    Double_t f4 = xpmax*tmpB1 - xpmin*tmpB2;
    Double_t f5 = xpmax*xpmax*tmpB1 - xpmin*xpmin*tmpB2;

    Double_t expc2 = exp(c*c);

    return -tau*( 2*f0 +
		  (4*c/rootpi)*((1-c*c)*f1 + c*f2) +
		  (2*c*c*(2*c*c-1) + 2)*expc2*f3 - (4*c*c-2)*expc2*f4 + expc2*f5
                );
  }

  assert(0) ;
  return 0 ;
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




Int_t RooGaussModel::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;  
  return 0 ;
}



void RooGaussModel::generateEvent(Int_t code)
{
  assert(code==1) ;
  Double_t xgen ;
  while(1) {    
    xgen = RooRandom::randomGenerator()->Gaus((mean*msf),(sigma*ssf));
    if (xgen<x.max() && xgen>x.min()) {
      x = xgen ;
      return ;
    }
  }
}

