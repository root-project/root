/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Class RooGaussModel implements a RooResolutionModel that models a Gaussian
// distribution. Object of class RooGaussModel can be used
// for analytical convolutions with classes inheriting from RooAbsAnaConvPdf
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include "RooGaussModel.h"
#include "RooMath.h"
#include "RooRealConstant.h"
#include "RooRandom.h"

ClassImp(RooGaussModel) 
;



//_____________________________________________________________________________
RooGaussModel::RooGaussModel(const char *name, const char *title, RooRealVar& xIn, 
			     RooAbsReal& _mean, RooAbsReal& _sigma) :
  RooResolutionModel(name,title,xIn), 
  _flatSFInt(kFALSE),
  _asympInt(kFALSE),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma),
  msf("msf","Mean Scale Factor",this,(RooRealVar&)RooRealConstant::value(1)),
  ssf("ssf","Sigma Scale Factor",this,(RooRealVar&)RooRealConstant::value(1))
{  
}



//_____________________________________________________________________________
RooGaussModel::RooGaussModel(const char *name, const char *title, RooRealVar& xIn, 
			     RooAbsReal& _mean, RooAbsReal& _sigma, 
			     RooAbsReal& _msSF) : 
  RooResolutionModel(name,title,xIn), 
  _flatSFInt(kFALSE),
  _asympInt(kFALSE),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma),
  msf("msf","Mean Scale Factor",this,_msSF),
  ssf("ssf","Sigma Scale Factor",this,_msSF)
{  
}



//_____________________________________________________________________________
RooGaussModel::RooGaussModel(const char *name, const char *title, RooRealVar& xIn, 
			     RooAbsReal& _mean, RooAbsReal& _sigma, 
			     RooAbsReal& _meanSF, RooAbsReal& _sigmaSF) : 
  RooResolutionModel(name,title,xIn), 
  _flatSFInt(kFALSE),
  _asympInt(kFALSE),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma),
  msf("msf","Mean Scale Factor",this,_meanSF),
  ssf("ssf","Sigma Scale Factor",this,_sigmaSF)
{  
}



//_____________________________________________________________________________
RooGaussModel::RooGaussModel(const RooGaussModel& other, const char* name) : 
  RooResolutionModel(other,name),
  _flatSFInt(other._flatSFInt),
  _asympInt(other._asympInt),
  mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma),
  msf("msf",this,other.msf),
  ssf("ssf",this,other.ssf)
{
}



//_____________________________________________________________________________
RooGaussModel::~RooGaussModel()
{
  // Destructor
}



//_____________________________________________________________________________
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
  if (!TString("exp(-@0/@1)*cosh(@0*@2/2)").CompareTo(name)) return coshBasisPlus;
  if (!TString("exp(@0/@1)*cosh(@0*@2/2)").CompareTo(name)) return coshBasisMinus;
  if (!TString("exp(-abs(@0)/@1)*cosh(@0*@2/2)").CompareTo(name)) return coshBasisSum;
  if (!TString("exp(-@0/@1)*sinh(@0*@2/2)").CompareTo(name)) return sinhBasisPlus;
  if (!TString("exp(@0/@1)*sinh(@0*@2/2)").CompareTo(name)) return sinhBasisMinus;
  if (!TString("exp(-abs(@0)/@1)*sinh(@0*@2/2)").CompareTo(name)) return sinhBasisSum;
  return 0 ;
} 



//_____________________________________________________________________________
Double_t RooGaussModel::evaluate() const 
{  
  //cout << "RooGaussModel::evaluate(" << GetName() << ") basisCode = " << _basisCode << endl ;
  
  // *** 1st form: Straight Gaussian, used for unconvoluted PDF or expBasis with 0 lifetime ***
  static Double_t root2(sqrt(2.)) ;
  static Double_t root2pi(sqrt(2*atan2(0.,-1.))) ;
  static Double_t rootpi(sqrt(atan2(0.,-1.))) ;

  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;

  Double_t tau = (_basisCode!=noBasis)?((RooAbsReal*)basis().getParameter(1))->getVal():0 ;
  if (basisType == coshBasis && _basisCode!=noBasis ) {
     Double_t dGamma = ((RooAbsReal*)basis().getParameter(2))->getVal();
     if (dGamma==0) basisType = expBasis;
  }

  if (basisType==none || ((basisType==expBasis || basisType==cosBasis) && tau==0.)) {
    Double_t xprime = (x-(mean*msf))/(sigma*ssf) ;
    if (verboseEval()>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 1st form" << endl ;
    
    Double_t result = exp(-0.5*xprime*xprime)/(sigma*ssf*root2pi) ;
    if (_basisCode!=0 && basisSign==Both) result *= 2 ;
    //cout << "1st form " << "x= " << x << " result= " << result << endl;
    return result ;
  }

  // *** 2nd form: 0, used for sinBasis, linBasis, and quadBasis with tau=0 ***
  if (tau==0) {
    if (verboseEval()>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 2nd form" << endl ;
    return 0. ;
  }

  // *** 3nd form: Convolution with exp(-t/tau), used for expBasis and cosBasis(omega=0) ***
  Double_t omega = (basisType==sinBasis || basisType==cosBasis)
    ? ((RooAbsReal*)basis().getParameter(2))->getVal() : 0 ;
  Double_t xprime = (x-(mean*msf))/tau ;
  Double_t c = (sigma*ssf)/(root2*tau) ; 
  Double_t u = xprime/(2*c) ;

  if (basisType==expBasis || (basisType==cosBasis && omega==0.)) {
    if (verboseEval()>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 3d form tau=" << tau << endl ;
    Double_t result(0) ;
    if (basisSign!=Minus) result += exp(-xprime+c*c) * RooMath::erfc(-u+c) ;
    if (basisSign!=Plus)  result += exp(xprime+c*c) * RooMath::erfc(u+c) ;
    // equivalent form, added FMV, 07/24/03
    //if (basisSign!=Minus) result += evalCerfRe(-u,c) ; 
    //if (basisSign!=Plus)  result += evalCerfRe( u,c) ; 
    //cout << "3rd form " << "x= " << x << " result= " << result << endl;
    return result ;
  }
  
  // *** 4th form: Convolution with exp(-t/tau)*sin(omega*t), used for sinBasis(omega<>0,tau<>0) ***
  Double_t wt = omega *tau ;
  if (basisType==sinBasis) {
    if (verboseEval()>2) cout << "RooGaussModel::evaluate(" << GetName() << ") 4th form omega = " 
			     << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (wt==0.) return result ;
    if (basisSign!=Minus) result += -1*evalCerfIm(-wt,-u,c) ; 
    if (basisSign!=Plus) result += -1*evalCerfIm(wt,u,c) ; 
    //cout << "4th form " << "x= " << x << " result= " << result << endl;
    return result ;
  }

  // *** 5th form: Convolution with exp(-t/tau)*cos(omega*t), used for cosBasis(omega<>0) ***
  if (basisType==cosBasis) {
    if (verboseEval()>2) cout << "RooGaussModel::evaluate(" << GetName() 
			     << ") 5th form omega = " << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (basisSign!=Minus) result += evalCerfRe(-wt,-u,c) ; 
    if (basisSign!=Plus) result += evalCerfRe(wt,u,c) ; 
    //cout << "5th form " << "x= " << x << " result= " << result << endl;
    return result ;  
  }

  // *** 6th form: Convolution with (t/tau)*exp(-t/tau), used for linBasis ***
  if (basisType==linBasis) {
    if (verboseEval()>2) cout << "RooGaussModel::evaluate(" << GetName() 
			     << ") 6th form tau = " << tau << endl ;

    assert(basisSign==Plus);  // This should only be for positive times

    Double_t f0 = exp(-xprime+c*c) * RooMath::erfc(-u+c);
    Double_t f1 = exp(-u*u);

    return (xprime - 2*c*c)*f0 + (2*c/rootpi)*f1 ; 
  }

  // *** 7th form: Convolution with (t/tau)^2*exp(-t/tau), used for quadBasis ***
  if (basisType==quadBasis) {
    if (verboseEval()>2) cout << "RooGaussModel::evaluate(" << GetName() 
			     << ") 7th form tau = " << tau << endl ;

    assert(basisSign==Plus);  // This should only be for positive times

    Double_t f0 = exp(-xprime+c*c) * RooMath::erfc(-u+c);
    Double_t f1 = exp(-u*u);

    Double_t x2c2 = xprime - 2*c*c; 

    return ( x2c2*x2c2*f0 + (2*c/rootpi)*x2c2*f1 + 2*c*c*f0 );
  }

  // ***8th form: Convolution with exp(-|t|/tau)*cosh(dgamma*t/2), used for         coshBasisSum ***
  if (basisType==coshBasis) {
    if (verboseEval()>2) cout << "RooGaussModel::evaluate(" << GetName() 
			     << ") 8th form tau = " << tau << endl ;

    Double_t dgamma = ((RooAbsReal*)basis().getParameter(2))->getVal();
    Double_t tau1 = 1/(1/tau-dgamma/2) ; 
    Double_t tau2 = 1/(1/tau+dgamma/2) ;
    Double_t xprime1 = (x-(mean*msf))/tau1 ;
    Double_t c1 = (sigma*ssf)/(root2*tau1) ; 
    Double_t u1 = xprime1/(2*c1) ;
    Double_t xprime2 = (x-(mean*msf))/tau2 ;
    Double_t c2 = (sigma*ssf)/(root2*tau2) ; 
    Double_t u2 = xprime2/(2*c2) ;
    //Double_t c12 = c1*c1;
    //Double_t c22 = c2*c2;

    Double_t result(0);   
    //if (basisSign!=Minus) result += 0.5*(exp(-xprime1+c12) * RooMath::erfc(-u1+c1)+exp(-xprime2+c22) * RooMath::erfc(-u2+c2)) ;
    //if (basisSign!=Plus)  result += 0.5*(exp(xprime1+c12) * RooMath::erfc(u1+c1)+exp(xprime2+c22) * RooMath::erfc(u2+c2)) ;
    // equivalent form, added FMV, 07/24/03
    if (basisSign!=Minus) result += 0.5*(evalCerfRe(-u1,c1)+evalCerfRe(-u2,c2)) ; 
    if (basisSign!=Plus)  result += 0.5*(evalCerfRe( u1,c1)+evalCerfRe( u2,c2)) ; 
    //cout << "8th form " << "x= " << x << " result= " << result << endl;
    return result ;
  }

  // *** 9th form: Convolution with exp(-|t|/tau)*sinh(dgamma*t/2), used for        sinhBasisSum ***
  if (basisType==sinhBasis) {
    if (verboseEval()>2) cout << "RooGaussModel::evaluate(" << GetName() 
			     << ") 9th form tau = " << tau << endl ;

    Double_t dgamma = ((RooAbsReal*)basis().getParameter(2))->getVal();
    Double_t tau1 = 1/(1/tau-dgamma/2) ; 
    Double_t tau2 = 1/(1/tau+dgamma/2) ;
    Double_t xprime1 = (x-(mean*msf))/tau1 ;
    Double_t c1 = (sigma*ssf)/(root2*tau1) ; 
    Double_t u1 = xprime1/(2*c1) ;
    Double_t xprime2 = (x-(mean*msf))/tau2 ;
    Double_t c2 = (sigma*ssf)/(root2*tau2) ; 
    Double_t u2 = xprime2/(2*c2) ;
    //Double_t c12 = c1*c1;
    //Double_t c22 = c2*c2;

    Double_t result(0);   
    //if (basisSign!=Minus) result += 0.5*(exp(-xprime1+c12) * RooMath::erfc(-u1+c1)-exp(-xprime2+c22) * RooMath::erfc(-u2+c2)) ;
    //if (basisSign!=Plus)  result += 0.5*(-exp(xprime1+c12) * RooMath::erfc(u1+c1)+exp(xprime2+c22) * RooMath::erfc(u2+c2)) ;
    // equivalent form, added FMV, 07/24/03
    if (basisSign!=Minus) result += 0.5*(evalCerfRe(-u1,c1)-evalCerfRe(-u2,c2)) ; 
    if (basisSign!=Plus)  result += 0.5*(evalCerfRe( u2,c2)-evalCerfRe( u1,c1)) ; 
    //cout << "9th form " << "x= " << x << " result= " << result << endl;
    return result ;
  }


  assert(0) ;
  return 0 ;
}



//_____________________________________________________________________________
Int_t RooGaussModel::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  switch(_basisCode) {    

  // Analytical integration capability of raw PDF
  case noBasis:
 
    // Optionally advertise flat integral over sigma scale factor
    if (_flatSFInt) {
      if (matchArgs(allVars,analVars,RooArgSet(convVar(),ssf.arg()))) {
	return 2 ;
      }
    }

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
  case coshBasisMinus:
  case coshBasisPlus:
  case coshBasisSum:
  case sinhBasisMinus:
  case sinhBasisPlus:
  case sinhBasisSum:

    // Optionally advertise flat integral over sigma scale factor
    if (_flatSFInt) {

      if (matchArgs(allVars,analVars,RooArgSet(convVar(),ssf.arg()))) {
	return 2 ;
      }
    }
    
    if (matchArgs(allVars,analVars,convVar())) return 1 ;
    break ;
  }
  
  return 0 ;
}



//_____________________________________________________________________________
Double_t RooGaussModel::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  static Double_t root2 = sqrt(2.) ;
  //static Double_t rootPiBy2 = sqrt(atan2(0.0,-1.0)/2.0);
  static Double_t rootpi = sqrt(atan2(0.0,-1.0));
  Double_t ssfInt(1.0) ;

  // Code must be 1 or 2
  assert(code==1||code==2) ;
  if (code==2) {
    ssfInt = (ssf.max(rangeName)-ssf.min(rangeName)) ;
  }

  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;

  // *** 1st form: Straight Gaussian, used for unconvoluted PDF or expBasis with 0 lifetime ***
  Double_t tau = (_basisCode!=noBasis)?((RooAbsReal*)basis().getParameter(1))->getVal():0 ;
  if (basisType == coshBasis && _basisCode!=noBasis ) {
     Double_t dGamma = ((RooAbsReal*)basis().getParameter(2))->getVal();
     if (dGamma==0) basisType = expBasis;
  }
  if (basisType==none || ((basisType==expBasis || basisType==cosBasis) && tau==0.)) {
    Double_t xscale = root2*(sigma*ssf);
    if (verboseEval()>0) cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 1st form" << endl ;
    
    Double_t xpmin = (x.min(rangeName)-(mean*msf))/xscale ;
    Double_t xpmax = (x.max(rangeName)-(mean*msf))/xscale ;
 
    Double_t result ;
    if (_asympInt) { // modified FMV, 07/24/03
      result = 1.0 ;
    } else {
      if (xpmin<-6 && xpmax>6) {
	// If integral is over >6 sigma, approximate with full integral
	result = 1.0 ;
      } else {
	result = 0.5*(RooMath::erf(xpmax)-RooMath::erf(xpmin)) ;
      }
    }

    if (_basisCode!=0 && basisSign==Both) result *= 2 ;    
    //cout << "Integral 1st form " << " result= " << result*ssfInt << endl;
    return result*ssfInt ;
  }


  Double_t omega = ((basisType==sinBasis)||(basisType==cosBasis)) ?
    ((RooAbsReal*)basis().getParameter(2))->getVal() : 0 ;

  // *** 2nd form: unity, used for sinBasis and linBasis with tau=0 (PDF is zero) ***
  if (tau==0) {
    if (verboseEval()>0) cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 2nd form" << endl ;
    return 0. ;
  }

  

  // *** 3rd form: Convolution with exp(-t/tau), used for expBasis and cosBasis(omega=0) ***
  Double_t c = (sigma*ssf)/(root2*tau) ; 
  Double_t xpmin = (x.min(rangeName)-(mean*msf))/tau ;
  Double_t xpmax = (x.max(rangeName)-(mean*msf))/tau ;
  Double_t umin = xpmin/(2*c) ;
  Double_t umax = xpmax/(2*c) ;

  if (basisType==expBasis || (basisType==cosBasis && omega==0.)) {
    if (verboseEval()>0) cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 3d form tau=" << tau << endl ;

    Double_t result(0) ;
    if (_asympInt) {   // modified FMV, 07/24/03
      if (basisSign!=Minus) result += 2 * tau ;
      if (basisSign!=Plus)  result += 2 * tau ;      
    } else {
      if (basisSign!=Minus) result += -1 * tau * ( RooMath::erf(-umax) - RooMath::erf(-umin) + 
						   exp(c*c) * ( exp(-xpmax)*RooMath::erfc(-umax+c)
								- exp(-xpmin)*RooMath::erfc(-umin+c) )) ;
      if (basisSign!=Plus)  result +=      tau * ( RooMath::erf(umax) - RooMath::erf(umin) + 
						   exp(c*c) * ( exp(xpmax)*RooMath::erfc(umax+c)
								- exp(xpmin)*RooMath::erfc(umin+c) )) ;     
      // equivalent form, added FMV, 07/24/03
      //if (basisSign!=Minus) result += evalCerfInt(+1,tau,-umin,-umax,c).re();   
      //if (basisSign!=Plus) result += evalCerfInt(-1,tau,umin,umax,c).re();
    }
    //cout << "Integral 3rd form " << " result= " << result*ssfInt << endl;
    return result*ssfInt ;
  }

  // *** 4th form: Convolution with exp(-t/tau)*sin(omega*t), used for sinBasis(omega<>0,tau<>0) ***
  Double_t wt = omega * tau ;
    
  if (basisType==sinBasis) {    
    if (verboseEval()>0) cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 4th form omega = " 
			     << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (wt==0) return result*ssfInt ;
    if (basisSign!=Minus) {
      RooComplex evalDif(evalCerf(-wt,-umax,c) - evalCerf(-wt,-umin,c)) ;
      //result += -tau/(1+wt*wt) * ( -evalDif.im() +   -wt*evalDif.re() -   -wt*(RooMath::erf(-umax) - RooMath::erf(-umin)) ) ; 
      // FMV, fixed wrong sign, 07/24/03
      result += -tau/(1+wt*wt) * ( -evalDif.im() +   wt*evalDif.re() -   -wt*(RooMath::erf(-umax) - RooMath::erf(-umin)) ) ; 
    }
    if (basisSign!=Plus) {
      RooComplex evalDif(evalCerf(wt,umax,c) - evalCerf(wt,umin,c)) ;
      //result +=  tau/(1+wt*wt) * ( -evalDif.im() +    wt*evalDif.re() -    wt*(RooMath::erf(umax) - RooMath::erf(umin)) ) ;
      // FMV, fixed wrong sign, 07/24/03
      result +=  tau/(1+wt*wt) * ( -evalDif.im() +   -wt*evalDif.re() -    wt*(RooMath::erf(umax) - RooMath::erf(umin)) ) ;
    }
    // equivalent form, added FMV, 07/24/03
    //if (basisSign!=Minus) result += -1*evalCerfInt(+1,-wt,tau,-umin,-umax,c).im();
    //if (basisSign!=Plus) result += -1*evalCerfInt(-1,wt,tau,umin,umax,c).im();      

    //cout << "Integral 4th form " << " result= " << result*ssfInt << endl;
    return result*ssfInt ;
  }

  // *** 5th form: Convolution with exp(-t/tau)*cos(omega*t), used for cosBasis(omega<>0) ***
  if (basisType==cosBasis) {
    if (verboseEval()>0) cout << "RooGaussModel::analyticalIntegral(" << GetName() 
			     << ") 5th form omega = " << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (basisSign!=Minus) {
      RooComplex evalDif(evalCerf(-wt,-umax,c) - evalCerf(-wt,-umin,c)) ;
      //result += -tau/(1+wt*wt) * ( evalDif.re() + -wt*evalDif.im() + RooMath::erf(-umax) - RooMath::erf(-umin) ) ;
      // FMV, fixed wrong sign, 07/24/03
      result += -tau/(1+wt*wt) * ( evalDif.re() + wt*evalDif.im() + RooMath::erf(-umax) - RooMath::erf(-umin) ) ;
    }
    if (basisSign!=Plus) {
      RooComplex evalDif(evalCerf(wt,umax,c) - evalCerf(wt,umin,c)) ;
      //result +=  tau/(1+wt*wt) * ( evalDif.re() +  wt*evalDif.im() + RooMath::erf(umax) - RooMath::erf(umin) ) ;
      // FMV, fixed wrong sign, 07/24/03
      result +=  tau/(1+wt*wt) * ( evalDif.re() + -wt*evalDif.im() + RooMath::erf(umax) - RooMath::erf(umin) ) ;
    }
    // equivalent form, added FMV, 07/24/03
    //if (basisSign!=Minus) result += evalCerfInt(+1,-wt,tau,-umin,-umax,c).re();
    //if (basisSign!=Plus) result += evalCerfInt(-1,wt,tau,umin,umax,c).re();      

    //cout << "Integral 5th form " << " result= " << result*ssfInt << endl;
    return result*ssfInt ;
  }

  // *** 6th form: Convolution with (t/tau)*exp(-t/tau), used for linBasis ***
  if (basisType==linBasis) {
    if (verboseEval()>0) cout << "RooGaussModel::analyticalIntegral(" << GetName()
			     << ") 6th form tau=" << tau << endl ;

    Double_t f0 = RooMath::erf(-umax) - RooMath::erf(-umin);
    Double_t f1 = exp(-umax*umax) - exp(-umin*umin);

    Double_t tmp1 = exp(-xpmax)*RooMath::erfc(-umax + c);
    Double_t tmp2 = exp(-xpmin)*RooMath::erfc(-umin + c);

    Double_t f2 = tmp1 - tmp2;
    Double_t f3 = xpmax*tmp1 - xpmin*tmp2;

    Double_t expc2 = exp(c*c);

    return -tau*(              f0 +
		  (2*c/rootpi)*f1 +
	     (1 - 2*c*c)*expc2*f2 +
			 expc2*f3
		)*ssfInt;
  }

  // *** 7th form: Convolution with (t/tau)*(t/tau)*exp(-t/tau), used for quadBasis ***
  if (basisType==quadBasis) {
    if (verboseEval()>0) cout << "RooGaussModel::analyticalIntegral(" << GetName()
			     << ") 7th form tau=" << tau << endl ;

    Double_t f0 = RooMath::erf(-umax) - RooMath::erf(-umin);

    Double_t tmpA1 = exp(-umax*umax);
    Double_t tmpA2 = exp(-umin*umin);

    Double_t f1 = tmpA1 - tmpA2;
    Double_t f2 = umax*tmpA1 - umin*tmpA2;

    Double_t tmpB1 = exp(-xpmax)*RooMath::erfc(-umax + c);
    Double_t tmpB2 = exp(-xpmin)*RooMath::erfc(-umin + c);

    Double_t f3 = tmpB1 - tmpB2;
    Double_t f4 = xpmax*tmpB1 - xpmin*tmpB2;
    Double_t f5 = xpmax*xpmax*tmpB1 - xpmin*xpmin*tmpB2;

    Double_t expc2 = exp(c*c);

    return -tau*( 2*f0 +
		  (4*c/rootpi)*((1-c*c)*f1 + c*f2) +
		  (2*c*c*(2*c*c-1) + 2)*expc2*f3 - (4*c*c-2)*expc2*f4 + expc2*f5
                )*ssfInt;
  }

  // *** 8th form: Convolution with exp(-|t|/tau)*cosh(dgamma*t/2), used for coshBasis ***
  if (basisType==coshBasis) {
    if (verboseEval()>0) {cout << "RooGaussModel::analyticalIntegral(" << GetName()                             << ") 8th form tau=" << tau << endl ; }
    
    Double_t dgamma = ((RooAbsReal*)basis().getParameter(2))->getVal();
    Double_t tau1 = 1/(1/tau-dgamma/2) ; 
    Double_t tau2 = 1/(1/tau+dgamma/2) ;
    Double_t c1 = (sigma*ssf)/(root2*tau1) ; 
    Double_t xpmin1 = (x.min(rangeName)-(mean*msf))/tau1 ;
    Double_t xpmax1 = (x.max(rangeName)-(mean*msf))/tau1 ;
    Double_t umin1 = xpmin1/(2*c1) ;
    Double_t umax1 = xpmax1/(2*c1) ;
    Double_t c2 = (sigma*ssf)/(root2*tau2) ; 
    Double_t xpmin2 = (x.min(rangeName)-(mean*msf))/tau2 ;
    Double_t xpmax2 = (x.max(rangeName)-(mean*msf))/tau2 ;
    Double_t umin2 = xpmin2/(2*c2) ;
    Double_t umax2 = xpmax2/(2*c2) ;
    //Double_t c12 = c1*c1;
    //Double_t c22 = c2*c2;
    //Double_t ec12 = exp(c12);
    //Double_t ec22 = exp(c22);
    
    Double_t result(0) ;
    
    /*
    if (basisSign!=Minus) result += -0.5*(tau1 * ( RooMath::erf(-umax1) - RooMath::erf(-umin1) + 
						   ec12 * ( exp(-xpmax1)*RooMath::erfc(-umax1+c1)
								- exp(-xpmin1)*RooMath::erfc(-umin1+c1) )) +   
                                                   tau2 * ( RooMath::erf(-umax2) - RooMath::erf(-umin2) + 
						   ec22 * ( exp(-xpmax2)*RooMath::erfc(-umax2+c2)
								- exp(-xpmin2)*RooMath::erfc(-umin2+c2) ))) ;
      if (basisSign!=Plus)  result +=  0.5*(tau1 * ( RooMath::erf(umax1) - RooMath::erf(umin1) + 
						   ec12 * ( exp(xpmax1)*RooMath::erfc(umax1+c1)
								- exp(xpmin1)*RooMath::erfc(umin1+c1) ))+ 
			                           tau2 * ( RooMath::erf(umax2) - RooMath::erf(umin2) + 
						   ec22 * ( exp(xpmax2)*RooMath::erfc(umax2+c2)
								- exp(xpmin2)*RooMath::erfc(umin2+c2) ))) ;  
    */
    // equivalent form, added FMV, 07/24/03
    if (basisSign!=Minus) result += 0.5*(evalCerfInt(+1,tau1,-umin1,-umax1,c1)+
					 evalCerfInt(+1,tau2,-umin2,-umax2,c2));
    if (basisSign!=Plus)  result += 0.5*(evalCerfInt(-1,tau1,umin1,umax1,c1)+
					 evalCerfInt(-1,tau2,umin2,umax2,c2));
    
    //cout << "Integral 8th form " << " result= " << result*ssfInt << endl;
    return result*ssfInt ;
  }
   
  // *** 9th form: Convolution with exp(-|t|/tau)*sinh(dgamma*t/2), used for sinhBasis ***
  if (basisType==sinhBasis) {
    if (verboseEval()>0) cout << "RooGaussModel::analyticalIntegral(" << GetName()                             << ") 9th form tau=" << tau << endl ; 
    
    Double_t dgamma = ((RooAbsReal*)basis().getParameter(2))->getVal();
    Double_t tau1 = 1/(1/tau-dgamma/2) ; 
    Double_t tau2 = 1/(1/tau+dgamma/2) ;
    Double_t c1 = (sigma*ssf)/(root2*tau1) ; 
    Double_t xpmin1 = (x.min(rangeName)-(mean*msf))/tau1 ;
    Double_t xpmax1 = (x.max(rangeName)-(mean*msf))/tau1 ;
    Double_t umin1 = xpmin1/(2*c1) ;
    Double_t umax1 = xpmax1/(2*c1) ;
    Double_t c2 = (sigma*ssf)/(root2*tau2) ; 
    Double_t xpmin2 = (x.min(rangeName)-(mean*msf))/tau2 ;
    Double_t xpmax2 = (x.max(rangeName)-(mean*msf))/tau2 ;
    Double_t umin2 = xpmin2/(2*c2) ;
    Double_t umax2 = xpmax2/(2*c2) ;
    //Double_t c12 = c1*c1;
    //Double_t c22 = c2*c2;
    //Double_t ec12 = exp(c12);
    //Double_t ec22 = exp(c22);
 
    Double_t result(0) ;

    /*
    if (basisSign!=Minus) result += 0.5*(-tau1 * ( RooMath::erf(-umax1) - RooMath::erf(-umin1) + 
						   ec12 * ( exp(-xpmax1)*RooMath::erfc(-umax1+c1)
							    - exp(-xpmin1)*RooMath::erfc(-umin1+c1) )) +   
                                                   tau2 * ( RooMath::erf(-umax2) - RooMath::erf(-umin2) + 
							    ec22 * ( exp(-xpmax2)*RooMath::erfc(-umax2+c2)
								     - exp(-xpmin2)*RooMath::erfc(-umin2+c2) ))) ;
    if (basisSign!=Plus)  result += 0.5*(-tau1 * ( RooMath::erf(umax1) - RooMath::erf(umin1) + 
						   ec12 * ( exp(xpmax1)*RooMath::erfc(umax1+c1)
							    - exp(xpmin1)*RooMath::erfc(umin1+c1) ))+ 
					 tau2 * ( RooMath::erf(umax2) - RooMath::erf(umin2) + 
						  ec22 * ( exp(xpmax2)*RooMath::erfc(umax2+c2)
							   - exp(xpmin2)*RooMath::erfc(umin2+c2) ))) ; 
    */
    // equivalent form, added FMV, 07/24/03
    if (basisSign!=Minus) result += 0.5*(evalCerfInt(+1,tau1,-umin1,-umax1,c1)-
    					 evalCerfInt(+1,tau2,-umin2,-umax2,c2));
    if (basisSign!=Plus)  result += 0.5*(evalCerfInt(-1,tau2,umin2,umax2,c2)-
					 evalCerfInt(-1,tau1,umin1,umax1,c1));

    //cout << "Integral 9th form " << " result= " << result*ssfInt << endl;
    return result*ssfInt ;
    
  }
  assert(0) ;
  return 0 ;
}



//_____________________________________________________________________________
RooComplex RooGaussModel::evalCerfApprox(Double_t swt, Double_t u, Double_t c) const
{
  // use the approximation: erf(z) = exp(-z*z)/(sqrt(pi)*z)
  // to explicitly cancel the divergent exp(y*y) behaviour of
  // CWERF for z = x + i y with large negative y

  static Double_t rootpi= sqrt(atan2(0.,-1.));
  RooComplex z(swt*c,u+c);  
  RooComplex zc(u+c,-swt*c);
  RooComplex zsq= z*z;
  RooComplex v= -zsq - u*u;

  return v.exp()*(-zsq.exp()/(zc*rootpi) + 1)*2 ;
}



// added FMV, 07/24/03
//_____________________________________________________________________________
RooComplex RooGaussModel::evalCerfInt(Double_t sign, Double_t wt, Double_t tau, Double_t umin, Double_t umax, Double_t c) const
{
  RooComplex diff;
  if (_asympInt) {
    diff = RooComplex(2,0) ;
  } else {
    diff = RooComplex(sign,0.)*(evalCerf(wt,umin,c) - evalCerf(wt,umax,c) + RooMath::erf(umin) - RooMath::erf(umax));
  }
  return RooComplex(tau/(1.+wt*wt),0)*RooComplex(1,wt)*diff;
}
// added FMV, 08/17/03

//_____________________________________________________________________________
Double_t RooGaussModel::evalCerfInt(Double_t sign, Double_t tau, Double_t umin, Double_t umax, Double_t c) const
{
  Double_t diff;
  if (_asympInt) {
    diff = 2. ;
  } else {
    if ((umin<-8 && umax>8)||(umax<-8 && umin>8)) {
      // If integral is over >8 sigma, approximate with full integral
      diff = 2. ;
    } else {
      diff = sign*(evalCerfRe(umin,c) - evalCerfRe(umax,c) + RooMath::erf(umin) - RooMath::erf(umax));
    }
  }
  return tau*diff;
}



//_____________________________________________________________________________
Int_t RooGaussModel::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;  
  return 0 ;
}



//_____________________________________________________________________________
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




