/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGExpModel.cc,v 1.6 2001/11/19 07:26:41 verkerke Exp $
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
#include "RooFitModels/RooGExpModel.hh"
#include "RooFitCore/RooMath.hh"
#include "RooFitCore/RooRealConstant.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooGExpModel) 
;


RooGExpModel::RooGExpModel(const char *name, const char *title, RooRealVar& x, 
			   RooAbsReal& _sigma, RooAbsReal& _rlife, 
			   Bool_t nlo, Type type) : 
  RooResolutionModel(name,title,x), 
  sigma("sigma","Width",this,_sigma),
  rlife("rlife","Life time",this,_rlife),
  ssf("ssf","Sigma Scale Factor",this,(RooRealVar&)RooRealConstant::value(1)),
  rsf("rsf","RLife Scale Factor",this,(RooRealVar&)RooRealConstant::value(1)),
  _nlo(nlo), _flip(type==Flipped)
{  
}


RooGExpModel::RooGExpModel(const char *name, const char *title, RooRealVar& x, 
			   RooAbsReal& _sigma, RooAbsReal& _rlife, 
			   RooAbsReal& _rsSF,
			   Bool_t nlo, Type type) : 
  RooResolutionModel(name,title,x), 
  sigma("sigma","Width",this,_sigma),
  rlife("rlife","Life time",this,_rlife),
  ssf("ssf","Sigma Scale Factor",this,_rsSF),
  rsf("rsf","RLife Scale Factor",this,_rsSF),
  _nlo(nlo), _flip(type==Flipped)
{  
}


RooGExpModel::RooGExpModel(const char *name, const char *title, RooRealVar& x, 
			   RooAbsReal& _sigma, RooAbsReal& _rlife, 
			   RooAbsReal& _sigmaSF, RooAbsReal& _rlifeSF,
			   Bool_t nlo, Type type) : 
  RooResolutionModel(name,title,x), 
  sigma("sigma","Width",this,_sigma),
  rlife("rlife","Life time",this,_rlife),
  ssf("ssf","Sigma Scale Factor",this,_sigmaSF),
  rsf("rsf","RLife Scale Factor",this,_rlifeSF),
  _nlo(nlo), _flip(type==Flipped)
{  
}


RooGExpModel::RooGExpModel(const RooGExpModel& other, const char* name) : 
  RooResolutionModel(other,name),
  sigma("sigma",this,other.sigma),
  rlife("rlife",this,other.rlife),
  ssf("ssf",this,other.ssf),
  rsf("rsf",this,other.rsf),
  _nlo(other._nlo),
  _flip(other._flip)
{
}


RooGExpModel::~RooGExpModel()
{
  // Destructor
}



Int_t RooGExpModel::basisCode(const char* name) const 
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
  return 0 ;
} 



Double_t RooGExpModel::evaluate() const 
{  
  static Double_t root2(sqrt(2)) ;
  static Double_t root2pi(sqrt(2*atan2(0,-1))) ;
  static Double_t rootpi(sqrt(atan2(0,-1)));

  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;

  Double_t arg=x/sigma;
  Double_t fsign = _flip?-1:1 ;

  Double_t sig = sigma*ssf ;
  Double_t rtau = rlife*rsf ;
 
  Double_t tau = (_basisCode!=noBasis)?((RooAbsReal*)basis().getParameter(1))->getVal():0. ;

  // *** 1st form: Straight GExp, used for unconvoluted PDF or expBasis with 0 lifetime ***
  if (basisType==none || ((basisType==expBasis || basisType==cosBasis) && tau==0.)) {
    if (_verboseEval>2) cout << "RooGExpModel::evaluate(" << GetName() << ") 1st form" << endl ;    

    Double_t result = 1/(2*rtau)
                    * exp(sig*sig/(2*rtau*rtau) + fsign*x/rtau)
                    * erfc(sig/(root2*rtau) + fsign*x/(root2*sig));  
    if (_basisCode!=0 && basisSign==Both) result *= 2 ;
    return result ;    
  }
  
  // *** 2nd form: 0, used for sinBasis and cosBasis with tau=0 ***
  if (tau==0) {
    if (_verboseEval>2) cout << "RooGExpModel::evaluate(" << GetName() << ") 2nd form" << endl ;
    return 0. ;
  }

  Double_t omega = (basisType!=expBasis)?((RooAbsReal*)basis().getParameter(2))->getVal():0. ;

  // *** 3nd form: Convolution with exp(-t/tau), used for expBasis and cosBasis(omega=0) ***
  if (basisType==expBasis || (basisType==cosBasis && omega==0.)) {
    if (_verboseEval>2) cout << "RooGExpModel::evaluate(" << GetName() << ") 3d form tau=" << tau << endl ;
    Double_t result(0) ;
    if (basisSign!=Minus) result += calcDecayConv(-1,tau,sig,rtau) ;
    if (basisSign!=Plus)  result += calcDecayConv(+1,tau,sig,rtau) ;
    return result ;
  }
  
  // *** 4th form: Convolution with exp(-t/tau)*sin(omega*t), used for sinBasis(omega<>0,tau<>0) ***
  Double_t wt = omega *tau ;
  if (basisType==sinBasis) {
    if (_verboseEval>2) cout << "RooGExpModel::evaluate(" << GetName() << ") 4th form omega = " 
			     << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (wt==0.) return result ;
    if (basisSign!=Minus) result += -1*calcSinConv(-1,sig,tau,omega,rtau,fsign).im() ;
    if (basisSign!=Plus) result += -1*calcSinConv(+1,sig,tau,omega,rtau,fsign).im() ;
    return result ;
  }

  // *** 5th form: Convolution with exp(-t/tau)*cos(omega*t), used for cosBasis(omega<>0) ***
  if (basisType==cosBasis) {
    if (_verboseEval>2) cout << "RooGExpModel::evaluate(" << GetName() 
			     << ") 5th form omega = " << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (basisSign!=Minus) result += calcSinConv(-1,sig,tau,omega,rtau,fsign).re() ;
    if (basisSign!=Plus) result += calcSinConv(+1,sig,tau,omega,rtau,fsign).re() ;
    return result ;  
  }

  assert(0) ;
  return 0 ;
}




RooComplex RooGExpModel::calcSinConv(Double_t sign, Double_t sig, Double_t tau, Double_t omega, Double_t rtau, Double_t fsign) const
{
  static Double_t root2(sqrt(2)) ;

  Double_t s1= -sign*x/tau;
  Double_t c1= sig/(root2*tau);
  Double_t u1= s1/(2*c1);  
  Double_t s2= x/rtau;
  Double_t c2= sig/(root2*rtau);
  Double_t u2= fsign*s2/(2*c2) ;

  RooComplex eins(1,0);
  RooComplex k(1/tau,sign*omega);  
  return (evalCerf(-sign*omega*tau,u1,c1)+evalCerf(0,u2,c2)*fsign*sign) / (eins + k*fsign*sign*rtau) ;
}




Double_t RooGExpModel::calcDecayConv(Double_t sign, Double_t tau, Double_t sig, Double_t rtau) const
{
  static Double_t root2(sqrt(2)) ;
  static Double_t root2pi(sqrt(2*atan2(0,-1))) ;
  static Double_t rootpi(sqrt(atan2(0,-1)));

  // Process flip status
  Double_t xp(x) ;
  if (_flip) {
    xp   *= -1 ;
    sign *= -1 ;
  }
  
  Double_t cFly;
  if ((sign<0)&&(fabs(tau-rtau)<tau/260)) {
    
    Double_t MeanTau=0.5*(tau+rtau);
    cFly=1./(MeanTau*MeanTau*root2pi)
      *exp(-(-xp/MeanTau-sig*sig/(2*MeanTau*MeanTau)))
      *(sig*exp(-1/(2*sig*sig)*pow((sig*sig/MeanTau+xp),2))
	-(sig*sig/MeanTau+xp)*(rootpi/root2)
	*erfc(sig/(root2*MeanTau)+xp/(root2*sig)));
    
    if(_nlo) {
      Double_t epsilon=0.5*(tau-rtau);
      Double_t a=sig/(root2*MeanTau)+xp/(root2*sig);
      cFly += 1./(MeanTau*MeanTau)
	*exp(-(-xp/MeanTau-sig*sig/(2*MeanTau*MeanTau)))
	*0.5/MeanTau*epsilon*epsilon*
	(exp(-a*a)*(sig/MeanTau*root2/rootpi
		    -(4*a*sig*sig)/(2*rootpi*MeanTau*MeanTau)
		    +(-4/rootpi+8*a*a/rootpi)/6
		    *pow(sig/(root2*MeanTau),3)
		    +2/rootpi*(sig*sig/(MeanTau*MeanTau)+xp/MeanTau)*
		    (sig/(root2*MeanTau)-a*(sig*sig)/(2*MeanTau*MeanTau))
		    +2/rootpi*((3*sig*sig)/(2*MeanTau*MeanTau)+xp/MeanTau+
			       0.5*pow(sig*sig/(MeanTau*MeanTau)+xp/MeanTau,2))*sig/(root2*MeanTau))
	 -(2*sig*sig/(MeanTau*MeanTau)+xp/MeanTau+(sig*sig/(MeanTau*MeanTau)+xp/MeanTau)*
	   (3*sig*sig/(2*MeanTau*MeanTau)+xp/MeanTau)
	   +pow(sig*sig/(MeanTau*MeanTau)+xp/MeanTau,3)/6)*erfc(a));
    }
    
  } else {
    
    cFly=(exp(sig*sig/(2*tau*tau)-sign*xp/tau)
	  *erfc(sig/(root2*tau)-sign*xp/(root2*sig))
	  +sign*exp(sig*sig/(2*rtau*rtau)+xp/rtau)
	  *erfc(sig/(root2*rtau)+xp/(root2*sig)))/(2*(tau+sign*rtau));
  }
  
  return cFly*2*tau ;    
}



Int_t RooGExpModel::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
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
    if (matchArgs(allVars,analVars,convVar())) return 1 ;
    break ;
  }
  
  return 0 ;
}



Double_t RooGExpModel::analyticalIntegral(Int_t code) const 
{
  static Double_t root2 = sqrt(2) ;
  static Double_t rootPiBy2 = sqrt(atan2(0.0,-1.0)/2.0);

  // Code must be 1
  assert(code==1) ;

  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;
   
  Double_t tau = (_basisCode!=noBasis)?((RooAbsReal*)basis().getParameter(1))->getVal():0 ;

  // *** 1st form????
  if (basisType==none || ((basisType==expBasis || basisType==cosBasis) && tau==0.)) {
    if (_verboseEval>0) cout << "RooGExpModel::analyticalIntegral(" << GetName() << ") 1st form" << endl ;

    Double_t result = 1.0 ; // WVE inferred from limit(tau->0) of cosBasisNorm
    if (_basisCode!=0 && basisSign==Both) result *= 2 ;    
    return result ;    
  }

  Double_t omega = (basisType!=expBasis) ?((RooAbsReal*)basis().getParameter(2))->getVal() : 0 ;

  // *** 2nd form: unity, used for sinBasis and cosBasis with tau=0 (PDF is zero) ***
  if (tau==0&&omega!=0) {
    if (_verboseEval>0) cout << "RooGExpModel::analyticalIntegral(" << GetName() << ") 2nd form" << endl ;
    return 0. ;
  }

  // *** 3rd form: Convolution with exp(-t/tau), used for expBasis and cosBasis(omega=0) ***
  if (basisType==expBasis || (basisType==cosBasis && omega==0.)) {
    Double_t result = 2*tau ;
    if (basisSign==Both) result *= 2 ;
    return result ;
  }
  
  // *** 4th form: Convolution with exp(-t/tau)*sin(omega*t), used for sinBasis(omega<>0,tau<>0) ***
  Double_t wt = omega * tau ;    
  if (basisType==sinBasis) {    
    if (_verboseEval>0) cout << "RooGExpModel::analyticalIntegral(" << GetName() << ") 4th form omega = " 
			     << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (wt==0) return result ;
    if (basisSign!=Minus) result += calcSinConvNorm(-1,tau,omega).im() ;
    if (basisSign!=Plus) result += calcSinConvNorm(+1,tau,omega).im() ;
    return result ;
  }
 
  // *** 5th form: Convolution with exp(-t/tau)*cos(omega*t), used for cosBasis(omega<>0) ***
  if (basisType==cosBasis) {
    if (_verboseEval>0) cout << "RooGExpModel::analyticalIntegral(" << GetName() 
			     << ") 5th form omega = " << omega << ", tau = " << tau << endl ;
    Double_t result(0) ;
    if (basisSign!=Minus) result += calcSinConvNorm(-1,tau,omega).re() ;
    if (basisSign!=Plus) result += calcSinConvNorm(+1,tau,omega).re() ;
    return result ;
  }
  
  assert(0) ;
  return 1 ;
}



RooComplex RooGExpModel::calcSinConvNorm(Double_t sign, Double_t tau, Double_t omega) const
{
  RooComplex z(1/tau,sign*omega);
  return z*2/(omega*omega+1/(tau*tau));
}



RooComplex RooGExpModel::evalCerfApprox(Double_t swt, Double_t u, Double_t c) const
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


Int_t RooGExpModel::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ; 
  return 0 ;
}


void RooGExpModel::generateEvent(Int_t code)
{
  assert(code==1) ;
  Double_t xgen ;
  while(1) {
    Double_t xgau = RooRandom::randomGenerator()->Gaus(0,(sigma*ssf));
    Double_t xexp = RooRandom::uniform();
    if (!_flip) xgen= xgau + (rlife*rsf)*log(xexp);
    else xgen= xgau - (rlife*rsf)*log(xexp);
    if (xgen<x.max() && xgen>x.min()) {
      x = xgen ;
      return ;
    }
  }
}


