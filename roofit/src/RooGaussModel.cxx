/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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
  switch(_basisCode) {
  case noBasis:  
    {
      Double_t xprime = (x-mean)/sigma ;
      return exp(-0.5*xprime*xprime) ;
    }
  case expBasisPlus: 
  case expBasisMinus: 
    {
      Double_t sign = (_basisCode==expBasisPlus)?-1:1 ;

      Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
      Double_t xprime = sign*(x-mean)/tau ;
      Double_t c = sigma/sqrt(2*tau) ;
      Double_t result = 0.25*tau * exp(xprime+c*c) * erfc(xprime/(2*c)+c) ;

      //cout << "RooGaussModel::evaluate_expBasis result(x=" << x << ") : " << result << endl ;
      return result ;
    }

  case sinBasisPlus: return 0 ;
  case sinBasisMinus: return 0 ;
  case cosBasisPlus: return 0 ;
  case cosBasisMinus: return 0 ;
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
    if (matchArgs(allVars,analVars,convVar())) {
      return 1 ;
    }
    break ;

  case sinBasisPlus:
  case sinBasisMinus:
  case cosBasisPlus:
  case cosBasisMinus:
    return 0 ;
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

  switch(_basisCode) {
  case noBasis:  
    {      
      Double_t xscale = root2*sigma;
      return rootPiBy2*sigma*(erf((x.max()-mean)/xscale)-erf((x.min()-mean)/xscale));
    }

  case expBasisPlus: 
  case expBasisMinus:
    {
      Double_t sign = (_basisCode==expBasisPlus)?-1:1 ;
      Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
      Double_t c = sigma/sqrt(2*tau) ; 
      Double_t xpmin = sign*(x.min()-mean)/tau ;
      Double_t xpmax = sign*(x.max()-mean)/tau ;

      Double_t result = sign * 0.25 * exp(c*c) * ( erf(xpmax/(2*c)) - erf(xpmin/(2*c)) 
                                                  + exp(xpmax)*erfc(xpmax/(2*c)+c)
                                                  - exp(xpmin)*erfc(xpmin/(2*c)+c) ) ;     
      return result ;
    }


  case sinBasisPlus: return 1 ;
  case sinBasisMinus: return 1 ;
  case cosBasisPlus: return 1 ;
  case cosBasisMinus: return 1 ;
  }

}
