/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTruthModel.cc,v 1.11 2001/11/05 18:50:50 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// RooTruthModel is the delta-function resolution model
//
// The truth model supports all basis functions because it evaluates each basis function as  
// as a RooFormulaVar. The 6 basis functions used in B mixing and decay have been
// hand coded for speed.

#include <iostream.h>
#include "RooFitCore/RooTruthModel.hh"

ClassImp(RooTruthModel) 
;


RooTruthModel::RooTruthModel(const char *name, const char *title, RooRealVar& x) : 
  RooResolutionModel(name,title,x)
{  
  // Constructor

}


RooTruthModel::RooTruthModel(const RooTruthModel& other, const char* name) : 
  RooResolutionModel(other,name)
{
  // Copy constructor
}


RooTruthModel::~RooTruthModel()
{
  // Destructor
}


Int_t RooTruthModel::basisCode(const char* name) const 
{
  // Check for optimized basis functions
  if (!TString("exp(-@0/@1)").CompareTo(name)) return expBasisPlus ;
  if (!TString("exp(@0/@1)").CompareTo(name)) return expBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)").CompareTo(name)) return expBasisSum ;
  if (!TString("exp(-@0/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisPlus ;
  if (!TString("exp(@0/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisSum ;
  if (!TString("exp(-@0/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisPlus ;
  if (!TString("exp(@0/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisSum ;

  // Truth model is delta function, i.e. convolution integral
  // is basis function, therefore we can handle any basis function
  return genericBasis ;
}




void RooTruthModel::changeBasis(RooFormulaVar* basis) 
{
  // Process change basis function. Since we actually
  // evaluate the basis function object, we need to
  // adjust our client-server links to the basis function here

  // Remove client-server link to old basis
  if (_basis) {
    removeServer(*_basis) ;
  }

  // Change basis pointer and update client-server link
  _basis = basis ;
  if (_basis) {
    addServer(*_basis,kTRUE,kFALSE) ;
  }

  _basisCode = basis?basisCode(basis->GetTitle()):0 ;
}





Double_t RooTruthModel::evaluate() const 
{
  // Evaluate the truth model: a delta function when used as PDF,
  // The basis function itself, when convoluted with a basis function.

  // No basis: delta function
  if (_basisCode == noBasis) {
    if (x==0) return 1 ;
    return 0 ;
  }

  // Generic basis: evaluate basis function object
  if (_basisCode == genericBasis) {
    return basis().getVal() ;
  }

  // Precompiled basis functions
  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;

  // Enforce sign compatibility
  if ((basisSign==Minus && x>0) || 
      (basisSign==Plus  && x<0)) return 0 ;


  Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
  // Return desired basis function
  switch(basisType) {    
  case expBasis: {
    return exp(-fabs(x)/tau) ;
  }
  case sinBasis: {
    Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ; 
    return exp(-fabs(x)/tau)*sin(x*dm) ;
  }
  case cosBasis: {
    Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ; 
    return exp(-fabs(x)/tau)*cos(x*dm) ;
  }
  }
}



Int_t RooTruthModel::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
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


Double_t RooTruthModel::analyticalIntegral(Int_t code) const 
{
  // Code must be 1
  assert(code==1) ;

  // Unconvoluted PDF
  if (_basisCode==noBasis) return 1 ;

  // Precompiled basis functions
  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;

  Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
  switch (basisType) {
  case expBasis:
    {
      Double_t result(0) ;
      if (basisSign != Minus) result += tau*(1-exp(-x.max()/tau)) ;
      if (basisSign != Plus) result += tau*(1-exp(x.min()/tau)) ;
      return result ;
    }
  case sinBasis:
    {
      Double_t result(0) ;
      Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
      if (basisSign != Minus) result += exp(-x.max()/tau)*(-1/tau*sin(dm*x.max()) - dm*cos(dm*x.max())) + 1/tau;
      if (basisSign != Plus)  result -= exp( x.min()/tau)*(-1/tau*sin(dm*(-x.min())) - dm*cos(dm*(-x.min()))) + 1/tau ;
      return result / (1/(tau*tau) + dm*dm) ;
    }
  case cosBasis:
    {
      Double_t result(0) ;
      Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
      if (basisSign != Minus) result += exp(-x.max()/tau)*(-1/tau*cos(dm*x.max()) + dm*sin(dm*x.max())) + 1/tau ;
      if (basisSign != Plus)  result += exp( x.min()/tau)*(-1/tau*cos(dm*(-x.min())) - dm*sin(dm*(-x.min()))) + 1/tau ;
      return result / (1/(tau*tau) + dm*dm) ;
    }
  }
  
}



Int_t RooTruthModel::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;  
  return 0 ;
}


void RooTruthModel::generateEvent(Int_t code)
{
  assert(code==1) ;
  Double_t zero(0.) ;
  x = zero ;
  return;
}
