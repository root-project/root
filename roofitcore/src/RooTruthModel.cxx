/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTruthModel.cc,v 1.7 2001/10/09 00:44:01 verkerke Exp $
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
  if (!TString("exp(-abs(@0)/@1)").CompareTo(name)) return expBasisPlus ;
  if (!TString("exp(-abs(-@0)/@1)").CompareTo(name)) return expBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisPlus ;
  if (!TString("exp(-abs(-@0)/@1)*sin(@0*@2)").CompareTo(name)) return sinBasisMinus ;
  if (!TString("exp(-abs(@0)/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisPlus ;
  if (!TString("exp(-abs(-@0)/@1)*cos(@0*@2)").CompareTo(name)) return cosBasisMinus ;

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

  _basisCode = basis?basisCode(basis->GetName()):0 ;
}





Double_t RooTruthModel::evaluate() const 
{
  // Evaluate the truth model: a delta function when used as PDF,
  // The basis function itself, when convoluted with a basis function.
  
  switch(_basisCode) {

  case 0:
    if (x==0) return 1 ;
    return 0 ;

  case expBasisPlus:
    {
      Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
      return x>0?exp(-x/tau):0 ;
    }
  case expBasisMinus:
    {
      Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
      return x<0?exp(x/tau):0 ;
    }

  case cosBasisPlus:
    {
      Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
      Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
      return x>0?exp(-x/tau)*cos(x*dm):0. ;
    }

  case cosBasisMinus:
    {
      Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
      Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
      return x<0?exp(x/tau)*cos(x*dm):0. ;
    }

  case sinBasisPlus:
    {
      Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
      Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
      return x>0?exp(-x/tau)*sin(x*dm):0. ;
    }

  case sinBasisMinus:
    {
      Double_t tau = ((RooAbsReal*)basis().getParameter(1))->getVal() ;
      Double_t dm = ((RooAbsReal*)basis().getParameter(2))->getVal() ;
      return x<0?exp(x/tau)*sin(x*dm):0. ;
    }

  default:
    return basis().getVal() ;
  }
}
