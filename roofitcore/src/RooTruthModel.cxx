/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTruthModel.cc,v 1.6 2001/10/08 05:20:23 verkerke Exp $
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
// as a RooFormulaVar. The downside of this technique is that RooTruthModel is not exceptionally 
// fast, but this is usually not a problem as this model is mostly used in debugging.
// If necessary, the performance can be improved by hard coding frequently used basis functions.

#include <iostream.h>
#include "RooFitCore/RooTruthModel.hh"

ClassImp(RooTruthModel) 
;


RooTruthModel::RooTruthModel(const char *name, const char *title, RooRealVar& x) : 
  RooResolutionModel(name,title,x)
{  
  // Constructor

  //addServer((RooAbsArg&)basis()) ;
}


RooTruthModel::RooTruthModel(const RooTruthModel& other, const char* name) : 
  RooResolutionModel(other,name)
{
  // Copy constructor

  //addServer((RooAbsArg&)basis()) ;
}


RooTruthModel::~RooTruthModel()
{
  // Destructor
}


Int_t RooTruthModel::basisCode(const char* name) const 
{
  // Truth model is delta function, i.e. convolution integral
  // is basis function, therefore we can handle any basis function
  return 1 ;
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
    {
      if (x==0) return 1 ;
      return 0 ;
    }
  default:
    return basis().getVal() ;
  }
}
