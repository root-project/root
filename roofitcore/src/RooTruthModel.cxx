/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTruthModel.cc,v 1.2 2001/06/09 05:08:48 verkerke Exp $
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
#include "RooFitCore/RooTruthModel.hh"

ClassImp(RooTruthModel) 
;


RooTruthModel::RooTruthModel(const char *name, const char *title, RooRealVar& x) : 
  RooResolutionModel(name,title,x)
{  
  //addServer((RooAbsArg&)basis()) ;
}


RooTruthModel::RooTruthModel(const RooTruthModel& other, const char* name) : 
  RooResolutionModel(other,name)
{
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
  //RooResolutionModel::changeBasis(basis) ;
  //return ;

  // Remove client-server link to old basis
  if (_basis) {
    //cout << "RooTruthModel::changeBasis(" << GetName() << "," << (void*)this << ") removing basis " << _basis->GetName() << " as server" << endl ;
    removeServer(*_basis) ;
  }

  // Change basis pointer and update client-server link
  _basis = basis ;
  if (_basis) {
    //cout << "RooTruthModel::changeBasis(" << GetName() << "," << (void*)this << " adding basis " << _basis->GetName() << " as server" << endl ;
    addServer(*_basis,kTRUE,kFALSE) ;
  }

  _basisCode = basis?basisCode(basis->GetName()):0 ;
}




Double_t RooTruthModel::evaluate(const RooDataSet* dset) const 
{
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
