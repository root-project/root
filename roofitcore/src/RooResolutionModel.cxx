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
#include "RooFitCore/RooResolutionModel.hh"

ClassImp(RooResolutionModel) 
;


RooResolutionModel::RooResolutionModel(const char *name, const char *title) : 
  RooAbsPdf(name,title), _basis(0)
{
  
}


RooResolutionModel::RooResolutionModel(const RooResolutionModel& other, const char* name) : 
  RooAbsPdf(other,name),  _basis(other._basis)
{
  // Copy constructor
  if (_basis) addServer(*_basis,kTRUE,kFALSE) ;
}



RooResolutionModel::~RooResolutionModel()
{
  // Destructor
}



RooResolutionModel* RooResolutionModel::convolution(RooAbsReal* basis) const
{
  RooResolutionModel* conv = (RooResolutionModel*) clone() ;  

  conv->SetName(TString(conv->GetName()).Append("_conv_").Append(basis->GetName())) ;
  conv->SetTitle(TString(conv->GetTitle()).Append(" convoluted with basis function ").Append(basis->GetName())) ;
  conv->changeBasis(basis) ;

  return conv ;
}



void RooResolutionModel::changeBasis(RooAbsReal* basis) 
{
  // Remove client-server link to old basis
  if (_basis) {
    removeServer(*_basis) ;
  }

  // Change basis pointer and update client-server link
  _basis = basis ;
  if (_basis) {
    addServer(*_basis,kTRUE,kFALSE) ;
  }
}


const RooAbsReal& RooResolutionModel::basis() const {
  static RooRealVar identity("identity","Identity basis function",1) ;
  return _basis?*_basis:identity ;
}


const RooRealVar& RooResolutionModel::convVar() const 
{
  // Convolution variable is by definition first server of basis function
  TIterator* sIter = basis().serverIterator() ;
  RooRealVar* var = (RooRealVar*) sIter->Next() ;
  delete sIter ;

  return *var ;
}



Double_t RooResolutionModel::getVal(const RooDataSet* dset) const
{
  if (!_basis) return RooAbsPdf::getVal(dset) ;

  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty()) {
    if (_verboseEval) cout << "RooResolutionModel::getVal(" << GetName() << "): recalculating value" << endl ;    
    _value = traceEval(dset) ; 

    setValueDirty(kFALSE) ;
    setShapeDirty(kFALSE) ;    
  }

  return _value ;
}




Bool_t RooResolutionModel::redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll) 
{
  if (!_basis) return kFALSE ;

  RooAbsReal* newBasis = (RooAbsReal*) newServerList.find(_basis->GetName()) ;
  if (newBasis) _basis = newBasis ;

  return (mustReplaceAll && !newBasis) ;
}
