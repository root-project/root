/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooResolutionModel.cc,v 1.2 2001/06/09 05:08:48 verkerke Exp $
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


RooResolutionModel::RooResolutionModel(const char *name, const char *title, RooRealVar& _x) : 
  RooAbsPdf(name,title), _basis(0), _basisCode(0), x("x","Dependent or convolution variable",this,_x)
{
  
}


RooResolutionModel::RooResolutionModel(const RooResolutionModel& other, const char* name) : 
  RooAbsPdf(other,name),  _basis(other._basis), _basisCode(other._basisCode), x("x",this,other.x)
{
  // Copy constructor
  if (_basis) {
    TIterator* bsIter = _basis->serverIterator() ;
    RooAbsArg* basisServer ;
    while(basisServer = (RooAbsArg*)bsIter->Next()) {
      addServer(*basisServer,kTRUE,kFALSE) ;
    }
    delete bsIter ;
  }
}



RooResolutionModel::~RooResolutionModel()
{
  // Destructor
}



RooResolutionModel* RooResolutionModel::convolution(RooFormulaVar* basis) const
{
  // Check that primary variable of basis functions is our convolution variable  
  if (basis->findServer(0) != x.absArg()) {
    cout << "RooResolutionModel::convolution(" << GetName() 
	 << " convolution parameter of basis function and PDF don't match" << endl ;
    return 0 ;
  }

  TString newName(GetName()) ;
  newName.Append("_conv_") ;
  newName.Append(basis->GetName()) ;

  RooResolutionModel* conv = (RooResolutionModel*) clone(newName) ;
  
  conv->SetTitle(TString(conv->GetTitle()).Append(" convoluted with basis function ").Append(basis->GetName())) ;
  conv->changeBasis(basis) ;

  return conv ;
}



void RooResolutionModel::changeBasis(RooFormulaVar* basis) 
{
  // Remove client-server link to old basis
  if (_basis) {
    TIterator* bsIter = _basis->serverIterator() ;
    RooAbsArg* basisServer ;
    while(basisServer = (RooAbsArg*)bsIter->Next()) {
      removeServer(*basisServer) ;
    }
    delete bsIter ;
  }

  // Change basis pointer and update client-server link
  _basis = basis ;
  if (_basis) {
    TIterator* bsIter = _basis->serverIterator() ;
    RooAbsArg* basisServer ;
    while(basisServer = (RooAbsArg*)bsIter->Next()) {
      addServer(*basisServer,kTRUE,kFALSE) ;
    }
    delete bsIter ;
  }

  _basisCode = basis?basisCode(basis->GetName()):0 ;
}


const RooFormulaVar& RooResolutionModel::basis() const {
  static RooRealVar one("one","one",1) ;
  static RooFormulaVar identity("identity","1*one",RooArgSet(one)) ;
  return _basis?*_basis:identity ;
}


const RooRealVar& RooResolutionModel::basisConvVar() const 
{
  // Convolution variable is by definition first server of basis function
  TIterator* sIter = basis().serverIterator() ;
  RooRealVar* var = (RooRealVar*) sIter->Next() ;
  delete sIter ;

  return *var ;
}


RooRealVar& RooResolutionModel::convVar() const 
{
  return (RooRealVar&) x.arg() ;
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

  RooFormulaVar* newBasis = (RooFormulaVar*) newServerList.find(_basis->GetName()) ;
  if (newBasis) _basis = newBasis ;

  return (mustReplaceAll && !newBasis) ;
}



Bool_t RooResolutionModel::traceEvalHook(Double_t value) const 
{
  // Floating point error checking and tracing for given float value

  // check for a math error or negative value
  return isnan(value) ;
}

