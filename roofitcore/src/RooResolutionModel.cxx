/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooResolutionModel.cc,v 1.10 2001/08/23 01:21:48 verkerke Exp $
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

RooFormulaVar* RooResolutionModel::_identity(0) ;


RooResolutionModel::RooResolutionModel(const char *name, const char *title, RooRealVar& _x) : 
  RooAbsPdf(name,title), _basis(0), _basisCode(0), x("x","Dependent or convolution variable",this,_x),
  _ownBasis(kFALSE) 
{
  if (!_identity) _identity = new RooFormulaVar("identity","1",RooArgSet("")) ;  
}


RooResolutionModel::RooResolutionModel(const RooResolutionModel& other, const char* name) : 
  RooAbsPdf(other,name), _basis(0), _basisCode(other._basisCode), x("x",this,other.x),
  _ownBasis(kFALSE) 
{
  if (other._basis) {
    _basis = (RooFormulaVar*) other._basis->Clone() ;
    _ownBasis = kTRUE ;
    //_basis = other._basis ;
  }

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
  if (_ownBasis && _basis) delete _basis ;
}



RooResolutionModel* RooResolutionModel::convolution(RooFormulaVar* basis, RooAbsArg* owner) const
{
  // Check that primary variable of basis functions is our convolution variable  
  if (basis->findServer(0) != x.absArg()) {
    cout << "RooResolutionModel::convolution(" << GetName() 
	 << ") convolution parameter of basis function and PDF don't match" << endl ;
    cout << "basis->findServer(0) = " << basis->findServer(0) << endl ;
    cout << "x.absArg()           = " << x.absArg() << endl ;
    return 0 ;
  }

  TString newName(GetName()) ;
  newName.Append("_conv_") ;
  newName.Append(basis->GetName()) ;
  newName.Append("_[") ;
  newName.Append(owner->GetName()) ;
  newName.Append("]") ;

  RooResolutionModel* conv = (RooResolutionModel*) clone(newName) ;
  
  TString newTitle(conv->GetTitle()) ;
  newTitle.Append(" convoluted with basis function ") ;
  newTitle.Append(basis->GetName()) ;
  conv->SetTitle(newTitle.Data()) ;

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



Double_t RooResolutionModel::getVal(const RooArgSet* nset) const
{
  if (!_basis) return RooAbsPdf::getVal(nset) ;

  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty()) {
    _value = evaluate() ; 

    // WVE insert traceEval traceEval
    if (_verboseDirty>1) cout << "RooResolutionModel(" << GetName() << ") value = " << _value << endl ;

    clearValueDirty() ; 
    clearShapeDirty() ; 
  }

  return _value ;
}



Bool_t RooResolutionModel::redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll) 
{
  if (!_basis) return kFALSE ;

  RooFormulaVar* newBasis = (RooFormulaVar*) newServerList.find(_basis->GetName()) ;
  if (newBasis) {
    _basis = newBasis ;
  }

  _basis->redirectServers(newServerList,mustReplaceAll) ;
    
  return (mustReplaceAll && !newBasis) ;
}



Bool_t RooResolutionModel::traceEvalHook(Double_t value) const 
{
  // Floating point error checking and tracing for given float value

  // check for a math error or negative value
  return isnan(value) ;
}



void RooResolutionModel::normLeafServerList(RooArgSet& list) const 
{
  // Fill list with leaf server nodes of normalization integral 
  _norm->leafNodeServerList(&list) ;
}
