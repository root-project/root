/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooResolutionModel.cc,v 1.22 2002/04/03 23:37:26 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// 
//  RooResolutionModel is the base class of for PDFs that represent a
//  resolution model that can be convoluted with physics a physics model of the form
//
//    Phys(x,a,b) = Sum_k coef_k(a) * basis_k(x,b)
//  
//  where basis_k are a limited number of functions in terms of the variable
//  to be convoluted and coef_k are coefficients independent of the convolution
//  variable.
//  
//  Classes derived from RooResolutionModel implement 
//         _ _                        _                  _
//   R_k(x,b,c) = Int(dx') basis_k(x',b) * resModel(x-x',c)
// 
//  which RooConvolutedPdf uses to construct the pdf for [ Phys (x) R ] :
//          _ _ _                 _          _ _
//    PDF(x,a,b,c) = Sum_k coef_k(a) * R_k(x,b,c)
//
//  A minimal implementation of a RooResolutionModel consists of a
//
//    Int_t basisCode(const char* name)   
//
//  function indication which basis functions this resolution model supports, and
//
//    Double_t evaluate() 
//
//  Implementing the resolution model, optionally convoluted with one of the
//  supported basis functions. RooResolutionModel objects can be used as regular
//  PDFs (They inherit from RooAbsPdf), or as resolution model convoluted with
//  a basis function. The implementation of evaluate() can identify the requested
//  from of use from the basisCode() function. If zero, the regular PDF value
//  should be calculated. If non-zero, the models value convoluted with the
//  basis function identified by the code should be calculated.
//
//  Optionally, analytical integrals can be advertised and implemented, in the
//  same way as done for regular PDFS (see RooAbsPdf for further details).
//  Also in getAnalyticalIntegral()/analyticalIntegral() the implementation
//  should use basisCode() to determine for which scenario the integral is
//  requested.
//
//  The choice of basis returned by basisCode() is guaranteed not to change
//  of the lifetime of a RooResolutionModel object.

#include <iostream.h>
#include "RooFitCore/RooResolutionModel.hh"

ClassImp(RooResolutionModel) 
;

RooFormulaVar* RooResolutionModel::_identity = 0;


RooResolutionModel::RooResolutionModel(const char *name, const char *title, RooRealVar& _x) : 
  RooAbsPdf(name,title), _basis(0), _basisCode(0), x("x","Dependent or convolution variable",this,_x),
  _ownBasis(kFALSE), _normSpecial(0), _lastNormSetSpecial(0)
{
  // Constructor with convolution variable 'x'
  if (!_identity) _identity = new RooFormulaVar("identity","1",RooArgSet("")) ;  
}


RooResolutionModel::RooResolutionModel(const RooResolutionModel& other, const char* name) : 
  RooAbsPdf(other,name), _basis(0), _basisCode(other._basisCode), x("x",this,other.x),
  _ownBasis(kFALSE), _normSpecial(0), _lastNormSetSpecial(0)
{
  // Copy constructor

  if (other._basis) {
    _basis = (RooFormulaVar*) other._basis->Clone() ;
    _ownBasis = kTRUE ;
    //_basis = other._basis ;
  }

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

  if (_ownBasis && _basis) {
    delete _basis ;
  }

  if (_normSpecial) delete _normSpecial ;
}



RooResolutionModel* RooResolutionModel::convolution(RooFormulaVar* basis, RooAbsArg* owner) const
{
  // Instantiate a clone of this resolution model representing a convolution with given
  // basis function. The owners object name is incorporated in the clones name
  // to avoid multiple convolution objects with the same name in complex PDF structures.

  // Check that primary variable of basis functions is our convolution variable  
  if (basis->getParameter(0) != x.absArg()) {
    cout << "RooResolutionModel::convolution(" << GetName() << "," << this  
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
  // Change the basis function we convolute with.
  // For one-time use by convolution() only.

  // Remove client-server link to old basis
  if (_basis) {
    TIterator* bsIter = _basis->serverIterator() ;
    RooAbsArg* basisServer ;
    while(basisServer = (RooAbsArg*)bsIter->Next()) {
      removeServer(*basisServer) ;
    }
    delete bsIter ;

    if (_ownBasis) {
      delete _basis ;
    }
  }
  _ownBasis = kFALSE ;

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

  _basisCode = basis?basisCode(basis->GetTitle()):0 ;
}



const RooRealVar& RooResolutionModel::basisConvVar() const 
{
  // Return the convolution variable of the selection basis function.
  // This is, by definition, the first parameter of the basis function

  // Convolution variable is by definition first server of basis function
  TIterator* sIter = basis().serverIterator() ;
  RooRealVar* var = (RooRealVar*) sIter->Next() ;
  delete sIter ;

  return *var ;
}


RooRealVar& RooResolutionModel::convVar() const 
{
  // Return the convolution variable of the resolution model
  return (RooRealVar&) x.arg() ;
}



Double_t RooResolutionModel::getVal(const RooArgSet* nset) const
{
  // Modified version of RooAbsPdf::getVal(). If used as regular PDF, 
  // call RooAbsPdf::getVal(), otherwise return unnormalized value
  // regardless of specified normalization set

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



Bool_t RooResolutionModel::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange) 
{
  // Forward redirectServers call to our basis function, which is not connected to either resolution
  // model or the physics model.

  if (!_basis) return kFALSE ;

  RooFormulaVar* newBasis = (RooFormulaVar*) newServerList.find(_basis->GetName()) ;
  if (newBasis) {

    if (_ownBasis) {
      delete _basis ;
    }

    _basis = newBasis ;
    _ownBasis = kFALSE ;
  }

  _basis->redirectServers(newServerList,mustReplaceAll,nameChange) ;
    
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
  // Return the list of servers used by our normalization integral
  _norm->leafNodeServerList(&list) ;
}



Double_t RooResolutionModel::getNormSpecial(const RooArgSet* nset) const 
{
  // Replica of RooAbsPdf::getNorm that uses a separate cache to store normalization
  // integral. Used by RooConvolutedPdf::analyticalIntegralWN(), which, for
  // normalized integrals, must retrieve two different integrals for each convolution 
  // object. Using RooAbsPdf::getNorm for both would lead to 100% cache misses.

  if (!nset) {
    return getVal() ;
  }

  if (nset != _lastNormSetSpecial) {

    if (_verboseEval>0) {
      cout << "RooResolutionModel::getNormSpecial(" << GetName() 
	   << ") recreating normalization integral(" 
	   << _lastNormSet << " -> " << nset << "=" ;
      if (nset) nset->Print("1") ; else cout << "<none>" ;
      cout << ")" << endl ;
    }

    if (_normSpecial) delete _normSpecial ;

    TString nname(GetName()) ; nname.Append("NormSpecial") ;
    TString ntitle(GetTitle()) ; ntitle.Append(" Integral (Special)") ;
    _normSpecial = new RooRealIntegral(nname.Data(),ntitle.Data(),*this,*(RooArgSet*)nset) ;

    _lastNormSetSpecial = (RooArgSet*)nset ;
  }

  return _normSpecial->getVal() ;
}


Double_t RooResolutionModel::getNorm(const RooArgSet* nset) const
{
  // Return the integral of this PDF over all elements of 'nset'. 
  if (!nset) {
    return getVal() ;
  }

  syncNormalization(nset) ;
  if (_verboseEval>1) cout << IsA()->GetName() << "::getNorm(" << GetName() 
			   << "): norm(" << _norm << ") = " << _norm->getVal() << endl ;

  Double_t ret = _norm->getVal() ;
  return ret ;
}


void RooResolutionModel::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsArg::printToStream() we add:
  //
  //     Shape : value, units, plot range
  //   Verbose : default binning and print label

  RooAbsPdf::printToStream(os,opt,indent) ;

  if(opt >= Verbose) {
    os << indent << "--- RooResolutionModel ---" << endl;
    os << indent << "basis function = " ; 
    if (_basis) {
      _basis->printToStream(os,opt,indent) ;
    } else {
      os << "<none>" << endl ;
    }
  }
}
