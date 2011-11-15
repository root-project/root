/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
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
//  which RooAbsAnaConvPdf uses to construct the pdf for [ Phys (x) R ] :
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
//

#include "RooFit.h"

#include "TClass.h"
#include "TMath.h"
#include "Riostream.h"
#include "RooResolutionModel.h"
#include "RooMsgService.h"
#include "RooSentinel.h"

ClassImp(RooResolutionModel) 
;

RooFormulaVar* RooResolutionModel::_identity = 0;



//_____________________________________________________________________________
void RooResolutionModel::cleanup()
{
  // Cleanup hook for RooSentinel atexit handler
  delete _identity ;
  _identity = 0 ;
}


//_____________________________________________________________________________
RooResolutionModel::RooResolutionModel(const char *name, const char *title, RooRealVar& _x) : 
  RooAbsPdf(name,title), 
  x("x","Dependent or convolution variable",this,_x),
  _basisCode(0), _basis(0), 
  _ownBasis(kFALSE)
{
  // Constructor with convolution variable 'x'

  if (!_identity) {
    _identity = identity() ; 
  }
}



//_____________________________________________________________________________
RooResolutionModel::RooResolutionModel(const RooResolutionModel& other, const char* name) : 
  RooAbsPdf(other,name), 
  x("x",this,other.x),
  _basisCode(other._basisCode), _basis(0),
  _ownBasis(kFALSE)
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
    while((basisServer = (RooAbsArg*)bsIter->Next())) {
      addServer(*basisServer,kTRUE,kFALSE) ;
    }
    delete bsIter ;
  }
}



//_____________________________________________________________________________
RooResolutionModel::~RooResolutionModel()
{
  // Destructor

  if (_ownBasis && _basis) {
    delete _basis ;
  }
}



//_____________________________________________________________________________
RooFormulaVar* RooResolutionModel::identity() 
{ 
  // Return identity formula pointer

  if (!_identity) {
    _identity = new RooFormulaVar("identity","1",RooArgSet("")) ;  
    RooSentinel::activate() ;
  }

  return _identity ; 
}



//_____________________________________________________________________________
RooResolutionModel* RooResolutionModel::convolution(RooFormulaVar* inBasis, RooAbsArg* owner) const
{
  // Instantiate a clone of this resolution model representing a convolution with given
  // basis function. The owners object name is incorporated in the clones name
  // to avoid multiple convolution objects with the same name in complex PDF structures.
  // 
  // Note: The 'inBasis' formula expression must be a RooFormulaVar that encodes the formula
  // in the title of the object and this expression must be an exact match against the
  // implemented basis function strings (see derived class implementation of method basisCode()
  // for those strings

  // Check that primary variable of basis functions is our convolution variable  
  if (inBasis->getParameter(0) != x.absArg()) {
    coutE(InputArguments) << "RooResolutionModel::convolution(" << GetName() << "," << this  
			  << ") convolution parameter of basis function and PDF don't match" << endl 
			  << "basis->findServer(0) = " << inBasis->findServer(0) << endl 
			  << "x.absArg()           = " << x.absArg() << endl ;
    return 0 ;
  }
  
  if (basisCode(inBasis->GetTitle())==0) {
    coutE(InputArguments) << "RooResolutionModel::convolution(" << GetName() << "," << this  
			  << ") basis function '" << inBasis->GetTitle() << "' is not supported." << endl ;
    return 0 ;
  }

  TString newName(GetName()) ;
  newName.Append("_conv_") ;
  newName.Append(inBasis->GetName()) ;
  newName.Append("_[") ;
  newName.Append(owner->GetName()) ;
  newName.Append("]") ;

  RooResolutionModel* conv = (RooResolutionModel*) clone(newName) ;
  
  TString newTitle(conv->GetTitle()) ;
  newTitle.Append(" convoluted with basis function ") ;
  newTitle.Append(inBasis->GetName()) ;
  conv->SetTitle(newTitle.Data()) ;

  conv->changeBasis(inBasis) ;

  return conv ;
}



//_____________________________________________________________________________
void RooResolutionModel::changeBasis(RooFormulaVar* inBasis) 
{
  // Change the basis function we convolute with.
  // For one-time use by convolution() only.

  // Remove client-server link to old basis
  if (_basis) {
    TIterator* bsIter = _basis->serverIterator() ;
    RooAbsArg* basisServer ;
    while((basisServer = (RooAbsArg*)bsIter->Next())) {
      removeServer(*basisServer) ;
    }
    delete bsIter ;

    if (_ownBasis) {
      delete _basis ;
    }
  }
  _ownBasis = kFALSE ;

  // Change basis pointer and update client-server link
  _basis = inBasis ;
  if (_basis) {
    TIterator* bsIter = _basis->serverIterator() ;
    RooAbsArg* basisServer ;
    while((basisServer = (RooAbsArg*)bsIter->Next())) {
      addServer(*basisServer,kTRUE,kFALSE) ;
    }
    delete bsIter ;
  }

  _basisCode = inBasis?basisCode(inBasis->GetTitle()):0 ;
}



//_____________________________________________________________________________
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



//_____________________________________________________________________________
RooRealVar& RooResolutionModel::convVar() const 
{
  // Return the convolution variable of the resolution model

  return (RooRealVar&) x.arg() ;
}



//_____________________________________________________________________________
Double_t RooResolutionModel::getValV(const RooArgSet* nset) const
{
  // Modified version of RooAbsPdf::getValF(). If used as regular PDF, 
  // call RooAbsPdf::getValF(), otherwise return unnormalized value
  // regardless of specified normalization set

  if (!_basis) return RooAbsPdf::getValV(nset) ;

  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty()) {
    _value = evaluate() ; 

    // WVE insert traceEval traceEval
    if (_verboseDirty) cxcoutD(Tracing) << "RooResolutionModel(" << GetName() << ") value = " << _value << endl ;

    clearValueDirty() ; 
    clearShapeDirty() ; 
  }

  return _value ;
}



//_____________________________________________________________________________
Bool_t RooResolutionModel::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t /*isRecursive*/) 
{
  // Forward redirectServers call to our basis function, which is not connected to either resolution
  // model or the physics model.

  if (!_basis) {
    _norm = 0 ;
    return kFALSE ; 
  } 

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



//_____________________________________________________________________________
Bool_t RooResolutionModel::traceEvalHook(Double_t value) const 
{
  // Floating point error checking and tracing for given float value

  // check for a math error or negative value
  return isnan(value) ;
}



//_____________________________________________________________________________
void RooResolutionModel::normLeafServerList(RooArgSet& list) const 
{
  // Return the list of servers used by our normalization integral

  _norm->leafNodeServerList(&list) ;
}



//_____________________________________________________________________________
Double_t RooResolutionModel::getNorm(const RooArgSet* nset) const
{
  // Return the integral of this PDF over all elements of 'nset'. 

  if (!nset) {
    return getVal() ;
  }

  syncNormalization(nset,kFALSE) ;
  if (_verboseEval>1) cxcoutD(Tracing) << IsA()->GetName() << "::getNorm(" << GetName() 
				       << "): norm(" << _norm << ") = " << _norm->getVal() << endl ;

  Double_t ret = _norm->getVal() ;
  return ret ;
}



//_____________________________________________________________________________
void RooResolutionModel::printMultiline(ostream& os, Int_t content, Bool_t verbose, TString indent) const
{
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsArg::printStream() we add:
  //
  //     Shape : value, units, plot range
  //   Verbose : default binning and print label

  RooAbsPdf::printMultiline(os,content,verbose,indent) ;

  if(verbose) {
    os << indent << "--- RooResolutionModel ---" << endl;
    os << indent << "basis function = " ; 
    if (_basis) {
      _basis->printStream(os,kName|kAddress|kTitle,kSingleLine,indent) ;
    } else {
      os << "<none>" << endl ;
    }
  }
}

