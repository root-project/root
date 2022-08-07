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
/**
 * \class RooResolutionModel
 *  RooResolutionModel is the base class for PDFs that represent a
 *  resolution model that can be convoluted with a physics model of the form
 *  \f[
 *    \mathrm{Phys}(x,a,b) = \sum_k \mathrm{coef}_k(a) * \mathrm{basis}_k(x,b)
 *  \f]
 *  where basis_k are a limited number of functions in terms of the variable
 *  to be convoluted and coef_k are coefficients independent of the convolution
 *  variable.
 *
 *  Classes derived from RooResolutionModel implement
 *  \f[
 *   R_k(x,\bar{b},\bar{c}) = \int \mathrm{basis}_k(x',\bar{b}) * \mathrm{resModel}(x-x',\bar{c}) \; \mathrm{d} x',
 *  \f]
 *  which RooAbsAnaConvPdf uses to construct the pdf for [ Phys (x) R ] :
 *  \f[
 *    \mathrm{PDF}(x,\bar a, \bar b, \bar c) = \sum_k \mathrm{coef}_k(\bar a) * R_k(x, \bar b, \bar c)
 *  \f]
 *
 *  A minimal implementation of a RooResolutionModel consists of a
 *  ```
 *    Int_t basisCode(const char* name)
 *  ```
 *  function indicating which basis functions this resolution model supports, and
 *  ```
 *    double evaluate(),
 *  ```
 *  which should implement the resolution model (optionally convoluted with one of the
 *  supported basis functions). RooResolutionModel objects can be used as regular
 *  PDFs (They inherit from RooAbsPdf), or as resolution model convoluted with
 *  a basis function. The implementation of evaluate() can identify the requested
 *  mode using basisCode(). If zero, the regular PDF value
 *  should be calculated. If non-zero, the model's value convoluted with the
 *  basis function identified by the code should be calculated.
 *
 *  Optionally, analytical integrals can be advertised and implemented, in the
 *  same way as done for regular PDFS (see RooAbsPdf for further details).
 *  Also in getAnalyticalIntegral() / analyticalIntegral(), the implementation
 *  should use basisCode() to determine for which scenario the integral is
 *  requested.
 *
 *  The choice of basis returned by basisCode() is guaranteed not to change
 *  during the lifetime of a RooResolutionModel object.
 *
 */

#include "TClass.h"
#include "TMath.h"
#include "Riostream.h"
#include "RooResolutionModel.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooResolutionModel);


////////////////////////////////////////////////////////////////////////////////
/// Constructor with convolution variable 'x'.
/// The convolution variable needs to be convertable to real values, and be able
/// to give information about its range. This is supported by e.g. RooRealVar or RooLinearVar, which
/// accepts offsetting and scaling an observable.
RooResolutionModel::RooResolutionModel(const char *name, const char *title, RooAbsRealLValue& _x) :
  RooAbsPdf(name,title),
  x("x","Dependent or convolution variable",this,_x),
  _basisCode(0), _basis(0),
  _ownBasis(false)
{

}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooResolutionModel::RooResolutionModel(const RooResolutionModel& other, const char* name) :
  RooAbsPdf(other,name),
  x("x",this,other.x),
  _basisCode(other._basisCode), _basis(0),
  _ownBasis(false)
{
  if (other._basis) {
    _basis = (RooFormulaVar*) other._basis->Clone() ;
    _ownBasis = true ;
    //_basis = other._basis ;
  }

  if (_basis) {
    for (RooAbsArg * basisServer : _basis->servers()) {
      addServer(*basisServer,true,false) ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooResolutionModel::~RooResolutionModel()
{
  if (_ownBasis && _basis) {
    delete _basis ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return identity formula pointer

RooFormulaVar* RooResolutionModel::identity()
{
  static RooFormulaVar identity("identity","1",RooArgSet(""));
  return &identity;
}



////////////////////////////////////////////////////////////////////////////////
/// Instantiate a clone of this resolution model representing a convolution with given
/// basis function. The owners object name is incorporated in the clones name
/// to avoid multiple convolution objects with the same name in complex PDF structures.
///
/// Note: The 'inBasis' formula expression must be a RooFormulaVar that encodes the formula
/// in the title of the object and this expression must be an exact match against the
/// implemented basis function strings (see derived class implementation of method basisCode()
/// for those strings

RooResolutionModel* RooResolutionModel::convolution(RooFormulaVar* inBasis, RooAbsArg* owner) const
{
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



////////////////////////////////////////////////////////////////////////////////
/// Change the basis function we convolute with.
/// For one-time use by convolution() only.

void RooResolutionModel::changeBasis(RooFormulaVar* inBasis)
{
  // Remove client-server link to old basis
  if (_basis) {
    for (RooAbsArg* basisServer : _basis->servers()) {
      removeServer(*basisServer) ;
    }

    if (_ownBasis) {
      delete _basis ;
    }
  }
  _ownBasis = false ;

  // Change basis pointer and update client-server link
  _basis = inBasis ;
  if (_basis) {
    for (RooAbsArg* basisServer : _basis->servers()) {
      addServer(*basisServer,true,false) ;
    }
  }

  _basisCode = inBasis?basisCode(inBasis->GetTitle()):0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the convolution variable of the selection basis function.
/// This is, by definition, the first parameter of the basis function

const RooRealVar& RooResolutionModel::basisConvVar() const
{
  // Convolution variable is by definition first server of basis function
  return *static_cast<RooRealVar const*>(*basis().servers().begin());
}


////////////////////////////////////////////////////////////////////////////////
/// Modified version of RooAbsPdf::getValF(). If used as regular PDF,
/// call RooAbsPdf::getValF(), otherwise return unnormalized value
/// regardless of specified normalization set

double RooResolutionModel::getValV(const RooArgSet* nset) const
{
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



////////////////////////////////////////////////////////////////////////////////
/// Forward redirectServers call to our basis function, which is not connected to either resolution
/// model or the physics model.

bool RooResolutionModel::redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive)
{
  if (!_basis) {
    _norm = 0 ;
    return false ;
  }

  RooFormulaVar* newBasis = (RooFormulaVar*) newServerList.find(_basis->GetName()) ;
  if (newBasis) {

    if (_ownBasis) {
      delete _basis ;
    }

    _basis = newBasis ;
    _ownBasis = false ;
  }

  _basis->redirectServers(newServerList,mustReplaceAll,nameChange) ;

  return (mustReplaceAll && !newBasis) || RooAbsPdf::redirectServersHook(newServerList, mustReplaceAll, nameChange, isRecursive);
}



////////////////////////////////////////////////////////////////////////////////
/// Floating point error checking and tracing for given float value

//bool RooResolutionModel::traceEvalHook(double value) const
//{
//  // check for a math error or negative value
//   return TMath::IsNaN(value) ;
//}



////////////////////////////////////////////////////////////////////////////////
/// Return the list of servers used by our normalization integral

void RooResolutionModel::normLeafServerList(RooArgSet& list) const
{
  _norm->leafNodeServerList(&list) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the integral of this PDF over all elements of 'nset'.

double RooResolutionModel::getNorm(const RooArgSet* nset) const
{
  if (!nset) {
    return getVal() ;
  }

  syncNormalization(nset,false) ;
  if (_verboseEval>1) cxcoutD(Tracing) << ClassName() << "::getNorm(" << GetName()
                   << "): norm(" << _norm << ") = " << _norm->getVal() << endl ;

  double ret = _norm->getVal() ;
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print info about this object to the specified stream. In addition to the info
/// from RooAbsArg::printStream() we add:
///
///     Shape : value, units, plot range
///   Verbose : default binning and print label

void RooResolutionModel::printMultiline(ostream& os, Int_t content, bool verbose, TString indent) const
{
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

