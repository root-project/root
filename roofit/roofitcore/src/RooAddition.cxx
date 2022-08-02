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

/**
\file RooAddition.cxx
\class RooAddition
\ingroup Roofitcore

RooAddition calculates the sum of a set of RooAbsReal terms, or
when constructed with two sets, it sums the product of the terms
in the two sets.
**/


#include "Riostream.h"
#include "RooAddition.h"
#include "RooRealSumFunc.h"
#include "RooRealSumPdf.h"
#include "RooProduct.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooNameReg.h"
#include "RooNLLVar.h"
#include "RooNLLVarNew.h"
#include "RooChi2Var.h"
#include "RooMsgService.h"
#include "RooBatchCompute.h"

#include <algorithm>
#include <cmath>

ClassImp(RooAddition);


////////////////////////////////////////////////////////////////////////////////
/// Constructor with a single set consisting of RooAbsReal.
/// \param[in] name Name of the PDF
/// \param[in] title Title
/// \param[in] sumSet The value of the function will be the sum of the values in this set
/// \param[in] takeOwnership If true, the RooAddition object will take ownership of the arguments in `sumSet`

RooAddition::RooAddition(const char* name, const char* title, const RooArgList& sumSet, bool takeOwnership)
  : RooAbsReal(name, title)
  , _set("!set","set of components",this)
  , _cacheMgr(this,10)
{
  for (const auto comp : sumSet) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp->GetName()
             << " is not of type RooAbsReal" << std::endl;
      RooErrorHandler::softAbort() ;
    }
    _set.add(*comp) ;
    if (takeOwnership) _ownedList.addOwned(*comp) ;
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with two sets of RooAbsReals.
///
/// The sum of pair-wise products of elements in the sets will be computed:
/// \f[
///  A = \sum_i \mathrm{Set1}[i] * \mathrm{Set2}[i]
/// \f]
///
/// \param[in] name Name of the PDF
/// \param[in] title Title
/// \param[in] sumSet1 Left-hand element of the pair-wise products
/// \param[in] sumSet2 Right-hand element of the pair-wise products
/// \param[in] takeOwnership If true, the RooAddition object will take ownership of the arguments in the `sumSets`
///
RooAddition::RooAddition(const char* name, const char* title, const RooArgList& sumSet1, const RooArgList& sumSet2, bool takeOwnership)
    : RooAbsReal(name, title)
    , _set("!set","set of components",this)
    , _cacheMgr(this,10)
{
  if (sumSet1.getSize() != sumSet2.getSize()) {
    coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: input lists should be of equal length" << std::endl;
    RooErrorHandler::softAbort() ;
  }

  for (unsigned int i = 0; i < sumSet1.size(); ++i) {
    const auto comp1 = &sumSet1[i];
    const auto comp2 = &sumSet2[i];

    if (!dynamic_cast<RooAbsReal*>(comp1)) {
      coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp1->GetName()
             << " in first list is not of type RooAbsReal" << std::endl;
      RooErrorHandler::softAbort() ;
    }

    if (!dynamic_cast<RooAbsReal*>(comp2)) {
      coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp2->GetName()
             << " in first list is not of type RooAbsReal" << std::endl;
      RooErrorHandler::softAbort() ;
    }
    // TODO: add flag to RooProduct c'tor to make it assume ownership...
    TString _name(name);
    _name.Append( "_[");
    _name.Append(comp1->GetName());
    _name.Append( "_x_");
    _name.Append(comp2->GetName());
    _name.Append( "]");
    RooProduct  *prod = new RooProduct( _name, _name , RooArgSet(*comp1, *comp2) /*, takeOwnership */ ) ;
    _set.add(*prod);
    _ownedList.addOwned(*prod) ;
    if (takeOwnership) {
        _ownedList.addOwned(*comp1) ;
        _ownedList.addOwned(*comp2) ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAddition::RooAddition(const RooAddition& other, const char* name)
    : RooAbsReal(other, name)
    , _set("!set",this,other._set)
    , _cacheMgr(other._cacheMgr,this)
{
  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate and return current value of self

double RooAddition::evaluate() const
{
  double sum(0);
  const RooArgSet* nset = _set.nset() ;

  for (auto* comp : static_range_cast<RooAbsReal*>(_set)) {
    const double tmp = comp->getVal(nset);
    sum += tmp ;
  }
  return sum ;
}


////////////////////////////////////////////////////////////////////////////////
/// Compute addition of PDFs in batches.
void RooAddition::computeBatch(cudaStream_t* stream, double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const
{
  RooBatchCompute::VarVector pdfs;
  RooBatchCompute::ArgVector coefs;
  pdfs.reserve(_set.size());
  coefs.reserve(_set.size());
  for (const auto arg : _set)
  {
    pdfs.push_back(dataMap.at(arg));
    coefs.push_back(1.0);
  }
  auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;
  dispatch->compute(stream, RooBatchCompute::AddPdf, output, nEvents, pdfs, coefs);
}


////////////////////////////////////////////////////////////////////////////////
/// Return the default error level for MINUIT error analysis
/// If the addition contains one or more RooNLLVars and
/// no RooChi2Vars, return the defaultErrorLevel() of
/// RooNLLVar. If the addition contains one ore more RooChi2Vars
/// and no RooNLLVars, return the defaultErrorLevel() of
/// RooChi2Var. If the addition contains neither or both
/// issue a warning message and return a value of 1

double RooAddition::defaultErrorLevel() const
{
  RooAbsReal* nllArg(0) ;
  RooAbsReal* chi2Arg(0) ;

  std::unique_ptr<RooArgSet> comps{getComponents()};
  for(RooAbsArg * arg : *comps) {
    if (dynamic_cast<RooNLLVar*>(arg) || dynamic_cast<ROOT::Experimental::RooNLLVarNew*>(arg)) {
      nllArg = (RooAbsReal*)arg ;
    }
    if (dynamic_cast<RooChi2Var*>(arg)) {
      chi2Arg = (RooAbsReal*)arg ;
    }
  }

  if (nllArg && !chi2Arg) {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName()
         << ") Summation contains a RooNLLVar, using its error level" << std::endl;
    return nllArg->defaultErrorLevel() ;
  } else if (chi2Arg && !nllArg) {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName()
         << ") Summation contains a RooChi2Var, using its error level" << std::endl;
    return chi2Arg->defaultErrorLevel() ;
  } else if (!nllArg && !chi2Arg) {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName() << ") WARNING: "
         << "Summation contains neither RooNLLVar nor RooChi2Var server, using default level of 1.0" << std::endl;
  } else {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName() << ") WARNING: "
         << "Summation contains BOTH RooNLLVar and RooChi2Var server, using default level of 1.0" << std::endl;
  }

  return 1.0 ;
}


////////////////////////////////////////////////////////////////////////////////

void RooAddition::enableOffsetting(bool flag)
{
  for (auto arg : _set) {
    static_cast<RooAbsReal*>(arg)->enableOffsetting(flag) ;
  }
}



////////////////////////////////////////////////////////////////////////////////

bool RooAddition::setData(RooAbsData& data, bool cloneData)
{
  for (const auto arg : _set) {
    static_cast<RooAbsReal*>(arg)->setData(data,cloneData) ;
  }
  return true ;
}



////////////////////////////////////////////////////////////////////////////////

void RooAddition::printMetaArgs(std::ostream& os) const
{
  // We can use the implementation of RooRealSumPdf with an empy coefficient list.
  static const RooArgList coefs{};
  RooRealSumPdf::printMetaArgs(_set, coefs, os);
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooAddition::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const
{
  // we always do things ourselves -- actually, always delegate further down the line ;-)
  analVars.add(allVars);

  // check if we already have integrals for this combination of factors
  Int_t sterileIndex(-1);
  CacheElem* cache = (CacheElem*) _cacheMgr.getObj(&analVars,&analVars,&sterileIndex,RooNameReg::ptr(rangeName));
  if (cache!=0) {
    Int_t code = _cacheMgr.lastIndex();
    return code+1;
  }

  // we don't, so we make it right here....
  cache = new CacheElem;
  for (const auto arg : _set) {// checked in c'tor that this will work...
      RooAbsReal *I = static_cast<const RooAbsReal*>(arg)->createIntegral(analVars,rangeName);
      cache->_I.addOwned(*I);
  }

  Int_t code = _cacheMgr.setObj(&analVars,&analVars,(RooAbsCacheElement*)cache,RooNameReg::ptr(rangeName));
  return 1+code;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate integral internally from appropriate integral cache

double RooAddition::analyticalIntegral(Int_t code, const char* rangeName) const
{
  // note: rangeName implicit encoded in code: see _cacheMgr.setObj in getPartIntList...
  CacheElem *cache = (CacheElem*) _cacheMgr.getObjByIndex(code-1);
  if (cache==0) {
    // cache got sterilized, trigger repopulation of this slot, then try again...
    std::unique_ptr<RooArgSet> vars( getParameters(RooArgSet()) );
    RooArgSet iset = _cacheMgr.selectFromSet2(*vars, code-1);
    RooArgSet dummy;
    Int_t code2 = getAnalyticalIntegral(iset,dummy,rangeName);
    assert(code==code2); // must have revived the right (sterilized) slot...
    return analyticalIntegral(code2,rangeName);
  }
  assert(cache!=0);

  // loop over cache, and sum...
  double result(0);
  for (auto I : cache->_I) {
    result += static_cast<const RooAbsReal*>(I)->getVal();
  }
  return result;

}


////////////////////////////////////////////////////////////////////////////////

std::list<double>* RooAddition::binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  return RooRealSumPdf::binBoundaries(_set, obs, xlo, xhi);
}


bool RooAddition::isBinnedDistribution(const RooArgSet& obs) const
{
  return RooRealSumPdf::isBinnedDistribution(_set, obs);
}


////////////////////////////////////////////////////////////////////////////////

std::list<double>* RooAddition::plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  return RooRealSumPdf::plotSamplingHint(_set, obs, xlo, xhi);
}
