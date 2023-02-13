/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofit:$Id$
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
\file RooHistPdf.cxx
\class RooHistPdf
\ingroup Roofitcore

RooHistPdf implements a probablity density function sampled from a
multidimensional histogram. The histogram distribution is explicitly
normalized by RooHistPdf and can have an arbitrary number of real or
discrete dimensions.
**/

#include "Riostream.h"

#include "RooHistPdf.h"
#include "RooDataHist.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooCategory.h"
#include "RooWorkspace.h"
#include "RooGlobalFunc.h"
#include "RooHelpers.h"

#include "TError.h"
#include "TBuffer.h"


ClassImp(RooHistPdf);


////////////////////////////////////////////////////////////////////////////////
/// Constructor from a RooDataHist. RooDataHist dimensions
/// can be either real or discrete. See RooDataHist::RooDataHist for details on the binning.
/// RooHistPdf neither owns or clone 'dhist' and the user must ensure the input histogram exists
/// for the entire life span of this PDF.

RooHistPdf::RooHistPdf(const char *name, const char *title, const RooArgSet& vars,
             const RooDataHist& dhist, Int_t intOrder) :
  RooAbsPdf(name,title),
  _pdfObsList("pdfObs","List of p.d.f. observables",this),
  _dataHist((RooDataHist*)&dhist),
  _codeReg(10),
  _intOrder(intOrder),
  _cdfBoundaries(false)
{
  _histObsList.addClone(vars) ;
  _pdfObsList.add(vars) ;

  // Verify that vars and dhist.get() have identical contents
  const RooArgSet* dvars = dhist.get() ;
  if (vars.size()!=dvars->size()) {
    coutE(InputArguments) << "RooHistPdf::ctor(" << GetName()
           << ") ERROR variable list and RooDataHist must contain the same variables." << std::endl ;
    assert(0) ;
  }
  for (const auto arg : vars) {
    if (!dvars->find(arg->GetName())) {
      coutE(InputArguments) << "RooHistPdf::ctor(" << GetName()
             << ") ERROR variable list and RooDataHist must contain the same variables." << std::endl ;
      assert(0) ;
    }
  }


  // Adjust ranges of _histObsList to those of _dataHist
  for (const auto hobs : _histObsList) {
    // Guaranteed to succeed, since checked above in ctor
    RooAbsArg* dhobs = dhist.get()->find(hobs->GetName()) ;
    RooRealVar* dhreal = dynamic_cast<RooRealVar*>(dhobs) ;
    if (dhreal){
      ((RooRealVar*)hobs)->setRange(dhreal->getMin(),dhreal->getMax()) ;
    }
  }

}




////////////////////////////////////////////////////////////////////////////////
/// Constructor from a RooDataHist. The first list of observables are the p.d.f.
/// observables, which may any RooAbsReal (function or variable). The second list
/// are the corresponding observables in the RooDataHist which must be of type
/// RooRealVar or RooCategory This constructor thus allows to apply a coordinate transformation
/// on the histogram data to be applied.

RooHistPdf::RooHistPdf(const char *name, const char *title, const RooArgList& pdfObs,
             const RooArgList& histObs, const RooDataHist& dhist, Int_t intOrder) :
  RooAbsPdf(name,title),
  _pdfObsList("pdfObs","List of p.d.f. observables",this),
  _dataHist((RooDataHist*)&dhist),
  _codeReg(10),
  _intOrder(intOrder),
  _cdfBoundaries(false)
{
  _histObsList.addClone(histObs) ;
  _pdfObsList.add(pdfObs) ;

  // Verify that vars and dhist.get() have identical contents
  const RooArgSet* dvars = dhist.get() ;
  if (histObs.size()!=dvars->size()) {
    coutE(InputArguments) << "RooHistPdf::ctor(" << GetName()
           << ") ERROR histogram variable list and RooDataHist must contain the same variables." << std::endl ;
    throw(std::string("RooHistPdf::ctor() ERROR: histogram variable list and RooDataHist must contain the same variables")) ;
  }

  for (const auto arg : histObs) {
    if (!dvars->find(arg->GetName())) {
      coutE(InputArguments) << "RooHistPdf::ctor(" << GetName()
             << ") ERROR variable list and RooDataHist must contain the same variables." << std::endl ;
      throw(std::string("RooHistPdf::ctor() ERROR: histogram variable list and RooDataHist must contain the same variables")) ;
    }
    if (!arg->isFundamental()) {
      coutE(InputArguments) << "RooHistPdf::ctor(" << GetName()
             << ") ERROR all elements of histogram observables set must be of type RooRealVar or RooCategory." << std::endl ;
      throw(std::string("RooHistPdf::ctor() ERROR all elements of histogram observables set must be of type RooRealVar or RooCategory.")) ;
    }
  }


  // Adjust ranges of _histObsList to those of _dataHist
  for (const auto hobs : _histObsList) {
    // Guaranteed to succeed, since checked above in ctor
    RooAbsArg* dhobs = dhist.get()->find(hobs->GetName()) ;
    RooRealVar* dhreal = dynamic_cast<RooRealVar*>(dhobs) ;
    if (dhreal){
      ((RooRealVar*)hobs)->setRange(dhreal->getMin(),dhreal->getMax()) ;
    }
  }
}


RooHistPdf::RooHistPdf(const char *name, const char *title, const RooArgSet& vars,
           std::unique_ptr<RooDataHist> dhist, int intOrder)
  : RooHistPdf{name, title, vars, *dhist, intOrder}
{
  _ownedDataHist = std::move(dhist);
}
RooHistPdf::RooHistPdf(const char *name, const char *title, const RooArgList& pdfObs, const RooArgList& histObs,
           std::unique_ptr<RooDataHist> dhist, int intOrder)
  : RooHistPdf{name, title, pdfObs, histObs, *dhist, intOrder}
{
  _ownedDataHist = std::move(dhist);
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooHistPdf::RooHistPdf(const RooHistPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _pdfObsList("pdfObs",this,other._pdfObsList),
  _dataHist(other._dataHist),
  _codeReg(other._codeReg),
  _intOrder(other._intOrder),
  _cdfBoundaries(other._cdfBoundaries),
  _totVolume(other._totVolume),
  _unitNorm(other._unitNorm)
{
  _histObsList.addClone(other._histObsList) ;

}




////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooHistPdf::~RooHistPdf()
{

}

RooDataHist* RooHistPdf::cloneAndOwnDataHist(const char* newname) {
   if (_ownedDataHist) return _ownedDataHist.get();
   _ownedDataHist.reset(static_cast<RooDataHist*>(_dataHist->Clone(newname)));
   _dataHist = _ownedDataHist.get();
   return _dataHist;
}

void RooHistPdf::computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const {

  // For interpolation and histograms of higher dimension, use base function
  if(_pdfObsList.size() > 1) {
      RooAbsReal::computeBatch(nullptr, output, nEvents, dataMap);
      return;
  }

  auto xVals = dataMap.at(_pdfObsList[0]);
  _dataHist->weights(output, xVals, _intOrder, true, _cdfBoundaries);
}


////////////////////////////////////////////////////////////////////////////////
/// Return the current value: The value of the bin enclosing the current coordinates
/// of the observables, normalized by the histograms contents. Interpolation
/// is applied if the RooHistPdf is configured to do that.

double RooHistPdf::evaluate() const
{
  // Transfer values from
  for (unsigned int i=0; i < _pdfObsList.size(); ++i) {
    RooAbsArg* harg = _histObsList[i];
    RooAbsArg* parg = _pdfObsList[i];

    if (harg != parg) {
      parg->syncCache() ;
      harg->copyCache(parg,true) ;
      if (!harg->inRange(0)) {
        return 0 ;
      }
    }
  }

  double ret = _dataHist->weightFast(_histObsList, _intOrder, !_unitNorm, _cdfBoundaries);

  return std::max(ret, 0.0);
}


////////////////////////////////////////////////////////////////////////////////
/// Return the total volume spanned by the observables of the RooHistPdf

double RooHistPdf::totVolume() const
{
  // Return previously calculated value, if any
  if (_totVolume>0) {
    return _totVolume ;
  }
  _totVolume = 1. ;

  for (const auto arg : _histObsList) {
    RooRealVar* real = dynamic_cast<RooRealVar*>(arg) ;
    if (real) {
      _totVolume *= (real->getMax()-real->getMin()) ;
    } else {
      RooCategory* cat = dynamic_cast<RooCategory*>(arg) ;
      if (cat) {
   _totVolume *= cat->numTypes() ;
      }
    }
  }

  return _totVolume ;
}

namespace {
bool fullRange(const RooAbsArg& x, const RooAbsArg& y ,const char* range)
{
  const RooAbsRealLValue *_x = dynamic_cast<const RooAbsRealLValue*>(&x);
  const RooAbsRealLValue *_y = dynamic_cast<const RooAbsRealLValue*>(&y);
  if (!_x || !_y) return false;
  if (!range || !strlen(range) || !_x->hasRange(range) ||
      _x->getBinningPtr(range)->isParameterized()) {
    // parameterized ranges may be full range now, but that might change,
    // so return false
    if (range && strlen(range) && _x->getBinningPtr(range)->isParameterized())
      return false;
    return (_x->getMin() == _y->getMin() && _x->getMax() == _y->getMax());
  }
  return (_x->getMin(range) == _y->getMin() && _x->getMax(range) == _y->getMax());
}
}


Int_t RooHistPdf::getAnalyticalIntegral(RooArgSet& allVars,
                                        RooArgSet& analVars,
                                        const char* rangeName,
                                        RooArgSet const& histObsList,
                                        RooSetProxy const& pdfObsList,
                                        Int_t intOrder) {
  // First make list of pdf observables to histogram observables
  // and select only those for which the integral is over the full range

  Int_t code = 0;
  Int_t frcode = 0;
  for (unsigned int n=0; n < pdfObsList.size() && n < histObsList.size(); ++n) {
    const auto pa = pdfObsList[n];
    const auto ha = histObsList[n];

    if (allVars.find(*pa)) {
      code |= 2 << n;
      analVars.add(*pa);
      if (fullRange(*pa, *ha, rangeName)) {
        frcode |= 2 << n;
      }
    }
  }

  if (code == frcode) {
    // integrate over full range of all observables - use bit 0 to indicate
    // full range integration over all observables
    code |= 1;
  }

  // Disable partial analytical integrals if interpolation is used, and we
  // integrate over sub-ranges, but leave them enabled when we integrate over
  // the full range of one or several variables
  if (intOrder > 1 && !(code & 1)) {
    analVars.removeAll();
    return 0;
  }
  return (code >= 2) ? code : 0;
}


double RooHistPdf::analyticalIntegral(Int_t code,
                                        const char* rangeName,
                                        RooArgSet const& histObsList,
                                        RooSetProxy const& pdfObsList,
                                        RooDataHist& dataHist,
                                        bool histFuncMode) {
  // Simplest scenario, full-range integration over all dependents
  if (((2 << histObsList.size()) - 1) == code) {
    return dataHist.sum(histFuncMode);
  }

  // Partial integration scenario, retrieve set of variables, calculate partial
  // sum, figure out integration ranges (if needed)
  RooArgSet intSet;
  std::map<const RooAbsArg*, std::pair<double, double> > ranges;
  for (unsigned int n=0; n < pdfObsList.size() && n < histObsList.size(); ++n) {
    const auto pa = pdfObsList[n];
    const auto ha = histObsList[n];

    if (code & (2 << n)) {
      intSet.add(*ha);
    }
    if (!(code & 1)) {
      ranges[ha] = RooHelpers::getRangeOrBinningInterval(pa, rangeName);
    }
    // WVE must sync hist slice list values to pdf slice list
    // Transfer values from
    if (ha != pa) {
      pa->syncCache();
      ha->copyCache(pa,true);
    }
  }

  double ret = (code & 1) ? dataHist.sum(intSet,histObsList,true,!histFuncMode) :
                              dataHist.sum(intSet,histObsList,true,!histFuncMode, ranges);

  return ret ;
}

////////////////////////////////////////////////////////////////////////////////
/// Determine integration scenario. If no interpolation is used,
/// RooHistPdf can perform all integrals over its dependents
/// analytically via partial or complete summation of the input
/// histogram. If interpolation is used on the integral over
/// all histogram observables is supported

Int_t RooHistPdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const
{
  return getAnalyticalIntegral(allVars, analVars, rangeName, _histObsList, _pdfObsList, _intOrder);
}


////////////////////////////////////////////////////////////////////////////////
/// Return integral identified by 'code'. The actual integration
/// is deferred to RooDataHist::sum() which implements partial
/// or complete summation over the histograms contents.

double RooHistPdf::analyticalIntegral(Int_t code, const char* rangeName) const
{
    return analyticalIntegral(code, rangeName, _histObsList, _pdfObsList, *_dataHist, false);
}


////////////////////////////////////////////////////////////////////////////////
/// Return sampling hint for making curves of (projections) of this function
/// as the recursive division strategy of RooCurve cannot deal efficiently
/// with the vertical lines that occur in a non-interpolated histogram

std::list<double>* RooHistPdf::plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  return plotSamplingHint(*_dataHist, _pdfObsList, _histObsList, _intOrder, obs, xlo, xhi);
}


std::list<double>* RooHistPdf::plotSamplingHint(RooDataHist const& dataHist,
                                                RooArgSet const& pdfObsList,
                                                RooArgSet const& histObsList,
                                                int intOrder,
                                                RooAbsRealLValue& obs,
                                                double xlo,
                                                double xhi)
{
  // No hints are required when interpolation is used
  if (intOrder>0) {
    return nullptr;
  }

  // Check that observable is in dataset, if not no hint is generated
  RooAbsArg* dhObs = nullptr;
  for (unsigned int i=0; i < pdfObsList.size(); ++i) {
    RooAbsArg* histObs = histObsList[i];
    RooAbsArg* pdfObs = pdfObsList[i];
    if (std::string(obs.GetName())==pdfObs->GetName()) {
      dhObs = dataHist.get()->find(histObs->GetName()) ;
      break;
    }
  }

  if (!dhObs) {
    return nullptr;
  }
  RooAbsLValue* lval = dynamic_cast<RooAbsLValue*>(dhObs) ;
  if (!lval) {
    return nullptr;
  }

  // Retrieve position of all bin boundaries

  const RooAbsBinning* binning = lval->getBinningPtr(nullptr);
  std::span<double> boundaries{binning->array(), static_cast<std::size_t>(binning->numBoundaries())};

  auto hint = new std::list<double> ;

  const double delta = (xhi-xlo)*1e-8 ;

  // Sample points right next to the plot limits
  hint->push_back(xlo + delta);
  hint->push_back(xhi - delta);

  // Sample points very close to the left and right of the bin boundaries that
  // are strictly in between the plot limits.
  for (const double x : boundaries) {
    if (x - xlo > delta && xhi - x > delta) {
      hint->push_back(x - delta);
      hint->push_back(x + delta);
    }
  }

  return hint ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return sampling hint for making curves of (projections) of this function
/// as the recursive division strategy of RooCurve cannot deal efficiently
/// with the vertical lines that occur in a non-interpolated histogram

std::list<double>* RooHistPdf::binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  // No hints are required when interpolation is used
  if (_intOrder>0) {
    return nullptr;
  }

  // Check that observable is in dataset, if not no hint is generated
  RooAbsLValue* lvarg = dynamic_cast<RooAbsLValue*>(_dataHist->get()->find(obs.GetName())) ;
  if (!lvarg) {
    return 0 ;
  }

  // Retrieve position of all bin boundaries
  const RooAbsBinning* binning = lvarg->getBinningPtr(nullptr);
  double* boundaries = binning->array() ;

  auto hint = new std::list<double> ;

  // Construct array with pairs of points positioned epsilon to the left and
  // right of the bin boundaries
  for (Int_t i=0 ; i<binning->numBoundaries() ; i++) {
    if (boundaries[i]>=xlo && boundaries[i]<=xhi) {
      hint->push_back(boundaries[i]) ;
    }
  }

  return hint ;
}




////////////////////////////////////////////////////////////////////////////////
/// Only handle case of maximum in all variables

Int_t RooHistPdf::getMaxVal(const RooArgSet& vars) const
{
  std::unique_ptr<RooAbsCollection> common{_pdfObsList.selectCommon(vars)};
  if (common->size()==_pdfObsList.size()) {
    return 1;
  }
  return 0 ;
}


////////////////////////////////////////////////////////////////////////////////

double RooHistPdf::maxVal(Int_t code) const
{
  R__ASSERT(code==1) ;

  double max(-1) ;
  for (Int_t i=0 ; i<_dataHist->numEntries() ; i++) {
    _dataHist->get(i) ;
    double wgt = _dataHist->weight() ;
    if (wgt>max) max=wgt ;
  }

  return max*1.05 ;
}




////////////////////////////////////////////////////////////////////////////////

bool RooHistPdf::areIdentical(const RooDataHist& dh1, const RooDataHist& dh2)
{
  if (std::abs(dh1.sumEntries()-dh2.sumEntries())>1e-8) return false ;
  if (dh1.numEntries() != dh2.numEntries()) return false ;
  for (int i=0 ; i < dh1.numEntries() ; i++) {
    dh1.get(i) ;
    dh2.get(i) ;
    if (std::abs(dh1.weight()-dh2.weight())>1e-8) return false ;
  }
  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if our datahist is already in the workspace

bool RooHistPdf::importWorkspaceHook(RooWorkspace& ws)
{
  for(auto const& data : ws.allData()) {
    // If your dataset is already in this workspace nothing needs to be done
    if (data == _dataHist) {
      return false ;
    }
  }

  // Check if dataset with given name already exists
  if (RooAbsData* wsdata = ws.embeddedData(_dataHist->GetName())) {

    // Yes it exists - now check if it is identical to our internal histogram
    if (wsdata->InheritsFrom(RooDataHist::Class())) {

      // Check if histograms are identical
      if (areIdentical((RooDataHist&)*wsdata,*_dataHist)) {

   // Exists and is of correct type, and identical -- adjust internal pointer to WS copy
   _dataHist = (RooDataHist*) wsdata ;
      } else {

   // not identical, clone rename and import
   auto uniqueName = std::string(_dataHist->GetName()) + "_" + GetName();
   bool flag = ws.import(*_dataHist,RooFit::Rename(uniqueName.c_str()),RooFit::Embedded()) ;
   if (flag) {
     coutE(ObjectHandling) << " RooHistPdf::importWorkspaceHook(" << GetName() << ") unable to import clone of underlying RooDataHist with unique name " << uniqueName << ", abort" << std::endl ;
     return true ;
   }
   _dataHist = (RooDataHist*) ws.embeddedData(uniqueName.c_str()) ;
      }

    } else {

      // Exists and is NOT of correct type: clone rename and import
      auto uniqueName = std::string(_dataHist->GetName()) + "_" + GetName();
      bool flag = ws.import(*_dataHist,RooFit::Rename(uniqueName.c_str()),RooFit::Embedded()) ;
      if (flag) {
   coutE(ObjectHandling) << " RooHistPdf::importWorkspaceHook(" << GetName() << ") unable to import clone of underlying RooDataHist with unique name " << uniqueName << ", abort" << std::endl ;
   return true ;
      }
      _dataHist = static_cast<RooDataHist*>(ws.embeddedData(uniqueName.c_str()));

    }
    return false ;
  }

  // We need to import our datahist into the workspace
  ws.import(*_dataHist,RooFit::Embedded()) ;

  // Redirect our internal pointer to the copy in the workspace
  _dataHist = (RooDataHist*) ws.embeddedData(_dataHist->GetName()) ;
  return false ;
}


////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class RooHistPdf.

void RooHistPdf::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RooHistPdf::Class(),this);
      // WVE - interim solution - fix proxies here
      //_proxyList.Clear() ;
      //registerProxy(_pdfObsList) ;
   } else {
      R__b.WriteClassBuffer(RooHistPdf::Class(),this);
   }
}

