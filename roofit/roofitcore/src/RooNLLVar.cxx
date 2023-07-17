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
\file RooNLLVar.cxx
\class RooNLLVar
\ingroup Roofitcore

Class RooNLLVar implements a -log(likelihood) calculation from a dataset
and a PDF. The NLL is calculated as
\f[
 \sum_\mathrm{data} -\log( \mathrm{pdf}(x_\mathrm{data}))
\f]
In extended mode, a
\f$ N_\mathrm{expect} - N_\mathrm{observed}*log(N_\mathrm{expect}) \f$ term is added.
**/

#include "RooNLLVar.h"

#include "RooAbsData.h"
#include "RooAbsPdf.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"
#include "RooAbsDataStore.h"
#include "RooRealMPFE.h"
#include "RooRealSumPdf.h"
#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooNaNPacker.h"
#include "RunContext.h"
#include "RooDataHist.h"

#ifdef ROOFIT_CHECK_CACHED_VALUES
#include <iomanip>
#endif

#include "TMath.h"
#include "Math/Util.h"

#include <algorithm>

namespace {
  template<class ...Args>
  RooAbsTestStatistic::Configuration makeRooAbsTestStatisticCfg(Args const& ... args) {
    RooAbsTestStatistic::Configuration cfg;
    cfg.rangeName = RooCmdConfig::decodeStringOnTheFly("RooNLLVar::RooNLLVar","RangeWithName",0,"",args...);
    cfg.addCoefRangeName = RooCmdConfig::decodeStringOnTheFly("RooNLLVar::RooNLLVar","AddCoefRange",0,"",args...);
    cfg.nCPU = RooCmdConfig::decodeIntOnTheFly("RooNLLVar::RooNLLVar","NumCPU",0,1,args...);
    cfg.interleave = RooFit::BulkPartition;
    cfg.verbose = static_cast<bool>(RooCmdConfig::decodeIntOnTheFly("RooNLLVar::RooNLLVar","Verbose",0,1,args...));
    cfg.splitCutRange = static_cast<bool>(RooCmdConfig::decodeIntOnTheFly("RooNLLVar::RooNLLVar","SplitRange",0,0,args...));
    cfg.cloneInputData = static_cast<bool>(RooCmdConfig::decodeIntOnTheFly("RooNLLVar::RooNLLVar","CloneData",0,1,args...));
    cfg.integrateOverBinsPrecision = RooCmdConfig::decodeDoubleOnTheFly("RooNLLVar::RooNLLVar", "IntegrateBins", 0, -1., {args...});
    return cfg;
  }
}

ClassImp(RooNLLVar)

RooNLLVar::~RooNLLVar() {}

RooArgSet RooNLLVar::_emptySet ;

////////////////////////////////////////////////////////////////////////////////
/// Construct likelihood from given p.d.f and (binned or unbinned dataset)
///
///  Argument                 | Description
///  -------------------------|------------
///  Extended()               | Include extended term in calculation
///  NumCPU()                 | Activate parallel processing feature
///  Range()                  | Fit only selected region
///  SumCoefRange()           | Set the range in which to interpret the coefficients of RooAddPdf components
///  SplitRange()             | Fit range is split by index category of simultaneous PDF
///  ConditionalObservables() | Define conditional observables
///  Verbose()                | Verbose output of GOF framework classes
///  CloneData()              | Clone input dataset for internal use (default is true)
///  BatchMode()              | Evaluate batches of data events (faster if PDFs support it)
///  IntegrateBins() | Integrate PDF within each bin. This sets the desired precision. Only useful for binned fits.
RooNLLVar::RooNLLVar(const char *name, const char* title, RooAbsPdf& pdf, RooAbsData& indata,
           const RooCmdArg& arg1, const RooCmdArg& arg2,const RooCmdArg& arg3,
           const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6,
           const RooCmdArg& arg7, const RooCmdArg& arg8,const RooCmdArg& arg9) :
  RooAbsOptTestStatistic(name,title,pdf,indata,
                         *RooCmdConfig::decodeSetOnTheFly(
                             "RooNLLVar::RooNLLVar","ProjectedObservables",0,&_emptySet,
                             arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
                         makeRooAbsTestStatisticCfg(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9))
{
  RooCmdConfig pc("RooNLLVar::RooNLLVar") ;
  pc.allowUndefined() ;
  pc.defineInt("extended","Extended",0,false) ;
  pc.defineInt("BatchMode", "BatchMode", 0, false);

  pc.process(arg1) ;  pc.process(arg2) ;  pc.process(arg3) ;
  pc.process(arg4) ;  pc.process(arg5) ;  pc.process(arg6) ;
  pc.process(arg7) ;  pc.process(arg8) ;  pc.process(arg9) ;

  _extended = pc.getInt("extended") ;
}


////////////////////////////////////////////////////////////////////////////////
/// Construct likelihood from given p.d.f and (binned or unbinned dataset)
/// For internal use.

RooNLLVar::RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& indata,
                     bool extended, RooAbsTestStatistic::Configuration const& cfg) :
  RooNLLVar{name, title, pdf, indata, RooArgSet(), extended, cfg} {}


////////////////////////////////////////////////////////////////////////////////
/// Construct likelihood from given p.d.f and (binned or unbinned dataset)
/// For internal use.

RooNLLVar::RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& indata,
                     const RooArgSet& projDeps,
                     bool extended, RooAbsTestStatistic::Configuration const& cfg) :
  RooAbsOptTestStatistic(name,title,pdf,indata,projDeps, cfg),
  _extended(extended)
{
  // If binned likelihood flag is set, pdf is a RooRealSumPdf representing a yield vector
  // for a binned likelihood calculation
  _binnedPdf = cfg.binnedL ? static_cast<RooRealSumPdf*>(_funcClone) : nullptr ;

  // Retrieve and cache bin widths needed to convert un-normalized binnedPdf values back to yields
  if (_binnedPdf) {

    // The Active label will disable pdf integral calculations
    _binnedPdf->setAttribute("BinnedLikelihoodActive") ;

    RooArgSet obs;
    _funcClone->getObservables(_dataClone->get(), obs);
    if (obs.size()!=1) {
      _binnedPdf = nullptr;
    } else {
      auto* var = static_cast<RooRealVar*>(obs.first());
      std::unique_ptr<std::list<double>> boundaries{_binnedPdf->binBoundaries(*var,var->getMin(),var->getMax())};
      auto biter = boundaries->begin() ;
      _binw.reserve(boundaries->size()-1) ;
      double lastBound = (*biter) ;
      ++biter ;
      while (biter!=boundaries->end()) {
        _binw.push_back((*biter) - lastBound);
        lastBound = (*biter) ;
        ++biter ;
      }
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooNLLVar::RooNLLVar(const RooNLLVar& other, const char* name) :
  RooAbsOptTestStatistic(other,name),
  _extended(other._extended),
  _weightSq(other._weightSq),
  _offsetSaveW2(other._offsetSaveW2),
  _binw(other._binw),
  _binnedPdf{other._binnedPdf}
{
}


////////////////////////////////////////////////////////////////////////////////
/// Create a test statistic using several properties of the current instance. This is used to duplicate
/// the test statistic in multi-processing scenarios.
RooAbsTestStatistic* RooNLLVar::create(const char *name, const char *title, RooAbsReal& pdf, RooAbsData& adata,
            const RooArgSet& projDeps, RooAbsTestStatistic::Configuration const& cfg) {
  RooAbsPdf & thePdf = dynamic_cast<RooAbsPdf&>(pdf);
  // check if pdf can be extended
  bool extendedPdf = _extended && thePdf.canBeExtended();

  auto testStat = new RooNLLVar(name, title, thePdf, adata, projDeps, extendedPdf, cfg);
  return testStat;
}


////////////////////////////////////////////////////////////////////////////////

void RooNLLVar::applyWeightSquared(bool flag)
{
  if (_gofOpMode==Slave) {
    if (flag != _weightSq) {
      _weightSq = flag;
      std::swap(_offset, _offsetSaveW2);
    }
    setValueDirty();
  } else if ( _gofOpMode==MPMaster) {
    for (int i=0 ; i<_nCPU ; i++)
      _mpfeArray[i]->applyNLLWeightSquared(flag);
  } else if ( _gofOpMode==SimMaster) {
    for(auto& gof : _gofArray)
      static_cast<RooNLLVar&>(*gof).applyWeightSquared(flag);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate and return likelihood on subset of data.
/// \param[in] firstEvent First event to be processed.
/// \param[in] lastEvent  First event not to be processed, any more.
/// \param[in] stepSize   Steps between events.
/// \note For batch computations, the step size **must** be one.
///
/// If this an extended likelihood, the extended term is added to the return likelihood
/// in the batch that encounters the event with index 0.

double RooNLLVar::evaluatePartition(std::size_t firstEvent, std::size_t lastEvent, std::size_t stepSize) const
{
  // Throughout the calculation, we use Kahan's algorithm for summing to
  // prevent loss of precision - this is a factor four more expensive than
  // straight addition, but since evaluating the PDF is usually much more
  // expensive than that, we tolerate the additional cost...
  ROOT::Math::KahanSum<double> result{0.0};
  double sumWeight{0.0};

  auto * pdfClone = static_cast<RooAbsPdf*>(_funcClone);

  // cout << "RooNLLVar::evaluatePartition(" << GetName() << ") projDeps = " << (_projDeps?*_projDeps:RooArgSet()) << endl ;

  _dataClone->store()->recalculateCache( _projDeps, firstEvent, lastEvent, stepSize, (_binnedPdf?false:true) ) ;



  // If pdf is marked as binned - do a binned likelihood calculation here (sum of log-Poisson for each bin)
  if (_binnedPdf) {
    ROOT::Math::KahanSum<double> sumWeightKahanSum{0.0};
    for (auto i=firstEvent ; i<lastEvent ; i+=stepSize) {

      _dataClone->get(i) ;

      double eventWeight = _dataClone->weight();


      // Calculate log(Poisson(N|mu) for this bin
      double N = eventWeight ;
      double mu = _binnedPdf->getVal()*_binw[i] ;
      //cout << "RooNLLVar::binnedL(" << GetName() << ") N=" << N << " mu = " << mu << endl ;

      if (mu<=0 && N>0) {

        // Catch error condition: data present where zero events are predicted
        logEvalError(Form("Observed %f events in bin %lu with zero event yield",N,(unsigned long)i)) ;

      } else if (std::abs(mu)<1e-10 && std::abs(N)<1e-10) {

        // Special handling of this case since log(Poisson(0,0)=0 but can't be calculated with usual log-formula
        // since log(mu)=0. No update of result is required since term=0.

      } else {

        result += -1*(-mu + N * std::log(mu) - TMath::LnGamma(N+1));
        sumWeightKahanSum += eventWeight;

      }
    }

    sumWeight = sumWeightKahanSum.Sum();

  } else { //unbinned PDF

    std::tie(result, sumWeight) = computeScalar(stepSize, firstEvent, lastEvent);

    // include the extended maximum likelihood term, if requested
    if(_extended && _setNum==_extSet) {
      result += pdfClone->extendedTerm(*_dataClone, _weightSq, _doBinOffset);
    }
  } //unbinned PDF


  // If part of simultaneous PDF normalize probability over
  // number of simultaneous PDFs: -sum(log(p/n)) = -sum(log(p)) + N*log(n)
  if (_simCount>1) {
    result += sumWeight * std::log(static_cast<double>(_simCount));
  }


  // At the end of the first full calculation, wire the caches
  if (_first) {
    _first = false ;
    _funcClone->wireAllCaches() ;
  }


  // Check if value offset flag is set.
  if (_doOffset) {

    // If no offset is stored enable this feature now
    if (_offset.Sum() == 0 && _offset.Carry() == 0 && (result.Sum() != 0 || result.Carry() != 0)) {
      coutI(Minimization) << "RooNLLVar::evaluatePartition(" << GetName() << ") first = "<< firstEvent << " last = " << lastEvent << " Likelihood offset now set to " << result.Sum() << std::endl ;
      _offset = result ;
    }

    // Subtract offset
    result -= _offset;
  }

  _evalCarry = result.Carry();
  return result.Sum() ;
}

RooNLLVar::ComputeResult RooNLLVar::computeScalar(std::size_t stepSize, std::size_t firstEvent, std::size_t lastEvent) const {
  auto pdfClone = static_cast<const RooAbsPdf*>(_funcClone);
  return computeScalarFunc(pdfClone, _dataClone, _normSet, _weightSq, stepSize, firstEvent, lastEvent, _doBinOffset);
}

// static function, also used from TestStatistics::RooUnbinnedL
RooNLLVar::ComputeResult RooNLLVar::computeScalarFunc(const RooAbsPdf *pdfClone, RooAbsData *dataClone,
                                                      RooArgSet *normSet, bool weightSq, std::size_t stepSize,
                                                      std::size_t firstEvent, std::size_t lastEvent, bool doBinOffset)
{
  ROOT::Math::KahanSum<double> kahanWeight;
  ROOT::Math::KahanSum<double> kahanProb;
  RooNaNPacker packedNaN(0.f);
  const double logSumW = std::log(dataClone->sumEntries());

  auto* dataHist = doBinOffset ? static_cast<RooDataHist*>(dataClone) : nullptr;

  for (auto i=firstEvent; i<lastEvent; i+=stepSize) {
    dataClone->get(i) ;

    double weight = dataClone->weight(); //FIXME
    const double ni = weight;

    if (0. == weight * weight) continue ;
    if (weightSq) weight = dataClone->weightSquared() ;

    double logProba = pdfClone->getLogVal(normSet);

    if(doBinOffset) {
      logProba -= std::log(ni) - std::log(dataHist->binVolume(i)) - logSumW;
    }

    const double term = -weight * logProba;

    kahanWeight.Add(weight);
    kahanProb.Add(term);
    packedNaN.accumulate(term);
  }

  if (packedNaN.getPayload() != 0.) {
    // Some events with evaluation errors. Return "badness" of errors.
    return {ROOT::Math::KahanSum<double>{packedNaN.getNaNWithPayload()}, kahanWeight.Sum()};
  }

  return {kahanProb, kahanWeight.Sum()};
}
