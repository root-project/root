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

Implements a -log(likelihood) calculation from a dataset
and a PDF. The NLL is calculated as
\f[
 \sum_\mathrm{data} -\log( \mathrm{pdf}(x_\mathrm{data}))
\f]
In extended mode, a
\f$ N_\mathrm{expect} - N_\mathrm{observed}*log(N_\mathrm{expect}) \f$ term is added.
**/

#include "RooNLLVar.h"

#include <RooAbsData.h>
#include <RooAbsDataStore.h>
#include <RooAbsPdf.h>
#include <RooCmdConfig.h>
#include <RooDataHist.h>
#include <RooHistPdf.h>
#include <RooMsgService.h>
#include <RooNaNPacker.h>
#include <RooProdPdf.h>
#include "RooRealMPFE.h"
#include <RooRealSumPdf.h>
#include <RooRealVar.h>

#include "TMath.h"
#include "Math/Util.h"

#include <algorithm>

RooNLLVar::~RooNLLVar() {}


////////////////////////////////////////////////////////////////////////////////
/// Construct likelihood from given p.d.f and (binned or unbinned dataset)
/// For internal use.

RooNLLVar::RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& indata,
                     bool extended, RooAbsTestStatistic::Configuration const& cfg) :
  RooNLLVar{name, title, pdf, indata, RooArgSet(), extended, cfg} {}


////////////////////////////////////////////////////////////////////////////////
/// Construct likelihood from given p.d.f and (binned or unbinned dataset)
/// For internal use.

RooNLLVar::RooNLLVar(const char *name, const char *title, RooAbsPdf &pdf, RooAbsData &indata, const RooArgSet &projDeps,
                     bool extended, RooAbsTestStatistic::Configuration const &cfg)
   : RooAbsOptTestStatistic(name, title, pdf, indata, projDeps, cfg),
     _extended(extended),
     _binnedPdf(cfg.binnedL ? static_cast<RooRealSumPdf *>(_funcClone) : nullptr)
{
  // If binned likelihood flag is set, pdf is a RooRealSumPdf representing a yield vector
  // for a binned likelihood calculation

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

    _skipZeroWeights = false;
  } else {
    _skipZeroWeights = true;
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


  // If pdf is marked as binned - do a binned likelihood calculation here (sum of log-Poisson for each bin)
  if (_binnedPdf) {
    ROOT::Math::KahanSum<double> sumWeightKahanSum{0.0};
    for (auto i=firstEvent ; i<lastEvent ; i+=stepSize) {

      _dataClone->get(i) ;

      double eventWeight = _dataClone->weight();


      // Calculate log(Poisson(N|mu) for this bin
      double N = eventWeight ;
      double mu = _binnedPdf->getVal()*_binw[i] ;
      //cout << "RooNLLVar::binnedL(" << GetName() << ") N=" << N << " mu = " << mu << std::endl ;

      if (mu<=0 && N>0) {

        // Catch error condition: data present where zero events are predicted
        logEvalError(Form("Observed %f events in bin %lu with zero event yield",N,(unsigned long)i)) ;

      } else if (std::abs(mu)<1e-10 && std::abs(N)<1e-10) {

        // Special handling of this case since log(Poisson(0,0)=0 but can't be calculated with usual log-formula
        // since log(mu)=0. No update of result is required since term=0.

      } else {

        double term = 0.0;
        if(_doBinOffset) {
          term -= -mu + N + N * (std::log(mu) - std::log(N));
        } else {
          term -= -mu + N * std::log(mu) - TMath::LnGamma(N+1);
        }
        result += term;
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
  // If we do bin-by bin offsetting, we don't do this because it cancels out
  if (!_doBinOffset && _simCount>1) {
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
  return computeScalarFunc(pdfClone, _dataClone, _normSet, _weightSq, stepSize, firstEvent, lastEvent, _offsetPdf.get());
}

RooNLLVar::ComputeResult RooNLLVar::computeScalarFunc(const RooAbsPdf *pdfClone, RooAbsData *dataClone,
                                                      RooArgSet *normSet, bool weightSq, std::size_t stepSize,
                                                      std::size_t firstEvent, std::size_t lastEvent, RooAbsPdf const* offsetPdf)
{
  ROOT::Math::KahanSum<double> kahanWeight;
  ROOT::Math::KahanSum<double> kahanProb;
  RooNaNPacker packedNaN(0.f);

  for (auto i=firstEvent; i<lastEvent; i+=stepSize) {
    dataClone->get(i) ;

    double weight = dataClone->weight(); //FIXME

    if (0. == weight * weight) continue ;
    if (weightSq) weight = dataClone->weightSquared() ;

    double logProba = pdfClone->getLogVal(normSet);

    if(offsetPdf) {
      logProba -= offsetPdf->getLogVal(normSet);
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

bool RooNLLVar::setDataSlave(RooAbsData &indata, bool cloneData, bool ownNewData)
{
   bool ret = RooAbsOptTestStatistic::setDataSlave(indata, cloneData, ownNewData);
   // To re-create the data template pdf if necessary
   _offsetPdf.reset();
   enableBinOffsetting(_doBinOffset);
   return ret;
}

void RooNLLVar::enableBinOffsetting(bool flag)
{
   if (!_init) {
      initialize();
   }

   _doBinOffset = flag;

   // If this is a "master" that delegates the actual work to "slaves", the
   // _offsetPdf will not be reset.
   bool needsResetting = true;

   switch (operMode()) {
   case Slave: break;
   case SimMaster: {
      for (auto &gof : _gofArray) {
         static_cast<RooNLLVar &>(*gof).enableBinOffsetting(flag);
      }
      needsResetting = false;
      break;
   }
   case MPMaster: {
      for (int i = 0; i < _nCPU; ++i) {
         static_cast<RooNLLVar &>(_mpfeArray[i]->arg()).enableBinOffsetting(flag);
      }
      needsResetting = false;
      break;
   }
   }

   if (!needsResetting)
      return;

   if (flag && !_offsetPdf) {
      std::string name = std::string{GetName()} + "_offsetPdf";
      std::unique_ptr<RooDataHist> dataTemplate;
      if (auto dh = dynamic_cast<RooDataHist *>(_dataClone)) {
         dataTemplate = std::make_unique<RooDataHist>(*dh);
      } else {
         dataTemplate = std::unique_ptr<RooDataHist>(static_cast<RooDataSet const &>(*_dataClone).binnedClone());
      }
      _offsetPdf = std::make_unique<RooHistPdf>(name.c_str(), name.c_str(), *_funcObsSet, std::move(dataTemplate));
      _offsetPdf->setOperMode(ADirty);
   }
   setValueDirty();
}
