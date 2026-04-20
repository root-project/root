/// \cond ROOFIT_INTERNAL

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

#include "RooChi2Var.h"

#include "FitHelpers.h"
#include "RooDataHist.h"
#include "RooAbsPdf.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"

#include "Riostream.h"
#include "TClass.h"

#include "RooRealVar.h"
#include "RooAbsDataStore.h"

#include <ROOT/StringUtils.hxx>

RooChi2Var::RooChi2Var(const char *name, const char *title, RooAbsReal &func, RooDataHist &data, bool extended,
                       RooDataHist::ErrorType etype, RooAbsTestStatistic::Configuration const &cfg)
   : RooAbsOptTestStatistic(name, title, func, data, RooArgSet{}, cfg),
     _etype{etype == RooAbsData::Auto ? (data.isNonPoissonWeighted() ? RooAbsData::SumW2 : RooAbsData::Expected)
                                      : etype},
     _funcMode{dynamic_cast<RooAbsPdf *>(&func) ? (extended ? ExtendedPdf : Pdf) : Function}
{
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooChi2Var::RooChi2Var(const RooChi2Var& other, const char* name) :
  RooAbsOptTestStatistic(other,name),
  _etype(other._etype),
  _funcMode(other._funcMode)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate chi^2 in partition from firstEvent to lastEvent using given stepSize
/// Throughout the calculation, we use Kahan's algorithm for summing to
/// prevent loss of precision - this is a factor four more expensive than
/// straight addition, but since evaluating the PDF is usually much more
/// expensive than that, we tolerate the additional cost...

double RooChi2Var::evaluatePartition(std::size_t firstEvent, std::size_t lastEvent, std::size_t stepSize) const
{
  double result(0);
  double carry(0);

  // Also consider the composite case of multiple ranges
  std::vector<std::string> rangeTokens;
  if (!_rangeName.empty()) {
    rangeTokens = ROOT::Split(_rangeName, ",");
  }

  // Determine normalization factor depending on type of input function
  double normFactor(1) ;
  switch (_funcMode) {
  case Function: normFactor=1 ; break ;
  case Pdf: normFactor = _dataClone->sumEntries() ; break ;
  case ExtendedPdf: normFactor = (static_cast<RooAbsPdf*>(_funcClone))->expectedEvents(_dataClone->get()) ; break ;
  }

  // Loop over bins of dataset
  RooDataHist* hdata = static_cast<RooDataHist*>(_dataClone) ;
  for (auto i=firstEvent ; i<lastEvent ; i+=stepSize) {

    // get the data values for this event
    RooArgSet const *row = hdata->get(i);

    // Skip bins that are outside of the selected range
    bool doSelect(true) ;
    if (!_rangeName.empty()) {
      doSelect = false;
      // A row is selected if it is inside at least one complete named range.
      for (const auto &rangeName : rangeTokens) {
        bool inThisRange = true;
        for (const auto arg : *row) {
          if (!arg->inRange(rangeName.c_str())) {
            inThisRange = false;
            break;
          }
        }
        if (inThisRange) {
          doSelect = true;
          break;
        }
      }
    }
    if (!doSelect) continue ;

    const double nData = hdata->weight(i) ;

    const double nPdf = _funcClone->getVal(_normSet) * normFactor * hdata->binVolume(i) ;

    const double eExt = nPdf-nData ;


    double eInt ;
    if (_etype != RooAbsData::Expected) {
       double eIntLo;
       double eIntHi;
       hdata->weightError(eIntLo, eIntHi, _etype);
       eInt = (eExt > 0) ? eIntHi : eIntLo;
    } else {
      eInt = sqrt(nPdf) ;
    }

    // Skip cases where pdf=0 and there is no data
    if (0. == eInt * eInt && 0. == nData * nData && 0. == nPdf * nPdf) continue ;

    // Return 0 if eInt=0, special handling in MINUIT will follow
    if (0. == eInt * eInt) {
      coutE(Eval) << "RooChi2Var::RooChi2Var(" << GetName() << ") INFINITY ERROR: bin " << i
        << " has zero error" << std::endl;
      return 0.;
    }

//     std::cout << "Chi2Var[" << i << "] nData = " << nData << " nPdf = " << nPdf << " errorExt = " << eExt << " errorInt = " << eInt << " contrib = " << eExt*eExt/(eInt*eInt) << std::endl ;

    double term = eExt*eExt/(eInt*eInt) ;
    double y = term - carry;
    double t = result + y;
    carry = (t - result) - y;
    result = t;
  }

  _evalCarry = carry;
  return result ;
}

/// \endcond
