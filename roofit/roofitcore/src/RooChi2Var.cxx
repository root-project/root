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
/** \class RooChi2Var
    \ingroup Roofitcore
    \brief Simple \f$ \chi^2 \f$ calculation from a binned dataset and a PDF.
 *
 * It calculates:
 *
 \f{align*}{
   \chi^2 &= \sum_{\mathrm{bins}}  \left( \frac{N_\mathrm{PDF,bin} - N_\mathrm{Data,bin}}{\Delta_\mathrm{bin}} \right)^2 \\
   N_\mathrm{PDF,bin} &=
     \begin{cases}
         \mathrm{pdf}(\text{bin centre}) \cdot V_\mathrm{bin} \cdot N_\mathrm{Data,tot}  &\text{normal PDF}\\
         \mathrm{pdf}(\text{bin centre}) \cdot V_\mathrm{bin} \cdot N_\mathrm{Data,expected} &\text{extended PDF}
     \end{cases} \\
   \Delta_\mathrm{bin} &=
     \begin{cases}
         \sqrt{N_\mathrm{PDF,bin}} &\text{if } \mathtt{DataError == RooAbsData::Expected}\\
         \mathtt{data{\rightarrow}weightError()} &\text{otherwise} \\
     \end{cases}
 \f}
 * If the dataset doesn't have user-defined errors, errors are assumed to be \f$ \sqrt{N} \f$.
 * In extended PDF mode, N_tot (total number of data events) is substituted with N_expected, the
 * expected number of events that the PDF predicts.
 *
 * \note If the dataset has errors stored, empty bins will prevent the calculation of \f$ \chi^2 \f$, because those have
 * zero error. This leads to messages like:
 * ```
 * [#0] ERROR:Eval -- RooChi2Var::RooChi2Var(chi2_GenPdf_data_hist) INFINITY ERROR: bin 2 has zero error
 * ```
 *
 * \note In this case, one can use the expected errors of the PDF instead of the data errors:
 * ```{.cpp}
 * RooChi2Var chi2(..., ..., RooFit::DataError(RooAbsData::Expected), ...);
 * ```
 */

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
    hdata->get(i);

    const double nData = hdata->weight() ;

    const double nPdf = _funcClone->getVal(_normSet) * normFactor * hdata->binVolume() ;

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

//     cout << "Chi2Var[" << i << "] nData = " << nData << " nPdf = " << nPdf << " errorExt = " << eExt << " errorInt = " << eInt << " contrib = " << eExt*eExt/(eInt*eInt) << endl ;

    double term = eExt*eExt/(eInt*eInt) ;
    double y = term - carry;
    double t = result + y;
    carry = (t - result) - y;
    result = t;
  }

  _evalCarry = carry;
  return result ;
}
