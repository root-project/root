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
    \brief RooChi2Var implements a simple \f$ \chi^2 \f$ calculation from a binned dataset and a PDF.
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
#include "RooDataHist.h"
#include "RooAbsPdf.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"

#include "Riostream.h"
#include "TClass.h"

#include "RooRealVar.h"
#include "RooAbsDataStore.h"


using namespace std;

namespace {
  template<class ...Args>
  RooAbsTestStatistic::Configuration makeRooAbsTestStatisticCfgForFunc(Args const& ... args) {
    RooAbsTestStatistic::Configuration cfg;
    cfg.rangeName = RooCmdConfig::decodeStringOnTheFly("RooChi2Var::RooChi2Var","RangeWithName",0,"",args...);
    cfg.nCPU = RooCmdConfig::decodeIntOnTheFly("RooChi2Var::RooChi2Var","NumCPU",0,1,args...);
    cfg.interleave = RooFit::Interleave;
    cfg.verbose = static_cast<bool>(RooCmdConfig::decodeIntOnTheFly("RooChi2Var::RooChi2Var","Verbose",0,1,args...));
    cfg.cloneInputData = false;
    cfg.integrateOverBinsPrecision = RooCmdConfig::decodeDoubleOnTheFly("RooChi2Var::RooChi2Var", "IntegrateBins", 0, -1., {args...});
    return cfg;
  }

  template<class ...Args>
  RooAbsTestStatistic::Configuration makeRooAbsTestStatisticCfgForPdf(Args const& ... args) {
    auto cfg = makeRooAbsTestStatisticCfgForFunc(args...);
    cfg.addCoefRangeName = RooCmdConfig::decodeStringOnTheFly("RooChi2Var::RooChi2Var","AddCoefRange",0,"",args...);
    cfg.splitCutRange = static_cast<bool>(RooCmdConfig::decodeIntOnTheFly("RooChi2Var::RooChi2Var","SplitRange",0,0,args...));
    return cfg;
  }
}

ClassImp(RooChi2Var);
;

RooArgSet RooChi2Var::_emptySet ;


////////////////////////////////////////////////////////////////////////////////
///  RooChi2Var constructor. Optional arguments are:
///  \param[in] name Name of the PDF
///  \param[in] title Title for plotting etc.
///  \param[in] func  Function
///  \param[in] hdata Data histogram
///  \param[in] arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9 Optional arguments according to table below.
///  <table>
///  <tr><th> Argument  <th> Effect
///  <tr><td>
///  DataError()  <td> Choose between Poisson errors and Sum-of-weights errors
///  <tr><td>
///  NumCPU()     <td> Activate parallel processing feature
///  <tr><td>
///  Range()      <td> Fit only selected region
///  <tr><td>
///  Verbose()    <td> Verbose output of GOF framework
///  <tr><td>
///  IntegrateBins()  <td> Integrate PDF within each bin. This sets the desired precision. Only useful for binned fits.

RooChi2Var::RooChi2Var(const char *name, const char* title, RooAbsReal& func, RooDataHist& hdata,
             const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3,
             const RooCmdArg& arg4,const RooCmdArg& arg5,const RooCmdArg& arg6,
             const RooCmdArg& arg7,const RooCmdArg& arg8,const RooCmdArg& arg9) :
  RooAbsOptTestStatistic(name,title,func,hdata,_emptySet,
          makeRooAbsTestStatisticCfgForFunc(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9))
{
  RooCmdConfig pc("RooChi2Var::RooChi2Var") ;
  pc.defineInt("etype","DataError",0,(Int_t)RooDataHist::Auto) ;
  pc.defineInt("extended","Extended",0,false) ;
  pc.allowUndefined() ;

  pc.process(arg1) ;  pc.process(arg2) ;  pc.process(arg3) ;
  pc.process(arg4) ;  pc.process(arg5) ;  pc.process(arg6) ;
  pc.process(arg7) ;  pc.process(arg8) ;  pc.process(arg9) ;

  if (func.IsA()->InheritsFrom(RooAbsPdf::Class())) {
    _funcMode = pc.getInt("extended") ? ExtendedPdf : Pdf ;
  } else {
    _funcMode = Function ;
  }
  _etype = (RooDataHist::ErrorType) pc.getInt("etype") ;

  if (_etype==RooAbsData::Auto) {
    _etype = hdata.isNonPoissonWeighted()? RooAbsData::SumW2 : RooAbsData::Expected ;
  }

}


////////////////////////////////////////////////////////////////////////////////
///  RooChi2Var constructor. Optional arguments taken
///
///  \param[in] name Name of the PDF
///  \param[in] title Title for plotting etc.
///  \param[in] pdf  PDF to fit
///  \param[in] hdata Data histogram
///  \param[in] arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9 Optional arguments according to table below.
///  <table>
///  <tr><th> Argument  <th> Effect
///  <tr><td>
///  Extended()   <td> Include extended term in calculation
///  <tr><td>
///  DataError()  <td> Choose between Poisson errors and Sum-of-weights errors
///  <tr><td>
///  NumCPU()     <td> Activate parallel processing feature
///  <tr><td>
///  Range()      <td> Fit only selected region
///  <tr><td>
///  SumCoefRange() <td> Set the range in which to interpret the coefficients of RooAddPdf components
///  <tr><td>
///  SplitRange() <td> Fit range is split by index category of simultaneous PDF
///  <tr><td>
///  ConditionalObservables() <td> Define projected observables
///  <tr><td>
///  Verbose()    <td> Verbose output of GOF framework
///  <tr><td>
///  IntegrateBins()  <td> Integrate PDF within each bin. This sets the desired precision.

RooChi2Var::RooChi2Var(const char *name, const char* title, RooAbsPdf& pdf, RooDataHist& hdata,
             const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3,
             const RooCmdArg& arg4,const RooCmdArg& arg5,const RooCmdArg& arg6,
             const RooCmdArg& arg7,const RooCmdArg& arg8,const RooCmdArg& arg9) :
  RooAbsOptTestStatistic(name,title,pdf,hdata,
                         *static_cast<const RooArgSet*>(RooCmdConfig::decodeObjOnTheFly("RooChi2Var::RooChi2Var","ProjectedObservables",0,&_emptySet,
                                 arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9)),
                         makeRooAbsTestStatisticCfgForPdf(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9))
{
  RooCmdConfig pc("RooChi2Var::RooChi2Var") ;
  pc.defineInt("extended","Extended",0,false) ;
  pc.defineInt("etype","DataError",0,(Int_t)RooDataHist::Auto) ;
  pc.allowUndefined() ;

  pc.process(arg1) ;  pc.process(arg2) ;  pc.process(arg3) ;
  pc.process(arg4) ;  pc.process(arg5) ;  pc.process(arg6) ;
  pc.process(arg7) ;  pc.process(arg8) ;  pc.process(arg9) ;

  _funcMode = pc.getInt("extended") ? ExtendedPdf : Pdf ;
  _etype = (RooDataHist::ErrorType) pc.getInt("etype") ;
  if (_etype==RooAbsData::Auto) {
    _etype = hdata.isNonPoissonWeighted()? RooAbsData::SumW2 : RooAbsData::Expected ;
  }
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

  double result(0), carry(0);

  _dataClone->store()->recalculateCache( _projDeps, firstEvent, lastEvent, stepSize, false) ;


  // Determine normalization factor depending on type of input function
  double normFactor(1) ;
  switch (_funcMode) {
  case Function: normFactor=1 ; break ;
  case Pdf: normFactor = _dataClone->sumEntries() ; break ;
  case ExtendedPdf: normFactor = ((RooAbsPdf*)_funcClone)->expectedEvents(_dataClone->get()) ; break ;
  }

  // Loop over bins of dataset
  RooDataHist* hdata = (RooDataHist*) _dataClone ;
  for (auto i=firstEvent ; i<lastEvent ; i+=stepSize) {

    // get the data values for this event
    hdata->get(i);

    const double nData = hdata->weight() ;

    const double nPdf = _funcClone->getVal(_normSet) * normFactor * hdata->binVolume() ;

    const double eExt = nPdf-nData ;


    double eInt ;
    if (_etype != RooAbsData::Expected) {
      double eIntLo,eIntHi ;
      hdata->weightError(eIntLo,eIntHi,_etype) ;
      eInt = (eExt>0) ? eIntHi : eIntLo ;
    } else {
      eInt = sqrt(nPdf) ;
    }

    // Skip cases where pdf=0 and there is no data
    if (0. == eInt * eInt && 0. == nData * nData && 0. == nPdf * nPdf) continue ;

    // Return 0 if eInt=0, special handling in MINUIT will follow
    if (0. == eInt * eInt) {
      coutE(Eval) << "RooChi2Var::RooChi2Var(" << GetName() << ") INFINITY ERROR: bin " << i
        << " has zero error" << endl ;
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
