/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara,   verkerke@slac.stanford.edu     *
 *   DK, David Kirkby,    UC Irvine,          dkirkby@uci.edu                *
 *   AL, Alfio Lazzaro,   INFN Milan,         alfio.lazzaro@mi.infn.it       *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *   VC, Vince Croft,     DIANA / NYU,        vincent.croft@cern.ch          *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooGradMinimizer.cxx
\class RooGradMinimizer
\ingroup Roofitcore

RooGradMinimizer is a wrapper class around ROOT::Fit:Fitter that
provides a seamless interface between the minimizer functionality
and the native RooFit interface.
It is based on the RooMinimizer class, but extends it by extracting
the numerical gradient functionality from Minuit2. This allows us to
schedule parallel calculation of gradient components.
**/

#include "RooFit.h"
#include "Riostream.h"

#include "TClass.h"

#include <fstream>
#include <iomanip>

#include "TH1.h"
#include "TH2.h"
#include "TMarker.h"
#include "TGraph.h"
#include "Fit/FitConfig.h"
#include "TStopwatch.h"
#include "TDirectory.h"
#include "TMatrixDSym.h"

#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooAbsReal.h"
#include "RooAbsRealLValue.h"
#include "RooRealVar.h"
#include "RooAbsPdf.h"
#include "RooSentinel.h"
#include "RooMsgService.h"
#include "RooPlot.h"

#include "RooMinimizer.h"
#include "RooGradMinimizer.h"
#include "RooGradMinimizerFcn.h"
#include "RooFitResult.h"

#include "Math/Minimizer.h"

#if (__GNUC__==3&&__GNUC_MINOR__==2&&__GNUC_PATCHLEVEL__==3)
char* operator+( streampos&, char* );
#endif

using namespace std;

ROOT::Fit::Fitter *RooGradMinimizer::_theFitter = 0 ;



////////////////////////////////////////////////////////////////////////////////
/// Cleanup method called by atexit handler installed by RooSentinel
/// to delete all global heap objects when the program is terminated

void RooGradMinimizer::cleanup()
{
  if (_theFitter) {
    delete _theFitter ;
    _theFitter =0 ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Construct MINUIT interface to given function. Function can be anything,
/// but is typically a -log(likelihood) implemented by RooNLLVar or a chi^2
/// (implemented by RooChi2Var). Other frequent use cases are a RooAddition
/// of a RooNLLVar plus a penalty or constraint term. This class propagates
/// all RooFit information (floating parameters, their values and errors)
/// to MINUIT before each MINUIT call and propagates all MINUIT information
/// back to the RooFit object at the end of each call (updated parameter
/// values, their (asymmetric errors) etc. The default MINUIT error level
/// for HESSE and MINOS error analysis is taken from the defaultErrorLevel()
/// value of the input function.

RooGradMinimizer::RooGradMinimizer(RooAbsReal& function)
{
  RooSentinel::activate() ;

  // Store function reference
  _extV = 0 ;
  _func = &function ;
  _optConst = kFALSE ;
  _verbose = kFALSE ;
  _profile = kFALSE ;
  _profileStart = kFALSE ;
  _printLevel = 1 ;
  _minimizerType = "Minuit"; // default minimizer

  if (_theFitter) delete _theFitter ;
  _theFitter = new ROOT::Fit::Fitter;
  //  RooMinimizer *theRooMinimizer = dynamic_cast<RooMinimizer*>(this);
  _fcn = new RooGradMinimizerFcn(_func,this,_verbose);
  _theFitter->Config().SetMinimizer(_minimizerType.c_str());
  // default max number of calls
  _theFitter->Config().MinimizerOptions().SetMaxIterations(500*_fcn->NDim());
  _theFitter->Config().MinimizerOptions().SetMaxFunctionCalls(500*_fcn->NDim());

  // Declare our parameters to MINUIT
  _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
		    _optConst,_verbose) ;

  // Send the parameters to the gradient calculator as well
  _fcn->SynchronizeGradient(_theFitter->Config().ParamsSettings());
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooGradMinimizer::~RooGradMinimizer()
{
  if (_extV) {
    delete _extV ;
  }

  if (_fcn) {
    delete _fcn;
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Change MINUIT strategy to istrat. Accepted codes
/// are 0,1,2 and represent MINUIT strategies for dealing
/// most efficiently with fast FCNs (0), expensive FCNs (2)
/// and 'intermediate' FCNs (1)
Int_t RooGradMinimizer::migrad()
{
  _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
		    _optConst,_verbose) ;
  //  profileStart() ;
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
  RooAbsReal::clearEvalErrorLog() ;

  _theFitter->Config().SetMinimizer(_minimizerType.c_str(),"migrad");
  bool ret = _theFitter->FitFCN(*_fcn);
  _status = ((ret) ? _theFitter->Result().Status() : -1);

  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
  //  profileStop() ;
  _fcn->BackProp(_theFitter->Result());

  saveStatus("MIGRAD",_status) ;

  return _status ;
}


////////////////////////////////////////////////////////////////////////////////
/// If flag is true, perform constant term optimization on
/// function being minimized.

void RooGradMinimizer::optimizeConst(Int_t flag)
{
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;

  if (_optConst && !flag){
    if (_printLevel>-1) coutI(Minimization) << "RooGradMinimizer::optimizeConst: deactivating const optimization" << endl ;
    _func->constOptimizeTestStatistic(RooAbsArg::DeActivate) ;
    _optConst = flag ;
  } else if (!_optConst && flag) {
    if (_printLevel>-1) coutI(Minimization) << "RooGradMinimizer::optimizeConst: activating const optimization" << endl ;
    _func->constOptimizeTestStatistic(RooAbsArg::Activate,flag>1) ;
    _optConst = flag ;
  } else if (_optConst && flag) {
    if (_printLevel>-1) coutI(Minimization) << "RooGradMinimizer::optimizeConst: const optimization already active" << endl ;
  } else {
    if (_printLevel>-1) coutI(Minimization) << "RooGradMinimizer::optimizeConst: const optimization wasn't active" << endl ;
  }

  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;

}


////////////////////////////////////////////////////////////////////////////////
/// Return underlying ROOT fitter object

ROOT::Fit::Fitter* RooMinimizer::fitter()
{
  return _theFitter ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return underlying ROOT fitter object

const ROOT::Fit::Fitter* RooMinimizer::fitter() const
{
  return _theFitter ;
}



