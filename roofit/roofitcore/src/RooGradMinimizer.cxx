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

//#include "RooMinimizer.h"
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

RooGradMinimizer::RooGradMinimizer(RooAbsReal& function, bool always_exactly_mimic_minuit2) {
  RooSentinel::activate() ;

  // Store function reference
  _func = &function ;

  if (_theFitter) delete _theFitter ;
  _theFitter = new ROOT::Fit::Fitter;
  _theFitter->Config().SetMinimizer(_minimizerType.c_str());
  setEps(1.0); // default tolerance

  _fcn = new RooGradMinimizerFcn(_func, this,
                                 (always_exactly_mimic_minuit2 ?
                                  RooGradientFunction::GradientCalculatorMode::ExactlyMinuit2 :
                                  RooGradientFunction::GradientCalculatorMode::AlmostMinuit2
                                 ),
                                 _verbose);

  // default max number of calls
  _theFitter->Config().MinimizerOptions().SetMaxIterations(500*_fcn->NDim());
  _theFitter->Config().MinimizerOptions().SetMaxFunctionCalls(500*_fcn->NDim());

  // Use +0.5 for 1-sigma errors
  setErrorLevel(_func->defaultErrorLevel()) ;

  // Declare our parameters to MINUIT
  _fcn->synchronize_parameter_settings(_theFitter->Config().ParamsSettings(),
                                       _optConst, _verbose);
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

void RooGradMinimizer::setStrategy(Int_t istrat) {
  _theFitter->Config().MinimizerOptions().SetStrategy(istrat);
  _fcn->set_strategy(static_cast<int>(istrat));
}


////////////////////////////////////////////////////////////////////////////////
/// Execute MIGRAD. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation

Int_t RooGradMinimizer::migrad()
{
  _fcn->synchronize_parameter_settings(_theFitter->Config().ParamsSettings(),
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
/// Execute HESSE. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation

Int_t RooGradMinimizer::hesse()
{
  if (_theFitter->GetMinimizer()==0) {
    coutW(Minimization) << "RooGaussMinimizer::hesse: Error, run Migrad before Hesse!"
                        << endl ;
    _status = -1;
  }
  else {

    _fcn->synchronize_parameter_settings(_theFitter->Config().ParamsSettings(),
                      _optConst,_verbose) ;
    //    profileStart() ;
    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
    RooAbsReal::clearEvalErrorLog() ;

    _theFitter->Config().SetMinimizer(_minimizerType.c_str());
    bool ret = _theFitter->CalculateHessErrors();
    _status = ((ret) ? _theFitter->Result().Status() : -1);

    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
    //    profileStop() ;
    _fcn->BackProp(_theFitter->Result());

    saveStatus("HESSE",_status) ;

  }

  return _status ;

}

////////////////////////////////////////////////////////////////////////////////
/// Execute MINOS. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation

Int_t RooGradMinimizer::minos()
{
  if (_theFitter->GetMinimizer()==0) {
    coutW(Minimization) << "RooGaussMinimizer::minos: Error, run Migrad before Minos!"
                        << endl ;
    _status = -1;
  }
  else {

    _fcn->synchronize_parameter_settings(_theFitter->Config().ParamsSettings(),
                      _optConst,_verbose) ;
    //    profileStart() ;
    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
    RooAbsReal::clearEvalErrorLog() ;

    _theFitter->Config().SetMinimizer(_minimizerType.c_str());
    bool ret = _theFitter->CalculateMinosErrors();
    _status = ((ret) ? _theFitter->Result().Status() : -1);

    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
    //    profileStop() ;
    _fcn->BackProp(_theFitter->Result());

    saveStatus("MINOS",_status) ;

  }

  return _status ;

}

////////////////////////////////////////////////////////////////////////////////
/// Execute MINOS for given list of parameters. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation

Int_t RooGradMinimizer::minos(const RooArgSet& minosParamList)
{
  if (_theFitter->GetMinimizer()==0) {
    coutW(Minimization) << "RooMinimizer::minos: Error, run Migrad before Minos!"
                        << endl ;
    _status = -1;
  }
  else if (minosParamList.getSize()>0) {

    _fcn->synchronize_parameter_settings(_theFitter->Config().ParamsSettings(),
                      _optConst,_verbose) ;
    profileStart() ;
    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
    RooAbsReal::clearEvalErrorLog() ;

    // get list of parameters for Minos
    TIterator* aIter = minosParamList.createIterator() ;
    RooAbsArg* arg ;
    std::vector<unsigned int> paramInd;
    while((arg=(RooAbsArg*)aIter->Next())) {
      RooAbsArg* par = _fcn->GetFloatParamList()->find(arg->GetName());
      if (par && !par->isConstant()) {
        Int_t index = _fcn->GetFloatParamList()->index(par);
        paramInd.push_back(index);
      }
    }
    delete aIter ;

    if (paramInd.size()) {
      // set the parameter indeces
      _theFitter->Config().SetMinosErrors(paramInd);

      _theFitter->Config().SetMinimizer(_minimizerType.c_str());
      bool ret = _theFitter->CalculateMinosErrors();
      _status = ((ret) ? _theFitter->Result().Status() : -1);
      // to avoid that following minimization computes automatically the Minos errors
      _theFitter->Config().SetMinosErrors(kFALSE);

    }

    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
    profileStop() ;
    _fcn->BackProp(_theFitter->Result());

    saveStatus("MINOS",_status) ;

  }

  return _status ;
}


////////////////////////////////////////////////////////////////////////////////
/// Change the MINUIT internal printing level

Int_t RooGradMinimizer::setPrintLevel(Int_t newLevel)
{
  Int_t ret = _printLevel ;
  _theFitter->Config().MinimizerOptions().SetPrintLevel(newLevel+1);
  _printLevel = newLevel+1 ;
  return ret ;
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
/// Start profiling timer

void RooGradMinimizer::profileStart()
{
  if (_profile) {
    _timer.Start() ;
    _cumulTimer.Start(_profileStart?kFALSE:kTRUE) ;
    _profileStart = kTRUE ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Stop profiling timer and report results of last session

void RooGradMinimizer::profileStop()
{
  if (_profile) {
    _timer.Stop() ;
    _cumulTimer.Stop() ;
    coutI(Minimization) << "Command timer: " ; _timer.Print() ;
    coutI(Minimization) << "Session timer: " ; _cumulTimer.Print() ;
  }
}


////////////////////////////////////////////////////////////////////////////////

Int_t RooGradMinimizer::evalCounter() const {
  return fitterFcn()->evalCounter() ;
}

////////////////////////////////////////////////////////////////////////////////

void RooGradMinimizer::zeroEvalCount() {
  fitterFcn()->zeroEvalCount() ;
}

////////////////////////////////////////////////////////////////////////////////


inline Int_t RooGradMinimizer::getNPar() const { return fitterFcn()->NDim() ; }
inline Double_t& RooGradMinimizer::maxFCN() { return fitterFcn()->GetMaxFCN() ; }


const RooGradMinimizerFcn* RooGradMinimizer::fitterFcn() const {  return ( fitter()->GetFCN() ? (dynamic_cast<RooGradMinimizerFcn*>(fitter()->GetFCN())) : _fcn ) ; }
RooGradMinimizerFcn* RooGradMinimizer::fitterFcn() { return ( fitter()->GetFCN() ? (dynamic_cast<RooGradMinimizerFcn*>(fitter()->GetFCN())) : _fcn ) ; }


////////////////////////////////////////////////////////////////////////////////
/// Set the level for MINUIT error analysis to the given
/// value. This function overrides the default value
/// that is taken in the RooMinimizer constructor from
/// the defaultErrorLevel() method of the input function

void RooGradMinimizer::setErrorLevel(Double_t level)
{
  _theFitter->Config().MinimizerOptions().SetErrorDef(level);
  _fcn->set_error_level(level);
}

////////////////////////////////////////////////////////////////////////////////
/// Change MINUIT epsilon

void RooGradMinimizer::setEps(Double_t eps)
{
  _theFitter->Config().MinimizerOptions().SetTolerance(eps);
}

////////////////////////////////////////////////////////////////////////////////
/// Choose the minimizer algorithm.

void RooGradMinimizer::setMinimizerType(const char* type) {
  if (strcmp(type, "Minuit2") != 0) {
    throw std::invalid_argument("In RooGradMinimizer::setMinimizerType: only Minuit2 is supported in RooGradMinimizer!");
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Return underlying ROOT fitter object

ROOT::Fit::Fitter* RooGradMinimizer::fitter()
{
  return _theFitter ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return underlying ROOT fitter object

const ROOT::Fit::Fitter* RooGradMinimizer::fitter() const
{
  return _theFitter ;
}

////////////////////////////////////////////////////////////////////////////////

void RooGradMinimizer::setVerbose(Bool_t flag) {
  _verbose = flag;
  fitterFcn()->SetVerbose(flag);
}


RooFitResult* RooGradMinimizer::lastMinuitFit(const RooArgList& varList)
{
  // Import the results of the last fit performed, interpreting
  // the fit parameters as the given varList of parameters.

  if (_theFitter==0 || _theFitter->GetMinimizer()==0) {
    oocoutE((TObject*)0,InputArguments) << "RooMinimizer::save: Error, run minimization before!"
                                        << endl ;
    return 0;
  }

  // Verify length of supplied varList
  if (varList.getSize()>0 && varList.getSize()!=Int_t(_theFitter->Result().NTotalParameters())) {
    oocoutE((TObject*)0,InputArguments)
        << "RooMinimizer::lastMinuitFit: ERROR: supplied variable list must be either empty " << endl
        << "                             or match the number of variables of the last fit ("
        << _theFitter->Result().NTotalParameters() << ")" << endl ;
    return 0 ;
  }


  // Verify that all members of varList are of type RooRealVar
  TIterator* iter = varList.createIterator() ;
  RooAbsArg* arg  ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dynamic_cast<RooRealVar*>(arg)) {
      oocoutE((TObject*)0,InputArguments) << "RooMinimizer::lastMinuitFit: ERROR: variable '"
                                          << arg->GetName() << "' is not of type RooRealVar" << endl ;
      return 0 ;
    }
  }
  delete iter ;

  RooFitResult* res = new RooFitResult("lastMinuitFit","Last MINUIT fit") ;

  // Extract names of fit parameters
  // and construct corresponding RooRealVars
  RooArgList constPars("constPars") ;
  RooArgList floatPars("floatPars") ;

  UInt_t i ;
  for (i = 0; i < _theFitter->Result().NTotalParameters(); ++i) {

    TString varName(_theFitter->Result().GetParameterName(i));
    Bool_t isConst(_theFitter->Result().IsParameterFixed(i)) ;

    Double_t xlo = _theFitter->Config().ParSettings(i).LowerLimit();
    Double_t xhi = _theFitter->Config().ParSettings(i).UpperLimit();
    Double_t xerr = _theFitter->Result().Error(i);
    Double_t xval = _theFitter->Result().Value(i);

    RooRealVar* var ;
    if (varList.getSize()==0) {

      if ((xlo<xhi) && !isConst) {
        var = new RooRealVar(varName,varName,xval,xlo,xhi) ;
      } else {
        var = new RooRealVar(varName,varName,xval) ;
      }
      var->setConstant(isConst) ;
    } else {

      var = (RooRealVar*) varList.at(i)->Clone() ;
      var->setConstant(isConst) ;
      var->setVal(xval) ;
      if (xlo<xhi) {
        var->setRange(xlo,xhi) ;
      }

      if (varName.CompareTo(var->GetName())) {
        oocoutI((TObject*)0,Eval)  << "RooMinimizer::lastMinuitFit: fit parameter '" << varName
                                   << "' stored in variable '" << var->GetName() << "'" << endl ;
      }

    }

    if (isConst) {
      constPars.addOwned(*var) ;
    } else {
      var->setError(xerr) ;
      floatPars.addOwned(*var) ;
    }
  }

  res->setConstParList(constPars) ;
  res->setInitParList(floatPars) ;
  res->setFinalParList(floatPars) ;
  res->setMinNLL(_theFitter->Result().MinFcnValue()) ;
  res->setEDM(_theFitter->Result().Edm()) ;
  res->setCovQual(_theFitter->GetMinimizer()->CovMatrixStatus()) ;
  res->setStatus(_theFitter->Result().Status()) ;
  std::vector<double> globalCC;
  TMatrixDSym corrs(_theFitter->Result().Parameters().size()) ;
  TMatrixDSym covs(_theFitter->Result().Parameters().size()) ;
  for (UInt_t ic=0; ic<_theFitter->Result().Parameters().size(); ic++) {
    globalCC.push_back(_theFitter->Result().GlobalCC(ic));
    for (UInt_t ii=0; ii<_theFitter->Result().Parameters().size(); ii++) {
      corrs(ic,ii) = _theFitter->Result().Correlation(ic,ii);
      covs(ic,ii) = _theFitter->Result().CovMatrix(ic,ii);
    }
  }
  res->fillCorrMatrix(globalCC,corrs,covs) ;

  return res;

}
