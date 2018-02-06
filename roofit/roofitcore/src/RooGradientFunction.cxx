/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *   VC, Vince Croft,     DIANA / NYU,        vincent.croft@cern.ch          *
 *   AL, Alfio Lazzaro,   INFN Milan,         alfio.lazzaro@mi.infn.it       *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// RooGradientFunction is an interface class to ROOT::Math::IMultiGradFunction
// which is necessary for Minuit2. It gives the gradient of a RooAbsReal to
// its parameters through Gradient(params, grad), where grad is the array in
// which to save the result, or the partial derivative of component i via
// Derivative(params, i), but also the RooAbsReal value itself with operator().
//
// The derivation is implemented using a replica of the Minuit2 derivator:
// the NumericalDerivatorMinuit2 class. This class has two modes: exact
// Minuit2 replication and a mode that deviates slightly, but keeps input
// parameters intact with slightly higher precision. Users of this class can
// choose between these modes using the grad_mode parameter, which is of type
// GradientCalculatorMode. In principle, this parameter can be used in the
// future to support alternative gradient calculating classes. Current values
// for this enum are ExactlyMinuit2 and AlmostMinuit2.
//
//////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "RooFit.h"

#include "Riostream.h"

#include "TIterator.h"
#include "TClass.h"

#include "RooAbsArg.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooAbsRealLValue.h"
#include "RooMsgService.h"

#include "RooGradientFunction.h"

#include <algorithm> // std::equal


RooGradientFunction::RooGradientFunction(RooAbsReal *funct, GradientCalculatorMode grad_mode, bool verbose) :
    _function(funct, verbose),
    _gradf(_function, grad_mode == GradientCalculatorMode::ExactlyMinuit2),
    _grad(_function.NDim()),
    _grad_params(_function.NDim()),
    _parameter_settings(_function.NDim()) {
  synchronize_parameter_settings(_parameter_settings);
}

RooGradientFunction::RooGradientFunction(const RooGradientFunction& other) :
    ROOT::Math::IMultiGradFunction(other),
    _function(other._function),
    _gradf(other._gradf),
    _grad(other._grad),
    _grad_params(other._grad_params),
    _parameter_settings(other._parameter_settings) {}


RooGradientFunction::Function::Function(RooAbsReal *funct, bool verbose) : _funct(funct), _verbose(verbose) {
  // Examine parameter list
  RooArgSet* paramSet = _funct->getParameters(RooArgSet());
  RooArgList paramList(*paramSet);
  delete paramSet;

  _floatParamList = (RooArgList*) paramList.selectByAttrib("Constant",kFALSE);
  if (_floatParamList->getSize()>1) {
    _floatParamList->sort();
  }
  _floatParamList->setName("floatParamList");

  _constParamList = (RooArgList*) paramList.selectByAttrib("Constant",kTRUE);
  if (_constParamList->getSize()>1) {
    _constParamList->sort();
  }
  _constParamList->setName("constParamList");

  // Remove all non-RooRealVar parameters from list (MINUIT cannot handle them)
  TIterator* pIter = _floatParamList->createIterator();
  RooAbsArg* arg;
  while ((arg=(RooAbsArg*)pIter->Next())) {
    if (!arg->IsA()->InheritsFrom(RooAbsRealLValue::Class())) {
      oocoutW(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::RooGradientFunction: removing parameter "
                        << arg->GetName()
                        << " from list because it is not of type RooRealVar" << std::endl;
      _floatParamList->remove(*arg);
    }
  }
  delete pIter;

  _nDim = _floatParamList->getSize();

  updateFloatVec();

  // Save snapshot of initial lists
  _initFloatParamList = (RooArgList*) _floatParamList->snapshot(kFALSE);
  _initConstParamList = (RooArgList*) _constParamList->snapshot(kFALSE);
}


RooGradientFunction::Function::Function(const RooGradientFunction::Function& other) :
    ROOT::Math::IMultiGenFunction(other),
    _evalCounter(other._evalCounter),
    _funct(other._funct),
    _maxFCN(other._maxFCN),
    _numBadNLL(other._numBadNLL),
    _printEvalErrors(other._printEvalErrors),
    _doEvalErrorWall(other._doEvalErrorWall),
    _nDim(other._nDim),
    _floatParamVec(other._floatParamVec),
    _verbose(other._verbose)
{
  _floatParamList = new RooArgList(*other._floatParamList);
  _constParamList = new RooArgList(*other._constParamList);
  _initFloatParamList = (RooArgList*) other._initFloatParamList->snapshot(kFALSE);
  _initConstParamList = (RooArgList*) other._initConstParamList->snapshot(kFALSE);
}


RooGradientFunction::Function::~Function() {
  delete _floatParamList;
  delete _initFloatParamList;
  delete _constParamList;
  delete _initConstParamList;
}


ROOT::Math::IMultiGenFunction* RooGradientFunction::Function::Clone() const {
  return new RooGradientFunction::Function(*this);
}
ROOT::Math::IMultiGradFunction* RooGradientFunction::Clone() const {
  return new RooGradientFunction(*this);
}

unsigned int RooGradientFunction::Function::NDim() const {
  return _nDim;
}

Bool_t RooGradientFunction::synchronize_parameter_settings(std::vector<ROOT::Fit::ParameterSettings>& parameter_settings,
                                                           Bool_t optConst, Bool_t verbose) {
  // Update parameter_settings with current information in RooAbsReal function parameters

  Bool_t constValChange(kFALSE);
  Bool_t constStatChange(kFALSE);

  Int_t index(0);

  // Handle eventual migrations from constParamList -> floatParamList
  for(index= 0; index < _function._constParamList->getSize() ; index++) {

    RooRealVar *par= dynamic_cast<RooRealVar*>(_function._constParamList->at(index));
    if (!par) continue;

    RooRealVar *oldpar= dynamic_cast<RooRealVar*>(_function._initConstParamList->at(index));
    if (!oldpar) continue;

    // Test if constness changed
    if (!par->isConstant()) {

      // Remove from constList, add to floatList
      _function._constParamList->remove(*par);
      _function._floatParamList->add(*par);
      _function._initFloatParamList->addClone(*oldpar);
      _function._initConstParamList->remove(*oldpar);
      constStatChange=kTRUE;
      _function._nDim++;

      if (verbose) {
        oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: parameter "
                          << par->GetName() << " is now floating." << std::endl ;
      }
    }

    // Test if value changed
    if (par->getVal()!= oldpar->getVal()) {
      constValChange=kTRUE;
      if (verbose) {
        oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradMinimizerFcn::synchronize: value of constant parameter "
                          << par->GetName()
                          << " changed from " << oldpar->getVal() << " to "
                          << par->getVal() << std::endl ;
      }
    }

  }

  // Update reference list
  *_function._initConstParamList = *_function._constParamList;

  // Synchronize MINUIT with function state
  // Handle floatParamList
  for(index= 0; index < _function._floatParamList->getSize(); index++) {
    RooRealVar *par= dynamic_cast<RooRealVar*>(_function._floatParamList->at(index));

    if (!par) continue;

    Double_t pstep(0);
    Double_t pmin(0);
    Double_t pmax(0);

    if(!par->isConstant()) {

      // Verify that floating parameter is indeed of type RooRealVar
      if (!par->IsA()->InheritsFrom(RooRealVar::Class())) {
        oocoutW(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::fit: Error, non-constant parameter "
                                       << par->GetName()
                                       << " is not of type RooRealVar, skipping" << std::endl;
        _function._floatParamList->remove(*par);
        index--;
        _function._nDim--;
        continue;
      }

      // Set the limits, if not infinite
      if (par->hasMin() )
        pmin = par->getMin();
      if (par->hasMax() )
        pmax = par->getMax();

      // Calculate step size
      pstep = par->getError();
      if(pstep <= 0) {
        // Floating parameter without error estitimate
        if (par->hasMin() && par->hasMax()) {
          pstep= 0.1*(pmax-pmin);

          // Trim default choice of error if within 2 sigma of limit
          if (pmax - par->getVal() < 2*pstep) {
            pstep = (pmax - par->getVal())/2;
          } else if (par->getVal() - pmin < 2*pstep) {
            pstep = (par->getVal() - pmin )/2;
          }

          // If trimming results in zero error, restore default
          if (pstep==0) {
            pstep= 0.1*(pmax-pmin);
          }

        } else {
          pstep=1;
        }
        if (verbose) {
          oocoutW(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: WARNING: no initial error estimate available for "
                            << par->GetName() << ": using " << pstep << std::endl;
        }
      }
    } else {
      pmin = par->getVal();
      pmax = par->getVal();
    }

    // new parameter
    if (index>=Int_t(parameter_settings.size())) {

      if (par->hasMin() && par->hasMax()) {
        parameter_settings.push_back(ROOT::Fit::ParameterSettings(par->GetName(),
                                                          par->getVal(),
                                                          pstep,
                                                          pmin,pmax));
      }
      else {
        parameter_settings.push_back(ROOT::Fit::ParameterSettings(par->GetName(),
                                                          par->getVal(),
                                                          pstep));
        if (par->hasMin() )
          parameter_settings.back().SetLowerLimit(pmin);
        else if (par->hasMax() )
          parameter_settings.back().SetUpperLimit(pmax);
      }

      continue;

    }

    Bool_t oldFixed = parameter_settings[index].IsFixed();
    Double_t oldVar = parameter_settings[index].Value();
    Double_t oldVerr = parameter_settings[index].StepSize();
    Double_t oldVlo = parameter_settings[index].LowerLimit();
    Double_t oldVhi = parameter_settings[index].UpperLimit();

    if (par->isConstant() && !oldFixed) {

      // Parameter changes floating -> constant : update only value if necessary
      if (oldVar!=par->getVal()) {
        parameter_settings[index].SetValue(par->getVal());
        if (verbose) {
          oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: value of parameter "
                            << par->GetName() << " changed from " << oldVar
                            << " to " << par->getVal() << std::endl ;
        }
      }
      parameter_settings[index].Fix();
      constStatChange=kTRUE;
      if (verbose) {
        oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: parameter "
                          << par->GetName() << " is now fixed." << std::endl ;
      }

    } else if (par->isConstant() && oldFixed) {

      // Parameter changes constant -> constant : update only value if necessary
      if (oldVar!=par->getVal()) {
        parameter_settings[index].SetValue(par->getVal());
        constValChange=kTRUE;

        if (verbose) {
          oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: value of fixed parameter "
                            << par->GetName() << " changed from " << oldVar
                            << " to " << par->getVal() << std::endl ;
        }
      }

    } else {
      // Parameter changes constant -> floating
      if (!par->isConstant() && oldFixed) {
        parameter_settings[index].Release();
        constStatChange=kTRUE;

        if (verbose) {
          oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: parameter "
                            << par->GetName() << " is now floating." << std::endl ;
        }
      }

      // Parameter changes constant -> floating : update all if necessary
      if (oldVar!=par->getVal() || oldVlo!=pmin || oldVhi != pmax || oldVerr!=pstep) {
        parameter_settings[index].SetValue(par->getVal());
        parameter_settings[index].SetStepSize(pstep);
        if (par->hasMin() && par->hasMax() )
          parameter_settings[index].SetLimits(pmin,pmax);
        else if (par->hasMin() )
          parameter_settings[index].SetLowerLimit(pmin);
        else if (par->hasMax() )
          parameter_settings[index].SetUpperLimit(pmax);
      }

      // Inform user about changes in verbose mode
      if (verbose) {
        // if ierr<0, par was moved from the const list and a message was already printed

        if (oldVar!=par->getVal()) {
          oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: value of parameter "
                            << par->GetName() << " changed from " << oldVar << " to "
                            << par->getVal() << std::endl ;
        }
        if (oldVlo!=pmin || oldVhi!=pmax) {
          oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: limits of parameter "
                            << par->GetName() << " changed from [" << oldVlo << "," << oldVhi
                            << "] to [" << pmin << "," << pmax << "]" << std::endl ;
        }

        // If oldVerr=0, then parameter was previously fixed
        if (oldVerr!=pstep && oldVerr!=0) {
          oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: error/step size of parameter "
                            << par->GetName() << " changed from " << oldVerr << " to " << pstep << std::endl ;
        }
      }
    }
  }

  if (optConst) {
    if (constStatChange) {

      RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors);

      oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: set of constant parameters changed, rerunning const optimizer" << std::endl;
      _function._funct->constOptimizeTestStatistic(RooAbsArg::ConfigChange);
    } else if (constValChange) {
      oocoutI(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction::synchronize: constant parameter values changed, rerunning const optimizer" << std::endl;
      _function._funct->constOptimizeTestStatistic(RooAbsArg::ValueChange);
    }

    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors);
  }

  _function.updateFloatVec();

  synchronize_gradient_parameter_settings(parameter_settings);

  return 0;
}


void RooGradientFunction::synchronize_gradient_parameter_settings(std::vector<ROOT::Fit::ParameterSettings>& parameter_settings) const {
  _gradf.SetInitialGradient(parameter_settings);
  _gradf.SetParameterHasLimits(parameter_settings);
}


Double_t RooGradientFunction::GetPdfParamVal(Int_t index)
{
  // Access PDF parameter value by ordinal index (needed by MINUIT)

  return ((RooRealVar*)_function._floatParamList->at(index))->getVal();
}

Double_t RooGradientFunction::GetPdfParamErr(Int_t index)
{
  // Access PDF parameter error by ordinal index (needed by MINUIT)
  return ((RooRealVar*)_function._floatParamList->at(index))->getError();
}


void RooGradientFunction::SetPdfParamErr(Int_t index, Double_t value)
{
  // Modify PDF parameter error by ordinal index (needed by MINUIT)

  ((RooRealVar*)_function._floatParamList->at(index))->setError(value);
}



void RooGradientFunction::ClearPdfParamAsymErr(Int_t index)
{
  // Modify PDF parameter error by ordinal index (needed by MINUIT)

  ((RooRealVar*)_function._floatParamList->at(index))->removeAsymError();
}


void RooGradientFunction::SetPdfParamErr(Int_t index, Double_t loVal, Double_t hiVal)
{
  // Modify PDF parameter error by ordinal index (needed by MINUIT)

  ((RooRealVar*)_function._floatParamList->at(index))->setAsymError(loVal,hiVal);
}


////////////////////////////////////////////////////////////////////////////////

void RooGradientFunction::Function::updateFloatVec()
{
  _floatParamVec.clear();
  RooFIter iter = _floatParamList->fwdIterator();
  RooAbsArg* arg;
  _floatParamVec = std::vector<RooAbsArg*>(_floatParamList->getSize());
  Int_t i(0);
  while((arg=iter.next())) {
    _floatParamVec[i++] = arg;
  }
}


double RooGradientFunction::DoEval(const double *x) const {
  return _function(x);
}

double RooGradientFunction::Function::DoEval(const double *x) const
{
  Bool_t parameters_changed = kFALSE;

  // Set the parameter values for this iteration
  for (unsigned index = 0; index < NDim(); index++) {
    // also check whether the function was already evaluated for this set of parameters
    parameters_changed |= SetPdfParamVal(index,x[index]);
  }

  // Calculate the function for these parameters
  RooAbsReal::setHideOffset(kFALSE);
  double fvalue = _funct->getVal();
  RooAbsReal::setHideOffset(kTRUE);

  if (!parameters_changed) {
    return fvalue;
  }

  if (RooAbsPdf::evalError() || RooAbsReal::numEvalErrors()>0 || fvalue>1e30) {

    if (_printEvalErrors>=0) {

      if (_doEvalErrorWall) {
        oocoutW(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction: Minimized function has error status." << std::endl
                                       << "Returning maximum FCN so far (" << _maxFCN
                                       << ") to force MIGRAD to back out of this region. Error log follows" << std::endl;
      } else {
        oocoutW(static_cast<RooAbsArg*>(nullptr),Evaluation) << "RooGradientFunction: Minimized function has error status but is ignored" << std::endl;
      }

      TIterator* iter = _floatParamList->createIterator();
      RooRealVar* var;
      Bool_t first(kTRUE);
      ooccoutW(static_cast<RooAbsArg*>(nullptr),Evaluation) << "Parameter values: ";
      while((var=(RooRealVar*)iter->Next())) {
        if (first) { first = kFALSE ; } else ooccoutW(static_cast<RooAbsArg*>(nullptr),Evaluation) << ", ";
        ooccoutW(static_cast<RooAbsArg*>(nullptr),Evaluation) << var->GetName() << "=" << var->getVal();
      }
      delete iter;
      ooccoutW(static_cast<RooAbsArg*>(nullptr),Evaluation) << std::endl;

      RooAbsReal::printEvalErrors(ooccoutW(static_cast<RooAbsArg*>(nullptr),Evaluation),_printEvalErrors);
      ooccoutW(static_cast<RooAbsArg*>(nullptr),Evaluation) << std::endl;
    }

    if (_doEvalErrorWall) {
      fvalue = _maxFCN+1;
    }

    RooAbsPdf::clearEvalError();
    RooAbsReal::clearEvalErrorLog();
    _numBadNLL++;
  } else if (fvalue>_maxFCN) {
    _maxFCN = fvalue;
  }

  // Optional logging
  if (_verbose) {
    std::cout << "\nprevFCN" << (_funct->isOffsetting()?"-offset":"") << " = "
              << std::setprecision(10) << fvalue << std::setprecision(4) << "  ";
    std::cout.flush();
  }

  _evalCounter++;
  return fvalue;
}


void RooGradientFunction::run_derivator(const double *x) const {
  // check whether the derivative was already calculated for this set of parameters
  if (!std::equal(_grad_params.begin(), _grad_params.end(), x)) {
    // if not, set the _grad_params to the current input parameters
    std::vector<double> new_grad_params(x, x + NDim());
    _grad_params = new_grad_params;

    // Set the parameter values for this iteration
    // TODO: this is already done in DoEval as well; find efficient way to do only once
    for (unsigned index = 0; index < NDim(); index++) {
      SetPdfParamVal(index,x[index]);
    }

    // Calculate the function for these parameters
    _grad = _gradf(x, parameter_settings());
  }
}


double RooGradientFunction::DoDerivative(const double *x, unsigned int icoord) const {
  run_derivator(x);
  return _grad.Grad()(icoord);
}


bool RooGradientFunction::hasG2ndDerivative() const {
  return true;
}

bool RooGradientFunction::hasGStepSize() const {
  return true;
}

double RooGradientFunction::DoSecondDerivative(const double *x, unsigned int icoord) const {
  run_derivator(x);
  return _grad.G2()(icoord);
}

double RooGradientFunction::DoStepSize(const double *x, unsigned int icoord) const {
  run_derivator(x);
  return _grad.Gstep()(icoord);
}

bool RooGradientFunction::returnsInMinuit2ParameterSpace() const {
  return true;
}


unsigned int RooGradientFunction::NDim() const { return _function.NDim(); }

RooArgList* RooGradientFunction::GetFloatParamList() { return _function._floatParamList; }
RooArgList* RooGradientFunction::GetConstParamList() { return _function._constParamList; }
RooArgList* RooGradientFunction::GetInitFloatParamList() { return _function._initFloatParamList; }
RooArgList* RooGradientFunction::GetInitConstParamList() { return _function._initConstParamList; }

void RooGradientFunction::SetEvalErrorWall(Bool_t flag) { _function._doEvalErrorWall = flag; }
void RooGradientFunction::SetPrintEvalErrors(Int_t numEvalErrors) { _function._printEvalErrors = numEvalErrors; }

Double_t& RooGradientFunction::GetMaxFCN() { return _function._maxFCN; }
Int_t RooGradientFunction::GetNumInvalidNLL() { return _function._numBadNLL; }

Int_t RooGradientFunction::evalCounter() const { return _function._evalCounter; }
void RooGradientFunction::zeroEvalCount() { _function._evalCounter = 0; }

void RooGradientFunction::SetVerbose(Bool_t flag) {
  _function._verbose = flag;
}

std::vector<ROOT::Fit::ParameterSettings> &RooGradientFunction::parameter_settings() const {
  return _parameter_settings;
}

void RooGradientFunction::set_step_tolerance(double step_tolerance) const {
  _gradf.set_step_tolerance(step_tolerance);
}
void RooGradientFunction::set_grad_tolerance(double grad_tolerance) const {
  _gradf.set_grad_tolerance(grad_tolerance);
}
void RooGradientFunction::set_ncycles(unsigned int ncycles) const {
  _gradf.set_ncycles(ncycles);
}
void RooGradientFunction::set_error_level(double error_level) const {
  _gradf.set_error_level(error_level);
}
