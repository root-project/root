/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooMinimizer is a wrapper class around ROOT::Fit:Fitter that
// provides a seamless interface between the minimizer functionality
// and the native RooFit interface.
// <p>
// By default the Minimizer is MINUIT.
// <p>
// RooMinimizer can minimize any RooAbsReal function with respect to
// its parameters. Usual choices for minimization are RooNLLVar
// and RooChi2Var
// <p>
// RooMinimizer has methods corresponding to MINUIT functions like
// hesse(), migrad(), minos() etc. In each of these function calls
// the state of the MINUIT engine is synchronized with the state
// of the RooFit variables: any change in variables, change
// in the constant status etc is forwarded to MINUIT prior to
// execution of the MINUIT call. Afterwards the RooFit objects
// are resynchronized with the output state of MINUIT: changes
// parameter values, errors are propagated.
// <p>
// Various methods are available to control verbosity, profiling,
// automatic PDF optimization.
// END_HTML
//

#ifndef __ROOFIT_NOROOMINIMIZER

#include "RooFit.h"
#include "Riostream.h"

#include "TClass.h"

#include <fstream>
#include <iomanip>

#include "TH1.h"
#include "TH2.h"
#include "TMarker.h"
#include "TGraph.h"
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
#include "RooFitResult.h"

#include "Math/Minimizer.h"

#if (__GNUC__==3&&__GNUC_MINOR__==2&&__GNUC_PATCHLEVEL__==3)
char* operator+( streampos&, char* );
#endif

using namespace std;

ClassImp(RooMinimizer)
;

ROOT::Fit::Fitter *RooMinimizer::_theFitter = 0 ;



//_____________________________________________________________________________
void RooMinimizer::cleanup()
{
  // Cleanup method called by atexit handler installed by RooSentinel
  // to delete all global heap objects when the program is terminated
  if (_theFitter) {
    delete _theFitter ;
    _theFitter =0 ;
  }
}



//_____________________________________________________________________________
RooMinimizer::RooMinimizer(RooAbsReal& function)
{
  // Construct MINUIT interface to given function. Function can be anything,
  // but is typically a -log(likelihood) implemented by RooNLLVar or a chi^2
  // (implemented by RooChi2Var). Other frequent use cases are a RooAddition
  // of a RooNLLVar plus a penalty or constraint term. This class propagates
  // all RooFit information (floating parameters, their values and errors)
  // to MINUIT before each MINUIT call and propagates all MINUIT information
  // back to the RooFit object at the end of each call (updated parameter
  // values, their (asymmetric errors) etc. The default MINUIT error level
  // for HESSE and MINOS error analysis is taken from the defaultErrorLevel()
  // value of the input function.

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
  _fcn = new RooMinimizerFcn(_func,this,_verbose);
  _theFitter->Config().SetMinimizer(_minimizerType.c_str());
  setEps(1.0); // default tolerance
  // default max number of calls
  _theFitter->Config().MinimizerOptions().SetMaxIterations(500*_fcn->NDim());
  _theFitter->Config().MinimizerOptions().SetMaxFunctionCalls(500*_fcn->NDim());

  // Shut up for now
  setPrintLevel(-1) ;

  // Use +0.5 for 1-sigma errors
  setErrorLevel(_func->defaultErrorLevel()) ;

  // Declare our parameters to MINUIT
  _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
		    _optConst,_verbose) ;

  // Now set default verbosity
  if (RooMsgService::instance().silentMode()) {
    setPrintLevel(-1) ;
  } else {
    setPrintLevel(1) ;
  }
}



//_____________________________________________________________________________
RooMinimizer::~RooMinimizer()
{
  // Destructor

  if (_extV) {
    delete _extV ;
  }

  if (_fcn) {
    delete _fcn;
  }

}



//_____________________________________________________________________________
void RooMinimizer::setStrategy(Int_t istrat)
{
  // Change MINUIT strategy to istrat. Accepted codes
  // are 0,1,2 and represent MINUIT strategies for dealing
  // most efficiently with fast FCNs (0), expensive FCNs (2)
  // and 'intermediate' FCNs (1)

  _theFitter->Config().MinimizerOptions().SetStrategy(istrat);

}



//_____________________________________________________________________________
void RooMinimizer::setMaxIterations(Int_t n) 
{
  // Change maximum number of MINUIT iterations 
  // (RooMinimizer default 500 * #parameters)
  _theFitter->Config().MinimizerOptions().SetMaxIterations(n);
}




//_____________________________________________________________________________
void RooMinimizer::setMaxFunctionCalls(Int_t n) 
{
  // Change maximum number of likelihood function calss from MINUIT
  // (RooMinimizer default 500 * #parameters)
  _theFitter->Config().MinimizerOptions().SetMaxFunctionCalls(n);
}




//_____________________________________________________________________________
void RooMinimizer::setErrorLevel(Double_t level)
{
  // Set the level for MINUIT error analysis to the given
  // value. This function overrides the default value
  // that is taken in the RooMinimizer constructor from
  // the defaultErrorLevel() method of the input function

  _theFitter->Config().MinimizerOptions().SetErrorDef(level);

}



//_____________________________________________________________________________
void RooMinimizer::setEps(Double_t eps)
{
  // Change MINUIT epsilon

  _theFitter->Config().MinimizerOptions().SetTolerance(eps);

}


//_____________________________________________________________________________
void RooMinimizer::setOffsetting(Bool_t flag) 
{
  // Enable internal likelihood offsetting for enhanced numeric precision
  _func->enableOffsetting(flag) ; 
}




//_____________________________________________________________________________
void RooMinimizer::setMinimizerType(const char* type)
{
  // Choose the minimzer algorithm.
  _minimizerType = type;
}




//_____________________________________________________________________________
ROOT::Fit::Fitter* RooMinimizer::fitter()
{
  // Return underlying ROOT fitter object 
  return _theFitter ;
}


//_____________________________________________________________________________
const ROOT::Fit::Fitter* RooMinimizer::fitter() const 
{
  // Return underlying ROOT fitter object 
  return _theFitter ;
}



//_____________________________________________________________________________
RooFitResult* RooMinimizer::fit(const char* options)
{
  // Parse traditional RooAbsPdf::fitTo driver options
  //
  //  m - Run Migrad only
  //  h - Run Hesse to estimate errors
  //  v - Verbose mode
  //  l - Log parameters after each Minuit steps to file
  //  t - Activate profile timer
  //  r - Save fit result
  //  0 - Run Migrad with strategy 0

  TString opts(options) ;
  opts.ToLower() ;

  // Initial configuration
  if (opts.Contains("v")) setVerbose(1) ;
  if (opts.Contains("t")) setProfile(1) ;
  if (opts.Contains("l")) setLogFile(Form("%s.log",_func->GetName())) ;
  if (opts.Contains("c")) optimizeConst(1) ;

  // Fitting steps
  if (opts.Contains("0")) setStrategy(0) ;
  migrad() ;
  if (opts.Contains("0")) setStrategy(1) ;
  if (opts.Contains("h")||!opts.Contains("m")) hesse() ;
  if (!opts.Contains("m")) minos() ;

  return (opts.Contains("r")) ? save() : 0 ;
}




//_____________________________________________________________________________
Int_t RooMinimizer::minimize(const char* type, const char* alg)
{
  _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
		    _optConst,_verbose) ;

  _theFitter->Config().SetMinimizer(type,alg);

  profileStart() ;
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
  RooAbsReal::clearEvalErrorLog() ;

  bool ret = _theFitter->FitFCN(*_fcn);
  _status = ((ret) ? _theFitter->Result().Status() : -1);

  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
  profileStop() ;
  _fcn->BackProp(_theFitter->Result());

  saveStatus("MINIMIZE",_status) ;

  return _status ;
}



//_____________________________________________________________________________
Int_t RooMinimizer::migrad()
{
  // Execute MIGRAD. Changes in parameter values
  // and calculated errors are automatically
  // propagated back the RooRealVars representing
  // the floating parameters in the MINUIT operation

  _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
		    _optConst,_verbose) ;
  profileStart() ;
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
  RooAbsReal::clearEvalErrorLog() ;

  _theFitter->Config().SetMinimizer(_minimizerType.c_str(),"migrad");
  bool ret = _theFitter->FitFCN(*_fcn);
  _status = ((ret) ? _theFitter->Result().Status() : -1);

  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
  profileStop() ;
  _fcn->BackProp(_theFitter->Result());

  saveStatus("MIGRAD",_status) ;

  return _status ;
}



//_____________________________________________________________________________
Int_t RooMinimizer::hesse()
{
  // Execute HESSE. Changes in parameter values
  // and calculated errors are automatically
  // propagated back the RooRealVars representing
  // the floating parameters in the MINUIT operation

  if (_theFitter->GetMinimizer()==0) {
    coutW(Minimization) << "RooMinimizer::hesse: Error, run Migrad before Hesse!"
			<< endl ;
    _status = -1;
  }
  else {

    _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
		    _optConst,_verbose) ;
    profileStart() ;
    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
    RooAbsReal::clearEvalErrorLog() ;

    _theFitter->Config().SetMinimizer(_minimizerType.c_str());
    bool ret = _theFitter->CalculateHessErrors();
    _status = ((ret) ? _theFitter->Result().Status() : -1);

    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
    profileStop() ;
    _fcn->BackProp(_theFitter->Result());

    saveStatus("HESSE",_status) ;
  
  }

  return _status ;

}

//_____________________________________________________________________________
Int_t RooMinimizer::minos()
{
  // Execute MINOS. Changes in parameter values
  // and calculated errors are automatically
  // propagated back the RooRealVars representing
  // the floating parameters in the MINUIT operation

  if (_theFitter->GetMinimizer()==0) {
    coutW(Minimization) << "RooMinimizer::minos: Error, run Migrad before Minos!"
			<< endl ;
    _status = -1;
  }
  else {

    _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
		      _optConst,_verbose) ;
    profileStart() ;
    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
    RooAbsReal::clearEvalErrorLog() ;

    _theFitter->Config().SetMinimizer(_minimizerType.c_str());
    bool ret = _theFitter->CalculateMinosErrors();
    _status = ((ret) ? _theFitter->Result().Status() : -1);

    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
    profileStop() ;
    _fcn->BackProp(_theFitter->Result());

    saveStatus("MINOS",_status) ;

  }

  return _status ;

}


//_____________________________________________________________________________
Int_t RooMinimizer::minos(const RooArgSet& minosParamList)
{
  // Execute MINOS for given list of parameters. Changes in parameter values
  // and calculated errors are automatically
  // propagated back the RooRealVars representing
  // the floating parameters in the MINUIT operation

  if (_theFitter->GetMinimizer()==0) {
    coutW(Minimization) << "RooMinimizer::minos: Error, run Migrad before Minos!"
			<< endl ;
    _status = -1;
  }
  else if (minosParamList.getSize()>0) {

    _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
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



//_____________________________________________________________________________
Int_t RooMinimizer::seek()
{
  // Execute SEEK. Changes in parameter values
  // and calculated errors are automatically
  // propagated back the RooRealVars representing
  // the floating parameters in the MINUIT operation

  _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
		    _optConst,_verbose) ;
  profileStart() ;
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
  RooAbsReal::clearEvalErrorLog() ;

  _theFitter->Config().SetMinimizer(_minimizerType.c_str(),"seek");
  bool ret = _theFitter->FitFCN(*_fcn);
  _status = ((ret) ? _theFitter->Result().Status() : -1);

  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
  profileStop() ;
  _fcn->BackProp(_theFitter->Result());

  saveStatus("SEEK",_status) ;

  return _status ;
}



//_____________________________________________________________________________
Int_t RooMinimizer::simplex()
{
  // Execute SIMPLEX. Changes in parameter values
  // and calculated errors are automatically
  // propagated back the RooRealVars representing
  // the floating parameters in the MINUIT operation

  _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
		    _optConst,_verbose) ;
  profileStart() ;
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
  RooAbsReal::clearEvalErrorLog() ;

  _theFitter->Config().SetMinimizer(_minimizerType.c_str(),"simplex");
  bool ret = _theFitter->FitFCN(*_fcn);
  _status = ((ret) ? _theFitter->Result().Status() : -1);

  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
  profileStop() ;
  _fcn->BackProp(_theFitter->Result());

  saveStatus("SEEK",_status) ;
    
  return _status ;
}



//_____________________________________________________________________________
Int_t RooMinimizer::improve()
{
  // Execute IMPROVE. Changes in parameter values
  // and calculated errors are automatically
  // propagated back the RooRealVars representing
  // the floating parameters in the MINUIT operation

  _fcn->Synchronize(_theFitter->Config().ParamsSettings(),
		    _optConst,_verbose) ;
  profileStart() ;
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
  RooAbsReal::clearEvalErrorLog() ;

  _theFitter->Config().SetMinimizer(_minimizerType.c_str(),"migradimproved");
  bool ret = _theFitter->FitFCN(*_fcn);
  _status = ((ret) ? _theFitter->Result().Status() : -1);

  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
  profileStop() ;
  _fcn->BackProp(_theFitter->Result());

  saveStatus("IMPROVE",_status) ;

  return _status ;
}



//_____________________________________________________________________________
Int_t RooMinimizer::setPrintLevel(Int_t newLevel)
{
  // Change the MINUIT internal printing level

  Int_t ret = _printLevel ;
  _theFitter->Config().MinimizerOptions().SetPrintLevel(newLevel+1);
  _printLevel = newLevel+1 ;
  return ret ;
}

//_____________________________________________________________________________
void RooMinimizer::optimizeConst(Int_t flag)
{
  // If flag is true, perform constant term optimization on
  // function being minimized.

  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;

  if (_optConst && !flag){
    if (_printLevel>-1) coutI(Minimization) << "RooMinimizer::optimizeConst: deactivating const optimization" << endl ;
    _func->constOptimizeTestStatistic(RooAbsArg::DeActivate) ;
    _optConst = flag ;
  } else if (!_optConst && flag) {
    if (_printLevel>-1) coutI(Minimization) << "RooMinimizer::optimizeConst: activating const optimization" << endl ;
    _func->constOptimizeTestStatistic(RooAbsArg::Activate,flag>1) ;
    _optConst = flag ;
  } else if (_optConst && flag) {
    if (_printLevel>-1) coutI(Minimization) << "RooMinimizer::optimizeConst: const optimization already active" << endl ;
  } else {
    if (_printLevel>-1) coutI(Minimization) << "RooMinimizer::optimizeConst: const optimization wasn't active" << endl ;
  }

  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;

}



//_____________________________________________________________________________
RooFitResult* RooMinimizer::save(const char* userName, const char* userTitle)
{
  // Save and return a RooFitResult snaphot of current minimizer status.
  // This snapshot contains the values of all constant parameters,
  // the value of all floating parameters at RooMinimizer construction and
  // after the last MINUIT operation, the MINUIT status, variance quality,
  // EDM setting, number of calls with evaluation problems, the minimized
  // function value and the full correlation matrix

  if (_theFitter->GetMinimizer()==0) {
    coutW(Minimization) << "RooMinimizer::save: Error, run minimization before!"
			<< endl ;
    return 0;
  }

  TString name,title ;
  name = userName ? userName : Form("%s", _func->GetName()) ;
  title = userTitle ? userTitle : Form("%s", _func->GetTitle()) ;
  RooFitResult* fitRes = new RooFitResult(name,title) ;

  // Move eventual fixed parameters in floatList to constList
  Int_t i ;
  RooArgList saveConstList(*(_fcn->GetConstParamList())) ;
  RooArgList saveFloatInitList(*(_fcn->GetInitFloatParamList())) ;
  RooArgList saveFloatFinalList(*(_fcn->GetFloatParamList())) ;
  for (i=0 ; i<_fcn->GetFloatParamList()->getSize() ; i++) {
    RooAbsArg* par = _fcn->GetFloatParamList()->at(i) ;
    if (par->isConstant()) {
      saveFloatInitList.remove(*saveFloatInitList.find(par->GetName()),kTRUE) ;
      saveFloatFinalList.remove(*par) ;
      saveConstList.add(*par) ;
    }
  }
  saveConstList.sort() ;

  fitRes->setConstParList(saveConstList) ;
  fitRes->setInitParList(saveFloatInitList) ;

  fitRes->setStatus(_status) ;
  fitRes->setCovQual(_theFitter->GetMinimizer()->CovMatrixStatus()) ;
  fitRes->setMinNLL(_theFitter->Result().MinFcnValue()) ;
  fitRes->setNumInvalidNLL(_fcn->GetNumInvalidNLL()) ;
  fitRes->setEDM(_theFitter->Result().Edm()) ;
  fitRes->setFinalParList(saveFloatFinalList) ;
  if (!_extV) {
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
    fitRes->fillCorrMatrix(globalCC,corrs,covs) ;
  } else {
    fitRes->setCovarianceMatrix(*_extV) ;
  }

  fitRes->setStatusHistory(_statusHistory) ;

  return fitRes ;

}

//_____________________________________________________________________________
RooPlot* RooMinimizer::contour(RooRealVar& var1, RooRealVar& var2,
			       Double_t n1, Double_t n2, Double_t n3,
			       Double_t n4, Double_t n5, Double_t n6)
{
  // Create and draw a TH2 with the error contours in parameters var1 and v2 at up to 6 'sigma' settings
  // where 'sigma' is calculated as n*n*errorLevel


  RooArgList* params = _fcn->GetFloatParamList() ;
  RooArgList* paramSave = (RooArgList*) params->snapshot() ;

  // Verify that both variables are floating parameters of PDF
  Int_t index1= _fcn->GetFloatParamList()->index(&var1);
  if(index1 < 0) {
    coutE(Minimization) << "RooMinimizer::contour(" << GetName()
			<< ") ERROR: " << var1.GetName()
			<< " is not a floating parameter of "
			<< _func->GetName() << endl ;
    return 0;
  }

  Int_t index2= _fcn->GetFloatParamList()->index(&var2);
  if(index2 < 0) {
    coutE(Minimization) << "RooMinimizer::contour(" << GetName()
			<< ") ERROR: " << var2.GetName()
			<< " is not a floating parameter of PDF "
			<< _func->GetName() << endl ;
    return 0;
  }

  // create and draw a frame
  RooPlot* frame = new RooPlot(var1,var2) ;

  // draw a point at the current parameter values
  TMarker *point= new TMarker(var1.getVal(), var2.getVal(), 8);
  frame->addObject(point) ;

  // remember our original value of ERRDEF
  Double_t errdef= _theFitter->GetMinimizer()->ErrorDef();

  Double_t n[6] ;
  n[0] = n1 ; n[1] = n2 ; n[2] = n3 ; n[3] = n4 ; n[4] = n5 ; n[5] = n6 ;
  unsigned int npoints(50);

  for (Int_t ic = 0 ; ic<6 ; ic++) {
    if(n[ic] > 0) {

       // set the value corresponding to an n1-sigma contour
       _theFitter->GetMinimizer()->SetErrorDef(n[ic]*n[ic]*errdef);

      // calculate and draw the contour
      Double_t *xcoor = new Double_t[npoints+1];
      Double_t *ycoor = new Double_t[npoints+1];
      bool ret = _theFitter->GetMinimizer()->Contour(index1,index2,npoints,xcoor,ycoor);

      if (!ret) {
	coutE(Minimization) << "RooMinimizer::contour("
			    << GetName()
			    << ") ERROR: MINUIT did not return a contour graph for n="
			    << n[ic] << endl ;
      } else {
	xcoor[npoints] = xcoor[0];
	ycoor[npoints] = ycoor[0];
	TGraph* graph = new TGraph(npoints+1,xcoor,ycoor);

	graph->SetName(Form("contour_%s_n%f",_func->GetName(),n[ic])) ;
	graph->SetLineStyle(ic+1) ;
	graph->SetLineWidth(2) ;
	graph->SetLineColor(kBlue) ;
	frame->addObject(graph,"L") ;
      }

      delete [] xcoor;
      delete [] ycoor;
    }
  }


  // restore the original ERRDEF
  _theFitter->Config().MinimizerOptions().SetErrorDef(errdef);

  // restore parameter values
  *params = *paramSave ;
  delete paramSave ;

  return frame ;

}


//_____________________________________________________________________________
void RooMinimizer::profileStart()
{
  // Start profiling timer
  if (_profile) {
    _timer.Start() ;
    _cumulTimer.Start(_profileStart?kFALSE:kTRUE) ;
    _profileStart = kTRUE ;
  }
}


//_____________________________________________________________________________
void RooMinimizer::profileStop()
{
  // Stop profiling timer and report results of last session
  if (_profile) {
    _timer.Stop() ;
    _cumulTimer.Stop() ;
    coutI(Minimization) << "Command timer: " ; _timer.Print() ;
    coutI(Minimization) << "Session timer: " ; _cumulTimer.Print() ;
  }
}





//_____________________________________________________________________________
void RooMinimizer::applyCovarianceMatrix(TMatrixDSym& V)
{
  // Apply results of given external covariance matrix. i.e. propagate its errors
  // to all RRV parameter representations and give this matrix instead of the
  // HESSE matrix at the next save() call

  _extV = (TMatrixDSym*) V.Clone() ;
  _fcn->ApplyCovarianceMatrix(*_extV);

}



RooFitResult* RooMinimizer::lastMinuitFit(const RooArgList& varList)
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

#endif
