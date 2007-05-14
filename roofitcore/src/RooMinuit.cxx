/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooMinuit.cxx,v 1.24 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [PDF] --
// RooMinuit is a wrapper class around TFitter/TMinuit that
// provides a seamless interface between the MINUIT functionality
// and the native RooFit interface.
//
// RooMinuit can minimize any RooAbsReal function with respect to
// its parameters. Usual choices for minimization are RooNLLVar
// and RooChi2Var
//
// RooMinuit has methods corresponding to MINUIT functions like
// hesse(), migrad(), minos() etc. In each of these function calls
// the state of the MINUIT engine is synchronized with the state
// of the RooFit variables: any change in variables, change
// in the constant status etc is forwarded to MINUIT prior to
// execution of the MINUIT call. Afterwards the RooFit objects
// are resynchronized with the output state of MINUIT: changes
// parameter values, errors are propagated.
//
// Various methods are available to control verbosity, profiling,
// automatic PDF optimization.

#include "RooFit.h"

#include "TClass.h"
#include <fstream>
#include <iomanip>
#include "TH1.h"
#include "TH2.h"
#include "TMarker.h"
#include "TGraph.h"
#include "TStopwatch.h"
#include "TFitter.h"
#include "TMinuit.h"
#include "TDirectory.h"
#include "RooMinuit.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooAbsReal.h"
#include "RooAbsRealLValue.h"
#include "RooRealVar.h"
#include "RooFitResult.h"
#include "RooAbsPdf.h"


#if (__GNUC__==3&&__GNUC_MINOR__==2&&__GNUC_PATCHLEVEL__==3)
char* operator+( streampos&, char* );
#endif

ClassImp(RooMinuit) 
;

static TVirtualFitter *_theFitter = 0;


RooMinuit::RooMinuit(RooAbsReal& function)
{
  // Constructor

  // Store function reference
  _func = &function ;
  _logfile = 0 ;
  _optConst = kFALSE ;
  _verbose = kFALSE ;
  _profile = kFALSE ;
  _handleLocalErrors = kFALSE ;
  _printLevel = 1 ;

  // Examine parameter list
  RooArgSet* paramSet = function.getParameters(RooArgSet()) ;
  RooArgList paramList(*paramSet) ;
  delete paramSet ;

  _floatParamList = (RooArgList*) paramList.selectByAttrib("Constant",kFALSE) ; 
  if (_floatParamList->getSize()>1) {
    _floatParamList->sort() ;
  }
  _floatParamList->setName("floatParamList") ;

  _constParamList = (RooArgList*) paramList.selectByAttrib("Constant",kTRUE) ;
  if (_constParamList->getSize()>1) {
    _constParamList->sort() ;
  }
  _constParamList->setName("constParamList") ;

  // Remove all non-RooRealVar parameters from list (MINUIT cannot handle them)
  TIterator* pIter = _floatParamList->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)pIter->Next())) {
    if (!arg->IsA()->InheritsFrom(RooAbsRealLValue::Class())) {
      cout << "RooMinuit::RooMinuit: removing parameter " << arg->GetName() 
	   << " from list because it is not of type RooRealVar" << endl ;
      _floatParamList->remove(*arg) ;
    }
  }
  _nPar      = _floatParamList->getSize() ;  
  delete pIter ;

  // Save snapshot of initial lists
  _initFloatParamList = (RooArgList*) _floatParamList->snapshot(kFALSE) ;
  _initConstParamList = (RooArgList*) _constParamList->snapshot(kFALSE) ;

  // Initialize MINUIT
  Int_t nPar= _floatParamList->getSize();
  if (_theFitter) delete _theFitter ;
  _theFitter = new TFitter(nPar*2) ; //WVE Kludge, nPar*2 works around TMinuit memory allocation bug
  _theFitter->SetObjectFit(this) ;

  // Shut up for now
  setPrintLevel(-1) ;
  _theFitter->Clear();
  
  // Tell MINUIT to use our global glue function
  _theFitter->SetFCN(RooMinuitGlue);

  // Use +0.5 for 1-sigma errors
  setErrorLevel(function.defaultErrorLevel()) ;

  // Declare our parameters to MINUIT
  synchronize(kFALSE) ;

  // Reset the *largest* negative log-likelihood value we have seen so far
  _maxFCN= -1e30 ;
  _numBadNLL = 0 ;

  // Now set default verbosity
  setWarnLevel(1) ;
  setPrintLevel(1) ;

}


RooMinuit::~RooMinuit() 
{
  // Destructor
  delete _floatParamList ;
  delete _initFloatParamList ;
  delete _constParamList ;
  delete _initConstParamList ;
}



void RooMinuit::setStrategy(Int_t istrat) 
{
  // Change MINUIT strategy 
  Double_t stratArg(istrat) ;
  _theFitter->ExecuteCommand("SET STR",&stratArg,1) ;
}



void RooMinuit::setErrorLevel(Double_t level)
{
  _theFitter->ExecuteCommand("SET ERR",&level,1);
}


void RooMinuit::setEps(Double_t eps)
{
  // Change MINUIT epsilon 
  _theFitter->ExecuteCommand("SET EPS",&eps,1) ;  
}



RooFitResult* RooMinuit::fit(const char* options)
{
  // Parse traditional RooAbsPdf::fitTo driver options
  // 
  //  s - Run Hesse first to estimate initial step size
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
  if (opts.Contains("s")) hesse() ;
  if (opts.Contains("0")) setStrategy(0) ;
  migrad() ;
  if (opts.Contains("0")) setStrategy(1) ;
  if (opts.Contains("h")||!opts.Contains("m")) hesse() ;
  if (!opts.Contains("m")) minos() ;
  
  return (!opts.Contains("r")) ? save() : 0 ; 
}



Int_t RooMinuit::migrad() 
{
  // Execute MIGRAD
  Double_t arglist[2];
  arglist[0]= 500*_nPar; // maximum iterations
  arglist[1]= 1.0;       // tolerance

  synchronize(kTRUE) ;
  profileStart() ;
  _status= _theFitter->ExecuteCommand("MIGRAD",arglist,2);
  profileStop() ;
  backProp() ;
  return _status ;
}



Int_t RooMinuit::hesse() 
{
  // Execute HESSE

  Double_t arglist[2];
  arglist[0]= 500*_nPar; // maximum iterations

  synchronize(kTRUE) ;
  profileStart() ;
  _status= _theFitter->ExecuteCommand("HESSE",arglist,1);
  profileStop() ;
  backProp() ;
  return _status ;
}



Int_t RooMinuit::minos() 
{
  // Execute MINOS
  Double_t arglist[2];
  arglist[0]= 500*_nPar; // maximum iterations

  synchronize(kTRUE) ;
  profileStart() ;
  _status= _theFitter->ExecuteCommand("MINOS",arglist,1);
  profileStop() ;
  backProp() ;
  return _status ;
}


// added FMV, 08/18/03 
Int_t RooMinuit::minos(const RooArgSet& minosParamList) 
{
  // Execute MINOS for given list of parameters
  Int_t nMinosPar(0) ;
  Double_t* arglist = new Double_t[_nPar+1];

  if (minosParamList.getSize()>0) {
    TIterator* aIter = minosParamList.createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)aIter->Next())) {
      RooAbsArg* par = _floatParamList->find(arg->GetName());
      if (par && !par->isConstant()) {
	Int_t index = _floatParamList->index(par);
	nMinosPar++;
        arglist[nMinosPar]=index+1;
      }
    }
  }
  arglist[0]= 500*_nPar; // maximum iterations

  synchronize(kTRUE) ;
  profileStart() ;
  _status= _theFitter->ExecuteCommand("MINOS",arglist,1+nMinosPar);
  profileStop() ;
  backProp() ;

  delete[] arglist ;
  return _status ;
}


Int_t RooMinuit::seek() 
{
  // Execute SEEK
  Double_t arglist[2];
  arglist[0]= 500*_nPar; // maximum iterations

  synchronize(kTRUE) ;
  profileStart() ;
  _status= _theFitter->ExecuteCommand("SEEK",arglist,1);
  profileStop() ;
  backProp() ;
  return _status ;
}


Int_t RooMinuit::simplex() 
{
  // Execute SIMPLEX 
  Double_t arglist[2];
  arglist[0]= 500*_nPar; // maximum iterations
  arglist[1]= 1.0;       // tolerance

  synchronize(kTRUE) ;
  profileStart() ;
  _status= _theFitter->ExecuteCommand("SIMPLEX",arglist,2);
  profileStop() ;
  backProp() ;
  return _status ;
}


Int_t RooMinuit::improve()
{
  // Execute IMPROVE
  Double_t arglist[2];
  arglist[0]= 500*_nPar; // maximum iterations

  synchronize(kTRUE) ;
  profileStart() ;
  _status= _theFitter->ExecuteCommand("IMPROVE",arglist,1);
  profileStop() ;
  backProp() ;
  return _status ;
}



Int_t RooMinuit::setPrintLevel(Int_t newLevel) 
{
  Int_t ret = _printLevel ;
  Double_t arg(newLevel) ;
  _theFitter->ExecuteCommand("SET PRINT",&arg,1);
  _printLevel = newLevel ;
  return ret ;
}


Int_t RooMinuit::setWarnLevel(Int_t newLevel) 
{
  Int_t ret = _warnLevel ;
  Double_t arg(newLevel) ;
  _theFitter->ExecuteCommand("SET WARNINGS",&arg,1);
  _warnLevel = newLevel ;
  return ret ;
}
      

Bool_t RooMinuit::synchronize(Bool_t verbose)
{
  Int_t oldPrint = setPrintLevel(-1) ;
  Int_t oldWarn = setWarnLevel(-1) ;
  Bool_t constValChange(kFALSE) ;
  Bool_t constStatChange(kFALSE) ;

  Int_t index(0) ;

  // Handle eventual migrations from constParamList -> floatParamList 
  for(index= 0; index < _constParamList->getSize() ; index++) {
    RooRealVar *par= dynamic_cast<RooRealVar*>(_constParamList->at(index)) ;
    if (!par) continue ;

    RooRealVar *oldpar= dynamic_cast<RooRealVar*>(_initConstParamList->at(index)) ;

    // Test if constness changed
    if (!par->isConstant()) {
      
      // Remove from constList, add to floatList
      _constParamList->remove(*par) ;
      _floatParamList->add(*par) ;
      _initFloatParamList->addClone(*oldpar) ;      
      _initConstParamList->remove(*oldpar) ;
      constStatChange=kTRUE ;
      _nPar++ ;

      if (verbose) {
	cout << "RooMinuit::synchronize: parameter " << par->GetName() << " is now floating." << endl ;
      }
    } 

    // Test if value changed
    if (par->getVal()!= oldpar->getVal()) {
      constValChange=kTRUE ;      
      if (verbose) {
	cout << "RooMinuit::synchronize: value of constant parameter " << par->GetName() 
	     << " changed from " << oldpar->getVal() << " to " << par->getVal() << endl ;
      }      
    }    

  }

  // Update reference list
  *_initConstParamList = *_constParamList ;


  // Synchronize MINUIT with function state
  for(index= 0; index < _nPar; index++) {
    RooRealVar *par= dynamic_cast<RooRealVar*>(_floatParamList->at(index)) ;

    Double_t pstep(0) ;
    Double_t pmin(0) ;
    Double_t pmax(0) ;

    if(!par->isConstant()) {

      // Verify that floating parameter is indeed of type RooRealVar 
      if (!par->IsA()->InheritsFrom(RooRealVar::Class())) {
	cout << "RooMinuit::fit: Error, non-constant parameter " << par->GetName() 
	     << " is not of type RooRealVar, skipping" << endl ;
	continue ;
      }

      // Set the limits, if not infinite
      if (par->hasMin() && par->hasMax()) {
	pmin = par->getMin();
	pmax = par->getMax();
      }
      
      // Calculate step size
      pstep= par->getError();
      if(pstep <= 0) {
	// Floating parameter without error estitimate
	if (par->hasMin() && par->hasMax()) {
	  pstep= 0.1*(pmax-pmin);
	} else {
	  pstep=1 ;
	}						  
	if(_verbose) {
	  cout << "RooMinuit::synchronize: WARNING: no initial error estimate available for "
	       << par->GetName() << ": using " << pstep << endl;
	}
      }       
    } else {
      pmin = par->getVal() ;
      pmax = par->getVal() ;      
    }

    // Extract previous information
    Double_t oldVar,oldVerr,oldVlo,oldVhi ;
    char oldParname[100] ;
    Int_t ierr = _theFitter->GetParameter(index,oldParname,oldVar,oldVerr,oldVlo,oldVhi)  ;

    // Determine if parameters is currently fixed in MINUIT

    Int_t ix ;
    Bool_t oldFixed(kFALSE) ;
    if (ierr>=0) {
      for (ix = 1; ix <= gMinuit->fNpfix; ++ix) {
        if (gMinuit->fIpfix[ix-1] == index+1) oldFixed=kTRUE ;
      }
    }
    
    if (par->isConstant() && !oldFixed) {

      // Parameter changes floating -> constant : update only value if necessary
      if (oldVar!=par->getVal()) {
	Double_t arglist[2] ;
	arglist[0] = index+1 ;
	arglist[1] = par->getVal() ;
	_theFitter->ExecuteCommand("SET PAR",arglist,2) ;
	if (verbose) {
	  cout << "RooMinuit::synchronize: value of parameter " << par->GetName() << " changed from " << oldVar << " to " << par->getVal() << endl ;
	}
      }

      _theFitter->FixParameter(index) ;
      constStatChange=kTRUE ;
      if (verbose) {
	cout << "RooMinuit::synchronize: parameter " << par->GetName() << " is now fixed." << endl ;
      }

    } else if (par->isConstant() && oldFixed) {
      
      // Parameter changes constant -> constant : update only value if necessary
      if (oldVar!=par->getVal()) {
	Double_t arglist[2] ;
	arglist[0] = index+1 ;
	arglist[1] = par->getVal() ;
	_theFitter->ExecuteCommand("SET PAR",arglist,2) ;
	constValChange=kTRUE ;

	if (verbose) {
	  cout << "RooMinuit::synchronize: value of fixed parameter " << par->GetName() << " changed from " << oldVar << " to " << par->getVal() << endl ;
	}
      }

    } else {
      
      if (!par->isConstant() && oldFixed) {
	_theFitter->ReleaseParameter(index) ;
	constStatChange=kTRUE ;
	
	if (verbose) {
	  cout << "RooMinuit::synchronize: parameter " << par->GetName() << " is now floating." << endl ;
	}
      } 

      // Parameter changes constant -> floating : update all if necessary
      if (oldVar!=par->getVal() || oldVlo!=pmin || oldVhi != pmax || oldVerr!=pstep) {
	_theFitter->SetParameter(index, par->GetName(), par->getVal(), pstep, pmin, pmax);
      }

      // Inform user about changes in verbose mode
      if (verbose && ierr>=0) {
	// if ierr<0, par was moved from the const list and a message was already printed

	if (oldVar!=par->getVal()) {
	  cout << "RooMinuit::synchronize: value of parameter " << par->GetName() << " changed from " << oldVar << " to " << par->getVal() << endl ;
	}
	if (oldVlo!=pmin || oldVhi!=pmax) {
	  cout << "RooMinuit::synchronize: limits of parameter " << par->GetName() << " changed from [" << oldVlo << "," << oldVhi 
	       << "] to [" << pmin << "," << pmax << "]" << endl ;
	}

	// If oldVerr=0, then parameter was previously fixed
	if (oldVerr!=pstep && oldVerr!=0) {
	cout << "RooMinuit::synchronize: error/step size of parameter " << par->GetName() << " changed from " << oldVerr << " to " << pstep << endl ;
	}
      }      
    }
  }




  oldWarn = setWarnLevel(oldWarn) ;
  oldPrint = setPrintLevel(oldPrint) ;

  if (_optConst) {
    if (constStatChange) {
      cout << "RooMinuit::synchronize: set of constant parameters changed, rerunning const optimizer" << endl ;
      _func->constOptimize(RooAbsArg::ConfigChange) ;
    } else if (constValChange) {
      cout << "RooMinuit::synchronize: constant parameter values changed, rerunning const optimizer" << endl ;
      _func->constOptimize(RooAbsArg::ValueChange) ;
    }
  }

  return 0 ;  
}
	


void RooMinuit::optimizeConst(Bool_t flag) 
{
  if (_optConst && !flag){ 
    if (_printLevel>-1) cout << "RooMinuit::optimizeConst: deactivating const optimization" << endl ;
    _func->constOptimize(RooAbsArg::DeActivate) ;
    _optConst = flag ;
  } else if (!_optConst && flag) {
    if (_printLevel>-1) cout << "RooMinuit::optimizeConst: activating const optimization" << endl ;
    _func->constOptimize(RooAbsArg::Activate) ;
    _optConst = flag ;
  } else if (_optConst && flag) {
    if (_printLevel>-1) cout << "RooMinuit::optimizeConst: const optimization already active" << endl ;
  } else {
    if (_printLevel>-1) cout << "RooMinuit::optimizeConst: const optimization wasn't active" << endl ;
  }
}


RooFitResult* RooMinuit::save(const char* userName, const char* userTitle) 
{
  // Save snaphot of current minimizer status

  TString name,title ;
  name = userName ? userName : Form(_func->GetName()) ;
  title = userTitle ? userTitle : Form(_func->GetTitle()) ;  
  RooFitResult* fitRes = new RooFitResult(name,title) ;

  // Move eventual fixed paramaters in floatList to constList
  Int_t i ;
  RooArgList saveConstList(*_constParamList) ;
  RooArgList saveFloatInitList(*_initFloatParamList) ;
  RooArgList saveFloatFinalList(*_floatParamList) ;
  for (i=0 ; i<_floatParamList->getSize() ; i++) {
    RooAbsArg* par = _floatParamList->at(i) ;
    if (par->isConstant()) {
      saveFloatInitList.remove(*saveFloatInitList.find(par->GetName()),kTRUE) ;
      saveFloatFinalList.remove(*par) ;
      saveConstList.add(*par) ;
    }
  }
  saveConstList.sort() ;
  
  fitRes->setConstParList(saveConstList) ;
  fitRes->setInitParList(saveFloatInitList) ;
  
  Double_t edm, errdef, minVal;
  Int_t nvpar, nparx;
  Int_t icode = _theFitter->GetStats(minVal, edm, errdef, nvpar, nparx);
  fitRes->setStatus(_status) ;
  fitRes->setCovQual(icode) ;
  fitRes->setMinNLL(minVal) ;
  fitRes->setNumInvalidNLL(_numBadNLL) ;
  fitRes->setEDM(edm) ;    
  fitRes->setFinalParList(saveFloatFinalList) ;
  fitRes->fillCorrMatrix() ;

  return fitRes ;
}




TH2F* RooMinuit::contour(RooRealVar& var1, RooRealVar& var2, Double_t n1, Double_t n2, Double_t n3) 
{
  // Verify that both variables are floating parameters of PDF
  Int_t index1= _floatParamList->index(&var1);
  if(index1 < 0) {
    cout << "RooMinuit::contour(" << GetName() 
	 << ") ERROR: " << var1.GetName() << " is not a floating parameter of " << _func->GetName() << endl ;
    return 0;
  }

  Int_t index2= _floatParamList->index(&var2);
  if(index2 < 0) {
    cout << "RooMinuit::contour(" << GetName() 
	 << ") ERROR: " << var2.GetName() << " is not a floating parameter of PDF " << _func->GetName() << endl ;
    return 0;
  }
  
  // create and draw a frame
  TH2F *frame = var1.createHistogram("contourPlot", var2, "-log(likelihood)") ;
  frame->SetStats(kFALSE);

  // draw a point at the current parameter values
  TMarker *point= new TMarker(var1.getVal(), var2.getVal(), 8);

  // remember our original value of ERRDEF
  Double_t errdef= gMinuit->fUp;

  TGraph* graph1 = 0;
  if(n1 > 0) {
    // set the value corresponding to an n1-sigma contour
    gMinuit->SetErrorDef(n1*n1*errdef);
    // calculate and draw the contour
    graph1= (TGraph*)gMinuit->Contour(25, index1, index2);
    gDirectory->Append(graph1) ;
  }

  TGraph* graph2 = 0;
  if(n2 > 0) {
    // set the value corresponding to an n2-sigma contour
    gMinuit->SetErrorDef(n2*n2*errdef);
    // calculate and draw the contour
    graph2= (TGraph*)gMinuit->Contour(25, index1, index2);
    graph2->SetLineStyle(2);
    gDirectory->Append(graph2) ;
  }

  TGraph* graph3 = 0;
  if(n3 > 0) {
    // set the value corresponding to an n3-sigma contour
    gMinuit->SetErrorDef(n3*n3*errdef);
    // calculate and draw the contour
    graph3= (TGraph*)gMinuit->Contour(25, index1, index2);
    graph3->SetLineStyle(3);
    gDirectory->Append(graph3) ;
  }
  // restore the original ERRDEF
  gMinuit->SetErrorDef(errdef);


  // Draw all objects
  frame->Draw();
  point->Draw();
  if (graph1) graph1->Draw() ;
  if (graph2) graph2->Draw();
  if (graph3) graph3->Draw();
  
  return frame;
}


Bool_t RooMinuit::setLogFile(const char* logfile) 
{
  if (_logfile) {
    cout << "RooMinuit::setLogFile: closing previous log file" << endl ;
    _logfile->close() ;
    delete _logfile ;
    _logfile = 0 ;
  }
  _logfile = new ofstream(logfile) ;
  if (!_logfile->good()) {
    cout << "RooMinuit::setLogFile: cannot open file " << logfile << endl ;
    _logfile->close() ;
    delete _logfile ;
    _logfile= 0;
  }  
  return kFALSE ;
}


Double_t RooMinuit::getPdfParamVal(Int_t index)
{
  // Access PDF parameter value by ordinal index (needed by MINUIT)
  return ((RooRealVar*)_floatParamList->at(index))->getVal() ;
}


Double_t RooMinuit::getPdfParamErr(Int_t index)
{
  // Access PDF parameter error by ordinal index (needed by MINUIT)
  return ((RooRealVar*)_floatParamList->at(index))->getError() ;  
}


Bool_t RooMinuit::setPdfParamVal(Int_t index, Double_t value, Bool_t verbose)
{
  // Modify PDF parameter value by ordinal index (needed by MINUIT)
  RooRealVar* par = (RooRealVar*)_floatParamList->at(index) ;

  if (par->getVal()!=value) {
    if (verbose) cout << par->GetName() << "=" << value << ", " ;
    par->setVal(value) ;  
    return kTRUE ;
  }

  return kFALSE ;
}


void RooMinuit::setPdfParamErr(Int_t index, Double_t value)
{
  // Modify PDF parameter error by ordinal index (needed by MINUIT)
  ((RooRealVar*)_floatParamList->at(index))->setError(value) ;    
}


void RooMinuit::clearPdfParamAsymErr(Int_t index) 
{
  // Modify PDF parameter error by ordinal index (needed by MINUIT)
  ((RooRealVar*)_floatParamList->at(index))->removeAsymError() ;      
}

void RooMinuit::setPdfParamErr(Int_t index, Double_t loVal, Double_t hiVal) 
{
  // Modify PDF parameter error by ordinal index (needed by MINUIT)
  ((RooRealVar*)_floatParamList->at(index))->setAsymError(loVal,hiVal) ;    
}


void RooMinuit::profileStart() 
{
  if (_profile) {
    _timer.Start() ;
    _cumulTimer.Start(kFALSE) ;
  }
}




void RooMinuit::profileStop() 
{
  if (_profile) {
    _timer.Stop() ;
    _cumulTimer.Stop() ;
    cout << "Command timer: " ; _timer.Print() ;
    cout << "Session timer: " ; _cumulTimer.Print() ;
  }
}





void RooMinuit::backProp() 
{
  // Transfer MINUIT fit results back into RooFit objects

  Double_t val,err,vlo,vhi, eplus, eminus, eparab, globcc;
  char buffer[10240];
  Int_t index ;
  for(index= 0; index < _nPar; index++) {
    _theFitter->GetParameter(index, buffer, val, err, vlo, vhi);
    setPdfParamVal(index, val);
    _theFitter->GetErrors(index, eplus, eminus, eparab, globcc);

    // Set the parabolic error
    setPdfParamErr(index, err);

    if(eplus > 0 || eminus < 0) {
      // Store the asymmetric error, if it is available
      setPdfParamErr(index, eminus,eplus);
    } else {
      // Clear the asymmetric error
      clearPdfParamAsymErr(index) ;
    }
  }
}



void RooMinuitGlue(Int_t& /*np*/, Double_t* /*gin*/,
		   Double_t &f, Double_t *par, Int_t /*flag*/)
{
  // Static function that interfaces minuit with RooMinuit

  // Retrieve fit context and its components
  RooMinuit* context = (RooMinuit*) _theFitter->GetObjectFit() ;
  ofstream* logf   = context->logfile() ;
  Double_t& maxFCN = context->maxFCN() ;
  Bool_t verbose   = context->_verbose ;

  // Set the parameter values for this iteration
  Int_t nPar= context->getNPar();
  for(Int_t index= 0; index < nPar; index++) {
    if (logf) (*logf) << par[index] << " " ;
    context->setPdfParamVal(index, par[index],verbose);
  }

  // Calculate the function for these parameters
  f= context->_func->getVal() ;
  if (f==0 || (context->_handleLocalErrors&&RooAbsPdf::evalError())) {
    cout << "RooFitGlue: Minimized function has error status. Returning maximum FCN" << endl
	 << "            so far (" << maxFCN << ") to force MIGRAD to back out of this region" << endl ;
    f = maxFCN ;
    RooAbsPdf::clearEvalError() ;
    context->_numBadNLL++ ;
  } else if (f>maxFCN) {
    maxFCN = f ;
  }

  // Optional logging
  if (logf) (*logf) << setprecision(15) << f << setprecision(4) << endl;
  if (verbose) {
    cout << "\nprevFCN = " << setprecision(10) << f << setprecision(4) << "  " ;
    cout.flush() ;
  }
}

