/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooFitContext.cc,v 1.45 2001/12/01 08:12:47 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// RooFitContext holds and combines a RooAbsPdf and a RooDataSet
// for unbinned maximum likelihood fitting. The PDF and DataSet 
// are both cloned and tied to each other.
//
// The context implements various optimization techniques that can
// only be made under the assumption that the dependent/parameter
// interpretation of all servers is fixed for the duration of the
// fit. (For example PDFs with exclusively constant parameters
// can be precalculated)
//
// This class also contains the interface to MINUIT to peform the
// actual fitting.


#include <fstream.h>
#include <iomanip.h>
#include "TH1.h"
#include "TH2.h"
#include "TMarker.h"
#include "TGraph.h"
#include "TStopwatch.h"
#include "TFitter.h"
#include "TMinuit.h"
#include "RooFitCore/RooFitContext.hh"
#include "RooFitCore/RooAbsData.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooResolutionModel.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFitResult.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooDataSet.hh"

ClassImp(RooFitContext)
;

static TVirtualFitter *_theFitter(0);


RooFitContext::RooFitContext(const RooAbsData* data, const RooAbsPdf* pdf, 
			     Bool_t cloneData, Bool_t clonePdf, const RooArgSet* projDeps) :
  TNamed(*pdf), _origLeafNodeList("origLeafNodeList"), _extendedMode(kFALSE), _doOptCache(kFALSE),
  _ownData(cloneData), _zombie(kFALSE), _projDeps(0)
{
  // Constructor

  if(0 == data) {
    _zombie=kTRUE ;
    cout << "RooFitContext: cannot create without valid dataset" << endl;
    return;
  }
  if(0 == pdf) {
    _zombie=kTRUE ;
    cout << "RooFitContext: cannot create without valid PDF" << endl;
    return;
  }

  // Clone data 
  if (cloneData) {

    // Check if the fit ranges of the dependents in the data and in the PDF are consistent
    RooArgSet* pdfDepSet = pdf->getDependents(data) ;
    const RooArgSet* dataDepSet = data->get() ;
    TIterator* iter = pdfDepSet->createIterator() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)iter->Next()) {
      RooRealVar* pdfReal = dynamic_cast<RooRealVar*>(arg) ;
      if (!pdfReal) continue ;

      RooRealVar* datReal = dynamic_cast<RooRealVar*>(dataDepSet->find(pdfReal->GetName())) ;
      if (!datReal) continue ;

      if (pdfReal->getFitMin()<(datReal->getFitMin()-1e-6)) {
	cout << "RooFitContxt: ERROR minimum of PDF variable " << arg->GetName() 
	     << "(" << pdfReal->getFitMin() << ") is smaller than that of " 
	     << arg->GetName() << " in the dataset (" << datReal->getFitMin() << ")" << endl ;
	_zombie=kTRUE ;
	return ;
      }

      if (pdfReal->getFitMax()>(datReal->getFitMax()+1e-6)) {
	cout << "RooFitContxt: ERROR maximum of PDF variable " << arg->GetName() 
	     << " is smaller than that of " << arg->GetName() << " in the dataset" << endl ;
	_zombie=kTRUE ;
	return ;
      }
      
    }
    delete iter ;
    
    // Copy data and strip entries lost by adjusted fit range
    _dataClone = ((RooAbsData*)data)->reduce(*pdfDepSet) ;

    delete pdfDepSet ;
  } else {
    _dataClone = (RooAbsData*) data ;
  }

  // Clone all PDF compents by copying all branch nodes
  RooArgSet tmp("PdfBranchNodeList") ;
  pdf->branchNodeServerList(&tmp) ;

  if (clonePdf) {
    _pdfCompList = (RooArgSet*) tmp.snapshot(kFALSE) ;
    
    // Find the top level PDF in the snapshot list
    _pdfClone = (RooAbsPdf*) _pdfCompList->find(pdf->GetName()) ;

  } else {
    _pdfCompList = (RooArgSet*) tmp.Clone() ;
    _pdfClone = (RooAbsPdf*)pdf ;
  }

  // Attach PDF to data set
  _pdfClone->attachDataSet(*_dataClone) ;
  _pdfClone->resetErrorCounters() ;

  // Cache parameter list  
  RooArgSet* paramSet = _pdfClone->getParameters(_dataClone) ;
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
  while(arg=(RooAbsArg*)pIter->Next()) {
    if (!arg->IsA()->InheritsFrom(RooAbsRealLValue::Class())) {
      cout << "RooFitContext::RooFitContext: removing parameter " << arg->GetName() 
	   << " from list because it is not of type RooRealVar" << endl ;
      _floatParamList->remove(*arg) ;
    }
  }

  _nPar      = _floatParamList->getSize() ;  
  delete pIter ;

  // Store the original leaf node list
  pdf->leafNodeServerList(&_origLeafNodeList) ;  

  // Store normalization set
  _normSet = (RooArgSet*) data->get()->snapshot(kFALSE) ;

  // Remove projected dependents from normalization set
  if (projDeps) {
    _projDeps = (RooArgSet*) projDeps->snapshot(kFALSE) ;
    _normSet->remove(*_projDeps,kTRUE,kTRUE) ;
  }
}



RooFitContext::~RooFitContext() 
{
  // Destructor
  delete _pdfCompList ;
  if (_ownData) {
    delete _dataClone ;
  }
  if (_projDeps) delete _projDeps ;
  delete _floatParamList ;
  delete _constParamList ;
  delete _normSet ;
}


void RooFitContext::printToStream(ostream &os, PrintOption opt, TString indent) const
{
  // Print contents 
  os << "DataSet clone:" << endl ;
  _dataClone->printToStream(os,opt,indent) ;

  os << indent << "PDF clone:" << endl ;
  _pdfClone->printToStream(os,opt,indent) ;

  os << indent << "PDF component list:" << endl ;
  _pdfCompList->printToStream(os,opt,indent) ;

  os << indent << "Parameter list:" << endl ;
  _constParamList->printToStream(os,opt,indent) ;
  _floatParamList->printToStream(os,opt,indent) ;

  return ;
}


Double_t RooFitContext::getPdfParamVal(Int_t index)
{
  // Access PDF parameter value by ordinal index (needed by MINUIT)
  return ((RooRealVar*)_floatParamList->at(index))->getVal() ;
}


Double_t RooFitContext::getPdfParamErr(Int_t index)
{
  // Access PDF parameter error by ordinal index (needed by MINUIT)
  return ((RooRealVar*)_floatParamList->at(index))->getError() ;  
}


Bool_t RooFitContext::setPdfParamVal(Int_t index, Double_t value, Bool_t verbose)
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


void RooFitContext::setPdfParamErr(Int_t index, Double_t value)
{
  // Modify PDF parameter error by ordinal index (needed by MINUIT)
  ((RooRealVar*)_floatParamList->at(index))->setError(value) ;    
}


void RooFitContext::setPdfParamErr(Int_t index, Double_t loVal, Double_t hiVal) 
{
  // Modify PDF parameter error by ordinal index (needed by MINUIT)
  ((RooRealVar*)_floatParamList->at(index))->setAsymError(loVal,hiVal) ;    
}


Double_t RooFitContext::getVal(Int_t evt) const 
{
  _dataClone->get(evt) ;
  return _pdfClone->getVal(_normSet) ; // WVE modified
}


Bool_t RooFitContext::optimize(Bool_t doPdf, Bool_t doData, Bool_t doCache) 
{
  // PDF/Dataset optimizer entry point

  // Sync PDF state by evaluating it once now

  // Find PDF nodes that can be cached in the data set
  RooArgSet cacheList("cacheList") ;

  if (doCache) {
    RooArgSet branchList("branchList") ;
    _pdfClone->setOperMode(RooAbsArg::ADirty) ;
    _pdfClone->branchNodeServerList(&branchList) ;
    TIterator* bIter = branchList.createIterator() ;
    RooAbsArg* branch ;
    while(branch=(RooAbsArg*)bIter->Next()) {
      if (branch->dependsOn(*_dataClone->get())) {
	branch->setOperMode(RooAbsArg::ADirty) ;
      }
    }
    delete bIter ;
  }

  if (doPdf) {
    findCacheableBranches(_pdfClone,_dataClone,cacheList) ;

    // Add cached branches from the data set
    _dataClone->cacheArgs(cacheList,_normSet) ;
  }


  if (doData) {
    // Find unused/unnecessary branches from the data set
    RooArgSet pruneList("pruneList") ;
    findUnusedDataVariables(_pdfClone,_dataClone,pruneList) ;

    if (doPdf) {
      findRedundantCacheServers(_pdfClone,_dataClone,cacheList,pruneList) ;
    }    

    if (pruneList.getSize()!=0) {
      // Created trimmed list of data variables
      RooArgSet newVarList(*_dataClone->get()) ;
      TIterator* iter = pruneList.createIterator() ;
      RooAbsArg* arg ;
      RooArgSet trimList ;
      while (arg = (RooAbsArg*) iter->Next()) {
	newVarList.remove(*arg) ;      
	trimList.add(*arg) ;
      }      
      delete iter ;
      if (trimList.getSize()>0) {
	cout << "RooFitContext::optimizePDF: dropping unused variables from dataset: " ;
	trimList.Print("1") ;
      }

      // Create trimmed data set
      RooAbsData *trimData = _dataClone->reduceEng(newVarList,0,kTRUE) ;
      
      // Unattach PDF clone from previous dataset
      _pdfClone->recursiveRedirectServers(_origLeafNodeList,kFALSE);
      
      // Reattach PDF clone to newly trimmed dataset
      _pdfClone->attachDataSet(*trimData) ;

      // Refresh normSet pointer (same contents) to force a normalization sync in all PDF components
      delete _normSet ;
      _normSet = (RooArgSet*) _dataClone->get()->snapshot(kFALSE) ;      
      // Remove projected dependents from normalization set
      if (_projDeps) {
	_normSet->remove(*_projDeps,kTRUE,kTRUE) ;
      }
      

      // WVE --- Is this still necessary now that we use RooNameSets? YES!!! --------
      //         Forces actual calculation of normalization of cached 
      //         variables while this is still posible
      TIterator* pcIter = _pdfCompList->createIterator() ;
      while(arg=(RooAbsArg*)pcIter->Next()){
	if (arg->IsA()->InheritsFrom(RooAbsPdf::Class())) {	  
	  ((RooAbsPdf*)arg)->getVal(_normSet) ; 
	}
      }
      delete pcIter ;
      //----------------------------------------------------------------------

      // Substitute new data for old data 
      if (_ownData) delete _dataClone ;
      _dataClone = trimData ;
      _ownData = kTRUE ;

      // Update _lastDataSet in cached variables to new trimmed dataset
      if (doCache) {
	TIterator* cIter = cacheList.createIterator() ;
	RooAbsArg *cacheArg ;
	while(cacheArg=(RooAbsArg*)cIter->Next()){
	  ((RooAbsReal*)cacheArg)->getVal(_normSet) ;
	}
	delete cIter ;

	_dataClone->setDirtyProp(kFALSE) ;
      }    

    }
  }

  // WVE --- Is this still necessary to do this last, now that we a fixed RooRealIntegral --------
  // This must be done last, otherwise the normalization of cached PDFs will not be calculated correctly
  if (doCache && doPdf) {
    TIterator* cIter = cacheList.createIterator() ;
    RooAbsArg *cacheArg ;
    while(cacheArg=(RooAbsArg*)cIter->Next()){
      cacheArg->setOperMode(RooAbsArg::AClean) ;
    }
    delete cIter ;
  }    
  //-----------------------------------------------------------------------------

  return kFALSE ;
}




Bool_t RooFitContext::findCacheableBranches(RooAbsPdf* pdf, RooAbsData* dset, 
					    RooArgSet& cacheList) 
{
  // Find branch PDFs with all-constant parameters, and add them
  // to the dataset cache list

  TIterator* sIter = pdf->serverIterator() ;
  RooAbsPdf* server ;

  while(server=(RooAbsPdf*)sIter->Next()) {
    if (server->isDerived()) {
      // Check if this branch node is eligible for precalculation
      Bool_t canOpt(kTRUE) ;

      RooArgSet* branchParamList = server->getParameters(dset) ;
      TIterator* pIter = branchParamList->createIterator() ;
      RooAbsArg* param ;
      while(param = (RooAbsArg*)pIter->Next()) {
	if (!param->isConstant()) canOpt=kFALSE ;
      }
      delete pIter ;
      delete branchParamList ;

      if (canOpt) {
	cout << "RooFitContext::optimizePDF: component PDF " 
	     << server->GetName() << " will be cached" << endl ;

	// Add to cache list
	cacheList.add(*server) ;

      } else {
	// Recurse if we cannot optimize at this level
	findCacheableBranches(server,dset,cacheList) ;
      }
    }
  }
  delete sIter ;
  return kFALSE ;
}



void RooFitContext::findUnusedDataVariables(RooAbsPdf* pdf,RooAbsData* dset,RooArgSet& pruneList) 
{
  TIterator* vIter = dset->get()->createIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*) vIter->Next()) {
    if (!pdf->dependsOn(*arg)) pruneList.add(*arg) ;
  }
  delete vIter ;
}


void RooFitContext::findRedundantCacheServers(RooAbsPdf* pdf,RooAbsData* dset,RooArgSet& cacheList, RooArgSet& pruneList) 
{
  TIterator* vIter = dset->get()->createIterator() ;
  RooAbsArg *var ;
  while (var=(RooAbsArg*) vIter->Next()) {
    if (allClientsCached(var,cacheList)) {
      pruneList.add(*var) ;
    }
  }
  delete vIter ;
}



Bool_t RooFitContext::allClientsCached(RooAbsArg* var, RooArgSet& cacheList)
{
  Bool_t ret(kTRUE), anyClient(kFALSE) ;

  TIterator* cIter = var->valueClientIterator() ;    
  RooAbsArg* client ;
  while (client=(RooAbsArg*) cIter->Next()) {
    anyClient = kTRUE ;
    if (!cacheList.find(client->GetName())) {
      // If client is not cached recurse
      ret &= allClientsCached(client,cacheList) ;
    }
  }
  delete cIter ;

  return anyClient?ret:kFALSE ;
}



RooFitResult* RooFitContext::fit(Option_t *fitOptions, Option_t* optOptions) 
{
  // Setup and perform MINUIT fit of PDF to dataset

  if (_zombie) return 0 ;

  // Parse our fit options string
  TString fitOpts= fitOptions;
  fitOpts.ToLower();
  Bool_t verbose       =!fitOpts.Contains("q") ;
  Bool_t migradOnly    = fitOpts.Contains("m") ;
  Bool_t estimateSteps = fitOpts.Contains("s") ;
  Bool_t performHesse  = fitOpts.Contains("h") ;
  Bool_t saveLog       = fitOpts.Contains("l") ;
        _verboseFit    = fitOpts.Contains("v") ;
  Bool_t profileTimer  = fitOpts.Contains("t") ;
         _extendedMode = fitOpts.Contains("e") ;
  Bool_t doStrat0      = fitOpts.Contains("0") ;
  Bool_t doSaveResult  = fitOpts.Contains("r") ;

  // Parse our optimizer options string
  TString optOpts= optOptions;
  optOpts.ToLower();
  Bool_t doOptPdf      = optOpts.Contains("p") ;
  Bool_t doOptData     = optOpts.Contains("d") ;
  Bool_t doOptCache    = optOpts.Contains("c") ;

  // Create fit result container if so requested
  RooFitResult* fitRes(0) ;

  if (doSaveResult) {
    TString name("fit(") ;
    name.Append(_pdfClone->GetName()) ;
    name.Append(",") ;
    name.Append(_dataClone->GetName()) ;
    name.Append(")") ;
    TString title("Result of fit of PDF ") ;
    title.Append(_pdfClone->GetName()) ;
    title.Append(" to dataset ") ;
    title.Append(_dataClone->GetName()) ;
    fitRes = new RooFitResult(name,title) ;
  }

  // Check if an extended ML fit is possible
  if(_extendedMode) {
    if(!_pdfClone->canBeExtended()) {
      cout << _pdfClone->GetName() << "::fitTo: this PDF does not support extended "
           << "maximum likelihood fits" << endl;
      return 0;
    }
    if(verbose) {
      cout << _pdfClone->GetName() << "::fitTo: will use extended maximum likelihood" << endl;
    }
  } else {
    if (_pdfClone->mustBeExtended()) {
      cout << _pdfClone->GetName() << "::fitTo: this PDF can only be used for extended "
           << "maximum likelihood fits" << endl;
      return 0;      
    }
  }

  // Check if there are any unprotected multiple occurrences of dependents
  if (_pdfClone->recursiveCheckDependents(_dataClone->get())) {
    cout << "RooFitContext::fit: Error in PDF dependents, abort" << endl ;
    return 0 ;
  }

  // Run the optimizer if requested
  if (doOptPdf||doOptData||doOptCache) optimize(doOptPdf,doOptData,doOptCache) ;

  // Save constant and initial parameters 
  if (doSaveResult) {
    fitRes->setConstParList(*_constParamList) ;
    fitRes->setInitParList(*_floatParamList) ;
  }

  // Create a log file if requested
  if(saveLog) {
    TString logname= fName;
    logname.Append(".log");
    _logfile= new ofstream(logname.Data());
    if(_logfile && _logfile->good()) {
      cout << fName << "::fitTo: saving fit log to " << logname << endl;
    } else {
      cout << fName << "::fitTo: unable to open logfile " << logname << endl;
      _logfile= 0;
      saveLog= kFALSE;
    }
  } else {
    _logfile = 0 ;
  }

  // Start a profiling timer if requested
  TStopwatch timer;
  if(profileTimer) timer.Start();

  // Initialize MINUIT
  Int_t nPar= _floatParamList->getSize();
  Double_t params[100], arglist[100];

  if (_theFitter) delete _theFitter ;
  _theFitter = new TFitter(nPar*2) ; //WVE Kludge, nPar*2 works around TMinuit memory allocation bug
  _theFitter->SetObjectFit(this) ;

  _theFitter->Clear();

  // Be quiet during the setup
  //arglist[0] = -1;
  //_theFitter->ExecuteCommand("SET PRINT",arglist,1);
  //_theFitter->ExecuteCommand("SET NOWARNINGS",arglist,0);

  // Tell MINUIT to use our global glue function
  _theFitter->SetFCN(RooFitGlue);

  // Use +0.5 for 1-sigma errors
  arglist[0]= 0.5;
  _theFitter->ExecuteCommand("SET ERR",arglist,1);

  // Declare our parameters
  Int_t index(0), nFree(nPar);
  for(index= 0; index < nPar; index++) {
    RooRealVar *par= dynamic_cast<RooRealVar*>(_floatParamList->at(index)) ;

    Double_t pstep(0) ;
    Double_t pmin(0) ;
    Double_t pmax(0) ;

    if(!par->isConstant()) {

      // Verify that floating parameter is indeed of type RooRealVar 
      if (!par->IsA()->InheritsFrom(RooRealVar::Class())) {
	cout << "RooFitContext::fit: Error, non-constant parameter " << par->GetName() 
	     << " is not of type RooRealVar" << endl ;
	return 0 ;
      }

      // Set the limits, if not infinite
      if (par->hasFitMin() && par->hasFitMax()) {
	pmin = par->getFitMin();
	pmax = par->getFitMax();
      }
      
      // Calculate step size
      pstep= par->getError();
      if(pstep <= 0) {
	// Floating parameter without error estitimate
	if (par->hasFitMin() && par->hasFitMax()) {
	  pstep= 0.1*(pmax-pmin);
	} else {
// 	  pstep= 0.1*(par->getPlotMax()-par->getPlotMin()) ;
// 	  if (pstep<=0) {
	    pstep=1 ;
// 	  }
	}						  
	if(!estimateSteps && verbose) {
	  cout << "*** WARNING: no initial error estimate available for "
	       << par->GetName() << ": using " << pstep << endl;
	}
      } 
      
    } else {
      pmin = par->getVal() ;
      pmax = par->getVal() ;      
    }

    _theFitter->SetParameter(index, par->GetName(), par->getVal(),
			     pstep, pmin, pmax);

    if(par->isConstant()) {
      // Declare fixed parameters (not necessary if range is zero)
      _theFitter->FixParameter(index);
      nFree--;
    }
  }

  // Now be verbose if requested
  if(verbose) {
    arglist[0] = 1;
    _theFitter->ExecuteCommand("SET PRINT",arglist,1);
    _theFitter->ExecuteCommand("SET WARNINGS",arglist,1);
  }

  // Reset the *largest* negative log-likelihood value we have seen so far
  _maxNLL= 0;

  // Do the fit
  arglist[0]= 250*nFree; // maximum iterations
  arglist[1]= 1.0;       // tolerance
  Int_t status(0);
  if(estimateSteps) {
    // Use HESSE to get reasonable starting step sizes for MIGRAD
    status= _theFitter->ExecuteCommand("HESSE",arglist,1);
  }

  // Always use MIGRAD unless an earlier step failed
  if(status == 0) {
    if (doStrat0) {
      Double_t stratArg(0.0) ;
      if (verbose)  cout << "RooFitContext: executing SET STRATEGY 0" << endl ;
      _theFitter->ExecuteCommand("SET STR",&stratArg,1) ;
    }

    if (verbose)  cout << "RooFitContext: executing MIGRAD" << endl ;
    status= _theFitter->ExecuteCommand("MIGRAD",arglist,2);

    if (doStrat0) {
      Double_t stratArg(1.0) ;
      if (verbose)  cout << "RooFitContext: executing SET STRATEGY 1" << endl ;
      _theFitter->ExecuteCommand("SET STR",&stratArg,1) ;
    }
  }

  // If the fit suceeded, follow with a HESSE analysis if requested
  if(status == 0 && performHesse) {
    arglist[0]= 250*nFree; // maximum iterations
    if (verbose)  cout << "RooFitContext: executing HESSE" << endl ;
    status= _theFitter->ExecuteCommand("HESSE",arglist,1);
  }

  // If the fit suceeded, follow with a MINOS analysis if requested
  if(status == 0 && !migradOnly) {
    arglist[0]= 250*nFree; // maximum iterations
    if (verbose)  cout << "RooFitContext: executing MINOS" << endl ;
    status= _theFitter->ExecuteCommand("MINOS",arglist,1);
  }

  // Get the fit results
  Double_t val,err,vlo,vhi, eplus, eminus, eparab, globcc;
  char buffer[10240];
  for(index= 0; index < nPar; index++) {
    _theFitter->GetParameter(index, buffer, val, err, vlo, vhi);
    setPdfParamVal(index, val);
    _theFitter->GetErrors(index, eplus, eminus, eparab, globcc);

    // Set the parabolic error
    setPdfParamErr(index, eparab);

    if(eplus > 0 || eminus < 0) {
      // Store the asymmetric error, if it is available
      setPdfParamErr(index, eminus,eplus);
    }
  }

  if(doSaveResult) { // Get the minimum function value if requested
    Double_t edm, errdef, minVal;
    Int_t nvpar, nparx;
    _theFitter->GetStats(minVal, edm, errdef, nvpar, nparx);
    fitRes->setStatus(status) ;
    fitRes->setMinNLL(minVal) ;
    fitRes->setEDM(edm) ;    
    fitRes->setFinalParList(*_floatParamList) ;
    fitRes->fillCorrMatrix() ;
  }

  // Print the time used, if requested
  if(profileTimer) {
    timer.Stop();
    cout << fName << "::fitTo: ";
    timer.Print();
  }

  // Close the log file now
  if(saveLog) {
    _logfile->close();
    delete _logfile;
    _logfile= 0;
  }

  return fitRes ;
}



Double_t RooFitContext::nLogLikelihood(Bool_t extended, Int_t nObserved) const 
{
  // Return the likelihood of this PDF for the given dataset
  Double_t result(0);
  const RooArgSet *values = _dataClone->get() ;
  if(!values) {
    cout << _dataClone->GetName() << "::nLogLikelihood: cannot get values from dataset " << endl ;
    return 0.0;
    }


  Stat_t events= _dataClone->numEntries();
  for(Int_t index= 0; index<events; index++) {

    // get the data values for this event
    _dataClone->get(index);

    Double_t term = _dataClone->weight() * _pdfClone->getLogVal(_normSet); // WVE modified
    //if(term == 0 && _dataClone->weight()) return 0;
    result-= term;
  }

  // include the extended maximum likelihood term, if requested
  nObserved = (nObserved==-1) ? _dataClone->numEntries() : nObserved ;
  if(extended) {
    result+= _pdfClone->extendedTerm(nObserved);
  }

  return result;
}




TH2F* RooFitContext::plotNLLContours(RooRealVar& var1, RooRealVar& var2, Double_t n1, Double_t n2, Double_t n3) 
{
  // Verify that both variables are floating parameters of PDF
  Int_t index1= _floatParamList->index(&var1);
  if(index1 < 0) {
    cout << "RooFitContext::plotNLLContours(" << GetName() 
	 << ") ERROR: " << var1.GetName() << " is not a floating parameter of PDF " << _pdfClone->GetName() << endl ;
    return 0;
  }

  Int_t index2= _floatParamList->index(&var2);
  if(index2 < 0) {
    cout << "RooFitContext::plotNLLContours(" << GetName() 
	 << ") ERROR: " << var2.GetName() << " is not a floating parameter of PDF " << _pdfClone->GetName() << endl ;
    return 0;
  }

  // Ensure function is minimized. Perform MIGRAD with strategy 0 only.
  fit("m0","cpds") ;

  // create and draw a frame
  TH2F *frame = var1.createHistogram("contourPlot", var2, "-log(likelihood)") ;
  frame->SetStats(kFALSE);

  // draw a point at the current parameter values
  TMarker *point= new TMarker(var1.getVal(), var2.getVal(), 8);

  // remember our original value of ERRDEF
  Double_t errdef= gMinuit->fUp;

  TGraph* graph1(0) ;
  if(n1 > 0) {
    // set the value corresponding to an n1-sigma contour
    gMinuit->SetErrorDef(n1*n1*errdef);
    // calculate and draw the contour
    graph1= (TGraph*)gMinuit->Contour(25, index1, index2);
  }

  TGraph* graph2(0) ;
  if(n2 > 0) {
    // set the value corresponding to an n1-sigma contour
    gMinuit->SetErrorDef(n2*n2*errdef);
    // calculate and draw the contour
    graph2= (TGraph*)gMinuit->Contour(25, index1, index2);
    graph2->SetLineStyle(2);
  }

  TGraph* graph3(0) ;
  if(n3 > 0) {
    // set the value corresponding to an n1-sigma contour
    gMinuit->SetErrorDef(n3*n3*errdef);
    // calculate and draw the contour
    graph3= (TGraph*)gMinuit->Contour(25, index1, index2);
    graph3->SetLineStyle(3);
  }
  // restore the original ERRDEF
  gMinuit->SetErrorDef(errdef);


  // Draw all objects
  frame->Draw();
  point->Draw();
  if (graph1) graph1->Draw();
  if (graph2) graph2->Draw();
  if (graph3) graph3->Draw();

  return frame;
}




void RooFitGlue(Int_t &np, Double_t *gin,
                Double_t &f, Double_t *par, Int_t flag)
{
  // Static function that interfaces minuit with RooFitContext

  // Retrieve fit context and its components
  RooFitContext* context = (RooFitContext*) _theFitter->GetObjectFit() ;
  ofstream* logf   = context->logfile() ;
  Double_t& maxNLL = context->maxNLL() ;
  Bool_t verbose   = context->_verboseFit ;

  // Set the parameter values for this iteration
  Int_t nPar= context->getNPar();
  for(Int_t index= 0; index < nPar; index++) {
    if (logf) (*logf) << par[index] << " " ;
    context->setPdfParamVal(index, par[index],verbose);
  }

  // Calculate the negative log-likelihood for these parameters
  f= context->nLogLikelihood(context->_extendedMode);
  if (f==0) {
    // if any event has a prob <=0 return a flat likelihood 
    // at the max value we have seen so far
    f = maxNLL ;
  } else if (f>maxNLL) {
    maxNLL = f ;
  }

  // Optional logging
  if (logf) (*logf) << setprecision(15) << f << setprecision(4) << endl;
  if (verbose) cout << "\nprevNLL = " << setprecision(10) << f << setprecision(4) << "  " ;
}



