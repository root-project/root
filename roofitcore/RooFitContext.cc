/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooFitContext.cc,v 1.18 2001/08/18 02:13:10 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
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
#include "TStopwatch.h"
#include "TFitter.h"
#include "TMinuit.h"
#include "RooFitCore/RooFitContext.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooResolutionModel.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFitResult.hh"

ClassImp(RooFitContext)
;

static TVirtualFitter *_theFitter(0);


RooFitContext::RooFitContext(const RooDataSet* data, const RooAbsPdf* pdf, Bool_t cloneData, Bool_t clonePdf) :
  TNamed(*pdf), _origLeafNodeList("origLeafNodeList"), _extendedMode(kFALSE), _doOptCache(kFALSE),
  _ownData(cloneData)
{
  // Constructor

  if(0 == data) {
    cout << "RooFitContext: cannot create without valid dataset" << endl;
    return;
  }
  if(0 == pdf) {
    cout << "RooFitContext: cannot create without valid PDF" << endl;
    return;
  }

  // Clone data 
  if (cloneData) {
    _dataClone = new RooDataSet(*data) ;
  } else {
    _dataClone = (RooDataSet*) data ;
  }

  // Clone all PDF compents by copying all branch nodes
  RooArgSet tmp("PdfBranchNodeList") ;
  pdf->branchNodeServerList(&tmp) ;

  if (clonePdf) {
    _pdfCompList = tmp.snapshot(kFALSE) ;
    
    // Find the top level PDF in the snapshot list
    _pdfClone = (RooAbsPdf*) _pdfCompList->FindObject(pdf->GetName()) ;

  } else {
    _pdfCompList = (RooArgSet*) tmp.Clone() ;
    _pdfClone = (RooAbsPdf*)pdf ;
  }

  // Attach PDF to data set
  _pdfClone->attachDataSet(*_dataClone) ;
  _pdfClone->resetErrorCounters() ;

  // Cache parameter list
  RooArgSet* paramList = _pdfClone->getParameters(_dataClone) ;

  _floatParamList = paramList->selectByAttrib("Constant",kFALSE) ; 
  _floatParamList->Sort() ;
  _floatParamList->SetName("floatParamList") ;

  _constParamList = paramList->selectByAttrib("Constant",kTRUE) ;
  _constParamList->Sort() ;
  _constParamList->SetName("constParamList") ;

  delete paramList ;

  // Remove all non-RooRealVar parameters from list (MINUIT cannot handle them)
  TIterator* pIter = _floatParamList->MakeIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)pIter->Next()) {
    if (!arg->IsA()->InheritsFrom(RooAbsRealLValue::Class())) {
      cout << "RooFitContext::RooFitContext: removing parameter " << arg->GetName() 
	   << " from list because it is not of type RooRealVar" << endl ;
      _floatParamList->remove(*arg) ;
    }
  }
  _nPar      = _floatParamList->GetSize() ;  
  delete pIter ;

  // Store the original leaf node list
  pdf->leafNodeServerList(&_origLeafNodeList) ;  
}



RooFitContext::~RooFitContext() 
{
  // Destructor
  delete _pdfCompList ;
  if (_ownData) {
    delete _dataClone ;
  }
  delete _floatParamList ;
  delete _constParamList ;
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
  return ((RooRealVar*)_floatParamList->At(index))->getVal() ;
}


Double_t RooFitContext::getPdfParamErr(Int_t index)
{
  // Access PDF parameter error by ordinal index (needed by MINUIT)
  return ((RooRealVar*)_floatParamList->At(index))->getError() ;  
}


Bool_t RooFitContext::setPdfParamVal(Int_t index, Double_t value, Bool_t verbose)
{
  // Modify PDF parameter value by ordinal index (needed by MINUIT)
  RooRealVar* par = (RooRealVar*)_floatParamList->At(index) ;

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
  ((RooRealVar*)_floatParamList->At(index))->setError(value) ;    
}


Double_t RooFitContext::getVal(Int_t evt) const 
{
  _dataClone->get(evt) ;
  return _pdfClone->getVal(_dataClone->get()) ;
}


Bool_t RooFitContext::optimize(Bool_t doPdf, Bool_t doData, Bool_t doCache) 
{
  // PDF/Dataset optimizer entry point

  // Find PDF nodes that can be cached in the data set
  RooArgSet cacheList("cacheList") ;

  if (doCache) {
    RooArgSet branchList("branchList") ;
    _pdfClone->setOperMode(RooAbsArg::ADirty) ;
    _pdfClone->branchNodeServerList(&branchList) ;
    TIterator* bIter = branchList.MakeIterator() ;
    RooAbsArg* branch ;
    while(branch=(RooAbsArg*)bIter->Next()) {
      branch->setOperMode(RooAbsArg::ADirty) ;
    }
    delete bIter ;
  }

  if (doPdf) {
    findCacheableBranches(_pdfClone,_dataClone,cacheList) ;
  
    // Add cached branches from the data set
    _dataClone->cacheArgs(cacheList) ;
  }


  if (doData) {
    // Find unused/unnecessary branches from the data set
    RooArgSet pruneList("pruneList") ;
    findUnusedDataVariables(_pdfClone,_dataClone,pruneList) ;

    if (doPdf)
      findRedundantCacheServers(_pdfClone,_dataClone,cacheList,pruneList) ;
    
    if (pruneList.GetSize()!=0) {
      // Created trimmed list of data variables
      RooArgSet newVarList(*_dataClone->get()) ;
      TIterator* iter = pruneList.MakeIterator() ;
      RooAbsArg* arg ;
      while (arg = (RooAbsArg*) iter->Next()) {
	cout << "RooFitContext::optimizePDF: dropping variable " 
	     << arg->GetName() << " from context data set" << endl ;
	newVarList.remove(*arg) ;      
      }      
      delete iter ;
      
      // Create trimmed data set
      RooDataSet *trimData = new RooDataSet("trimData","Reduced data set for fit context",
					    _dataClone,newVarList,kTRUE) ;
      
      // Unattach PDF clone from previous dataset
      _pdfClone->recursiveRedirectServers(_origLeafNodeList,kFALSE);
      
      // Reattach PDF clone to newly trimmed dataset
      _pdfClone->attachDataSet(*trimData) ;
      
      // Make sure PDF releases all handles to old data set before deleting it
      TIterator* pcIter = _pdfCompList->MakeIterator() ;
      while(arg=(RooAbsArg*)pcIter->Next()){
	if (arg->IsA()->InheritsFrom(RooAbsReal::Class())) {
	  ((RooAbsReal*)arg)->getVal(trimData->get()) ;
	}
      }
      delete pcIter ;

      // Substitute new data for old data 
      if (_ownData) delete _dataClone ;
      _dataClone = trimData ;
      _ownData = kTRUE ;

      // Update _lastDataSet in cached variables to new trimmed dataset
      if (doCache) {
	TIterator* cIter = cacheList.MakeIterator() ;
	RooAbsArg *cacheArg ;
	while(cacheArg=(RooAbsArg*)cIter->Next()){
	  ((RooAbsReal*)cacheArg)->getVal(_dataClone->get()) ;
	}
	delete cIter ;

	_dataClone->setDirtyProp(kFALSE) ;
      }    

    }
  }

  // This must be done last, otherwise the normalization of cached PDFs will not be calculated correctly
  if (doCache && doPdf) {
    TIterator* cIter = cacheList.MakeIterator() ;
    RooAbsArg *cacheArg ;
    while(cacheArg=(RooAbsArg*)cIter->Next()){
      cacheArg->setOperMode(RooAbsArg::AClean) ;
    }
    delete cIter ;
  }    

  return kFALSE ;
}




Bool_t RooFitContext::findCacheableBranches(RooAbsPdf* pdf, RooDataSet* dset, 
					    RooArgSet& cacheList) 
{
  // Find branch PDFs with all-constant parameters, and add them
  // to the dataset cache list

  TIterator* sIter = pdf->serverIterator() ;
  RooAbsPdf* server ;

  while(server=(RooAbsPdf*)sIter->Next()) {
//     if (server->isDerived() && server->IsA()->InheritsFrom(RooAbsPdf::Class())) {
    if (server->isDerived()) {
      // Check if this branch node is eligible for precalculation
      Bool_t canOpt(kTRUE) ;

      RooArgSet* branchParamList = server->getParameters(dset) ;
      TIterator* pIter = branchParamList->MakeIterator() ;
      RooAbsArg* param ;
      while(param = (RooAbsArg*)pIter->Next()) {
	if (!param->isConstant()) canOpt=kFALSE ;
      }
      delete pIter ;
      delete branchParamList ;

      if (canOpt) {
	cout << "RooFitContext::optimizePDF: component PDF " << server->GetName() 
	     << " of PDF " << pdf->GetName() << " will be cached" << endl ;

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



void RooFitContext::findUnusedDataVariables(RooAbsPdf* pdf,RooDataSet* dset,RooArgSet& pruneList) 
{
  TIterator* vIter = dset->get()->MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*) vIter->Next()) {
    if (!pdf->dependsOn(*arg)) pruneList.add(*arg) ;
  }
  delete vIter ;
}


void RooFitContext::findRedundantCacheServers(RooAbsPdf* pdf,RooDataSet* dset,RooArgSet& cacheList, RooArgSet& pruneList) 
{
  TIterator* vIter = dset->get()->MakeIterator() ;
  RooAbsArg *var ;
  while (var=(RooAbsArg*) vIter->Next()) {
    if (allClientsCached(var,cacheList)) pruneList.add(*var) ;
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
    if (!cacheList.FindObject(client)) {
      // If client is not cached recurse
      ret = allClientsCached(client,cacheList) ;
    }
  }
  delete cIter ;

  return anyClient?ret:kFALSE ;
}



const RooFitResult* RooFitContext::fit(Option_t *fitOptions, Option_t* optOptions) 
{
  // Setup and perform MINUIT fit of PDF to dataset

  // Parse our fit options string
  TString fitOpts= fitOptions;
  fitOpts.ToLower();
  Bool_t verbose       =!fitOpts.Contains("q") ;
  Bool_t migradOnly    = fitOpts.Contains("m") ;
  Bool_t estimateSteps = fitOpts.Contains("s") ;
  Bool_t performHesse  = fitOpts.Contains("h") ;
  Bool_t saveLog       = fitOpts.Contains("l") ;
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
  if (doSaveResult) fitRes = new RooFitResult ;

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
  }

  // Check if there are any unprotected multiple occurrences of dependents
  if (_pdfClone->checkDependents(_dataClone->get())) {
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
  Int_t nPar= _floatParamList->GetSize();
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
    RooRealVar *par= dynamic_cast<RooRealVar*>(_floatParamList->At(index)) ;

    Double_t pstep(0) ;
    Double_t pmin= par->getFitMin();
    Double_t pmax= par->getFitMax();

    if(!par->isConstant()) {

      // Verify that floating parameter is indeed of type RooRealVar 
      if (!par->IsA()->InheritsFrom(RooRealVar::Class())) {
	cout << "RooFitContext::fit: Error, non-constant parameter " << par->GetName() 
	     << " is not of type RooRealVar" << endl ;
	return 0 ;
      }

      // Calculate step size
      pstep= par->getError();
      if(pstep <= 0) {
	pstep= 0.1*(pmax-pmin);
	if(!estimateSteps && verbose) {
	  cout << "*** WARNING: no initial error estimate available for "
	       << par->GetName() << ": using " << pstep << endl;
	}
      }
    } 

    _theFitter->SetParameter(index, par->GetName(), par->getVal(),
			     pstep, pmin, pmax);

    if(par->isConstant() && (pmax > pmin) && (pstep > 0)) {
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
      _theFitter->ExecuteCommand("SET STR",&stratArg,1) ;
    }

    status= _theFitter->ExecuteCommand("MIGRAD",arglist,2);

    if (doStrat0) {
      Double_t stratArg(1.0) ;
      _theFitter->ExecuteCommand("SET STR",&stratArg,1) ;
    }
  }

  // If the fit suceeded, follow with a HESSE analysis if requested
  if(status == 0 && performHesse) {
    arglist[0]= 250*nFree; // maximum iterations
    status= _theFitter->ExecuteCommand("HESSE",arglist,1);
  }

  // If the fit suceeded, follow with a MINOS analysis if requested
  if(status == 0 && !migradOnly) {
    arglist[0]= 250*nFree; // maximum iterations
    status= _theFitter->ExecuteCommand("MINOS",arglist,1);
  }

  // Get the fit results
  Double_t val,err,vlo,vhi, eplus, eminus, eparab, globcc;
  char buffer[10240];
  for(index= 0; index < nPar; index++) {
    _theFitter->GetParameter(index, buffer, val, err, vlo, vhi);
    setPdfParamVal(index, val);
    _theFitter->GetErrors(index, eplus, eminus, eparab, globcc);
    if(eplus > 0 || eminus < 0) {
    // Use the average asymmetric error, if it is available
      setPdfParamErr(index, 0.5*(eplus-eminus));
    }
    else {
    // Otherwise, use the parabolic error
      setPdfParamErr(index, eparab);
    }
  }

  if(doSaveResult) { // Get the minimum function value if requested
    Double_t edm, errdef, minVal;
    Int_t nvpar, nparx;
    _theFitter->GetStats(minVal, edm, errdef, nvpar, nparx);
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


Double_t RooFitContext::nLogLikelihood(Bool_t dummy) const 
{
  // Return the likelihood of this PDF for the given dataset
  Double_t result(0);
  const RooArgSet *values = _dataClone->get() ;
  if(!values) {
    cout << _dataClone->GetName() << "::nLogLikelihood: cannot get values from dataset " << endl ;
    return 0.0;
    }

  Stat_t events= _dataClone->GetEntries();
  for(Int_t index= 0; index<events; index++) {

    // get the data values for this event
    _dataClone->get(index);

    Double_t term = _pdfClone->getLogVal(_dataClone->get());
    if(term == 0) return 0;
    result-= term;
  }

  // include the extended maximum likelihood term, if requested
  if(_extendedMode) {
    result+= _pdfClone->extendedTerm(events);
  }

  return result;
}



void RooFitGlue(Int_t &np, Double_t *gin,
                Double_t &f, Double_t *par, Int_t flag)
{
  // Static function that interfaces minuit with RooFitContext

  // Retrieve fit context and its components
  RooFitContext* context = (RooFitContext*) _theFitter->GetObjectFit() ;
  ofstream* logf   = context->logfile() ;
  Double_t& maxNLL = context->maxNLL() ;

  // Set the parameter values for this iteration
  Int_t nPar= context->getNPar();
  for(Int_t index= 0; index < nPar; index++) {
    if (logf) (*logf) << par[index] << " " ;
    context->setPdfParamVal(index, par[index],logf?kTRUE:kFALSE);
  }

  // Calculate the negative log-likelihood for these parameters
  f= context->nLogLikelihood();
  if (f==0) {
    // if any event has a prob <=0 return a flat likelihood 
    // at the max value we have seen so far
    f = maxNLL ;
  } else if (f>maxNLL) {
    maxNLL = f ;
  }

  // Optional logging
  if (logf) {
    (*logf) << setprecision(15) << f << setprecision(4) << endl;
    cout << "\nprevNLL = " << setprecision(10) << f << setprecision(4) << "  " ;
  }
}



