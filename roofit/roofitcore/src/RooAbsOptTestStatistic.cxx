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
// 
// BEGIN_HTML
// RooAbsOptTestStatistic is the abstract base class for test
// statistics objects that evaluate a function or PDF at each point of a given
// dataset.  This class provides generic optimizations, such as
// caching and precalculation of constant terms that can be made for
// all such quantities
//
// Implementations should define evaluatePartition(), which calculates the
// value of a (sub)range of the dataset and optionally combinedValue(),
// which combines the values calculated for each partition. If combinedValue()
// is not overloaded, the default implementation will add the partition results
// to obtain the combined result
//
// Support for calculation in partitions is needed to allow multi-core
// parallelized calculation of test statistics
// END_HTML
//
//

#include "RooFit.h"

#include "Riostream.h"
#include <string.h>


#include "RooAbsOptTestStatistic.h"
#include "RooMsgService.h"
#include "RooAbsPdf.h"
#include "RooAbsData.h"
#include "RooDataHist.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooErrorHandler.h"
#include "RooGlobalFunc.h"
#include "RooBinning.h"
#include "RooAbsDataStore.h"
#include "RooCategory.h"
#include "RooDataSet.h"
#include "RooProdPdf.h"
#include "RooAddPdf.h"
#include "RooProduct.h"
#include "RooRealSumPdf.h"
#include "RooTrace.h"
#include "RooVectorDataStore.h" 

using namespace std;

ClassImp(RooAbsOptTestStatistic)
;


//_____________________________________________________________________________
RooAbsOptTestStatistic:: RooAbsOptTestStatistic()
{
  // Default Constructor

  // Initialize all non-persisted data members

  _funcObsSet = 0 ;
  _funcCloneSet = 0 ;
  _funcClone = 0 ;

  _normSet = 0 ;
  _dataClone = 0 ;
  _projDeps = 0 ;

  _origFunc = 0 ;
  _origData = 0 ;

  _ownData = kTRUE ;
  _sealed = kFALSE ;
  _optimized = kFALSE ;
}



//_____________________________________________________________________________
RooAbsOptTestStatistic::RooAbsOptTestStatistic(const char *name, const char *title, RooAbsReal& real, RooAbsData& indata,
					       const RooArgSet& projDeps, const char* rangeName, const char* addCoefRangeName,
					       Int_t nCPU, RooFit::MPSplit interleave, Bool_t verbose, Bool_t splitCutRange, Bool_t /*cloneInputData*/) : 
  RooAbsTestStatistic(name,title,real,indata,projDeps,rangeName, addCoefRangeName, nCPU, interleave, verbose, splitCutRange),
  _projDeps(0),
  _sealed(kFALSE), 
  _optimized(kFALSE)
{
  // Constructor taking function (real), a dataset (data), a set of projected observables (projSet). If 
  // rangeName is not null, only events in the dataset inside the range will be used in the test
  // statistic calculation. If addCoefRangeName is not null, all RooAddPdf component of 'real' will be
  // instructed to fix their fraction definitions to the given named range. If nCPU is greater than
  // 1 the test statistic calculation will be paralellized over multiple processes. By default the data
  // is split with 'bulk' partitioning (each process calculates a contigious block of fraction 1/nCPU
  // of the data). For binned data this approach may be suboptimal as the number of bins with >0 entries
  // in each processing block many vary greatly thereby distributing the workload rather unevenly.
  // If interleave is set to true, the interleave partitioning strategy is used where each partition
  // i takes all bins for which (ibin % ncpu == i) which is more likely to result in an even workload.
  // If splitCutRange is true, a different rangeName constructed as rangeName_{catName} will be used
  // as range definition for each index state of a RooSimultaneous

  // Don't do a thing in master mode

  if (operMode()!=Slave) {
    _funcObsSet = 0 ;
    _funcCloneSet = 0 ;
    _funcClone = 0 ;
    _normSet = 0 ;
    _dataClone = 0 ;
    _projDeps = 0 ;    
    _origFunc = 0 ;
    _origData = 0 ;
    _ownData = kFALSE ;
    _sealed = kFALSE ;
    return ;
  }

  _origFunc = 0 ; //other._origFunc ;
  _origData = 0 ; // other._origData ;

  initSlave(real,indata,projDeps,rangeName,addCoefRangeName) ;
}

//_____________________________________________________________________________
RooAbsOptTestStatistic::RooAbsOptTestStatistic(const RooAbsOptTestStatistic& other, const char* name) : 
  RooAbsTestStatistic(other,name), _sealed(other._sealed), _sealNotice(other._sealNotice), _optimized(kFALSE)
{
  // Copy constructor

  // Don't do a thing in master mode
  if (operMode()!=Slave) {    

    _funcObsSet = 0 ;
    _funcCloneSet = 0 ;
    _funcClone = 0 ;
    _normSet = other._normSet ? ((RooArgSet*) other._normSet->snapshot()) : 0 ;   
    _dataClone = 0 ;
    _projDeps = 0 ;    
    _origFunc = 0 ;
    _origData = 0 ;
    _ownData = kFALSE ;
    return ;
  }
  
  _origFunc = 0 ; //other._origFunc ;
  _origData = 0 ; // other._origData ;
  _projDeps = 0 ;
  
  initSlave(*other._funcClone,*other._dataClone,other._projDeps?*other._projDeps:RooArgSet(),other._rangeName.c_str(),other._addCoefRangeName.c_str()) ;
}



//_____________________________________________________________________________
void RooAbsOptTestStatistic::initSlave(RooAbsReal& real, RooAbsData& indata, const RooArgSet& projDeps, const char* rangeName, 
				       const char* addCoefRangeName) 
{
  RooArgSet obs(*indata.get()) ;
  obs.remove(projDeps,kTRUE,kTRUE) ;


  // ******************************************************************
  // *** PART 1 *** Clone incoming pdf, attach to each other *
  // ******************************************************************

  // Clone FUNC
  _funcClone = (RooAbsReal*) real.cloneTree() ;
  _funcCloneSet = 0 ;

  // Attach FUNC to data set  
  _funcObsSet = _funcClone->getObservables(indata) ;

  if (_funcClone->getAttribute("BinnedLikelihood")) {
    _funcClone->setAttribute("BinnedLikelihoodActive") ;
  }

  // Reattach FUNC to original parameters  
  RooArgSet* origParams = (RooArgSet*) real.getParameters(indata) ;
  _funcClone->recursiveRedirectServers(*origParams) ;

    // Mark all projected dependents as such
  if (projDeps.getSize()>0) {
    RooArgSet *projDataDeps = (RooArgSet*) _funcObsSet->selectCommon(projDeps) ;
    projDataDeps->setAttribAll("projectedDependent") ;
    delete projDataDeps ;    
  }

  // If PDF is a RooProdPdf (with possible constraint terms)
  // analyze pdf for actual parameters (i.e those in unconnected constraint terms should be
  // ignored as here so that the test statistic will not be recalculated if those
  // are changed
  RooProdPdf* pdfWithCons = dynamic_cast<RooProdPdf*>(_funcClone) ;
  if (pdfWithCons) {
    
    RooArgSet* connPars = pdfWithCons->getConnectedParameters(*indata.get()) ;
    // Add connected parameters as servers
    _paramSet.removeAll() ;
    _paramSet.add(*connPars) ;
    delete connPars ;

  } else {
    // Add parameters as servers
    _paramSet.add(*origParams) ;
  }


  delete origParams ;

  // Store normalization set  
  _normSet = (RooArgSet*) indata.get()->snapshot(kFALSE) ;

  // Expand list of observables with any observables used in parameterized ranges
  RooAbsArg* realDep ;
  RooFIter iter = _funcObsSet->fwdIterator() ;
  while((realDep=iter.next())) {
    RooAbsRealLValue* realDepRLV = dynamic_cast<RooAbsRealLValue*>(realDep) ;
    if (realDepRLV && realDepRLV->isDerived()) {
      RooArgSet tmp2 ;
      realDepRLV->leafNodeServerList(&tmp2, 0, kTRUE) ;
      _funcObsSet->add(tmp2,kTRUE) ;
    }
  }



  // ******************************************************************
  // *** PART 2 *** Clone and adjust incoming data, attach to PDF     *
  // ******************************************************************
  
  // Check if the fit ranges of the dependents in the data and in the FUNC are consistent
  const RooArgSet* dataDepSet = indata.get() ;
  iter = _funcObsSet->fwdIterator() ;
  RooAbsArg* arg ;
  while((arg=iter.next())) {

    // Check that both dataset and function argument are of type RooRealVar
    RooRealVar* realReal = dynamic_cast<RooRealVar*>(arg) ;
    if (!realReal) continue ;
    RooRealVar* datReal = dynamic_cast<RooRealVar*>(dataDepSet->find(realReal->GetName())) ;
    if (!datReal) continue ;
    
    // Check that range of observables in pdf is equal or contained in range of observables in data
 
   if (!realReal->getBinning().lowBoundFunc() && realReal->getMin()<(datReal->getMin()-1e-6)) {
      coutE(InputArguments) << "RooAbsOptTestStatistic: ERROR minimum of FUNC observable " << arg->GetName() 
			    << "(" << realReal->getMin() << ") is smaller than that of " 
			    << arg->GetName() << " in the dataset (" << datReal->getMin() << ")" << endl ;
      RooErrorHandler::softAbort() ;
      return ;
    }
    
    if (!realReal->getBinning().highBoundFunc() && realReal->getMax()>(datReal->getMax()+1e-6)) {
      coutE(InputArguments) << "RooAbsOptTestStatistic: ERROR maximum of FUNC observable " << arg->GetName() 
			    << " is larger than that of " << arg->GetName() << " in the dataset" << endl ;
      RooErrorHandler::softAbort() ;
      return ;
    }
    
  }
  
  // Copy data and strip entries lost by adjusted fit range, _dataClone ranges will be copied from realDepSet ranges
  if (rangeName && strlen(rangeName)) {
    _dataClone = ((RooAbsData&)indata).reduce(RooFit::SelectVars(*_funcObsSet),RooFit::CutRange(rangeName)) ;  
//     cout << "RooAbsOptTestStatistic: reducing dataset to fit in range named " << rangeName << " resulting dataset has " << _dataClone->sumEntries() << " events" << endl ;
  } else {
    _dataClone = (RooAbsData*) indata.Clone() ;
  }
  _ownData = kTRUE ;  
 

  // ******************************************************************
  // *** PART 3 *** Make adjustments for fit ranges, if specified     *
  // ****************************************************************** 

  RooArgSet* origObsSet = real.getObservables(indata) ;
  RooArgSet* dataObsSet = (RooArgSet*) _dataClone->get() ;
  if (rangeName && strlen(rangeName)) {    
    cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName() << ") constructing test statistic for sub-range named " << rangeName << endl ;    
    //cout << "now adjusting observable ranges to requested fit range" << endl ;

    // Adjust FUNC normalization ranges to requested fitRange, store original ranges for RooAddPdf coefficient interpretation
    iter = _funcObsSet->fwdIterator() ;
    while((arg=iter.next())) {

      RooRealVar* realObs = dynamic_cast<RooRealVar*>(arg) ;
      if (realObs) {

	// If no explicit range is given for RooAddPdf coefficients, create explicit named range equivalent to original observables range
        if (!(addCoefRangeName && strlen(addCoefRangeName))) {
	  realObs->setRange(Form("NormalizationRangeFor%s",rangeName),realObs->getMin(),realObs->getMax()) ;
// 	  cout << "RAOTS::ctor() setting range " << Form("NormalizationRangeFor%s",rangeName) << " on observable " 
// 	       << realObs->GetName() << " to [" << realObs->getMin() << "," << realObs->getMax() << "]" << endl ;
	}

	// Adjust range of function observable to those of given named range
	realObs->setRange(realObs->getMin(rangeName),realObs->getMax(rangeName)) ;	
//  	cout << "RAOTS::ctor() setting normalization range on observable " 
//  	     << realObs->GetName() << " to [" << realObs->getMin() << "," << realObs->getMax() << "]" << endl ;

	// Adjust range of data observable to those of given named range
	RooRealVar* dataObs = (RooRealVar*) dataObsSet->find(realObs->GetName()) ;
	dataObs->setRange(realObs->getMin(rangeName),realObs->getMax(rangeName)) ;	
       
	// Keep track of list of fit ranges in string attribute fit range of original p.d.f.
	if (!_splitRange) {
	  const char* origAttrib = real.getStringAttribute("fitrange") ;	  
	  if (origAttrib) {
	    real.setStringAttribute("fitrange",Form("%s,fit_%s",origAttrib,GetName())) ;
	  } else {
	    real.setStringAttribute("fitrange",Form("fit_%s",GetName())) ;
	  }
	  RooRealVar* origObs = (RooRealVar*) origObsSet->find(arg->GetName()) ;
	  if (origObs) {
	    origObs->setRange(Form("fit_%s",GetName()),realObs->getMin(rangeName),realObs->getMax(rangeName)) ;
	  }
	}
	
      }
    }
  }
  delete origObsSet ;

  // If dataset is binned, activate caching of bins that are invalid because the're outside the
  // updated range definition (WVE need to add virtual interface here)
  RooDataHist* tmph = dynamic_cast<RooDataHist*>(_dataClone) ;
  if (tmph) {
    tmph->cacheValidEntries() ;
  }
        
  // Fix RooAddPdf coefficients to original normalization range
  if (rangeName && strlen(rangeName)) {
    
    // WVE Remove projected dependents from normalization
    _funcClone->fixAddCoefNormalization(*_dataClone->get(),kFALSE) ;
    
    if (addCoefRangeName && strlen(addCoefRangeName)) {
      cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName() 
		       << ") fixing interpretation of coefficients of any RooAddPdf component to range " << addCoefRangeName << endl ;
      _funcClone->fixAddCoefRange(addCoefRangeName,kFALSE) ;
    } else {
	cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName() 
			 << ") fixing interpretation of coefficients of any RooAddPdf to full domain of observables " << endl ;
	_funcClone->fixAddCoefRange(Form("NormalizationRangeFor%s",rangeName),kFALSE) ;
    }
  }


  // This is deferred from part 2 - but must happen after part 3 - otherwise invalid bins cannot be properly marked in cacheValidEntries
  _dataClone->attachBuffers(*_funcObsSet) ;
  setEventCount(_dataClone->numEntries()) ;




  // *********************************************************************
  // *** PART 4 *** Adjust normalization range for projected observables *
  // *********************************************************************

  // Remove projected dependents from normalization set
  if (projDeps.getSize()>0) {

    _projDeps = (RooArgSet*) projDeps.snapshot(kFALSE) ;
    
    //RooArgSet* tobedel = (RooArgSet*) _normSet->selectCommon(*_projDeps) ;
    _normSet->remove(*_projDeps,kTRUE,kTRUE) ;

//     // Delete owned projected dependent copy in _normSet
//     TIterator* ii = tobedel->createIterator() ;
//     RooAbsArg* aa ;
//     while((aa=(RooAbsArg*)ii->Next())) {
//       delete aa ;
//     }
//     delete ii ;
//     delete tobedel ;

    // Mark all projected dependents as such
    RooArgSet *projDataDeps = (RooArgSet*) _funcObsSet->selectCommon(*_projDeps) ;
    projDataDeps->setAttribAll("projectedDependent") ;
    delete projDataDeps ;    
  } 


  coutI(Optimization) << "RooAbsOptTestStatistic::ctor(" << GetName() << ") optimizing internal clone of p.d.f for likelihood evaluation." 
			<< "Lazy evaluation and associated change tracking will disabled for all nodes that depend on observables" << endl ;


  // *********************************************************************
  // *** PART 4 *** Finalization and activation of optimization          *
  // *********************************************************************

  //_origFunc = _func ;
  //_origData = _data ;

  // Redirect pointers of base class to clone 
  _func = _funcClone ;
  _data = _dataClone ;

  _funcClone->getVal(_normSet) ;

//   cout << "ROATS::ctor(" << GetName() << ") funcClone structure dump BEFORE opt" << endl ;
//   _funcClone->Print("t") ;

  optimizeCaching() ;


//   cout << "ROATS::ctor(" << GetName() << ") funcClone structure dump AFTER opt" << endl ;
//   _funcClone->Print("t") ;

}


//_____________________________________________________________________________
RooAbsOptTestStatistic::~RooAbsOptTestStatistic()
{
  // Destructor

  if (operMode()==Slave) {
    delete _funcClone ;
    delete _funcObsSet ;
    if (_projDeps) {
      delete _projDeps ;
    }
    if (_ownData) {
      delete _dataClone ;
    }
  } 
  delete _normSet ;
}



//_____________________________________________________________________________
Double_t RooAbsOptTestStatistic::combinedValue(RooAbsReal** array, Int_t n) const
{
  // Method to combined test statistic results calculated into partitions into
  // the global result. This default implementation adds the partition return
  // values
  
  // Default implementation returns sum of components
  Double_t sum(0), carry(0);
  for (Int_t i = 0; i < n; ++i) {
    Double_t y = array[i]->getValV();
    carry += reinterpret_cast<RooAbsOptTestStatistic*>(array[i])->getCarry();
    y -= carry;
    const Double_t t = sum + y;
    carry = (t - sum) - y;
    sum = t;
  }
  _evalCarry = carry;
  return sum ;
}



//_____________________________________________________________________________
Bool_t RooAbsOptTestStatistic::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) 
{
  // Catch server redirect calls and forward to internal clone of function

  RooAbsTestStatistic::redirectServersHook(newServerList,mustReplaceAll,nameChange,isRecursive) ;
  if (operMode()!=Slave) return kFALSE ;  
  Bool_t ret = _funcClone->recursiveRedirectServers(newServerList,kFALSE,nameChange) ;
  return ret ;
}



//_____________________________________________________________________________
void RooAbsOptTestStatistic::printCompactTreeHook(ostream& os, const char* indent) 
{
  // Catch print hook function and forward to function clone

  RooAbsTestStatistic::printCompactTreeHook(os,indent) ;
  if (operMode()!=Slave) return ;
  TString indent2(indent) ;
  indent2 += "opt >>" ;
  _funcClone->printCompactTree(os,indent2.Data()) ;
  os << indent2 << " dataset clone = " << _dataClone << " first obs = " << _dataClone->get()->first() << endl ;
}



//_____________________________________________________________________________
void RooAbsOptTestStatistic::constOptimizeTestStatistic(ConstOpCode opcode, Bool_t doAlsoTrackingOpt) 
{
  // Driver function to propagate constant term optimizations in test statistic.
  // If code Activate is sent, constant term optimization will be executed.
  // If code Deacivate is sent, any existing constant term optimizations will
  // be abanoned. If codes ConfigChange or ValueChange are sent, any existing
  // constant term optimizations will be redone.

  //   cout << "ROATS::constOpt(" << GetName() << ") funcClone structure dump BEFORE const-opt" << endl ;
  //   _funcClone->Print("t") ;

  RooAbsTestStatistic::constOptimizeTestStatistic(opcode,doAlsoTrackingOpt);
  if (operMode()!=Slave) return ;
  
  if (_dataClone->hasFilledCache() && _dataClone->store()->cacheOwner()!=this) {
    if (opcode==Activate) {
      cxcoutW(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() 
			    << ") dataset cache is owned by another object, no constant term optimization can be applied" << endl ;
    }
    return ;
  }

  if (!allowFunctionCache()) {
    if (opcode==Activate) {
      cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() 
			    << ") function caching prohibited by test statistic, no constant term optimization is applied" << endl ;
    }
    return ;
  }

  if (_dataClone->hasFilledCache() && opcode==Activate) {
    opcode=ValueChange ;
  }

  switch(opcode) {
  case Activate:     
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() 
			  << ") optimizing evaluation of test statistic by finding all nodes in p.d.f that depend exclusively"
			  << " on observables and constant parameters and precalculating their values" << endl ;
    optimizeConstantTerms(kTRUE,doAlsoTrackingOpt) ;
    break ;
    
  case DeActivate:  
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() 
			  << ") deactivating optimization of constant terms in test statistic" << endl ;
    optimizeConstantTerms(kFALSE) ;
    break ;

  case ConfigChange: 
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() 
			  << ") one ore more parameter were changed from constant to floating or vice versa, "
			  << "re-evaluating constant term optimization" << endl ;
    optimizeConstantTerms(kFALSE) ;
    optimizeConstantTerms(kTRUE,doAlsoTrackingOpt) ;
    break ;

  case ValueChange: 
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() 
			  << ") the value of one ore more constant parameter were changed re-evaluating constant term optimization" << endl ;
    // Request a forcible cache update of all cached nodes
    _dataClone->store()->forceCacheUpdate() ;

    break ;
  }

//   cout << "ROATS::constOpt(" << GetName() << ") funcClone structure dump AFTER const-opt" << endl ;
//   _funcClone->Print("t") ;
}



//_____________________________________________________________________________
void RooAbsOptTestStatistic::optimizeCaching() 
{
  // This method changes the value caching logic for all nodes that depends on any of the observables
  // as defined by the given dataset. When evaluating a test statistic constructed from the RooAbsReal
  // with a dataset the observables are guaranteed to change with every call, thus there is no point
  // in tracking these changes which result in a net overhead. Thus for observable-dependent nodes, 
  // the evaluation mechanism is changed from being dependent on a 'valueDirty' flag to guaranteed evaluation. 
  // On the dataset side, the observables objects are modified to no longer send valueDirty messages
  // to their client 

//   cout << "RooAbsOptTestStatistic::optimizeCaching(" << GetName() << "," << this << ")" << endl ;

  // Trigger create of all object caches now in nodes that have deferred object creation
  // so that cache contents can be processed immediately
  _funcClone->getVal(_normSet) ;

  // Set value caching mode for all nodes that depend on any of the observables to ADirty
  _funcClone->optimizeCacheMode(*_funcObsSet) ;

  // Disable propagation of dirty state flags for observables
  _dataClone->setDirtyProp(kFALSE) ;  

  // Disable reading of observables that are not used
  _dataClone->optimizeReadingWithCaching(*_funcClone, RooArgSet(),requiredExtraObservables()) ;
}



//_____________________________________________________________________________
void RooAbsOptTestStatistic::optimizeConstantTerms(Bool_t activate, Bool_t applyTrackingOpt)
{
  // Driver function to activate global constant term optimization.
  // If activated constant terms are found and cached with the dataset
  // The operation mode of cached nodes is set to AClean meaning that
  // their getVal() call will never result in an evaluate call.
  // Finally the branches in the dataset that correspond to observables
  // that are exclusively used in constant terms are disabled as
  // they serve no more purpose

  if(activate) {

    if (_optimized) {
      return ;
    }
    
    // Trigger create of all object caches now in nodes that have deferred object creation
    // so that cache contents can be processed immediately
    _funcClone->getVal(_normSet) ;

    // Apply tracking optimization here. Default strategy is to track components
    // of RooAddPdfs and RooRealSumPdfs. If these components are a RooProdPdf
    // or a RooProduct respectively, track the components of these products instead
    // of the product term
    RooArgSet trackNodes ;


    // Add safety check here - applyTrackingOpt will only be applied if present
    // dataset is constructed in terms of a RooVectorDataStore
    if (applyTrackingOpt) {
      if (!dynamic_cast<RooVectorDataStore*>(_dataClone->store())) {
	coutW(Optimization) << "RooAbsOptTestStatistic::optimizeConstantTerms(" << GetName() 
			    << ") WARNING Cache-and-track optimization (Optimize level 2) is only available for datasets"
			    << " implement in terms of RooVectorDataStore - ignoring this option for current dataset" << endl ;
	applyTrackingOpt = kFALSE ;
      }
    }

    if (applyTrackingOpt) {
      RooArgSet branches ;
      _funcClone->branchNodeServerList(&branches) ;
      RooFIter iter = branches.fwdIterator() ;
      RooAbsArg* arg ;
      while((arg=iter.next())) {
	arg->setCacheAndTrackHints(trackNodes) ;
      }
      // Do not set CacheAndTrack on constant expressions
      RooArgSet* constNodes = (RooArgSet*) trackNodes.selectByAttrib("Constant",kTRUE) ;
      trackNodes.remove(*constNodes) ;
      delete constNodes ;

      // Set CacheAndTrack flag on all remaining nodes
      trackNodes.setAttribAll("CacheAndTrack",kTRUE) ;
    }
    
    // Find all nodes that depend exclusively on constant parameters
    _cachedNodes.removeAll() ;

    _funcClone->findConstantNodes(*_dataClone->get(),_cachedNodes) ;

//     cout << "ROATS::oCT(" << GetName() << ") funcClone structure dump BEFORE cacheArgs" << endl ;
//     _funcClone->Print("t") ;


    // Cache constant nodes with dataset - also cache entries corresponding to zero-weights in data when using BinnedLikelihood
    _dataClone->cacheArgs(this,_cachedNodes,_normSet,!_funcClone->getAttribute("BinnedLikelihood")) ;  

//     cout << "ROATS::oCT(" << GetName() << ") funcClone structure dump AFTER cacheArgs" << endl ;
//     _funcClone->Print("t") ;

    
    // Put all cached nodes in AClean value caching mode so that their evaluate() is never called
    TIterator* cIter = _cachedNodes.createIterator() ;
    RooAbsArg *cacheArg ;
    while((cacheArg=(RooAbsArg*)cIter->Next())){
      cacheArg->setOperMode(RooAbsArg::AClean) ;
    }
    delete cIter ;  


//     cout << "_cachedNodes = " << endl ;
//     RooFIter i = _cachedNodes.fwdIterator() ;
//     RooAbsArg* aa ;
//     while ((aa=i.next())) {
//       cout << aa->IsA()->GetName() << "::" << aa->GetName() << (aa->getAttribute("ConstantExpressionCached")?" CEC":"   ") << (aa->getAttribute("CacheAndTrack")?" CAT":"   ") << endl ;
//     }

    RooArgSet* constNodes = (RooArgSet*) _cachedNodes.selectByAttrib("ConstantExpressionCached",kTRUE) ;
    RooArgSet actualTrackNodes(_cachedNodes) ;
    actualTrackNodes.remove(*constNodes) ;
    if (constNodes->getSize()>0) {
      if (constNodes->getSize()<20) {
	coutI(Minimization) << " The following expressions have been identified as constant and will be precalculated and cached: " << *constNodes << endl ;
      } else {
	coutI(Minimization) << " A total of " << constNodes->getSize() << " expressions have been identified as constant and will be precalculated and cached." << endl ;
      }
//       RooFIter i = constNodes->fwdIterator() ;
//       RooAbsArg* cnode ;
//       while((cnode=i.next())) {
// 	cout << cnode->IsA()->GetName() << "::" << cnode->GetName() << endl ;
//       }      
    }
    if (actualTrackNodes.getSize()>0) {
      if (actualTrackNodes.getSize()<20) {
	coutI(Minimization) << " The following expressions will be evaluated in cache-and-track mode: " << actualTrackNodes << endl ;
// 	RooFIter iter = actualTrackNodes.fwdIterator() ;
// 	RooAbsArg* atn ;
// 	while((atn = iter.next())) {
// 	  cout << atn->IsA()->GetName() << "::" << atn->GetName() << endl ;
// 	}
      } else {
	coutI(Minimization) << " A total of " << constNodes->getSize() << " expressions will be evaluated in cache-and-track-mode." << endl ;
      }
    }
    delete constNodes ;
    
    // Disable reading of observables that are no longer used
    _dataClone->optimizeReadingWithCaching(*_funcClone, _cachedNodes,requiredExtraObservables()) ;

    _optimized = kTRUE ;

  } else {
    
    // Delete the cache
    _dataClone->resetCache() ;
    
    // Reactivate all tree branches
    _dataClone->setArgStatus(*_dataClone->get(),kTRUE) ;
    
    // Reset all nodes to ADirty   
    optimizeCaching() ;

    // Disable propagation of dirty state flags for observables
    _dataClone->setDirtyProp(kFALSE) ;  

    _cachedNodes.removeAll() ;


    _optimized = kFALSE ;
  }
}



//_____________________________________________________________________________
Bool_t RooAbsOptTestStatistic::setDataSlave(RooAbsData& indata, Bool_t cloneData, Bool_t ownNewData) 
{ 
//   cout << "RAOTS::setDataSlave(" << this << ") START" << endl ;
  // Change dataset that is used to given one. If cloneData is kTRUE, a clone of
  // in the input dataset is made.  If the test statistic was constructed with
  // a range specification on the data, the cloneData argument is ignore and
  // the data is always cloned.


  if (operMode()==SimMaster) {
    //cout << "ROATS::setDataSlave() ERROR this is SimMaster _funcClone = " << _funcClone << endl ;    
    return kFALSE ;
  }
  
  //cout << "ROATS::setDataSlave() new dataset size = " << indata.numEntries() << endl ;
  //indata.Print("v") ;


  // Delete previous dataset now, if it was owned
  if (_ownData) {
    delete _dataClone ;
    _dataClone = 0 ;
  }

  if (!cloneData && _rangeName.size()>0) {
    coutW(InputArguments) << "RooAbsOptTestStatistic::setData(" << GetName() << ") WARNING: test statistic was constructed with range selection on data, "
			 << "ignoring request to _not_ clone the input dataset" << endl ; 
    cloneData = kTRUE ;
  }

  if (cloneData) {
    // Cloning input dataset
    if (_rangeName.size()==0) {
      _dataClone = (RooAbsData*) indata.reduce(*indata.get()) ;
    } else {
      _dataClone = ((RooAbsData&)indata).reduce(RooFit::SelectVars(*indata.get()),RooFit::CutRange(_rangeName.c_str())) ;  
    }
    _ownData = kTRUE ;

  } else {
 
    // Taking input dataset
    _dataClone = &indata ;
    _ownData = ownNewData ;
    
  }    
  
  // Attach function clone to dataset
  _dataClone->attachBuffers(*_funcObsSet) ;
  _dataClone->setDirtyProp(kFALSE) ;  
  _data = _dataClone ;

  // ReCache constant nodes with dataset 
  if (_cachedNodes.getSize()>0) {
    _dataClone->cacheArgs(this,_cachedNodes,_normSet) ;      
  }

  // Adjust internal event count
  setEventCount(indata.numEntries()) ;

  setValueDirty() ;

//   cout << "RAOTS::setDataSlave(" << this << ") END" << endl ;

  return kTRUE ;
}




//_____________________________________________________________________________
RooAbsData& RooAbsOptTestStatistic::data() 
{ 
  if (_sealed) {
    Bool_t notice = (sealNotice() && strlen(sealNotice())) ;
    coutW(ObjectHandling) << "RooAbsOptTestStatistic::data(" << GetName() 
			  << ") WARNING: object sealed by creator - access to data is not permitted: " 
			  << (notice?sealNotice():"<no user notice>") << endl ;
    static RooDataSet dummy ("dummy","dummy",RooArgSet()) ;
    return dummy ;
  }
  return *_dataClone ; 
}


//_____________________________________________________________________________
const RooAbsData& RooAbsOptTestStatistic::data() const 
{ 
  if (_sealed) {
    Bool_t notice = (sealNotice() && strlen(sealNotice())) ;
    coutW(ObjectHandling) << "RooAbsOptTestStatistic::data(" << GetName() 
			  << ") WARNING: object sealed by creator - access to data is not permitted: " 
			  << (notice?sealNotice():"<no user notice>") << endl ;
    static RooDataSet dummy ("dummy","dummy",RooArgSet()) ;
    return dummy ;
  }
  return *_dataClone ; 
}


