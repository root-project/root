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
}



//_____________________________________________________________________________
RooAbsOptTestStatistic::RooAbsOptTestStatistic(const char *name, const char *title, RooAbsReal& real, RooAbsData& indata,
					       const RooArgSet& projDeps, const char* rangeName, const char* addCoefRangeName,
					       Int_t nCPU, Bool_t interleave, Bool_t verbose, Bool_t splitCutRange, Bool_t cloneInputData) : 
  RooAbsTestStatistic(name,title,real,indata,projDeps,rangeName, addCoefRangeName, nCPU, interleave, verbose, splitCutRange),
  _projDeps(0),
  _sealed(kFALSE)
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

  cloneInputData=1;
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
  RooAbsTestStatistic(other,name), _sealed(other._sealed), _sealNotice(other._sealNotice)
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

  // Reattach FUNC to original parameters  
  RooArgSet* origParams = (RooArgSet*) real.getParameters(indata) ;
  _funcClone->recursiveRedirectServers(*origParams) ;

  // Add parameters as servers
  _paramSet.add(*origParams) ;
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
 
  _dataClone->attachBuffers(*_funcObsSet) ;
  setEventCount(_dataClone->numEntries()) ;



  // ******************************************************************
  // *** PART 3 *** Make adjustments for fit ranges, if specified     *
  // ****************************************************************** 

  RooArgSet* origObsSet = real.getObservables(indata) ;
  RooArgSet* dataObsSet = (RooArgSet*) _dataClone->get() ;
  if (rangeName && strlen(rangeName)) {    
    cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName() << ") constructing test statistic for sub-range named " << rangeName << endl ;    
//     cout << "now adjusting observable ranges to requested fit range" << endl ;

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
// 	cout << "RAOTS::ctor() setting normalization range on observable " 
// 	     << realObs->GetName() << " to [" << realObs->getMin() << "," << realObs->getMax() << "]" << endl ;

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



  // *********************************************************************
  // *** PART 4 *** Adjust normalization range for projected observables *
  // *********************************************************************

  // Remove projected dependents from normalization set
  if (projDeps.getSize()>0) {

    _projDeps = (RooArgSet*) projDeps.snapshot(kFALSE) ;
    
    RooArgSet* tobedel = (RooArgSet*) _normSet->selectCommon(*_projDeps) ;
    _normSet->remove(*_projDeps,kTRUE,kTRUE) ;

    // Delete owned projected dependent copy in _normSet
    TIterator* ii = tobedel->createIterator() ;
    RooAbsArg* aa ;
    while((aa=(RooAbsArg*)ii->Next())) {
      delete aa ;
    }
    delete ii ;
    delete tobedel ;

    // Mark all projected dependents as such
    RooArgSet *projDataDeps = (RooArgSet*) _dataClone->get()->selectCommon(*_projDeps) ;
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
  Double_t sum(0) ;
  Int_t i ;
  for (i=0 ; i<n ; i++) {
    Double_t tmp = array[i]->getVal() ;
    // if (tmp==0) return 0 ; WVE no longer needed
    sum += tmp ;
  }
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
void RooAbsOptTestStatistic::constOptimizeTestStatistic(ConstOpCode opcode) 
{
  // Driver function to propagate constant term optimizations in test statistic.
  // If code Activate is sent, constant term optimization will be executed.
  // If code Deacivate is sent, any existing constant term optimizations will
  // be abanoned. If codes ConfigChange or ValueChange are sent, any existing
  // constant term optimizations will be redone.

//   cout << "ROATS::constOpt(" << GetName() << ") funcClone structure dump BEFORE const-opt" << endl ;
//   _funcClone->Print("t") ;

  RooAbsTestStatistic::constOptimizeTestStatistic(opcode);
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

  switch(opcode) {
  case Activate:     
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() 
			  << ") optimizing evaluation of test statistic by finding all nodes in p.d.f that depend exclusively"
			  << " on observables and constant parameters and precalculating their values" << endl ;
    optimizeConstantTerms(kTRUE) ;
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
    optimizeConstantTerms(kTRUE) ;
    break ;

  case ValueChange: 
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() 
			  << ") the value of one ore more constant parameter were changed re-evaluating constant term optimization" << endl ;
    optimizeConstantTerms(kFALSE) ;
    optimizeConstantTerms(kTRUE) ;
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
  _funcClone->optimizeCacheMode(*_dataClone->get()) ;

  // Disable propagation of dirty state flags for observables
  _dataClone->setDirtyProp(kFALSE) ;  

  // Disable reading of observables that are not used
  _dataClone->optimizeReadingWithCaching(*_funcClone, RooArgSet(),requiredExtraObservables()) ;
}



//_____________________________________________________________________________
void RooAbsOptTestStatistic::optimizeConstantTerms(Bool_t activate)
{
  // Driver function to activate global constant term optimization.
  // If activated constant terms are found and cached with the dataset
  // The operation mode of cached nodes is set to AClean meaning that
  // their getVal() call will never result in an evaluate call.
  // Finally the branches in the dataset that correspond to observables
  // that are exclusively used in constant terms are disabled as
  // they serve no more purpose

  if(activate) {
    // Trigger create of all object caches now in nodes that have deferred object creation
    // so that cache contents can be processed immediately
    _funcClone->getVal(_normSet) ;
    
    // Find all nodes that depend exclusively on constant parameters
    _cachedNodes.removeAll() ;

    _funcClone->findConstantNodes(*_dataClone->get(),_cachedNodes) ;

//     cout << "ROATS::oCT(" << GetName() << ") funcClone structure dump BEFORE cacheArgs" << endl ;
//     _funcClone->Print("t") ;


    // Cache constant nodes with dataset 
    _dataClone->cacheArgs(this,_cachedNodes,_normSet) ;  

//     cout << "ROATS::oCT(" << GetName() << ") funcClone structure dump AFTER cacheArgs" << endl ;
//     _funcClone->Print("t") ;

    
    // Put all cached nodes in AClean value caching mode so that their evaluate() is never called
    TIterator* cIter = _cachedNodes.createIterator() ;
    RooAbsArg *cacheArg ;
    while((cacheArg=(RooAbsArg*)cIter->Next())){
      cacheArg->setOperMode(RooAbsArg::AClean) ;
    }
    delete cIter ;  

    coutI(Minimization) << " The following expressions have been identified as constant and will be precalculated and cached: " << _cachedNodes << endl ;
    
    // Disable reading of observables that are no longer used
    _dataClone->optimizeReadingWithCaching(*_funcClone, _cachedNodes,requiredExtraObservables()) ;

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
    
  }
}



//_____________________________________________________________________________
Bool_t RooAbsOptTestStatistic::setDataSlave(RooAbsData& indata, Bool_t cloneData) 
{ 
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
    _ownData = kFALSE ;
    
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


