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
  _normSet = 0 ;
  _funcCloneSet = 0 ;
  _dataClone = 0 ;
  _funcClone = 0 ;
  _projDeps = 0 ;
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

//   cout << "RooAbsOptTestStatistic::ctor(" << GetName() << "," << this << endl ;
  //FK: Desperate times, desperate measures. How can RooNLLVar call this ctor with dataClone=kFALSE?
  //   cout<<"Setting clonedata to 1"<<endl;
  cloneInputData=1;
  // Don't do a thing in master mode
  if (operMode()!=Slave) {
    //cout << "RooAbsOptTestStatistic::ctor not slave mode, do nothing" << endl ;
    _normSet = 0 ;
    _funcCloneSet = 0 ;
    _dataClone = 0 ;
    _funcClone = 0 ;
    _projDeps = 0 ;
    _ownData = kFALSE ;
    _sealed = kFALSE ;
    return ;
  }

  RooArgSet obs(*indata.get()) ;
  obs.remove(projDeps,kTRUE,kTRUE) ;

  // + ALEX
  //   // Check that the FUNC is valid for use with this dataset
  //   // Check if there are any unprotected multiple occurrences of dependents
  //   if (real.recursiveCheckObservables(&obs)) {
  //     coutE(InputArguments) << "RooAbsOptTestStatistic: ERROR in FUNC dependents, abort" << endl ;
  //     RooErrorHandler::softAbort() ;
  //     return ;
  //   }
  // - ALEX  


  // Get list of actual observables of test statistic function
  RooArgSet* realDepSet = real.getObservables(&indata) ;

  // Expand list of observables with any observables used in parameterized ranges
  TIterator* iter9 = realDepSet->createIterator() ;
  RooAbsArg* realDep ;
  while((realDep=(RooAbsArg*)iter9->Next())) {
    RooAbsRealLValue* realDepRLV = dynamic_cast<RooAbsRealLValue*>(realDep) ;
    if (realDepRLV && realDepRLV->isDerived()) {

      RooArgSet tmp ;
      realDepRLV->leafNodeServerList(&tmp, 0, kTRUE) ;
      realDepSet->add(tmp,kTRUE) ;
    }
  }
  delete iter9 ;


  // Check if the fit ranges of the dependents in the data and in the FUNC are consistent
  const RooArgSet* dataDepSet = indata.get() ;
  TIterator* iter = realDepSet->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooRealVar* realReal = dynamic_cast<RooRealVar*>(arg) ;
    if (!realReal) continue ;

    
    RooRealVar* datReal = dynamic_cast<RooRealVar*>(dataDepSet->find(realReal->GetName())) ;
    if (!datReal) continue ;
    
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
    if (!cloneInputData) {
      coutW(InputArguments) << "RooAbsOptTestStatistic::ctor(" << GetName() 
			    << ") WARNING: Must clone input data when a range specification is given, ignoring request to use original input dataset" << endl ; 
    }    
    _dataClone = ((RooAbsData&)indata).reduce(RooFit::SelectVars(*realDepSet),RooFit::CutRange(rangeName)) ;  
//     cout << "RooAbsOptTestStatistic: reducing dataset to fit in range named " << rangeName << " resulting dataset has " << _dataClone->sumEntries() << " events" << endl ;
    _ownData = kTRUE ;
  } else {
    if (cloneInputData) {
      _dataClone = (RooAbsData*) indata.Clone() ;
      //reduce(RooFit::SelectVars(*indata.get())) ; //  ((RooAbsData&)data).reduce(RooFit::SelectVars(*realDepSet)) ;  

      _ownData = kTRUE ;
    } else {
      // coverity[DEADCODE]
      _dataClone = &indata ;
      _ownData = kFALSE ;
    }
  }



  // Copy any non-shared parameterized range definitions from pdf observables to dataset observables
  iter9 = realDepSet->createIterator() ;
  while((realDep=(RooAbsRealLValue*)iter9->Next())) {
    RooAbsRealLValue* realDepRLV = dynamic_cast<RooAbsRealLValue*>(realDep) ;
    if (realDepRLV && !realDepRLV->getBinning().isShareable()) {

      RooRealVar* datReal = dynamic_cast<RooRealVar*>(_dataClone->get()->find(realDepRLV->GetName())) ;
      if (datReal) {
	datReal->setBinning(realDepRLV->getBinning()) ;
	datReal->attachDataSet(*_dataClone) ;
      }
    }
  }
  delete iter9 ;

  if (rangeName && strlen(rangeName)) {
    
    cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName() << ") constructing test statistic for sub-range named " << rangeName << endl ;

    // Adjust FUNC normalization ranges to requested fitRange, store original ranges for RooAddPdf coefficient interpretation
    TIterator* iter2 = _dataClone->get()->createIterator() ;
    while((arg=(RooAbsArg*)iter2->Next())) {
      RooRealVar* realReal = dynamic_cast<RooRealVar*>(arg) ;
      if (realReal) {
        if (!(addCoefRangeName && strlen(addCoefRangeName))) {
	  realReal->setRange(Form("NormalizationRangeFor%s",rangeName),realReal->getMin(),realReal->getMax()) ;
	}

	realReal->setRange(realReal->getMin(rangeName),realReal->getMax(rangeName)) ;	
      }
    }

    // If dataset is binned, activate caching of bins that are invalid because the're outside the
    // updated range definition (WVE need to add virtual interface here)
    RooDataHist* tmp = dynamic_cast<RooDataHist*>(_dataClone) ;
    if (tmp) {
      tmp->cacheValidEntries() ;
    }


    // Mark fitted range in original FUNC dependents for future use
    if (!_splitRange) {
      iter->Reset() ;
      while((arg=(RooAbsArg*)iter->Next())) {      
	RooRealVar* realReal = dynamic_cast<RooRealVar*>(arg) ;
	if (realReal) {
	  realReal->setStringAttribute("fitrange",Form("fit_%s",GetName())) ;
	  realReal->setRange(Form("fit_%s",GetName()),realReal->getMin(rangeName),realReal->getMax(rangeName)) ;

	  // Keep track of list of fit ranges in string attribute fit range of original p.d.f.
	  const char* origAttrib = real.getStringAttribute("fitrange") ;	  
	  if (origAttrib) {
	    real.setStringAttribute("fitrange",Form("%s,fit_%s",origAttrib,GetName())) ;
	  } else {
	    real.setStringAttribute("fitrange",Form("fit_%s",GetName())) ;
	  }
	}
      }
    }
    delete iter2 ;
  }
  delete iter ;

  setEventCount(_dataClone->numEntries()) ;
 
  // Clone all FUNC compents by copying all branch nodes
  RooArgSet tmp("RealBranchNodeList") ;
  real.branchNodeServerList(&tmp) ;
  _funcCloneSet = (RooArgSet*) tmp.snapshot(kFALSE) ;
  
  // Find the top level FUNC in the snapshot list
  _funcClone = (RooAbsReal*) _funcCloneSet->find(real.GetName()) ;


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


  // Attach FUNC to data set
  _funcClone->attachDataSet(*_dataClone) ;


  // Store normalization set  
  _normSet = (RooArgSet*) indata.get()->snapshot(kFALSE) ;

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

//   cout << "RAOTS: Now evaluating funcClone with _normSet = " << _normSet << " = " << *_normSet << endl ;
  _funcClone->getVal(_normSet) ;

  // Add parameters as servers
  RooArgSet* params = _funcClone->getParameters(_dataClone) ;
  _paramSet.add(*params) ;
  delete params ;

  // Mark all projected dependents as such
  if (_projDeps) {
    RooArgSet *projDataDeps = (RooArgSet*) _dataClone->get()->selectCommon(*_projDeps) ;
    projDataDeps->setAttribAll("projectedDependent") ;
    delete projDataDeps ;
  }

  coutI(Optimization) << "RooAbsOptTestStatistic::ctor(" << GetName() << ") optimizing internal clone of p.d.f for likelihood evaluation." 
			<< "Lazy evaluation and associated change tracking will disabled for all nodes that depend on observables" << endl ;



  delete realDepSet ;  

  // Redirect pointers of base class to clone 
  _func = _funcClone ;
  _data = _dataClone ;

  _funcClone->getVal(_normSet) ;

  optimizeCaching() ;
}


//_____________________________________________________________________________
RooAbsOptTestStatistic::RooAbsOptTestStatistic(const RooAbsOptTestStatistic& other, const char* name) : 
  RooAbsTestStatistic(other,name), _sealed(other._sealed), _sealNotice(other._sealNotice)
{
  // Copy constructor
//   cout << "RooAbsOptTestStatistic::cctor(" << GetName() << "," << this << endl ;

  // Don't do a thing in master mode
  if (operMode()!=Slave) {
//     cout << "RooAbsOptTestStatistic::cctor not slave mode, do nothing" << endl ;
    _projDeps = 0 ;
    _normSet = other._normSet ? ((RooArgSet*) other._normSet->snapshot()) : 0 ;   
    _ownData = kFALSE ;
    return ;
  }

  _funcCloneSet = (RooArgSet*) other._funcCloneSet->snapshot(kFALSE) ;
  _funcClone = (RooAbsReal*) _funcCloneSet->find(other._funcClone->GetName()) ;

  // Copy the operMode attribute of all branch nodes
  TIterator* iter = _funcCloneSet->createIterator() ;
  RooAbsArg* branch ;
  while((branch=(RooAbsArg*)iter->Next())) {
    branch->setOperMode(other._funcCloneSet->find(branch->GetName())->operMode()) ;
  }
  delete iter ;

  // WVE Must use clone with cache redirection here
  if (other._ownData || other._dataClone->hasFilledCache()) {    
    _dataClone = (RooAbsData*) other._dataClone->cacheClone(this,_funcCloneSet) ;
    _ownData = kTRUE ;
  } else {
    _dataClone = other._dataClone ;
    _ownData = kFALSE ;
    
    // Revert any AClean nodes imported from original to ADirty as not optimization is applicable to test statistics with borrowed data
    Bool_t wasOpt(kFALSE) ;
    TIterator* biter = _funcCloneSet->createIterator() ;
    RooAbsArg *branch2 ;
    while((branch2=(RooAbsArg*)biter->Next())){
      if (branch2->operMode()==RooAbsArg::AClean) {
// 	cout << "RooAbsOptTestStatistic::cctor(" << GetName() << " setting branch " << branch2->GetName() << " to ADirty" << endl ;
	branch2->setOperMode(RooAbsArg::ADirty) ;
	wasOpt=kTRUE ;
      }
    }
    delete biter ;  

    if (wasOpt) {
      coutW(Optimization) << "RooAbsOptTestStatistic::cctor(" << GetName() << ") WARNING clone of optimized test statistic with unowned data will not be optimized, "
			  << "to retain optimization behavior in cloning, construct test statistic with CloneData(kTRUE)" << endl ;
    }
  }

  // Attach function clone to dataset
  _funcClone->attachDataSet(*_dataClone) ;

  // Explicit attach function clone to current parameter instances
  _funcClone->recursiveRedirectServers(_paramSet) ;

  _normSet = (RooArgSet*) other._normSet->snapshot() ;
  
  if (other._projDeps) {
    _projDeps = (RooArgSet*) other._projDeps->snapshot() ;
  } else {
    _projDeps = 0 ;
  }

  _func = _funcClone ;
  _data = _dataClone ;

//   cout << "RooAbsOptTestStatistic::cctor call getVal()" << endl ;
  _funcClone->getVal(_normSet) ;
//   cout << "RooAbsOptTestStatistic::cctor start caching" << endl ;
  optimizeCaching() ;
}



//_____________________________________________________________________________
RooAbsOptTestStatistic::~RooAbsOptTestStatistic()
{
  // Destructor

  if (operMode()==Slave) {
    delete _funcCloneSet ;
    if (_ownData) {
      delete _dataClone ;
    } else {
      // If dataclone survives the test statistic, clean its cache transer
      // ownership of observables back to dataset
      if (RooAbsData::releaseVars(_dataClone)==kFALSE) {
	_ownedDataObs.releaseOwnership() ;
      } 
//       if (RooAbsData::isAlive(_dataClone)) {
// 	_dataClone->resetCache() ;
//       }      
    }
    delete _projDeps ;
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
    RooArgSet cacheableNodes ;

    _funcClone->findConstantNodes(*_dataClone->get(),cacheableNodes) ;

    // Cache constant nodes with dataset 
    _dataClone->cacheArgs(this,cacheableNodes,_normSet) ;  
    
    // Put all cached nodes in AClean value caching mode so that their evaluate() is never called
    TIterator* cIter = cacheableNodes.createIterator() ;
    RooAbsArg *cacheArg ;
    while((cacheArg=(RooAbsArg*)cIter->Next())){
      cacheArg->setOperMode(RooAbsArg::AClean) ;
    }
    delete cIter ;  
    
    // Disable reading of observables that are no longer used
    _dataClone->optimizeReadingWithCaching(*_funcClone, cacheableNodes,requiredExtraObservables()) ;

  } else {
    
    // Delete the cache
    _dataClone->resetCache() ;
    
    // Reactivate all tree branches
    _dataClone->setArgStatus(*_dataClone->get(),kTRUE) ;
    
    // Reset all nodes to ADirty   
    optimizeCaching() ;

    // Disable propagation of dirty state flags for observables
    _dataClone->setDirtyProp(kFALSE) ;  
    
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

  RooAbsData* origData = _dataClone ;
  Bool_t deleteOrigData = _ownData ;

  if (!cloneData && _rangeName.size()>0) {
    coutW(InputArguments) << "RooAbsOptTestStatistic::setData(" << GetName() << ") WARNING: test statistic was constructed with range selection on data, "
			 << "ignoring request to _not_ clone the input dataset" << endl ; 
    cloneData = kTRUE ;
  }

  RooArgSet obsToOwn ;

  if (cloneData) {
    if (_rangeName.size()==0) {
      _dataClone = (RooAbsData*) indata.reduce(*indata.get()) ;
    } else {
      _dataClone = ((RooAbsData&)indata).reduce(RooFit::SelectVars(*indata.get()),RooFit::CutRange(_rangeName.c_str())) ;  
    }
    _ownData = kTRUE ;
  } else {
    
    _dataClone = &indata ;
    _ownData = kFALSE ;
    
    // Add claim on observables to prevent these from being deleted when _dataClone is deleted
    RooAbsData::claimVars(_dataClone) ;
    
    // Prepare totake ownership of data observables
    obsToOwn.add(_dataClone->_vars) ;
  }    
  
  // Attach function clone to dataset
  Bool_t save = RooObjCacheManager::clearObsList() ;
  RooObjCacheManager::doClearObsList(kTRUE) ;

  _funcClone->attachDataSet(*_dataClone) ;

  RooObjCacheManager::doClearObsList(save) ;

  _data = _dataClone ;

  if (deleteOrigData) {
    delete origData ;
  } else {
    if (RooAbsData::releaseVars(origData)==kFALSE) {
      _ownedDataObs.releaseOwnership() ;
    } else {
    }
    _ownedDataObs.removeAll() ;
  }

  // Take ownership of observables of dataset 
  if (obsToOwn.getSize()>0) {
    _ownedDataObs.addOwned(obsToOwn) ;
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
