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

// -- CLASS DESCRIPTION [PDF] --
// RooAbsOptTestStatistic is the abstract base class for goodness-of-fit
// variables that evaluate the PDF at each point of a given dataset.
// This class provides generic optimizations, such as caching and precalculation
// of constant terms that can be made for all such quantities
//
// Implementations should define evaluatePartition(), which calculates the
// value of a (sub)range of the dataset and optionally combinedValue(),
// which combines the values calculated for each partition. If combinedValue()
// is overloaded, the default implementation will add the partition results
// to obtain the combined result
//
// Support for calculation in partitions is needed to allow parallel calculation
// of goodness-of-fit values.

#include "RooFit.h"

#include "Riostream.h"
#include <string.h>


#include "RooAbsOptTestStatistic.h"
#include "RooMsgService.h"
#include "RooAbsPdf.h"
#include "RooAbsData.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooErrorHandler.h"
#include "RooGlobalFunc.h"

ClassImp(RooAbsOptTestStatistic)
;

RooAbsOptTestStatistic:: RooAbsOptTestStatistic()
{
  _normSet = 0 ;
  _funcCloneSet = 0 ;
  _dataClone = 0 ;
  _funcClone = 0 ;
  _projDeps = 0 ;
}

RooAbsOptTestStatistic::RooAbsOptTestStatistic(const char *name, const char *title, RooAbsReal& real, RooAbsData& data,
					       const RooArgSet& projDeps, const char* rangeName, const char* addCoefRangeName,
					       Int_t nCPU, Bool_t interleave, Bool_t verbose, Bool_t splitCutRange) : 
  RooAbsTestStatistic(name,title,real,data,projDeps,rangeName, addCoefRangeName, nCPU, interleave, verbose, splitCutRange),
  _projDeps(0)
{

  // Don't do a thing in master mode
  if (operMode()!=Slave) {
    _normSet = 0 ;
    return ;
  }

  RooArgSet obs(*data.get()) ;
  obs.remove(projDeps,kTRUE,kTRUE) ;

  // Check that the FUNC is valid for use with this dataset
  // Check if there are any unprotected multiple occurrences of dependents
  if (real.recursiveCheckObservables(&obs)) {
    coutE(InputArguments) << "RooAbsOptTestStatistic: ERROR in FUNC dependents, abort" << endl ;
    RooErrorHandler::softAbort() ;
    return ;
  }
  
  
  // Check if the fit ranges of the dependents in the data and in the FUNC are consistent
  RooArgSet* realDepSet = real.getObservables(&data) ;
  const RooArgSet* dataDepSet = data.get() ;
  TIterator* iter = realDepSet->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooRealVar* realReal = dynamic_cast<RooRealVar*>(arg) ;
    if (!realReal) continue ;
    
    RooRealVar* datReal = dynamic_cast<RooRealVar*>(dataDepSet->find(realReal->GetName())) ;
    if (!datReal) continue ;
    
    if (realReal->getMin()<(datReal->getMin()-1e-6)) {
      coutE(InputArguments) << "RooAbsOptTestStatistic: ERROR minimum of FUNC variable " << arg->GetName() 
			    << "(" << realReal->getMin() << ") is smaller than that of " 
			    << arg->GetName() << " in the dataset (" << datReal->getMin() << ")" << endl ;
      RooErrorHandler::softAbort() ;
      return ;
    }
    
    if (realReal->getMax()>(datReal->getMax()+1e-6)) {
      coutE(InputArguments) << "RooAbsOptTestStatistic: ERROR maximum of FUNC variable " << arg->GetName() 
			    << " is smaller than that of " << arg->GetName() << " in the dataset" << endl ;
      RooErrorHandler::softAbort() ;
      return ;
    }
    
  }

  
  // Copy data and strip entries lost by adjusted fit range, _dataClone ranges will be copied from realDepSet ranges
  if (rangeName && strlen(rangeName)) {
    _dataClone = ((RooAbsData&)data).reduce(RooFit::SelectVars(*realDepSet),RooFit::CutRange(rangeName)) ;  
  } else {
    _dataClone = ((RooAbsData&)data).reduce(RooFit::SelectVars(*realDepSet)) ;  
  }
  
  if (rangeName && strlen(rangeName)) {
    
    cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName() << ") constructing likelihood for sub-range named " << rangeName << endl ;

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

    // Mark fitted range in original FUNC dependents for future use
    if (!_splitRange) {
      iter->Reset() ;
      while((arg=(RooAbsArg*)iter->Next())) {      
	RooRealVar* realReal = dynamic_cast<RooRealVar*>(arg) ;
	if (realReal) {
	  realReal->setRange("fit",realReal->getMin(rangeName),realReal->getMax(rangeName)) ;
	}
      }
    }
    delete iter2 ;
  }
  delete iter ;

  delete realDepSet ;  

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
    _funcClone->fixAddCoefNormalization(*_dataClone->get()) ;
    
    if (addCoefRangeName) {
      cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName() << ") fixing interpretation of coefficients of any RooAddPdf component to range " << addCoefRangeName << endl ;
      _funcClone->fixAddCoefRange(addCoefRangeName,kFALSE) ;
    } else {
	cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName() << ") fixing interpretation of coefficients of any RooAddPdf to full domain of observables " << endl ;
	_funcClone->fixAddCoefRange(Form("NormalizationRangeFor%s",rangeName),kFALSE) ;
    }
  }

  // Attach FUNC to data set
  _funcClone->attachDataSet(*_dataClone) ;

  // Store normalization set  
  _normSet = (RooArgSet*) data.get()->snapshot(kFALSE) ;

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

  optimizeCaching() ;
  
}


RooAbsOptTestStatistic::RooAbsOptTestStatistic(const RooAbsOptTestStatistic& other, const char* name) : 
  RooAbsTestStatistic(other,name)
{
  // Don't do a thing in master mode
  if (operMode()!=Slave) {
    _projDeps = 0 ;
    _normSet = other._normSet ? ((RooArgSet*) other._normSet->snapshot()) : 0 ;   
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
  _dataClone = (RooAbsData*) other._dataClone->cacheClone(_funcCloneSet) ;

  _funcClone->attachDataSet(*_dataClone) ;
  _normSet = (RooArgSet*) other._normSet->snapshot() ;
  
  if (other._projDeps) {
    _projDeps = (RooArgSet*) other._projDeps->snapshot() ;
  } else {
    _projDeps = 0 ;
  }
}



RooAbsOptTestStatistic::~RooAbsOptTestStatistic()
{
  if (operMode()==Slave) {
    delete _funcCloneSet ;
    delete _dataClone ;
    delete _projDeps ;
  } 
  delete _normSet ;
}



Double_t RooAbsOptTestStatistic::combinedValue(RooAbsReal** array, Int_t n) const
{
  // Default implementation returns sum of components
  Double_t sum(0) ;
  Int_t i ;
  for (i=0 ; i<n ; i++) {
    Double_t tmp = array[i]->getVal() ;
    if (tmp==0) return 0 ;
    sum += tmp ;
  }
  return sum ;
}


Bool_t RooAbsOptTestStatistic::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) 
{
  RooAbsTestStatistic::redirectServersHook(newServerList,mustReplaceAll,nameChange,isRecursive) ;
  if (operMode()!=Slave) return kFALSE ;  
  Bool_t ret = _funcClone->recursiveRedirectServers(newServerList,kFALSE,nameChange) ;
  return ret ;
}


void RooAbsOptTestStatistic::printCompactTreeHook(ostream& os, const char* indent) 
{
  RooAbsTestStatistic::printCompactTreeHook(os,indent) ;
  if (operMode()!=Slave) return ;
  TString indent2(indent) ;
  indent2 += ">>" ;
  _funcClone->printCompactTree(os,indent2) ;
}



void RooAbsOptTestStatistic::constOptimizeTestStatistic(ConstOpCode opcode) 
{
  // Driver function to propagate const optimization
  RooAbsTestStatistic::constOptimizeTestStatistic(opcode);
  if (operMode()!=Slave) return ;

  switch(opcode) {
  case Activate:     
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() << ") optimizing evaluation of test statistic by finding all nodes in p.d.f that depend exclusively"
			    << " on observables and constant parameters and precalculating their values" << endl ;
    optimizeConstantTerms(kTRUE) ;
    break ;
  case DeActivate:  
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() << ") deactivating optimization of constant terms in test statistic" << endl ;
    optimizeConstantTerms(kFALSE) ;
    break ;
  case ConfigChange: 
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() << ") one ore more parameter were changed from constant to floating or vice versa, "
			    << "re-evaluating constant term optimization" << endl ;
    optimizeConstantTerms(kFALSE) ;
    optimizeConstantTerms(kTRUE) ;
    break ;
  case ValueChange: 
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName() << ") the value of one ore more constant parameter were changed re-evaluating constant term optimization" << endl ;
    optimizeConstantTerms(kFALSE) ;
    optimizeConstantTerms(kTRUE) ;
    break ;
  }
}


void RooAbsOptTestStatistic::optimizeCaching() 
{
  // This method changes the value caching logic for all nodes that depends on any of the observables
  // as defined by the given dataset. When evaluating a test statistic constructed from the RooAbsReal
  // with a dataset the observables are guaranteed to change with every call, thus there is no point
  // in tracking these changes which result in a net overhead. Thus for observable-dependent nodes, 
  // the evaluation mechanism is changed from being dependent on a 'valueDirty' flag to guaranteed evaluation. 
  // On the dataset side, the observables objects are modified to no longer send valueDirty messages
  // to their client 

  // Trigger create of all object caches now in nodes that have deferred object creation
  // so that cache contents can be processed immediately
  _funcClone->getVal(_normSet) ;

  // Set value caching mode for all nodes that depend on any of the observables to ADirty
  _funcClone->optimizeCacheMode(*_dataClone->get()) ;

  // Disable propagation of dirty state flags for observables
  _dataClone->setDirtyProp(kFALSE) ;  

  // Disable reading of observables that are not used
  _dataClone->optimizeReadingWithCaching(*_funcClone, RooArgSet()) ;
}


void RooAbsOptTestStatistic::optimizeConstantTerms(Bool_t activate)
{
  if(activate) {
    // Trigger create of all object caches now in nodes that have deferred object creation
    // so that cache contents can be processed immediately
    _funcClone->getVal(_normSet) ;
    
    // Find all nodes that depend exclusively on constant parameters
    RooArgSet cacheableNodes ;
    _funcClone->findConstantNodes(*_dataClone->get(),cacheableNodes) ;

    // Cache constant nodes with dataset 
    _dataClone->cacheArgs(cacheableNodes,_normSet) ;  
    
    // Put all cached nodes in AClean value caching mode so that their evaluate() is never called
    TIterator* cIter = cacheableNodes.createIterator() ;
    RooAbsArg *cacheArg ;
    while((cacheArg=(RooAbsArg*)cIter->Next())){
      cacheArg->setOperMode(RooAbsArg::AClean) ;
    }
    delete cIter ;  
    
    // Disable reading of observables that are no longer used
    _dataClone->optimizeReadingWithCaching(*_funcClone, cacheableNodes) ;

  } else {
    
    // Delete the cache
    _dataClone->resetCache() ;
    
    // Reactivate all tree branches
    _dataClone->setArgStatus(*_dataClone->get(),kTRUE) ;
    
    // Reset all nodes to ADirty   
    optimizeCaching() ;
    
  }
}



