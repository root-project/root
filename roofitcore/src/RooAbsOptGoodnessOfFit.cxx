/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsOptGoodnessOfFit.cc,v 1.11 2003/04/01 22:34:42 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// RooAbsOptGoodnessOfFit is the abstract base class for goodness-of-fit
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

#include "RooFitCore/RooAbsOptGoodnessOfFit.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooAbsData.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooErrorHandler.hh"

ClassImp(RooAbsOptGoodnessOfFit)
;

RooAbsOptGoodnessOfFit::RooAbsOptGoodnessOfFit(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
					 const RooArgSet& projDeps, Int_t nCPU) : 
  RooAbsGoodnessOfFit(name,title,pdf,data,projDeps,nCPU),
  _projDeps(0)
{
  // Don't do a thing in master mode
  if (operMode()!=Slave) {
    _normSet = 0 ;
    return ;
  }

  RooArgSet obs(*data.get()) ;
  obs.remove(projDeps,kTRUE,kTRUE) ;

  // Check that the PDF is valid for use with this dataset
  // Check if there are any unprotected multiple occurrences of dependents
  if (pdf.recursiveCheckDependents(&obs)) {
    cout << "RooAbsOptGoodnessOfFit: ERROR in PDF dependents, abort" << endl ;
    RooErrorHandler::softAbort() ;
    return ;
  }
  
  
  // Check if the fit ranges of the dependents in the data and in the PDF are consistent
  RooArgSet* pdfDepSet = pdf.getDependents(&data) ;
  const RooArgSet* dataDepSet = data.get() ;
  TIterator* iter = pdfDepSet->createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
    RooRealVar* pdfReal = dynamic_cast<RooRealVar*>(arg) ;
    if (!pdfReal) continue ;
    
    RooRealVar* datReal = dynamic_cast<RooRealVar*>(dataDepSet->find(pdfReal->GetName())) ;
    if (!datReal) continue ;
    
    if (pdfReal->getFitMin()<(datReal->getFitMin()-1e-6)) {
      cout << "RooAbsOptGoodnessOfFit: ERROR minimum of PDF variable " << arg->GetName() 
	   << "(" << pdfReal->getFitMin() << ") is smaller than that of " 
	   << arg->GetName() << " in the dataset (" << datReal->getFitMin() << ")" << endl ;
      RooErrorHandler::softAbort() ;
      return ;
    }
    
    if (pdfReal->getFitMax()>(datReal->getFitMax()+1e-6)) {
      cout << "RooAbsOptGoodnessOfFit: ERROR maximum of PDF variable " << arg->GetName() 
	   << " is smaller than that of " << arg->GetName() << " in the dataset" << endl ;
      RooErrorHandler::softAbort() ;
      return ;
    }
    
  }
  delete iter ;
  
  // Copy data and strip entries lost by adjusted fit range
  _dataClone = ((RooAbsData&)data).reduce(*pdfDepSet) ;  
  delete pdfDepSet ;

  setEventCount(_dataClone->numEntries()) ;
 
 
  // Clone all PDF compents by copying all branch nodes
  RooArgSet tmp("PdfBranchNodeList") ;
  pdf.branchNodeServerList(&tmp) ;
  _pdfCloneSet = (RooArgSet*) tmp.snapshot(kFALSE) ;
  
  // Find the top level PDF in the snapshot list
  _pdfClone = (RooAbsPdf*) _pdfCloneSet->find(pdf.GetName()) ;

  // Attach PDF to data set
  _pdfClone->attachDataSet(*_dataClone) ;

  // Store normalization set
  _normSet = (RooArgSet*) data.get()->snapshot(kFALSE) ;

  // Remove projected dependents from normalization set
  if (projDeps.getSize()>0) {
    _projDeps = (RooArgSet*) projDeps.snapshot(kFALSE) ;
    _normSet->remove(*_projDeps,kTRUE,kTRUE) ;

    // Mark all projected dependents as such
    RooArgSet *projDataDeps = (RooArgSet*) _dataClone->get()->selectCommon(*_projDeps) ;
    projDataDeps->setAttribAll("projectedDependent") ;
    delete projDataDeps ;
  } 

  // Add parameters as servers
  RooArgSet* params = _pdfClone->getParameters(_dataClone) ;
  _paramSet.add(*params) ;
  delete params ;

  // Mark all projected dependents as such
  if (_projDeps) {
    RooArgSet *projDataDeps = (RooArgSet*) _dataClone->get()->selectCommon(*_projDeps) ;
    projDataDeps->setAttribAll("projectedDependent") ;
    delete projDataDeps ;
  }

  optimizeDirty() ;
}




RooAbsOptGoodnessOfFit::RooAbsOptGoodnessOfFit(const RooAbsOptGoodnessOfFit& other, const char* name) : 
  RooAbsGoodnessOfFit(other,name)
{
  // Don't do a thing in master mode
  if (operMode()!=Slave) {
    _normSet = other._normSet ? ((RooArgSet*) other._normSet->snapshot()) : 0 ;   
    return ;
  }

  _pdfCloneSet = (RooArgSet*) other._pdfCloneSet->snapshot(kFALSE) ;
  _pdfClone = (RooAbsPdf*) _pdfCloneSet->find(other._pdfClone->GetName()) ;

  // Copy the operMode attribute of all branch nodes
  TIterator* iter = _pdfCloneSet->createIterator() ;
  RooAbsArg* branch ;
  while(branch=(RooAbsArg*)iter->Next()) {
    branch->setOperMode(other._pdfCloneSet->find(branch->GetName())->operMode()) ;
  }
  delete iter ;

  // WVE Must use clone with cache redirection here
  _dataClone = (RooAbsData*) other._dataClone->cacheClone(_pdfCloneSet) ;

  _pdfClone->attachDataSet(*_dataClone) ;
  _normSet = (RooArgSet*) other._normSet->snapshot() ;
  
  if (other._projDeps) {
    _projDeps = (RooArgSet*) other._projDeps->snapshot() ;
  } else {
    _projDeps = 0 ;
  }
}



RooAbsOptGoodnessOfFit::~RooAbsOptGoodnessOfFit()
{
  if (operMode()==Slave) {
    delete _pdfCloneSet ;
    delete _dataClone ;
    delete _normSet ;
    delete _projDeps ;
  } 
}



Double_t RooAbsOptGoodnessOfFit::combinedValue(RooAbsReal** array, Int_t n) const
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


Bool_t RooAbsOptGoodnessOfFit::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) 
{
  RooAbsGoodnessOfFit::redirectServersHook(newServerList,mustReplaceAll,nameChange,isRecursive) ;
  if (operMode()!=Slave) return kFALSE ;  
  Bool_t ret = _pdfClone->recursiveRedirectServers(newServerList,kFALSE,nameChange) ;
  return ret ;
}


void RooAbsOptGoodnessOfFit::printCompactTreeHook(const char* indent) 
{
  RooAbsGoodnessOfFit::printCompactTreeHook(indent) ;
  if (operMode()!=Slave) return ;
  TString indent2(indent) ;
  indent2 += ">>" ;
  _pdfClone->printCompactTree(indent2) ;
}



void RooAbsOptGoodnessOfFit::constOptimize(ConstOpCode opcode) 
{
  // Driver function to propagate const optimization
  RooAbsGoodnessOfFit::constOptimize(opcode);
  if (operMode()!=Slave) return ;
  
  cout << "RooAbsOptGoodnessOfFit::constOptimize(" << GetName() << ") Action=" ;

  switch(opcode) {
  case Activate: 
    cout << "Activate" << endl ;
    doConstOpt() ;
    break ;
  case DeActivate:  
    cout << "DeActivate" << endl ;
    undoConstOpt() ;
    break ;
  case ConfigChange: 
    cout << "ConfigChange" << endl ;
    undoConstOpt() ;
    doConstOpt() ;
    break ;
  case ValueChange: 
    cout << "ValueChange" << endl ;
    undoConstOpt() ;
    doConstOpt() ;
    break ;
  }
}



void RooAbsOptGoodnessOfFit::optimizeDirty()
{
  _pdfClone->getVal(_normSet) ;

  RooArgSet branchList("branchList") ;
  _pdfClone->setOperMode(RooAbsArg::ADirty) ;
  _pdfClone->branchNodeServerList(&branchList) ;
  TIterator* bIter = branchList.createIterator() ;
  RooAbsArg* branch ;
  while(branch=(RooAbsArg*)bIter->Next()) {
    if (branch->dependsOn(*_dataClone->get())) {

      RooArgSet* bdep = branch->getDependents(_dataClone->get()) ;
      if (bdep->getSize()>0) {
	branch->setOperMode(RooAbsArg::ADirty) ;
      } else {
	//cout << "using lazy evaluation for node " << branch->GetName() << endl ;
      }
      delete bdep ;
    }
  }
  delete bIter ;

//   cout << "   disabling data dirty state prop" << endl ;
  _dataClone->setDirtyProp(kFALSE) ;
}



void RooAbsOptGoodnessOfFit::doConstOpt()
{
  // optimizeDirty must have been run first!

  // Find cachable branches and cache them with the data set
  RooArgSet cacheList ;
  findCacheableBranches(_pdfClone,_dataClone,cacheList) ;
  _dataClone->cacheArgs(cacheList,_normSet) ;  

  // Find unused data variables after caching and disable them
  RooArgSet pruneList("pruneList") ;
  findUnusedDataVariables(_pdfClone,_dataClone,pruneList) ;
  findRedundantCacheServers(_pdfClone,_dataClone,cacheList,pruneList) ;

  if (pruneList.getSize()!=0) {
    // Deactivate tree branches here
    cout << "RooAbsOptGoodnessOfFit::optimize: The following unused tree branches are deactivated: " ; 
    pruneList.Print("1") ;
    _dataClone->setArgStatus(pruneList,kFALSE) ;
  }

  TIterator* cIter = cacheList.createIterator() ;
  RooAbsArg *cacheArg ;
  while(cacheArg=(RooAbsArg*)cIter->Next()){
    cacheArg->setOperMode(RooAbsArg::AClean) ;
    //cout << "setting cached branch " << cacheArg->GetName() << " to AClean" << endl ;
  }
  delete cIter ;
}


void RooAbsOptGoodnessOfFit::undoConstOpt()
{
  // Delete the cache
  _dataClone->resetCache() ;

  // Reactivate all tree branches
  _dataClone->setArgStatus(*_dataClone->get(),kTRUE) ;
  
  // Reset all nodes to ADirty 
  optimizeDirty() ;
}





Bool_t RooAbsOptGoodnessOfFit::findCacheableBranches(RooAbsArg* arg, RooAbsData* dset, 
					    RooArgSet& cacheList) 
{
  // Find branch PDFs with all-constant parameters, and add them
  // to the dataset cache list

  // Evaluate function with current normalization in case servers
  // are created on the fly
  RooAbsReal* realArg = dynamic_cast<RooAbsReal*>(arg) ;
  if (realArg) {
    realArg->getVal(_normSet) ;
  }

  TIterator* sIter = arg->serverIterator() ;
  RooAbsArg* server ;

  while(server=(RooAbsArg*)sIter->Next()) {
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
	cout << "RooAbsOptGoodnessOfFit::optimize: component " 
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



void RooAbsOptGoodnessOfFit::findUnusedDataVariables(RooAbsPdf* pdf,RooAbsData* dset,RooArgSet& pruneList) 
{
  TIterator* vIter = dset->get()->createIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*) vIter->Next()) {
    if (!pdf->dependsOn(*arg)) pruneList.add(*arg) ;
  }
  delete vIter ;
}


void RooAbsOptGoodnessOfFit::findRedundantCacheServers(RooAbsPdf* pdf,RooAbsData* dset,RooArgSet& cacheList, RooArgSet& pruneList) 
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



Bool_t RooAbsOptGoodnessOfFit::allClientsCached(RooAbsArg* var, RooArgSet& cacheList)
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



