/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsOptGoodnessOfFit.cc,v 1.13 2004/04/05 22:43:55 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
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

  _pdfClone->optimizeDirty(*_dataClone,_normSet) ;
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


void RooAbsOptGoodnessOfFit::printCompactTreeHook(ostream& os, const char* indent) 
{
  RooAbsGoodnessOfFit::printCompactTreeHook(os,indent) ;
  if (operMode()!=Slave) return ;
  TString indent2(indent) ;
  indent2 += ">>" ;
  _pdfClone->printCompactTree(os,indent2) ;
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
    _pdfClone->doConstOpt(*_dataClone,_normSet) ;
    break ;
  case DeActivate:  
    cout << "DeActivate" << endl ;
    _pdfClone->undoConstOpt(*_dataClone,_normSet) ;
    break ;
  case ConfigChange: 
    cout << "ConfigChange" << endl ;
    _pdfClone->undoConstOpt(*_dataClone,_normSet) ;
    _pdfClone->doConstOpt(*_dataClone,_normSet) ;
    break ;
  case ValueChange: 
    cout << "ValueChange" << endl ;
    _pdfClone->undoConstOpt(*_dataClone,_normSet) ;
    _pdfClone->doConstOpt(*_dataClone,_normSet) ;
    break ;
  }
}





