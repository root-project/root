/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooSimultaneous.cxx,v 1.63 2007/05/11 09:11:58 verkerke Exp $
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
// RooSimultaneous facilitates simultaneous fitting of multiple PDFs
// to subsets of a given dataset.
//
// The class takes an index category, which is interpreted as
// the data subset indicator, and a list of PDFs, each associated
// with a state of the index category. RooSimultaneous always returns
// the value of the PDF that is associated with the current value
// of the index category
//
// Extended likelihood fitting is supported if all components support
// extended likelihood mode. As for the returned probability density,
// the expected number of events for the PDF associated with the current
// state of the index category is returned.

#include "RooFit.h"

#include "TObjString.h"
#include "TObjString.h"
#include "RooSimultaneous.h"
#include "RooAbsCategoryLValue.h"
#include "RooPlot.h"
#include "RooCurve.h"
#include "RooRealVar.h"
#include "RooAddPdf.h"
#include "RooAbsData.h"
#include "Roo1DTable.h"
#include "RooSimGenContext.h"
#include "RooDataSet.h"
#include "RooCmdConfig.h"
#include "RooNameReg.h"
#include "RooGlobalFunc.h"
#include "RooNameReg.h"

ClassImp(RooSimultaneous)
;


RooSimultaneous::RooSimultaneous(const char *name, const char *title, 
				 RooAbsCategoryLValue& indexCat) : 
  RooAbsPdf(name,title), 
  _plotCoefNormSet("plotCoefNormSet","plotCoefNormSet",this,kFALSE,kFALSE),
  _normListMgr(10),
  _indexCat("indexCat","Index category",this,indexCat),
  _numPdf(0),
  _anyCanExtend(kFALSE),
  _anyMustExtend(kFALSE)
{
  // Constructor from index category. PDFs associated with indexCat
  // states can be added after construction with the addPdf() function.
  // 
  // RooSimultaneous can function without having a PDF associated
  // with every single state. The normalization in such cases is taken
  // from the number of registered PDFs, but getVal() will assert if
  // when called for an unregistered index state.
}


RooSimultaneous::RooSimultaneous(const char *name, const char *title, 
				 const RooArgList& pdfList, RooAbsCategoryLValue& indexCat) :
  RooAbsPdf(name,title), 
  _plotCoefNormSet("plotCoefNormSet","plotCoefNormSet",this,kFALSE,kFALSE),
  _normListMgr(10),
  _indexCat("indexCat","Index category",this,indexCat),
  _numPdf(0),
  _anyCanExtend(kFALSE),
  _anyMustExtend(kFALSE)
{
  // Constructor from index category and full list of PDFs. 
  // In this constructor form, a PDF must be supplied for each indexCat state
  // to avoid ambiguities. The PDFS are associated in order with the state of the
  // index category as listed by the index categories type iterator.
  //
  // PDFs may not overlap (i.e. share any variables) with the index category

  if (pdfList.getSize() != indexCat.numTypes()) {
    cout << "RooSimultaneous::ctor(" << GetName() 
	 << " ERROR: Number PDF list entries must match number of index category states, no PDFs added" << endl ;
    return ;    
  }

  // Iterator over PDFs and index cat states and add each pair
  TIterator* pIter = pdfList.createIterator() ;
  TIterator* cIter = indexCat.typeIterator() ;
  RooAbsPdf* pdf ;
  RooCatType* type ;
  while ((pdf=(RooAbsPdf*)pIter->Next())) {
    type = (RooCatType*) cIter->Next() ;
    addPdf(*pdf,type->GetName()) ;
    if (pdf->canBeExtended()) _anyCanExtend = kTRUE ;
    if (pdf->mustBeExtended()) _anyMustExtend = kTRUE ;
  }

  delete pIter ;
  delete cIter ;
}



RooSimultaneous::RooSimultaneous(const RooSimultaneous& other, const char* name) : 
  RooAbsPdf(other,name),
  _plotCoefNormSet("plotCoefNormSet",this,other._plotCoefNormSet),
  _normListMgr(other._normListMgr),
  _indexCat("indexCat",this,other._indexCat), 
  _numPdf(other._numPdf),
  _anyCanExtend(other._anyCanExtend),
  _anyMustExtend(other._anyMustExtend)
{
  // Copy constructor

  // Copy proxy list 
  TIterator* pIter = other._pdfProxyList.MakeIterator() ;
  RooRealProxy* proxy ;
  while ((proxy=(RooRealProxy*)pIter->Next())) {
    _pdfProxyList.Add(new RooRealProxy(proxy->GetName(),this,*proxy)) ;
  }
  delete pIter ;
}


RooSimultaneous::~RooSimultaneous() 
{
  // Destructor

  _pdfProxyList.Delete() ;
}


RooAbsPdf* RooSimultaneous::getPdf(const char* catName) const 
{
  // Retrieve the proxy by index name
  RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject(catName) ;
  return proxy ? ((RooAbsPdf*)proxy->absArg()) : 0 ;
}



Bool_t RooSimultaneous::addPdf(const RooAbsPdf& pdf, const char* catLabel)
{
  // Associate given PDF with index category state label 'catLabel'.
  // The names state must be already defined in the index category
  //
  // RooSimultaneous can function without having a PDF associated
  // with every single state. The normalization in such cases is taken
  // from the number of registered PDFs, but getVal() will assert if
  // when called for an unregistered index state.
  //
  // PDFs may not overlap (i.e. share any variables) with the index category

  // PDFs cannot overlap with the index category
  if (pdf.dependsOn(_indexCat.arg())) {
    cout << "RooSimultaneous::addPdf(" << GetName() << "): ERROR, PDF " << pdf.GetName() 
	 << " overlaps with index category " << _indexCat.arg().GetName() << endl ;
    return kTRUE ;
  }

  // Each index state can only have one PDF associated with it
  if (_pdfProxyList.FindObject(catLabel)) {
    cout << "RooSimultaneous::addPdf(" << GetName() << "): ERROR, index state " 
	 << catLabel << " has already an associated PDF" << endl ;
    return kTRUE ;
  }


  // Create a proxy named after the associated index state
  TObject* proxy = new RooRealProxy(catLabel,catLabel,this,(RooAbsPdf&)pdf) ;
  _pdfProxyList.Add(proxy) ;
  _numPdf += 1 ;

  if (pdf.canBeExtended()) _anyCanExtend = kTRUE ;
  if (pdf.mustBeExtended()) _anyMustExtend = kTRUE ;

  return kFALSE ;
}



Double_t RooSimultaneous::evaluate() const
{  
  // Return the current value: 
  // the value of the PDF associated with the current index category state

  // Retrieve the proxy by index name
  RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject((const char*) _indexCat) ;
  
  assert(proxy!=0) ;


  // Return the selected PDF value, normalized by the number of index states
  return ((RooAbsPdf*)(proxy->absArg()))->getVal(_normMgr.lastNormSet()) ; 
}



Double_t RooSimultaneous::expectedEvents(const RooArgSet* nset) const 
{
  // Return the number of expected events:
  // If the index is in nset, then return the sum of the expected events of all components,
  // otherwise return the number of expected events of the PDF associated with the current index category state

  if (nset->contains(_indexCat.arg())) {

    Double_t sum(0) ;

    TIterator* iter = _pdfProxyList.MakeIterator() ;
    RooRealProxy* proxy ;
    while((proxy=(RooRealProxy*)iter->Next())) {      
      sum += ((RooAbsPdf*)(proxy->absArg()))->expectedEvents(nset) ;
    }
    delete iter ;

    return sum ;
    
  } else {

    // Retrieve the proxy by index name
    RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject((const char*) _indexCat) ;
    
    assert(proxy!=0) ;
    
    // Return the selected PDF value, normalized by the number of index states
    return ((RooAbsPdf*)(proxy->absArg()))->expectedEvents(nset); 
  }
}



Int_t RooSimultaneous::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
					       const RooArgSet* normSet, const char* rangeName) const 
{
  // Distributed integration implementation
  
  // Declare that we can analytically integrate all requested observables
  analVars.add(allVars) ;

  // Retrieve (or create) the required partial integral list
  Int_t code ;

  // Check if this configuration was created before
  RooArgList* normList = _normListMgr.getNormList(this,normSet,&analVars,0,RooNameReg::ptr(rangeName)) ;
  if (normList) {
    code = _normListMgr.lastIndex() ;
    return code+1 ;
  }

  // Create the partial integral set for this request
  TIterator* iter = _pdfProxyList.MakeIterator() ;
  normList = new RooArgList("normList") ;
  RooRealProxy* proxy ;
  while((proxy=(RooRealProxy*)iter->Next())) {
    RooAbsReal* pdfInt = proxy->arg().createIntegral(analVars,normSet,0,rangeName) ;
    normList->addOwned(*pdfInt) ;
  }
  delete iter ;

  // Store the partial integral list and return the assigned code ;
  code = _normListMgr.setNormList(this,normSet,&analVars,normList,RooNameReg::ptr(rangeName)) ;
  
  return code+1 ;
}


Double_t RooSimultaneous::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* /*rangeName*/) const 
{
  // Return analytical integral defined by given scenario code

  // No integration scenario
  if (code==0) {
    return getVal(normSet) ;
  }

  // Partial integration scenarios, rangeName already encoded in 'code'
  RooArgList* normIntList = _normListMgr.getNormListByIndex(code-1) ;

  RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject((const char*) _indexCat) ;
  Int_t idx = _pdfProxyList.IndexOf(proxy) ;
  return ((RooAbsReal*)normIntList->at(idx))->getVal(normSet) ;
}





Bool_t RooSimultaneous::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t /*isRecursive*/) 
{
  Bool_t ret(kFALSE) ;  

  Int_t i ;
  for (i=0 ; i<_normListMgr.cacheSize() ; i++) {
    RooArgList* nlist = _normListMgr.getNormListByIndex(i) ;
    TIterator* iter = nlist->createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      ret |= arg->recursiveRedirectServers(newServerList,mustReplaceAll,nameChange) ;
    }
    delete iter ;
  }
  return ret ;
}




RooPlot* RooSimultaneous::plotOn(RooPlot *frame, RooLinkedList& cmdList) const
{
  // New experimental plotOn() with varargs...

  // See RooAbsPdf::plotOn() for description. Because a RooSimultaneous PDF cannot project out
  // its index category via integration, plotOn() will abort if this is requested without 
  // providing a projection dataset

  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;
  
  // Extract projection configuration from command list
  RooCmdConfig pc(Form("RooSimultaneous::plotOn(%s)",GetName())) ;
  pc.defineDouble("scaleFactor","Normalization",0,1.0) ;
  pc.defineInt("scaleType","Normalization",0,RooAbsPdf::Relative) ;
  pc.defineObject("projSet","Project",0) ;
  pc.defineObject("sliceSet","SliceVars",0) ;
  pc.defineObject("projDataSet","ProjData",0) ;
  pc.defineObject("projData","ProjData",1) ;
  pc.defineMutex("Project","SliceVars") ;
  pc.allowUndefined() ; // there may be commands we don't handle here

  // Process and check varargs 
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    return frame ;
  }

  const RooAbsData* projData = (const RooAbsData*) pc.getObject("projData") ;
  const RooArgSet* projDataSet = (const RooArgSet*) pc.getObject("projDataSet") ;
  const RooArgSet* sliceSet = (const RooArgSet*) pc.getObject("sliceSet") ;
  const RooArgSet* projSet = (const RooArgSet*) pc.getObject("projSet") ;  
  Double_t scaleFactor = pc.getDouble("scaleFactor") ;
  ScaleType stype = (ScaleType) pc.getInt("scaleType") ;

  // Check if we have a projection dataset
  if (!projData) {
    cout << "RooSimultaneous::plotOn(" << GetName() << ") ERROR: must have a projection dataset for index category" << endl ;
    return frame ;
  }

  // Make list of variables to be projected
  RooArgSet projectedVars ;
  if (sliceSet) {
    //cout << "frame->getNormVars() = " ; frame->getNormVars()->Print("1") ;

    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
    
    // Take out the sliced variables
    TIterator* iter = sliceSet->createIterator() ;
    RooAbsArg* sliceArg ;
    while((sliceArg=(RooAbsArg*)iter->Next())) {
      RooAbsArg* arg = projectedVars.find(sliceArg->GetName()) ;
      if (arg) {
	projectedVars.remove(*arg) ;
      } else {
	cout << "RooAbsReal::plotOn(" << GetName() << ") slice variable " 
	     << sliceArg->GetName() << " was not projected anyway" << endl ;
      }
    }
    delete iter ;
  } else if (projSet) {
    makeProjectionSet(frame->getPlotVar(),projSet,projectedVars,kFALSE) ;
  } else {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  }

  Bool_t projIndex(kFALSE) ;

  if (!_indexCat.arg().isDerived()) {
    // *** Error checking for a fundamental index category ***
    //cout << "RooSim::plotOn: index is fundamental" << endl ;
      
    // Check that the provided projection dataset contains our index variable
    if (!projData->get()->find(_indexCat.arg().GetName())) {
      cout << "RooSimultaneous::plotOn(" << GetName() << ") ERROR: Projection over index category "
	   << "requested, but projection data set doesn't contain index category" << endl ;
      return frame ;
    }

    if (projectedVars.find(_indexCat.arg().GetName())) {
      projIndex=kTRUE ;
    }

  } else {
    // *** Error checking for a composite index category ***

    // Determine if any servers of the index category are in the projectedVars
    TIterator* sIter = _indexCat.arg().serverIterator() ;
    RooAbsArg* server ;
    RooArgSet projIdxServers ;
    Bool_t anyServers(kFALSE) ;
    while((server=(RooAbsArg*)sIter->Next())) {
      if (projectedVars.find(server->GetName())) {
	anyServers=kTRUE ;
	projIdxServers.add(*server) ;
      }
    }
    delete sIter ;

    // Check that the projection dataset contains all the 
    // index category components we're projecting over

    // Determine if all projected servers of the index category are in the projection dataset
    sIter = projIdxServers.createIterator() ;
    Bool_t allServers(kTRUE) ;
    while((server=(RooAbsArg*)sIter->Next())) {
      if (!projData->get()->find(server->GetName())) {
	allServers=kFALSE ;
      }
    }
    delete sIter ;
    
    if (!allServers) {      
      cout << "RooSimultaneous::plotOn(" << GetName() 
	   << ") ERROR: Projection dataset doesn't contain complete set of index category dependents" << endl ;
      return frame ;
    }

    if (anyServers) {
      projIndex = kTRUE ;
    }
  } 
  
  // Calculate relative weight fractions of components
  Roo1DTable* wTable = projData->table(_indexCat.arg()) ;

  // If we don't project over the index, just do the regular plotOn
  if (!projIndex) {

    cout << "RooSimultaneous::plotOn(" << GetName() << ") plot on " << frame->getPlotVar()->GetName() 
	 << " represents a slice in the index category ("  << _indexCat.arg().GetName() << ")" << endl ;

    // Reduce projData: take out fitCat (component) columns and entries that don't match selected slice
    // Construct cut string to only select projection data event that match the current slice

    const RooAbsData* projDataTmp(projData) ;
    if (projData) {
      // Make list of categories columns to exclude from projection data
      RooArgSet* indexCatComps = _indexCat.arg().getObservables(frame->getNormVars());
      
      // Make cut string to exclude rows from projection data
      TString cutString ;
      TIterator* compIter =  indexCatComps->createIterator() ;    
      RooAbsCategory* idxComp ;
      Bool_t first(kTRUE) ;
      while((idxComp=(RooAbsCategory*)compIter->Next())) {
	if (!first) {
	  cutString.Append("&&") ;
	} else {
	  first=kFALSE ;
	}
	cutString.Append(Form("%s==%d",idxComp->GetName(),idxComp->getIndex())) ;
      }
      delete compIter ;

      // Make temporary projData without RooSim index category components
      RooArgSet projDataVars(*projData->get()) ;
      projDataVars.remove(*indexCatComps,kTRUE,kTRUE) ;
      
      projDataTmp = ((RooAbsData*)projData)->reduce(projDataVars,cutString) ;
      delete indexCatComps ;
    }

    // Multiply scale factor with fraction of events in current state of index

//     RooPlot* retFrame =  getPdf(_indexCat.arg().getLabel())->plotOn(frame,drawOptions,
// 					   scaleFactor*wTable->getFrac(_indexCat.arg().getLabel()),
// 					   stype,projDataTmp,projSet) ;
    
    // Override normalization and projection dataset
    RooLinkedList cmdList2(cmdList) ;
    RooCmdArg tmp1 = RooFit::Normalization(scaleFactor*wTable->getFrac(_indexCat.arg().getLabel()),stype) ;
    RooCmdArg tmp2 = RooFit::ProjWData(*projDataSet,*projDataTmp) ;

    // WVE -- do not adjust normalization for asymmetry plots
    if (!cmdList.find("Asymmetry")) {
      cmdList2.Add(&tmp1) ;
    }
    cmdList2.Add(&tmp2) ;

    // Plot single component
    RooPlot* retFrame =  getPdf(_indexCat.arg().getLabel())->plotOn(frame,cmdList2) ;

    delete wTable ;
    return retFrame ;
  }

  // If we project over the index, plot using a temporary RooAddPdf
  // using the weights from the data as coefficients

  // Make a deep clone of our index category
  RooArgSet* idxCloneSet = (RooArgSet*) RooArgSet(_indexCat.arg()).snapshot(kTRUE) ;
  RooAbsCategoryLValue* idxCatClone = (RooAbsCategoryLValue*) idxCloneSet->find(_indexCat.arg().GetName()) ;

  // Build the list of indexCat components that are sliced
  RooArgSet* idxCompSliceSet = _indexCat.arg().getObservables(frame->getNormVars()) ;
  idxCompSliceSet->remove(projectedVars,kTRUE,kTRUE) ;
  TIterator* idxCompSliceIter = idxCompSliceSet->createIterator() ;

  // Make a new expression that is the weighted sum of requested components
  RooArgList pdfCompList ;
  RooArgList wgtCompList ;
//RooAbsPdf* pdf ;
  RooRealProxy* proxy ;
  TIterator* pIter = _pdfProxyList.MakeIterator() ;
  Double_t sumWeight(0) ;
  while((proxy=(RooRealProxy*)pIter->Next())) {

    idxCatClone->setLabel(proxy->name()) ;

    // Determine if this component is the current slice (if we slice)
    Bool_t skip(kFALSE) ;
    idxCompSliceIter->Reset() ;
    RooAbsCategory* idxSliceComp ;
    while((idxSliceComp=(RooAbsCategory*)idxCompSliceIter->Next())) {
      RooAbsCategory* idxComp = (RooAbsCategory*) idxCloneSet->find(idxSliceComp->GetName()) ;
      if (idxComp->getIndex()!=idxSliceComp->getIndex()) {
	skip=kTRUE ;
	break ;
      }
    }
    if (skip) continue ;
 
    // Instantiate a RRV holding this pdfs weight fraction
    RooRealVar *wgtVar = new RooRealVar(proxy->name(),"coef",wTable->getFrac(proxy->name())) ;
    wgtCompList.addOwned(*wgtVar) ;
    sumWeight += wTable->getFrac(proxy->name()) ;

    // Add the PDF to list list
    pdfCompList.add(proxy->arg()) ;
  }

  TString plotVarName(GetName()) ;
  RooAddPdf *plotVar = new RooAddPdf(plotVarName,"weighted sum of RS components",pdfCompList,wgtCompList) ;

  // Fix appropriate coefficient normalization in plot function
  if (_plotCoefNormSet.getSize()>0) {
    plotVar->fixAddCoefNormalization(_plotCoefNormSet) ;
  }

  RooAbsData* projDataTmp(0) ;
  RooArgSet projSetTmp ;
  if (projData) {
    
    // Construct cut string to only select projection data event that match the current slice
    TString cutString ;
    if (idxCompSliceSet->getSize()>0) {
      idxCompSliceIter->Reset() ;
      RooAbsCategory* idxSliceComp ;
      Bool_t first(kTRUE) ;
      while((idxSliceComp=(RooAbsCategory*)idxCompSliceIter->Next())) {
	if (!first) {
	  cutString.Append("&&") ;
	} else {
	  first=kFALSE ;
	}
	cutString.Append(Form("%s==%d",idxSliceComp->GetName(),idxSliceComp->getIndex())) ;
      }
    }

    // Make temporary projData without RooSim index category components
    RooArgSet projDataVars(*projData->get()) ;
    RooArgSet* idxCatServers = _indexCat.arg().getObservables(frame->getNormVars()) ;

    projDataVars.remove(*idxCatServers,kTRUE,kTRUE) ;

    if (idxCompSliceSet->getSize()>0) {
      projDataTmp = ((RooAbsData*)projData)->reduce(projDataVars,cutString) ;
    } else {
      projDataTmp = ((RooAbsData*)projData)->reduce(projDataVars) ;      
    }

    

    if (projSet) {
      projSetTmp.add(*projSet) ;
      projSetTmp.remove(*idxCatServers,kTRUE,kTRUE);
    }

    
    delete idxCatServers ;
  }


  if (_indexCat.arg().isDerived() && idxCompSliceSet->getSize()>0) {
    cout << "RooSimultaneous::plotOn(" << GetName() << ") plot on " << frame->getPlotVar()->GetName() 
	 << " represents a slice in index category components " ; idxCompSliceSet->Print("1") ;

    RooArgSet* idxCompProjSet = _indexCat.arg().getObservables(frame->getNormVars()) ;
    idxCompProjSet->remove(*idxCompSliceSet,kTRUE,kTRUE) ;
    if (idxCompProjSet->getSize()>0) {
      cout << "RooSimultaneous::plotOn(" << GetName() << ") plot on " << frame->getPlotVar()->GetName() 
	   << " averages with data index category components " ; idxCompProjSet->Print("1") ;
    }
    delete idxCompProjSet ;
  } else {
    cout << "RooSimultaneous::plotOn(" << GetName() << ") plot on " << frame->getPlotVar()->GetName() 
	 << " averages with data index category (" << _indexCat.arg().GetName() << ")" << endl ;
  }


  // Override normalization and projection dataset
  RooLinkedList cmdList2(cmdList) ;

  RooCmdArg tmp1 = RooFit::Normalization(scaleFactor*sumWeight,stype) ;
  RooCmdArg tmp2 = RooFit::ProjWData(*projDataSet,*projDataTmp) ;
  // WVE -- do not adjust normalization for asymmetry plots
  if (!cmdList.find("Asymmetry")) {
    cmdList2.Add(&tmp1) ;
  }
  cmdList2.Add(&tmp2) ;

  RooPlot* frame2 ;
  if (projSetTmp.getSize()>0) {
    // Plot temporary function  
    RooCmdArg tmp3 = RooFit::Project(projSetTmp) ;
    cmdList2.Add(&tmp3) ;
    frame2 = plotVar->plotOn(frame,cmdList2) ;
  } else {
    // Plot temporary function  
    frame2 = plotVar->plotOn(frame,cmdList2) ;
  }

  // Cleanup
  delete pIter ;
  delete wTable ;
  delete idxCloneSet ;
  delete idxCompSliceIter ;
  delete idxCompSliceSet ;
  delete plotVar ;

  if (projDataTmp) delete projDataTmp ;

  return frame2 ;
}



RooPlot* RooSimultaneous::plotOn(RooPlot *frame, Option_t* drawOptions, Double_t scaleFactor, 
				 ScaleType stype, const RooAbsData* projData, const RooArgSet* projSet,
				 Double_t /*precision*/, Bool_t /*shiftToZero*/, const RooArgSet* /*projDataSet*/,
				 Double_t /*rangeLo*/, Double_t /*rangeHi*/, RooCurve::WingMode /*wmode*/) const
{
  // Forward to new implementation

  // Make command list
  RooLinkedList cmdList ;
  cmdList.Add(new RooCmdArg(RooFit::DrawOption(drawOptions))) ;
  cmdList.Add(new RooCmdArg(RooFit::Normalization(scaleFactor,stype))) ;
  if (projData) cmdList.Add(new RooCmdArg(RooFit::ProjWData(*projData))) ;
  if (projSet) cmdList.Add(new RooCmdArg(RooFit::Project(*projSet))) ;

  // Call new method
  RooPlot* ret = plotOn(frame,cmdList) ;

  // Cleanup
  cmdList.Delete() ;
  return ret ;  
}



void RooSimultaneous::selectNormalization(const RooArgSet* normSet, Bool_t /*force*/) 
{
  _plotCoefNormSet.removeAll() ;
  if (normSet) _plotCoefNormSet.add(*normSet) ;
}

void RooSimultaneous::selectNormalizationRange(const char* normRange, Bool_t /*force*/) 
{
  _plotCoefNormRange = RooNameReg::ptr(normRange) ;
}




RooAbsGenContext* RooSimultaneous::genContext(const RooArgSet &vars, const RooDataSet *prototype, 
					      const RooArgSet* auxProto, Bool_t verbose) const 
{
  const char* idxCatName = _indexCat.arg().GetName() ;
  const RooArgSet* protoVars = prototype ? prototype->get() : 0 ;

  if (vars.find(idxCatName) || (protoVars && protoVars->find(idxCatName))) {
    // Generating index category: return special sim-context
    return new RooSimGenContext(*this,vars,prototype,auxProto,verbose) ;
  } else if (_indexCat.arg().isDerived()) {
    // Generating dependents of a derived index category

    // Determine if we none,any or all servers
    Bool_t anyServer(kFALSE), allServers(kTRUE) ;
    if (prototype) {
      TIterator* sIter = _indexCat.arg().serverIterator() ;
      RooAbsArg* server ;
      while((server=(RooAbsArg*)sIter->Next())) {
	if (prototype->get()->find(server->GetName())) {
	  anyServer=kTRUE ;
	} else {
	  allServers=kFALSE ;
	}
      }
      delete sIter ;
    } else {
      allServers=kTRUE ;
    }

    if (allServers) {
      // Use simcontext if we have all servers
      return new RooSimGenContext(*this,vars,prototype,auxProto,verbose) ;
    } else if (!allServers && anyServer) {
      // Abort if we have only part of the servers
      cout << "RooSimultaneous::genContext: ERROR: prototype must include either all "
	   << " components of the RooSimultaneous index category or none " << endl ;
      return 0 ; 
    } 
    // Otherwise make single gencontext for current state
  } 

  // Not generating index cat: return context for pdf associated with present index state
  RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject(_indexCat.arg().getLabel()) ;
  if (!proxy) {
    cout << "RooSimultaneous::genContext(" << GetName() 
	 << ") ERROR: no PDF associated with current state (" 
         << _indexCat.arg().GetName() << "=" << _indexCat.arg().getLabel() << ")" << endl ; 
    return 0 ;
  }
  return ((RooAbsPdf*)proxy->absArg())->genContext(vars,prototype,auxProto,verbose) ;
}

