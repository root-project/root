/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimultaneous.cc,v 1.38 2002/06/08 00:45:01 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   25-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
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

#include "TObjString.h"
#include "RooFitCore/RooSimultaneous.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooCurve.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooAddPdf.hh"
#include "RooFitCore/RooAbsData.hh"
#include "RooFitCore/Roo1DTable.hh"
#include "RooFitCore/RooSimGenContext.hh"
#include "RooFitCore/RooDataSet.hh"

ClassImp(RooSimultaneous)
;


RooSimultaneous::RooSimultaneous(const char *name, const char *title, 
				 RooAbsCategoryLValue& indexCat) : 
  RooAbsPdf(name,title), _numPdf(0.),
  _indexCat("indexCat","Index category",this,indexCat),
  _plotCoefNormSet("plotCoefNormSet","plotCoefNormSet",this,kFALSE,kFALSE),
  _codeReg(10),
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
  RooAbsPdf(name,title), _numPdf(0.),
  _indexCat("indexCat","Index category",this,indexCat),
  _plotCoefNormSet("plotCoefNormSet","plotCoefNormSet",this,kFALSE,kFALSE),
  _codeReg(10),
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
  while (pdf=(RooAbsPdf*)pIter->Next()) {
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
  _indexCat("indexCat",this,other._indexCat), _numPdf(other._numPdf),
  _plotCoefNormSet("plotCoefNormSet",this,other._plotCoefNormSet),
  _codeReg(other._codeReg),
  _anyCanExtend(other._anyCanExtend),
  _anyMustExtend(other._anyMustExtend)
{
  // Copy constructor

  // Copy proxy list 
  TIterator* pIter = other._pdfProxyList.MakeIterator() ;
  RooRealProxy* proxy ;
  while (proxy=(RooRealProxy*)pIter->Next()) {
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
  _numPdf += 1.0 ;

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
  return ((RooAbsPdf*)(proxy->absArg()))->getVal(_lastNormSet) ; 
}



Double_t RooSimultaneous::expectedEvents() const 
{
  // Return the number of expected events:
  // the number of expected events of the PDF associated with the current index category state

  // Retrieve the proxy by index name
  RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject((const char*) _indexCat) ;
  
  assert(proxy!=0) ;

  // Return the selected PDF value, normalized by the number of index states
  return ((RooAbsPdf*)(proxy->absArg()))->expectedEvents() ;
}



RooFitContext* RooSimultaneous::fitContext(const RooAbsData& dset, const RooArgSet* projDeps) const 
{
  return new RooSimFitContext(&dset,this,projDeps) ;
}



Int_t RooSimultaneous::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code
  //
  // RooSimultaneous queries each component PDF for its analytical integration capability of the requested
  // set ('allVars'). It finds the largest common set of variables that can be integrated
  // by all components. If such a set exists, it reconfirms that each component is capable of
  // analytically integrating the common set, and combines the components individual integration
  // codes into a single integration code valid for RooSimultaneous.
  
  TIterator* pdfIter = _pdfProxyList.MakeIterator() ;

//RooAbsPdf* pdf ;
  RooRealProxy* proxy ;
  RooArgSet allAnalVars(allVars) ;
  TIterator* avIter = allVars.createIterator() ;

  Int_t n(0) ;
  // First iteration, determine what each component can integrate analytically
  while(proxy=(RooRealProxy*)pdfIter->Next()) {
    RooArgSet subAnalVars ;
    Int_t subCode = proxy->arg().getAnalyticalIntegralWN(allVars,subAnalVars,normSet) ;
    
    // If a dependent is not supported by any of the components, 
    // it is dropped from the combined analytic list
    avIter->Reset() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)avIter->Next()) {
      if (!subAnalVars.find(arg->GetName())) {
	allAnalVars.remove(*arg,kTRUE) ;
      }
    }
    n++ ;
  }

  if (allAnalVars.getSize()==0) {
    delete avIter ;
    return 0 ;
  }

  // Now retrieve the component codes for the common set of analytic dependents 
  pdfIter->Reset() ;
  n=0 ;
  Int_t* subCode = new Int_t[_pdfProxyList.GetSize()] ;
  Bool_t allOK(kTRUE) ;
  while(proxy=(RooRealProxy*)pdfIter->Next()) {
    RooArgSet subAnalVars ;
    subCode[n] = proxy->arg().getAnalyticalIntegralWN(allAnalVars,subAnalVars,normSet) ;
    if (subCode[n]==0) {
      cout << "RooSimultaneous::getAnalyticalIntegral(" << GetName() << ") WARNING: component PDF " << proxy->arg().GetName() 
	   << "   advertises inconsistent set of integrals (e.g. (X,Y) but not X or Y individually.)"
	   << "   Distributed analytical integration disabled. Please fix PDF" << endl ;
      allOK = kFALSE ;
    }
    n++ ;
  }  
  if (!allOK) return 0 ;

  analVars.add(allAnalVars) ;
  Int_t masterCode = _codeReg.store(subCode,_pdfProxyList.GetSize())+1 ;

  delete[] subCode ;
  delete avIter ;
  delete pdfIter ;
  return masterCode ;
}


Double_t RooSimultaneous::analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const 
{
  // Return analytical integral defined by given scenario code
 
  if (code==0) return getVal(normSet) ;

  const Int_t* subCode = _codeReg.retrieve(code-1) ;
  if (!subCode) {
    cout << "RooSimultaneous::analyticalIntegral(" << GetName() << "): ERROR unrecognized integration code, " << code << endl ;
    assert(0) ;    
  }

  // Calculate the current value of this object
  RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject((const char*) _indexCat) ;
  Int_t idx = _pdfProxyList.IndexOf(proxy) ;

  return proxy->arg().analyticalIntegralWN(subCode[idx],normSet) ;
}





RooPlot* RooSimultaneous::plotOn(RooPlot *frame, Option_t* drawOptions, Double_t scaleFactor, 
				 ScaleType stype, const RooAbsData* projData, const RooArgSet* projSet) const 
{
  // See RooAbsPdf::plotOn() for description. Because a RooSimultaneous PDF cannot project out
  // its index category via integration, plotOn() will abort if this is requested without 
  // providing a projection dataset
  
  // Check if we have a projection dataset
  if (!projData) {
    cout << "RooSimultaneous::plotOn(" << GetName() << ") ERROR: must have a projection dataset for index category" << endl ;
    return frame ;
  }

  // Make list of variables to be projected
  RooArgSet projectedVars ;
  if (projSet) {
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
    while(server=(RooAbsArg*)sIter->Next()) {
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
    while(server=(RooAbsArg*)sIter->Next()) {
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

    if (anyServers) projIndex = kTRUE ;
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
      RooArgSet* indexCatComps = _indexCat.arg().getDependents(frame->getNormVars());
      
      // Make cut string to exclude rows from projection data
      TString cutString ;
      TIterator* compIter =  indexCatComps->createIterator() ;    
      RooAbsCategory* idxComp ;
      Bool_t first(kTRUE) ;
      while(idxComp=(RooAbsCategory*)compIter->Next()) {
	if (!first) {
	  cutString.Append("&&") ;
	} else {
	  first=kFALSE ;
	}
	cutString.Append(Form("%s==%d",idxComp->GetName(),idxComp->getIndex())) ;
      }
      
      // Make temporary projData without RooSim index category components
      RooArgSet projDataVars(*projData->get()) ;
      projDataVars.remove(*indexCatComps,kTRUE,kTRUE) ;
      
      projDataTmp = ((RooAbsData*)projData)->reduce(projDataVars,cutString) ;
      delete indexCatComps ;
    }

    // Multiply scale factor with fraction of events in current state of index
    //cout << "wTable->getFrac(" << _indexCat.arg().getLabel() << ") = " << wTable->getFrac(_indexCat.arg().getLabel()) << endl ;
    RooPlot* retFrame =  RooAbsPdf::plotOn(frame,drawOptions,
					   scaleFactor*wTable->getFrac(_indexCat.arg().getLabel()),
					   stype,projDataTmp,projSet) ;
    delete wTable ;
    return retFrame ;
  }

  // If we project over the index, plot using a temporary RooAddPdf
  // using the weights from the data as coefficients

  // Make a deep clone of our index category
  RooArgSet* idxCloneSet = (RooArgSet*) RooArgSet(_indexCat.arg()).snapshot(kTRUE) ;
  RooAbsCategoryLValue* idxCatClone = (RooAbsCategoryLValue*) idxCloneSet->find(_indexCat.arg().GetName()) ;

  // Build the list of indexCat components that are sliced
  RooArgSet* idxCompSliceSet = _indexCat.arg().getDependents(frame->getNormVars()) ;
  idxCompSliceSet->remove(projectedVars,kTRUE,kTRUE) ;
  TIterator* idxCompSliceIter = idxCompSliceSet->createIterator() ;

  // Make a new expression that is the weighted sum of requested components
  RooArgList pdfCompList ;
  RooArgList wgtCompList ;
//RooAbsPdf* pdf ;
  RooRealProxy* proxy ;
  TIterator* pIter = _pdfProxyList.MakeIterator() ;
  Double_t plotFrac(0) ;
  Double_t sumWeight(0) ;
  while(proxy=(RooRealProxy*)pIter->Next()) {

    idxCatClone->setLabel(proxy->name()) ;

    // Determine if this component is the current slice (if we slice)
    Bool_t skip(kFALSE) ;
    idxCompSliceIter->Reset() ;
    RooAbsCategory* idxSliceComp ;
    while(idxSliceComp=(RooAbsCategory*)idxCompSliceIter->Next()) {
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
    plotVar->fixCoefNormalization(_plotCoefNormSet) ;
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
      while(idxSliceComp=(RooAbsCategory*)idxCompSliceIter->Next()) {
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
    RooArgSet* idxCatServers = _indexCat.arg().getDependents(frame->getNormVars()) ;
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

    RooArgSet* idxCompProjSet = _indexCat.arg().getDependents(frame->getNormVars()) ;
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

  // Plot temporary function  
  RooPlot* frame2 = plotVar->plotOn(frame,drawOptions,scaleFactor*sumWeight,stype,projDataTmp,projSet?&projSetTmp:0) ;

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



void RooSimultaneous::selectNormalization(const RooArgSet* normSet, Bool_t force) 
{
  _plotCoefNormSet.removeAll() ;
  if (normSet) _plotCoefNormSet.add(*normSet) ;
}





RooAbsGenContext* RooSimultaneous::genContext(const RooArgSet &vars, 
					const RooDataSet *prototype, Bool_t verbose) const 
{
  const char* idxCatName = _indexCat.arg().GetName() ;
  const RooArgSet* protoVars = prototype ? prototype->get() : 0 ;

  if (vars.find(idxCatName) || (protoVars && protoVars->find(idxCatName))) {
    // Generating index category: return special sim-context
    return new RooSimGenContext(*this,vars,prototype,verbose) ;
  } else if (_indexCat.arg().isDerived()) {
    // Generating dependents of a derived index category

    // Determine if we none,any or all servers
    Bool_t anyServer(kFALSE), allServers(kTRUE) ;
    if (prototype) {
      TIterator* sIter = _indexCat.arg().serverIterator() ;
      RooAbsArg* server ;
      while(server=(RooAbsArg*)sIter->Next()) {
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
      return new RooSimGenContext(*this,vars,prototype,verbose) ;
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
  return ((RooAbsPdf*)proxy->absArg())->genContext(vars,prototype,verbose) ;
}

