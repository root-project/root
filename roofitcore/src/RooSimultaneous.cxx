/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimultaneous.cc,v 1.15 2001/09/28 21:59:29 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   25-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "TObjString.h"
#include "RooFitCore/RooSimultaneous.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooCurve.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooAddPdf.hh"
#include "RooFitCore/RooAbsData.hh"
#include "RooFitCore/Roo1DTable.hh"

ClassImp(RooSimultaneous)
;


RooSimultaneous::RooSimultaneous(const char *name, const char *title, 
				 RooAbsCategoryLValue& indexCat) : 
  RooAbsPdf(name,title), _numPdf(0.),
  _indexCat("indexCat","Index category",this,indexCat),
  _codeReg(10),
  _allExtendable(kTRUE)
{
}


RooSimultaneous::RooSimultaneous(const char *name, const char *title, 
				 const RooArgList& pdfList, RooAbsCategoryLValue& indexCat) :
  RooAbsPdf(name,title), _numPdf(0.),
  _indexCat("indexCat","Index category",this,indexCat),
  _codeReg(10),
  _allExtendable(kTRUE)
{
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
    if (!pdf->canBeExtended()) _allExtendable = kFALSE ;
  }

  delete pIter ;
  delete cIter ;
}



RooSimultaneous::RooSimultaneous(const RooSimultaneous& other, const char* name) : 
  RooAbsPdf(other,name),
  _indexCat("indexCat",this,other._indexCat), _numPdf(other._numPdf),
  _codeReg(other._codeReg),
  _allExtendable(other._allExtendable)
{
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
  _pdfProxyList.Delete() ;
}



Bool_t RooSimultaneous::addPdf(const RooAbsPdf& pdf, const char* catLabel)
{
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

  if (!pdf.canBeExtended()) _allExtendable = kFALSE ;

  return kFALSE ;
}



Double_t RooSimultaneous::evaluate() const
{  
  // Retrieve the proxy by index name
  RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject((const char*) _indexCat) ;
  
  assert(proxy!=0) ;

  // Return the selected PDF value, normalized by the number of index states
  return ((RooAbsPdf*)(proxy->absArg()))->getVal(_lastNormSet) / _numPdf ;
}


Double_t RooSimultaneous::expectedEvents() const 
{
  // Retrieve the proxy by index name
  RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject((const char*) _indexCat) ;
  
  assert(proxy!=0) ;

  // Return the selected PDF value, normalized by the number of index states
  return ((RooAbsPdf*)(proxy->absArg()))->expectedEvents() ;
}



const RooFitResult* RooSimultaneous::fitTo(RooAbsData& data, Option_t *fitOpt, Option_t *optOpt) 
{
  TString opts = optOpt ;
  opts.ToLower() ;

  if (!opts.Contains("s")) {
  // Fit this PDF to given data set using a regular fit context    
    return RooAbsPdf::fitTo(data,fitOpt,optOpt) ;
  } 

  // Fit this PDF to given data set using a SimFit context
  RooSimFitContext context(&data,this) ;
  return context.fit(fitOpt,optOpt) ;  
}



Int_t RooSimultaneous::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code
    // This PDF is by construction normalized
  TIterator* pdfIter = _pdfProxyList.MakeIterator() ;

  RooAbsPdf* pdf ;
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
  //cout << "RooSimultaneous::aI(" << GetName() << ") code = " << code << " normSet = " << normSet << endl ;

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
				 ScaleType stype, const RooArgSet* projSet) const 
{
  // We need a dataset for plotting...
  cout << "RooSimultaneous::plotOn(" << GetName() << ") Cannot plot simultaneous PDF without data set" << endl
       << "to determine relative fractions of PDF components." << endl
       << "Please use RooSimultaneous::plot(RooPlot*,RooAbsData*,...)" << endl ;
  return frame ;
}



RooPlot* RooSimultaneous::plotOn(RooPlot *frame, RooAbsData* wdata, Option_t* drawOptions, Double_t scaleFactor, 
				 ScaleType stype, const RooArgSet* projSet) const
{
  RooArgSet allComponents ;
  TIterator *iter = _pdfProxyList.MakeIterator() ;
  RooRealProxy* proxy ;
  while(proxy=(RooRealProxy*)iter->Next()) {
    allComponents.add(proxy->arg()) ;
  }
  delete iter ;
  return plotCompOn(frame,wdata,allComponents,drawOptions,scaleFactor,stype,projSet) ;
}




RooPlot* RooSimultaneous::plotCompOn(RooPlot *frame, RooAbsData* wdata, const char* indexLabelList, Option_t* drawOptions,
				     Double_t scaleFactor, ScaleType stype, const RooArgSet* projSet) const
{
  RooArgSet allComponents ;

  // Process comma separated index label list
  char labelList[1024] ;
  strcpy(labelList,indexLabelList) ;  
  char* label  = strtok(labelList,",") ;
  while(label) {
    // Look for a pdf proxy with this labels name
    RooRealProxy* proxy =  (RooRealProxy*) _pdfProxyList.FindObject(label) ;

    // Add to list if found, ignore with warning otherwise
    if (proxy) {
      allComponents.add(proxy->arg()) ;
    } else {
      cout << "RooSimultaneous::plotCompOn(" << GetName() 
	   << ") WARNING: There is no component PDF associated with index label " 
	   << label << ", ignoring" << endl ;
    }
    label = strtok(0,",") ;
  }

  return plotCompOn(frame,wdata,allComponents,drawOptions,scaleFactor,stype,projSet) ;
}



RooPlot* RooSimultaneous::plotCompOn(RooPlot *frame, RooAbsData* wdata, const RooArgSet& compSet, Option_t* drawOptions,
				     Double_t scaleFactor, ScaleType stype, const RooArgSet* projSet) const
{
  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // Calculate relative weight fractions of components
  Roo1DTable* wTable = wdata->table(_indexCat.arg()) ;

  // Make a new expression that is the weighted sum of requested components
  RooArgList pdfCompList ;
  RooArgList wgtCompList ;
  RooAbsPdf* pdf ;
  RooRealProxy* proxy ;
  TIterator* cIter = compSet.createIterator() ;
  TIterator* pIter = _pdfProxyList.MakeIterator() ;
  Double_t plotFrac(0) ;
  while(pdf=(RooAbsPdf*)cIter->Next()) {

    // Check if listed component is indeed contained in this PDF
    if (!dependsOn(*pdf)) {
      cout << "RooSimultaneous::plotCompOn(" << GetName() << ") WARNING " 
	   << pdf->GetName() << " is not a component of this pdf, ignoring" << endl ;
      continue ;
    }
    
    // Find proxy for this pdf (we need the proxy name to look up the weight table
    pIter->Reset() ;
    while(proxy=(RooRealProxy*)pIter->Next()) {
      if (!TString(proxy->arg().GetName()).CompareTo(pdf->GetName())) break ;
    }
    
    // Add pdf to plot list
    pdfCompList.add(*pdf) ;

    // Instantiate a RRV holding this pdfs weight fraction
    RooRealVar *wgtVar = new RooRealVar(proxy->name(),"coef",wTable->getFrac(proxy->name())) ;
    plotFrac += wgtVar->getVal() ;
    wgtCompList.addOwned(*wgtVar) ;
  }
  delete pIter ;
  delete cIter ;
  delete wTable ;

  // Did we select anything
  if (plotFrac==0) {
    cout << "RooSimultaneous::plotCompOn(" << GetName() << ") no components selected, plotting aborted" << endl ;
    return frame ;
  }

  RooAddPdf *plotVar = new RooAddPdf("plotVar","weighted sum of RS components",pdfCompList,wgtCompList) ;

  // Plot temporary function
  cout << "RooSimultaneous::plotCompOn(" << GetName() << ") plotting components " ; pdfCompList.Print("1") ;
  RooPlot* frame2 = plotVar->plotOn(frame,drawOptions,scaleFactor*plotFrac,stype,projSet) ;

  // Cleanup
  delete plotVar ;

  return frame2 ;
  
}
