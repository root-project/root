/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimultaneous.cc,v 1.11 2001/09/11 00:30:32 verkerke Exp $
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
  _codeReg(10)
{
}

RooSimultaneous::RooSimultaneous(const RooSimultaneous& other, const char* name) : 
  RooAbsPdf(other,name),
  _indexCat("indexCat",this,other._indexCat), _numPdf(other._numPdf),
  _codeReg(other._codeReg)
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

  return kFALSE ;
}



Double_t RooSimultaneous::evaluate() const
{
//   // Require that all states have an associated PDF
//   if (_pdfProxyList.GetSize() != _indexCat.arg().numTypes()) {
//     cout << "RooSimultaneous::evaluate(" << GetName() 
// 	 << "): ERROR, number of PDFs and number of index states do not match" << endl ;
//     return 0 ;
//   }

  // Retrieve the proxy by index name
  RooRealProxy* proxy = (RooRealProxy*) _pdfProxyList.FindObject((const char*) _indexCat) ;
  
  assert(proxy!=0) ;

  // Return the selected PDF value, normalized by the number of index states
  return ((RooAbsPdf*)(proxy->absArg()))->getVal(_lastNormSet) / _numPdf ;
}



Int_t RooSimultaneous::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const 
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
    Int_t subCode = proxy->arg().getAnalyticalIntegral(allVars,subAnalVars,normSet) ;
    cout << "RooSimultaneous::getAI(" << GetName() << ") ITER1 subCode(" << n << "," << pdf->GetName() << ") = " << subCode << endl ;
    
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
    subCode[n] = proxy->arg().getAnalyticalIntegral(allAnalVars,subAnalVars,normSet) ;
    cout << "RooSimultaneous::getAI(" << GetName() << ") ITER2 subCode(" << n << "," << pdf->GetName() << ") = " << subCode << endl ;
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


Double_t RooSimultaneous::analyticalIntegral(Int_t code, const RooArgSet* normSet) const 
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

  return proxy->arg().analyticalIntegral(subCode[idx],normSet) ;
}


RooPlot *RooSimultaneous::plotOn(RooPlot* frame, RooAbsData* wdata, Option_t* drawOptions, Double_t scaleFactor) const {
  // Plot a smooth curve of this object's value on the specified frame.

  // check that we are passed a valid plot frame to use
  if(0 == frame) {
    cout << ClassName() << "::" << GetName() << ":plotOn: frame is null" << endl;
    return 0;
  }

  // check that this frame knows what variable to plot
  RooAbsReal *var= frame->getPlotVar();
  if(0 == var) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: frame does not specify a plot variable" << endl;
    return 0;
  }

  // check that the plot variable is not derived
  RooRealVar* realVar= dynamic_cast<RooRealVar*>(var);
  if(0 == realVar) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: cannot plot derived variable \"" << var->GetName() << "\"" << endl;
    return 0;
  }

  // check if we actually depend on the plot variable
  if(!this->dependsOn(*realVar)) {
    cout << GetName() << "::plotOn:WARNING: variable is not an explicit dependent: "
	 << realVar->GetName() << endl;
  }

  // deep-clone ourselves so that the plotting process will not disturb
  // our original expression tree
  RooArgSet *cloneList = (RooArgSet*) RooArgSet(*this).snapshot() ;
  RooSimultaneous *clone= (RooSimultaneous*) cloneList->find(GetName()) ;

  // redirect our clone to use the plot variable
  RooArgSet plotSet(*realVar);
  clone->recursiveRedirectServers(plotSet);

  // Make a new expression that is the weighted sum of the RooSimultaneous components
  RooArgList pdfCompList ;
  RooArgList wgtCompList ;
  TIterator* pIter = clone->_pdfProxyList.MakeIterator() ;
  RooRealProxy *proxy ;
  Roo1DTable* wTable = wdata->table(clone->_indexCat.arg()) ;
  Int_t n = _pdfProxyList.GetSize() ;
  while(proxy=(RooRealProxy*)pIter->Next()) {
    pdfCompList.add(proxy->arg()) ;
    if (--n) wgtCompList.addOwned(*(new RooRealVar(proxy->name(),
				    "coef",wTable->getFrac(proxy->name())))) ;
  }
  delete pIter ;
  delete wTable ;
  RooAddPdf *plotSumVar = new RooAddPdf("plotSumVar","weighted sum of RS components",pdfCompList,wgtCompList) ;


  // normalize ourself to any previous contents in the frame
  if(frame->getFitRangeNorm() > 0) scaleFactor*= frame->getFitRangeNorm();
  frame->updateNormVars(plotSet);

  // create a new curve of our function using the clone to do the evaluations
  RooCurve* curve= new RooCurve(*plotSumVar,*realVar,scaleFactor,frame->getNormVars());

  // add this new curve to the specified plot frame
  frame->addPlotable(curve, drawOptions);

  // cleanup 
  delete plotSumVar ;
  delete cloneList;

  return frame;
}
