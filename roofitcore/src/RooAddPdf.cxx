/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooAddPdf.cc,v 1.15 2001/09/27 18:22:28 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   06-Jan-2000 DK Created initial version
 *   19-Apr-2000 DK Add the printEventStats() method
 *   26-Jun-2000 DK Add support for extended likelihood fits
 *   02-Jul-2000 DK Add support for multiple terms (instead of only 2)
 *   05-Jul-2000 DK Add support for extended maximum likelihood and a
 *                  new method for this: setNPar()
 *   03-May02001 WV Port to RooFitCore/RooFitModels
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooAddPdf is an efficient implementation of a sum of PDFs of the form 
//
//  (c_1*PDF_1 + c_2*PDF_2 + ... (1-sum(c_1...c_n-1))*PDF_n 
//
// The coefficient of the last PDF is calculated automatically from the
// normalization condition. RooAddPdf relies on each component PDF
// to be normalized and will perform no normalization other than calculating
// the proper last coefficient c_n.
// An additional condition for this is that each coefficient c_k may
// not overlap (i.e. share servers) with pdf_k.

#include "TIterator.h"
#include "TList.h"
#include "RooFitCore/RooAddPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooAddPdf)
;


RooAddPdf::RooAddPdf(const char *name, const char *title) :
  RooAbsPdf(name,title), 
  _coefList("coefList","List of coefficients",this),
  _pdfList("pdfList","List of PDFs",this),
  _codeReg(10),
  _haveLastCoef(kFALSE)
{
  // Dummy constructor 
  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
}


RooAddPdf::RooAddPdf(const char *name, const char *title,
		     RooAbsPdf& pdf1, RooAbsPdf& pdf2, RooAbsReal& coef1) : 
  RooAbsPdf(name,title),
  _coefList("coefList","List of coefficients",this),
  _pdfList("pdfProxyList","List of PDFs",this),
  _codeReg(10),
  _haveLastCoef(kFALSE)
{
  // Constructor with two PDFs

  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;

  _pdfList.add(pdf1) ;
  _pdfList.add(pdf2) ;
  _coefList.add(coef1) ;
}

RooAddPdf::RooAddPdf(const char *name, const char *title, const RooArgList& pdfList, const RooArgList& coefList) :
  RooAbsPdf(name,title),
  _coefList("coefList","List of coefficients",this),
  _pdfList("pdfProxyList","List of PDFs",this),
  _codeReg(10),
  _haveLastCoef(kFALSE)
{ 
  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
 
  // Constructor with N PDFs and N or N-1 coefs
  TIterator* pdfIter = pdfList.createIterator() ;
  TIterator* coefIter = coefList.createIterator() ;
  RooAbsPdf* pdf ;
  RooAbsReal* coef ;
  while(coef = (RooAbsPdf*)coefIter->Next()) {
    pdf = (RooAbsPdf*) pdfIter->Next() ;
    if (!pdf) {
      cout << "RooAddPdf::RooAddPdf(" << GetName() 
	   << ") number of pdfs and coefficients inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1" << endl ;
      assert(0) ;
    }
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      cout << "RooAddPdf::RooAddPdf(" << GetName() << ") coefficient " << coef->GetName() << " is not of type RooAbsReal, ignored" << endl ;
      continue ;
    }
    if (!dynamic_cast<RooAbsReal*>(pdf)) {
      cout << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << coef->GetName() << " is not of type RooAbsPdf, ignored" << endl ;
      continue ;
    }
    _pdfList.add(*pdf) ;
    _coefList.add(*coef) ;    
  }

  pdf = (RooAbsPdf*) pdfIter->Next() ;
  if (pdf) {
    if (!dynamic_cast<RooAbsReal*>(pdf)) {
      cout << "RooAddPdf::RooAddPdf(" << GetName() << ") last pdf " << coef->GetName() << " is not of type RooAbsPdf, fatal error" << endl ;
      assert(0) ;
    }
    _pdfList.add(*pdf) ;  
  } else {
    _haveLastCoef=kTRUE ;
  }

  delete pdfIter ;
  delete coefIter  ;
}



RooAddPdf::RooAddPdf(const RooAddPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _coefList("coefList",this,other._coefList),
  _pdfList("pdfProxyList",this,other._pdfList),
  _codeReg(other._codeReg),
  _haveLastCoef(other._haveLastCoef)
{
  // Copy constructor

  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;

}


RooAddPdf::~RooAddPdf()
{
  // Destructor
  delete _pdfIter ;
  delete _coefIter ;
}



Double_t RooAddPdf::evaluate() const 
{
  // Calculate the current value of this object
  const RooArgSet* nset = _pdfList.nset() ;
  
  Double_t value(0) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* pdf ;

  if (_haveLastCoef) {

    // N pdfs, N coefficients (use extended likelihood)
    Double_t coefSum(0) ;
    while(coef=(RooAbsReal*)_coefIter->Next()) {
      pdf = (RooAbsReal*)_pdfIter->Next() ;
      Double_t coefVal = coef->getVal(nset) ;
      value += pdf->getVal(nset)*coefVal ;
      coefSum += coefVal ;
    }
    value /= coefSum ;    

  } else {

    // N pdfs, N-1 coefficients 
    Double_t lastCoef(1) ;
    while(coef=(RooAbsReal*)_coefIter->Next()) {
      pdf = (RooAbsReal*)_pdfIter->Next() ;
      value += pdf->getVal(nset)*coef->getVal(nset) ;
      lastCoef -= coef->getVal(nset) ;
    }
    
    // Add last pdf with correct coefficient
    pdf = (RooAbsReal*) _pdfIter->Next() ;
    value += pdf->getVal(nset)*lastCoef;

    // Warn about coefficient degeneration
    if (lastCoef<0 || lastCoef>1) {
      cout << "RooAddPdf::evaluate(" << GetName() 
	   << " WARNING: sum of PDF coefficients not in range [0-1], value=" 
	   << 1-lastCoef << endl ;
    } 
  }

  return value ;
}


Bool_t RooAddPdf::checkDependents(const RooArgSet* nset) const 
{
  // Check if PDF is valid with dependent configuration given by specified data set

  // Coeffient and PDF should be non-overlapping, but coef/pdf pairs can overlap each other
  Bool_t ret(kFALSE) ;

  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* pdf ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    pdf = (RooAbsReal*)_pdfIter->Next() ;
    ret |= pdf->checkDependents(nset) ;
    ret |= coef->checkDependents(nset) ;
    if (pdf->dependentOverlaps(nset,*coef)) {
      cout << "RooAddPdf::checkDependents(" << GetName() << "): ERROR: coefficient " << coef->GetName() 
	   << " and PDF " << pdf->GetName() << " have one or more dependents in common" << endl ;
      ret = kTRUE ;
    }
  }
  
  return ret ;
}


Int_t RooAddPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code
    // This PDF is by construction normalized
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  RooArgSet allAnalVars(allVars) ;
  TIterator* avIter = allVars.createIterator() ;

  Int_t n(0) ;
  // First iteration, determine what each component can integrate analytically
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    RooArgSet subAnalVars ;
    Int_t subCode = pdf->getAnalyticalIntegralWN(allVars,subAnalVars,normSet) ;
//     cout << "RooAddPdf::getAI(" << GetName() << ") ITER1 subCode(" << n << "," << pdf->GetName() << ") = " << subCode << endl ;

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
  _pdfIter->Reset() ;
  n=0 ;
  Int_t* subCode = new Int_t[_pdfList.getSize()] ;
  Bool_t allOK(kTRUE) ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    RooArgSet subAnalVars ;
    subCode[n] = pdf->getAnalyticalIntegralWN(allAnalVars,subAnalVars,normSet) ;
//     cout << "RooAddPdf::getAI(" << GetName() << ") ITER2 subCode(" << n << "," << pdf->GetName() << ") = " << subCode[n] << endl ;
    if (subCode[n]==0) {
      cout << "RooAddPdf::getAnalyticalIntegral(" << GetName() << ") WARNING: component PDF " << pdf->GetName() 
	   << "   advertises inconsistent set of integrals (e.g. (X,Y) but not X or Y individually."
	   << "   Distributed analytical integration disabled. Please fix PDF" << endl ;
      allOK = kFALSE ;
    }
    n++ ;
  }  
  if (!allOK) return 0 ;

  analVars.add(allAnalVars) ;
  Int_t masterCode = _codeReg.store(subCode,_pdfList.getSize())+1 ;

  delete[] subCode ;
  delete avIter ;
  return masterCode ;
}


Double_t RooAddPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const 
{
  // Return analytical integral defined by given scenario code
  if (code==0) return getVal() ;

  const Int_t* subCode = _codeReg.retrieve(code-1) ;
  if (!subCode) {
    cout << "RooAddPdf::analyticalIntegral(" << GetName() << "): ERROR unrecognized integration code, " << code << endl ;
    assert(0) ;    
  }

  // Calculate the current value of this object  
  Double_t value(0) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* pdf ;
  Int_t i(0) ;

  if (_haveLastCoef) {

    // N pdfs, N coefficients (use extended likelihood)
    Double_t coefSum(0) ;
    while(coef=(RooAbsReal*)_coefIter->Next()) {
      pdf = (RooAbsReal*)_pdfIter->Next() ;
      Double_t coefVal = coef->getVal(normSet) ;
      value += pdf->analyticalIntegralWN(subCode[i],normSet)*coefVal ;      
      coefSum += coefVal ;
      i++ ;
    }    
    value /= coefSum ;

  } else {

    // N pdfs, N-1 coefficients
    Double_t lastCoef(1) ;
    while(coef=(RooAbsReal*)_coefIter->Next()) {
      pdf = (RooAbsReal*)_pdfIter->Next() ;
      Double_t coefVal = coef->getVal(normSet) ;
      value += pdf->analyticalIntegralWN(subCode[i],normSet)*coefVal ;
      lastCoef -= coefVal ;
      i++ ;
    }

    pdf = (RooAbsReal*) _pdfIter->Next() ;
    value += pdf->analyticalIntegralWN(subCode[i],normSet)*lastCoef ;

    // Warn about coefficient degeneration
    if (lastCoef<0 || lastCoef>1) {
      cout << "RooAddPdf::analyticalIntegral(" << GetName() 
	   << " WARNING: sum of PDF coefficients not in range [0-1], value=" 
	   << 1-lastCoef << endl ;
    }     
  }

  return value ;
}




Double_t RooAddPdf::expectedEvents() const 
{  
  // Calculate the current value of this object
  _coefIter->Reset() ;

  // Do running sum of coef - for expectedTotal
  Double_t expectedTotal(0.0);
  RooAbsReal* coef ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    expectedTotal += coef->getVal() ;
  }   

  return expectedTotal;
}




RooPlot* RooAddPdf::plotCompOn(RooPlot *frame, const RooArgSet& compSet, Option_t* drawOptions,
			       Double_t scaleFactor, ScaleType stype, const RooArgSet* projSet) const 
{
  // Plot selected components 

  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;
  
  // Build a temporary consisting only of the components to be plotted
  TString newName(GetName()) ;
  TString newTitle("Components ") ;
  newName.Append("[") ;

  RooArgList plotPdfList ;
  RooArgList plotCoefList ;
  RooAbsPdf* pdf ;
  RooAbsReal* coef ;
  RooRealVar* lastCoefVar(0) ;
  Double_t lastCoef(1-expectedEvents()) ;
  Double_t coefPartSum(0) ;
  Double_t coefSum(0) ;
  Bool_t first(kTRUE) ;
  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {

    coef = (RooAbsReal*) _coefIter->Next() ;
    if (!coef) {
      lastCoefVar = new RooRealVar("lastCoef","lastCoef",lastCoef) ;
      coef = lastCoefVar ;
    }
    if (compSet.find(pdf->GetName())) {
      coefPartSum += coef->getVal() ;
      plotPdfList.add(*pdf) ;
      plotCoefList.add(*coef) ;

      // Append name of component to name of pdf subset
      if (first) {
	first=kFALSE ;
      } else {
	newName.Append(",") ;
	newTitle.Append(",") ;
      }
      newName.Append(pdf->GetName()) ;
      newTitle.Append(pdf->GetName()) ;
    }
    coefSum += coef->getVal() ;
  } 
  newName.Append("]") ;
  newTitle.Append(" of ") ;
  newTitle.Append(GetTitle()) ;
  
  RooAddPdf* plotVar = new RooAddPdf(newName.Data(),newTitle.Data(),plotPdfList,plotCoefList) ;

  // Plot temporary function
  cout << "RooAddPdf::plotCompOn(" << GetName() << ") plotting components " ; plotPdfList.Print("1") ;
  RooPlot* frame2 = plotVar->plotOn(frame,drawOptions,scaleFactor*coefPartSum/coefSum,stype,projSet) ;

  // Cleanup
  delete plotVar ;
  if (lastCoefVar) delete lastCoefVar ;

  return frame2 ;
}
