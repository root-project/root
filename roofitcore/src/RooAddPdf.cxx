/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooAddPdf.cc,v 1.23 2001/10/22 07:12:12 verkerke Exp $
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

// -- CLASS DESCRIPTION [PDF] --
// RooAddPdf is an efficient implementation of a sum of PDFs of the form 
//
//  c_1*PDF_1 + c_2*PDF_2 + ... c_n*PDF_n 
//
// or 
//
//  c_1*PDF_1 + c_2*PDF_2 + ... (1-sum(c_1...c_n-1))*PDF_n 
//
// The first form is for extended likelihood fits, where the
// expected number of events is Sum(i) c_i. The coefficients c_i
// can either be explicitly provided, or, if all components support
// extended likelihood fits, they can be calculated the contribution
// of each PDF to the total number of expected events.
//
// In the second form, the sum of the coefficients is enforced to be one,
// and the coefficient of the last PDF is calculated from that condition.
//
// RooAddPdf relies on each component PDF to be normalized and will perform 
// no normalization other than calculating the proper last coefficient c_n, if requested.
// An (enforced) condition for this assuption is that each PDF_i is independent
// of each coefficient_i.
//
// 

#include "TIterator.h"
#include "TList.h"
#include "RooFitCore/RooAddPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooAddGenContext.hh"

ClassImp(RooAddPdf)
;


RooAddPdf::RooAddPdf(const char *name, const char *title) :
  RooAbsPdf(name,title), 
  _coefList("coefList","List of coefficients",this),
  _pdfList("pdfList","List of PDFs",this),
  _codeReg(10),
  _haveLastCoef(kFALSE),
  _allExtendable(kTRUE)
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
  _haveLastCoef(kFALSE),
  _allExtendable(kFALSE)
{
  // Special constructor with two PDFs and one coefficient (most frequent use case)

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
  _haveLastCoef(kFALSE),
  _allExtendable(kFALSE)
{ 
  // Generic constructor from list of PDFs and list of coefficients.
  // Each pdf list element (i) is paired with coefficient list element (i).
  // The number of coefficients must be either equal to the number of PDFs,
  // in which case extended MLL fitting is enabled, or be one less.
  //
  // All PDFs must inherit from RooAbsPdf. All coefficients must inherit from RooAbsReal

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
      cout << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << pdf->GetName() << " is not of type RooAbsPdf, ignored" << endl ;
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





RooAddPdf::RooAddPdf(const char *name, const char *title, const RooArgList& pdfList) :
  RooAbsPdf(name,title),
  _coefList("coefList","List of coefficients",this),
  _pdfList("pdfProxyList","List of PDFs",this),
  _codeReg(10),
  _haveLastCoef(kFALSE),
  _allExtendable(kTRUE)
{ 
  // Generic constructor from list of extended PDFs. There are no coefficients as the expected
  // number of events from each components determine the relative weight of the PDFs.
  // 
  // All PDFs must inherit from RooAbsPdf. 

  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
 
  // Constructor with N PDFs 
  TIterator* pdfIter = pdfList.createIterator() ;
  RooAbsPdf* pdf ;
  while(pdf = (RooAbsPdf*) pdfIter->Next()) {
    
    if (!dynamic_cast<RooAbsReal*>(pdf)) {
      cout << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << pdf->GetName() << " is not of type RooAbsPdf, ignored" << endl ;
      continue ;
    }
    if (!pdf->canBeExtended()) {
      cout << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << pdf->GetName() << " is not extendable, ignored" << endl ;
      continue ;
    }
    _pdfList.add(*pdf) ;    
  }

  delete pdfIter ;
}



RooAddPdf::RooAddPdf(const RooAddPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _coefList("coefList",this,other._coefList),
  _pdfList("pdfProxyList",this,other._pdfList),
  _codeReg(other._codeReg),
  _haveLastCoef(other._haveLastCoef),
  _allExtendable(other._allExtendable)
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
  // Calculate the current value

  const RooArgSet* nset = _pdfList.nset() ;

  Double_t value(0) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsPdf* pdf ;

  if (_allExtendable) {

    Double_t totExpected(0) ;
    
    // N pdfs, no coefficients
    while(pdf = (RooAbsPdf*)_pdfIter->Next()) {
      Double_t nExpected = pdf->expectedEvents() ;
      if (nExpected) {
	value += pdf->getVal(nset)*nExpected ; 
	totExpected += nExpected ;
      }
    }	    
    if (totExpected==0.) {
      cout << "RooAddPdf::evaluate(" << GetName() << ") WARNING: total number of expected events is 0" << endl ;
    } else {
      value /= totExpected ;
    }

  } else {
    if (_haveLastCoef) {
      
      // N pdfs, N coefficients (use extended likelihood)
      Double_t coefSum(0) ;
      while(coef=(RooAbsReal*)_coefIter->Next()) {
	pdf = (RooAbsPdf*)_pdfIter->Next() ;
	Double_t coefVal = coef->getVal(nset) ;
	if (coefVal) {
	  value += pdf->getVal(nset)*coefVal ;
	  coefSum += coefVal ;
	}
      }
      value /= coefSum ;    
      
    } else {
      
      // N pdfs, N-1 coefficients 
      Double_t lastCoef(1) ;
      while(coef=(RooAbsReal*)_coefIter->Next()) {
	pdf = (RooAbsPdf*)_pdfIter->Next() ;
	Double_t coefVal = coef->getVal(nset) ;
	if (coefVal) {
	  value += pdf->getVal(nset)*coefVal ;
	  lastCoef -= coef->getVal(nset) ;
	}
      }
      
      // Add last pdf with correct coefficient
      pdf = (RooAbsPdf*) _pdfIter->Next() ;
      value += pdf->getVal(nset)*lastCoef;
      
      // Warn about coefficient degeneration
      if (lastCoef<0 || lastCoef>1) {
	cout << "RooAddPdf::evaluate(" << GetName() 
	     << " WARNING: sum of PDF coefficients not in range [0-1], value=" 
	     << 1-lastCoef << endl ;
      } 
    }
  }

  return value ;
}


Bool_t RooAddPdf::checkDependents(const RooArgSet* nset) const 
{
  // Check if PDF is valid for given normalization set.
  // Coeffient and PDF must be non-overlapping, but pdf-coefficient 
  // pairs may overlap each other

  Bool_t ret(kFALSE) ;

  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* pdf ;
  while(coef=(RooAbsReal*)_coefIter->Next()) {
    pdf = (RooAbsReal*)_pdfIter->Next() ;
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
  //
  // RooAddPdf queries each component PDF for its analytical integration capability of the requested
  // set ('allVars'). It finds the largest common set of variables that can be integrated
  // by all components. If such a set exists, it reconfirms that each component is capable of
  // analytically integrating the common set, and combines the components individual integration
  // codes into a single integration code valid for RooAddPdf.

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

  if (code==0) return getVal(normSet) ;

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
  RooAbsPdf* pdf ;
  Int_t i(0) ;

  if (_allExtendable) {
    
    Double_t totExpected(0) ;
    
    // N pdfs, no coefficients
    while(pdf = (RooAbsPdf*)_pdfIter->Next()) {
      Double_t nExpected = pdf->expectedEvents() ;
      if (nExpected) {
	value += pdf->analyticalIntegralWN(subCode[i],normSet)*nExpected ;
	totExpected += nExpected ; 
      }
    }	    
    if (totExpected==0.) {
      cout << "RooAddPdf::analyticalIntegral(" << GetName() << ") WARNING: total number of expected events is 0" << endl ;
    } else {
      value /= totExpected ;
    }

  } else {
    if (_haveLastCoef) {
      
      // N pdfs, N coefficients (use extended likelihood)
      Double_t coefSum(0) ;
      while(coef=(RooAbsReal*)_coefIter->Next()) {
	pdf = (RooAbsPdf*)_pdfIter->Next() ;
	Double_t coefVal = coef->getVal(normSet) ;
	if (coefVal) {
	  value += pdf->analyticalIntegralWN(subCode[i],normSet)*coefVal ;      
	  coefSum += coefVal ;
	}
	i++ ;
      }    
      value /= coefSum ;
      
    } else {
      
      // N pdfs, N-1 coefficients
      Double_t lastCoef(1) ;
      while(coef=(RooAbsReal*)_coefIter->Next()) {
	pdf = (RooAbsPdf*)_pdfIter->Next() ;
	Double_t coefVal = coef->getVal(normSet) ;
	if (coefVal) {
	  value += pdf->analyticalIntegralWN(subCode[i],normSet)*coefVal ;
	  lastCoef -= coefVal ;
	}
	i++ ;
      }
      
      pdf = (RooAbsPdf*) _pdfIter->Next() ;
      value += pdf->analyticalIntegralWN(subCode[i],normSet)*lastCoef ;
      
      // Warn about coefficient degeneration
      if (lastCoef<0 || lastCoef>1) {
	cout << "RooAddPdf::analyticalIntegral(" << GetName() 
	     << " WARNING: sum of PDF coefficients not in range [0-1], value=" 
	     << 1-lastCoef << endl ;
      }     
    }
  }

  return value ;
}




Double_t RooAddPdf::expectedEvents() const 
{  
  // Return the number of expected events, which is either the sum of all coefficients
  // or the sum of the components extended terms

  Double_t expectedTotal(0.0);
  RooAbsReal* coef ;
  RooAbsPdf* pdf ;
    
  if (_allExtendable) {
    
    // Sum of the extended terms
    _pdfIter->Reset() ;
    while(pdf = (RooAbsPdf*)_pdfIter->Next()) {      
      expectedTotal += pdf->expectedEvents() ;
    }   
    
  } else {
    
    // Sum the coefficients
    _coefIter->Reset() ;
    RooAbsReal* coef ;
    while(coef=(RooAbsReal*)_coefIter->Next()) {
      expectedTotal += coef->getVal() ;
    }   
  }

  return expectedTotal;
}




RooPlot* RooAddPdf::plotCompOn(RooPlot *frame, const RooArgSet& compSet, Option_t* drawOptions,
			       Double_t scaleFactor, ScaleType stype, const RooArgSet* projSet) const 
{
  // Plot only the PDF components listed in 'compSet' of this PDF on 'frame'. 
  // See RooAbsReal::plotOn() for a description of the remaining arguments and other features

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
  RooPlot* frame2 = plotVar->plotOn(frame,drawOptions,scaleFactor*coefPartSum/coefSum,stype,0,projSet) ;

  // Cleanup
  delete plotVar ;
  if (lastCoefVar) delete lastCoefVar ;

  return frame2 ;
}


RooAbsGenContext* RooAddPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype, Bool_t verbose) const 
{
  return new RooAddGenContext(*this,vars,prototype,verbose) ;
}

