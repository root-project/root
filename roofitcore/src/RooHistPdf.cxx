/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooHistPdf.cc,v 1.7 2002/04/03 23:37:25 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   26-Sep-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// RooHistPdf implements a probablity density function sample from a 
// multidimensional histogram. The histogram distribution is explicitly
// normalized by RooHistPdf and can have an arbitrary number of real or 
// discrete dimensions.

#include "RooFitCore/RooHistPdf.hh"
#include "RooFitCore/RooDataHist.hh"

ClassImp(RooHistPdf)
;


RooHistPdf::RooHistPdf(const char *name, const char *title, const RooArgSet& vars, 
		       const RooDataHist& dhist, Int_t intOrder) :
  RooAbsPdf(name,title), 
  _dataHist((RooDataHist*)&dhist), 
  _depList("depList","List of dependents",this),
  _codeReg(10),
  _intOrder(intOrder)
{
  // Constructor from a RooDataHist. The variable listed in 'vars' control the dimensionality of the
  // PDF. Any additional dimensions present in 'dhist' will be projected out. RooDataHist dimensions
  // can be either real or discrete. See RooDataHist::RooDataHist for details on the binning.
  // RooHistPdf neither owns or clone 'dhist' and the user must ensure the input histogram exists
  // for the entire life span of this PDF.

  _depList.add(vars) ;

  // Verify that vars and dhist.get() have identical contents
  const RooArgSet* dvars = dhist.get() ;
  if (vars.getSize()!=dvars->getSize()) {
    cout << "RooHistPdf::ctor(" << GetName() 
	 << ") ERROR variable list and RooDataHist must contain the same variables." << endl ;
    assert(0) ;
  }
  TIterator* iter = vars.createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
    if (!dvars->find(arg->GetName())) {
      cout << "RooHistPdf::ctor(" << GetName() 
	   << ") ERROR variable list and RooDataHist must contain the same variables." << endl ;
      assert(0) ;
    }
  }
}


RooHistPdf::RooHistPdf(const RooHistPdf& other, const char* name) :
  RooAbsPdf(other,name), 
  _dataHist(other._dataHist),
  _depList("depList",this,other._depList),
  _codeReg(other._codeReg),
  _intOrder(other._intOrder)
{
  // Copy constructor
}


Double_t RooHistPdf::evaluate() const
{
  // Return the current value: The value of the bin enclosing the current coordinates
  // of the dependents, normalized by the histograms contents
  return _dataHist->weight(_depList,_intOrder,kTRUE) ;
}


Int_t RooHistPdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  // Determine integration scenario. RooHistPdf can perform all integrals over 
  // its dependents analytically via partial or complete summation of the input histogram.

  // Simplest scenario, integrate over all dependents
  if ((allVars.getSize()==_depList.getSize()) && 
      matchArgs(allVars,analVars,_depList)) return 1000 ;

  // Find subset of _depList that integration is requested over
  RooArgSet* allVarsSel = (RooArgSet*) allVars.selectCommon(_depList) ;
  if (allVarsSel->getSize()==0) {
    delete allVarsSel ;
    return 0 ;
  }

  // Partial integration scenarios.
  // Build unique code from bit mask of integrated variables in depList
  Int_t code(0),n(0) ;
  TIterator* iter = _depList.createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
    if (allVars.find(arg->GetName())) code |= (1<<n) ;
    n++ ;
  }
  delete iter ;
  analVars.add(*allVarsSel) ;

  // Register bit pattern and store with associated argset of variable to be integrated
  Int_t masterCode =  _codeReg.store(&code,1,new RooArgSet(*allVarsSel))+1 ;
  delete allVarsSel ;
  return masterCode ;
}



Double_t RooHistPdf::analyticalIntegral(Int_t code) const 
{
  // Return integral identified by 'code'. The actual integration
  // is deferred to RooDataHist::sum() which implements partial
  // or complete summation over the histograms contents

  // Simplest scenario, integration over all dependents
  if (code==1000) return _dataHist->sum(kFALSE) ;

  // Partial integration scenario, retrieve set of variables, calculate partial sum
  RooArgSet* intSet = 0;
  _codeReg.retrieve(code-1,intSet) ;
  
  return _dataHist->sum(*intSet,_depList,kFALSE) ;
}


