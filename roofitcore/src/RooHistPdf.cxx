/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   26-Sep-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooHistPdf.hh"
#include "RooFitCore/RooDataHist.hh"

ClassImp(RooHistPdf)
;


RooHistPdf::RooHistPdf(const char *name, const char *title, const RooArgSet& vars, const RooDataHist& dhist) :
  RooAbsPdf(name,title), 
  _dataHist((RooDataHist*)&dhist), 
  _depList("depList","List of dependents",this),
  _codeReg(10)
{
  // Constructor
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
  _codeReg(other._codeReg)
{
  // Copy constructor
}


Double_t RooHistPdf::evaluate() const
{
  return _dataHist->weight(_depList) ;
}


Int_t RooHistPdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  // Simplest scenario, integrate over all dependents
  if ((allVars.getSize()==_depList.getSize()) && 
      matchArgs(allVars,analVars,_depList)) return 1000 ;
  
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
  analVars.add(allVars) ;

  // Register bit pattern and store with associated argset of variable to be integrated
  return _codeReg.store(&code,1,new RooArgSet(allVars))+1 ;
}



Double_t RooHistPdf::analyticalIntegral(Int_t code) const 
{
  // Simplest scenario, integration over all dependents
  if (code==1000) return _dataHist->sum(kTRUE) ;

  // Partial integration scenario, retrieve set of variables, calculate partial sum
  RooArgSet* intSet(0) ;
  _codeReg.retrieve(code-1,intSet) ;
  
  return _dataHist->sum(*intSet,_depList,kTRUE) ;
}


