/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooNLLBinding.cc,v 1.5 2002/04/03 23:37:26 verkerke Exp $
 * Authors:
 *   DK, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// Lightweight interface adaptor that binds an analytic integral of a
// RooAbsReal object (specified by a code) to a set of dependent variables.

#include "RooFitCore/RooNLLBinding.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooAbsPdf.hh"

#include <assert.h>

ClassImp(RooNLLBinding)
;


RooNLLBinding::RooNLLBinding(const RooAbsPdf &pdf, const RooAbsData& data, const RooArgSet &vars, Bool_t extended) :
  RooRealBinding(pdf,vars,0), _extended(extended)
{  
  // Constructor
  _context = pdf.fitContext(data) ;
}



RooNLLBinding::~RooNLLBinding()
{
  // Destructor
  delete _context ;
}



Double_t RooNLLBinding::operator()(const Double_t xvector[]) const 
{
  assert(isValid());
  loadValues(xvector);      
  Double_t NLL =  _context->nLogLikelihood(_extended) ;
  cout << "RooNLLBinding::operator() " << (_extended?"extended ":"") << "NLL(" ;
  UInt_t i ;  
  for (i=0 ; i<_dimension ; i++) {
    if (i!=0) cout << ", " ;
    cout << _vars[i]->GetName() << " = " << _vars[i]->getVal() ;
  }
  cout << ") = " << NLL << endl ;
  return NLL ;
}
