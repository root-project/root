/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// Lightweight interface adaptor that binds an analytic integral of a
// RooAbsReal object (specified by a code) to a set of dependent variables.

#include "RooFitCore/RooNLLBinding.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooAbsPdf.hh"

#include <assert.h>

ClassImp(RooNLLBinding)
;


RooNLLBinding::RooNLLBinding(const RooAbsPdf &pdf, const RooAbsData& data, const RooArgSet &vars) :
  RooRealBinding(pdf,vars,0), _context(&data,&pdf,kFALSE,kFALSE)
{  
}



Double_t RooNLLBinding::operator()(const Double_t xvector[]) const {
  // Evaluate our analytic integral at the specified values of the dependents.

  assert(isValid());
  loadValues(xvector);    
  return _context.nLogLikelihood() ;
}
