/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenContext.rdl,v 1.5 2001/10/10 00:22:24 david Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   11-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX} --
// RooAbsGenContext is the abstract base class for generator contexts

#include "RooFitCore/RooAbsGenContext.hh"
#include "RooFitCore/RooAbsPdf.hh"

ClassImp(RooAbsGenContext)
;


RooAbsGenContext::RooAbsGenContext(const RooAbsPdf& model, Bool_t verbose) :
  TNamed(model), _verbose(verbose), _isValid(kTRUE) 
{
  // Constructor
}


RooAbsGenContext::~RooAbsGenContext()
{
  // Destructor
}
