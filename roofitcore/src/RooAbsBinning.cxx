/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsBinning.cc,v 1.2 2002/04/03 23:37:22 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   01-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// RooAbsBinning is the abstract base class for RooRealVar binning definitions
// This class defines the interface to retrieve bin boundaries, ranges etc.

#include "RooFitCore/RooAbsBinning.hh"

ClassImp(RooAbsBinning)
;


RooAbsBinning::RooAbsBinning() 
{
}


RooAbsBinning::~RooAbsBinning() 
{
}


void RooAbsBinning::printToStream(ostream &os, PrintOption opt, TString indent) const
{
  if (opt==Standard) {

    Bool_t first(kTRUE) ;
    Int_t n = numBins() ;
    os << "VB(" ;

    Int_t i ;
    for (i=0 ; i<n ; i++) {
      if (!first) {
	os << " : " ;
      } else {
	first = kFALSE ;
      }
      os << binLow(i) ;
    }
    os << " : " << binHigh(n-1) ;
    os << ")" << endl ;
    return ;
  }
}
