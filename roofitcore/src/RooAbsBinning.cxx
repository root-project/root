/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsBinning.cc,v 1.1 2002/03/07 06:22:18 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   01-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

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
