/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
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
