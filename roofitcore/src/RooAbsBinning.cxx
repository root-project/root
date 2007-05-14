/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooAbsBinning.cxx,v 1.15 2007/05/11 09:11:58 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// RooAbsBinning is the abstract base class for RooRealVar binning definitions
// This class defines the interface to retrieve bin boundaries, ranges etc.

#include "RooFit.h"

#include "RooAbsBinning.h"
#include "RooAbsBinning.h"

ClassImp(RooAbsBinning)
;


RooAbsBinning::RooAbsBinning(const char* name) : TNamed(name,name)
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
    os << "B(" ;

    Int_t i ;
    for (i=0 ; i<n ; i++) {
      if (!first) {
	os << indent << " : " ;
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

void RooAbsBinning::Streamer(TBuffer &R__b)
{
   // Stream an object of class RooAbsBinning.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      if (R__v==1) {
	TObject::Streamer(R__b);
      } else {
	TNamed::Streamer(R__b);
      }
      RooPrintable::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, RooAbsBinning::IsA());
   } else {
      R__c = R__b.WriteVersion(RooAbsBinning::IsA(), kTRUE);
      TNamed::Streamer(R__b);
      RooPrintable::Streamer(R__b);
      R__b.SetByteCount(R__c, kTRUE);
   }
}

