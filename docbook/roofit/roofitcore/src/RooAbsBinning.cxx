/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooAbsBinning is the abstract base class for RooRealVar binning definitions
// This class defines the interface to retrieve bin boundaries, ranges etc.
// END_HTML
//
//

#include "RooFit.h"

#include "RooAbsBinning.h"
#include "RooAbsReal.h"
#include "TClass.h"

#include "Riostream.h"

ClassImp(RooAbsBinning)
;


//_____________________________________________________________________________
RooAbsBinning::RooAbsBinning(const char* name) : TNamed(name,name)
{
  // Constructor
}



//_____________________________________________________________________________
RooAbsBinning::~RooAbsBinning() 
{
  // Destructor
}



//_____________________________________________________________________________
void RooAbsBinning::printName(ostream& os) const 
{
  // Print binning name

  os << GetName() ;
}



//_____________________________________________________________________________
void RooAbsBinning::printTitle(ostream& os) const 
{
  // Print binning title

  os << GetTitle() ;
}



//_____________________________________________________________________________
void RooAbsBinning::printClassName(ostream& os) const 
{
  // Print binning class name

  os << IsA()->GetName() ;
}



//_____________________________________________________________________________
void RooAbsBinning::printArgs(ostream& os) const 
{
  // Print binning arguments (the RooAbsReal objects represening
  // the variable bin boundaries for parameterized binning implementations

  os << "[ " ;    
  if (lowBoundFunc()) {
    os << "lowerBound=" << lowBoundFunc()->GetName() ;
  }
  if (highBoundFunc()) {
    if (lowBoundFunc()) {
      os << " " ;
    }
    os << "upperBound=" << highBoundFunc()->GetName() ;
  }
  os << " ]" ;  
}



//_____________________________________________________________________________
void RooAbsBinning::printValue(ostream &os) const
{
  // Print binning value, i.e the bin boundary positions
  Int_t n = numBins() ;
  os << "B(" ;
  
  Int_t i ;
  for (i=0 ; i<n ; i++) {
    if (i>0) {
      os << " : " ;
    }
    os << binLow(i) ;
  }
  os << " : " << binHigh(n-1) ;
  os << ")" ;

}



//_____________________________________________________________________________
void RooAbsBinning::Streamer(TBuffer &R__b)
{
  // Custom streamer implementing schema evolution between V1 and V2 persistent binnings

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

