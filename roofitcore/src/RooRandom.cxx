/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenContext.cc,v 1.9 2001/08/09 01:02:14 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   20-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// This class provides a static interface for generating random numbers.

#include "RooFitCore/RooRandom.hh"
#include "RooFitCore/RooQuasiRandomGenerator.hh"

#include "TRandom3.h"

#include <assert.h>

ClassImp(RooRandom)
  ;

static const char rcsid[] =
"$Id: RooGenContext.cc,v 1.9 2001/08/09 01:02:14 verkerke Exp $";

TRandom *RooRandom::randomGenerator() {
  // Return a pointer to a singleton random-number generator
  // implementation. Creates the object the first time it is called.

  static TRandom *_theGenerator= 0;
  if(0 == _theGenerator) _theGenerator= new TRandom3();
  return _theGenerator;
}

RooQuasiRandomGenerator *RooRandom::quasiGenerator() {
  // Return a pointer to a singleton quasi-random generator
  // implementation. Creates the object the first time it is called.

  static RooQuasiRandomGenerator *_theGenerator= 0;
  if(0 == _theGenerator) _theGenerator= new RooQuasiRandomGenerator();
  return _theGenerator;
}

Double_t RooRandom::uniform(TRandom *generator) {
  // Return a number uniformly distributed from (0,1)

  return generator->Rndm();
}

void RooRandom::uniform(UInt_t dimension, Double_t vector[], TRandom *generator) {
  // Fill the vector provided with random numbers uniformly distributed from (0,1)

  for(UInt_t index= 0; index < dimension; index++) vector[index]= uniform(generator);
}

UInt_t RooRandom::integer(UInt_t n, TRandom *generator) {
  // Return an integer uniformly distributed from [0,n-1]

  return generator->Integer(n);
}

Bool_t RooRandom::quasi(UInt_t dimension, Double_t vector[], RooQuasiRandomGenerator *generator) {
  // Return a quasi-random number in the range (0,1) using the
  // Niederreiter base 2 generator described in Bratley, Fox, Niederreiter,
  // ACM Trans. Model. Comp. Sim. 2, 195 (1992).

  return generator->generate(dimension,vector);
}
