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
// This class provides a static interface for generating random numbers.
// By default a private copy of TRandom3 is used to generate all random numbers.
// END_HTML
//

#include "RooFit.h"

#include "RooRandom.h"
#include "RooRandom.h"
#include "RooQuasiRandomGenerator.h"

#include "TRandom3.h"

#include <assert.h>

using namespace std;

ClassImp(RooRandom)
  ;



//_____________________________________________________________________________
TRandom *RooRandom::randomGenerator() 
{
  // Return a pointer to a singleton random-number generator
  // implementation. Creates the object the first time it is called.
  
  static TRandom *_theGenerator= 0;
  if(0 == _theGenerator) _theGenerator= new TRandom3();
  return _theGenerator;
}


//_____________________________________________________________________________
RooQuasiRandomGenerator *RooRandom::quasiGenerator() 
{
  // Return a pointer to a singleton quasi-random generator
  // implementation. Creates the object the first time it is called.
  
  static RooQuasiRandomGenerator *_theGenerator= 0;
  if(0 == _theGenerator) _theGenerator= new RooQuasiRandomGenerator();
  return _theGenerator;
}


//_____________________________________________________________________________
Double_t RooRandom::uniform(TRandom *generator) 
{
  // Return a number uniformly distributed from (0,1)

  return generator->Rndm();
}


//_____________________________________________________________________________
void RooRandom::uniform(UInt_t dimension, Double_t vector[], TRandom *generator) 
{
  // Fill the vector provided with random numbers uniformly distributed from (0,1)
  
  for(UInt_t index= 0; index < dimension; index++) vector[index]= uniform(generator);
}


//_____________________________________________________________________________
UInt_t RooRandom::integer(UInt_t n, TRandom *generator) 
{
  // Return an integer uniformly distributed from [0,n-1]

  return generator->Integer(n);
}


//_____________________________________________________________________________
Double_t RooRandom::gaussian(TRandom *generator) 
{
  // Return a Gaussian random variable with mean 0 and variance 1.

  return generator->Gaus();
}


//_____________________________________________________________________________
Bool_t RooRandom::quasi(UInt_t dimension, Double_t vector[], RooQuasiRandomGenerator *generator) 
{
  // Return a quasi-random number in the range (0,1) using the
  // Niederreiter base 2 generator described in Bratley, Fox, Niederreiter,
  // ACM Trans. Model. Comp. Sim. 2, 195 (1992).

  return generator->generate(dimension,vector);
}
