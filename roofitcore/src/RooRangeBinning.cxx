/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRangeBinning.cc,v 1.4 2005/06/20 15:44:56 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// RooRangeBinning is a single bin binning used to indicate alternative
// ranges for integration etc...

#include "RooFit.h"

#include "RooNumber.h"
#include "RooNumber.h"
#include "Riostream.h"


#include "RooRangeBinning.h"

ClassImp(RooRangeBinning) 
;


RooRangeBinning::RooRangeBinning(const char* name) :
  RooAbsBinning(name)
{
  // Constructor
  _range[0] = -RooNumber::infinity ;
  _range[1] = +RooNumber::infinity ;

}

RooRangeBinning::RooRangeBinning(Double_t xmin, Double_t xmax, const char* name) :
  RooAbsBinning(name)
{
  // Constructor
  _range[0] = xmin ;
  _range[1] = xmax ;
}



RooRangeBinning::RooRangeBinning(const RooRangeBinning& other, const char* name) :
  RooAbsBinning(name)
{
  // Copy constructor
  _range[0] = other._range[0] ;
  _range[1] = other._range[1] ;
}



RooRangeBinning::~RooRangeBinning() 
{
  // Destructor 
}


void RooRangeBinning::setRange(Double_t xlo, Double_t xhi) 
{
  // Change limits
  if (xlo>xhi) {
    cout << "RooRangeBinning::setRange: ERROR low bound > high bound" << endl ;
    return ;
  }

  _range[0] = xlo ;
  _range[1] = xhi ;
}
