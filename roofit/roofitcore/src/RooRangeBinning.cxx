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
// RooRangeBinning is binning/range definition that only defines a range
// but no binning. It it used to store named ranges created by
// the RooRealVar::setRange() method
// END_HTML
//

#include "RooFit.h"

#include "RooNumber.h"
#include "RooNumber.h"
#include "RooMsgService.h"
#include "Riostream.h"
#include "RooMsgService.h"

#include "RooRangeBinning.h"

ClassImp(RooRangeBinning) 
;



//_____________________________________________________________________________
RooRangeBinning::RooRangeBinning(const char* name) :
  RooAbsBinning(name)
{
  // Default constructor
  _range[0] = -RooNumber::infinity() ;
  _range[1] = +RooNumber::infinity() ;

}


//_____________________________________________________________________________
RooRangeBinning::RooRangeBinning(Double_t xmin, Double_t xmax, const char* name) :
  RooAbsBinning(name)
{
  // Construct binning with range [xmin,xmax] with no binning substructure
  _range[0] = xmin ;
  _range[1] = xmax ;
}



//_____________________________________________________________________________
RooRangeBinning::RooRangeBinning(const RooRangeBinning& other, const char* name) :
  RooAbsBinning(name)
{
  // Copy constructor

  _range[0] = other._range[0] ;
  _range[1] = other._range[1] ;
}



//_____________________________________________________________________________
RooRangeBinning::~RooRangeBinning() 
{
  // Destructor 
}



//_____________________________________________________________________________
void RooRangeBinning::setRange(Double_t xlo, Double_t xhi) 
{
  // Change limits of the binning to [xlo,xhi]

  if (xlo>xhi) {
    oocoutE((TObject*)0,InputArguments) << "RooRangeBinning::setRange: ERROR low bound > high bound" << endl ;
    return ;
  }

  _range[0] = xlo ;
  _range[1] = xhi ;
}
