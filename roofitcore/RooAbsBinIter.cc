/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsBinIter.cc,v 1.2 2001/08/23 01:21:44 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// RooAbsBinIter is the abstract base class for iterators of bins
// of a RooAbsArg used in binned fits


#include "RooFitCore/RooAbsBinIter.hh"

ClassImp(RooAbsBinIter) 
;


RooAbsBinIter::RooAbsBinIter(const RooAbsArg& arg) : 
  _arg((RooAbsArg*)&arg), _curBin(0)
{
  // Constructor
}

RooAbsBinIter::RooAbsBinIter(const RooAbsBinIter& other) :
  _arg(other._arg), _curBin(other._curBin)
{
  // Copy constructor
}


RooAbsBinIter::~RooAbsBinIter() 
{
  // Destructor
}
  
