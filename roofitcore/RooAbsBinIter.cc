/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooAbsBinIter.hh"

ClassImp(RooAbsBinIter) 
;


RooAbsBinIter::RooAbsBinIter(const RooAbsArg& arg) : 
  _arg((RooAbsArg*)&arg), _curBin(0)
{
}

RooAbsBinIter::RooAbsBinIter(const RooAbsBinIter& other) :
  _arg(other._arg), _curBin(other._curBin)
{
}


RooAbsBinIter::~RooAbsBinIter() 
{
}
  
