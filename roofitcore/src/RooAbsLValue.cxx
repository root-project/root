/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsLValue.cc,v 1.1 2001/08/23 01:21:45 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   21-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
//
//  Abstract base class for objects that are lvalues, i.e. objects
//  whose value can be modified directly. This class implements
//  abstract methods for binned fits that return the number
//  of fit bins and change the value of the object to the central
//  value of a given fit bin, regardless of the type of value.

#include "RooFitCore/RooAbsLValue.hh"

ClassImp(RooAbsLValue)
;

RooAbsLValue::RooAbsLValue() 
{
}

RooAbsLValue::~RooAbsLValue() 
{
}
