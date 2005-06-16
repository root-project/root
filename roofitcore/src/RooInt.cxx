/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooInt.cc,v 1.7 2005/02/25 14:22:57 wverkerke Exp $
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
// RooInt is a minimal implementation of a TObject holding a Int_t
// value.

#include "RooFitCore/RooFit.hh"

#include "RooFitCore/RooInt.hh"
#include "RooFitCore/RooInt.hh"

ClassImp(RooInt)
;


Int_t RooInt::Compare(const TObject* other) const 
{
  const RooInt* otherD = dynamic_cast<const RooInt*>(other) ;
  if (!other) return 0 ;
  return (_value>otherD->_value) ? 1 : -1 ;
}
