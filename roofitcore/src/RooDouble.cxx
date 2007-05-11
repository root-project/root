/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooDouble.cc,v 1.9 2005/06/16 09:31:27 wverkerke Exp $
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
// RooDouble is a minimal implementation of a TObject holding a Double_t
// value.

#include "RooFit.h"

#include "RooDouble.h"
#include "RooDouble.h"

ClassImp(RooDouble)
;


Int_t RooDouble::Compare(const TObject* other) const 
{
  const RooDouble* otherD = dynamic_cast<const RooDouble*>(other) ;
  if (!other) return 0 ;
  return (_value>otherD->_value) ? 1 : -1 ;
}
