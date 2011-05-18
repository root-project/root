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
// RooDouble is a minimal implementation of a TObject holding a Double_t
// value.
// END_HTML
//

#include "RooFit.h"
#include "RooDouble.h"
#include <string>

ClassImp(RooDouble)
;



//_____________________________________________________________________________
RooDouble::RooDouble(Double_t value) : TNamed(), _value(value) 
{
  SetName(Form("%f",value)) ;
}


//_____________________________________________________________________________
Int_t RooDouble::Compare(const TObject* other) const 
{
  // Implement comparison to other TObjects that are also RooDouble
  // to faciliate sorting of RooDoubles in a ROOT container

  const RooDouble* otherD = dynamic_cast<const RooDouble*>(other) ;
  if (!otherD) return 0 ;
  return (_value>otherD->_value) ? 1 : -1 ;
}
