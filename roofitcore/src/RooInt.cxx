/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   01-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// RooInt is a minimal implementation of a TObject holding a Int_t
// value.

#include "RooFitCore/RooInt.hh"

ClassImp(RooInt)
;


Int_t RooInt::Compare(const TObject* other) const 
{
  const RooInt* otherD = dynamic_cast<const RooInt*>(other) ;
  if (!other) return 0 ;
  return (_value>otherD->_value) ? 1 : -1 ;
}
