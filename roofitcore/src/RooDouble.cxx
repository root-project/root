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

#include "RooFitCore/RooDouble.hh"

ClassImp(RooDouble)
;


Int_t RooDouble::Compare(const TObject* other) const 
{
  const RooDouble* otherD = dynamic_cast<const RooDouble*>(other) ;
  if (!other) return 0 ;
  return (_value>otherD->_value) ? 1 : -1 ;
}
