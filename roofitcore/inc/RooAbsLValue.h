/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsLValue.rdl,v 1.1 2001/08/23 01:21:45 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   21-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_LVALUE
#define ROO_ABS_LVALUE

#include <iostream.h>
#include "Rtypes.h"
class RooAbsBinIter ;


class RooAbsLValue {
public:

  // Constructors, cloning and assignment
  RooAbsLValue() ;
  virtual ~RooAbsLValue();

  virtual void setFitBin(Int_t ibin) = 0 ;
  virtual Int_t getFitBin() const = 0 ;
  virtual Int_t numFitBins() const = 0 ;
  virtual Double_t getFitBinWidth() const = 0 ;
  virtual RooAbsBinIter* createFitBinIterator() const = 0 ;

protected:

  ClassDef(RooAbsLValue,1) // Abstract variable
};

#endif
