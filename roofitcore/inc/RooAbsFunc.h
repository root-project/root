/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsFunc1D.rdl,v 1.2 2001/05/14 22:54:19 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   02-Aug-2001 DK Created initial version from RooAbsFunc1D
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_ABS_FUNC
#define ROO_ABS_FUNC

#include "Rtypes.h"

class RooAbsFunc {
public:
  inline RooAbsFunc(UInt_t dimension) : _dimension(dimension), _valid(kTRUE) { }
  inline virtual ~RooAbsFunc() { }
  inline UInt_t getDimension() const { return _dimension; }
  inline Bool_t isValid() const { return _valid; }

  virtual Double_t operator()(const Double_t xvector[]) const = 0;
  virtual Double_t getMinLimit(UInt_t dimension) const = 0;
  virtual Double_t getMaxLimit(UInt_t dimension) const = 0;

protected:
  UInt_t _dimension;
  Bool_t _valid;
  ClassDef(RooAbsFunc,0) // Abstract real-valued function interface
};

#endif

