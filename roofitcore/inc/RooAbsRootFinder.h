/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsIntegrator.rdl,v 1.9 2001/08/24 23:55:15 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   14-Nov-2001 DK Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_ROOT_FINDER
#define ROO_ABS_ROOT_FINDER

#include "Rtypes.h"

class RooAbsFunc;

class RooAbsRootFinder {
public:
  RooAbsRootFinder(const RooAbsFunc& function);
  inline virtual ~RooAbsRootFinder() { }

  virtual Bool_t findRoot(Double_t &result, Double_t xlo, Double_t xhi, Double_t value= 0) const = 0;

protected:
  const RooAbsFunc *_function;
  Bool_t _valid;

  ClassDef(RooAbsRootFinder,0) // Abstract interface for 1-dim real-valued function root finders
};

#endif
