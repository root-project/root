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
#ifndef ROO_BRENT_ROOT_FINDER
#define ROO_BRENT_ROOT_FINDER

#include "RooFitCore/RooAbsRootFinder.hh"

class RooBrentRootFinder : public RooAbsRootFinder {
public:
  RooBrentRootFinder(const RooAbsFunc& function);
  inline virtual ~RooBrentRootFinder() { }

  virtual Bool_t findRoot(Double_t &result, Double_t xlo, Double_t xhi, Double_t value= 0) const;

protected:
  enum { MaxIterations = 100 };

  ClassDef(RooBrentRootFinder,0) // Abstract interface for 1-dim real-valued function root finders
};

#endif
