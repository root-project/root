/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_CONSTRAINT_SUM
#define ROO_CONSTRAINT_SUM

#include "RooAbsReal.h"
#include "RooListProxy.h"
#include "RooSetProxy.h"
#include "TStopwatch.h"

class RooRealVar;
class RooArgList ;

class RooConstraintSum : public RooAbsReal {
public:

  RooConstraintSum() ;
  RooConstraintSum(const char *name, const char *title, const RooArgSet& constraintSet, const RooArgSet& paramSet) ;
  virtual ~RooConstraintSum() ;

  RooConstraintSum(const RooConstraintSum& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooConstraintSum(*this, newname); }

  const RooArgList& list() { return _set1 ; }

protected:

  RooListProxy _set1 ;    // Set of constraint terms
  RooSetProxy _paramSet ; // Set of parameters to which constraints apply

  Double_t evaluate() const;

  ClassDef(RooConstraintSum,2) // sum of -log of set of RooAbsPdf representing parameter constraints
};

#endif
