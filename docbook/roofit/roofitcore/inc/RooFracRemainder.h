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
#ifndef ROO_FRAC_REMAINDER
#define ROO_FRAC_REMAINDER

#include "RooAbsReal.h"
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class RooFracRemainder : public RooAbsReal {
public:

  RooFracRemainder() ;
  RooFracRemainder(const char *name, const char *title, const RooArgSet& sumSet) ;
  virtual ~RooFracRemainder() ;

  RooFracRemainder(const RooFracRemainder& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooFracRemainder(*this, newname); }

protected:

  RooListProxy _set1 ;            // Set of input fractions
  mutable TIterator* _setIter1 ;  //! Iterator over set of input fractions

  Double_t evaluate() const;

  ClassDef(RooFracRemainder,1) // Utility function calculating remainder fraction, i.e. 1-sum_i(a_i)
};

#endif
