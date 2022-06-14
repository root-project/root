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
#ifndef ROO_TFOAM_BINDING
#define ROO_TFOAM_BINDING

#include "TFoamIntegrand.h"
#include "RooArgSet.h"
#include "RooRealBinding.h"
class RooAbsPdf ;

class RooTFoamBinding : public TFoamIntegrand {
public:
  RooTFoamBinding(const RooAbsReal& pdf, const RooArgSet& observables) ;
  ~RooTFoamBinding() override;

  double Density(Int_t ndim, double *) override ;

  RooRealBinding& binding() { return *_binding ; }

protected:

  RooArgSet       _nset ;
  RooRealBinding* _binding ;

  ClassDefOverride(RooTFoamBinding,0) // Function binding to RooAbsReal object
};

#endif

