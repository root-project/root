/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsLValue.rdl,v 1.8 2004/08/09 00:00:52 bartoldu Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_ABS_LVALUE
#define ROO_ABS_LVALUE

#include <iostream>
#include "Rtypes.h"


class RooAbsLValue {
public:

  // Constructors, cloning and assignment
  RooAbsLValue() ;
  virtual ~RooAbsLValue();

  virtual void setFitBin(Int_t ibin) = 0 ;
  virtual Int_t getFitBin() const = 0 ;
  virtual Int_t numFitBins() const = 0 ;
  virtual Double_t getFitBinWidth(Int_t i) const = 0 ;

  virtual void randomize() = 0 ;

protected:

  ClassDef(RooAbsLValue,1) // Abstract variable
};

#endif
