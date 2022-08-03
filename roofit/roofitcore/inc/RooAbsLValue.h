/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsLValue.h,v 1.12 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ABS_LVALUE
#define ROO_ABS_LVALUE

#include <list>
#include <string>

#include "Rtypes.h"

class RooAbsBinning ;

class RooAbsLValue {
public:

  // Constructors, cloning and assignment
  RooAbsLValue() ;
  virtual ~RooAbsLValue();

  virtual void setBin(Int_t ibin, const char* rangeName=nullptr) = 0 ;
  virtual Int_t getBin(const char* rangeName=nullptr) const = 0 ;
  virtual Int_t numBins(const char* rangeName=nullptr) const = 0 ;
  virtual double getBinWidth(Int_t i, const char* rangeName=nullptr) const = 0 ;
  virtual double volume(const char* rangeName) const = 0 ;
  virtual void randomize(const char* rangeName=nullptr) = 0 ;

  virtual const RooAbsBinning* getBinningPtr(const char* rangeName) const = 0 ;
  virtual std::list<std::string> getBinningNames() const = 0;
  virtual Int_t getBin(const RooAbsBinning*) const = 0 ;

protected:

  ClassDef(RooAbsLValue,1) // Abstract variable
};

#endif
