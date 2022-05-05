/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooDouble.h,v 1.8 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_DOUBLE
#define ROO_DOUBLE

#include "Rtypes.h"
#include "TNamed.h"

class RooDouble : public TNamed {
public:

  /// Default constructor
  RooDouble() : _value(0) {
  } ;
  RooDouble(Double_t value) ;
  RooDouble(const RooDouble& other) : TNamed(other), _value(other._value) {}
  /// Destructor
  ~RooDouble() override {
  } ;

  // Double_t cast operator
  /// Return value of contained double
  inline operator Double_t() const {
    return _value ;
  }
  /// Return true if contained double equals value
  RooDouble& operator=(Double_t value) {
    _value = value ; return *this ;
  }

  // Sorting interface ;
  Int_t Compare(const TObject* other) const override ;
  /// We are a sortable object
  bool IsSortable() const override {
    return true ;
  }

protected:

  Double_t _value ; ///< Value payload
  ClassDefOverride(RooDouble,1) // Container class for Double_t
};

#endif
