/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooDouble.rdl,v 1.7 2005/02/25 14:22:56 wverkerke Exp $
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

  RooDouble() {} ;
  RooDouble(Double_t value) : TNamed(), _value(value) {} ;
  RooDouble(const RooDouble& other) : TNamed(other), _value(other._value) {}
  virtual ~RooDouble() {} ;

  // Double_t cast operator 
  inline operator Double_t() const { return _value ; }
  RooDouble& operator=(Double_t value) { _value = value ; return *this ; }
  
  // Sorting interface ;
  Int_t Compare(const TObject* other) const ;
  virtual Bool_t IsSortable() const { return kTRUE ; }

protected:

  Double_t _value ;
  ClassDef(RooDouble,1) // Container class for Double_t
};

#endif
