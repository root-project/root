/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   01-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/
#ifndef ROO_DOUBLE
#define ROO_DOUBLE

#include "Rtypes.h"
#include "TObject.h"

class RooDouble : public TObject {
public:

  RooDouble(Double_t value) : TObject(), _value(value) {} ;
  RooDouble(const RooDouble& other) : TObject(other), _value(other._value) {}
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
