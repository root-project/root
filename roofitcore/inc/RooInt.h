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
#ifndef ROO_INT
#define ROO_INT

#include "Rtypes.h"
#include "TNamed.h"

class RooInt : public TNamed {
public:

  RooInt() {} ;
  RooInt(Int_t value) : TNamed(), _value(value) {} ;
  RooInt(const RooInt& other) : TNamed(other), _value(other._value) {}
  virtual ~RooInt() {} ;

  // Double_t cast operator 
  inline operator Int_t() const { return _value ; }
  RooInt& operator=(Int_t value) { _value = value ; return *this ; }

  // Sorting interface ;
  Int_t Compare(const TObject* other) const ;
  virtual Bool_t IsSortable() const { return kTRUE ; }

protected:

  Int_t _value ;
  ClassDef(RooInt,1) // Container class for Int_t
};

#endif
