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
#ifndef ROO_TOBJ_WRAP
#define ROO_TOBJ_WRAP

#include "Rtypes.h"
#include "TNamed.h"

class RooTObjWrap : public TNamed {
public:

  RooTObjWrap() {} ;
  RooTObjWrap(TObject* obj) : TNamed(), _obj(obj) {} ;
  RooTObjWrap(const RooTObjWrap& other) : TNamed(other), _obj(other._obj) {}
  virtual ~RooTObjWrap() {} ;

  TObject* obj() const { return _obj ; }
  void setObj(TObject* obj) { _obj = obj ; }

protected:

  TObject* _obj ;
  ClassDef(RooTObjWrap,1) // Container class for Int_t
};

#endif
