/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
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
