/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooTObjWrap.rdl,v 1.6 2005/02/25 14:23:03 wverkerke Exp $
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
#ifndef ROO_TOBJ_WRAP
#define ROO_TOBJ_WRAP

#include "Rtypes.h"
#include "TNamed.h"
#include "RooLinkedList.h"

class RooTObjWrap : public TNamed {
public:

  RooTObjWrap(Bool_t isArray=kFALSE) : _isArray(isArray) {} ;
  RooTObjWrap(TObject* obj, Bool_t isArray=kFALSE) : TNamed(), _isArray(isArray) { _list.Add(obj) ; } 
  RooTObjWrap(const RooTObjWrap& other) : TNamed(other), _list(other._list) {}
  virtual ~RooTObjWrap() {} ;

  TObject* obj() const { return _list.At(0) ; }
  const RooLinkedList& objList() const { return _list ; }

  void setObj(TObject* obj) { 
     if (!_isArray) {
         _list.Clear() ;
     }
    _list.Add(obj) ; 
   }

protected:

  Bool_t _isArray ;
  RooLinkedList _list ;
  ClassDef(RooTObjWrap,1) // Container class for Int_t
};

#endif
