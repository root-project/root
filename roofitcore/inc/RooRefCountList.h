/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRefCountList.rdl,v 1.1 2002/09/06 22:41:29 verkerke Exp $
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
#ifndef ROO_REF_COUNT_LIST
#define ROO_REF_COUNT_LIST

#include "THashList.h"

class RooRefCountList : public THashList {
public:
  inline RooRefCountList() : THashList() { }
  virtual ~RooRefCountList() {} ;

  virtual void AddLast(TObject* obj) ;
  virtual void AddLast(TObject* obj, Option_t* opt) { AddLast(obj) ; }
  virtual TObject* Remove(TObject* obj) ;
  virtual TObject* Remove(TObjLink* link) { return Remove(link->GetObject()) ; } ;
  virtual TObject* RemoveAll(TObject* obj) ;
  virtual void RemoveAll(TCollection* coll) { return THashList::RemoveAll(coll) ; } ;
  Int_t refCount(TObject* obj) ;
  
protected:  
  ClassDef(RooRefCountList,1) // TList with extra support for Option_t associations
};

#endif
