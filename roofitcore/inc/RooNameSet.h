/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNameSet.rdl,v 1.9 2002/09/05 04:33:45 verkerke Exp $
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
#ifndef ROO_NAME_SET
#define ROO_NAME_SET

#include "TList.h"
#include "TString.h"
#include "TClass.h"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooPrintable.hh"
class RooArgSet ;

class RooNameSet : public TObject, public RooPrintable {
public:

  // Constructors, assignment etc.
  RooNameSet();
  RooNameSet(const RooArgSet& argSet);
  RooNameSet(const RooNameSet& other) ;
  virtual TObject* Clone(const char* newname=0) const { return new RooNameSet(*this) ; }
  virtual ~RooNameSet() ;

  void refill(const RooArgSet& argSet) ;
  RooArgSet* select(const RooArgSet& list) const ;
  Bool_t operator==(const RooNameSet& other) ;  
  RooNameSet& operator=(const RooNameSet&) ;

  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  Int_t _len ;
  char* _nameList ; //[_len]

protected:

  void extendBuffer(Int_t inc) ;

  ClassDef(RooNameSet,1) // A sterile version of RooArgSet, containing only the names of the contained RooAbsArgs
};

#endif
