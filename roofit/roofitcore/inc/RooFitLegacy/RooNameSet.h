/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNameSet.h,v 1.16 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_NAME_SET
#define ROO_NAME_SET

#include "TObject.h"
#include "RooPrintable.h"

#include <ROOT/RConfig.hxx>

class RooArgSet;

class RooNameSet : public TObject, public RooPrintable {
public:

  // Constructors, assignment etc.
  RooNameSet();
  RooNameSet(const RooArgSet& argSet);
  RooNameSet(const RooNameSet& other) ;
  TObject* Clone(const char*) const override { return new RooNameSet(*this) ; }
  ~RooNameSet() override ;

  void refill(const RooArgSet& argSet) ;
  RooArgSet* select(const RooArgSet& list) const ;
  Bool_t operator==(const RooNameSet& other) const;
  RooNameSet& operator=(const RooNameSet&) ;
  Bool_t operator<(const RooNameSet& other) const ;

  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printValue(std::ostream& os) const override ;

  inline void Print(Option_t *options= 0) const override {
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  void setNameList(const char* givenList) ;
  const char* content() const { return _nameList ? _nameList : ""; }

private:
  Int_t _len ;
  char* _nameList ; ///<[_len]

protected:

  void extendBuffer(Int_t inc) ;
  static void strdup(Int_t& dstlen, char* &dstbuf, const char* str);

  ClassDefOverride(RooNameSet,1) // A sterile version of RooArgSet, containing only the names of the contained RooAbsArgs
} R__SUGGEST_ALTERNATIVE("Please use RooHelpers::getColonSeparatedNameString() and RooHelpers::selectFromArgSet().");

#endif
