/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooPrintable.h,v 1.12 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_PRINTABLE
#define ROO_PRINTABLE


class TNamed ;

#include "Riosfwd.h"
#include "Rtypes.h"
#include "TString.h"

class RooPrintable {
public:
  inline RooPrintable() { }
  inline virtual ~RooPrintable() { }

  // New-style printing

  // Master print function
  enum ContentsOption { kName=1, kClassName=2, kValue=4, kArgs=8, kExtras=16, kAddress=32, kTitle=64,  kCollectionHeader=128} ; // Can be ORed
  enum StyleOption { kInline=1, kSingleLine=2, kStandard=3, kVerbose=4, kTreeStructure=5 } ; // Exclusive
  virtual void printStream(ostream& os, Int_t contents, StyleOption style, TString indent="") const ;
  
  // Virtual hook function for class-specific content implementation
  virtual void printAddress(ostream& os) const ;
  virtual void printName(ostream& os) const ;
  virtual void printTitle(ostream& os) const ;
  virtual void printClassName(ostream& os) const ;
  virtual void printValue(ostream& os) const ;
  virtual void printArgs(ostream& os) const ;
  virtual void printExtras(ostream& os) const ;
  virtual void printMultiline(ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const ;
  virtual void printTree(ostream& os, TString indent="") const ;

  static ostream& defaultPrintStream(ostream *os= 0);
  virtual Int_t defaultPrintContents(Option_t* opt) const ;
  virtual StyleOption defaultPrintStyle(Option_t* opt) const ;
   
  // Formatting control
  static void nameFieldLength(Int_t newLen) ;

protected:

  static Int_t _nameLength ;

  ClassDef(RooPrintable,1) // Interface for printable objects


};

namespace RooFit {
ostream& operator<<(ostream& os, const RooPrintable& rp) ; 
}

#ifndef __CINT__
using RooFit::operator<< ;
#endif

#endif
