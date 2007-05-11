/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooPrintable.rdl,v 1.11 2005/06/20 15:44:56 wverkerke Exp $
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

#include "Riostream.h"
#include "Rtypes.h"
#include "TString.h"

class RooPrintable {
public:
  inline RooPrintable() { }
  inline virtual ~RooPrintable() { }
  enum PrintOption { InLine=0, OneLine=1, Standard=2, Shape=3, Verbose=4 } ;
  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const;
  PrintOption parseOptions(Option_t *options) const;
  PrintOption lessVerbose(PrintOption opt) const;
  static void inLinePrint(ostream& os, const TNamed &named);
  static void oneLinePrint(ostream& os, const TNamed &named);
  static ostream& defaultStream(ostream *os= 0);


  ClassDef(RooPrintable,1) // Interface for printable objects
};

namespace RooFit {
ostream& operator<<(ostream& os, const RooPrintable& rp) ; 
}

#ifndef __CINT__
using RooFit::operator<< ;
#endif

#endif
