/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooPrintable.rdl,v 1.6 2002/09/05 04:33:49 verkerke Exp $
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
#ifndef ROO_PRINTABLE
#define ROO_PRINTABLE


class TNamed ;

#include <iostream.h>
#include "Rtypes.h"
#include "TString.h"

class RooPrintable {
public:
  inline RooPrintable() { }
  inline virtual ~RooPrintable() { }
  enum PrintOption { OneLine=0, Standard=1, Shape=2, Verbose=3 } ;
  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const;
  PrintOption parseOptions(Option_t *options) const;
  PrintOption lessVerbose(PrintOption opt) const;
  static void oneLinePrint(ostream& os, const TNamed &named);
  static ostream& defaultStream(ostream *os= 0);
  ClassDef(RooPrintable,1) // Interface for printable objects
};

#endif
