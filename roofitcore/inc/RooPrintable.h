/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPrintable.rdl,v 1.2 2001/04/11 23:25:27 davidk Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   10-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
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
  ClassDef(RooPrintable,1) // A utility class for printing object info
};

#endif
