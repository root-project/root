/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataSet.rdl,v 1.6 2001/03/29 01:06:44 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   10-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#include "BaBar/BaBar.hh"

#include "RooFitCore/RooPrintable.hh"

#include <iostream.h>
#include <iomanip.h>
#include "TString.h"
#include "TNamed.h"

ClassImp(RooPrintable)

static const char rcsid[] =
"$Id: RooPlot.cc,v 1.1 2001/03/28 19:21:48 davidk Exp $";

void RooPrintable::printToStream(ostream& os, PrintOption opt, const char *indent) const {
  // Print information about this object to the specified stream. The possible
  // PrintOptions are:
  //
  //   OneLine  - print a one line summary (see oneLinePrint())
  //   Standard - the default level of printing
  //   Shape    - add information about our "shape"
  //   Verbose  - the most detailed level of printing
  // 
  // Use the optional indent parameter to prefix all output after the first line.
  // The default implementation prints a one-line warning message.

  os << "*** PrintToStream not implemented ***" << endl;
}

RooPrintable::PrintOption RooPrintable::parseOptions(Option_t *options) const {
  // Apply the following PrintOption mapping:
  //
  //  "1" - OneLine
  //  "S" - Shape
  //  "V" - Verbose
  //
  // The default is Standard. If multiple options are specified,
  // the most verbose level is used.

  TString opts(options);
  opts.ToUpper();
  PrintOption popt(Standard);
  if(opts.Contains("1")) { popt= OneLine ; }
  if(opts.Contains("S")) { popt= Shape; }
  if(opts.Contains("V")) { popt= Verbose; }

  return popt;
}

void RooPrintable::oneLinePrint(ostream& os, const TNamed &named) {
  // Provide a standard implementation of one-line printing consisting of
  //
  // <classname>::<name>: "<title>"
  //
  // The title is omitted if it is empty. Subclasses should call this method
  // to generate the first line of output in their printToStream() implementations.

  os << named.ClassName() << "::" << named.GetName() << ": \"" << named.GetTitle() << "\"" << endl;
}

ostream &RooPrintable::defaultStream(ostream *os) {
  // Return a reference to the default stream to use in Print(). Use the optional
  // parameter to specify a new default stream. This method allows subclasses to
  // provide an inline implementation of Print() without pulling in iostream.h.

  static ostream *_defaultStream = &cout;
  if(0 != os) _defaultStream= os;
  return *_defaultStream;
}
