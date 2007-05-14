/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooPrintable.cxx,v 1.19 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [MISC] --
// Classes inheriting from this class can be plotted and printed.

#include "RooFit.h"

#include "RooPrintable.h"
#include "RooPrintable.h"

#include "Riostream.h"
#include <iomanip>
#include "TNamed.h"

ClassImp(RooPrintable)
;

// Implement ostream operator on RooPrintable in terms of printToStream(InLine)  
namespace RooFit {
ostream& operator<<(ostream& os, const RooPrintable& rp) { 
  rp.printToStream(os,RooPrintable::InLine) ; return os ; 
}
}

void RooPrintable::printToStream(ostream& os, PrintOption /*opt*/, TString /*indent*/) const {
  // Print information about this object to the specified stream. The possible
  // PrintOptions are:
  //
  //   InLine   - print contents suitable for inlining (i.e. no newlines in printout)
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
  //  "I" - InLine
  //  "1" - OneLine
  //  "S" - Shape
  //  "V" - Verbose
  //
  // The default is Standard. If multiple options are specified,
  // the most verbose level is used.

  TString opts(options);
  opts.ToUpper();
  PrintOption popt(Standard);
  if(opts.Contains("I")) { popt= InLine ; }
  if(opts.Contains("1")) { popt= OneLine ; }
  if(opts.Contains("S")) { popt= Shape; }
  if(opts.Contains("V")) { popt= Verbose; }

  return popt;
}

RooPrintable::PrintOption RooPrintable::lessVerbose(PrintOption opt) const {
  // Return a PrintOption that is one degree less verbose than the input option.
  // Useful for being less verbose when printing info about sub-objects.

  switch(opt) {
  case InLine:
    return InLine ;
  case OneLine:
    return InLine;
  case Standard:
    return OneLine;
  case Shape:
  case Verbose:
  default:
    return Standard;
  }
}

void RooPrintable::inLinePrint(ostream& os, const TNamed &named) {
  // Provide a standard implementation of in-line printing consisting of
  //
  // <classname>::<name> 
  //
  os << named.ClassName() << "::" << named.GetName() ;
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
  // Return a reference to the current default stream to use in
  // Print(). Use the optional parameter to specify a new default
  // stream (a reference to the old one is still returned). This
  // method allows subclasses to provide an inline implementation of
  // Print() without pulling in iostream.h.

  static ostream *_defaultStream = &cout;

  ostream& _oldDefault= *_defaultStream;
  if(0 != os) _defaultStream= os;
  return _oldDefault;
}
