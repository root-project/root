/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooPlotable.cc,v 1.12 2005/06/20 15:44:56 wverkerke Exp $
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

// -- CLASS DESCRIPTION [PLOT] --
// Classes inheriting from this class can be plotted and printed, and can
// be dynamically cross-cast into TObject's.

#include "RooFit.h"

#include "RooPlotable.h"
#include "RooPlotable.h"
#include "TObject.h"
#include "Riostream.h"

ClassImp(RooPlotable)
;

void RooPlotable::printToStream(ostream& os, PrintOption opt, TString indent) const {
  if(opt >= Verbose) {
    os << indent << "--- RooPlotable ---" << endl;
    os << indent << "  y-axis min = " << getYAxisMin() << endl
       << indent << "  y-axis max = " << getYAxisMax() << endl
       << indent << "  y-axis label \"" << getYAxisLabel() << "\"" << endl;
  }
}

TObject *RooPlotable::crossCast() {
  return dynamic_cast<TObject*>(this);
}
