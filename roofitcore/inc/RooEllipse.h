/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooEllipse.rdl,v 1.3 2002/09/05 04:33:25 verkerke Exp $
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
#ifndef ROO_ELLIPSE
#define ROO_ELLIPSE

#include "TGraph.h"
#include "RooFitCore/RooPlotable.hh"

class RooEllipse : public TGraph, public RooPlotable {
public:
  RooEllipse();
  RooEllipse(const char *name, Double_t x1, Double_t x2, Double_t s1, Double_t s2, Double_t rho= 0, Int_t points= 100);
  virtual ~RooEllipse();

  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  };

  // These methods return zero to indicate that they do not support
  // this interface. See RooPlot::updateFitRangeNorm() for details.
  inline virtual Double_t getFitRangeNEvt() const { return 0; }
  inline virtual Double_t getFitRangeBinW() const { return 0; }

  ClassDef(RooEllipse,1) // 2-dimensional contour
};

#endif
