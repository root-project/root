/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooEllipse.h,v 1.8 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ELLIPSE
#define ROO_ELLIPSE

#include "TGraph.h"
#include "RooPlotable.h"

class RooEllipse : public TGraph, public RooPlotable {
public:
  RooEllipse();
  RooEllipse(const char *name, Double_t x1, Double_t x2, Double_t s1, Double_t s2, Double_t rho= 0, Int_t points= 100);
  ~RooEllipse() override;


  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const override;

  /// Printing interface
  inline void Print(Option_t *options= 0) const override {
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  // These methods return zero to indicate that they do not support
  // this interface. See RooPlot::updateFitRangeNorm() for details.
  inline Double_t getFitRangeNEvt() const override { return 0; }
  inline Double_t getFitRangeNEvt(Double_t, Double_t) const override { return 0; }
  inline Double_t getFitRangeBinW() const override { return 0; }

  ClassDefOverride(RooEllipse,1) // 2-dimensional contour
};

#endif
