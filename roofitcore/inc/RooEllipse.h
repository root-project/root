/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCurve.rdl,v 1.12 2001/11/19 18:03:20 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   31-Jan-2002 DK Created initial version
 *
 * Copyright (C) 2002 Stanford University
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
