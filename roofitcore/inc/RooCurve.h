/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHist.rdl,v 1.3 2001/04/22 18:15:32 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   29-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_CURVE
#define ROO_CURVE

#include "TGraph.h"
#include "RooFitCore/RooPlotable.hh"

class RooAbsReal;
class RooRealVar;
class RooAbsFunc1D;

class RooCurve : public TGraph, public RooPlotable {
public:
  RooCurve();
  RooCurve(const RooAbsReal &func, RooRealVar &x, Double_t prec= 1e-3);
  RooCurve(const char *name, const char *title, const RooAbsFunc1D &func, Double_t xlo, Double_t xhi,
	   UInt_t minPoints, Double_t prec= 1e-3);
  virtual ~RooCurve();
  void addPoint(Double_t x, Double_t y);
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }
protected:
  void addPoints(const RooAbsFunc1D &func, Double_t xlo, Double_t xhi,
		 Int_t minPoints, Double_t prec);
  Double_t addRange(const RooAbsFunc1D& func, Double_t x1, Double_t x2, Double_t y1);
  ClassDef(RooCurve,1) // a 1-dim smooth curve
};

#endif
