/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCurve.rdl,v 1.4 2001/06/16 20:28:20 david Exp $
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
class RooAbsFunc;
class RooArgSet;

class RooCurve : public TGraph, public RooPlotable {
public:
  RooCurve();
  RooCurve(const RooAbsReal &func, RooRealVar &x, Double_t scaleFactor= 1,
	   const RooArgSet *normVars= 0, Double_t prec= 1e-3);
  RooCurve(const char *name, const char *title, const RooAbsFunc &func, Double_t xlo,
	   Double_t xhi, UInt_t minPoints, Double_t prec= 1e-3);
  virtual ~RooCurve();

  void addPoint(Double_t x, Double_t y);

  Double_t getFitRangeNorm() const;

  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

protected:
  void initialize();
  void addPoints(const RooAbsFunc &func, Double_t xlo, Double_t xhi,
		 Int_t minPoints, Double_t prec);
  void addRange(const RooAbsFunc& func, Double_t x1, Double_t x2, Double_t y1,
		Double_t y2, Double_t prec);
  ClassDef(RooCurve,1) // 1-dimensional smooth curve
};

#endif
