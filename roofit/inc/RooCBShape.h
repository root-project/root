/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooCBShape.rdl,v 1.3 2000/11/13 19:06:21 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   07-Feb-2000 DK Created initial version from RooGaussianProb
 *   19-Jun-2001 JB Ported to RooFitModels
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/
#ifndef ROO_CB_SHAPE
#define ROO_CB_SHAPE

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;

class RooCBShape : public RooAbsPdf {
public:
  RooCBShape(const char *name, const char *title, RooAbsReal& _m,
	     RooAbsReal& _m0, RooAbsReal& _sigma,
	     RooAbsReal& _alpha, RooAbsReal& _n);

  RooCBShape(const RooCBShape& other, const char* name = 0);
  virtual TObject* clone() const { return new RooCBShape(*this); }

  inline virtual ~RooCBShape() { }

protected:

  RooRealProxy m;
  RooRealProxy m0;
  RooRealProxy sigma;
  RooRealProxy alpha;
  RooRealProxy n;

  Double_t evaluate(const RooDataSet* dset) const;

private:

  ClassDef(RooCBShape,0) // Crystal Ball lineshape PDF
};

#endif
