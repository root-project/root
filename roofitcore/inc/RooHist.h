/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHist.rdl,v 1.3 2001/04/22 18:15:32 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/
#ifndef ROO_HIST
#define ROO_HIST

#include "TGraphAsymmErrors.h"
#include "RooFitCore/RooPlotable.hh"

class TH1;

class RooHist : public TGraphAsymmErrors, public RooPlotable {
public:
  RooHist(Double_t nSigma= 1);
  RooHist(const TH1 &data, Double_t nSigma= 1);
  virtual ~RooHist();
  // add a datapoint for a bin with n entries, using a Poisson error
  void addBin(Axis_t binCenter, Int_t n);
  // add a datapoint for the asymmetry (n1-n2)/(n1+n2), using a binomial error
  void addAsymmetryBin(Axis_t binCenter, Int_t n1, Int_t n2);
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }
protected:
  void initialize();
  Int_t roundBin(Stat_t y);
private:
  Double_t _nSigma;
  ClassDef(RooHist,1) // a 1-dim histogram with error bars
};

#endif
