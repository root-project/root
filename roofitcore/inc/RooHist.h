/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlotWithErrors.rdl,v 1.1 2001/03/28 19:21:48 davidk Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/
#ifndef ROO_HIST
#define ROO_HIST

#include "TGraphAsymmErrors.h"

class RooHist : public TGraphAsymmErrors {
public:
  RooHist(Double_t nSigma= 1);
  virtual ~RooHist();
  // add a datapoint for a bin with n entries, using a Poisson error
  void addBin(Float_t binCenter, Int_t n);
  // add a datapoint for the asymmetry (n1-n2)/(n1+n2), using a binomial error
  void addAsymmetryBin(Float_t binCenter, Int_t n1, Int_t n2);
  // return the maximum extent of an error bar
  inline Float_t getPlotMax() { return _ymax; }
private:
  void initialize();
  Double_t _nSigma,_ymax;
  ClassDef(RooHist,1) // a plot with error bars
};

#endif
