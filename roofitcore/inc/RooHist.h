/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHist.rdl,v 1.2 2001/04/21 01:13:11 david Exp $
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
#include "RooFitCore/RooPrintable.hh"

class TH1;

class RooHist : public TGraphAsymmErrors, public RooPrintable {
public:
  RooHist(Double_t nSigma= 1);
  RooHist(const TH1 &data, Double_t nSigma= 1);
  virtual ~RooHist();
  // add a datapoint for a bin with n entries, using a Poisson error
  void addBin(Axis_t binCenter, Int_t n);
  // add a datapoint for the asymmetry (n1-n2)/(n1+n2), using a binomial error
  void addAsymmetryBin(Axis_t binCenter, Int_t n1, Int_t n2);
  // return the maximum extent of an error bar
  inline Float_t getPlotMax() { return _ymax; }
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }
  inline const char* getYAxisLabel() const { return _yAxisLabel.Data(); }
  inline setYAxisLabel(const char *label) { _yAxisLabel= label; }
protected:
  void initialize();
  Int_t roundBin(Stat_t y);
private:
  TString _yAxisLabel;
  Double_t _nSigma,_ymax;
  ClassDef(RooHist,1) // a plot with error bars
};

#endif
