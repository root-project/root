/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHist.rdl,v 1.9 2001/11/09 21:25:40 verkerke Exp $
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
  RooHist() {} ;
  RooHist(Double_t nominalBinWidth, Double_t nSigma= 1);
  RooHist(const TH1 &data, Double_t nominalBinWidth= 0, Double_t nSigma= 1);
  RooHist(const TH1 &data1, const TH1 &data2, Double_t nominalBinWidth= 0, Double_t nSigma= 1);
  virtual ~RooHist();

  // add a datapoint for a bin with n entries, using a Poisson error
  void addBin(Axis_t binCenter, Int_t n, Double_t binWidth= 0);
  // add a datapoint for the asymmetry (n1-n2)/(n1+n2), using a binomial error
  void addAsymmetryBin(Axis_t binCenter, Int_t n1, Int_t n2, Double_t binWidth= 0);

  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  Double_t getFitRangeNEvt() const;
  Double_t getFitRangeBinW() const;
  inline Double_t getNominalBinWidth() const { return _nominalBinWidth; }

protected:
  void initialize();
  Int_t roundBin(Stat_t y);
private:
  Double_t _nominalBinWidth,_nSigma,_entries;
  ClassDef(RooHist,1) // 1-dimensional histogram with error bars
};

#endif
