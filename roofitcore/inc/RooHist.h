/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooHist.rdl,v 1.21 2006/07/03 15:37:11 wverkerke Exp $
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
#ifndef ROO_HIST
#define ROO_HIST

#include "TGraphAsymmErrors.h"
#include "RooPlotable.h"
#include "RooAbsData.h"

class TH1;
class RooCurve ;

class RooHist : public TGraphAsymmErrors, public RooPlotable {
public:
  RooHist() {} ;
  RooHist(Double_t nominalBinWidth, Double_t nSigma= 1, Double_t xErrorFrac=1.0);
  RooHist(const TH1 &data, Double_t nominalBinWidth= 0, Double_t nSigma= 1, RooAbsData::ErrorType=RooAbsData::Poisson, Double_t xErrorFrac=1.0);
  RooHist(const TH1 &data1, const TH1 &data2, Double_t nominalBinWidth= 0, Double_t nSigma= 1, Double_t xErrorFrac=1.0);
  RooHist(const RooHist& hist1, const RooHist& hist2, Double_t wgt1=1.0, Double_t wgt2=1.0, RooAbsData::ErrorType etype=RooAbsData::Poisson, Double_t xErrorFrac=1.0) ;
  virtual ~RooHist();

  // add a datapoint for a bin with n entries, using a Poisson error
  void addBin(Axis_t binCenter, Int_t n, Double_t binWidth= 0, Double_t xErrorFrac=1.0);
  // add a datapoint for a bin with n entries, using a given error
  void addBinWithError(Axis_t binCenter, Double_t n, Double_t elow, Double_t ehigh, Double_t binWidth= 0, Double_t xErrorFrac=1.0);
  // add a datapoint for the asymmetry (n1-n2)/(n1+n2), using a binomial error
  void addAsymmetryBin(Axis_t binCenter, Int_t n1, Int_t n2, Double_t binWidth= 0, Double_t xErrorFrac=1.0);

  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  Double_t getFitRangeNEvt() const;
  Double_t getFitRangeNEvt(Double_t xlo, Double_t xhi) const ;
  Double_t getFitRangeBinW() const;
  inline Double_t getNominalBinWidth() const { return _nominalBinWidth; }
  inline void setRawEntries(Double_t n) { _rawEntries = n ; } 

  Bool_t hasIdenticalBinning(const RooHist& other) const ;

  RooHist* makeResidHist(const RooCurve& curve,bool normalize=false) const;
  RooHist* makePullHist(const RooCurve& curve) const {return makeResidHist(curve,false); }

protected:
  void initialize();
  Int_t roundBin(Double_t y);

private:
  Double_t _nominalBinWidth,_nSigma,_entries,_rawEntries;
  ClassDef(RooHist,1) // 1-dimensional histogram with error bars
};

#endif
