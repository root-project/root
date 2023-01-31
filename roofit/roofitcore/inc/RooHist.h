/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooHist.h,v 1.22 2007/05/11 09:11:30 verkerke Exp $
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

class RooAbsRealLValue;
class RooCurve;
class RooFitResult;

class TH1;

class RooHist : public TGraphAsymmErrors, public RooPlotable {
public:
  RooHist() {}
  RooHist(double nominalBinWidth, double nSigma= 1, double xErrorFrac=1.0, double scaleFactor=1.0);
  RooHist(const TH1 &data, double nominalBinWidth= 0, double nSigma= 1, RooAbsData::ErrorType=RooAbsData::Poisson,
     double xErrorFrac=1.0, bool correctForBinWidth=true, double scaleFactor=1.);
  RooHist(const TH1 &data1, const TH1 &data2, double nominalBinWidth= 0, double nSigma= 1, RooAbsData::ErrorType=RooAbsData::Poisson,
     double xErrorFrac=1.0, bool efficiency=false, double scaleFactor=1.0);
  RooHist(const RooHist& hist1, const RooHist& hist2, double wgt1=1.0, double wgt2=1.0,
     RooAbsData::ErrorType etype=RooAbsData::Poisson, double xErrorFrac=1.0) ;
  RooHist(const RooAbsReal &f, RooAbsRealLValue &x, double xErrorFrac=1.0, double scaleFactor=1.0, const RooArgSet *normVars = nullptr, const RooFitResult* fr = nullptr);

  // add a datapoint for a bin with n entries, using a Poisson error
  void addBin(Axis_t binCenter, double n, double binWidth= 0, double xErrorFrac=1.0, double scaleFactor=1.0);
  // add a datapoint for a bin with n entries, using a given error
  void addBinWithError(Axis_t binCenter, double n, double elow, double ehigh, double binWidth= 0,
             double xErrorFrac=1.0, bool correctForBinWidth=true, double scaleFactor=1.0);
  // add a datapoint for a bin with n entries, using a given x and y error
  void addBinWithXYError(Axis_t binCenter, double n, double exlow, double exhigh, double eylow, double eyhigh,
                         double scaleFactor=1.0);
  // add a datapoint for the asymmetry (n1-n2)/(n1+n2), using a binomial error
  void addAsymmetryBin(Axis_t binCenter, Int_t n1, Int_t n2, double binWidth= 0, double xErrorFrac=1.0, double scaleFactor=1.0);
  // add a datapoint for the asymmetry (n1-n2)/(n1+n2), using sum-of-weights error
  void addAsymmetryBinWithError(Axis_t binCenter, double n1, double n2, double en1, double en2, double binWidth= 0, double xErrorFrac=1.0, double scaleFactor=1.0);

  // add a datapoint for the efficiency (n1)/(n1+n2), using a binomial error
  void addEfficiencyBin(Axis_t binCenter, Int_t n1, Int_t n2, double binWidth= 0, double xErrorFrac=1.0, double scaleFactor=1.0);
  // add a datapoint for the efficiency (n1)/(n1+n2), using a sum-of-weights error
  void addEfficiencyBinWithError(Axis_t binCenter, double n1, double n2, double en1, double en2, double binWidth= 0, double xErrorFrac=1.0, double scaleFactor=1.0);

  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent= "") const override;

  inline void Print(Option_t *options= nullptr) const override {
    // Printing interface
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  double getFitRangeNEvt() const override;
  double getFitRangeNEvt(double xlo, double xhi) const override ;
  /// Return (average) bin width of this RooHist.
  double getFitRangeBinW() const override { return _nominalBinWidth ; }
  inline double getNominalBinWidth() const { return _nominalBinWidth; }
  inline void setRawEntries(double n) { _rawEntries = n ; }

  bool hasIdenticalBinning(const RooHist& other) const ;

  RooHist* makeResidHist(const RooCurve& curve,bool normalize=false, bool useAverage=false) const;
  RooHist* makePullHist(const RooCurve& curve, bool useAverage=false) const
    {return makeResidHist(curve,true,useAverage); }

  bool isIdentical(const RooHist& other, double tol=1e-6, bool verbose=true) const ;


protected:
  void initialize();
  Int_t roundBin(double y);

  friend class RooPlot;

  void fillResidHist(RooHist & residHist, const RooCurve& curve,bool normalize=false, bool useAverage=false) const;
  std::unique_ptr<RooHist> createEmptyResidHist(const RooCurve& curve, bool normalize=false) const;

private:

  void addPoint(Axis_t binCenter, double y, double yscale, double exlow, double exhigh, double eylow, double eyhigh);

  double _nominalBinWidth = 1.0; ///< Average bin width
  double _nSigma = 1.0;          ///< Number of 'sigmas' error bars represent
  double _entries = 0.0;         ///< Number of entries in histogram
  double _rawEntries = 0.0;      ///< Number of entries in source dataset

  std::vector<double> _originalWeights; ///< The original bin weights that were passed to the `RooHist::addBin` functions before scaling and bin width correction

  ClassDefOverride(RooHist,2) // 1-dimensional histogram with error bars
};

#endif
