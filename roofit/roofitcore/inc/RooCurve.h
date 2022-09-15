/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCurve.h,v 1.24 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_CURVE
#define ROO_CURVE

#include "TGraph.h"
#include "RooPlotable.h"
#include <list>
#include <vector>
#include "TMatrixDfwd.h"

class RooAbsReal;
class RooRealVar;
class RooAbsFunc;
class RooArgSet;
class RooAbsRealLValue ;
class RooHist ;

class RooCurve : public TGraph, public RooPlotable {
public:
  RooCurve();
  enum WingMode { NoWings=0 ,Straight=1, Extended=2 } ;
  RooCurve(const RooAbsReal &func, RooAbsRealLValue &x, double xlo, double xhi, Int_t xbins,
      double scaleFactor= 1, const RooArgSet *normVars= nullptr, double prec= 1e-3, double resolution= 1e-3,
      bool shiftToZero=false, WingMode wmode=Extended, Int_t nEvalError=-1, Int_t doEEVal=false, double eeVal=0.0,
      bool showProgress=false);
  RooCurve(const char *name, const char *title, const RooAbsFunc &func, double xlo,
      double xhi, UInt_t minPoints, double prec= 1e-3, double resolution= 1e-3,
      bool shiftToZero=false, WingMode wmode=Extended, Int_t nEvalError=-1, Int_t doEEVal=false, double eeVal=0.0);
  ~RooCurve() override;

  RooCurve(const char* name, const char* title, const RooCurve& c1, const RooCurve& c2, double scale1=1., double scale2=1.) ;

  void addPoint(double x, double y);

  double getFitRangeBinW() const override;
  double getFitRangeNEvt(double xlo, double xhi) const override ;
  double getFitRangeNEvt() const override;


  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override;

  inline void Print(Option_t *options= nullptr) const override {
    // Printing interface
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  double chiSquare(const RooHist& hist, int nFitParam) const ;
  Int_t findPoint(double value, double tolerance=1e-10) const ;
  double average(double lo, double hi) const ;
  double interpolate(double x, double tolerance=1e-10) const ;

  bool isIdentical(const RooCurve& other, double tol=1e-6, bool verbose=true) const ;

  RooCurve* makeErrorBand(const std::vector<RooCurve*>& variations, double Z=1) const ;
  RooCurve* makeErrorBand(const std::vector<RooCurve*>& plusVar, const std::vector<RooCurve*>& minusVar, const TMatrixD& V, double Z=1) const ;

protected:

  void calcBandInterval(const std::vector<RooCurve*>& variations,Int_t i,double Z,double& lo, double& hi, bool approxGauss) const ;
  void calcBandInterval(const std::vector<RooCurve*>& plusVar, const std::vector<RooCurve*>& minusVar, Int_t i, const TMatrixD& V,
         double Z,double& lo, double& hi) const ;

  void initialize();
  void addPoints(const RooAbsFunc &func, double xlo, double xhi,
       Int_t minPoints, double prec, double resolution, WingMode wmode,
       Int_t numee=0, bool doEEVal=false, double eeVal=0.0,std::list<double>* samplingHint=nullptr) ;
  void addRange(const RooAbsFunc& func, double x1, double x2, double y1,
      double y2, double minDy, double minDx,
      Int_t numee=0, bool doEEVal=false, double eeVal=0.0);


  void shiftCurveToZero(double prevYMax) ;

  bool _showProgress ; ///<! Show progress indication when adding points

  ClassDefOverride(RooCurve,1) // 1-dimensional smooth curve for use in RooPlots
};

#endif
