/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCurve.rdl,v 1.23 2005/06/20 15:44:50 wverkerke Exp $
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
  RooCurve(const RooAbsReal &func, RooAbsRealLValue &x, Double_t xlo, Double_t xhi, Int_t xbins,
	   Double_t scaleFactor= 1, const RooArgSet *normVars= 0, Double_t prec= 1e-3, Double_t resolution= 1e-3,
	   Bool_t shiftToZero=kFALSE, WingMode wmode=Extended);
  RooCurve(const char *name, const char *title, const RooAbsFunc &func, Double_t xlo,
	   Double_t xhi, UInt_t minPoints, Double_t prec= 1e-3, Double_t resolution= 1e-3,
	   Bool_t shiftToZero=kFALSE, WingMode wmode=Extended);
  virtual ~RooCurve();

  RooCurve(const char* name, const char* title, const RooCurve& c1, const RooCurve& c2, Double_t scale1=1., Double_t scale2=1.) ;

  void addPoint(Double_t x, Double_t y);

  Double_t getFitRangeBinW() const;
  Double_t getFitRangeNEvt(Double_t xlo, Double_t xhi) const ;
  Double_t getFitRangeNEvt() const;

  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  Double_t chiSquare(const RooHist& hist, int nFitParam) const ;
  Int_t findPoint(Double_t value, Double_t tolerance=1e-10) const ;
  Double_t average(Double_t lo, Double_t hi) const ;
  Double_t interpolate(Double_t x, Double_t tolerance=1e-10) const ;

protected:
  void initialize();
  void addPoints(const RooAbsFunc &func, Double_t xlo, Double_t xhi,
		 Int_t minPoints, Double_t prec, Double_t resolution, WingMode wmode);
  void addRange(const RooAbsFunc& func, Double_t x1, Double_t x2, Double_t y1,
		Double_t y2, Double_t minDy, Double_t minDx);

  void shiftCurveToZero(Double_t prevYMax) ;

  ClassDef(RooCurve,1) // 1-dimensional smooth curve
};

#endif
