/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooKeysPdf.h,v 1.10 2007/05/11 09:13:07 verkerke Exp $
 * Authors:                                                                  *
 *   GR, Gerhard Raven,   UC San Diego,        raven@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_KEYS
#define ROO_KEYS

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;

class RooKeysPdf : public RooAbsPdf {
public:
  enum Mirror { NoMirror, MirrorLeft, MirrorRight, MirrorBoth,
      MirrorAsymLeft, MirrorAsymLeftRight,
      MirrorAsymRight, MirrorLeftAsymRight,
      MirrorAsymBoth };
  RooKeysPdf() ;
  RooKeysPdf(const char *name, const char *title,
             RooAbsReal& x, RooDataSet& data, Mirror mirror= NoMirror,
        double rho=1);
  RooKeysPdf(const char *name, const char *title,
             RooAbsReal& x, RooRealVar& xdata, RooDataSet& data, Mirror mirror= NoMirror,
        double rho=1);
  RooKeysPdf(const RooKeysPdf& other, const char* name=0);
  TObject* clone(const char* newname) const override {return new RooKeysPdf(*this,newname); }
  ~RooKeysPdf() override;

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars,
     const char* rangeName = 0) const override;
  double analyticalIntegral(Int_t code, const char* rangeName = 0) const override;
  Int_t getMaxVal(const RooArgSet& vars) const override;
  double maxVal(Int_t code) const override;

  void LoadDataSet( RooDataSet& data);

protected:

  RooRealProxy _x ;
  double evaluate() const override;

private:
  // how far you have to go out in a Gaussian until it is smaller than the
  // machine precision
  static const double _nSigma; //!

  Int_t _nEvents;
  double *_dataPts;  //[_nEvents]
  double *_dataWgts; //[_nEvents]
  double *_weights;  //[_nEvents]
  double _sumWgt ;

  enum { _nPoints = 1000 };
  double _lookupTable[_nPoints+1];

  double g(double x,double sigma) const;

  bool _mirrorLeft, _mirrorRight;
  bool _asymLeft, _asymRight;

  // cached info on variable
  Char_t _varName[128];
  double _lo, _hi, _binWidth;
  double _rho;

  ClassDefOverride(RooKeysPdf,2) // One-dimensional non-parametric kernel estimation p.d.f.
};

#endif
