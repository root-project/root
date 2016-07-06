/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooParamKeysPdf.h 888 2014-08-01 19:54:39Z adye $
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
#ifndef ROO_PARAM_KEYS
#define ROO_PARAM_KEYS

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;

class RooParamKeysPdf : public RooAbsPdf {
public:
  enum Mirror { NoMirror, MirrorLeft, MirrorRight, MirrorBoth,
		MirrorAsymLeft, MirrorAsymLeftRight,
		MirrorAsymRight, MirrorLeftAsymRight,
		MirrorAsymBoth };
  RooParamKeysPdf() ;
  RooParamKeysPdf(
    const char *name, const char *title,
    RooAbsReal& x, RooAbsReal& deltax, 
    RooDataSet& data, Mirror mirror= NoMirror, Double_t rho=1, Int_t nPoints=1000
  );
  RooParamKeysPdf(
    const char *name, const char *title,
    RooAbsReal& x, RooAbsReal& deltax, double centralValue, RooAbsReal& multiplicativeShift,
    RooDataSet& data, Mirror mirror= NoMirror, Double_t rho=1, Int_t nPoints=1000
  );
  RooParamKeysPdf(const RooParamKeysPdf& other, const char* name=0);
  virtual TObject* clone(const char* newname) const {return new RooParamKeysPdf(*this,newname); }
  virtual ~RooParamKeysPdf();
  
  void LoadDataSet( RooDataSet& data);

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

protected:
  
  RooRealProxy _x ;
  RooRealProxy _deltax ;
  double _centralValue;
  RooRealProxy _multiplicativeShift;
  Double_t evaluate() const;

private:
  
  Double_t evaluateFull(Double_t x) const;

  Int_t _nEvents;
  Double_t *_dataPts;  //[_nEvents]
  Double_t *_dataWgts; //[_nEvents]
  Double_t *_weights;  //[_nEvents]
  Double_t _sumWgt ;
  mutable Double_t _normVal ;
  
  Int_t _nPoints;

  //enum { _nPoints = 1000 };
  Double_t *_lookupTable; //[_nPoints] 
  
  Double_t g(Double_t x,Double_t sigma) const;

  Bool_t _mirrorLeft, _mirrorRight;
  Bool_t _asymLeft, _asymRight;

  // cached info on variable
  Char_t _varName[128];
  Double_t _lo, _hi, _binWidth;
  Double_t _rho;
  
  ClassDef(RooParamKeysPdf,4) // One-dimensional non-parametric kernel estimation p.d.f.
};

#ifdef __CINT__
// Specify schema conversion rule here, rather than in LinkDef1.h, so it is included if we compile with ACLiC.
#pragma read sourceClass="RooParamKeysPdf" \
  targetClass="RooParamKeysPdf" \
  version="[-2]" \
  source="Double_t _lookupTable[1001]" \
  target="_nPoints, _lookupTable" \
  code="{ _nPoints=1000; _lookupTable=new Double_t[_nPoints]; for (Int_t i=0; i<_nPoints; i++) _lookupTable[i]= onfile._lookupTable[i]; }"
#endif

#endif
