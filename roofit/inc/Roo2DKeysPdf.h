/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: Roo2DKeysPdf.rdl,v 1.1 2001/08/26 17:48:48 bevan Exp $
 * Authors:
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
 *
 * History:
 *   25-Jul-2001 AB Created 2D KEYS version based on the 1D template of RooKeysPdf
 *                  by Gerhard Raven.
 *   Wed Aug 15  AB changed grid to 100*100 instead of 50*50
 *   25-Aug-2001 AB Ported to RooFitCore/RooFitModels
 *
 * Copyright (C) 2001, Liverpool University
 *****************************************************************************/
#ifndef ROO_2DKEYS
#define ROO_2DKEYS

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooDataSet.hh"

//mirror correction limit for boundary correction
const Double_t ROO2DKEYSPDF_NSIGMAMIROOR = 3.0;

class Roo2DKeysPdf : public RooAbsPdf {
public:
  Roo2DKeysPdf(const char *name, const char *title,
             RooAbsReal& x, RooAbsReal &y, RooDataSet& data, TString options = "a");
  Roo2DKeysPdf(const Roo2DKeysPdf& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new Roo2DKeysPdf(*this,newname); }

  virtual ~Roo2DKeysPdf();

// choose the kernel bandwith to use.  The default is 0                                               
//    0 = use adaptive kernel estimator (uses local population to vary with of kernels)               
//    1 = use trivial kernel estimator (uses all data and sigma to estimate uniform kernel bandwidth)
  void  calculateBandWidth(Int_t kernel);

// The Roo2DKeys options available are:
//	a = select an adaptove bandwidth [default]
//      n = select a normal bandwidth
//      m = mirror kernal contributions at edges [fold gaussians back into the x,y plane]
  void  setOptions(TString options);
  Int_t getBandWidthType();
  Double_t getMean(const char * axis);
  Double_t getSigma(const char * axis);

  RooRealProxy x;
  RooRealProxy y;

  Double_t evaluate() const;
protected:

private:
  // these are used in calculating bandwidths for x and y
  Double_t evaluateFull(Double_t thisX, Double_t thisY);
  Double_t g(Double_t var1, Double_t * _var1, Double_t sigma1, Double_t var2, 
	     Double_t * _var2, Double_t sigma2); 

  //mirror corrections for the boundaries
  Double_t xBoundaryCorrection(Double_t thisX, Int_t ix);
  Double_t yBoundaryCorrection(Double_t thisY, Int_t iy);


  Double_t * _x;
  Double_t * _hx;
  Double_t * _y;
  Double_t * _hy;
  Double_t   _norm;

  Int_t      _nEvents;

  Double_t   _xMean;
  Double_t   _xSigma;
  Double_t   _yMean;
  Double_t   _ySigma;
  Double_t   _xbinWidth;
  Double_t   _ybinWidth;
  Double_t   _n;         //coefficient of the kernel estimation sum
  Double_t   _n16;       //pow(_nEvents, -1/6)
  Double_t   _sqrt2pi;
  Double_t   _2pi;
  Double_t   _lox,_hix;
  Double_t   _loy,_hiy;

  Int_t      _BandWidthType;
  Int_t      _MirrorAtBoundary;

  enum     { _nPoints = 100 };         //granularity of LUT - use a linear extrapolation
                                       //between points to improve upon the resolution
  Double_t   _p[_nPoints][_nPoints];   //probability of the PDF in (x,y) plane

  ClassDef(Roo2DKeysPdf,0) // Non-Parametric Multi Variate KEYS PDF
};

#endif
