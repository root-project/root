/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: Roo2DKeysPdf.h,v 1.12 2007/05/11 09:13:07 verkerke Exp $
 * Authors:                                                                  *
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu         *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California,         *
 *                          Liverpool University,                            *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_2DKEYS
#define ROO_2DKEYS

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"
#include "RooDataSet.h"

////////////////////////////////////////////////////////////////////////////////////
class Roo2DKeysPdf : public RooAbsPdf
{
public:
  Roo2DKeysPdf(const char *name, const char *title,
             RooAbsReal& xx, RooAbsReal &yy, RooDataSet& data, TString options = "a", double widthScaleFactor = 1.0);
  Roo2DKeysPdf(const Roo2DKeysPdf& other, const char* name=0);
  TObject* clone(const char* newname) const override { return new Roo2DKeysPdf(*this,newname); }

  ~Roo2DKeysPdf() override;

//load in a new dataset and re-calculate the PDF
//return 0 if successful
  Int_t    loadDataSet(RooDataSet& data, TString options);

// The Roo2DKeys options available are:
//      a = select an adaptive bandwidth [default]
//      n = select a normal bandwidth
//      m = mirror kernel contributions at edges [fold gaussians back into the x,y plane]
//      d = print debug statements [useful for development only; default is off]
//      v  = print verbose debug statements [useful for development only; default is off]
//      vv = print ludicrously verbose debug statements [useful for development only; default is off]
  void     setOptions(TString options);
  void     getOptions(void) const;

// Set the value of a scale factor to modify the bandwidth by. The default value for this is unity.
// Modification of 'normal' bandwidths is useful when the data are not 'normally distributed',
// otherwise one expects little departure from that behavior.  Note that both the normal and adaptive
// bandwidth selections are modified by this factor.  Useful for systematic studies.
//           ***********
//           *IMPORTANT* The kernel is proportional to 1/widthScaleFactor.
//           ***********
  inline void     setWidthScaleFactor(double widthScaleFactor);

// choose the kernel bandwith to use.  The default is 0
//    0 = use adaptive kernel estimator (uses local population to vary with of kernels)
//    1 = use trivial kernel estimator (uses all data and sigma to estimate uniform kernel bandwidth)
  Int_t    calculateBandWidth(Int_t kernel = -999);

  Int_t    getBandWidthType() const;
  double getMean(const char * axis) const;
  double getSigma(const char * axis) const;

// print content and basic information about the data
  void     PrintInfo(std::ostream &) const;

// save PDF to a file as a TH2F *, TTree * or both
// this is so that you only need to compute the PDF once and
// are free to use the much faster Roo2DHistPdf class in order
// to perform fits/do toy studies etc.
  void     writeToFile(char * outputFile, const char * name) const;
  void     writeHistToFile(char * outputFile, const char * histName) const;
  void     writeNTupleToFile(char * outputFile, const char * name) const;

  RooRealProxy x;
  RooRealProxy y;

  double evaluate() const override;

protected:

private:
  // these are used in calculating bandwidths for x and y
  double evaluateFull(double thisX, double thisY) const;
  double g(double var1, double * _var1, double sigma1, double var2,
        double * _var2, double sigma2) const;

  //mirror corrections for the boundaries
  double highBoundaryCorrection(double thisVar, double thisH, double high, double tVar) const;
  double lowBoundaryCorrection(double thisVar, double thisH, double low, double tVar) const;

  double * _x;
  double * _hx;
  double * _y;
  double * _hy;
  double   _norm;
  double   _xMean;    // the (x,y) mean and sigma are properties of the data, not of the PDF
  double   _xSigma;
  double   _yMean;
  double   _ySigma;
  double   _n;         //coefficient of the kernel estimation sum
  double   _n16;       //pow(_nEvents, -1/6)
  double   _sqrt2pi;
  double   _2pi;       // = M_PI*2
  double   _lox,_hix;
  double   _loy,_hiy;
  double   _xoffset;
  double   _yoffset;
  double   _widthScaleFactor; //allow manipulation of the bandwidth by a scale factor

  Int_t      _nEvents;
  Int_t      _BandWidthType;
  Int_t      _MirrorAtBoundary;
  Int_t      _debug;
  Int_t      _verbosedebug;
  Int_t      _vverbosedebug;

  ClassDefOverride(Roo2DKeysPdf,0) // Two-dimensional kernel estimation p.d.f.
};

inline void  Roo2DKeysPdf::setWidthScaleFactor(double widthScaleFactor) { _widthScaleFactor = widthScaleFactor; }

#endif
