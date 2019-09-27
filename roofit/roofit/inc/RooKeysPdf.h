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
#ifndef ROOKEYS
#define ROOKEYS

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include <vector>
#include <string>

class RooRealVar;

class RooKeysPdf : public RooAbsPdf {
public:
  enum Mirror { NoMirror, MirrorLeft, MirrorRight, MirrorBoth,
                MirrorAsymLeft, MirrorAsymLeftRight,
                MirrorAsymRight, MirrorLeftAsymRight,
                MirrorAsymBoth,
                Default };
  struct Data {
    double x;
    double w;
    static inline bool compare(const Data& A, const Data& B)  {
      return A.x < B.x;
    }
  };
  
  RooKeysPdf();
  RooKeysPdf(const RooKeysPdf& other, const char* name=0);
  RooKeysPdf(const char* name, const char* title, RooAbsReal& x, RooDataSet& data, 
             Mirror mirror=Default, double rho=1, int nBins=1000);
  RooKeysPdf(const char* name, const char* title, RooAbsReal& x, RooRealVar& xdata, RooDataSet& data,           Mirror mirror=Default, double rho=1, int nBins=1000);
  virtual ~RooKeysPdf();
  
  virtual TObject* clone(const char* newname) const {
    return new RooKeysPdf(*this,newname); 
  }
  
  virtual int getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars,
                                    const char* rangeName = 0) const;
  virtual double analyticalIntegral(int code, const char* rangeName = 0) const;
  
  virtual int getMaxVal(const RooArgSet& vars) const;
  virtual double maxVal(int code) const;

  void LoadDataSet( RooDataSet& data);

protected:

  RooRealProxy x ;
  double evaluate() const;

private:
  //storage
  size_t nEvents;
  size_t nTotalEvents;
  const int nBins;
  std::vector<Data> dataArr;
  std::vector<double> adaptedWidthArr;
  std::vector<double> lookupTable;
  
  //details
  double rho;
  std::string varName;
  double lo, hi, binWidth;
  bool mirrorLeft, mirrorRight;
  bool asymLeft, asymRight;
  
  //utilities
  double gaussian(double x, double sigma, unsigned int& start, unsigned int& end) const;
  void fillLookupTable(double x, double weight, double adaptedWidth, double sign);
  // how far you have to go out in a Gaussian until it is smaller than the machine precision
  const double sigmaLowLimit = std::sqrt(-2.0*std::log(std::numeric_limits<double>::epsilon()));

  ClassDef(RooKeysPdf,3) // One-dimensional non-parametric kernel estimation p.d.f.
};

#endif
