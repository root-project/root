/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAddition.h,v 1.3 2007/05/11 09:11:30 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,        dkirkby@uci.edu                  *
 *   PB, Patrick Bos,     Netherlands eScience Center,                       *
 *                                          p.bos@esciencecenter.nl          *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University.                         *
 *               2016,      Netherlands eScience Center.                     *
 *                          All rights reserved.                             *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_TEST_STAT_MP_DRIVER
#define ROO_TEST_STAT_MP_DRIVER

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooRealMPFE.h"

class RooRealVar;
class RooArgList ;

typedef RooAbsData* pRooAbsData ;
typedef RooRealMPFE* pRooRealMPFE ;

class RooTestStatMPDriver : public RooAbsReal {
public:

  RooTestStatMPDriver() ;
  RooTestStatMPDriver(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data, int nCPU=0) ;
  virtual ~RooTestStatMPDriver() ;

  RooTestStatMPDriver(const RooTestStatMPDriver& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooTestStatMPDriver(*this, newname); }

protected:
  
  void init() const ; 

  Double_t evaluate() const;

  RooRealProxy _nll ; // Likelihood object to be used for calculations
  mutable pRooRealMPFE* _mpfeArray = 0 ; // array of MP front-ends (WARNING array of pointers is fragile - copying of test stat may not work! )
  int _nCPU ; // Number of CPUs to be used

  ClassDef(RooTestStatMPDriver,1) // Multi-processor driver for evaluation of test statistics
};

#endif
