/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooNovosibirsk.h,v 1.7 2007/07/12 20:30:49 wouter Exp $
 * Authors:                                                                  *
 *   DB, Dieter Best,     UC Irvine,         best@slac.stanford.edu          *
 *   HT, Hirohisa Tanaka  SLAC               tanaka@slac.stanford.edu        *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_NOVOSIBIRSK
#define ROO_NOVOSIBIRSK

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;
class RooAbsReal;

class RooNovosibirsk : public RooAbsPdf {
public:
  // Your constructor needs a name and title and then a list of the
  // dependent variables and parameters used by this PDF. Use an
  // underscore in the variable names to distinguish them from your
  // own local versions.
  RooNovosibirsk() {} ;
  RooNovosibirsk(const char *name, const char *title,
       RooAbsReal& _x,     RooAbsReal& _peak,
       RooAbsReal& _width, RooAbsReal& _tail);

  RooNovosibirsk(const RooNovosibirsk& other,const char* name=0) ;

  TObject* clone(const char* newname) const override { return new RooNovosibirsk(*this,newname);   }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

  // An empty constructor is usually ok
  inline ~RooNovosibirsk() override { }

protected:
  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

private:
  RooRealProxy x;
  RooRealProxy width;
  RooRealProxy peak;
  RooRealProxy tail;

  ClassDefOverride(RooNovosibirsk,1) // Novosibirsk PDF
};

#endif
