/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROONUMCDF
#define ROONUMCDF

#include "RooNumRunningInt.h"

class RooNumCdf : public RooNumRunningInt {
public:
  RooNumCdf(const char *name, const char *title, RooAbsPdf& _pdf, RooRealVar& _x, const char* binningName="cache");
  RooNumCdf(const RooNumCdf& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooNumCdf(*this,newname); }
  ~RooNumCdf() override ;

protected:

  void fillCacheObject(FuncCacheElem& cacheFunc) const override ;

private:

  ClassDefOverride(RooNumCdf,1) // Numeric calculator for CDF for a given PDF

};

#endif
