/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooUnblindUniform.h,v 1.5 2007/05/11 10:15:52 verkerke Exp $
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
#ifndef ROO_UNBLIND_UNIFORM
#define ROO_UNBLIND_UNIFORM

#include "RooAbsHiddenReal.h"
#include "RooRealProxy.h"
#include "RooBlindTools.h"

class RooUnblindUniform : public RooAbsHiddenReal {
public:
  // Constructors, assignment etc
  RooUnblindUniform() ;
  RooUnblindUniform(const char *name, const char *title, 
		      const char *blindString, Double_t scale, RooAbsReal& blindValue);
  RooUnblindUniform(const RooUnblindUniform& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooUnblindUniform(*this,newname); }  
  virtual ~RooUnblindUniform();

protected:

  // Function evaluation
  virtual Double_t evaluate() const ;

  RooRealProxy _value ;
  RooBlindTools _blindEngine ;

  ClassDef(RooUnblindUniform,1) // Uniform unblinding transformation
};

#endif
