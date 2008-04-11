/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooUnblindCPAsymVar.h,v 1.15 2007/05/11 10:15:52 verkerke Exp $
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
#ifndef ROO_UNBLIND_CPASYM_VAR
#define ROO_UNBLIND_CPASYM_VAR

#include "RooAbsHiddenReal.h"
#include "RooAbsCategory.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooBlindTools.h"

class RooCategory ;

class RooUnblindCPAsymVar : public RooAbsHiddenReal {
public:
  // Constructors, assignment etc
  RooUnblindCPAsymVar() ;
  RooUnblindCPAsymVar(const char *name, const char *title, 
			const char *blindString, RooAbsReal& cpasym);
  RooUnblindCPAsymVar(const char *name, const char *title, 
		      const char *blindString, RooAbsReal& cpasym, RooAbsCategory& blindState);
  RooUnblindCPAsymVar(const RooUnblindCPAsymVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooUnblindCPAsymVar(*this,newname); }  
  virtual ~RooUnblindCPAsymVar();

protected:

  // Function evaluation
  virtual Double_t evaluate() const ;

  RooRealProxy _asym ;
  RooBlindTools _blindEngine ;

  ClassDef(RooUnblindCPAsymVar,1) // CP-Asymmetry unblinding transformation
};

#endif
