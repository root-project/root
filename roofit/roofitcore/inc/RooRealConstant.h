/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealConstant.h,v 1.13 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_REAL_CONSTANT
#define ROO_REAL_CONSTANT

#include "Rtypes.h"

class RooAbsReal ;
class RooArgList ;
#include "RooConstVar.h"

class RooRealConstant {
public:

  inline RooRealConstant() {} ;
  virtual ~RooRealConstant() {} ;
  static RooConstVar& value(Double_t value) ;

  static RooConstVar& removalDummy() ;

protected:

  static RooArgList& constDB();

  ClassDef(RooRealConstant,0) // RooRealVar constants factory
};


#endif
