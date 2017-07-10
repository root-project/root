/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealMPFE.h,v 1.7 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_TASK_SPEC
#define ROO_TASK_SPEC

#include "RooFit.h"
#include <cstdlib>
#include <sstream>
#include "RooMsgService.h"
#include "RooNLLVar.h"
#include "RooAbsTestStatistic.h"
#include "RooAbsOptTestStatistic.h"
#include "RooAddition.h"
#include "RooAbsTestStatistic.h"

class RooTaskSpec {
 public:
  //  RooTaskSpec(const Int_t _fit_case, const pdfName name,const Bool_t binned);
 RooTaskSpec(RooAbsOptTestStatistic* nll);
 RooTaskSpec(RooAbsReal* nll);
 // virtual TObject* clone(const char* newname) const { return new RooTaskSpec(*this,newname); }
 // virtual ~RooTaskSpec();

 private:
 void _initialise(RooAbsOptTestStatistic* rats);
 Int_t _fit_case;
 Bool_t _binned;
 // ClassDef(RooTaskSpec,0)
};

#endif
