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
#include <list>
#include <sstream>
#include "RooMsgService.h"
#include "RooNLLVar.h"
#include "RooAbsTestStatistic.h"
#include "RooAbsOptTestStatistic.h"
#include "RooAddition.h"
#include "RooAbsTestStatistic.h"

class RooArgList;

class RooTaskSpec {
 public:
  RooTaskSpec();
  RooTaskSpec(RooAbsOptTestStatistic* nll);
  RooTaskSpec(RooAbsReal* nll);
  struct Task {
    Int_t id;
    Bool_t binned;
    const char* name;
    Int_t entries;
    Bool_t is_done;
  };
  std::list<Task> tasks;
 private:
  void _initialise(RooAbsOptTestStatistic* rats);
  Task _fill_task(const Int_t n, RooAbsOptTestStatistic* rats);
  Int_t _fit_case;
  Int_t n_tasks = tasks.size();


};
#endif
