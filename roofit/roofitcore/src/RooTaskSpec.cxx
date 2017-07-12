/****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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
/**
\file RooTaskSpec.cxx
\class RooTaskSpec
\ingroup Roofitcore

RooTaskSpec is a precursor to an upgrade of the the multi-processor front-end for 
parallel calculation of RooAbsReal objects. The RooTaskSpec should return a table
containing the information that will be passed to the RooRealMPFE process. The
first calculation that is returned is the case being calculated since Binned and
Unbinned likelihood calcualtions need to be optimised for multiprocessing in very
different ways.
Several cases are possible:
Either the return value of createNLL() is a RooAbsTestStatisic or a RooAddition of
RooAbsTestStatistics and/or a RooConstraintSum. 
Case 1: return value is a RooAbsTestStatistic -> this class returns the tools to
analyse the  components of this likelihood.
Case 2: Multiple return values. The first of which is always a RooAbsTestStatistic
followed by either more RooAbsTestStatistics or a RooConstraintSum which also
signifies the last term. Here we only consider the first term since the others
represent odd and rare test cases. 
ToDo: Simple use demonstration.
**/


#include "Riostream.h"
#include "RooFit.h"
#include <cstdlib>
#include <sstream>
#include "RooMsgService.h"
#include "RooNLLVar.h"
#include "RooAbsTestStatistic.h"
#include "RooAbsOptTestStatistic.h"
#include "RooAddition.h"
#include "RooAbsTestStatistic.h"
#include "RooTaskSpec.h"
#include "RooAbsData.h"

using namespace std;
using namespace RooFit;

//ClassImp(RooTaskSpec);

//RooTaskSpec::RooTaskSpec(const Int_t case, const pdfName name,const Bool_t binned);
RooTaskSpec::RooTaskSpec(RooAbsOptTestStatistic* nll){
  RooAbsOptTestStatistic* rats = dynamic_cast<RooAbsOptTestStatistic*>(nll) ;
  if (rats) {
    cout << " NLL is a RooAbsOptTestStatistic (Case 1)" << endl ;
    rats->Print();
    _fit_case = 1;
    _initialise(rats);
  } else {
    _fit_case = 0;
  }
}

RooTaskSpec::RooTaskSpec(RooAbsReal* nll){
  RooAddition* ra = dynamic_cast<RooAddition*>(nll) ;
  ra->Print();
  if (ra) {
    RooAbsOptTestStatistic* rats = dynamic_cast<RooAbsOptTestStatistic*>(ra->list().at(0)) ;
    if (!rats) {
      cout << "ERROR: NLL is a RooAddition, but first element of addition is not a RooAbsOptTestStatistic!" << endl ;
      _fit_case = 0;
      cout << "It is a "<<ra->list().at(0)<<endl;
    } else {
      _fit_case = 1;
      cout << "NLL is a RooAddition (Case 2), first element is a RooAbsOptTestStatistic" << endl ;
      _initialise(rats);
    }
  }
}


void RooTaskSpec::_initialise (RooAbsOptTestStatistic* rats){
  // Check if nll is a AbsTestStatistic
  if (rats->numSimultaneous()==0){
    //    _set.add(_fill_task(0, rats));
    tasks.push_back(_fill_task(0, rats));
  }  else {
    for (Int_t i=0 ; i < rats->numSimultaneous() ; i++) {
      cout << "SimComponent #" << i << " = " ; 
      rats->simComponents()[i]->Print() ;
      RooAbsOptTestStatistic* comp = (RooAbsOptTestStatistic*) rats->simComponents()[i] ;
      tasks.push_back(_fill_task(i, comp));
    }
  }
}

RooTaskSpec::Task RooTaskSpec::_fill_task(Int_t n, RooAbsOptTestStatistic* rats){
  Bool_t b = rats->function().getAttribute("BinnedLikelihood") ;
  Task t;
  t.id = n;
  t.binned = b;
  if (b) {
    t.name = rats->function().GetName();
    t.entries = rats->data().numEntries();
    //   cout << "Binned Likelihood has probability model named " << rats->function().GetName()
    //	 << " and a binned dataset with " << rats->data().numEntries()
    //   << " bins with a total event count of " << rats->data().sumEntries() << endl ;
  } else {
    t.name = rats->function().GetName();
    t.entries = rats->data().numEntries();

    //    cout << "Unbinned likelihood has probability density model named " << rats->function().GetName()
    //    << " and an dataset with " << rats->data().numEntries() 
    //    << " events with a weight of " << rats->data().sumEntries() << endl ;
  } return t;
} 

