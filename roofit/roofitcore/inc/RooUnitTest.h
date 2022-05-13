/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, NIKHEF, verkerke@nikhef.nl                         *
 *                                                                           *
 * Copyright (c) 2000-2011, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_UNIT_TEST
#define ROO_UNIT_TEST

#include "RooTable.h"
#include "RooWorkspace.h"
#include "RooFitResult.h"
#include "RooPlot.h"

#include "TNamed.h"
#include "TFile.h"
#include "TH1.h"

#include <list>
#include <string>
#include <utility>

/*
 * The tolerance for the curve test is put to 0.4 instead of 0.2 to take into
 * account the small variations in the values of the likelihood which can occur
 * in presence of a different treatment of floating point numbers.
 */

class RooUnitTest : public TNamed {
public:
  RooUnitTest(const char* name, TFile* refFile, bool writeRef, Int_t verbose, std::string const& batchMode="off") ;
  ~RooUnitTest() override ;

  void setDebug(bool flag) { _debug = flag ; }
  void setSilentMode() ;
  void clearSilentMode() ;
  void regPlot(RooPlot* frame, const char* refName) ;
  void regResult(RooFitResult* r, const char* refName) ;
  void regValue(double value, const char* refName) ;
  void regTable(RooTable* t, const char* refName) ;
  void regWS(RooWorkspace* ws, const char* refName) ;
  void regTH(TH1* h, const char* refName) ;
  RooWorkspace* getWS(const char* refName) ;
  bool runTest() ;
  bool runCompTests() ;
  bool areTHidentical(TH1* htest, TH1* href) ;

  virtual bool isTestAvailable() { return true ; }
  virtual bool testCode() = 0 ;

  virtual double htol() { return 5e-4 ; }  ///< histogram test tolerance (KS dist != prob)
#ifdef R__FAST_MATH
  virtual double ctol() { return 2e-3 ; }  ///< curve test tolerance
#else
  virtual double ctol() { return 4e-3 ; }  ///< curve test tolerance
#endif
  virtual double fptol() { return 1e-5 ; } ///< fit parameter test tolerance
  virtual double fctol() { return 1e-4 ; } ///< fit correlation test tolerance
  virtual double vtol() { return 1e-3 ; }  ///< value test tolerance

  static void setMemDir(TDirectory* memDir);

protected:

  static TDirectory* gMemDir ;

  TFile* _refFile ;
  bool _debug ;
  bool _write ;
  Int_t _verb ;
  std::string _batchMode="off";
   std::list<std::pair<RooPlot*, std::string> > _regPlots ;
   std::list<std::pair<RooFitResult*, std::string> > _regResults ;
   std::list<std::pair<double, std::string> > _regValues ;
   std::list<std::pair<RooTable*,std::string> > _regTables ;
   std::list<std::pair<RooWorkspace*,std::string> > _regWS ;
   std::list<std::pair<TH1*,std::string> > _regTH ;

  ClassDefOverride(RooUnitTest,0) ; // Abstract base class for RooFit/RooStats unit regression tests
} ;
#endif
