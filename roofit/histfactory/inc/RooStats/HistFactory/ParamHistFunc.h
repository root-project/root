/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: ParamHistFunc.h,v 1.3 2007/05/11 09:11:30 verkerke Exp $
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

#ifndef ROO_PARAMHISTFUNC
#define ROO_PARAMHISTFUNC

#include <map>
#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "RooObjCacheManager.h"
#include "RooDataHist.h"

// Forward Declarations
class RooRealVar;
class RooArgList ;
class RooWorkspace;
class RooBinning;

class ParamHistFunc : public RooAbsReal {
public:

  ParamHistFunc() ;
  ParamHistFunc(const char *name, const char *title, const RooArgList& vars, const RooArgList& paramSet );
  ParamHistFunc(const char *name, const char *title, const RooArgList& vars, const RooArgList& paramSet, const TH1* hist );
  // Not yet fully implemented:
  //ParamHistFunc(const char *name, const char *title, const RooRealVar& var, const RooArgList& paramSet, const RooAbsReal& nominal );
  virtual ~ParamHistFunc() ;


  ParamHistFunc(const ParamHistFunc& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new ParamHistFunc(*this, newname); }

  //  void printMetaArgs(std::ostream& os) const ;

  const RooArgList& paramList() const { return _paramSet ; }


  Int_t numBins() const { return _dataSet.numEntries(); } // Number of bins (called numEntries in RooDataHist)

  void setParamConst( Int_t, Bool_t=kTRUE );

  RooRealVar& getParameter() const ;
  RooRealVar& getParameter( Int_t masterIdx ) const ;

  const RooArgSet* get(Int_t masterIdx) const { return _dataSet.get( masterIdx ) ; } 
  const RooArgSet* get(const RooArgSet& coord) const { return _dataSet.get( coord ) ; } 

  double binVolume() const { return _dataSet.binVolume(); }

  virtual Bool_t forceAnalyticalInt(const RooAbsArg&) const { return kTRUE ; }

  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet,const char* rangeName=0) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;



  static RooArgList createParamSet(RooWorkspace& w, const std::string&, const RooArgList& Vars);
  static RooArgList createParamSet(RooWorkspace& w, const std::string&, const RooArgList& Vars, Double_t, Double_t);
  static RooArgList createParamSet(const std::string&, Int_t, Double_t, Double_t);

  virtual std::list<Double_t>* binBoundaries(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const ;
  virtual std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const ; 
  virtual Bool_t isBinnedDistribution(const RooArgSet& /*obs*/) const {return kTRUE;}


protected:

  class CacheElem : public RooAbsCacheElement {
  public:
    CacheElem()  {} ;
    virtual ~CacheElem() {} ; 
    virtual RooArgList containedArgs(Action) { 
      RooArgList ret(_funcIntList) ; 
      ret.add(_lowIntList); 
      ret.add(_highIntList);
      return ret ; 
    }
    RooArgList _funcIntList ;
    RooArgList _lowIntList ;
    RooArgList _highIntList ;
    // will want std::vector<RooRealVar*> for low and high also
  } ;
  mutable RooObjCacheManager _normIntMgr ; // The integration cache manager

  // Turn into a RooListProxy
  //RooRealProxy _dataVar;       // The RooRealVar
  RooListProxy _dataVars;       // The RooRealVars
  RooListProxy _paramSet ;            // interpolation parameters
  //RooAbsBinning* _binning;  // Holds the binning of the dataVar (at construction time)

  Int_t _numBins;
  mutable std::map<Int_t, Int_t> _binMap;
  mutable RooDataHist _dataSet;

  // std::vector< Double_t > _nominalVals; // The nominal vals when gamma = 1.0 ( = 1.0 by default)
  RooArgList   _ownedList ;       // List of owned components

  Int_t getCurrentBin() const ;
  Int_t addVarSet( const RooArgList& vars );
  Int_t addParamSet( const RooArgList& params );
  static Int_t GetNumBins( const RooArgSet& vars );
  Double_t evaluate() const;

  ClassDef(ParamHistFunc,4) // Sum of RooAbsReal objects
};

#endif
