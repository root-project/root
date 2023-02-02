// @(#)root/roostats:$Id:  cranmer $
// Author: George Lewis, Kyle Cranmer
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROO_PARAMHISTFUNC
#define ROO_PARAMHISTFUNC

#include "RooAbsReal.h"
#include "RooListProxy.h"
#include "RooObjCacheManager.h"
#include "RooDataHist.h"

// Forward Declarations
class RooRealVar;
class RooWorkspace;

class ParamHistFunc : public RooAbsReal {
public:

  ParamHistFunc() ;
  ParamHistFunc(const char *name, const char *title, const RooArgList& vars, const RooArgList& paramSet );
  ParamHistFunc(const char *name, const char *title, const RooArgList& vars, const RooArgList& paramSet, const TH1* hist );

  ParamHistFunc(const ParamHistFunc& other, const char *name = nullptr);
  TObject* clone(const char* newname) const override { return new ParamHistFunc(*this, newname); }

  const RooArgList& paramList() const { return _paramSet ; }

  Int_t numBins() const { return _dataSet.numEntries(); } // Number of bins (called numEntries in RooDataHist)

  void setParamConst( Int_t, bool=true );
  void setConstant(bool constant);

  void setShape(TH1* shape);

  RooAbsReal& getParameter() const ;
  RooAbsReal& getParameter( Int_t masterIdx ) const ;

  const RooArgSet* get(Int_t masterIdx) const { return _dataSet.get( masterIdx ) ; }
  const RooArgSet* get(const RooArgSet& coord) const { return _dataSet.get( coord ) ; }

  double binVolume() const { return _dataSet.binVolume(); }

  bool forceAnalyticalInt(const RooAbsArg&) const override { return true ; }

  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet,const char* rangeName=nullptr) const override;
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=nullptr) const override;

  static RooArgList createParamSet(RooWorkspace& w, const std::string&, const RooArgList& Vars);
  static RooArgList createParamSet(RooWorkspace& w, const std::string&, const RooArgList& Vars, double, double);
  static RooArgList createParamSet(const std::string&, Int_t, double, double);

  std::list<double>* binBoundaries(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const override;
  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override;
  bool isBinnedDistribution(const RooArgSet& obs) const override { return _dataVars.overlaps(obs); }


protected:

  class CacheElem : public RooAbsCacheElement {
  public:
    RooArgList containedArgs(Action) override {
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
  mutable RooObjCacheManager _normIntMgr ; ///<! The integration cache manager

  RooListProxy _dataVars;             ///< The RooRealVars
  RooListProxy _paramSet ;            ///< interpolation parameters

  Int_t _numBins = 0;
  struct NumBins {
    NumBins() {}
    NumBins(int nx, int ny, int nz) : x{nx}, y{ny}, z{nz}, xy{x*y}, xz{x*z}, yz{y*z}, xyz{xy*z} {}
    int x = 0;
    int y = 0;
    int z = 0;
    int xy = 0;
    int xz = 0;
    int yz = 0;
    int xyz = 0;
  };
  mutable NumBins _numBinsPerDim; //!
  mutable RooDataHist _dataSet;

  Int_t getCurrentBin() const;
  Int_t addVarSet( const RooArgList& vars );
  Int_t addParamSet( const RooArgList& params );
  static Int_t GetNumBins( const RooArgSet& vars );
  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t size, RooFit::Detail::DataMap const&) const override;

private:
  static NumBins getNumBinsPerDim(RooArgSet const& vars);

  ClassDefOverride(ParamHistFunc, 7)
};

#endif
