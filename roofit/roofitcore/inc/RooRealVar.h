/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealVar.h,v 1.54 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_REAL_VAR
#define ROO_REAL_VAR

#include "RooAbsRealLValue.h"
#include "RooSharedProperties.h"

#include "TString.h"

#include <list>
#include <string>
#include <map>
#include <memory>
#include <unordered_map>


class RooArgSet ;
class RooErrorVar ;
class RooVectorDataStore ;
class RooExpensiveObjectCache ;
class RooRealVarSharedProperties;
namespace RooBatchCompute{
struct RunContext;
}

class RooRealVar : public RooAbsRealLValue {
public:
  // Constructors, assignment etc.
  RooRealVar() ;
  RooRealVar(const char *name, const char *title,
        double value, const char *unit= "") ;
  RooRealVar(const char *name, const char *title, double minValue,
      double maxValue, const char *unit= "");
  RooRealVar(const char *name, const char *title, double value,
      double minValue, double maxValue, const char *unit= "") ;
  RooRealVar(const RooRealVar& other, const char* name=0);
  RooRealVar& operator=(const RooRealVar& other);
  TObject* clone(const char* newname) const override { return new RooRealVar(*this,newname); }
  ~RooRealVar() override;

  // Parameter value and error accessors
  double getValV(const RooArgSet* nset=0) const override ;
  RooSpan<const double> getValues(RooBatchCompute::RunContext& inputData, const RooArgSet* = nullptr) const final;

  /// Returns how many times the value of this RooRealVar was reset.
  std::size_t valueResetCounter() const { return _valueResetCounter; }
  void setVal(double value) override;
  void setVal(double value, const char* rangeName) override;
  inline double getError() const { return _error>=0?_error:0. ; }
  inline bool hasError(bool allowZero=true) const { return allowZero ? (_error>=0) : (_error>0) ; }
  inline void setError(double value) { _error= value ; }
  inline void removeError() { _error = -1 ; }
  inline double getAsymErrorLo() const { return _asymErrLo<=0?_asymErrLo:0. ; }
  inline double getAsymErrorHi() const { return _asymErrHi>=0?_asymErrHi:0. ; }
  inline bool hasAsymError(bool allowZero=true) const { return allowZero ? ((_asymErrHi>=0 && _asymErrLo<=0)) :  ((_asymErrHi>0 && _asymErrLo<0)) ; }
  inline void removeAsymError() { _asymErrLo = 1 ; _asymErrHi = -1 ; }
  inline void setAsymError(double lo, double hi) { _asymErrLo = lo ; _asymErrHi = hi ; }
  inline double getErrorLo() const { return _asymErrLo<=0?_asymErrLo:-1*_error ; }
  inline double getErrorHi() const { return _asymErrHi>=0?_asymErrHi:_error ; }

  RooErrorVar* errorVar() const ;

  // Set/get finite fit range limits
  void setMin(const char* name, double value) ;
  void setMax(const char* name, double value) ;
  void setRange(const char* name, double min, double max) ;
  void setRange(const char* name, RooAbsReal& min, RooAbsReal& max) ;
  inline void setMin(double value) { setMin(0,value) ; }
  inline void setMax(double value) { setMax(0,value) ; }
  /// Set the limits of the default range.
  inline void setRange(double min, double max) { setRange(0,min,max) ; }
  /// Set parameterised limits of the default range. See setRange(const char*, RooAbsReal&, RooAbsReal&).
  inline void setRange(RooAbsReal& min, RooAbsReal& max) { setRange(0,min,max) ; }

  void setBins(Int_t nBins, const char* name=0);
  void setBinning(const RooAbsBinning& binning, const char* name=0) ;

  // RooAbsRealLValue implementation
  bool hasBinning(const char* name) const override ;
  const RooAbsBinning& getBinning(const char* name=0, bool verbose=true, bool createOnTheFly=false) const override ;
  RooAbsBinning& getBinning(const char* name=0, bool verbose=true, bool createOnTheFly=false) override ;
  std::list<std::string> getBinningNames() const override ;

  // Set infinite fit range limits
  /// Remove lower range limit for binning with given name. Empty name means default range.
  void removeMin(const char* name=0);
  /// Remove upper range limit for binning with given name. Empty name means default range.
  void removeMax(const char* name=0);
  /// Remove range limits for binning with given name. Empty name means default range.
  void removeRange(const char* name=0);

  // I/O streaming interface (machine readable)
  bool readFromStream(std::istream& is, bool compact, bool verbose=false) override ;
  void writeToStream(std::ostream& os, bool compact) const override ;

  // We implement a fundamental type of AbsArg that can be stored in a dataset
  inline bool isFundamental() const override { return true; }

  // Force to be a leaf-node of any expression tree, even if we have (shape) servers
  bool isDerived() const override {
    // Does value or shape of this arg depend on any other arg?
    return !_serverList.empty() || _proxyList.GetEntries()>0;
  }

  // Printing interface (human readable)
  void printValue(std::ostream& os) const override ;
  void printExtras(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override ;
  Int_t defaultPrintContents(Option_t* opt) const override ;


  TString* format(const RooCmdArg& formatArg) const ;
  TString* format(Int_t sigDigits, const char *options) const ;

  static void printScientific(bool flag=false) ;
  static void printSigDigits(Int_t ndig=5) ;

  using RooAbsRealLValue::operator= ;

  void deleteSharedProperties() ;

  void copyCacheFast(const RooRealVar& other, bool setValDirty=true) { _value = other._value ; if (setValDirty) setValueDirty() ; }

  static void cleanup() ;

  protected:

  static bool _printScientific ;
  static Int_t  _printSigDigits ;

  friend class RooAbsRealLValue ;
  void setValFast(double value) override { _value = value ; setValueDirty() ; }


  double evaluate() const override { return _value ; } // dummy because we overloaded getVal()
  void copyCache(const RooAbsArg* source, bool valueOnly=false, bool setValDirty=true) override ;
  void attachToTree(TTree& t, Int_t bufSize=32000) override ;
  void attachToVStore(RooVectorDataStore& vstore) override ;
  void fillTreeBranch(TTree& t) override ;

  double chopAt(double what, Int_t where) const ;

  double _error;      ///< Symmetric error associated with current value
  double _asymErrLo ; ///< Low side of asymmetric error associated with current value
  double _asymErrHi ; ///< High side of asymmetric error associated with current value
  std::unique_ptr<RooAbsBinning> _binning;
  std::unordered_map<std::string,std::unique_ptr<RooAbsBinning>> _altNonSharedBinning ; ///<! Non-shareable alternative binnings

  std::shared_ptr<RooRealVarSharedProperties> sharedProp() const;
  void installSharedProp(std::shared_ptr<RooRealVarSharedProperties>&& prop);

  void setExpensiveObjectCache(RooExpensiveObjectCache&) override { ; } ///< variables don't need caches
  static RooRealVarSharedProperties& _nullProp(); ///< Null property

  using SharedPropertiesMap = std::map<RooSharedProperties::UUID,std::weak_ptr<RooRealVarSharedProperties>>;

  static SharedPropertiesMap* sharedPropList(); ///< List of properties shared among clones of a variable

  std::shared_ptr<RooRealVarSharedProperties> _sharedProp; ///<! Shared binnings associated with this instance

  std::size_t _valueResetCounter = 0; ///<! How many times the value of this variable was reset

  ClassDefOverride(RooRealVar,8) // Real-valued variable
};




#endif
