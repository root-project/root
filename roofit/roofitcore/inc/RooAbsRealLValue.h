/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsRealLValue.h,v 1.37 2007/07/13 21:50:24 wouter Exp $
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
#ifndef ROO_ABS_REAL_LVALUE
#define ROO_ABS_REAL_LVALUE

#include <cmath>
#include <cfloat>
#include <utility>
#include "TString.h"

#include "RooAbsReal.h"
#include "RooNumber.h"
#include "RooAbsLValue.h"
#include "RooAbsBinning.h"

class RooArgSet ;

class RooAbsRealLValue : public RooAbsReal, public RooAbsLValue {
public:
  // Constructors, assignment etc.
  inline RooAbsRealLValue() { }
  RooAbsRealLValue(const char *name, const char *title, const char *unit= "") ;
  RooAbsRealLValue(const RooAbsRealLValue& other, const char* name=nullptr);
  ~RooAbsRealLValue() override;

  // Parameter value and error accessors
  /// Set the current value of the object. Needs to be overridden by implementations.
  virtual void setVal(double value)=0;
  /// Set the current value of the object. The rangeName is ignored.
  /// Can be overridden by derived classes to e.g. check if the value fits in the given range.
  virtual void setVal(double value, const char* /*rangeName*/) {
    return setVal(value) ;
  }
  virtual RooAbsArg& operator=(double newValue);

  // Implementation of RooAbsLValue
  void setBin(Int_t ibin, const char* rangeName=nullptr) override ;
  Int_t getBin(const char* rangeName=nullptr) const override { return getBinning(rangeName).binNumber(getVal()) ; }
  Int_t numBins(const char* rangeName=nullptr) const override { return getBins(rangeName) ; }
  double getBinWidth(Int_t i, const char* rangeName=nullptr) const override { return getBinning(rangeName).binWidth(i) ; }
  double volume(const char* rangeName) const override { return getMax(rangeName)-getMin(rangeName) ; }
  void randomize(const char* rangeName=nullptr) override;

  const RooAbsBinning* getBinningPtr(const char* rangeName) const override { return &getBinning(rangeName) ; }
  Int_t getBin(const RooAbsBinning* ptr) const override { return ptr->binNumber(getVal()) ; }

  virtual void setBin(Int_t ibin, const RooAbsBinning& binning) ;
  virtual Int_t getBin(const RooAbsBinning& binning) const { return binning.binNumber(getVal()) ; }
  virtual Int_t numBins(const RooAbsBinning& binning) const { return binning.numBins() ; }
  virtual double getBinWidth(Int_t i, const RooAbsBinning& binning) const { return binning.binWidth(i) ; }
  virtual double volume(const RooAbsBinning& binning) const { return binning.highBound() - binning.lowBound() ; }
  virtual void randomize(const RooAbsBinning& binning) ;


  // Get fit range limits

  /// Retrive binning configuration with given name or default binning.
  virtual const RooAbsBinning& getBinning(const char* name=nullptr, bool verbose=true, bool createOnTheFly=false) const = 0 ;
  /// Retrive binning configuration with given name or default binning.
  virtual RooAbsBinning& getBinning(const char* name=nullptr, bool verbose=true, bool createOnTheFly=false) = 0 ;
  /// Check if binning with given name has been defined.
  virtual bool hasBinning(const char* name) const = 0 ;
  bool inRange(const char* name) const override ;
  /// Get number of bins of currently defined range.
  /// \param name Optionally, request number of bins for range with given name.
  virtual Int_t getBins(const char* name=nullptr) const { return getBinning(name).numBins(); }
  /// Get minimum of currently defined range.
  /// \param name Optionally, request minimum of range with given name.
  virtual double getMin(const char* name=nullptr) const { return getBinning(name).lowBound(); }
  /// Get maximum of currently defined range.
  /// \param name Optionally, request maximum of range with given name.
  virtual double getMax(const char* name=nullptr) const { return getBinning(name).highBound(); }
  /// Get low and high bound of the variable.
  /// \param name Optional range name. If not given, the default range will be used.
  /// \return A pair with [lowerBound, upperBound]
  std::pair<double, double> getRange(const char* name = 0) const {
    const auto& binning = getBinning(name);
    return {binning.lowBound(), binning.highBound()};
  }
  /// Check if variable has a lower bound.
  inline bool hasMin(const char* name=nullptr) const { return !RooNumber::isInfinite(getMin(name)); }
  /// Check if variable has an upper bound.
  inline bool hasMax(const char* name=nullptr) const { return !RooNumber::isInfinite(getMax(name)); }
  /// Check if variable has a binning with given name.
  bool hasRange(const char* name) const override { return hasBinning(name) ; }

  // Jacobian term management
  virtual bool isJacobianOK(const RooArgSet& depList) const ;
  virtual double jacobian() const { return 1 ; }

  inline bool isLValue() const override { return true; }

  // Test a value against our fit range
  bool inRange(double value, const char* rangeName, double* clippedValue=nullptr) const;
  void inRange(std::span<const double> values, std::string const& rangeName, std::vector<bool>& out) const;
  bool isValidReal(double value, bool printError=false) const override ;

  // Constant and Projected flags
  inline void setConstant(bool value= true) { setAttribute("Constant",value); setValueDirty() ; setShapeDirty() ; }

  // I/O streaming interface (machine readable)
  bool readFromStream(std::istream& is, bool compact, bool verbose=false) override ;
  void writeToStream(std::ostream& os, bool compact) const override ;

  // Printing interface (human readable)
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override ;


  // Build 1-dimensional plots
  RooPlot* frame(const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(),
                 const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),
                 const RooCmdArg& arg6=RooCmdArg::none(), const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;
  RooPlot *frame(const RooLinkedList& cmdList) const ;
  RooPlot *frame(double lo, double hi, Int_t nbins) const;
  RooPlot *frame(double lo, double hi) const;
  RooPlot *frame(Int_t nbins) const;
  RooPlot *frame() const;

  // Create empty 1,2, and 3D histograms from a list of 1-3 RooAbsReals
  TH1 *createHistogram(const char *name,
                       const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
                       const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
                       const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
                       const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;
  TH1 *createHistogram(const char *name, const RooLinkedList& cmdList) const ;

  TH1F *createHistogram(const char *name, const char *yAxisLabel) const ;
  TH1F *createHistogram(const char *name, const char *yAxisLabel, double xlo, double xhi, Int_t nBins) const ;
  TH1F *createHistogram(const char *name, const char *yAxisLabel, const RooAbsBinning& bins) const ;

  TH2F *createHistogram(const char *name, const RooAbsRealLValue &yvar, const char *zAxisLabel=nullptr,
         double* xlo=nullptr, double* xhi=nullptr, Int_t* nBins=nullptr) const ;
  TH2F *createHistogram(const char *name, const RooAbsRealLValue &yvar, const char *zAxisLabel, const RooAbsBinning** bins) const ;


  TH3F *createHistogram(const char *name, const RooAbsRealLValue &yvar, const RooAbsRealLValue &zvar,
         const char *tAxisLabel, double* xlo=nullptr, double* xhi=nullptr, Int_t* nBins=nullptr) const ;
  TH3F *createHistogram(const char *name, const RooAbsRealLValue &yvar, const RooAbsRealLValue &zvar, const char* tAxisLabel, const RooAbsBinning** bins) const ;

  static TH1* createHistogram(const char *name, RooArgList &vars, const char *tAxisLabel, double* xlo, double* xhi, Int_t* nBins) ;
  static TH1* createHistogram(const char *name, RooArgList &vars, const char *tAxisLabel, const RooAbsBinning** bins) ;

protected:

  virtual void setValFast(double value) { setVal(value) ; }

  bool fitRangeOKForPlotting() const ;
  void copyCache(const RooAbsArg* source, bool valueOnly=false, bool setValDirty=true) override ;

  ClassDefOverride(RooAbsRealLValue,1) // Abstract modifiable real-valued object
};

#endif
