/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooDataSet.h,v 1.59 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_DATA_SET
#define ROO_DATA_SET

class RooAbsRealLValue;
class RooCategory;
class RooDataHist;
class RooRealVar;

class TDirectory;

#include "RooAbsData.h"
#include "RooDirItem.h"

#include <ROOT/RConfig.hxx> // for R__DEPRECATED

#include <list>
#include <string_view>

class RooDataSet : public RooAbsData, public RooDirItem {
public:

  // Constructors, factory methods etc.
  RooDataSet() ;

  // Universal constructor
  RooDataSet(RooStringView name, RooStringView title, const RooArgSet& vars, const RooCmdArg& arg1={}, const RooCmdArg& arg2={},
             const RooCmdArg& arg3={}, const RooCmdArg& arg4={},const RooCmdArg& arg5={},
             const RooCmdArg& arg6={},const RooCmdArg& arg7={},const RooCmdArg& arg8={}) ;

    // Constructor for subset of existing dataset
  RooDataSet(RooStringView name, RooStringView title, RooDataSet *data, const RooArgSet& vars,
             const char *cuts=nullptr, const char* wgtVarName=nullptr)
#ifndef ROOFIT_BUILDS_ITSELF
  R__DEPRECATED(6,38, "Use RooAbsData::reduce(), or if you need to change the weight column, the universal constructor with the Import(), Cut(), and WeightVar() arguments.")
#endif
  ;
  RooDataSet(RooStringView name, RooStringView title, RooDataSet *data, const RooArgSet& vars,
             const RooFormulaVar& cutVar, const char* wgtVarName=nullptr)
  R__DEPRECATED(6,38, "Use RooAbsData::reduce(), or if you need to change the weight column, the universal constructor with the Import(), Cut(), and WeightVar() arguments.");

  RooDataSet(RooDataSet const & other, const char* newname=nullptr) ;
  TObject* Clone(const char* newname = "") const override {
    return new RooDataSet(*this, newname && newname[0] != '\0' ? newname : GetName());
  }
  ~RooDataSet() override ;

  RooFit::OwningPtr<RooAbsData> emptyClone(const char* newName=nullptr, const char* newTitle=nullptr, const RooArgSet* vars=nullptr, const char* wgtVarName=nullptr) const override;

  RooFit::OwningPtr<RooDataHist> binnedClone(const char* newName=nullptr, const char* newTitle=nullptr) const ;

  double sumEntries() const override;
  double sumEntries(const char* cutSpec, const char* cutRange=nullptr) const override;

  virtual RooPlot* plotOnXY(RooPlot* frame,
             const RooCmdArg& arg1={}, const RooCmdArg& arg2={},
             const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
             const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
             const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) const ;


  /// Read data from a text file and create a dataset from it.
  /// The possible options are: (D)ebug, (Q)uiet.
  static RooDataSet *read(const char *filename, const RooArgList &variables,
           const char *opts= "", const char* commonPath="",
           const char *indexCatName=nullptr) ;
  bool write(const char* filename) const;
  bool write(std::ostream & ofs) const;


  bool isWeighted() const override;
  bool isNonPoissonWeighted() const override;

  double weight() const override;
  /// Returns a pointer to the weight variable (if set).
  RooRealVar* weightVar() const { return _wgtVar; }
  double weightSquared() const override;
  void weightError(double& lo, double& hi,ErrorType etype=SumW2) const override;
  double weightError(ErrorType etype=SumW2) const override;

  const RooArgSet* get(Int_t index) const override;
  const RooArgSet* get() const override;

  std::span<const double> getWeightBatch(std::size_t first, std::size_t len, bool sumW2) const override;

  /// Add one ore more rows of data
  void add(const RooArgSet& row, double weight, double weightError);
  void add(const RooArgSet& row, double weight=1.0) override { add(row, weight, 0.0); }
  virtual void add(const RooArgSet& row, double weight, double weightErrorLo, double weightErrorHi);

  virtual void addFast(const RooArgSet& row, double weight=1.0, double weightError=0.0);

  void append(RooDataSet& data) ;
  bool merge(RooDataSet* data1, RooDataSet* data2=nullptr, RooDataSet* data3=nullptr,
           RooDataSet* data4=nullptr, RooDataSet* data5=nullptr, RooDataSet* data6=nullptr) ;
  bool merge(std::list<RooDataSet*> dsetList) ;

  virtual RooAbsArg* addColumn(RooAbsArg& var, bool adjustRange=true) ;
  virtual RooArgSet* addColumns(const RooArgList& varList) ;

  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override;
  void printArgs(std::ostream& os) const override;
  void printValue(std::ostream& os) const override;

  void SetName(const char *name) override;
  void SetNameTitle(const char *name, const char* title) override;

  static void cleanup();

  void convertToTreeStore() override;

protected:

  friend class RooProdGenContext ;

  void initialize(const char* wgtVarName) ;

  // Cache copy feature is not publicly accessible
  std::unique_ptr<RooAbsData> reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, const char* cutRange=nullptr,
                        std::size_t nStart=0, std::size_t nStop = std::numeric_limits<std::size_t>::max()) const override;

  RooArgSet _varsNoWgt;          ///< Vars without weight variable
  RooRealVar *_wgtVar = nullptr; ///< Pointer to weight variable (if set)

private:

  void loadValuesFromSlices(RooCategory &indexCat, std::map<std::string, RooAbsData *> const &slices,
                            const char *rangeName, RooFormulaVar const *cutVar, const char *cutSpec);

  unsigned short _errorMsgCount{0}; ///<! Counter to silence error messages when filling dataset.
  bool _doWeightErrorCheck{true};   ///<! When adding events with weights, check that weights can actually be stored.

  mutable std::unique_ptr<std::vector<double>> _sumW2Buffer; ///<! Buffer for sumW2 in case a batch of values is requested.

  ClassDefOverride(RooDataSet,2) // Unbinned data set
};

#endif
