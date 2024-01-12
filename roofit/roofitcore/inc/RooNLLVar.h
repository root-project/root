/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNLLVar.h,v 1.10 2007/07/21 21:32:52 wouter Exp $
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
#ifndef ROO_NLL_VAR
#define ROO_NLL_VAR

// We can't print deprecation warnings when including headers in cling, because
// this will be done automatically anyway.
#ifdef __CLING__
#ifndef ROOFIT_BUILDS_ITSELF
// These warnings should only be suppressed when building ROOT itself!
#warning "Including RooNLLVar.h is deprecated, and this header will be removed in ROOT v6.34: please use RooAbsPdf::createNLL() to create likelihood objects"
#else
// If we are builting RooFit itself, this will serve as a reminder to actually
// remove this deprecate public header. Here is now this needs to be done:
//    1. Move this header file from inc/ to src/
//    2. Remove the LinkDef entry, ClassDefOverride, and ClassImpl macros for
//       this class
//    3. If there are are tests using this class in the test/ directory, change
//       the include to use a relative path the moved header file in the src/
//       directory, e.g. #include <RemovedInterface.h> becomes #include
//       "../src/RemovedInterface.h"
//    4. Remove this ifndef-else-endif block from the header
//    5. Remove the deprecation warning at the end of the class declaration
#include <RVersion.h>
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 34, 00)
#error "Please remove this deprecated public interface."
#endif
#endif
#endif

#include "RooAbsOptTestStatistic.h"
#include "RooCmdArg.h"
#include "RooAbsPdf.h"
#include <vector>
#include <utility>

class RooNLLVar : public RooAbsOptTestStatistic {
public:

  // Constructors, assignment etc
  RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
            bool extended,
            RooAbsTestStatistic::Configuration const& cfg=RooAbsTestStatistic::Configuration{});

  RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
            const RooArgSet& projDeps, bool extended = false,
            RooAbsTestStatistic::Configuration const& cfg=RooAbsTestStatistic::Configuration{});

  RooNLLVar(const RooNLLVar& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooNLLVar(*this,newname); }

  RooAbsTestStatistic* create(const char *name, const char *title, RooAbsReal& pdf, RooAbsData& adata,
                                      const RooArgSet& projDeps, RooAbsTestStatistic::Configuration const& cfg) override;

  ~RooNLLVar() override;

  void applyWeightSquared(bool flag) override;

  double defaultErrorLevel() const override { return 0.5 ; }

  void enableBinOffsetting(bool on = true);

  using ComputeResult = std::pair<ROOT::Math::KahanSum<double>, double>;

  static RooNLLVar::ComputeResult computeScalarFunc(const RooAbsPdf *pdfClone, RooAbsData *dataClone, RooArgSet *normSet,
                                                bool weightSq, std::size_t stepSize, std::size_t firstEvent,
                                                std::size_t lastEvent, RooAbsPdf const* offsetPdf = nullptr);

  bool setDataSlave(RooAbsData& data, bool cloneData=true, bool ownNewDataAnyway=false) override;

protected:

  bool processEmptyDataSets() const override { return _extended ; }
  double evaluatePartition(std::size_t firstEvent, std::size_t lastEvent, std::size_t stepSize) const override;

  static RooArgSet _emptySet ; // Supports named argument constructor

private:
  ComputeResult computeScalar(std::size_t stepSize, std::size_t firstEvent, std::size_t lastEvent) const;

  bool _extended{false};
  bool _doBinOffset{false};
  bool _weightSq{false}; ///< Apply weights squared?
  mutable bool _first{true}; ///<!
  ROOT::Math::KahanSum<double> _offsetSaveW2{0.0}; ///<!

  mutable std::vector<double> _binw ; ///<!
  mutable RooAbsPdf* _binnedPdf{nullptr}; ///<!
  std::unique_ptr<RooAbsPdf> _offsetPdf; ///<! An optional per-bin likelihood offset

  ClassDefOverride(RooNLLVar,0) // Function representing (extended) -log(L) of p.d.f and dataset

#ifndef ROOFIT_BUILDS_ITSELF
} R__DEPRECATED(6,34, "Please use RooAbsPdf::createNLL() to create likelihood objects");
#else
};
#endif

#endif

