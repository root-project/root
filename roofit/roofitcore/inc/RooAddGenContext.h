/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAddGenContext.h,v 1.12 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ADD_GEN_CONTEXT
#define ROO_ADD_GEN_CONTEXT

#include "RooAbsGenContext.h"
#include "RooArgSet.h"
#include "RooAddPdf.h"
#include "RooAddModel.h"
#include "RooGenContext.h"
#include "RooMsgService.h"

#include <memory>
#include <vector>

class AddCacheElem;
class RooDataSet;

class RooAddGenContext : public RooAbsGenContext {
public:
  RooAddGenContext(const RooAddPdf &model, const RooArgSet &vars, const RooDataSet *prototype=nullptr,
                   const RooArgSet* auxProto=nullptr, bool _verbose= false);
  RooAddGenContext(const RooAddModel &model, const RooArgSet &vars, const RooDataSet *prototype=nullptr,
                   const RooArgSet* auxProto=nullptr, bool _verbose= false);

  void setProtoDataOrder(Int_t* lut) override ;

  void attach(const RooArgSet& params) override ;

  void printMultiline(std::ostream &os, Int_t content, bool verbose=false, TString indent="") const override ;

  template<class Pdf_t>
  static std::unique_ptr<RooAbsGenContext> create(const Pdf_t &pdf, const RooArgSet &vars,
                                                  const RooDataSet *prototype,
                                                  const RooArgSet* auxProto, bool verbose);

protected:

  void initGenerator(const RooArgSet &theEvent) override;
  void generateEvent(RooArgSet &theEvent, Int_t remaining) override;
  void updateThresholds() ;

  RooAddGenContext(const RooAddGenContext& other) ;

  std::unique_ptr<RooArgSet> _vars ;
  std::unique_ptr<RooArgSet> _pdfSet ;              ///<  Set owned all nodes of internal clone of p.d.f
  RooAbsPdf *_pdf ;                 ///<  Pointer to cloned p.d.f
  std::vector<std::unique_ptr<RooAbsGenContext>> _gcList ;  ///<  List of component generator contexts
  Int_t  _nComp ;                   ///<  Number of PDF components
  std::vector<double> _coefThresh ;           ///<[_nComp] Array of coefficient thresholds
  bool _isModel ;                 ///< Are we generating from a RooAddPdf or a RooAddModel
  AddCacheElem* _pcache = nullptr;   ///<! RooAddPdf cache element

  ClassDefOverride(RooAddGenContext,0) // Specialized context for generating a dataset from a RooAddPdf
};


/// Returns a RooAddGenContext if possible, or, if the RooAddGenContext doesn't
/// support this particular RooAddPdf or RooAddModel because it has negative
/// coefficients, returns a generic RooGenContext.
///
/// Templated function to support both RooAddPdf and RooAddModel without code
/// duplication and without type checking at runtime.

template<class Pdf_t>
std::unique_ptr<RooAbsGenContext> RooAddGenContext::create(const Pdf_t &pdf, const RooArgSet &vars,
                                                           const RooDataSet *prototype,
                                                           const RooArgSet* auxProto, bool verbose)
{
  // Check if any coefficient is negative. We can use getVal() without the
  // normalization set, as normalization doesn't change the coefficients sign.
  auto hasNegativeCoefs = [&]() {
    for(auto * coef : static_range_cast<RooAbsReal*>(pdf._coefList)) {
      if(coef->getVal() < 0) return true;
    }
    return false;
  };

  // The RooAddGenContext doesn't support negative coefficients, so we create a
  // generic RooGenContext.
  if(hasNegativeCoefs()) {
    oocxcoutI(&pdf, Generation) << pdf.ClassName() << "::genContext():"
        << " using a generic generator context instead of the RooAddGenContext for the "
        << pdf.ClassName() << " \"" << pdf.GetName() <<  "\", because the pdf has negative coefficients." << std::endl;
    return std::make_unique<RooGenContext>(pdf, vars, prototype, auxProto, verbose);
  }

  return std::make_unique<RooAddGenContext>(pdf, vars, prototype, auxProto,verbose) ;
}

#endif
