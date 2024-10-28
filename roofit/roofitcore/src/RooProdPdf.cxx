/*****************************************************************************
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
\file RooProdPdf.cxx
\class RooProdPdf
\ingroup Roofitcore

Efficient implementation of a product of PDFs of the form
\f[ \prod_{i=1}^{N} \mathrm{PDF}_i (x, \ldots) \f]

PDFs may share observables. If that is the case any irreducible subset
of PDFs that share observables will be normalised with explicit numeric
integration as any built-in normalisation will no longer be valid.

Alternatively, products using conditional PDFs can be defined, *e.g.*

\f[ F(x|y) \cdot G(y), \f]

meaning a PDF \f$ F(x) \f$ **given** \f$ y \f$ and a PDF \f$ G(y) \f$.
In this construction, \f$ F \f$ is only
normalised w.r.t \f$ x\f$, and \f$ G \f$ is normalised w.r.t \f$ y \f$. The product in this construction
is properly normalised.

If exactly one of the component PDFs supports extended likelihood fits, the
product will also be usable in extended mode, returning the number of expected
events from the extendable component PDF. The extendable component does not
have to appear in any specific place in the list.
**/

#include "RooProdPdf.h"
#include "RooBatchCompute.h"
#include "RooRealProxy.h"
#include "RooProdGenContext.h"
#include "RooGenProdProj.h"
#include "RooProduct.h"
#include "RooNameReg.h"
#include "RooMsgService.h"
#include "RooFormulaVar.h"
#include "RooRealVar.h"
#include "RooAddition.h"
#include "RooGlobalFunc.h"
#include "RooConstVar.h"
#include "RooWorkspace.h"
#include "RooRangeBoolean.h"
#include "RooCustomizer.h"
#include "RooRealIntegral.h"
#include "RooTrace.h"
#include "RooFitImplHelpers.h"
#include "strtok.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>

#ifndef _WIN32
#include <strings.h>
#endif

using std::endl, std::string, std::vector, std::list, std::ostream, std::map, std::ostringstream;

ClassImp(RooFit::Detail::RooFixedProdPdf);
ClassImp(RooProdPdf);


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooProdPdf::RooProdPdf() :
  _cacheMgr(this,10)
{
  // Default constructor
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor with 2 PDFs (most frequent use case).
///
/// The optional cutOff parameter can be used as a speed optimization if
/// one or more of the PDF have sizable regions with very small values,
/// which would pull the entire product of PDFs to zero in those regions.
///
/// After each PDF multiplication, the running product is compared with
/// the cutOff parameter. If the running product is smaller than the
/// cutOff value, the product series is terminated and remaining PDFs
/// are not evaluated.
///
/// There is no magic value of the cutOff, the user should experiment
/// to find the appropriate balance between speed and precision.
/// If a cutoff is specified, the PDFs most likely to be small should
/// be put first in the product. The default cutOff value is zero.
///

RooProdPdf::RooProdPdf(const char *name, const char *title,
             RooAbsPdf& pdf1, RooAbsPdf& pdf2, double cutOff) :
  RooAbsPdf(name,title),
  _cacheMgr(this,10),
  _cutOff(cutOff),
  _pdfList("!pdfs","List of PDFs",this)
{
  _pdfList.add(pdf1) ;
  _pdfNSetList.emplace_back(std::make_unique<RooArgSet>("nset")) ;
  if (pdf1.canBeExtended()) {
    _extendedIndex = _pdfList.index(&pdf1) ;
  }

  _pdfList.add(pdf2) ;
  _pdfNSetList.emplace_back(std::make_unique<RooArgSet>("nset")) ;

  if (pdf2.canBeExtended()) {
    if (_extendedIndex>=0) {
      // Protect against multiple extended terms
      coutW(InputArguments) << "RooProdPdf::RooProdPdf(" << GetName()
             << ") multiple components with extended terms detected,"
             << " product will not be extendible." << endl ;
      _extendedIndex=-1 ;
    } else {
      _extendedIndex=_pdfList.index(&pdf2) ;
    }
  }
  TRACE_CREATE;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor from a list of PDFs.
///
/// The optional cutOff parameter can be used as a speed optimization if
/// one or more of the PDF have sizable regions with very small values,
/// which would pull the entire product of PDFs to zero in those regions.
///
/// After each PDF multiplication, the running product is compared with
/// the cutOff parameter. If the running product is smaller than the
/// cutOff value, the product series is terminated and remaining PDFs
/// are not evaluated.
///
/// There is no magic value of the cutOff, the user should experiment
/// to find the appropriate balance between speed and precision.
/// If a cutoff is specified, the PDFs most likely to be small should
/// be put first in the product. The default cutOff value is zero.

RooProdPdf::RooProdPdf(const char* name, const char* title, const RooArgList& inPdfList, double cutOff) :
  RooAbsPdf(name,title),
  _cacheMgr(this,10),
  _cutOff(cutOff),
  _pdfList("!pdfs","List of PDFs",this)
{
  addPdfs(inPdfList);
  TRACE_CREATE;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor from named argument list.
/// \param[in] name Name used by RooFit
/// \param[in] title Title used for plotting
/// \param[in] fullPdfSet Set of "regular" PDFs that are normalised over all their observables
/// \param[in] arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8 Optional arguments according to table below.
///
/// <table>
/// <tr><th> Argument                 <th> Description
/// <tr><td> `Conditional(pdfSet,depSet,depsAreCond=false)` <td> Add PDF to product with condition that it
/// only be normalized over specified observables. Any remaining observables will be conditional observables.
/// (Setting `depsAreCond` to true inverts this, so the observables in depSet will be the conditional observables.)
/// </table>
///
/// For example, given a PDF \f$ F(x,y) \f$ and \f$ G(y) \f$,
///
/// `RooProdPdf("P", "P", G, Conditional(F,x))` will construct a 2-dimensional PDF as follows:
/// \f[
///     P(x,y) = \frac{G(y)}{\int_y G(y)} \cdot \frac{F(x,y)}{\int_x F(x,y)},
/// \f]
///
/// which is a well normalised and properly defined PDF, but different from
/// \f[
///     P'(x,y) = \frac{F(x,y) \cdot G(y)}{\int_x\int_y F(x,y) \cdot G(y)}.
/// \f]
///
/// In the former case, the \f$ y \f$ distribution of \f$ P \f$ is identical to that of \f$ G \f$, while
/// \f$ F \f$ only is used to determine the correlation between \f$ X \f$ and \f$ Y \f$. In the latter
/// case, the \f$ Y \f$ distribution is defined by the product of \f$ F \f$ and \f$ G \f$.
///
/// This \f$ P(x,y) \f$ construction is analogous to generating events from \f$ F(x,y) \f$ with
/// a prototype dataset sampled from \f$ G(y) \f$.

RooProdPdf::RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet,
             const RooCmdArg& arg1, const RooCmdArg& arg2,
             const RooCmdArg& arg3, const RooCmdArg& arg4,
             const RooCmdArg& arg5, const RooCmdArg& arg6,
             const RooCmdArg& arg7, const RooCmdArg& arg8) :
  RooAbsPdf(name,title),
  _cacheMgr(this,10),
  _pdfList("!pdfs","List of PDFs",this)
{
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;

  initializeFromCmdArgList(fullPdfSet,l) ;
  TRACE_CREATE;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor from named argument list

RooProdPdf::RooProdPdf(const char* name, const char* title,
             const RooCmdArg& arg1, const RooCmdArg& arg2,
             const RooCmdArg& arg3, const RooCmdArg& arg4,
             const RooCmdArg& arg5, const RooCmdArg& arg6,
             const RooCmdArg& arg7, const RooCmdArg& arg8) :
  RooAbsPdf(name,title),
  _cacheMgr(this,10),
  _pdfList("!pdfList","List of PDFs",this)
{
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;

  initializeFromCmdArgList(RooArgSet(),l) ;
  TRACE_CREATE;
}



////////////////////////////////////////////////////////////////////////////////
/// Internal constructor from list of named arguments

RooProdPdf::RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet, const RooLinkedList& cmdArgList) :
  RooAbsPdf(name,title),
  _cacheMgr(this,10),
  _pdfList("!pdfs","List of PDFs",this)
{
  initializeFromCmdArgList(fullPdfSet, cmdArgList) ;
  TRACE_CREATE;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooProdPdf::RooProdPdf(const RooProdPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _cacheMgr(other._cacheMgr,this),
  _genCode(other._genCode),
  _cutOff(other._cutOff),
  _pdfList("!pdfs",this,other._pdfList),
  _extendedIndex(other._extendedIndex),
  _useDefaultGen(other._useDefaultGen),
  _refRangeName(other._refRangeName),
  _selfNorm(other._selfNorm),
  _defNormSet(other._defNormSet)
{
  // Clone contents of normalizarion set list
  for(auto const& nset : other._pdfNSetList) {
    _pdfNSetList.emplace_back(std::make_unique<RooArgSet>(nset->GetName()));
    nset->snapshot(*_pdfNSetList.back());
  }
  TRACE_CREATE;
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize RooProdPdf configuration from given list of RooCmdArg configuration arguments
/// and set of 'regular' p.d.f.s in product

void RooProdPdf::initializeFromCmdArgList(const RooArgSet& fullPdfSet, const RooLinkedList& l)
{
  Int_t numExtended(0) ;

  // Process set of full PDFS
  for(auto const* pdf : static_range_cast<RooAbsPdf*>(fullPdfSet)) {
    _pdfList.add(*pdf) ;
    _pdfNSetList.emplace_back(std::make_unique<RooArgSet>("nset")) ;

    if (pdf->canBeExtended()) {
      _extendedIndex = _pdfList.index(pdf) ;
      numExtended++ ;
    }

  }

  // Process list of conditional PDFs
  for(auto * carg : static_range_cast<RooCmdArg*>(l)) {

    if (0 == strcmp(carg->GetName(), "Conditional")) {

      Int_t argType = carg->getInt(0) ;
      auto pdfSet = static_cast<RooArgSet const*>(carg->getSet(0));
      auto normSet = static_cast<RooArgSet const*>(carg->getSet(1));

      for(auto * thePdf : static_range_cast<RooAbsPdf*>(*pdfSet)) {
        _pdfList.add(*thePdf) ;

        _pdfNSetList.emplace_back(std::make_unique<RooArgSet>(0 == argType ? "nset" : "cset"));
        normSet->snapshot(*_pdfNSetList.back());

        if (thePdf->canBeExtended()) {
          _extendedIndex = _pdfList.index(thePdf) ;
          numExtended++ ;
        }

      }

    } else if (0 != strlen(carg->GetName())) {
      coutW(InputArguments) << "Unknown arg: " << carg->GetName() << endl ;
    }
  }

  // Protect against multiple extended terms
  if (numExtended>1) {
    coutW(InputArguments) << "RooProdPdf::RooProdPdf(" << GetName()
           << ") WARNING: multiple components with extended terms detected,"
           << " product will not be extendible." << endl ;
    _extendedIndex = -1 ;
  }


}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooProdPdf::~RooProdPdf()
{
  TRACE_DESTROY;
}


RooProdPdf::CacheElem* RooProdPdf::getCacheElem(RooArgSet const* nset) const {
  int code ;
  auto cache = static_cast<CacheElem*>(_cacheMgr.getObj(nset, nullptr, &code)) ;

  // If cache doesn't have our configuration, recalculate here
  if (!cache) {
    code = getPartIntList(nset, nullptr) ;
    cache = static_cast<CacheElem*>(_cacheMgr.getObj(nset, nullptr, &code)) ;
  }
  return cache;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate current value of object

double RooProdPdf::evaluate() const
{
  return calculate(*getCacheElem(_normSet)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate running product of pdfs terms, using the supplied
/// normalization set in 'normSetList' for each component

double RooProdPdf::calculate(const RooProdPdf::CacheElem& cache, bool /*verbose*/) const
{
  if (cache._isRearranged) {
    if (dologD(Eval)) {
      cxcoutD(Eval) << "RooProdPdf::calculate(" << GetName() << ") rearranged product calculation"
                    << " calculate: num = " << cache._rearrangedNum->GetName() << " = " << cache._rearrangedNum->getVal() << endl ;
//       cache._rearrangedNum->printComponentTree("",0,5) ;
      cxcoutD(Eval) << "calculate: den = " << cache._rearrangedDen->GetName() << " = " << cache._rearrangedDen->getVal() << endl ;
//       cache._rearrangedDen->printComponentTree("",0,5) ;
    }

    return cache._rearrangedNum->getVal() / cache._rearrangedDen->getVal();
  } else {

    double value = 1.0;
    assert(cache._normList.size() == cache._partList.size());
    for (std::size_t i = 0; i < cache._partList.size(); ++i) {
      const auto& partInt = static_cast<const RooAbsReal&>(cache._partList[i]);
      const auto normSet = cache._normList[i].get();

      const double piVal = partInt.getVal(!normSet->empty() ? normSet : nullptr);
      value *= piVal ;
      if (value <= _cutOff) break;
    }

    return value ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluate product of PDFs in batch mode.
void RooProdPdf::doEvalImpl(RooAbsArg const *caller, const RooProdPdf::CacheElem &cache, RooFit::EvalContext &ctx) const
{
   if (cache._isRearranged) {
      auto numerator = ctx.at(cache._rearrangedNum.get());
      auto denominator = ctx.at(cache._rearrangedDen.get());
      RooBatchCompute::compute(ctx.config(caller), RooBatchCompute::Ratio, ctx.output(), {numerator, denominator});
   } else {
      std::vector<std::span<const double>> factors;
      factors.reserve(cache._partList.size());
      for (const RooAbsArg *i : cache._partList) {
         auto span = ctx.at(i);
         factors.push_back(span);
      }
      std::array<double, 1> special{static_cast<double>(factors.size())};
      RooBatchCompute::compute(ctx.config(caller), RooBatchCompute::ProdPdf, ctx.output(), factors, special);
   }
}

namespace {

template<class T>
void eraseNullptrs(std::vector<T*>& v) {
  v.erase(std::remove_if(v.begin(), v.end(), [](T* x){ return x == nullptr; } ), v.end());
}

void removeCommon(std::vector<RooAbsArg*> &v, std::span<RooAbsArg * const> other) {

  for (auto const& arg : other) {
    auto namePtrMatch = [&arg](const RooAbsArg* elm) {
      return elm != nullptr && elm->namePtr() == arg->namePtr();
    };

    auto found = std::find_if(v.begin(), v.end(), namePtrMatch);
    if(found != v.end()) {
      *found = nullptr;
    }
  }
  eraseNullptrs(v);
}

void addCommon(std::vector<RooAbsArg*> &v, std::vector<RooAbsArg*> const& o1, std::vector<RooAbsArg*> const& o2) {

  for (auto const& arg : o1) {
    auto namePtrMatch = [&arg](const RooAbsArg* elm) {
      return elm->namePtr() == arg->namePtr();
    };

    if(std::find_if(o2.begin(), o2.end(), namePtrMatch) != o2.end()) {
      v.push_back(arg);
    }
  }
}

}


////////////////////////////////////////////////////////////////////////////////
/// Factorize product in irreducible terms for given choice of integration/normalization

void RooProdPdf::factorizeProduct(const RooArgSet& normSet, const RooArgSet& intSet,
              RooLinkedList& termList, RooLinkedList& normList,
              RooLinkedList& impDepList, RooLinkedList& crossDepList,
              RooLinkedList& intList) const
{
  // List of all term dependents: normalization and imported
  std::vector<RooArgSet> depAllList;
  std::vector<RooArgSet> depIntNoNormList;

  // Setup lists for factorization terms and their dependents
  RooArgSet* term(nullptr);
  RooArgSet* termNormDeps(nullptr);
  RooArgSet* termIntDeps(nullptr);
  RooArgSet* termIntNoNormDeps(nullptr);

  std::vector<RooAbsArg*> pdfIntNoNormDeps;
  std::vector<RooAbsArg*> pdfIntSet;
  std::vector<RooAbsArg*> pdfNSet;
  std::vector<RooAbsArg*> pdfCSet;
  std::vector<RooAbsArg*> pdfNormDeps; // Dependents to be normalized for the PDF
  std::vector<RooAbsArg*> pdfAllDeps; // All dependents of this PDF

  // Loop over the PDFs
  for(std::size_t iPdf = 0; iPdf < _pdfList.size(); ++iPdf) {
    RooAbsPdf& pdf = static_cast<RooAbsPdf&>(_pdfList[iPdf]);
    RooArgSet& pdfNSetOrig = *_pdfNSetList[iPdf];

    pdfNSet.clear();
    pdfCSet.clear();

    // Make iterator over tree leaf node list to get the observables.
    // This code is borrowed from RooAbsPdf::getObservables().
    // RooAbsArg::treeNodeServer list is relatively expensive, so we only do it
    // once and use it in a lambda function.
    RooArgSet pdfLeafList("leafNodeServerList") ;
    pdf.treeNodeServerList(&pdfLeafList,nullptr,false,true,true) ;
    auto getObservablesOfCurrentPdf = [&pdfLeafList](
            std::vector<RooAbsArg*> & out,
            const RooArgSet& dataList) {
      for (const auto arg : pdfLeafList) {
        if (arg->dependsOnValue(dataList) && arg->isLValue()) {
          out.push_back(arg) ;
        }
      }
    };

    // Reduce pdfNSet to actual dependents
    if (0 == strcmp("cset", pdfNSetOrig.GetName())) {
      getObservablesOfCurrentPdf(pdfNSet, normSet);
      removeCommon(pdfNSet, pdfNSetOrig.get());
      pdfCSet = pdfNSetOrig.get();
    } else {
      // Interpret at NSet
      getObservablesOfCurrentPdf(pdfNSet, pdfNSetOrig);
    }


    pdfNormDeps.clear();
    pdfAllDeps.clear();

    // Make list of all dependents of this PDF
    getObservablesOfCurrentPdf(pdfAllDeps, normSet);


//     cout << GetName() << ": pdf = " << pdf->GetName() << " pdfAllDeps = " << pdfAllDeps << " pdfNSet = " << *pdfNSet << " pdfCSet = " << *pdfCSet << endl;

    // Make list of normalization dependents for this PDF;
    if (!pdfNSet.empty()) {
      // PDF is conditional
      addCommon(pdfNormDeps, pdfAllDeps, pdfNSet);
    } else {
      // PDF is regular
      pdfNormDeps = pdfAllDeps;
    }

//     cout << GetName() << ": pdfNormDeps for " << pdf->GetName() << " = " << pdfNormDeps << endl;

    pdfIntSet.clear();
    getObservablesOfCurrentPdf(pdfIntSet, intSet) ;

    // WVE if we have no norm deps, conditional observables should be taken out of pdfIntSet
    if (pdfNormDeps.empty() && !pdfCSet.empty()) {
      removeCommon(pdfIntSet, pdfCSet);
//       cout << GetName() << ": have no norm deps, removing conditional observables from intset" << endl;
    }

    pdfIntNoNormDeps.clear();
    pdfIntNoNormDeps = pdfIntSet;
    removeCommon(pdfIntNoNormDeps, pdfNormDeps);

//     cout << GetName() << ": pdf = " << pdf->GetName() << " intset = " << *pdfIntSet << " pdfIntNoNormDeps = " << pdfIntNoNormDeps << endl;

    // Check if this PDF has dependents overlapping with one of the existing terms
    bool done = false;
    int j = 0;
    auto lIter = termList.begin();
    auto ldIter = normList.begin();
    for(;lIter != termList.end(); (++lIter, ++ldIter, ++j)) {
      termNormDeps = static_cast<RooArgSet*>(*ldIter);
      term = static_cast<RooArgSet*>(*lIter);
      // PDF should be added to existing term if
      // 1) It has overlapping normalization dependents with any other PDF in existing term
      // 2) It has overlapping dependents of any class for which integration is requested
      // 3) If normalization happens over multiple ranges, and those ranges are both defined
      //    in either observable

      bool normOverlap = termNormDeps->overlaps(pdfNormDeps.begin(), pdfNormDeps.end());
      //bool intOverlap =  pdfIntSet->overlaps(*termAllDeps);

      if (normOverlap) {
//    cout << GetName() << ": this term overlaps with term " << (*term) << " in normalization observables" << endl;

   term->add(pdf);
   termNormDeps->add(pdfNormDeps.begin(), pdfNormDeps.end(), false);
   depAllList[j].add(pdfAllDeps.begin(), pdfAllDeps.end(), false);
   if (termIntDeps) {
     termIntDeps->add(pdfIntSet.begin(), pdfIntSet.end(), false);
   }
   if (termIntNoNormDeps) {
     termIntNoNormDeps->add(pdfIntNoNormDeps.begin(), pdfIntNoNormDeps.end(), false);
   }
   termIntNoNormDeps->add(pdfIntNoNormDeps.begin(), pdfIntNoNormDeps.end(), false);
   done = true;
      }
    }

    // If not, create a new term
    if (!done) {
      if (!(pdfNormDeps.empty() && pdfAllDeps.empty() &&
       pdfIntSet.empty()) || normSet.empty()) {
   term = new RooArgSet("term");
   termNormDeps = new RooArgSet("termNormDeps");
   depAllList.emplace_back(pdfAllDeps.begin(), pdfAllDeps.end(), "termAllDeps");
   termIntDeps = new RooArgSet(pdfIntSet.begin(), pdfIntSet.end(), "termIntDeps");
   depIntNoNormList.emplace_back(pdfIntNoNormDeps.begin(), pdfIntNoNormDeps.end(), "termIntNoNormDeps");
   termIntNoNormDeps = &depIntNoNormList.back();

   term->add(pdf);
   termNormDeps->add(pdfNormDeps.begin(), pdfNormDeps.end(), false);

   termList.Add(term);
   normList.Add(termNormDeps);
   intList.Add(termIntDeps);
      }
    }

  }

  // Loop over list of terms again to determine 'imported' observables
  int i = 0;
  RooArgSet *normDeps;
  auto lIter = termList.begin();
  auto ldIter = normList.begin();
  for(;lIter != termList.end(); (++lIter, ++ldIter, ++i)) {
    normDeps = static_cast<RooArgSet*>(*ldIter);
    term = static_cast<RooArgSet*>(*lIter);
    // Make list of wholly imported dependents
    RooArgSet impDeps(depAllList[i]);
    impDeps.remove(*normDeps, true, true);
    auto snap = new RooArgSet;
    impDeps.snapshot(*snap);
    impDepList.Add(snap);
//     cout << GetName() << ": list of imported dependents for term " << (*term) << " set to " << impDeps << endl ;

    // Make list of cross dependents (term is self contained for these dependents,
    // but components import dependents from other components)
    auto crossDeps = std::unique_ptr<RooAbsCollection>{depIntNoNormList[i].selectCommon(*normDeps)};
    snap = new RooArgSet;
    crossDeps->snapshot(*snap);
    crossDepList.Add(snap);
//     cout << GetName() << ": list of cross dependents for term " << (*term) << " set to " << *crossDeps << endl ;
  }

  return;
}




////////////////////////////////////////////////////////////////////////////////
/// Return list of (partial) integrals of product terms for integration
/// of p.d.f over observables iset while normalization over observables nset.
/// Also return list of normalization sets to be used to evaluate
/// each component in the list correctly.

Int_t RooProdPdf::getPartIntList(const RooArgSet* nset, const RooArgSet* iset, const char* isetRangeName) const
{
  // Check if this configuration was created before
  Int_t sterileIdx(-1);

  if (static_cast<CacheElem*>(_cacheMgr.getObj(nset,iset,&sterileIdx,isetRangeName))) {
    return _cacheMgr.lastIndex();
  }

  std::unique_ptr<CacheElem> cache = createCacheElem(nset, iset, isetRangeName);

  // Store the partial integral list and return the assigned code
  return _cacheMgr.setObj(nset, iset, cache.release(), RooNameReg::ptr(isetRangeName));
}



std::unique_ptr<RooProdPdf::CacheElem> RooProdPdf::createCacheElem(const RooArgSet* nset,
                                                       const RooArgSet* iset,
                                                       const char* isetRangeName) const
{
//    cout << "   FOLKERT::RooProdPdf::getPartIntList(" << GetName() <<")  nset = " << (nset?*nset:RooArgSet()) << endl
//         << "   _normRange = " << _normRange << endl
//         << "   iset = " << (iset?*iset:RooArgSet()) << endl
//         << "   isetRangeName = " << (isetRangeName?isetRangeName:"<null>") << endl ;

  // Create containers for partial integral components to be generated
  auto cache = std::make_unique<CacheElem>();

  // Factorize the product in irreducible terms for this nset
  RooLinkedList terms;
  RooLinkedList norms;
  RooLinkedList imp;
  RooLinkedList ints;
  RooLinkedList cross;
  //   cout << "RooProdPdf::getPIL -- now calling factorizeProduct()" << endl ;


  // Normalization set used for factorization
  RooArgSet factNset(nset ? (*nset) : _defNormSet);
//   cout << GetName() << "factNset = " << factNset << endl ;

  factorizeProduct(factNset, iset ? (*iset) : RooArgSet(), terms, norms, imp, cross, ints);

  RooArgSet *norm;
  RooArgSet *integ;
  RooArgSet *xdeps;
  RooArgSet *imps;

  // Group irriducible terms that need to be (partially) integrated together
  std::list<std::vector<RooArgSet*>> groupedList;
  RooArgSet outerIntDeps;
//   cout << "RooProdPdf::getPIL -- now calling groupProductTerms()" << endl;
  groupProductTerms(groupedList, outerIntDeps, terms, norms, imp, ints, cross);

  // Loop over groups
//   cout<<"FK: pdf("<<GetName()<<") Starting selecting F(x|y)!"<<endl;
  // Find groups of type F(x|y), i.e. termImpSet!=0, construct ratio object
  std::map<std::string, RooArgSet> ratioTerms;
  for (auto const& group : groupedList) {
    if (1 == group.size()) {
//       cout<<"FK: Starting Single Term"<<endl;

      RooArgSet* term = group[0];

      Int_t termIdx = terms.IndexOf(term);
      norm=static_cast<RooArgSet*>(norms.At(termIdx));
      imps=static_cast<RooArgSet*>(imp.At(termIdx));
      RooArgSet termNSet(*norm);
      RooArgSet termImpSet(*imps);

      //       cout<<"FK: termImpSet.size()  = "<<termImpSet.size()<< " " << termImpSet << endl;
      //       cout<<"FK: _refRangeName = "<<_refRangeName<<endl;

      if (!termImpSet.empty() && nullptr != _refRangeName) {

//    cout << "WVE now here" << endl;

   // WVE we can skip this if the ref range is equal to the normalization range
   bool rangeIdentical(true);
//    cout << "_normRange = " << _normRange << " _refRangeName = " << RooNameReg::str(_refRangeName) << endl ;
   for (auto const* normObs : static_range_cast<RooRealVar*>(termNSet)) {
     //FK: Here the refRange should be compared to _normRange, if it's set, and to the normObs range if it's not set
     if (_normRange.Length() > 0) {
       if (normObs->getMin(_normRange.Data()) != normObs->getMin(RooNameReg::str(_refRangeName))) rangeIdentical = false;
       if (normObs->getMax(_normRange.Data()) != normObs->getMax(RooNameReg::str(_refRangeName))) rangeIdentical = false;
     }
     else{
       if (normObs->getMin() != normObs->getMin(RooNameReg::str(_refRangeName))) rangeIdentical = false;
       if (normObs->getMax() != normObs->getMax(RooNameReg::str(_refRangeName))) rangeIdentical = false;
     }
   }
//    cout<<"FK: rangeIdentical Single = "<<(rangeIdentical ? 'T':'F')<<endl;
   // coverity[CONSTANT_EXPRESSION_RESULT]
   // LM : avoid making integral ratio if range is the same. Why was not included ??? (same at line 857)
   if (!rangeIdentical ) {
//      cout << "PREPARING RATIO HERE (SINGLE TERM)" << endl ;
     auto ratio = makeCondPdfRatioCorr(*static_cast<RooAbsReal*>(term->first()), termNSet, termImpSet, normRange(), RooNameReg::str(_refRangeName));
     std::ostringstream str; termImpSet.printValue(str);
//      cout << GetName() << "inserting ratio term" << endl;
     ratioTerms[str.str()].addOwned(std::move(ratio));
   }
      }

    } else {
//       cout<<"FK: Starting Composite Term"<<endl;

      for (auto const& term : group) {

   Int_t termIdx = terms.IndexOf(term);
   norm=static_cast<RooArgSet*>(norms.At(termIdx));
   imps=static_cast<RooArgSet*>(imp.At(termIdx));
   RooArgSet termNSet(*norm);
   RooArgSet termImpSet(*imps);

   if (!termImpSet.empty() && nullptr != _refRangeName) {

     // WVE we can skip this if the ref range is equal to the normalization range
     bool rangeIdentical(true);
     //FK: Here the refRange should be compared to _normRange, if it's set, and to the normObs range if it's not set
     if(_normRange.Length() > 0) {
       for (auto const* normObs : static_range_cast<RooRealVar*>(termNSet)) {
         if (normObs->getMin(_normRange.Data()) != normObs->getMin(RooNameReg::str(_refRangeName))) rangeIdentical = false;
         if (normObs->getMax(_normRange.Data()) != normObs->getMax(RooNameReg::str(_refRangeName))) rangeIdentical = false;
       }
     } else {
       for (auto const* normObs : static_range_cast<RooRealVar*>(termNSet)) {
         if (normObs->getMin() != normObs->getMin(RooNameReg::str(_refRangeName))) rangeIdentical = false;
         if (normObs->getMax() != normObs->getMax(RooNameReg::str(_refRangeName))) rangeIdentical = false;
       }
     }
//      cout<<"FK: rangeIdentical Composite = "<<(rangeIdentical ? 'T':'F') <<endl;
     if (!rangeIdentical ) {
//        cout << "PREPARING RATIO HERE (COMPOSITE TERM)" << endl ;
       auto ratio = makeCondPdfRatioCorr(*static_cast<RooAbsReal*>(term->first()), termNSet, termImpSet, normRange(), RooNameReg::str(_refRangeName));
       std::ostringstream str; termImpSet.printValue(str);
       ratioTerms[str.str()].addOwned(std::move(ratio));
     }
   }
      }
    }

  }

  // Find groups with y as termNSet
  // Replace G(y) with (G(y),ratio)
  for (auto const& group : groupedList) {
      for (auto const& term : group) {
   Int_t termIdx = terms.IndexOf(term);
   norm = static_cast<RooArgSet*>(norms.At(termIdx));
   imps = static_cast<RooArgSet*>(imp.At(termIdx));
   RooArgSet termNSet(*norm);
   RooArgSet termImpSet(*imps);

   // If termNset matches index of ratioTerms, insert ratio here
   ostringstream str; termNSet.printValue(str);
   if (!ratioTerms[str.str()].empty()) {
//      cout << "MUST INSERT RATIO OBJECT IN TERM (COMPOSITE)" << *term << endl;
     term->add(ratioTerms[str.str()]);
     cache->_ownedList.addOwned(std::move(ratioTerms[str.str()]));
   }
      }
  }

  for (auto const& group : groupedList) {
//     cout << GetName() << ":now processing group" << endl;
//      group->Print("1");

    if (1 == group.size()) {
//       cout << "processing atomic item" << endl;
      RooArgSet* term = group[0];

        Int_t termIdx = terms.IndexOf(term);
        norm = static_cast<RooArgSet*>(norms.At(termIdx));
        integ = static_cast<RooArgSet*>(ints.At(termIdx));
        xdeps = static_cast<RooArgSet*>(cross.At(termIdx));
        imps = static_cast<RooArgSet*>(imp.At(termIdx));

        RooArgSet termNSet;
        RooArgSet termISet;
        RooArgSet termXSet;
        RooArgSet termImpSet;

        // Take list of normalization, integrated dependents from factorization algorithm
        termISet.add(*integ);
        termNSet.add(*norm);

        // Cross-imported integrated dependents
        termXSet.add(*xdeps);
        termImpSet.add(*imps);

        // Add prefab term to partIntList.
        bool isOwned(false);
        vector<RooAbsReal*> func = processProductTerm(nset, iset, isetRangeName, term, termNSet, termISet, isOwned);
        if (func[0]) {
          cache->_partList.add(*func[0]);
          if (isOwned) cache->_ownedList.addOwned(std::unique_ptr<RooAbsArg>{func[0]});

          cache->_normList.emplace_back(std::make_unique<RooArgSet>());
          norm->snapshot(*cache->_normList.back(), false);

          cache->_numList.addOwned(std::unique_ptr<RooAbsArg>{func[1]});
          cache->_denList.addOwned(std::unique_ptr<RooAbsArg>{func[2]});
        }
      } else {
//        cout << "processing composite item" << endl;
        RooArgSet compTermSet;
        RooArgSet compTermNorm;
        RooArgSet compTermNum;
        RooArgSet compTermDen;
        for (auto const &term : group) {
          //    cout << GetName() << ": processing term " << (*term) << " of composite item" << endl ;
          Int_t termIdx = terms.IndexOf(term);
          norm = static_cast<RooArgSet *>(norms.At(termIdx));
          integ = static_cast<RooArgSet *>(ints.At(termIdx));
          xdeps = static_cast<RooArgSet *>(cross.At(termIdx));
          imps = static_cast<RooArgSet *>(imp.At(termIdx));

          RooArgSet termNSet;
          RooArgSet termISet;
          RooArgSet termXSet;
          RooArgSet termImpSet;
          termISet.add(*integ);
          termNSet.add(*norm);
          termXSet.add(*xdeps);
          termImpSet.add(*imps);

          // Remove outer integration dependents from termISet
          termISet.remove(outerIntDeps, true, true);

          bool isOwned = false;
          vector<RooAbsReal *> func =
             processProductTerm(nset, iset, isetRangeName, term, termNSet, termISet, isOwned, true);
          //       cout << GetName() << ": created composite term component " << func[0]->GetName() << endl;
          if (func[0]) {
     compTermSet.add(*func[0]);
     if (isOwned) cache->_ownedList.addOwned(std::unique_ptr<RooAbsArg>{func[0]});
     compTermNorm.add(*norm, false);

     compTermNum.add(*func[1]);
     compTermDen.add(*func[2]);
     //cache->_numList.add(*func[1]);
     //cache->_denList.add(*func[2]);

   }
      }

//       cout << GetName() << ": constructing special composite product" << endl;
//       cout << GetName() << ": compTermSet = " ; compTermSet.Print("1");

      // WVE THIS NEEDS TO BE REARRANGED

      // compTermset is set van partial integrals to be multiplied
      // prodtmp = product (compTermSet)
      // inttmp = int ( prodtmp ) d (outerIntDeps) _range_isetRangeName

      const std::string prodname = makeRGPPName("SPECPROD", compTermSet, outerIntDeps, RooArgSet(), isetRangeName);
      auto prodtmp = std::make_unique<RooProduct>(prodname.c_str(), prodname.c_str(), compTermSet);

      const std::string intname = makeRGPPName("SPECINT", compTermSet, outerIntDeps, RooArgSet(), isetRangeName);
      auto inttmp = std::make_unique<RooRealIntegral>(intname.c_str(), intname.c_str(), *prodtmp, outerIntDeps, nullptr, nullptr, isetRangeName);
      inttmp->setStringAttribute("PROD_TERM_TYPE", "SPECINT");

      cache->_partList.add(*inttmp);

      // Product of numerator terms
      const string prodname_num = makeRGPPName("SPECPROD_NUM", compTermNum, RooArgSet(), RooArgSet(), nullptr);
      auto prodtmp_num = std::make_unique<RooProduct>(prodname_num.c_str(), prodname_num.c_str(), compTermNum);
      prodtmp_num->addOwnedComponents(compTermNum);

      // Product of denominator terms
      const string prodname_den = makeRGPPName("SPECPROD_DEN", compTermDen, RooArgSet(), RooArgSet(), nullptr);
      auto prodtmp_den = std::make_unique<RooProduct>(prodname_den.c_str(), prodname_den.c_str(), compTermDen);
      prodtmp_den->addOwnedComponents(compTermDen);

      // Ratio
      std::string name = Form("SPEC_RATIO(%s,%s)", prodname_num.c_str(), prodname_den.c_str());
      auto ndr = std::make_unique<RooFormulaVar>(name.c_str(), "@0/@1", RooArgList(*prodtmp_num, *prodtmp_den));

      // Integral of ratio
      std::unique_ptr<RooAbsReal> numtmp{ndr->createIntegral(outerIntDeps,isetRangeName)};
      numtmp->addOwnedComponents(std::move(ndr));

      cache->_ownedList.addOwned(std::move(prodtmp));
      cache->_ownedList.addOwned(std::move(inttmp));
      cache->_ownedList.addOwned(std::move(prodtmp_num));
      cache->_ownedList.addOwned(std::move(prodtmp_den));
      cache->_numList.addOwned(std::move(numtmp));
      cache->_denList.addOwned(std::unique_ptr<RooAbsArg>{static_cast<RooAbsArg*>(RooFit::RooConst(1).clone("1"))});
      cache->_normList.emplace_back(std::make_unique<RooArgSet>());
      compTermNorm.snapshot(*cache->_normList.back(), false);
    }
  }

  // Need to rearrange product in case of multiple ranges
  if (_normRange.Contains(",")) {
    rearrangeProduct(*cache);
  }

  // We own contents of all lists filled by factorizeProduct()
  terms.Delete();
  ints.Delete();
  imp.Delete();
  norms.Delete();
  cross.Delete();

  return cache;
}



////////////////////////////////////////////////////////////////////////////////
/// For single normalization ranges

std::unique_ptr<RooAbsReal> RooProdPdf::makeCondPdfRatioCorr(RooAbsReal& pdf, const RooArgSet& termNset, const RooArgSet& /*termImpSet*/, const char* normRangeTmp, const char* refRange) const
{
  std::unique_ptr<RooAbsReal> ratio_num{pdf.createIntegral(termNset,normRangeTmp)};
  std::unique_ptr<RooAbsReal> ratio_den{pdf.createIntegral(termNset,refRange)};
  auto ratio = std::make_unique<RooFormulaVar>(Form("ratio(%s,%s)",ratio_num->GetName(),ratio_den->GetName()),"@0/@1",
                  RooArgList(*ratio_num,*ratio_den)) ;

  ratio->addOwnedComponents(std::move(ratio_num));
  ratio->addOwnedComponents(std::move(ratio_den));
  ratio->setAttribute("RATIO_TERM") ;
  return ratio ;
}




////////////////////////////////////////////////////////////////////////////////

void RooProdPdf::rearrangeProduct(RooProdPdf::CacheElem& cache) const
{
  RooAbsReal *part;
  RooAbsReal *num;
  RooAbsReal *den;
  RooArgSet nomList ;

  list<string> rangeComps ;
  {
    std::vector<char> buf(strlen(_normRange.Data()) + 1);
    strcpy(buf.data(),_normRange.Data()) ;
    char* save(nullptr) ;
    char* token = R__STRTOK_R(buf.data(),",",&save) ;
    while(token) {
      rangeComps.push_back(token) ;
      token = R__STRTOK_R(nullptr,",",&save) ;
    }
  }


  std::map<std::string,RooArgSet> denListList ;
  RooArgSet specIntDeps ;
  string specIntRange ;

//   cout << "THIS IS REARRANGEPRODUCT" << endl ;

  for (std::size_t i = 0; i < cache._partList.size(); i++) {

    part = static_cast<RooAbsReal*>(cache._partList.at(i));
    num = static_cast<RooAbsReal*>(cache._numList.at(i));
    den = static_cast<RooAbsReal*>(cache._denList.at(i));
    i++;

//     cout << "now processing part " << part->GetName() << " of type " << part->getStringAttribute("PROD_TERM_TYPE") << endl ;
//     cout << "corresponding numerator = " << num->GetName() << endl ;
//     cout << "corresponding denominator = " << den->GetName() << endl ;


    RooFormulaVar* ratio(nullptr) ;
    RooArgSet origNumTerm ;

    if (string("SPECINT")==part->getStringAttribute("PROD_TERM_TYPE")) {

   RooRealIntegral* orig = static_cast<RooRealIntegral*>(num);
   auto specratio = static_cast<RooFormulaVar const*>(&orig->integrand()) ;
   RooProduct* func = static_cast<RooProduct*>(specratio->getParameter(0)) ;

   std::unique_ptr<RooArgSet> components{orig->getComponents()};
   for(RooAbsArg * carg : *components) {
     if (carg->getAttribute("RATIO_TERM")) {
       ratio = static_cast<RooFormulaVar*>(carg) ;
       break ;
     }
   }

   if (ratio) {
     RooCustomizer cust(*func,"blah") ;
     cust.replaceArg(*ratio,RooFit::RooConst(1)) ;
     RooAbsArg* funcCust = cust.build() ;
//      cout << "customized function = " << endl ;
//      funcCust->printComponentTree() ;
     nomList.add(*funcCust) ;
   } else {
     nomList.add(*func) ;
   }


    } else {

      // Find the ratio term
      RooAbsReal* func = num;
      // If top level object is integral, navigate to integrand
      if (func->InheritsFrom(RooRealIntegral::Class())) {
   func = const_cast<RooAbsReal*>(&static_cast<RooRealIntegral*>(func)->integrand());
      }
      if (func->InheritsFrom(RooProduct::Class())) {
//    cout << "product term found: " ; func->Print() ;
   for(RooAbsArg * arg : static_cast<RooProduct*>(func)->components()) {
     if (arg->getAttribute("RATIO_TERM")) {
       ratio = static_cast<RooFormulaVar*>(arg) ;
     } else {
       origNumTerm.add(*arg) ;
     }
   }
      }

      if (ratio) {
//    cout << "Found ratio term in numerator: " << ratio->GetName() << endl ;
//    cout << "Adding only original term to numerator: " << origNumTerm << endl ;
   nomList.add(origNumTerm) ;
      } else {
   nomList.add(*num) ;
      }

    }

    for (list<string>::iterator iter = rangeComps.begin() ; iter != rangeComps.end() ; ++iter) {
      // If denominator is an integral, make a clone with the integration range adjusted to
      // the selected component of the normalization integral
//       cout << "NOW PROCESSING DENOMINATOR " << den->ClassName() << "::" << den->GetName() << endl ;

      if (string("SPECINT")==part->getStringAttribute("PROD_TERM_TYPE")) {

//    cout << "create integral: SPECINT case" << endl ;
   RooRealIntegral* orig = static_cast<RooRealIntegral*>(num);
   auto specRatio = static_cast<RooFormulaVar const*>(&orig->integrand()) ;
   specIntDeps.add(orig->intVars()) ;
   if (orig->intRange()) {
     specIntRange = orig->intRange() ;
   }
   //RooProduct* numtmp = (RooProduct*) specRatio->getParameter(0) ;
   RooProduct* dentmp = static_cast<RooProduct*>(specRatio->getParameter(1)) ;

//    cout << "numtmp = " << numtmp->ClassName() << "::" << numtmp->GetName() << endl ;
//    cout << "dentmp = " << dentmp->ClassName() << "::" << dentmp->GetName() << endl ;

//    cout << "denominator components are " << dentmp->components() << endl ;
   for (auto* parg : static_range_cast<RooAbsReal*>(dentmp->components())) {
//      cout << "now processing denominator component " << parg->ClassName() << "::" << parg->GetName() << endl ;

     if (ratio && parg->dependsOn(*ratio)) {
//        cout << "depends in value of ratio" << endl ;

       // Make specialize ratio instance
       std::unique_ptr<RooAbsReal> specializedRatio{specializeRatio(*(RooFormulaVar*)ratio,iter->c_str())};

//        cout << "specRatio = " << endl ;
//        specializedRatio->printComponentTree() ;

       // Replace generic ratio with specialized ratio
       RooAbsArg *partCust(nullptr) ;
       if (parg->InheritsFrom(RooAddition::Class())) {



         RooAddition* tmpadd = static_cast<RooAddition*>(parg) ;

         RooCustomizer cust(*tmpadd->list1().first(),Form("blah_%s",iter->c_str())) ;
         cust.replaceArg(*ratio,*specializedRatio) ;
         partCust = cust.build() ;

       } else {
         RooCustomizer cust(*parg,Form("blah_%s",iter->c_str())) ;
         cust.replaceArg(*ratio,*specializedRatio) ;
         partCust = cust.build() ;
       }

       // Print customized denominator
//        cout << "customized function = " << endl ;
//        partCust->printComponentTree() ;

       std::unique_ptr<RooAbsReal> specializedPartCust{specializeIntegral(*static_cast<RooAbsReal*>(partCust),iter->c_str())};

       // Finally divide again by ratio
       string name = Form("%s_divided_by_ratio",specializedPartCust->GetName()) ;
       auto specIntFinal = std::make_unique<RooFormulaVar>(name.c_str(),"@0/@1",RooArgList(*specializedPartCust,*specializedRatio)) ;
       specIntFinal->addOwnedComponents(std::move(specializedPartCust));
       specIntFinal->addOwnedComponents(std::move(specializedRatio));

       denListList[*iter].addOwned(std::move(specIntFinal));
     } else {

//        cout << "does NOT depend on value of ratio" << endl ;
//        parg->Print("t") ;

       denListList[*iter].addOwned(specializeIntegral(*parg,iter->c_str()));

     }
   }
//    cout << "end iteration over denominator components" << endl ;
      } else {

   if (ratio) {

     std::unique_ptr<RooAbsReal> specRatio{specializeRatio(*(RooFormulaVar*)ratio,iter->c_str())};

     // If integral is 'Int r(y)*g(y) dy ' then divide a posteriori by r(y)
//      cout << "have ratio, orig den = " << den->GetName() << endl ;

     RooArgSet tmp(origNumTerm) ;
     tmp.add(*specRatio) ;
     const string pname = makeRGPPName("PROD",tmp,RooArgSet(),RooArgSet(),nullptr) ;
     auto specDenProd = std::make_unique<RooProduct>(pname.c_str(),pname.c_str(),tmp) ;
     std::unique_ptr<RooAbsReal> specInt;

     if (den->InheritsFrom(RooRealIntegral::Class())) {
       specInt = std::unique_ptr<RooAbsReal>{specDenProd->createIntegral((static_cast<RooRealIntegral*>(den))->intVars(),iter->c_str())};
       specInt->addOwnedComponents(std::move(specDenProd));
     } else if (den->InheritsFrom(RooAddition::Class())) {
       RooAddition* orig = static_cast<RooAddition*>(den) ;
       RooRealIntegral* origInt = static_cast<RooRealIntegral*>(orig->list1().first()) ;
       specInt = std::unique_ptr<RooAbsReal>{specDenProd->createIntegral(origInt->intVars(),iter->c_str())};
       specInt->addOwnedComponents(std::move(specDenProd));
     } else {
       throw string("this should not happen") ;
     }

     //RooAbsReal* specInt = specializeIntegral(*den,iter->c_str()) ;
     string name = Form("%s_divided_by_ratio",specInt->GetName()) ;
     auto specIntFinal = std::make_unique<RooFormulaVar>(name.c_str(),"@0/@1",RooArgList(*specInt,*specRatio)) ;
     specIntFinal->addOwnedComponents(std::move(specInt));
     specIntFinal->addOwnedComponents(std::move(specRatio));
     denListList[*iter].addOwned(std::move(specIntFinal));
   } else {
     denListList[*iter].addOwned(specializeIntegral(*den,iter->c_str()));
   }

      }
    }

  }

  // Do not rearrage terms if numerator and denominator are effectively empty
  if (nomList.empty()) {
    return ;
  }

  string name = Form("%s_numerator",GetName()) ;
  // WVE FIX THIS (2)

  std::unique_ptr<RooAbsReal> numerator = std::make_unique<RooProduct>(name.c_str(),name.c_str(),nomList) ;

  RooArgSet products ;
//   cout << "nomList = " << nomList << endl ;
  for (map<string,RooArgSet>::iterator iter = denListList.begin() ; iter != denListList.end() ; ++iter) {
//     cout << "denList[" << iter->first << "] = " << iter->second << endl ;
    name = Form("%s_denominator_comp_%s",GetName(),iter->first.c_str()) ;
    // WVE FIX THIS (2)
    RooProduct* prod_comp = new RooProduct(name.c_str(),name.c_str(),iter->second) ;
    prod_comp->addOwnedComponents(std::move(iter->second));
    products.add(*prod_comp) ;
  }
  name = Form("%s_denominator_sum",GetName()) ;
  RooAbsReal* norm = new RooAddition(name.c_str(),name.c_str(),products) ;
  norm->addOwnedComponents(products) ;

  if (!specIntDeps.empty()) {
    // Apply posterior integration required for SPECINT case

    string namesr = Form("SPEC_RATIO(%s,%s)",numerator->GetName(),norm->GetName()) ;
    RooFormulaVar* ndr = new RooFormulaVar(namesr.c_str(),"@0/@1",RooArgList(*numerator,*norm)) ;
    ndr->addOwnedComponents(std::move(numerator));

    // Integral of ratio
    numerator = std::unique_ptr<RooAbsReal>{ndr->createIntegral(specIntDeps,specIntRange.c_str())};

    norm = static_cast<RooAbsReal*>(RooFit::RooConst(1).Clone()) ;
  }


//   cout << "numerator" << endl ;
//   numerator->printComponentTree("",0,5) ;
//   cout << "denominator" << endl ;
//   norm->printComponentTree("",0,5) ;


  // WVE DEBUG
  //RooMsgService::instance().debugWorkspace()->import(RooArgSet(*numerator,*norm)) ;

  cache._rearrangedNum = std::move(numerator);
  cache._rearrangedDen.reset(norm);
  cache._isRearranged = true ;

}


////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<RooAbsReal> RooProdPdf::specializeRatio(RooFormulaVar& input, const char* targetRangeName) const
{
  RooRealIntegral* numint = static_cast<RooRealIntegral*>(input.getParameter(0)) ;
  RooRealIntegral* denint = static_cast<RooRealIntegral*>(input.getParameter(1)) ;

  std::unique_ptr<RooAbsReal> numint_spec{specializeIntegral(*numint,targetRangeName)};

  std::unique_ptr<RooAbsReal> ret = std::make_unique<RooFormulaVar>(Form("ratio(%s,%s)",numint_spec->GetName(),denint->GetName()),"@0/@1",RooArgList(*numint_spec,*denint)) ;
  ret->addOwnedComponents(std::move(numint_spec));

  return ret;
}



////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<RooAbsReal> RooProdPdf::specializeIntegral(RooAbsReal& input, const char* targetRangeName) const
{
  if (input.InheritsFrom(RooRealIntegral::Class())) {

    // If input is integral, recreate integral but override integration range to be targetRangeName
    RooRealIntegral* orig = static_cast<RooRealIntegral*>(&input) ;
//     cout << "creating integral: integrand =  " << orig->integrand().GetName() << " vars = " << orig->intVars() << " range = " << targetRangeName << endl ;
    return std::unique_ptr<RooAbsReal>{orig->integrand().createIntegral(orig->intVars(),targetRangeName)};

  } else if (input.InheritsFrom(RooAddition::Class())) {

    // If input is sum of integrals, recreate integral from first component of set, but override integration range to be targetRangeName
    RooAddition* orig = static_cast<RooAddition*>(&input) ;
    RooRealIntegral* origInt = static_cast<RooRealIntegral*>(orig->list1().first()) ;
//     cout << "creating integral from addition: integrand =  " << origInt->integrand().GetName() << " vars = " << origInt->intVars() << " range = " << targetRangeName << endl ;
    return std::unique_ptr<RooAbsReal>{origInt->integrand().createIntegral(origInt->intVars(),targetRangeName)};
  }

  std::stringstream errMsg;
  errMsg << "specializeIntegral: unknown input type " << input.ClassName() << "::" << input.GetName();
  throw std::runtime_error(errMsg.str());
}


////////////////////////////////////////////////////////////////////////////////
/// Group product into terms that can be calculated independently

void RooProdPdf::groupProductTerms(std::list<std::vector<RooArgSet*>>& groupedTerms, RooArgSet& outerIntDeps,
               const RooLinkedList& terms, const RooLinkedList& norms,
               const RooLinkedList& imps, const RooLinkedList& ints, const RooLinkedList& /*cross*/) const
{
  // Start out with each term in its own group
  for(auto * term : static_range_cast<RooArgSet*>(terms)) {
    groupedTerms.emplace_back();
    groupedTerms.back().emplace_back(term) ;
  }

  // Make list of imported dependents that occur in any term
  RooArgSet allImpDeps ;
  for(auto * impDeps : static_range_cast<RooArgSet*>(imps)) {
    allImpDeps.add(*impDeps,false) ;
  }

  // Make list of integrated dependents that occur in any term
  RooArgSet allIntDeps ;
  for(auto * intDeps : static_range_cast<RooArgSet*>(ints)) {
    allIntDeps.add(*intDeps,false) ;
  }

  outerIntDeps.removeAll() ;
  outerIntDeps.add(*std::unique_ptr<RooArgSet>{allIntDeps.selectCommon(allImpDeps)});

  // Now iteratively merge groups that should be (partially) integrated together
  for(RooAbsArg * outerIntDep : outerIntDeps) {

    // Collect groups that feature this dependent
    std::vector<RooArgSet*>* newGroup = nullptr ;

    // Loop over groups
    bool needMerge = false ;
    auto group = groupedTerms.begin();
    auto nGroups = groupedTerms.size();
    for (size_t iGroup = 0; iGroup < nGroups; ++iGroup) {

      // See if any term in this group depends in any ay on outerDepInt
      for (auto const& term2 : *group) {

   Int_t termIdx = terms.IndexOf(term2) ;
   RooArgSet* termNormDeps = static_cast<RooArgSet*>(norms.At(termIdx)) ;
   RooArgSet* termIntDeps = static_cast<RooArgSet*>(ints.At(termIdx)) ;
   RooArgSet* termImpDeps = static_cast<RooArgSet*>(imps.At(termIdx)) ;

   if (termNormDeps->contains(*outerIntDep) ||
       termIntDeps->contains(*outerIntDep) ||
       termImpDeps->contains(*outerIntDep)) {
     needMerge = true ;
   }

      }

      if (needMerge) {
   // Create composite group if not yet existing
   if (newGroup==nullptr) {
     groupedTerms.emplace_back() ;
     newGroup = &groupedTerms.back() ;
   }

   // Add terms of this group to new term
   for (auto& term2 : *group) {
     newGroup->emplace_back(term2) ;
   }

   // Remove this non-owning group from list
   group = groupedTerms.erase(group);
      } else {
        ++group;
      }
    }

  }
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate integrals of factorized product terms over observables iset while normalized
/// to observables in nset.

std::vector<RooAbsReal*> RooProdPdf::processProductTerm(const RooArgSet* nset, const RooArgSet* iset, const char* isetRangeName,
                     const RooArgSet* term,const RooArgSet& termNSet, const RooArgSet& termISet,
                     bool& isOwned, bool forceWrap) const
{
  vector<RooAbsReal*> ret(3) ; ret[0] = nullptr ; ret[1] = nullptr ; ret[2] = nullptr ;

  // CASE I: factorizing term: term is integrated over all normalizing observables
  // -----------------------------------------------------------------------------
  // Check if all observbales of this term are integrated. If so the term cancels
  if (!termNSet.empty() && termNSet.size()==termISet.size() && isetRangeName==nullptr) {


    //cout << "processProductTerm(" << GetName() << ") case I " << endl ;

    // Term factorizes
    return ret ;
  }

  // CASE II: Dropped terms: if term is entirely unnormalized, it should be dropped
  // ------------------------------------------------------------------------------
  if (nset && termNSet.empty()) {

    //cout << "processProductTerm(" << GetName() << ") case II " << endl ;

    // Drop terms that are not asked to be normalized
    return ret ;
  }

  if (iset && !termISet.empty()) {
    if (term->size()==1) {

      // CASE IIIa: Normalized and partially integrated single PDF term
      //---------------------------------------------------------------

      RooAbsPdf* pdf = static_cast<RooAbsPdf*>(term->first()) ;

      RooAbsReal* partInt = std::unique_ptr<RooAbsReal>{pdf->createIntegral(termISet,termNSet,isetRangeName)}.release();
      partInt->setOperMode(operMode()) ;
      partInt->setStringAttribute("PROD_TERM_TYPE","IIIa") ;

      isOwned=true ;

      //cout << "processProductTerm(" << GetName() << ") case IIIa func = " << partInt->GetName() << endl ;

      ret[0] = partInt ;

      // Split mode results
      ret[1] = std::unique_ptr<RooAbsReal>{pdf->createIntegral(termISet,isetRangeName)}.release();
      ret[2] = std::unique_ptr<RooAbsReal>{pdf->createIntegral(termNSet,normRange())}.release();

      return ret ;


    } else {

      // CASE IIIb: Normalized and partially integrated composite PDF term
      //---------------------------------------------------------------

      // Use auxiliary class RooGenProdProj to calculate this term
      const std::string name = makeRGPPName("GENPROJ_",*term,termISet,termNSet,isetRangeName) ;
      RooAbsReal* partInt = new RooGenProdProj(name.c_str(),name.c_str(),*term,termISet,termNSet,isetRangeName) ;
      partInt->setStringAttribute("PROD_TERM_TYPE","IIIb") ;
      partInt->setOperMode(operMode()) ;

      //cout << "processProductTerm(" << GetName() << ") case IIIb func = " << partInt->GetName() << endl ;

      isOwned=true ;
      ret[0] = partInt ;

      const std::string name1 = makeRGPPName("PROD",*term,RooArgSet(),RooArgSet(),nullptr) ;

      // WVE FIX THIS
      RooProduct* tmp_prod = new RooProduct(name1.c_str(),name1.c_str(),*term) ;

      ret[1] = std::unique_ptr<RooAbsReal>{tmp_prod->createIntegral(termISet,isetRangeName)}.release();
      ret[2] = std::unique_ptr<RooAbsReal>{tmp_prod->createIntegral(termNSet,normRange())}.release();

      return ret ;
    }
  }

  // CASE IVa: Normalized non-integrated composite PDF term
  // -------------------------------------------------------
  if (nset && !nset->empty() && term->size()>1) {
    // Composite term needs normalized integration

    const std::string name = makeRGPPName("GENPROJ_",*term,termISet,termNSet,isetRangeName) ;
    RooAbsReal* partInt = new RooGenProdProj(name.c_str(),name.c_str(),*term,termISet,termNSet,isetRangeName,normRange()) ;
    partInt->setExpensiveObjectCache(expensiveObjectCache()) ;

    partInt->setStringAttribute("PROD_TERM_TYPE","IVa") ;
    partInt->setOperMode(operMode()) ;

    //cout << "processProductTerm(" << GetName() << ") case IVa func = " << partInt->GetName() << endl ;

    isOwned=true ;
    ret[0] = partInt ;

    const std::string name1 = makeRGPPName("PROD",*term,RooArgSet(),RooArgSet(),nullptr) ;

    // WVE FIX THIS
    RooProduct* tmp_prod = new RooProduct(name1.c_str(),name1.c_str(),*term) ;

    ret[1] = std::unique_ptr<RooAbsReal>{tmp_prod->createIntegral(termISet,isetRangeName)}.release();
    ret[2] = std::unique_ptr<RooAbsReal>{tmp_prod->createIntegral(termNSet,normRange())}.release();

    return ret ;
  }

  // CASE IVb: Normalized, non-integrated single PDF term
  // -----------------------------------------------------
  for (auto* pdf : static_range_cast<RooAbsPdf*>(*term)) {

    if (forceWrap) {

      // Construct representative name of normalization wrapper
      TString name(pdf->GetName()) ;
      name.Append("_NORM[") ;
      bool first(true) ;
      for (auto const* arg : termNSet) {
   if (!first) {
     name.Append(",") ;
   } else {
     first=false ;
   }
   name.Append(arg->GetName()) ;
      }
      if (normRange()) {
   name.Append("|") ;
   name.Append(normRange()) ;
      }
      name.Append("]") ;

      RooAbsReal* partInt = new RooRealIntegral(name.Data(),name.Data(),*pdf,RooArgSet(),&termNSet) ;
      partInt->setStringAttribute("PROD_TERM_TYPE","IVb") ;
      isOwned=true ;

      //cout << "processProductTerm(" << GetName() << ") case IVb func = " << partInt->GetName() << endl ;

      ret[0] = partInt ;

      ret[1] = std::unique_ptr<RooAbsReal>{pdf->createIntegral(RooArgSet())}.release();
      ret[2] = std::unique_ptr<RooAbsReal>{pdf->createIntegral(termNSet,normRange())}.release();

      return ret ;


    } else {
      isOwned=false ;

      //cout << "processProductTerm(" << GetName() << ") case IVb func = " << pdf->GetName() << endl ;


      pdf->setStringAttribute("PROD_TERM_TYPE","IVb") ;
      ret[0] = pdf ;

      ret[1] = std::unique_ptr<RooAbsReal>{pdf->createIntegral(RooArgSet())}.release();
      ret[2] = !termNSet.empty() ? std::unique_ptr<RooAbsReal>{pdf->createIntegral(termNSet,normRange())}.release()
                                 : (static_cast<RooAbsReal*>(RooFit::RooConst(1).clone("1")));
      return ret  ;
    }
  }

  coutE(Eval) << "RooProdPdf::processProductTerm(" << GetName() << ") unidentified term!!!" << endl ;
  return ret ;
}




////////////////////////////////////////////////////////////////////////////////
/// Make an appropriate automatic name for a RooGenProdProj object in getPartIntList()

std::string RooProdPdf::makeRGPPName(const char* pfx, const RooArgSet& term, const RooArgSet& iset,
                 const RooArgSet& nset, const char* isetRangeName) const
{
  // Make an appropriate automatic name for a RooGenProdProj object in getPartIntList()

  std::ostringstream os(pfx);
  os << "[";

  // Encode component names
  bool first(true) ;
  for (auto const* pdf : static_range_cast<RooAbsPdf*>(term)) {
    if (!first) os << "_X_";
    first = false;
    os << pdf->GetName();
  }
  os << "]" << integralNameSuffix(iset,&nset,isetRangeName,true);

  return os.str();
}



////////////////////////////////////////////////////////////////////////////////
/// Force RooRealIntegral to offer all observables for internal integration

bool RooProdPdf::forceAnalyticalInt(const RooAbsArg& /*dep*/) const
{
  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Determine which part (if any) of given integral can be performed analytically.
/// If any analytical integration is possible, return integration scenario code.
///
/// RooProdPdf implements two strategies in implementing analytical integrals
///
/// First, PDF components whose entire set of dependents are requested to be integrated
/// can be dropped from the product, as they will integrate out to 1 by construction
///
/// Second, RooProdPdf queries each remaining component PDF for its analytical integration
/// capability of the requested set ('allVars'). It finds the largest common set of variables
/// that can be integrated by all remaining components. If such a set exists, it reconfirms that
/// each component is capable of analytically integrating the common set, and combines the components
/// individual integration codes into a single integration code valid for RooProdPdf.

Int_t RooProdPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars,
                 const RooArgSet* normSet, const char* rangeName) const
{
  if (_forceNumInt) return 0 ;

  // Declare that we can analytically integrate all requested observables
  analVars.add(allVars) ;

  // Retrieve (or create) the required partial integral list
  Int_t code = getPartIntList(normSet,&allVars,rangeName);

  return code+1 ;
}




////////////////////////////////////////////////////////////////////////////////
/// Return analytical integral defined by given scenario code

double RooProdPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
  // No integration scenario
  if (code==0) {
    return getVal(normSet) ;
  }


  // WVE needs adaptation for rangename feature

  // Partial integration scenarios
  CacheElem* cache = static_cast<CacheElem*>(_cacheMgr.getObjByIndex(code-1)) ;

  // If cache has been sterilized, revive this slot
  if (cache==nullptr) {
    std::unique_ptr<RooArgSet> vars{getParameters(RooArgSet())} ;
    RooArgSet nset = _cacheMgr.selectFromSet1(*vars, code-1) ;
    RooArgSet iset = _cacheMgr.selectFromSet2(*vars, code-1) ;

    Int_t code2 = getPartIntList(&nset, &iset, rangeName) ;

    // preceding call to getPartIntList guarantees non-null return
    // coverity[NULL_RETURNS]
    cache = static_cast<CacheElem*>(_cacheMgr.getObj(&nset,&iset,&code2,rangeName)) ;
  }

  double val = calculate(*cache,true) ;
//   cout << "RPP::aIWN(" << GetName() << ") ,code = " << code << ", value = " << val << endl ;

  return val ;
}



////////////////////////////////////////////////////////////////////////////////
/// If this product contains exactly one extendable p.d.f return the extension abilities of
/// that p.d.f, otherwise return CanNotBeExtended

RooAbsPdf::ExtendMode RooProdPdf::extendMode() const
{
  return (_extendedIndex>=0) ? (static_cast<RooAbsPdf*>(_pdfList.at(_extendedIndex)))->extendMode() : CanNotBeExtended ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the expected number of events associated with the extendable input PDF
/// in the product. If there is no extendable term, abort.

double RooProdPdf::expectedEvents(const RooArgSet* nset) const
{
  if (_extendedIndex<0) {
    coutF(Generation) << "Requesting expected number of events from a RooProdPdf that does not contain an extended p.d.f" << endl ;
    throw std::logic_error(std::string("RooProdPdf ") + GetName() + " could not be extended.");
  }

  return static_cast<RooAbsPdf*>(_pdfList.at(_extendedIndex))->expectedEvents(nset) ;
}

std::unique_ptr<RooAbsReal> RooProdPdf::createExpectedEventsFunc(const RooArgSet* nset) const
{
  if (_extendedIndex<0) {
    coutF(Generation) << "Requesting expected number of events from a RooProdPdf that does not contain an extended p.d.f" << endl ;
    throw std::logic_error(std::string("RooProdPdf ") + GetName() + " could not be extended.");
  }

  return static_cast<RooAbsPdf*>(_pdfList.at(_extendedIndex))->createExpectedEventsFunc(nset);
}


////////////////////////////////////////////////////////////////////////////////
/// Return generator context optimized for generating events from product p.d.f.s

RooAbsGenContext* RooProdPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype,
                const RooArgSet* auxProto, bool verbose) const
{
  if (_useDefaultGen) return RooAbsPdf::genContext(vars,prototype,auxProto,verbose) ;
  return new RooProdGenContext(*this,vars,prototype,auxProto,verbose) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Query internal generation capabilities of component p.d.f.s and aggregate capabilities
/// into master configuration passed to the generator context

Int_t RooProdPdf::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK) const
{
  if (!_useDefaultGen) return 0 ;

  // Find the subset directVars that only depend on a single PDF in the product
  RooArgSet directSafe ;
  for (auto const* arg : directVars) {
    if (isDirectGenSafe(*arg)) directSafe.add(*arg) ;
  }


  // Now find direct integrator for relevant components ;
  std::vector<Int_t> code;
  code.reserve(64);
  for (auto const* pdf : static_range_cast<RooAbsPdf*>(_pdfList)) {
    RooArgSet pdfDirect ;
    Int_t pdfCode = pdf->getGenerator(directSafe,pdfDirect,staticInitOK);
    code.push_back(pdfCode);
    if (pdfCode != 0) {
      generateVars.add(pdfDirect) ;
    }
  }


  if (!generateVars.empty()) {
    Int_t masterCode = _genCode.store(code) ;
    return masterCode+1 ;
  } else {
    return 0 ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Forward one-time initialization call to component generation initialization
/// methods.

void RooProdPdf::initGenerator(Int_t code)
{
  if (!_useDefaultGen) return ;

  const std::vector<Int_t>& codeList = _genCode.retrieve(code-1) ;
  Int_t i(0) ;
  for (auto* pdf : static_range_cast<RooAbsPdf*>(_pdfList)) {
    if (codeList[i]!=0) {
      pdf->initGenerator(codeList[i]) ;
    }
    i++ ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Generate a single event with configuration specified by 'code'
/// Defer internal generation to components as encoded in the _genCode
/// registry for given generator code.

void RooProdPdf::generateEvent(Int_t code)
{
  if (!_useDefaultGen) return ;

  const std::vector<Int_t>& codeList = _genCode.retrieve(code-1) ;
  Int_t i(0) ;
  for (auto* pdf : static_range_cast<RooAbsPdf*>(_pdfList)) {
    if (codeList[i]!=0) {
      pdf->generateEvent(codeList[i]) ;
    }
    i++ ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return RooAbsArg components contained in the cache

RooArgList RooProdPdf::CacheElem::containedArgs(Action)
{
  RooArgList ret ;
  ret.add(_partList) ;
  ret.add(_numList) ;
  ret.add(_denList) ;
  if (_rearrangedNum) ret.add(*_rearrangedNum) ;
  if (_rearrangedDen) ret.add(*_rearrangedDen) ;
  return ret ;

}



////////////////////////////////////////////////////////////////////////////////
/// Hook function to print cache contents in tree printing of RooProdPdf

void RooProdPdf::CacheElem::printCompactTreeHook(ostream& os, const char* indent, Int_t curElem, Int_t maxElem)
{
   if (curElem==0) {
     os << indent << "RooProdPdf begin partial integral cache" << endl ;
   }

   auto indent2 = std::string(indent) +  "[" + std::to_string(curElem) + "]";
   for(auto const& arg : _partList) {
     arg->printCompactTree(os,indent2.c_str()) ;
   }

   if (curElem==maxElem) {
     os << indent << "RooProdPdf end partial integral cache" << endl ;
   }
}



////////////////////////////////////////////////////////////////////////////////
/// Forward determination of safety of internal generator code to
/// component p.d.f that would generate the given observable

bool RooProdPdf::isDirectGenSafe(const RooAbsArg& arg) const
{
  // Only override base class behaviour if default generator method is enabled
  if (!_useDefaultGen) return RooAbsPdf::isDirectGenSafe(arg) ;

  // Argument may appear in only one PDF component
  RooAbsPdf* thePdf(nullptr) ;
  for (auto* pdf : static_range_cast<RooAbsPdf*>(_pdfList)) {

    if (pdf->dependsOn(arg)) {
      // Found PDF depending on arg

      // If multiple PDFs depend on arg directGen is not safe
      if (thePdf) return false ;

      thePdf = pdf ;
    }
  }
  // Forward call to relevant component PDF
  return thePdf?(thePdf->isDirectGenSafe(arg)):false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Look up user specified normalization set for given input PDF component

RooArgSet* RooProdPdf::findPdfNSet(RooAbsPdf const& pdf) const
{
  Int_t idx = _pdfList.index(&pdf) ;
  if (idx<0) return nullptr;
  return _pdfNSetList[idx].get() ;
}



/// Add some full PDFs to the factors of this RooProdPdf.
void RooProdPdf::addPdfs(RooAbsCollection const& pdfs)
{
   size_t numExtended = (_extendedIndex==-1) ? 0 : 1;

   for(auto arg : pdfs) {
      RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(arg);
      if (!pdf) {
         coutW(InputArguments) << "RooProdPdf::addPdfs(" << GetName() << ") list arg "
                               << arg->GetName() << " is not a PDF, ignored" << endl ;
         continue;
      }
      if(pdf->canBeExtended()) {
         if (_extendedIndex == -1) {
            _extendedIndex = _pdfList.size();
         } else {
            numExtended++;
         }
      }
      _pdfList.add(*pdf);
      _pdfNSetList.emplace_back(std::make_unique<RooArgSet>("nset"));
   }

   // Protect against multiple extended terms
   if (numExtended>1) {
      coutW(InputArguments) << "RooProdPdf::addPdfs(" << GetName()
                            << ") WARNING: multiple components with extended terms detected,"
                            << " product will not be extendible." << endl ;
      _extendedIndex = -1 ;
   }

   // Reset cache
   _cacheMgr.reset() ;

}

/// Remove some PDFs from the factors of this RooProdPdf.
void RooProdPdf::removePdfs(RooAbsCollection const& pdfs)
{
  // Remember what the extended PDF is
  RooAbsArg const* extPdf = _extendedIndex >= 0 ? &_pdfList[_extendedIndex] : nullptr;

  // Actually remove the PDFs and associated nsets
  for(size_t i=0;i < _pdfList.size(); i++) {
     if(pdfs.contains(_pdfList[i])) {
        _pdfList.remove(_pdfList[i]);
        _pdfNSetList.erase(_pdfNSetList.begin()+i);
        i--;
     }
  }

  // Since we may have removed PDFs from the list, the index of the extended
  // PDF in the list needs to be updated. The new index might also be -1 if the
  // extended PDF got removed.
  if(extPdf) {
     _extendedIndex = _pdfList.index(*extPdf);
  }

  // Reset cache
  _cacheMgr.reset() ;
}


namespace {

std::vector<TNamed const*> sortedNamePtrs(RooAbsCollection const& col)
{
   std::vector<TNamed const*> ptrs;
   ptrs.reserve(col.size());
   for(RooAbsArg* arg : col) {
     ptrs.push_back(arg->namePtr());
   }
   std::sort(ptrs.begin(), ptrs.end());
   return ptrs;
}

bool sortedNamePtrsOverlap(std::vector<TNamed const*> const& ptrsA, std::vector<TNamed const*> const& ptrsB)
{
   auto pA = ptrsA.begin();
   auto pB = ptrsB.begin();
   while (pA != ptrsA.end() && pB != ptrsB.end()) {
      if (*pA < *pB) {
          ++pA;
      } else if (*pB < *pA) {
          ++pB;
      } else {
          return true;
      }
   }
   return false;
}

} // namespace


////////////////////////////////////////////////////////////////////////////////
/// Return all parameter constraint p.d.f.s on parameters listed in constrainedParams.
/// The observables set is required to distinguish unambiguously p.d.f in terms
/// of observables and parameters, which are not constraints, and p.d.fs in terms
/// of parameters only, which can serve as constraints p.d.f.s
/// The pdfParams output parameter communicates to the caller which parameter
/// are used in the pdfs that are not constraints.

RooArgSet* RooProdPdf::getConstraints(const RooArgSet& observables, RooArgSet const& constrainedParams, RooArgSet &pdfParams) const
{
  auto constraints = new RooArgSet{"constraints"};

  // For the optimized implementation of checking if two collections overlap by name.
  auto observablesNamePtrs = sortedNamePtrs(observables);
  auto constrainedParamsNamePtrs = sortedNamePtrs(constrainedParams);

  // Loop over PDF components
  for (std::size_t iPdf = 0; iPdf < _pdfList.size(); ++iPdf) {
    auto * pdf = static_cast<RooAbsPdf*>(&_pdfList[iPdf]);

    RooArgSet tmp;
    pdf->getParameters(nullptr, tmp);

    // A constraint term is a p.d.f that doesn't contribute to the
    // expectedEvents() and does not depend on any of the listed observables
    // but does depends on any of the parameters that should be constrained
    bool isConstraint = false;

    if(static_cast<int>(iPdf) != _extendedIndex) {
      auto tmpNamePtrs = sortedNamePtrs(tmp);
      // Before, there were calls to `pdf->dependsOn()` here, but they were very
      // expensive for large computation graphs! Given that we have to traverse
      // the computation graph with a call to `pdf->getParameters()` anyway, we
      // can just check if the set of all variables operlaps with the observables
      // or constraind parameters.
      //
      // We are using an optimized implementation of overlap checking. Because
      // the overlap is checked by name, we can check overlap of the
      // corresponding name pointers. The optimization can't be in
      // RooAbsCollection itself, because it is crucial that the memory for the
      // non-tmp name pointers is not reallocated for each pdf.
      isConstraint = !sortedNamePtrsOverlap(tmpNamePtrs, observablesNamePtrs) &&
                     sortedNamePtrsOverlap(tmpNamePtrs, constrainedParamsNamePtrs);
    }
    if (isConstraint) {
      constraints->add(*pdf) ;
    } else {
      // We only want to add parameter, not observables. Since a call like
      // `pdf->getParameters(&observables)` would be expensive, we take the set
      // of all variables and remove the ovservables, which is much cheaper. In
      // a call to `pdf->getParameters(&observables)`, the observables are
      // matched by name, so we have to pass the `matchByNameOnly` here.
      tmp.remove(observables, /*silent=*/false, /*matchByNameOnly=*/true);
      pdfParams.add(tmp,true) ;
    }
  }

  return constraints;
}




////////////////////////////////////////////////////////////////////////////////
/// Return all parameter constraint p.d.f.s on parameters listed in constrainedParams.
/// The observables set is required to distinguish unambiguously p.d.f in terms
/// of observables and parameters, which are not constraints, and p.d.fs in terms
/// of parameters only, which can serve as constraints p.d.f.s

RooArgSet* RooProdPdf::getConnectedParameters(const RooArgSet& observables) const
{
  RooArgSet* connectedPars  = new RooArgSet("connectedPars") ;
  for (std::size_t iPdf = 0; iPdf < _pdfList.size(); ++iPdf) {
    auto * pdf = static_cast<RooAbsPdf*>(&_pdfList[iPdf]);
    // Check if term is relevant, either because it provides a propablity
    // density in the observables or because it is used for the expected
    // events.
    if (static_cast<int>(iPdf) == _extendedIndex || pdf->dependsOn(observables)) {
      RooArgSet tmp;
      pdf->getParameters(&observables, tmp);
      connectedPars->add(tmp) ;
    }
  }
  return connectedPars ;
}




////////////////////////////////////////////////////////////////////////////////

void RooProdPdf::getParametersHook(const RooArgSet* nset, RooArgSet* params, bool stripDisconnected) const
{
  if (!stripDisconnected) return ;
  if (!nset || nset->empty()) return ;

  // Get/create appropriate term list for this normalization set
  Int_t code = getPartIntList(nset, nullptr);
  RooArgList & plist = static_cast<CacheElem*>(_cacheMgr.getObj(nset, &code))->_partList;

  // Strip any terms from params that do not depend on any term
  RooArgSet tostrip ;
  for (auto param : *params) {
    bool anyDep(false) ;
    for (auto term : plist) {
      if (term->dependsOnValue(*param)) {
        anyDep=true ;
      }
    }
    if (!anyDep) {
      tostrip.add(*param) ;
    }
  }

  if (!tostrip.empty()) {
    params->remove(tostrip,true,true);
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Interface function used by test statistics to freeze choice of range
/// for interpretation of conditional product terms

void RooProdPdf::selectNormalizationRange(const char* rangeName, bool force)
{
  if (!force && _refRangeName) {
    return ;
  }

  fixRefRange(rangeName) ;
}




////////////////////////////////////////////////////////////////////////////////

void RooProdPdf::fixRefRange(const char* rangeName)
{
  _refRangeName = const_cast<TNamed*>(RooNameReg::ptr(rangeName));
}



////////////////////////////////////////////////////////////////////////////////
/// Forward the plot sampling hint from the p.d.f. that defines the observable obs

std::list<double>* RooProdPdf::plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  for (auto const* pdf : static_range_cast<RooAbsPdf*>(_pdfList)) {
    if (std::list<double>* hint = pdf->plotSamplingHint(obs,xlo,xhi)) {
      return hint ;
    }
  }

  return nullptr;
}



////////////////////////////////////////////////////////////////////////////////
/// If all components that depend on obs are binned that so is the product

bool RooProdPdf::isBinnedDistribution(const RooArgSet& obs) const
{
  for (auto const* pdf : static_range_cast<RooAbsPdf*>(_pdfList)) {
    if (pdf->dependsOn(obs) && !pdf->isBinnedDistribution(obs)) {
      return false ;
    }
  }

  return true  ;
}






////////////////////////////////////////////////////////////////////////////////
/// Forward the plot sampling hint from the p.d.f. that defines the observable obs

std::list<double>* RooProdPdf::binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  for (auto const* pdf : static_range_cast<RooAbsPdf*>(_pdfList)) {
    if (std::list<double>* hint = pdf->binBoundaries(obs,xlo,xhi)) {
      return hint ;
    }
  }

  return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Label OK'ed components of a RooProdPdf with cache-and-track, _and_ label all RooProdPdf
/// descendants with extra information about (conditional) normalization, needed to be able
/// to Cache-And-Track them outside the RooProdPdf context.

void RooProdPdf::setCacheAndTrackHints(RooArgSet& trackNodes)
{
  for (const auto parg : _pdfList) {

    if (parg->canNodeBeCached()==Always) {
      trackNodes.add(*parg) ;
//      cout << "tracking node RooProdPdf component " << parg << " " << parg->ClassName() << "::" << parg->GetName() << endl ;

      // Additional processing to fix normalization sets in case product defines conditional observables
      if (RooArgSet* pdf_nset = findPdfNSet(static_cast<RooAbsPdf&>(*parg))) {
        // Check if conditional normalization is specified
        using RooHelpers::getColonSeparatedNameString;
        if (string("nset")==pdf_nset->GetName() && !pdf_nset->empty()) {
          parg->setStringAttribute("CATNormSet",getColonSeparatedNameString(*pdf_nset).c_str()) ;
        }
        if (string("cset")==pdf_nset->GetName()) {
          parg->setStringAttribute("CATCondSet",getColonSeparatedNameString(*pdf_nset).c_str()) ;
        }
      } else {
        coutW(Optimization) << "RooProdPdf::setCacheAndTrackHints(" << GetName() << ") WARNING product pdf does not specify a normalization set for component " << parg->GetName() << endl ;
      }
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a RooProdPdf to more intuitively reflect the contents of the
/// product operator construction

void RooProdPdf::printMetaArgs(ostream& os) const
{
  for (std::size_t i=0 ; i<_pdfList.size() ; i++) {
    if (i>0) os << " * " ;
    RooArgSet* ncset = _pdfNSetList[i].get() ;
    os << _pdfList.at(i)->GetName() ;
    if (!ncset->empty()) {
      if (string("nset")==ncset->GetName()) {
   os << *ncset  ;
      } else {
   os << "|" ;
   bool first(true) ;
   for (auto const* arg : *ncset) {
     if (!first) {
       os << "," ;
     } else {
       first = false ;
     }
     os << arg->GetName() ;
   }
      }
    }
  }
  os << " " ;
}



////////////////////////////////////////////////////////////////////////////////
/// Implement support for node removal

bool RooProdPdf::redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive)
{
  if (nameChange && _pdfList.find("REMOVAL_DUMMY")) {

    cxcoutD(LinkStateMgmt) << "RooProdPdf::redirectServersHook(" << GetName() << "): removing REMOVAL_DUMMY" << endl ;

    // Remove node from _pdfList proxy and remove corresponding entry from normset list
    RooAbsArg* pdfDel = _pdfList.find("REMOVAL_DUMMY") ;

    _pdfNSetList.erase(_pdfNSetList.begin() + _pdfList.index("REMOVAL_DUMMY")) ;
    _pdfList.remove(*pdfDel) ;

    // Clear caches
    _cacheMgr.reset() ;
  }

  // If the replaced server is an observable that is used in any of the
  // normalization sets for conditional fits, replace the element in the
  // normalization set too.
  for(std::unique_ptr<RooArgSet> const& normSet : _pdfNSetList) {
    for(RooAbsArg * arg : *normSet) {
      if(RooAbsArg * newArg = arg->findNewServer(newServerList, nameChange)) {
        // Since normSet is owning, the original arg is now deleted.
        normSet->replace(arg, std::unique_ptr<RooAbsArg>{newArg->cloneTree()});
      }
    }
  }

  return RooAbsPdf::redirectServersHook(newServerList, mustReplaceAll, nameChange, isRecursive);
}

void RooProdPdf::CacheElem::writeToStream(std::ostream& os) const {
  using namespace RooHelpers;
  os << "_partList\n";
  os << getColonSeparatedNameString(_partList) << "\n";
  os << "_numList\n";
  os << getColonSeparatedNameString(_numList) << "\n";
  os << "_denList\n";
  os << getColonSeparatedNameString(_denList) << "\n";
  os << "_ownedList\n";
  os << getColonSeparatedNameString(_ownedList) << "\n";
  os << "_normList\n";
  for(auto const& set : _normList) {
    os << getColonSeparatedNameString(*set) << "\n";
  }
  os << "_isRearranged" << "\n";
  os << _isRearranged << "\n";
  os << "_rearrangedNum" << "\n";
  if(_rearrangedNum) {
    os << getColonSeparatedNameString(*_rearrangedNum) << "\n";
  } else {
    os << "nullptr" << "\n";
  }
  os << "_rearrangedDen" << "\n";
  if(_rearrangedDen) {
    os << getColonSeparatedNameString(*_rearrangedDen) << "\n";
  } else {
    os << "nullptr" << "\n";
  }
}

std::unique_ptr<RooArgSet> RooProdPdf::fillNormSetForServer(RooArgSet const &normSet, RooAbsArg const &server) const
{
   if (normSet.empty())
      return nullptr;
   auto *pdfNset = findPdfNSet(static_cast<RooAbsPdf const &>(server));
   if (pdfNset && !pdfNset->empty()) {
      std::unique_ptr<RooArgSet> out;
      if (0 == strcmp("cset", pdfNset->GetName())) {
         // If the name of the normalization set is "cset", it doesn't contain the
         // normalization set but the conditional observables that should *not* be
         // normalized over.
         out = std::make_unique<RooArgSet>(normSet);
         RooArgSet common;
         out->selectCommon(*pdfNset, common);
         out->remove(common);
      } else {
         out = std::make_unique<RooArgSet>(*pdfNset);
      }
      // prefix also the arguments in the normSets if they have not already been
      if (auto prefix = getStringAttribute("__prefix__")) {
         for (RooAbsArg *arg : *out) {
            if (!arg->getStringAttribute("__prefix__")) {
               arg->SetName((std::string(prefix) + arg->GetName()).c_str());
               arg->setStringAttribute("__prefix__", prefix);
            }
         }
      }
      return out;
   } else {
      return nullptr;
   }
}

std::unique_ptr<RooAbsArg>
RooProdPdf::compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext &ctx) const
{
   if (ctx.likelihoodMode()) {
      auto binnedInfo = RooHelpers::getBinnedL(*this);
      if (binnedInfo.binnedPdf && binnedInfo.binnedPdf != this) {
         return binnedInfo.binnedPdf->compileForNormSet(normSet, ctx);
      }
   }

   std::unique_ptr<RooProdPdf> prodPdfClone{static_cast<RooProdPdf *>(this->Clone())};
   ctx.markAsCompiled(*prodPdfClone);

   for (const auto server : prodPdfClone->servers()) {
      auto nsetForServer = fillNormSetForServer(normSet, *server);
      RooArgSet const &nset = nsetForServer ? *nsetForServer : normSet;

      RooArgSet depList;
      server->getObservables(&nset, depList);

      ctx.compileServer(*server, *prodPdfClone, depList);
   }

   auto fixedProdPdf = std::make_unique<RooFit::Detail::RooFixedProdPdf>(std::move(prodPdfClone), normSet);
   ctx.markAsCompiled(*fixedProdPdf);

   return fixedProdPdf;
}

namespace RooFit {
namespace Detail {

RooFixedProdPdf::RooFixedProdPdf(std::unique_ptr<RooProdPdf> &&prodPdf, RooArgSet const &normSet)
   : RooAbsPdf(prodPdf->GetName(), prodPdf->GetTitle()),
     _normSet{normSet},
     _servers("!servers", "List of servers", this),
     _prodPdf{std::move(prodPdf)}
{
   initialize();
}

RooFixedProdPdf::RooFixedProdPdf(const RooFixedProdPdf &other, const char *name)
   : RooAbsPdf(other, name),
     _normSet{other._normSet},
     _servers("!servers", "List of servers", this),
     _prodPdf{static_cast<RooProdPdf *>(other._prodPdf->Clone())}
{
   initialize();
}

void RooFixedProdPdf::initialize()
{
   _cache = _prodPdf->createCacheElem(&_normSet, nullptr);
   auto &cache = *_cache;

   // The actual servers for a given normalization set depend on whether the
   // cache is rearranged or not. See RooProdPdf::calculateBatch to see
   // which args in the cache are used directly.
   if (cache._isRearranged) {
      _servers.add(*cache._rearrangedNum);
      _servers.add(*cache._rearrangedDen);
   } else {
      for (std::size_t i = 0; i < cache._partList.size(); ++i) {
         _servers.add(cache._partList[i]);
      }
   }
}

} // namespace Detail
} // namespace RooFit
