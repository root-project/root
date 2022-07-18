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

//////////////////////////////////////////////////////////////////////////////
/** \class RooAddPdf
    \ingroup Roofitcore

RooAddPdf is an efficient implementation of a sum of PDFs of the form

\f[
 \sum_{i=1}^{n} c_i \cdot \mathrm{PDF}_i
\f]

or
\f[
 c_1\cdot\mathrm{PDF}_1 + c_2\cdot\mathrm{PDF}_2 \; + \; ... \; + \; \left( 1-\sum_{i=1}^{n-1}c_i \right) \cdot \mathrm{PDF}_n
\f]

The first form is for extended likelihood fits, where the
expected number of events is \f$ \sum_i c_i \f$. The coefficients \f$ c_i \f$
can either be explicitly provided, or, if all components support
extended likelihood fits, they can be calculated from the contribution
of each PDF to the total expected number of events.

In the second form, the sum of the coefficients is required to be 1 or less,
and the coefficient of the last PDF is calculated automatically from the condition
that the sum of all coefficients has to be 1.

### Recursive coefficients
It is also possible to parameterise the coefficients recursively

\f[
 \sum_{i=1}^n c_i \prod_{j=1}^{i-1} \left[ (1-c_j) \right] \cdot \mathrm{PDF}_i \\
 = c_1 \cdot \mathrm{PDF}_1 + (1-c_1)\, c_2 \cdot \mathrm{PDF}_2 + \ldots + (1-c_1)\ldots(1-c_{n-1}) \cdot 1 \cdot \mathrm{PDF}_n \\
\f]

In this form the sum of the coefficients is always less than 1.0
for all possible values of the individual coefficients between 0 and 1.
\note Don't pass the \f$ n^\mathrm{th} \f$ coefficient. It is always 1, since the normalisation condition removes one degree of freedom.

RooAddPdf relies on each component PDF to be normalized and will perform
no normalization other than calculating the proper last coefficient \f$ c_n \f$, if requested.
An (enforced) condition for this assumption is that each \f$ \mathrm{PDF}_i \f$ is independent of each \f$ c_i \f$.

## Difference between RooAddPdf / RooRealSumFunc / RooRealSumPdf
- RooAddPdf is a PDF of PDFs, *i.e.* its components need to be normalised and non-negative.
- RooRealSumPdf is a PDF of functions, *i.e.*, its components can be negative, but their sum cannot be. The normalisation
  is computed automatically, unless the PDF is extended (see above).
- RooRealSumFunc is a sum of functions. It is neither normalised, nor need it be positive.

*/

#include "RooAddPdf.h"

#include "RooAddGenContext.h"
#include "RooBatchCompute.h"
#include "RooDataSet.h"
#include "RooGlobalFunc.h"
#include "RooNaNPacker.h"
#include "RooRealProxy.h"
#include "RooRealVar.h"
#include "RooRealConstant.h"
#include "RooRealIntegral.h"
#include "RooRecursiveFraction.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <set>

using namespace std;

ClassImp(RooAddPdf);


////////////////////////////////////////////////////////////////////////////////
/// Dummy constructor

RooAddPdf::RooAddPdf(const char *name, const char *title) :
  RooAbsPdf(name,title),
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,false,false),
  _projCacheMgr(this,10),
  _pdfList("!pdfs","List of PDFs",this),
  _coefList("!coefficients","List of coefficients",this),
  _coefErrCount{_errorCount}
{
  TRACE_CREATE
}


void RooAddPdf::finalizeConstruction() {

  // Two pdfs with the same name are only allowed in the input list if they are
  // actually the same object.
  using PdfInfo = std::pair<std::string,RooAbsArg*>;
  std::set<PdfInfo> seen;
  for(auto const& pdf : _pdfList) {
    PdfInfo elem{pdf->GetName(), pdf};
    auto comp = [&](PdfInfo const& p){ return p.first == elem.first && p.second != elem.second; };
    auto found = std::find_if(seen.begin(), seen.end(), comp);
    if(found != seen.end()) {
      std::stringstream errorMsg;
      errorMsg << "RooAddPdf::RooAddPdf(" << GetName()
               << ") pdf list contains pdfs with duplicate name \"" << pdf->GetName() << "\"."
               << std::endl;
      coutE(InputArguments) << errorMsg.str();
      throw std::invalid_argument(errorMsg.str().c_str());
    }
    seen.insert(elem);
  }

  _coefCache.resize(_pdfList.size());
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor with two PDFs and one coefficient

RooAddPdf::RooAddPdf(const char *name, const char *title,
		     RooAbsPdf& pdf1, RooAbsPdf& pdf2, RooAbsReal& coef1) :
  RooAddPdf(name, title)
{
  _pdfList.add(pdf1) ;
  _pdfList.add(pdf2) ;
  _coefList.add(coef1) ;

  finalizeConstruction();
}


////////////////////////////////////////////////////////////////////////////////
/// Generic constructor from list of PDFs and list of coefficients.
/// Each pdf list element (i) is paired with coefficient list element (i).
/// The number of coefficients must be either equal to the number of PDFs,
/// in which case extended MLL fitting is enabled, or be one less.
///
/// All PDFs must inherit from RooAbsPdf. All coefficients must inherit from RooAbsReal
///
/// If the recursiveFraction flag is true, the coefficients are interpreted as recursive
/// coefficients as explained in the class description.

RooAddPdf::RooAddPdf(const char *name, const char *title, const RooArgList& inPdfList, const RooArgList& inCoefList, bool recursiveFractions) :
  RooAddPdf(name,title)
{
  _recursive = recursiveFractions;

  if (inPdfList.size()>inCoefList.size()+1 || inPdfList.size()<inCoefList.size()) {
    std::stringstream errorMsg;
    errorMsg << "RooAddPdf::RooAddPdf(" << GetName()
			  << ") number of pdfs and coefficients inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1." << endl ;
    coutE(InputArguments) << errorMsg.str();
    throw std::invalid_argument(errorMsg.str().c_str());
  }

  if (recursiveFractions && inPdfList.size()!=inCoefList.size()+1) {
    std::stringstream errorMsg;
    errorMsg << "RooAddPdf::RooAddPdf(" << GetName()
			  << "): Recursive fractions option can only be used if Npdf=Ncoef+1." << endl;
    coutE(InputArguments) << errorMsg.str();
    throw std::invalid_argument(errorMsg.str());
  }

  // Constructor with N PDFs and N or N-1 coefs
  RooArgList partinCoefList ;

  bool first(true) ;

  for (auto i = 0u; i < inCoefList.size(); ++i) {
    auto coef = dynamic_cast<RooAbsReal*>(inCoefList.at(i));
    auto pdf  = dynamic_cast<RooAbsPdf*>(inPdfList.at(i));
    if (inPdfList.at(i) == nullptr) {
      std::stringstream errorMsg;
      errorMsg << "RooAddPdf::RooAddPdf(" << GetName()
			        << ") number of pdfs and coefficients inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1" << endl ;
      coutE(InputArguments) << errorMsg.str();
      throw std::invalid_argument(errorMsg.str());
    }
    if (!coef) {
      std::stringstream errorMsg;
      errorMsg << "RooAddPdf::RooAddPdf(" << GetName() << ") coefficient " << (coef ? coef->GetName() : "") << " is not of type RooAbsReal, ignored" << endl ;
      coutE(InputArguments) << errorMsg.str();
      throw std::invalid_argument(errorMsg.str());
    }
    if (!pdf) {
      std::stringstream errorMsg;
      errorMsg << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << (pdf ? pdf->GetName() : "") << " is not of type RooAbsPdf, ignored" << endl ;
      coutE(InputArguments) << errorMsg.str();
      throw std::invalid_argument(errorMsg.str());
    }
    _pdfList.add(*pdf) ;

    // Process recursive fraction mode separately
    if (recursiveFractions) {
      partinCoefList.add(*coef) ;
      if (first) {

        // The first fraction is the first plain fraction
        first = false ;
        _coefList.add(*coef) ;

      } else {

        // The i-th recursive fraction = (1-f1)*(1-f2)*...(fi) and is calculated from the list (f1,...,fi) by RooRecursiveFraction)
        RooAbsReal* rfrac = new RooRecursiveFraction(Form("%s_recursive_fraction_%s",GetName(),pdf->GetName()),"Recursive Fraction",partinCoefList) ;
        addOwnedComponents(*rfrac) ;
        _coefList.add(*rfrac) ;

      }

    } else {
      _coefList.add(*coef) ;
    }
  }

  if (inPdfList.size() == inCoefList.size() + 1) {
    auto pdf = dynamic_cast<RooAbsPdf*>(inPdfList.at(inCoefList.size()));

    if (!pdf) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") last argument " << inPdfList.at(inCoefList.size())->GetName() << " is not of type RooAbsPdf." << endl ;
      throw std::invalid_argument("Last argument for RooAddPdf is not a PDF.");
    }
    _pdfList.add(*pdf) ;

    // Process recursive fractions mode. Above, we verified that we don't have a last coefficient
    if (recursiveFractions) {

      // The last recursive fraction = (1-f1)*(1-f2)*...(1-fN) and is calculated from the list (f1,...,fN,1) by RooRecursiveFraction
      partinCoefList.add(RooFit::RooConst(1)) ;
      RooAbsReal* rfrac = new RooRecursiveFraction(Form("%s_recursive_fraction_%s",GetName(),pdf->GetName()),"Recursive Fraction",partinCoefList) ;
      addOwnedComponents(*rfrac) ;
      _coefList.add(*rfrac) ;

      // In recursive mode we always have Ncoef=Npdf, since we added it just above
      _haveLastCoef=true ;
    }

  } else {
    _haveLastCoef=true ;
  }

  finalizeConstruction();
}


////////////////////////////////////////////////////////////////////////////////
/// Generic constructor from list of extended PDFs. There are no coefficients as the expected
/// number of events from each components determine the relative weight of the PDFs.
///
/// All PDFs must inherit from RooAbsPdf.

RooAddPdf::RooAddPdf(const char *name, const char *title, const RooArgList& inPdfList) :
  RooAddPdf(name,title)
{
  _allExtendable = true;

  // Constructor with N PDFs
  for (const auto pdfArg : inPdfList) {
    auto pdf = dynamic_cast<const RooAbsPdf*>(pdfArg);

    if (!pdf) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << (pdf ? pdf->GetName() : "") << " is not of type RooAbsPdf, ignored" << endl ;
      continue ;
    }
    if (!pdf->canBeExtended()) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << pdf->GetName() << " is not extendable, ignored" << endl ;
      continue ;
    }
    _pdfList.add(*pdf) ;
  }

  finalizeConstruction();
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAddPdf::RooAddPdf(const RooAddPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _refCoefNorm("!refCoefNorm",this,other._refCoefNorm),
  _refCoefRangeName((TNamed*)other._refCoefRangeName),
  _projectCoefs(other._projectCoefs),
  _projCacheMgr(other._projCacheMgr,this),
  _codeReg(other._codeReg),
  _pdfList("!pdfs",this,other._pdfList),
  _coefList("!coefficients",this,other._coefList),
  _haveLastCoef(other._haveLastCoef),
  _allExtendable(other._allExtendable),
  _recursive(other._recursive)
{
  _coefErrCount = _errorCount ;
  finalizeConstruction();
  TRACE_CREATE
}


////////////////////////////////////////////////////////////////////////////////
/// By default the interpretation of the fraction coefficients is
/// performed in the contextual choice of observables. This makes the
/// shape of the p.d.f explicitly dependent on the choice of
/// observables. This method instructs RooAddPdf to freeze the
/// interpretation of the coefficients to be done in the given set of
/// observables. If frozen, fractions are automatically transformed
/// from the reference normalization set to the contextual normalization
/// set by ratios of integrals.

void RooAddPdf::fixCoefNormalization(const RooArgSet& refCoefNorm)
{
  if (refCoefNorm.empty()) {
    _projectCoefs = false ;
    return ;
  }
  _projectCoefs = true ;

  _refCoefNorm.removeAll() ;
  _refCoefNorm.add(refCoefNorm) ;

  _projCacheMgr.reset() ;
}



////////////////////////////////////////////////////////////////////////////////
/// By default, fraction coefficients are assumed to refer to the default
/// fit range. This makes the shape of a RooAddPdf
/// explicitly dependent on the range of the observables. Calling this function
/// allows for a range-independent definition of the fractions, because it
/// ties all coefficients to the given
/// named range. If the normalisation range is different
/// from this reference range, the appropriate fraction coefficients
/// are automatically calculated from the reference fractions by
/// integrating over the ranges, and comparing these integrals.

void RooAddPdf::fixCoefRange(const char* rangeName)
{
  _refCoefRangeName = (TNamed*)RooNameReg::ptr(rangeName) ;
  if (_refCoefRangeName) _projectCoefs = true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Retrieve cache element for the computation of the PDF normalisation.
/// \param[in] nset Current normalisation set (integration over these variables yields 1).
/// \param[in] iset Integration set. Variables to be integrated over (if integrations are performed).
/// \param[in] rangeName Reference range for the integrals.
///
/// If a cache element does not exist, create and fill it on the fly. The cache also contains
/// - Supplemental normalization terms (in case not all added p.d.f.s have the same observables)
/// - Projection integrals to calculate transformed fraction coefficients when a frozen reference frame is provided
/// - Projection integrals for similar transformations when a frozen reference range is provided.

RooAddPdf::CacheElem* RooAddPdf::getProjCache(const RooArgSet* nset, const RooArgSet* iset, const char* rangeName) const
{

  // Check if cache already exists
  CacheElem* cache = (CacheElem*) _projCacheMgr.getObj(nset,iset,0,rangeName) ;
  if (cache) {
    return cache ;
  }

  //Create new cache
  cache = new CacheElem ;

  // *** PART 1 : Create supplemental normalization list ***

  // Retrieve the combined set of dependents of this PDF ;
  auto fullDepList = std::unique_ptr<RooArgSet>{getObservables(nset)} ;
  if (iset) {
    fullDepList->remove(*iset,true,true) ;
  }

  // Fill with dummy unit RRVs for now
  for (std::size_t i = 0; i < _pdfList.size(); ++i) {
    auto pdf  = static_cast<const RooAbsPdf *>(_pdfList.at(i));
    auto coef = static_cast<const RooAbsReal*>(_coefList.at(i));

    // Start with full list of dependents
    RooArgSet supNSet(*fullDepList) ;

    // Remove PDF dependents
    if (auto pdfDeps = std::unique_ptr<RooArgSet>{pdf->getObservables(nset)}) {
      supNSet.remove(*pdfDeps,true,true) ;
    }

    // Remove coef dependents
    if (auto coefDeps = std::unique_ptr<RooArgSet>{coef ? coef->getObservables(nset) : nullptr}) {
      supNSet.remove(*coefDeps,true,true) ;
    }

    RooAbsReal* snorm ;
    auto name = std::string(GetName()) + "_" + pdf->GetName() + "_SupNorm";
    cache->_needSupNorm = false ;
    if (!supNSet.empty()) {
      snorm = new RooRealIntegral(name.c_str(),"Supplemental normalization integral",RooRealConstant::value(1.0),supNSet) ;
      cxcoutD(Caching) << "RooAddPdf " << GetName() << " making supplemental normalization set " << supNSet << " for pdf component " << pdf->GetName() << endl ;
      cache->_needSupNorm = true ;
    } else {
      snorm = new RooRealVar(name.c_str(),"Unit Supplemental normalization integral",1.0) ;
    }
    cache->_suppNormList.addOwned(*snorm) ;
  }

  if (_verboseEval>1) {
    cxcoutD(Caching) << "RooAddPdf::syncSuppNormList(" << GetName() << ") synching supplemental normalization list for norm" << (nset?*nset:RooArgSet()) << endl ;
    if dologD(Caching) {
      cache->_suppNormList.Print("v") ;
    }
  }


  // *** PART 2 : Create projection coefficients ***

//   cout << " this = " << this << " (" << GetName() << ")" << endl ;
//   cout << "projectCoefs = " << (_projectCoefs?"T":"F") << endl ;
//   cout << "_normRange.Length() = " << _normRange.Length() << endl ;

  // If no projections required stop here
  if (!_projectCoefs && !rangeName) {
    _projCacheMgr.setObj(nset,iset,cache,RooNameReg::ptr(rangeName)) ;
//     cout << " no projection required" << endl ;
    return cache ;
  }


//   cout << "calculating projection" << endl ;

  // Reduce iset/nset to actual dependents of this PDF
  auto nset2 = std::unique_ptr<RooArgSet>{nset ? getObservables(nset) : new RooArgSet()} ;
  cxcoutD(Caching) << "RooAddPdf(" << GetName() << ")::getPC nset = " << (nset?*nset:RooArgSet()) << " nset2 = " << *nset2 << endl ;

  if (nset2->empty() && !_refCoefNorm.empty()) {
    //cout << "WVE: evaluating RooAddPdf without normalization, but have reference normalization for coefficient definition" << endl ;

    nset2->add(_refCoefNorm) ;
    if (_refCoefRangeName) {
      rangeName = RooNameReg::str(_refCoefRangeName) ;
    }
  }


  // Check if requested transformation is not identity
  if (!nset2->equals(_refCoefNorm) || _refCoefRangeName !=0 || rangeName !=0 || _normRange.Length()>0) {

    cxcoutD(Caching) << "ALEX:     RooAddPdf::syncCoefProjList(" << GetName() << ") projecting coefficients from "
		   << *nset2 << (rangeName?":":"") << (rangeName?rangeName:"")
		   << " to "  << ((!_refCoefNorm.empty())?_refCoefNorm:*nset2) << (_refCoefRangeName?":":"") << (_refCoefRangeName?RooNameReg::str(_refCoefRangeName):"") << endl ;

    // Recalculate projection integrals of PDFs
    for (auto arg : _pdfList) {
      auto thePdf = static_cast<const RooAbsPdf*>(arg);

      // Calculate projection integral
      RooAbsReal* pdfProj ;
      if (!nset2->equals(_refCoefNorm)) {
	pdfProj = thePdf->createIntegral(*nset2,_refCoefNorm,_normRange.Length()>0?_normRange.Data():0) ;
	pdfProj->setOperMode(operMode()) ;
	cxcoutD(Caching) << "RooAddPdf(" << GetName() << ")::getPC nset2(" << *nset2 << ")!=_refCoefNorm(" << _refCoefNorm << ") --> pdfProj = " << pdfProj->GetName() << endl ;
      } else {
        auto name = std::string(GetName()) + "_" + thePdf->GetName() + "_ProjectNorm";
        pdfProj = new RooRealVar(name.c_str(),"Unit Projection normalization integral",1.0) ;
        cxcoutD(Caching) << "RooAddPdf(" << GetName() << ")::getPC nset2(" << *nset2 << ")==_refCoefNorm("
                         << _refCoefNorm << ") --> pdfProj = " << pdfProj->GetName() << std::endl;
      }

      cache->_projList.addOwned(*pdfProj) ;
      cxcoutD(Caching) << " RooAddPdf::syncCoefProjList(" << GetName() << ") PP = " << pdfProj->GetName() << endl ;

      // Calculation optional supplemental normalization term
      RooArgSet supNormSet(_refCoefNorm) ;
      auto deps = std::unique_ptr<RooArgSet>{thePdf->getParameters(RooArgSet())} ;
      supNormSet.remove(*deps,true,true) ;

      RooAbsReal* snorm ;
      auto name = std::string(GetName()) + "_" + thePdf->GetName() + "_ProjSupNorm";
      if (!supNormSet.empty() && !nset2->equals(_refCoefNorm) ) {
        snorm = new RooRealIntegral(name.c_str(),"Projection Supplemental normalization integral",
                                    RooRealConstant::value(1.0),supNormSet) ;
      } else {
        snorm = new RooRealVar(name.c_str(),"Unit Projection Supplemental normalization integral",1.0) ;
      }
      cxcoutD(Caching) << " RooAddPdf::syncCoefProjList(" << GetName() << ") SN = " << snorm->GetName() << endl ;
      cache->_suppProjList.addOwned(*snorm) ;

      // Calculate reference range adjusted projection integral
      RooAbsReal* rangeProj1 ;

   //    cout << "ALEX >>>> RooAddPdf(" << GetName() << ")::getPC _refCoefRangeName WVE = "
// 	   <<(_refCoefRangeName?":":"") << (_refCoefRangeName?RooNameReg::str(_refCoefRangeName):"")
// 	   <<" _refCoefRangeName AK = "  << (_refCoefRangeName?_refCoefRangeName->GetName():"")
// 	   << " && _refCoefNorm" << _refCoefNorm << " with size = _refCoefNorm.size() " << _refCoefNorm.size() << endl ;

      // Check if _refCoefRangeName is identical to default range for all observables,
      // If so, substitute by unit integral

      // ----------
      auto tmpObs = std::unique_ptr<RooArgSet>{thePdf->getObservables(_refCoefNorm)} ;
      bool allIdent = true ;
      for (auto const& obsArg : *tmpObs) {
	RooRealVar* rvarg = dynamic_cast<RooRealVar*>(obsArg) ;
	if (rvarg) {
	  if (rvarg->getMin(RooNameReg::str(_refCoefRangeName))!=rvarg->getMin() ||
	      rvarg->getMax(RooNameReg::str(_refCoefRangeName))!=rvarg->getMax()) {
	    allIdent=false ;
	  }
	}
      }
      // -------------

      if (_refCoefRangeName && !_refCoefNorm.empty() && !allIdent) {


	auto tmp = std::unique_ptr<RooArgSet>{thePdf->getObservables(_refCoefNorm)} ;
	rangeProj1 = thePdf->createIntegral(*tmp,*tmp,RooNameReg::str(_refCoefRangeName)) ;

	//rangeProj1->setOperMode(operMode()) ;

      } else {

	auto theName = std::string(GetName()) + "_" + thePdf->GetName() + "_RangeNorm1";
	rangeProj1 = new RooRealVar(theName.c_str(),"Unit range normalization integral",1.0) ;

      }
      cxcoutD(Caching) << " RooAddPdf::syncCoefProjList(" << GetName() << ") R1 = " << rangeProj1->GetName() << endl ;
      cache->_refRangeProjList.addOwned(*rangeProj1) ;


      // Calculate range adjusted projection integral
      RooAbsReal* rangeProj2 ;
      cxcoutD(Caching) << "RooAddPdf::syncCoefProjList(" << GetName() << ") rangename = " << (rangeName?rangeName:"<null>")
		       << " nset = " << (nset?*nset:RooArgSet()) << endl ;
      if (rangeName && !_refCoefNorm.empty()) {

	rangeProj2 = thePdf->createIntegral(_refCoefNorm,_refCoefNorm,rangeName) ;
	//rangeProj2->setOperMode(operMode()) ;

      } else if (_normRange.Length()>0) {

	auto tmp = std::unique_ptr<RooArgSet>{thePdf->getObservables(_refCoefNorm)} ;
	rangeProj2 = thePdf->createIntegral(*tmp,*tmp,_normRange.Data()) ;

      } else {

	auto theName = std::string(GetName()) + "_" + thePdf->GetName() + "_RangeNorm2";
	rangeProj2 = new RooRealVar(theName.c_str(),"Unit range normalization integral",1.0) ;

      }
      cxcoutD(Caching) << " RooAddPdf::syncCoefProjList(" << GetName() << ") R2 = " << rangeProj2->GetName() << endl ;
      cache->_rangeProjList.addOwned(*rangeProj2) ;

    }

  }

  _projCacheMgr.setObj(nset,iset,cache,RooNameReg::ptr(rangeName)) ;

  return cache ;
}


////////////////////////////////////////////////////////////////////////////////
/// Update the coefficient values in the given cache element: calculate new remainder
/// fraction, normalize fractions obtained from extended ML terms to unity, and
/// multiply the various range and dimensional corrections needed in the
/// current use context.

void RooAddPdf::updateCoefficients(CacheElem& cache, const RooArgSet* nset) const
{
  // Since this function updates the cache, it obviously needs write access:
  auto& myCoefCache = const_cast<std::vector<double>&>(_coefCache);
  myCoefCache.resize(_haveLastCoef ? _coefList.size() : _pdfList.size(), 0.);

  // Straight coefficients
  if (_allExtendable) {

    // coef[i] = expectedEvents[i] / SUM(expectedEvents)
    double coefSum(0) ;
    std::size_t i = 0;
    for (auto arg : _pdfList) {
      auto pdf = static_cast<RooAbsPdf*>(arg);
      myCoefCache[i] = pdf->expectedEvents(!_refCoefNorm.empty()?&_refCoefNorm:nset) ;
      coefSum += myCoefCache[i] ;
      i++ ;
    }

    if (coefSum==0.) {
      coutW(Eval) << "RooAddPdf::updateCoefCache(" << GetName() << ") WARNING: total number of expected events is 0" << endl ;
    } else {
      for (std::size_t j=0; j < _pdfList.size(); j++) {
        myCoefCache[j] /= coefSum ;
      }
    }

  } else {
    if (_haveLastCoef) {

      // coef[i] = coef[i] / SUM(coef)
      double coefSum(0) ;
      std::size_t i=0;
      for (auto coefArg : _coefList) {
        auto coef = static_cast<RooAbsReal*>(coefArg);
        myCoefCache[i] = coef->getVal(nset) ;
        coefSum += myCoefCache[i++];
      }
      if (coefSum==0.) {
        coutW(Eval) << "RooAddPdf::updateCoefCache(" << GetName() << ") WARNING: sum of coefficients is zero 0" << endl ;
      } else {
        const double invCoefSum = 1./coefSum;
        for (std::size_t j=0; j < _coefList.size(); j++) {
          myCoefCache[j] *= invCoefSum;
        }
      }
    } else {

      // coef[i] = coef[i] ; coef[n] = 1-SUM(coef[0...n-1])
      double lastCoef(1) ;
      std::size_t i=0;
      for (auto coefArg : _coefList) {
        auto coef = static_cast<RooAbsReal*>(coefArg);
        myCoefCache[i] = coef->getVal(nset) ;
        lastCoef -= myCoefCache[i++];
      }
      myCoefCache[_coefList.size()] = lastCoef ;

      // Treat coefficient degeneration
      const float coefDegen = lastCoef < 0. ? -lastCoef : (lastCoef > 1. ? lastCoef - 1. : 0.);
      if (coefDegen > 1.E-5) {
        myCoefCache[_coefList.size()] = RooNaNPacker::packFloatIntoNaN(100.f*coefDegen);

        std::stringstream msg;
        if (_coefErrCount-->0) {
          msg << "RooAddPdf::updateCoefCache(" << GetName()
              << " WARNING: sum of PDF coefficients not in range [0-1], value="
		      << 1-lastCoef ;
          if (_coefErrCount==0) {
            msg << " (no more will be printed)"  ;
          }
          coutW(Eval) << msg.str() << std::endl;
        }
      }
    }
  }


  // Stop here if not projection is required or needed
  if ((!_projectCoefs && _normRange.Length()==0) || cache._projList.empty()) {
    return ;
  }

  // Adjust coefficients for given projection
  double coefSum(0) ;
  {
    RooAbsReal::GlobalSelectComponentRAII compRAII(true);

    for (std::size_t i = 0; i < _pdfList.size(); i++) {

      RooAbsReal* pp = ((RooAbsReal*)cache._projList.at(i)) ;
      RooAbsReal* sn = ((RooAbsReal*)cache._suppProjList.at(i)) ;
      RooAbsReal* r1 = ((RooAbsReal*)cache._refRangeProjList.at(i)) ;
      RooAbsReal* r2 = ((RooAbsReal*)cache._rangeProjList.at(i)) ;

      double proj = pp->getVal()/sn->getVal()*(r2->getVal()/r1->getVal()) ;

      myCoefCache[i] *= proj ;
      coefSum += myCoefCache[i] ;
    }
  }


  if ((RooMsgService::_debugCount>0) && RooMsgService::instance().isActive(this,RooFit::Caching,RooFit::DEBUG)) {
    for (std::size_t i=0; i < _pdfList.size(); ++i) {
      ccoutD(Caching) << " ALEX:   POST-SYNC coef[" << i << "] = " << myCoefCache[i]
                      << " ( _coefCache[i]/coefSum = " << myCoefCache[i]*coefSum << "/" << coefSum << " ) "<< endl ;
    }
  }

  if (coefSum==0.) {
    coutE(Eval) << "RooAddPdf::updateCoefCache(" << GetName() << ") sum of coefficients is zero." << endl ;
  }

  for (std::size_t i=0; i < _pdfList.size(); i++) {
    myCoefCache[i] /= coefSum ;
  }

}

////////////////////////////////////////////////////////////////////////////////
/// Look up projection cache and per-PDF norm sets. If a PDF doesn't have a special
/// norm set, use the `defaultNorm`. If `defaultNorm == nullptr`, use the member
/// _normSet.
std::pair<const RooArgSet*, RooAddPdf::CacheElem*> RooAddPdf::getNormAndCache(const RooArgSet* nset) const {

  // Treat empty normalization set and nullptr the same way.
  if(nset && nset->empty()) nset = nullptr;

  if (nset == nullptr) {
    if (!_refCoefNorm.empty()) {
      nset = &_refCoefNorm ;
    }
  }

  // A RooAddPdf needs to have a normalization set defined, otherwise its
  // coefficient will not be uniquely defined. Its shape depends on the
  // normalization provided. Un-normalized calls to RooAddPdf can happen in
  // Roofit, when printing the pdf's or when computing integrals. In these case,
  // if the pdf has a normalization set previously defined (i.e. stored as a
  // datamember in _copyOfLastNormSet) it should use it by default when the pdf
  // is evaluated without passing a normalizations set (in pdf->getVal(nullptr) )
  // In the case of no pre-defined normalization set exists, a warning will be
  // produced, since the obtained value will be arbitrary. Note that to avoid
  // unnecessary warning messages, when calling RooAbsPdf::printValue or
  // RooAbsPdf::graphVizTree, the printing of the warning messages for the
  // RooFit::Eval topic is explicitly disabled.
  {
    // If nset is still nullptr, get the pointer to a copy of the last-used
    // normalization set.  It nset is not nullptr, check whether the copy of
    // the last-used normalization set needs an update.
    if(nset == nullptr) {
      nset = _copyOfLastNormSet.get();
    } else if(nset->uniqueId() != _idOfLastUsedNormSet) {
      _copyOfLastNormSet = std::make_unique<const RooArgSet>(*nset);
      _idOfLastUsedNormSet = nset->uniqueId();
    }

    // If nset is STILL nullptr, print a warning.
    if (nset == nullptr) {
       oocoutW(this, Eval) << "Evaluating RooAddPdf without a defined normalization set. This can lead to ambiguos "
          "coefficients definition and incorrect results."
                           << " Use RooAddPdf::fixCoefNormalization(nset) to provide a normalization set for "
          "defining uniquely RooAddPdf coefficients!"
                           << std::endl;
    }
  }


  CacheElem* cache = getProjCache(nset) ;
  updateCoefficients(*cache,nset) ;

  return {nset, cache};
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate and return the current value

double RooAddPdf::getValV(const RooArgSet* normSet) const
{
  auto normAndCache = getNormAndCache(normSet);
  const RooArgSet* nset = normAndCache.first;
  CacheElem* cache = normAndCache.second;

  // Process change in last data set used
  bool nsetChanged(false) ;
  if (nset!=_normSet || _norm==0) {
    nsetChanged = syncNormalization(nset) ;
  }

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  if (isValueDirty() || nsetChanged) {
    _value = 0.0;

    for (unsigned int i=0; i < _pdfList.size(); ++i) {
      const auto& pdf = static_cast<RooAbsPdf&>(_pdfList[i]);
      double snormVal = 1.;
      if (cache->_needSupNorm) {
        snormVal = ((RooAbsReal*)cache->_suppNormList.at(i))->getVal();
      }

      double pdfVal = pdf.getVal(nset);
      if (pdf.isSelectedComp()) {
        _value += pdfVal*_coefCache[i]/snormVal;
      }
    }
    clearValueAndShapeDirty();
  }

  return _value;
}


////////////////////////////////////////////////////////////////////////////////
/// Compute addition of PDFs in batches.
void RooAddPdf::computeBatch(cudaStream_t* stream, double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const
{
  RooBatchCompute::VarVector pdfs;
  RooBatchCompute::ArgVector coefs;
  CacheElem* cache = getNormAndCache(nullptr).second;
  for (unsigned int pdfNo = 0; pdfNo < _pdfList.size(); ++pdfNo)
  {
    auto pdf = static_cast<RooAbsPdf*>(&_pdfList[pdfNo]);
    if (pdf->isSelectedComp())
    {
      pdfs.push_back(dataMap.at(pdf));
      coefs.push_back(_coefCache[pdfNo] / (cache->_needSupNorm ?
        static_cast<RooAbsReal*>(cache->_suppNormList.at(pdfNo))->getVal() : 1) );
    }
  }
  auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;
  dispatch->compute(stream, RooBatchCompute::AddPdf, output, nEvents, pdfs, coefs);
}


////////////////////////////////////////////////////////////////////////////////
/// Reset error counter to given value, limiting the number
/// of future error messages for this pdf to 'resetValue'

void RooAddPdf::resetErrorCounters(Int_t resetValue)
{
  RooAbsPdf::resetErrorCounters(resetValue) ;
  _coefErrCount = resetValue ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if PDF is valid for given normalization set.
/// Coeffient and PDF must be non-overlapping, but pdf-coefficient
/// pairs may overlap each other

bool RooAddPdf::checkObservables(const RooArgSet* nset) const
{
  bool ret(false) ;

  // There may be fewer coefficients than PDFs.
  std::size_t end = std::min(_pdfList.size(), _coefList.size());
  for (std::size_t i = 0; i < end; ++i) {
    auto pdf  = static_cast<const RooAbsPdf *>(_pdfList.at(i));
    auto coef = static_cast<const RooAbsReal*>(_coefList.at(i));
    if (pdf->observableOverlaps(nset,*coef)) {
      coutE(InputArguments) << "RooAddPdf::checkObservables(" << GetName() << "): ERROR: coefficient " << coef->GetName()
			    << " and PDF " << pdf->GetName() << " have one or more dependents in common" << endl ;
      ret = true ;
    }
  }

  return ret ;
}


////////////////////////////////////////////////////////////////////////////////
/// Determine which part (if any) of given integral can be performed analytically.
/// If any analytical integration is possible, return integration scenario code
///
/// RooAddPdf queries each component PDF for its analytical integration capability of the requested
/// set ('allVars'). It finds the largest common set of variables that can be integrated
/// by all components. If such a set exists, it reconfirms that each component is capable of
/// analytically integrating the common set, and combines the components individual integration
/// codes into a single integration code valid for RooAddPdf.

Int_t RooAddPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars,
					 const RooArgSet* normSet, const char* rangeName) const
{

  RooArgSet allAnalVars(*std::unique_ptr<RooArgSet>{getObservables(allVars)}) ;

  Int_t n(0) ;

  // First iteration, determine what each component can integrate analytically
  for (const auto pdfArg : _pdfList) {
    auto pdf = static_cast<const RooAbsPdf *>(pdfArg);
    RooArgSet subAnalVars ;
    pdf->getAnalyticalIntegralWN(allVars,subAnalVars,normSet,rangeName) ;

    // Observables that cannot be integrated analytically by this component are dropped from the common list
    for (const auto arg : allVars) {
      if (!subAnalVars.find(arg->GetName()) && pdf->dependsOn(*arg)) {
        allAnalVars.remove(*arg,true,true) ;
      }
    }
    n++ ;
  }

  // If no observables can be integrated analytically, return code 0 here
  if (allAnalVars.empty()) {
    return 0 ;
  }


  // Now retrieve codes for integration over common set of analytically integrable observables for each component
  n=0 ;
  std::vector<Int_t> subCode(_pdfList.size());
  bool allOK(true) ;
  for (const auto arg : _pdfList) {
    auto pdf = static_cast<const RooAbsPdf *>(arg);
    RooArgSet subAnalVars ;
    auto allAnalVars2 = std::unique_ptr<RooArgSet>{pdf->getObservables(allAnalVars)} ;
    subCode[n] = pdf->getAnalyticalIntegralWN(*allAnalVars2,subAnalVars,normSet,rangeName) ;
    if (subCode[n]==0 && !allAnalVars2->empty()) {
      coutE(InputArguments) << "RooAddPdf::getAnalyticalIntegral(" << GetName() << ") WARNING: component PDF " << pdf->GetName()
			    << "   advertises inconsistent set of integrals (e.g. (X,Y) but not X or Y individually."
			    << "   Distributed analytical integration disabled. Please fix PDF" << endl ;
      allOK = false ;
    }
    n++ ;
  }
  if (!allOK) {
    return 0 ;
  }

  // Mare all analytically integrated observables as such
  analVars.add(allAnalVars) ;

  // Store set of variables analytically integrated
  RooArgSet* intSet = new RooArgSet(allAnalVars) ;
  Int_t masterCode = _codeReg.store(subCode,intSet)+1 ;

  return masterCode ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return analytical integral defined by given scenario code

Double_t RooAddPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
  // WVE needs adaptation to handle new rangeName feature
  if (code==0) {
    return getVal(normSet) ;
  }

  // Retrieve analytical integration subCodes and set of observabels integrated over
  RooArgSet* intSet ;
  const std::vector<Int_t>& subCode = _codeReg.retrieve(code-1,intSet) ;
  if (subCode.empty()) {
    coutE(InputArguments) << "RooAddPdf::analyticalIntegral(" << GetName() << "): ERROR unrecognized integration code, " << code << endl ;
    assert(0) ;
  }

  cxcoutD(Caching) << "RooAddPdf::aiWN(" << GetName() << ") calling getProjCache with nset = " << (normSet?*normSet:RooArgSet()) << endl ;

  if ((normSet==0 || normSet->empty()) && !_refCoefNorm.empty()) {
//     cout << "WVE integration of RooAddPdf without normalization, but have reference set, using ref set for normalization" << endl ;
    normSet = &_refCoefNorm ;
  }

  CacheElem* cache = getProjCache(normSet,intSet,0) ; // WVE rangename here?
  updateCoefficients(*cache,normSet) ;

  // Calculate the current value of this object
  double value(0) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  double snormVal ;

  //cout << "ROP::aIWN updateCoefCache with rangeName = " << (rangeName?rangeName:"<null>") << endl ;
  RooArgList* snormSet = (!cache->_suppNormList.empty()) ? &cache->_suppNormList : nullptr ;
  for (std::size_t i = 0; i < _pdfList.size(); ++i ) {
    auto pdf = static_cast<const RooAbsPdf*>(_pdfList.at(i));

    if (_coefCache[i]) {
      snormVal = snormSet ? ((RooAbsReal*) cache->_suppNormList.at(i))->getVal() : 1.0 ;

      // WVE swap this?
      double val = pdf->analyticalIntegralWN(subCode[i],normSet,rangeName) ;
      if (pdf->isSelectedComp()) {
        value += val*_coefCache[i]/snormVal ;
      }
    }
  }

  return value ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the number of expected events, which is either the sum of all coefficients
/// or the sum of the components extended terms, multiplied with the fraction that
/// is in the current range w.r.t the reference range

Double_t RooAddPdf::expectedEvents(const RooArgSet* nset) const
{
  double expectedTotal{0.0};

  cxcoutD(Caching) << "RooAddPdf::expectedEvents(" << GetName() << ") calling getProjCache with nset = " << (nset?*nset:RooArgSet()) << endl ;
  CacheElem& cache = *getProjCache(nset) ;
  updateCoefficients(cache,nset) ;

  if (!cache._rangeProjList.empty()) {

    for (std::size_t i = 0; i < _pdfList.size(); ++i) {
      auto const& r1 = static_cast<RooAbsReal&>(cache._refRangeProjList[i]);
      auto const& r2 = static_cast<RooAbsReal&>(cache._rangeProjList[i]);
      double ncomp = _allExtendable ? static_cast<RooAbsPdf&>(_pdfList[i]).expectedEvents(nset)
                                    : static_cast<RooAbsReal&>(_coefList[i]).getVal(nset);
      expectedTotal += (r2.getVal()/r1.getVal()) * ncomp ;

    }

  } else {

    if (_allExtendable) {
      for(auto const& arg : _pdfList) {
        expectedTotal += static_cast<RooAbsPdf*>(arg)->expectedEvents(nset) ;
      }
    } else {
      for(auto const& arg : _coefList) {
        expectedTotal += static_cast<RooAbsReal*>(arg)->getVal(nset) ;
      }
    }

  }
  return expectedTotal ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface function used by test statistics to freeze choice of observables
/// for interpretation of fraction coefficients

void RooAddPdf::selectNormalization(const RooArgSet* depSet, bool force)
{

  if (!force && !_refCoefNorm.empty()) {
    return ;
  }

  if (!depSet) {
    fixCoefNormalization(RooArgSet()) ;
    return ;
  }

  fixCoefNormalization(*std::unique_ptr<RooArgSet>{getObservables(depSet)}) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface function used by test statistics to freeze choice of range
/// for interpretation of fraction coefficients

void RooAddPdf::selectNormalizationRange(const char* rangeName, bool force)
{
  if (!force && _refCoefRangeName) {
    return ;
  }

  fixCoefRange(rangeName) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return specialized context to efficiently generate toy events from RooAddPdfs
/// return RooAbsPdf::genContext(vars,prototype,auxProto,verbose) ; // WVE DEBUG

RooAbsGenContext* RooAddPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype,
					const RooArgSet* auxProto, bool verbose) const
{
  return new RooAddGenContext(*this,vars,prototype,auxProto,verbose) ;
}



////////////////////////////////////////////////////////////////////////////////
/// List all RooAbsArg derived contents in this cache element

RooArgList RooAddPdf::CacheElem::containedArgs(Action)
{
  RooArgList allNodes;
  allNodes.add(_projList) ;
  allNodes.add(_suppProjList) ;
  allNodes.add(_refRangeProjList) ;
  allNodes.add(_rangeProjList) ;

  return allNodes ;
}



////////////////////////////////////////////////////////////////////////////////
/// Loop over components for plot sampling hints and merge them if there are multiple

std::list<Double_t>* RooAddPdf::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  std::unique_ptr<std::list<Double_t>> sumHint = nullptr ;
  bool needClean = false;

  // Loop over components pdf
  for (const auto arg : _pdfList) {
    auto pdf = static_cast<const RooAbsPdf*>(arg);

    std::unique_ptr<std::list<Double_t>> pdfHint{pdf->plotSamplingHint(obs,xlo,xhi)} ;

    // Process hint
    if (pdfHint) {
      if (!sumHint) {

	// If this is the first hint, then just save it
	sumHint = std::move(pdfHint) ;

      } else {

	auto newSumHint = std::make_unique<std::list<Double_t>>(sumHint->size()+pdfHint->size());

	// Merge hints into temporary array
	merge(pdfHint->begin(),pdfHint->end(),sumHint->begin(),sumHint->end(),newSumHint->begin()) ;

	// Copy merged array without duplicates to new sumHintArrau
	sumHint = std::move(newSumHint) ;
	needClean = true ;

      }
    }
  }
  if (needClean) {
    sumHint->erase(std::unique(sumHint->begin(),sumHint->end()), sumHint->end()) ;
  }

  return sumHint.release() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Loop over components for plot sampling hints and merge them if there are multiple

std::list<Double_t>* RooAddPdf::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  std::unique_ptr<list<Double_t>> sumBinB = nullptr ;
  bool needClean = false;

  // Loop over components pdf
  for (auto arg : _pdfList) {
    auto pdf = static_cast<const RooAbsPdf *>(arg);
    std::unique_ptr<list<Double_t>> pdfBinB{pdf->binBoundaries(obs,xlo,xhi)};

    // Process hint
    if (pdfBinB) {
      if (!sumBinB) {

	// If this is the first hint, then just save it
	sumBinB = std::move(pdfBinB) ;

      } else {

	auto newSumBinB = std::make_unique<list<Double_t>>(sumBinB->size()+pdfBinB->size()) ;

	// Merge hints into temporary array
	merge(pdfBinB->begin(),pdfBinB->end(),sumBinB->begin(),sumBinB->end(),newSumBinB->begin()) ;

	// Copy merged array without duplicates to new sumBinBArrau
	sumBinB = std::move(newSumBinB) ;
	needClean = true ;
      }
    }
  }

  // Remove consecutive duplicates
  if (needClean) {
    sumBinB->erase(std::unique(sumBinB->begin(),sumBinB->end()), sumBinB->end()) ;
  }

  return sumBinB.release() ;
}


////////////////////////////////////////////////////////////////////////////////
/// If all components that depend on obs are binned, so is their sum.
bool RooAddPdf::isBinnedDistribution(const RooArgSet& obs) const
{
  for (const auto arg : _pdfList) {
    auto pdf = static_cast<const RooAbsPdf*>(arg);
    if (pdf->dependsOn(obs) && !pdf->isBinnedDistribution(obs)) {
      return false ;
    }
  }

  return true  ;
}


////////////////////////////////////////////////////////////////////////////////
/// Label OK'ed components of a RooAddPdf with cache-and-track

void RooAddPdf::setCacheAndTrackHints(RooArgSet& trackNodes)
{
  RooFIter aiter = pdfList().fwdIterator() ;
  RooAbsArg* aarg ;
  while ((aarg=aiter.next())) {
    if (aarg->canNodeBeCached()==Always) {
      trackNodes.add(*aarg) ;
      //cout << "tracking node RooAddPdf component " << aarg->IsA()->GetName() << "::" << aarg->GetName() << endl ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a RooAddPdf to more intuitively reflect the contents of the
/// product operator construction

void RooAddPdf::printMetaArgs(ostream& os) const
{
  bool first(true) ;

  if (!_coefList.empty()) {
    for (std::size_t i = 0; i < _pdfList.size(); ++i ) {
      const RooAbsArg * coef = _coefList.at(i);
      const RooAbsArg * pdf  = _pdfList.at(i);
      if (!first) {
        os << " + " ;
      } else {
        first = false ;
      }

      if (i < _coefList.size()) {
        os << coef->GetName() << " * " << pdf->GetName();
      } else {
        os << "[%] * " << pdf->GetName();
      }
    }
  } else {

    for (const auto pdf : _pdfList) {
      if (!first) {
        os << " + " ;
      } else {
        first = false ;
      }
      os << pdf->GetName() ;
    }
  }

  os << " " ;
}
