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

#include "RooAddHelpers.h"
#include "RooAddGenContext.h"
#include "RooBatchCompute.h"
#include "RooDataSet.h"
#include "RooGlobalFunc.h"
#include "RooRealProxy.h"
#include "RooRealVar.h"
#include "RooRealConstant.h"
#include "RooRealSumPdf.h"
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

  auto addRecursiveCoef = [this,&partinCoefList](RooAbsPdf& pdf, RooAbsReal& coef) -> RooAbsReal & {
    partinCoefList.add(coef) ;
    if(partinCoefList.size() == 1) {
      // The first fraction is the first plain fraction
      return coef;
    }
    // The i-th recursive fraction = (1-f1)*(1-f2)*...(fi) and is calculated from the list (f1,...,fi) by RooRecursiveFraction)
    std::stringstream rfracName;
    rfracName << GetName() << "_recursive_fraction_" << pdf.GetName() << "_" << partinCoefList.size();
    auto rfrac = std::make_unique<RooRecursiveFraction>(rfracName.str().c_str(),"Recursive Fraction",partinCoefList) ;
    auto & rfracRef = *rfrac;
    addOwnedComponents(std::move(rfrac)) ;
    return rfracRef;
  };

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
    _coefList.add(recursiveFractions ? addRecursiveCoef(*pdf, *coef) : *coef);
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
      _coefList.add(addRecursiveCoef(*pdf, RooFit::RooConst(1)));
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
  auto* newNamePtr = const_cast<TNamed*>(RooNameReg::ptr(rangeName));
  if(newNamePtr != _refCoefRangeName) {
    _projCacheMgr.reset() ;
  }
  _refCoefRangeName = newNamePtr;
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

AddCacheElem* RooAddPdf::getProjCache(const RooArgSet* nset, const RooArgSet* iset, const char* rangeName) const
{
  // Check if cache already exists
  auto cache = static_cast<AddCacheElem*>(_projCacheMgr.getObj(nset,iset,0,rangeName));
  if (cache) {
    return cache ;
  }

  //Create new cache
  cache = new AddCacheElem{*this, _pdfList, _coefList, nset, iset, rangeName,
                        _projectCoefs, _refCoefNorm, _refCoefRangeName, _verboseEval};

  _projCacheMgr.setObj(nset,iset,cache,RooNameReg::ptr(rangeName)) ;

  return cache;
}


////////////////////////////////////////////////////////////////////////////////
/// Update the coefficient values in the given cache element: calculate new remainder
/// fraction, normalize fractions obtained from extended ML terms to unity, and
/// multiply the various range and dimensional corrections needed in the
/// current use context.
///
/// param[in] cache The cache element for the given normalization set that
///                 stores the supplementary normalization values and
///                 projection-related objects.
/// param[in] nset The set of variables to normalize over.
/// param[in] syncCoefValues If the initial values of the coefficients still
///                          need to be copied from the `_coefList` elements to
///                          the `_coefCache`. True by default.

void RooAddPdf::updateCoefficients(AddCacheElem& cache, const RooArgSet* nset, bool syncCoefValues) const
{
  _coefCache.resize(_pdfList.size());
  if(syncCoefValues) {
    for(std::size_t i = 0; i < _coefList.size(); ++i) {
      _coefCache[i] = static_cast<RooAbsReal const&>(_coefList[i]).getVal(nset);
    }
  }
  RooAddHelpers::updateCoefficients(*this, _coefCache, _pdfList, _haveLastCoef, cache, nset, _projectCoefs,
                                    _refCoefNorm, _allExtendable, _coefErrCount);
}

////////////////////////////////////////////////////////////////////////////////
/// Look up projection cache and per-PDF norm sets. If a PDF doesn't have a special
/// norm set, use the `defaultNorm`. If `defaultNorm == nullptr`, use the member
/// _normSet.
std::pair<const RooArgSet*, AddCacheElem*> RooAddPdf::getNormAndCache(const RooArgSet* nset) const {

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
       coutW(Eval) << "Evaluating RooAddPdf without a defined normalization set. This can lead to ambiguos "
          "coefficients definition and incorrect results."
                           << " Use RooAddPdf::fixCoefNormalization(nset) to provide a normalization set for "
          "defining uniquely RooAddPdf coefficients!"
                           << std::endl;
    }
  }


  AddCacheElem* cache = getProjCache(nset) ;

  return {nset, cache};
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate and return the current value

double RooAddPdf::getValV(const RooArgSet* normSet) const
{
  auto normAndCache = getNormAndCache(normSet);
  const RooArgSet* nset = normAndCache.first;
  AddCacheElem* cache = normAndCache.second;
  updateCoefficients(*cache, nset);

  // Process change in last data set used
  bool nsetChanged(false) ;
  if (RooFit::getUniqueId(nset) != RooFit::getUniqueId(_normSet) || _norm==0) {
    nsetChanged = syncNormalization(nset) ;
  }

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  if (isValueDirty() || nsetChanged) {
    _value = 0.0;

    for (unsigned int i=0; i < _pdfList.size(); ++i) {
      const auto& pdf = static_cast<RooAbsPdf&>(_pdfList[i]);
      double snormVal = 1.;
      snormVal = cache->suppNormVal(i);

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
  _coefCache.resize(_pdfList.size());
  for(std::size_t i = 0; i < _coefList.size(); ++i) {
    auto coefVals = dataMap.at(&_coefList[i]);
    // We don't support per-event coefficients in this function. If the CPU
    // mode is used, we can just fall back to the RooAbsReal implementation.
    // With CUDA, we can't do that because the inputs might be on the device.
    // That's why we throw an exception then.
    if(coefVals.size() > 1) {
      if(stream) {
        throw std::runtime_error("The RooAddPdf doesn't support per-event coefficients in CUDA mode yet!");
      }
      RooAbsReal::computeBatch(stream, output, nEvents, dataMap);
      return;
    }
    _coefCache[i] = coefVals[0];
  }

  RooBatchCompute::VarVector pdfs;
  RooBatchCompute::ArgVector coefs;
  auto normAndCache = getNormAndCache(nullptr);
  const RooArgSet* nset = normAndCache.first;
  AddCacheElem* cache = normAndCache.second;
  // We don't sync the coefficient values from the _coefList to the _coefCache
  // because we have already done it using the dataMap.
  updateCoefficients(*cache, nset, /*syncCoefValues=*/false);

  for (unsigned int pdfNo = 0; pdfNo < _pdfList.size(); ++pdfNo)
  {
    auto pdf = static_cast<RooAbsPdf*>(&_pdfList[pdfNo]);
    if (pdf->isSelectedComp())
    {
      pdfs.push_back(dataMap.at(pdf));
      coefs.push_back(_coefCache[pdfNo] / cache->suppNormVal(pdfNo) );
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
  return RooRealSumPdf::checkObservables(*this, nset, _pdfList, _coefList);
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

double RooAddPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
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

  AddCacheElem* cache = getProjCache(normSet,intSet,0) ; // WVE rangename here?
  updateCoefficients(*cache,normSet);

  // Calculate the current value of this object
  double value(0) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  double snormVal ;

  //cout << "ROP::aIWN updateCoefCache with rangeName = " << (rangeName?rangeName:"<null>") << endl ;
  for (std::size_t i = 0; i < _pdfList.size(); ++i ) {
    auto pdf = static_cast<const RooAbsPdf*>(_pdfList.at(i));

    if (_coefCache[i]) {
      snormVal = cache->suppNormVal(i);

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

double RooAddPdf::expectedEvents(const RooArgSet* nset) const
{
  double expectedTotal{0.0};

  cxcoutD(Caching) << "RooAddPdf::expectedEvents(" << GetName() << ") calling getProjCache with nset = " << (nset?*nset:RooArgSet()) << endl ;
  AddCacheElem& cache = *getProjCache(nset) ;
  updateCoefficients(cache, nset);

  if (cache.doProjection()) {

    for (std::size_t i = 0; i < _pdfList.size(); ++i) {
      double ncomp = _allExtendable ? static_cast<RooAbsPdf&>(_pdfList[i]).expectedEvents(nset)
                                    : static_cast<RooAbsReal&>(_coefList[i]).getVal(nset);
      expectedTotal += cache.rangeProjScaleFactor(i) * ncomp ;

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
  return RooAddGenContext::create(*this,vars,prototype,auxProto,verbose).release();
}



////////////////////////////////////////////////////////////////////////////////
/// Loop over components for plot sampling hints and merge them if there are multiple

std::list<double>* RooAddPdf::plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  return RooRealSumPdf::plotSamplingHint(_pdfList, obs, xlo, xhi);
}


////////////////////////////////////////////////////////////////////////////////
/// Loop over components for plot sampling hints and merge them if there are multiple

std::list<double>* RooAddPdf::binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  return RooRealSumPdf::binBoundaries(_pdfList, obs, xlo, xhi);
}


////////////////////////////////////////////////////////////////////////////////
/// If all components that depend on obs are binned, so is their sum.
bool RooAddPdf::isBinnedDistribution(const RooArgSet& obs) const
{
  return RooRealSumPdf::isBinnedDistribution(_pdfList, obs);
}


////////////////////////////////////////////////////////////////////////////////
/// Label OK'ed components of a RooAddPdf with cache-and-track

void RooAddPdf::setCacheAndTrackHints(RooArgSet& trackNodes)
{
  RooRealSumPdf::setCacheAndTrackHints(_pdfList, trackNodes);
}



////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a RooAddPdf to more intuitively reflect the contents of the
/// product operator construction

void RooAddPdf::printMetaArgs(std::ostream& os) const
{
  RooRealSumPdf::printMetaArgs(_pdfList, _coefList, os);
}
