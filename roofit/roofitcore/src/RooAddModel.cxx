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
/// \class RooAddModel
///
/// RooAddModel is an efficient implementation of a sum of PDFs of the form
/// \f[
///  c_1 \cdot \mathrm{PDF}_1 + c_2 \cdot \mathrm{PDF}_2 + ... + c_n \cdot \mathrm{PDF}_n
/// \f]
/// or
/// \f[
///  c_1 \cdot \mathrm{PDF}_1 + c_2 \cdot \mathrm{PDF}_2 + ... + \left( 1-\sum_{i=1}^{n-1} c_i \right) \cdot \mathrm{PDF}_n
/// \f]
/// The first form is for extended likelihood fits, where the
/// expected number of events is \f$ \sum_i c_i \f$. The coefficients \f$ c_i \f$
/// can either be explicitly provided, or, if all components support
/// extended likelihood fits, they can be calculated from the contribution
/// of each PDF to the total number of expected events.
///
/// In the second form, the sum of the coefficients is enforced to be one,
/// and the coefficient of the last PDF is calculated from that condition.
///
/// RooAddModel relies on each component PDF to be normalized, and will perform
/// no normalization other than calculating the proper last coefficient \f$ c_n \f$, if requested.
/// An (enforced) condition for this assumption is that each \f$ \mathrm{PDF}_i \f$ is independent
/// of each coefficient \f$ i \f$.
///
///

#include "RooAddModel.h"

#include "RooAddHelpers.h"
#include "RooMsgService.h"
#include "RooDataSet.h"
#include "RooRealProxy.h"
#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooAddGenContext.h"
#include "RooNameReg.h"
#include "RooBatchCompute.h"

using namespace std;

ClassImp(RooAddModel);


////////////////////////////////////////////////////////////////////////////////

RooAddModel::RooAddModel() :
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,false,false),
  _projCacheMgr(this,10),
  _intCacheMgr(this,10)
{
  _coefErrCount = _errorCount ;
}



////////////////////////////////////////////////////////////////////////////////
/// Generic constructor from list of PDFs and list of coefficients.
/// Each pdf list element (i) is paired with coefficient list element (i).
/// The number of coefficients must be either equal to the number of PDFs,
/// in which case extended MLL fitting is enabled, or be one less.
///
/// All PDFs must inherit from RooAbsPdf. All coefficients must inherit from RooAbsReal.

RooAddModel::RooAddModel(const char *name, const char *title, const RooArgList& inPdfList, const RooArgList& inCoefList, bool ownPdfList) :
  RooResolutionModel(name,title,(static_cast<RooResolutionModel*>(inPdfList.at(0)))->convVar()),
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,false,false),
  _refCoefRangeName(0),
  _projCacheMgr(this,10),
  _intCacheMgr(this,10),
  _codeReg(10),
  _pdfList("!pdfs","List of PDFs",this),
  _coefList("!coefficients","List of coefficients",this),
  _haveLastCoef(false),
  _allExtendable(false)
{
   const std::string ownName(GetName() ? GetName() : "");
   if (inPdfList.size() > inCoefList.size() + 1 || inPdfList.size() < inCoefList.size()) {
      std::stringstream msgSs;
      msgSs << "RooAddModel::RooAddModel(" << ownName
            << ") number of pdfs and coefficients inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1";
      const std::string msgStr = msgSs.str();
      coutE(InputArguments) << msgStr << "\n";
      throw std::runtime_error(msgStr);
   }

   // Constructor with N PDFs and N or N-1 coefs
   std::size_t i = 0;
   for (auto const &coef : inCoefList) {
      auto pdf = inPdfList.at(i);
      if (!pdf) {
         std::stringstream msgSs;
         msgSs << "RooAddModel::RooAddModel(" << ownName
               << ") number of pdfs and coefficients inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1";
         const std::string msgStr = msgSs.str();
         coutE(InputArguments) << msgStr << "\n";
         throw std::runtime_error(msgStr);
      }
      if (!coef) {
         coutE(InputArguments) << "RooAddModel::RooAddModel(" << ownName
                               << ") encountered and undefined coefficient, ignored\n";
         continue;
      }
      if (!dynamic_cast<RooAbsReal *>(coef)) {
         auto coefName = coef->GetName();
         coutE(InputArguments) << "RooAddModel::RooAddModel(" << ownName << ") coefficient "
                               << (coefName != nullptr ? coefName : "") << " is not of type RooAbsReal, ignored\n";
         continue;
      }
      if (!dynamic_cast<RooAbsPdf *>(pdf)) {
         coutE(InputArguments) << "RooAddModel::RooAddModel(" << ownName << ") pdf "
                               << (pdf->GetName() ? pdf->GetName() : "") << " is not of type RooAbsPdf, ignored\n";
         continue;
      }
      _pdfList.add(*pdf);
      _coefList.add(*coef);
      i++;
   }

   if (i < inPdfList.size()) {
      auto pdf = inPdfList.at(i);
      if (!dynamic_cast<RooAbsPdf *>(pdf)) {
         std::stringstream msgSs;
         msgSs << "RooAddModel::RooAddModel(" << ownName << ") last pdf " << (pdf->GetName() ? pdf->GetName() : "")
               << " is not of type RooAbsPdf, fatal error";
         const std::string msgStr = msgSs.str();
         coutE(InputArguments) << msgStr << "\n";
         throw std::runtime_error(msgStr);
      }
      _pdfList.add(*pdf);
   } else {
      _haveLastCoef = true;
   }

   _coefErrCount = _errorCount;

   if (ownPdfList) {
      _ownedComps.addOwned(_pdfList);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAddModel::RooAddModel(const RooAddModel& other, const char* name) :
  RooResolutionModel(other,name),
  _refCoefNorm("!refCoefNorm",this,other._refCoefNorm),
  _refCoefRangeName((TNamed*)other._refCoefRangeName),
  _projCacheMgr(other._projCacheMgr,this),
  _intCacheMgr(other._intCacheMgr,this),
  _codeReg(other._codeReg),
  _pdfList("!pdfs",this,other._pdfList),
  _coefList("!coefficients",this,other._coefList),
  _haveLastCoef(other._haveLastCoef),
  _allExtendable(other._allExtendable)
{
  _coefErrCount = _errorCount ;
}



////////////////////////////////////////////////////////////////////////////////
/// By default the interpretation of the fraction coefficients is
/// performed in the contextual choice of observables. This makes the
/// shape of the p.d.f explicitly dependent on the choice of
/// observables. This method instructs RooAddModel to freeze the
/// interpretation of the coefficients to be done in the given set of
/// observables. If frozen, fractions are automatically transformed
/// from the reference normalization set to the contextual normalization
/// set by ratios of integrals

void RooAddModel::fixCoefNormalization(const RooArgSet& refCoefNorm)
{
  if (refCoefNorm.empty()) {
    return ;
  }

  _refCoefNorm.removeAll() ;
  _refCoefNorm.add(refCoefNorm) ;

  _projCacheMgr.reset() ;
}



////////////////////////////////////////////////////////////////////////////////
/// By default the interpretation of the fraction coefficients is
/// performed in the default range. This make the shape of a RooAddModel
/// explicitly dependent on the range of the observables. To allow
/// a range independent definition of the fraction this function
/// instructs RooAddModel to freeze its interpretation in the given
/// named range. If the current normalization range is different
/// from the reference range, the appropriate fraction coefficients
/// are automically calculation from the reference fractions using
/// ratios if integrals

void RooAddModel::fixCoefRange(const char* rangeName)
{
  _refCoefRangeName = (TNamed*)RooNameReg::ptr(rangeName) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Instantiate a clone of this resolution model representing a convolution with given
/// basis function. The owners object name is incorporated in the clones name
/// to avoid multiple convolution objects with the same name in complex PDF structures.
///
/// RooAddModel will clone all the component models to create a composite convolution object

RooResolutionModel* RooAddModel::convolution(RooFormulaVar* inBasis, RooAbsArg* owner) const
{
  // Check that primary variable of basis functions is our convolution variable
  if (inBasis->getParameter(0) != x.absArg()) {
    coutE(InputArguments) << "RooAddModel::convolution(" << GetName()
           << ") convolution parameter of basis function and PDF don't match" << endl ;
    ccoutE(InputArguments) << "basis->findServer(0) = " << inBasis->findServer(0) << " " << inBasis->findServer(0)->GetName() << endl ;
    ccoutE(InputArguments) << "x.absArg()           = " << x.absArg() << " " << x.absArg()->GetName() << endl ;
    inBasis->Print("v") ;
    return 0 ;
  }

  TString newName(GetName()) ;
  newName.Append("_conv_") ;
  newName.Append(inBasis->GetName()) ;
  newName.Append("_[") ;
  newName.Append(owner->GetName()) ;
  newName.Append("]") ;

  TString newTitle(GetTitle()) ;
  newTitle.Append(" convoluted with basis function ") ;
  newTitle.Append(inBasis->GetName()) ;

  RooArgList modelList ;
  for (auto obj : _pdfList) {
    auto model = static_cast<RooResolutionModel*>(obj);
    // Create component convolution
    RooResolutionModel* conv = model->convolution(inBasis,owner) ;
    modelList.add(*conv) ;
  }

  RooArgList theCoefList ;
  for (auto coef : _coefList) {
    theCoefList.add(*coef) ;
  }

  RooAddModel* convSum = new RooAddModel(newName,newTitle,modelList,theCoefList,true) ;
  for (std::set<std::string>::const_iterator attrIt = _boolAttrib.begin();
      attrIt != _boolAttrib.end(); ++attrIt) {
    convSum->setAttribute((*attrIt).c_str()) ;
  }
  for (std::map<std::string,std::string>::const_iterator attrIt = _stringAttrib.begin();
      attrIt != _stringAttrib.end(); ++attrIt) {
    convSum->setStringAttribute((attrIt->first).c_str(), (attrIt->second).c_str()) ;
  }
  convSum->changeBasis(inBasis) ;
  return convSum ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return code for basis function representing by 'name' string.
/// The basis code of the first component model will be returned,
/// if the basis is supported by all components. Otherwise 0
/// is returned

Int_t RooAddModel::basisCode(const char* name) const
{
  bool first(true), code(0) ;
  for (auto obj : _pdfList) {
    auto model = static_cast<RooResolutionModel*>(obj);
    Int_t subCode = model->basisCode(name) ;
    if (first) {
      code = subCode ;
      first = false ;
    } else if (subCode==0) {
      code = 0 ;
    }
  }

  return code ;
}



////////////////////////////////////////////////////////////////////////////////
/// Retrieve cache element with for calculation of p.d.f value with normalization set nset and integrated over iset
/// in range 'rangeName'. If cache element does not exist, create and fill it on the fly. The cache contains
/// suplemental normalization terms (in case not all added p.d.f.s have the same observables), projection
/// integrals to calculated transformed fraction coefficients when a frozen reference frame is provided
/// and projection integrals for similar transformations when a frozen reference range is provided.

AddCacheElem* RooAddModel::getProjCache(const RooArgSet* nset, const RooArgSet* iset) const
{
  // Check if cache already exists
  auto cache = static_cast<AddCacheElem*>(_projCacheMgr.getObj(nset,iset,0,normRange()));
  if (cache) {
    return cache ;
  }

  //Create new cache
  cache = new AddCacheElem{*this, _pdfList, _coefList, nset, iset, _refCoefNorm,
                           _refCoefRangeName ? RooNameReg::str(_refCoefRangeName) : "",
                           _verboseEval};

  _projCacheMgr.setObj(nset,iset,cache,RooNameReg::ptr(normRange())) ;

  return cache;
}


////////////////////////////////////////////////////////////////////////////////
/// Update the coefficient values in the given cache element: calculate new remainder
/// fraction, normalize fractions obtained from extended ML terms to unity, and
/// multiply the various range and dimensional corrections needed in the
/// current use context.

void RooAddModel::updateCoefficients(AddCacheElem& cache, const RooArgSet* nset) const
{
  _coefCache.resize(_pdfList.getSize());
  for(std::size_t i = 0; i < _coefList.size(); ++i) {
    _coefCache[i] = static_cast<RooAbsReal const&>(_coefList[i]).getVal(nset);
  }
  RooAddHelpers::updateCoefficients(*this, _coefCache, _pdfList, _haveLastCoef, cache, nset,
                                    _refCoefNorm, _allExtendable, _coefErrCount);
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate the current value

double RooAddModel::evaluate() const
{
  const RooArgSet* nset = _normSet ;
  AddCacheElem* cache = getProjCache(nset) ;

  updateCoefficients(*cache,nset) ;


  // Do running sum of coef/pdf pairs, calculate lastCoef.
  double snormVal ;
  double value(0) ;
  Int_t i(0) ;
  for (auto obj : _pdfList) {
    auto pdf = static_cast<RooAbsPdf*>(obj);

    if (_coefCache[i]!=0.) {
      snormVal = nset ? cache->suppNormVal(i) : 1.0 ;
      double pdfVal = pdf->getVal(nset) ;
      // double pdfNorm = pdf->getNorm(nset) ;
      if (pdf->isSelectedComp()) {
   value += pdfVal*_coefCache[i]/snormVal ;
   cxcoutD(Eval) << "RooAddModel::evaluate(" << GetName() << ")  value += ["
         << pdf->GetName() << "] " << pdfVal << " * " << _coefCache[i] << " / " << snormVal << endl ;
      }
    }
    i++ ;
  }

  return value ;
}

void RooAddModel::computeBatch(cudaStream_t *stream, double *output, size_t nEvents,
                               RooFit::Detail::DataMap const &dataMap) const
{
   // Like many other functions in this class, the implementation was copy-pasted from the RooAddPdf

   _coefCache.resize(_pdfList.size());
   for (std::size_t i = 0; i < _coefList.size(); ++i) {
      auto coefVals = dataMap.at(&_coefList[i]);
      // We don't support per-event coefficients in this function. If the CPU
      // mode is used, we can just fall back to the RooAbsReal implementation.
      // With CUDA, we can't do that because the inputs might be on the device.
      // That's why we throw an exception then.
      if (coefVals.size() > 1) {
         if (stream) {
            throw std::runtime_error("The RooAddPdf doesn't support per-event coefficients in CUDA mode yet!");
         }
         RooAbsReal::computeBatch(stream, output, nEvents, dataMap);
         return;
      }
      _coefCache[i] = coefVals[0];
   }

   RooBatchCompute::VarVector pdfs;
   RooBatchCompute::ArgVector coefs;
   const RooArgSet *nset = nullptr;
   AddCacheElem *cache = getProjCache(nullptr);
   // We don't sync the coefficient values from the _coefList to the _coefCache
   // because we have already done it using the dataMap.
   updateCoefficients(*cache, nset);

   for (unsigned int pdfNo = 0; pdfNo < _pdfList.size(); ++pdfNo) {
      auto pdf = static_cast<RooAbsPdf *>(&_pdfList[pdfNo]);
      if (pdf->isSelectedComp()) {
         pdfs.push_back(dataMap.at(pdf));
         coefs.push_back(_coefCache[pdfNo] / cache->suppNormVal(pdfNo));
      }
   }
   auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;
   dispatch->compute(stream, RooBatchCompute::AddPdf, output, nEvents, pdfs, coefs);
}


////////////////////////////////////////////////////////////////////////////////
/// Reset error counter to given value, limiting the number
/// of future error messages for this pdf to 'resetValue'

void RooAddModel::resetErrorCounters(Int_t resetValue)
{
  RooAbsPdf::resetErrorCounters(resetValue) ;
  _coefErrCount = resetValue ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if PDF is valid for given normalization set.
/// Coeffient and PDF must be non-overlapping, but pdf-coefficient
/// pairs may overlap each other

bool RooAddModel::checkObservables(const RooArgSet* nset) const
{
  bool ret(false) ;

  for (unsigned int i = 0; i < _coefList.size(); ++i) {
    auto pdf = &_pdfList[i];
    auto coef = &_coefList[i];

    if (pdf->observableOverlaps(nset,*coef)) {
      coutE(InputArguments) << "RooAddModel::checkObservables(" << GetName() << "): ERROR: coefficient " << coef->GetName()
             << " and PDF " << pdf->GetName() << " have one or more dependents in common" << endl ;
      ret = true ;
    }
  }

  return ret ;
}



////////////////////////////////////////////////////////////////////////////////

Int_t RooAddModel::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars,
                const RooArgSet* normSet, const char* rangeName) const
{
  if (_forceNumInt) return 0 ;

  // Declare that we can analytically integrate all requested observables
  analVars.add(allVars) ;

  // Retrieve (or create) the required component integral list
  Int_t code ;
  RooArgList *cilist ;
  getCompIntList(normSet,&allVars,cilist,code,rangeName) ;

  return code+1 ;

}



////////////////////////////////////////////////////////////////////////////////
/// Check if this configuration was created before

void RooAddModel::getCompIntList(const RooArgSet* nset, const RooArgSet* iset, pRooArgList& compIntList, Int_t& code, const char* isetRangeName) const
{
  Int_t sterileIdx(-1) ;

  IntCacheElem* cache = (IntCacheElem*) _intCacheMgr.getObj(nset,iset,&sterileIdx,RooNameReg::ptr(isetRangeName)) ;
  if (cache) {
    code = _intCacheMgr.lastIndex() ;
    compIntList = &cache->_intList ;

    return ;
  }

  // Create containers for partial integral components to be generated
  cache = new IntCacheElem ;

  // Fill Cache
  for (auto obj : _pdfList) {
    auto model = static_cast<RooResolutionModel*>(obj);

    RooAbsReal* intPdf = model->createIntegral(*iset,nset,0,isetRangeName) ;
    cache->_intList.addOwned(*intPdf) ;
  }

  // Store the partial integral list and return the assigned code ;
  code = _intCacheMgr.setObj(nset,iset,(RooAbsCacheElement*)cache,RooNameReg::ptr(isetRangeName)) ;

  // Fill references to be returned
  compIntList = &cache->_intList ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return analytical integral defined by given scenario code

double RooAddModel::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
  // No integration scenario
  if (code==0) {
    return getVal(normSet) ;
  }

  // Partial integration scenarios
  IntCacheElem* cache = (IntCacheElem*) _intCacheMgr.getObjByIndex(code-1) ;

  RooArgList* compIntList ;

  // If cache has been sterilized, revive this slot
  if (cache==0) {
    std::unique_ptr<RooArgSet> vars{getParameters(RooArgSet())} ;
    RooArgSet nset = _intCacheMgr.selectFromSet1(*vars, code-1) ;
    RooArgSet iset = _intCacheMgr.selectFromSet2(*vars, code-1) ;

    int code2 = -1 ;
    getCompIntList(&nset,&iset,compIntList,code2,rangeName) ;
  } else {

    compIntList = &cache->_intList ;

  }

  // Calculate the current value
  const RooArgSet* nset = _normSet ;
  AddCacheElem* pcache = getProjCache(nset) ;

  updateCoefficients(*pcache,nset) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  double snormVal ;
  double value(0) ;
  Int_t i(0) ;
  for (const auto obj : *compIntList) {
    auto pdfInt = static_cast<const RooAbsReal*>(obj);
    if (_coefCache[i]!=0.) {
      snormVal = nset ? pcache->suppNormVal(i) : 1.0 ;
      double intVal = pdfInt->getVal(nset) ;
      value += intVal*_coefCache[i]/snormVal ;
      cxcoutD(Eval) << "RooAddModel::evaluate(" << GetName() << ")  value += ["
            << pdfInt->GetName() << "] " << intVal << " * " << _coefCache[i] << " / " << snormVal << endl ;
    }
    i++ ;
  }

  return value ;

}



////////////////////////////////////////////////////////////////////////////////
/// Return the number of expected events, which is either the sum of all coefficients
/// or the sum of the components extended terms

double RooAddModel::expectedEvents(const RooArgSet* nset) const
{
  double expectedTotal(0.0);

  if (_allExtendable) {

    // Sum of the extended terms
    for (auto obj : _pdfList) {
      auto pdf = static_cast<RooAbsPdf*>(obj);
      expectedTotal += pdf->expectedEvents(nset) ;
    }

  } else {

    // Sum the coefficients
    for (const auto obj : _coefList) {
      auto coef = static_cast<RooAbsReal*>(obj);
      expectedTotal += coef->getVal() ;
    }
  }

  return expectedTotal;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface function used by test statistics to freeze choice of observables
/// for interpretation of fraction coefficients

void RooAddModel::selectNormalization(const RooArgSet* depSet, bool force)
{
  if (!force && _refCoefNorm.getSize()!=0) {
    return ;
  }

  if (!depSet) {
    fixCoefNormalization(RooArgSet()) ;
    return ;
  }

  RooArgSet myDepSet;
  getObservables(depSet, myDepSet);
  fixCoefNormalization(myDepSet);
}



////////////////////////////////////////////////////////////////////////////////
/// Interface function used by test statistics to freeze choice of range
/// for interpretation of fraction coefficients

void RooAddModel::selectNormalizationRange(const char* rangeName, bool force)
{
  if (!force && _refCoefRangeName) {
    return ;
  }

  fixCoefRange(rangeName) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return specialized context to efficiently generate toy events from RooAddModels.

RooAbsGenContext* RooAddModel::genContext(const RooArgSet &vars, const RooDataSet *prototype,
               const RooArgSet* auxProto, bool verbose) const
{
  return RooAddGenContext::create(*this,vars,prototype,auxProto,verbose).release();
}



////////////////////////////////////////////////////////////////////////////////
/// Direct generation is safe if all components say so

bool RooAddModel::isDirectGenSafe(const RooAbsArg& arg) const
{
  for (auto obj : _pdfList) {
    auto pdf = static_cast<RooAbsPdf*>(obj);

    if (!pdf->isDirectGenSafe(arg)) {
      return false ;
    }
  }
  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return pseud-code that indicates if all components can do internal generation (1) or not (0)

Int_t RooAddModel::getGenerator(const RooArgSet& directVars, RooArgSet &/*generateVars*/, bool /*staticInitOK*/) const
{
  for (auto obj : _pdfList) {
    auto pdf = static_cast<RooAbsPdf*>(obj);

    RooArgSet tmp ;
    if (pdf->getGenerator(directVars,tmp)==0) {
      return 0 ;
    }
  }
  return 1 ;
}




////////////////////////////////////////////////////////////////////////////////
/// This function should never be called as RooAddModel implements a custom generator context

void RooAddModel::generateEvent(Int_t /*code*/)
{
  assert(0) ;
}


////////////////////////////////////////////////////////////////////////////////
/// List all RooAbsArg derived contents in this cache element

RooArgList RooAddModel::IntCacheElem::containedArgs(Action)
{
  RooArgList allNodes(_intList) ;
  return allNodes ;
}


////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a RooAddModel to more intuitively reflect the contents of the
/// product operator construction

void RooAddModel::printMetaArgs(ostream& os) const
{
  bool first(true) ;

  os << "(" ;
  for (unsigned int i=0; i < _coefList.size(); ++i) {
    auto coef = &_coefList[i];
    auto pdf = &_pdfList[i];
    if (!first) {
      os << " + " ;
    } else {
      first = false ;
    }
    os << coef->GetName() << " * " << pdf->GetName() ;
  }
  if (_pdfList.size() > _coefList.size()) {
    os << " + [%] * " << _pdfList[_pdfList.size()-1].GetName() ;
  }
  os << ") " ;
}

