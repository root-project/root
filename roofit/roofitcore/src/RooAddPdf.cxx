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
 \sum_{i=1}^{n} c_i * \mathrm{PDF}_i
\f]

or 
\f[
 c_1*\mathrm{PDF}_1 + c_2*\mathrm{PDF}_2 + ... + (1-\sum_{i=1}^{n-1}c_i)*\mathrm{PDF}_n
\f]

The first form is for extended likelihood fits, where the
expected number of events is \f$ \sum_i c_i \f$. The coefficients \f$ c_i \f$
can either be explicitly provided, or, if all components support
extended likelihood fits, they can be calculated the contribution
of each PDF to the total number of expected events.

In the second form, the sum of the coefficients is required to be one or less,
and the coefficient of the last PDF is calculated from that condition.

It is also possible to parameterize the coefficients recursively

\f[
c_1*\mathrm{PDF}_1 + (1-c_1)(c_2*\mathrm{PDF}_2 + (1-c_2)*(c_3*\mathrm{PDF}_3 + ...))
\f]

In this form the sum of the coefficients is always less than 1.0
for all possible values of the individual coefficients between 0 and 1.

RooAddPdf relies on each component PDF to be normalized and will perform 
no normalization other than calculating the proper last coefficient \f$ c_n \f$, if requested.
An (enforced) condition for this assumption is that each \f$ \mathrm{PDF}_i \f$ is independent
of each \f$ c_i \f$.

### Difference between RooAddPdf / RooRealSum{Func|Pdf}
- RooAddPdf is a PDF of PDFs, *i.e.* its components need to be normalised and non-negative.
- RooRealSumPdf is a PDF of functions, *i.e.*, its components can be negative, but their sum cannot be. The normalisation
  is computed automatically, unless the PDF is extended (see above).
- RooRealSumFunc is a sum of functions. It is neither normalised, nor need it be positive.

*/


#include "RooFit.h"
#include "RooMsgService.h"

#include "TIterator.h"
#include "TIterator.h"
#include "TList.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooRealProxy.h"
#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooAddGenContext.h"
#include "RooRealConstant.h"
#include "RooNameReg.h"
#include "RooMsgService.h"
#include "RooRecursiveFraction.h"
#include "RooGlobalFunc.h"
#include "RooRealIntegral.h"
#include "RooTrace.h"

#include "Riostream.h"
#include <algorithm>


using namespace std;

ClassImp(RooAddPdf);
;


////////////////////////////////////////////////////////////////////////////////
/// Default constructor used for persistence

RooAddPdf::RooAddPdf() :
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,kFALSE,kFALSE),
  _refCoefRangeName(0),
  _codeReg(10),
  _snormList(0),
  _recursive(kFALSE)
{
  _coefErrCount = _errorCount ;
  TRACE_CREATE 
}



////////////////////////////////////////////////////////////////////////////////
/// Dummy constructor 

RooAddPdf::RooAddPdf(const char *name, const char *title) :
  RooAbsPdf(name,title), 
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,kFALSE,kFALSE),
  _refCoefRangeName(0),
  _projectCoefs(kFALSE),
  _projCacheMgr(this,10),
  _codeReg(10),
  _pdfList("!pdfs","List of PDFs",this),
  _coefList("!coefficients","List of coefficients",this),
  _snormList(0),
  _haveLastCoef(kFALSE),
  _allExtendable(kFALSE),
  _recursive(kFALSE)
{
  _coefErrCount = _errorCount ;
  TRACE_CREATE 
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with two PDFs and one coefficient

RooAddPdf::RooAddPdf(const char *name, const char *title,
		     RooAbsPdf& pdf1, RooAbsPdf& pdf2, RooAbsReal& coef1) : 
  RooAbsPdf(name,title),
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,kFALSE,kFALSE),
  _refCoefRangeName(0),
  _projectCoefs(kFALSE),
  _projCacheMgr(this,10),
  _codeReg(10),
  _pdfList("!pdfs","List of PDFs",this),
  _coefList("!coefficients","List of coefficients",this),
  _haveLastCoef(kFALSE),
  _allExtendable(kFALSE),
  _recursive(kFALSE)
{
  _pdfList.add(pdf1) ;  
  _pdfList.add(pdf2) ;
  _coefList.add(coef1) ;

  _coefCache.resize(_pdfList.size());
  _coefErrCount = _errorCount ;
  TRACE_CREATE 
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

RooAddPdf::RooAddPdf(const char *name, const char *title, const RooArgList& inPdfList, const RooArgList& inCoefList, Bool_t recursiveFractions) :
  RooAbsPdf(name,title),
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,kFALSE,kFALSE),
  _refCoefRangeName(0),
  _projectCoefs(kFALSE),
  _projCacheMgr(this,10),
  _codeReg(10),
  _pdfList("!pdfs","List of PDFs",this),
  _coefList("!coefficients","List of coefficients",this),
  _haveLastCoef(kFALSE),
  _allExtendable(kFALSE),
  _recursive(kFALSE)
{ 
  if (inPdfList.getSize()>inCoefList.getSize()+1 || inPdfList.getSize()<inCoefList.getSize()) {
    coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() 
			  << ") number of pdfs and coefficients inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1" << endl ;
    assert(0) ;
  }

  if (recursiveFractions && inPdfList.getSize()!=inCoefList.getSize()+1) {
    coutW(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() 
			  << ") WARNING inconsistent input: recursive fractions options can only be used if Npdf=Ncoef+1, ignoring recursive fraction setting" << endl ;
  }
 
  // Constructor with N PDFs and N or N-1 coefs
  RooArgList partinCoefList ;

  Bool_t first(kTRUE) ;

  for (auto i = 0u; i < inCoefList.size(); ++i) {
    auto coef = dynamic_cast<RooAbsReal*>(inCoefList.at(i));
    auto pdf  = dynamic_cast<RooAbsPdf*>(inPdfList.at(i));
    if (inPdfList.at(i) == nullptr) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() 
			    << ") number of pdfs and coefficients inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1" << endl ;
      assert(0) ;
    }
    if (!coef) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") coefficient " << (coef ? coef->GetName() : "") << " is not of type RooAbsReal, ignored" << endl ;
      continue ;
    }
    if (!pdf) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << (pdf ? pdf->GetName() : "") << " is not of type RooAbsPdf, ignored" << endl ;
      continue ;
    }
    _pdfList.add(*pdf) ;

    // Process recursive fraction mode separately
    if (recursiveFractions) {
      partinCoefList.add(*coef) ;
      if (first) {	

	// The first fraction is the first plain fraction
	first = kFALSE ;
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
      coutF(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") last pdf " << pdf->GetName() << " is not of type RooAbsPdf, fatal error" << endl ;
      throw std::invalid_argument("Last argument for RooAddPdf is not a PDF.");
    }
    _pdfList.add(*pdf) ;  

    // Process recursive fractions mode
    if (recursiveFractions) {

      // The last recursive fraction = (1-f1)*(1-f2)*...(1-fN) and is calculated from the list (f1,...,fN,1) by RooRecursiveFraction)
      partinCoefList.add(RooFit::RooConst(1)) ;
      RooAbsReal* rfrac = new RooRecursiveFraction(Form("%s_recursive_fraction_%s",GetName(),pdf->GetName()),"Recursive Fraction",partinCoefList) ;
      addOwnedComponents(*rfrac) ;
      _coefList.add(*rfrac) ;      	

      // In recursive mode we always have Ncoef=Npdf
      _haveLastCoef=kTRUE ;
    }

  } else {
    _haveLastCoef=kTRUE ;
  }


  _coefCache.resize(_pdfList.size());
  _coefErrCount = _errorCount ;
  _recursive = recursiveFractions ;

  TRACE_CREATE 
}



////////////////////////////////////////////////////////////////////////////////
/// Generic constructor from list of extended PDFs. There are no coefficients as the expected
/// number of events from each components determine the relative weight of the PDFs.
/// 
/// All PDFs must inherit from RooAbsPdf. 

RooAddPdf::RooAddPdf(const char *name, const char *title, const RooArgList& inPdfList) :
  RooAbsPdf(name,title),
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,kFALSE,kFALSE),
  _refCoefRangeName(0),
  _projectCoefs(kFALSE),
  _projCacheMgr(this,10),
  _pdfList("!pdfs","List of PDFs",this),
  _coefList("!coefficients","List of coefficients",this),
  _haveLastCoef(kFALSE),
  _allExtendable(kTRUE),
  _recursive(kFALSE)
{ 
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

  _coefCache.resize(_pdfList.size());
  _coefErrCount = _errorCount ;
  TRACE_CREATE 
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
  _coefCache.resize(_pdfList.size());
  _coefErrCount = _errorCount ;
  TRACE_CREATE 
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAddPdf::~RooAddPdf()
{
  TRACE_DESTROY
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
  if (refCoefNorm.getSize()==0) {
    _projectCoefs = kFALSE ;
    return ;
  }
  _projectCoefs = kTRUE ;  

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
  if (_refCoefRangeName) _projectCoefs = kTRUE ;
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
  RooArgSet *fullDepList = getObservables(nset) ;
  if (iset) {
    fullDepList->remove(*iset,kTRUE,kTRUE) ;
  }    

  // Fill with dummy unit RRVs for now
  for (int i = 0; i < _pdfList.getSize(); ++i) {
    auto pdf  = static_cast<const RooAbsPdf *>(_pdfList.at(i));
    auto coef = static_cast<const RooAbsReal*>(_coefList.at(i));

    // Start with full list of dependents
    RooArgSet supNSet(*fullDepList) ;

    // Remove PDF dependents
    RooArgSet* pdfDeps = pdf->getObservables(nset) ;
    if (pdfDeps) {
      supNSet.remove(*pdfDeps,kTRUE,kTRUE) ;
      delete pdfDeps ; 
    }

    // Remove coef dependents
    RooArgSet* coefDeps = coef ? coef->getObservables(nset) : 0 ;
    if (coefDeps) {
      supNSet.remove(*coefDeps,kTRUE,kTRUE) ;
      delete coefDeps ;
    }
    
    RooAbsReal* snorm ;
    TString name(GetName()) ;
    name.Append("_") ;
    name.Append(pdf->GetName()) ;
    name.Append("_SupNorm") ;
    cache->_needSupNorm = kFALSE ;
    if (supNSet.getSize()>0) {
      snorm = new RooRealIntegral(name,"Supplemental normalization integral",RooRealConstant::value(1.0),supNSet) ;
      cxcoutD(Caching) << "RooAddPdf " << GetName() << " making supplemental normalization set " << supNSet << " for pdf component " << pdf->GetName() << endl ;
      cache->_needSupNorm = kTRUE ;
    } else {
      snorm = new RooRealVar(name,"Unit Supplemental normalization integral",1.0) ;
    }
    cache->_suppNormList.addOwned(*snorm) ;
  }

  delete fullDepList ;
    
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
  RooArgSet* nset2 = nset ? getObservables(nset) : new RooArgSet() ;
  cxcoutD(Caching) << "RooAddPdf(" << GetName() << ")::getPC nset = " << (nset?*nset:RooArgSet()) << " nset2 = " << *nset2 << endl ;

  if (nset2->getSize()==0 && _refCoefNorm.getSize()!=0) {
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
		   << " to "  << ((_refCoefNorm.getSize()>0)?_refCoefNorm:*nset2) << (_refCoefRangeName?":":"") << (_refCoefRangeName?RooNameReg::str(_refCoefRangeName):"") << endl ;
    
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
	TString name(GetName()) ;
	name.Append("_") ;
	name.Append(thePdf->GetName()) ;
	name.Append("_ProjectNorm") ;
	pdfProj = new RooRealVar(name,"Unit Projection normalization integral",1.0) ;
	cxcoutD(Caching) << "RooAddPdf(" << GetName() << ")::getPC nset2(" << *nset2 << ")==_refCoefNorm(" << _refCoefNorm << ") --> pdfProj = " << pdfProj->GetName() << endl ;
      }

      cache->_projList.addOwned(*pdfProj) ;
      cxcoutD(Caching) << " RooAddPdf::syncCoefProjList(" << GetName() << ") PP = " << pdfProj->GetName() << endl ;

      // Calculation optional supplemental normalization term
      RooArgSet supNormSet(_refCoefNorm) ;
      RooArgSet* deps = thePdf->getParameters(RooArgSet()) ;
      supNormSet.remove(*deps,kTRUE,kTRUE) ;
      delete deps ;

      RooAbsReal* snorm ;
      TString name(GetName()) ;
      name.Append("_") ;
      name.Append(thePdf->GetName()) ;
      name.Append("_ProjSupNorm") ;
      if (supNormSet.getSize()>0 && !nset2->equals(_refCoefNorm) ) {
	snorm = new RooRealIntegral(name,"Projection Supplemental normalization integral",
				    RooRealConstant::value(1.0),supNormSet) ;
      } else {
	snorm = new RooRealVar(name,"Unit Projection Supplemental normalization integral",1.0) ;
      }
      cxcoutD(Caching) << " RooAddPdf::syncCoefProjList(" << GetName() << ") SN = " << snorm->GetName() << endl ;
      cache->_suppProjList.addOwned(*snorm) ;

      // Calculate reference range adjusted projection integral
      RooAbsReal* rangeProj1 ;

   //    cout << "ALEX >>>> RooAddPdf(" << GetName() << ")::getPC _refCoefRangeName WVE = " 	   
// 	   <<(_refCoefRangeName?":":"") << (_refCoefRangeName?RooNameReg::str(_refCoefRangeName):"")
// 	   <<" _refCoefRangeName AK = "  << (_refCoefRangeName?_refCoefRangeName->GetName():"")  	   
// 	   << " && _refCoefNorm" << _refCoefNorm << " with size = _refCoefNorm.getSize() " << _refCoefNorm.getSize() << endl ;

      // Check if _refCoefRangeName is identical to default range for all observables, 
      // If so, substitute by unit integral 

      // ----------
      RooArgSet* tmpObs = thePdf->getObservables(_refCoefNorm) ;
      RooAbsArg* obsArg ;
      TIterator* iter = tmpObs->createIterator() ;
      Bool_t allIdent = kTRUE ;
      while((obsArg=(RooAbsArg*)iter->Next())) {
	RooRealVar* rvarg = dynamic_cast<RooRealVar*>(obsArg) ;
	if (rvarg) {
	  if (rvarg->getMin(RooNameReg::str(_refCoefRangeName))!=rvarg->getMin() ||
	      rvarg->getMax(RooNameReg::str(_refCoefRangeName))!=rvarg->getMax()) {
	    allIdent=kFALSE ;
	  }
	}
      }
      delete iter ;
      delete tmpObs ;
      // -------------

      if (_refCoefRangeName && _refCoefNorm.getSize()>0 && !allIdent) {
	

	RooArgSet* tmp = thePdf->getObservables(_refCoefNorm) ;
	rangeProj1 = thePdf->createIntegral(*tmp,*tmp,RooNameReg::str(_refCoefRangeName)) ;
	
	//rangeProj1->setOperMode(operMode()) ;

	delete tmp ;
      } else {

	TString theName(GetName()) ;
	theName.Append("_") ;
	theName.Append(thePdf->GetName()) ;
	theName.Append("_RangeNorm1") ;
	rangeProj1 = new RooRealVar(theName,"Unit range normalization integral",1.0) ;

      }
      cxcoutD(Caching) << " RooAddPdf::syncCoefProjList(" << GetName() << ") R1 = " << rangeProj1->GetName() << endl ;
      cache->_refRangeProjList.addOwned(*rangeProj1) ;
      

      // Calculate range adjusted projection integral
      RooAbsReal* rangeProj2 ;
      cxcoutD(Caching) << "RooAddPdf::syncCoefProjList(" << GetName() << ") rangename = " << (rangeName?rangeName:"<null>") 
		       << " nset = " << (nset?*nset:RooArgSet()) << endl ;
      if (rangeName && _refCoefNorm.getSize()>0) {

	rangeProj2 = thePdf->createIntegral(_refCoefNorm,_refCoefNorm,rangeName) ;
	//rangeProj2->setOperMode(operMode()) ;

      } else if (_normRange.Length()>0) {

	RooArgSet* tmp = thePdf->getObservables(_refCoefNorm) ;
	rangeProj2 = thePdf->createIntegral(*tmp,*tmp,_normRange.Data()) ;
	delete tmp ;

      } else {

	TString theName(GetName()) ;
	theName.Append("_") ;
	theName.Append(thePdf->GetName()) ;
	theName.Append("_RangeNorm2") ;
	rangeProj2 = new RooRealVar(theName,"Unit range normalization integral",1.0) ;

      }
      cxcoutD(Caching) << " RooAddPdf::syncCoefProjList(" << GetName() << ") R2 = " << rangeProj2->GetName() << endl ;
      cache->_rangeProjList.addOwned(*rangeProj2) ;

    }               

  }

  delete nset2 ;

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
    Double_t coefSum(0) ;
    std::size_t i = 0;
    for (auto arg : _pdfList) {
      auto pdf = static_cast<RooAbsPdf*>(arg);
      myCoefCache[i] = pdf->expectedEvents(_refCoefNorm.getSize()>0?&_refCoefNorm:nset) ;
      coefSum += myCoefCache[i] ;
      i++ ;
    }

    if (coefSum==0.) {
      coutW(Eval) << "RooAddPdf::updateCoefCache(" << GetName() << ") WARNING: total number of expected events is 0" << endl ;
    } else {
      for (int j=0; j < _pdfList.getSize(); j++) {
        myCoefCache[j] /= coefSum ;
      }			            
    }

  } else {
    if (_haveLastCoef) {

      // coef[i] = coef[i] / SUM(coef)
      Double_t coefSum(0) ;
      std::size_t i=0;
      for (auto coefArg : _coefList) {
        auto coef = static_cast<RooAbsReal*>(coefArg);
        myCoefCache[i] = coef->getVal(nset) ;
        coefSum += myCoefCache[i++];
      }		
      if (coefSum==0.) {
        coutW(Eval) << "RooAddPdf::updateCoefCache(" << GetName() << ") WARNING: sum of coefficients is zero 0" << endl ;
      } else {	
        for (int j=0; j < _coefList.getSize(); j++) {
          myCoefCache[j] /= coefSum;
        }
      }
    } else {

      // coef[i] = coef[i] ; coef[n] = 1-SUM(coef[0...n-1])
      Double_t lastCoef(1) ;
      std::size_t i=0;
      for (auto coefArg : _coefList) {
        auto coef = static_cast<RooAbsReal*>(coefArg);
        myCoefCache[i] = coef->getVal(nset) ;
        //cxcoutD(Caching) << "SYNC: orig coef[" << i << "] = " << myCoefCache[i] << endl ;
        lastCoef -= myCoefCache[i++];
      }			
      myCoefCache[_coefList.getSize()] = lastCoef ;
      //cxcoutD(Caching) << "SYNC: orig coef[" << _coefList.getSize() << "] = " << myCoefCache[_coefList.getSize()] << endl ;


      // Warn about coefficient degeneration
      if ((lastCoef<-1e-05 || (lastCoef-1)>1e-5) && _coefErrCount-->0) {
        coutW(Eval) << "RooAddPdf::updateCoefCache(" << GetName()
		        << " WARNING: sum of PDF coefficients not in range [0-1], value="
		        << 1-lastCoef ;
        if (_coefErrCount==0) {
          coutW(Eval) << " (no more will be printed)"  ;
        }
        coutW(Eval) << endl ;
      } 
    }
  }

  

//    cout << "XXXX" << GetName() << "updateCoefs _projectCoefs = " << (_projectCoefs?"T":"F") << " cache._projList.getSize()= " << cache._projList.getSize() << endl ;

  // Stop here if not projection is required or needed
  if ((!_projectCoefs && _normRange.Length()==0) || cache._projList.getSize()==0) {
    //if (cache._projList.getSize()==0) {
//     cout << GetName() << " SYNC no projection required rangeName = " << (_normRange.Length()>0?_normRange.Data():"<none>") << endl ;
    return ;
  }

//    cout << "XXXX" << GetName() << " updateCoefs, applying correction" << endl ;
//    cout << "PROJLIST = " << endl ;
//    cache._projList.Print("v") ;
   
   
  // Adjust coefficients for given projection
  Double_t coefSum(0) ;
  {
    RooAbsReal::GlobalSelectComponentRAII compRAII(true);

    for (int i = 0; i < _pdfList.getSize(); i++) {

      RooAbsReal* pp = ((RooAbsReal*)cache._projList.at(i)) ;
      RooAbsReal* sn = ((RooAbsReal*)cache._suppProjList.at(i)) ;
      RooAbsReal* r1 = ((RooAbsReal*)cache._refRangeProjList.at(i)) ;
      RooAbsReal* r2 = ((RooAbsReal*)cache._rangeProjList.at(i)) ;

      Double_t proj = pp->getVal()/sn->getVal()*(r2->getVal()/r1->getVal()) ;

      //     cxcoutD(Caching) << "ALEX:    RooAddPdf::updateCoef(" << GetName() << ") with nset = " << (nset?*nset:RooArgSet()) << "for pdf component #" << i << " = " << _pdfList.at(i)->GetName() << endl
      // 	 << "ALEX:   pp = " << pp->GetName() << " = " << pp->getVal() << endl
      // 	 << "ALEX:   sn = " << sn->GetName() << " = " << sn->getVal() <<  endl
      // 	 << "ALEX:   r1 = " << r1->GetName() << " = " << r1->getVal() <<  endl
      // 	 << "ALEX:   r2 = " << r2->GetName() << " = " << r2->getVal() <<  endl
      // 	 << "ALEX: proj = (" << pp->getVal() << "/" << sn->getVal() << ")*(" << r2->getVal() << "/" << r1->getVal() << ") = " << proj << endl ;

      myCoefCache[i] *= proj ;
      coefSum += myCoefCache[i] ;
    }
  }


  if ((RooMsgService::_debugCount>0) && RooMsgService::instance().isActive(this,RooFit::Caching,RooFit::DEBUG)) {
    for (int i=0; i < _pdfList.getSize(); ++i) {
      ccoutD(Caching) << " ALEX:   POST-SYNC coef[" << i << "] = " << myCoefCache[i]
                      << " ( _coefCache[i]/coefSum = " << myCoefCache[i]*coefSum << "/" << coefSum << " ) "<< endl ;
    }
  }

  if (coefSum==0.) {
    coutE(Eval) << "RooAddPdf::updateCoefCache(" << GetName() << ") sum of coefficients is zero." << endl ;
  }

  for (int i=0; i < _pdfList.getSize(); i++) {
    myCoefCache[i] /= coefSum ;
  }


  
}

std::pair<const RooArgSet*, RooAddPdf::CacheElem*> RooAddPdf::getNormAndCache() const {
  const RooArgSet* nset = _normSet ; 
  //cxcoutD(Caching) << "RooAddPdf::evaluate(" << GetName() << ") calling getProjCache with nset = " << nset << " = " << (nset?*nset:RooArgSet()) << endl ;

  if (nset==0 || nset->getSize()==0) {
    if (_refCoefNorm.getSize()!=0) {
      nset = &_refCoefNorm ;
    }
  }

  CacheElem* cache = getProjCache(nset) ;
  updateCoefficients(*cache,nset) ;

  return {nset, cache};
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate and return the current value

Double_t RooAddPdf::evaluate() const
{
  auto normAndCache = getNormAndCache();
  const RooArgSet* nset = normAndCache.first;
  CacheElem* cache = normAndCache.second;

  
  // Do running sum of coef/pdf pairs, calculate lastCoef.
  Double_t value(0);

  for (unsigned int i=0; i < _pdfList.size(); ++i) {
    const auto& pdf = static_cast<RooAbsPdf&>(_pdfList[i]);
    double snormVal = 1.;
    if (cache->_needSupNorm) {
      snormVal = ((RooAbsReal*)cache->_suppNormList.at(i))->getVal();
    }
    
    Double_t pdfVal = pdf.getVal(nset);
    if (pdf.isSelectedComp()) {
      value += pdfVal*_coefCache[i]/snormVal;
    }
  }

  return value;
}




////////////////////////////////////////////////////////////////////////////////
/// Compute addition of PDFs in batches.

RooSpan<double> RooAddPdf::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  auto normAndCache = getNormAndCache();
  const RooArgSet* nset = normAndCache.first;
  CacheElem* cache = normAndCache.second;


  auto output = _batchData.makeWritableBatchInit(begin, batchSize, 0.);
  const std::size_t n = output.size();


  for (unsigned int pdfNo = 0; pdfNo < _pdfList.size(); ++pdfNo) {
    const auto& pdf = static_cast<RooAbsPdf&>(_pdfList[pdfNo]);
    auto pdfOutputs = pdf.getValBatch(begin, batchSize, nset);
    assert(pdfOutputs.size() == output.size());

    const double coef = _coefCache[pdfNo] / (cache->_needSupNorm ?
        static_cast<RooAbsReal*>(cache->_suppNormList.at(pdfNo))->getVal() :
        1.);

    if (pdf.isSelectedComp()) {
      for (std::size_t i = 0; i < n; ++i) { //CHECK_VECTORISE
        output[i] += pdfOutputs[i] * coef;
      }
    }
  }

  return output;
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

Bool_t RooAddPdf::checkObservables(const RooArgSet* nset) const 
{
  Bool_t ret(kFALSE) ;

  // There may be fewer coefficients than PDFs.
  int end = std::min(_pdfList.getSize(), _coefList.getSize());
  for (int i = 0; i < end; ++i) {
    auto pdf  = static_cast<const RooAbsPdf *>(_pdfList.at(i));
    auto coef = static_cast<const RooAbsReal*>(_coefList.at(i));
    if (pdf->observableOverlaps(nset,*coef)) {
      coutE(InputArguments) << "RooAddPdf::checkObservables(" << GetName() << "): ERROR: coefficient " << coef->GetName() 
			    << " and PDF " << pdf->GetName() << " have one or more dependents in common" << endl ;
      ret = kTRUE ;
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

  RooArgSet* allDepVars = getObservables(allVars) ;
  RooArgSet allAnalVars(*allDepVars) ;
  delete allDepVars ;

  Int_t n(0) ;

  // First iteration, determine what each component can integrate analytically
  for (const auto pdfArg : _pdfList) {
    auto pdf = static_cast<const RooAbsPdf *>(pdfArg);
    RooArgSet subAnalVars ;
    pdf->getAnalyticalIntegralWN(allVars,subAnalVars,normSet,rangeName) ;

    // Observables that cannot be integrated analytically by this component are dropped from the common list
    for (const auto arg : allVars) {
      if (!subAnalVars.find(arg->GetName()) && pdf->dependsOn(*arg)) {
        allAnalVars.remove(*arg,kTRUE,kTRUE) ;
      }	
    }
    n++ ;
  }

  // If no observables can be integrated analytically, return code 0 here
  if (allAnalVars.getSize()==0) {
    return 0 ;
  }


  // Now retrieve codes for integration over common set of analytically integrable observables for each component
  n=0 ;
  std::vector<Int_t> subCode(_pdfList.getSize());
  Bool_t allOK(kTRUE) ;
  for (const auto arg : _pdfList) {
    auto pdf = static_cast<const RooAbsPdf *>(arg);
    RooArgSet subAnalVars ;
    RooArgSet* allAnalVars2 = pdf->getObservables(allAnalVars) ;
    subCode[n] = pdf->getAnalyticalIntegralWN(*allAnalVars2,subAnalVars,normSet,rangeName) ;
    if (subCode[n]==0 && allAnalVars2->getSize()>0) {
      coutE(InputArguments) << "RooAddPdf::getAnalyticalIntegral(" << GetName() << ") WARNING: component PDF " << pdf->GetName() 
			    << "   advertises inconsistent set of integrals (e.g. (X,Y) but not X or Y individually."
			    << "   Distributed analytical integration disabled. Please fix PDF" << endl ;
      allOK = kFALSE ;
    }
    delete allAnalVars2 ; 
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

  if ((normSet==0 || normSet->getSize()==0) && _refCoefNorm.getSize()>0) {
//     cout << "WVE integration of RooAddPdf without normalization, but have reference set, using ref set for normalization" << endl ;
    normSet = &_refCoefNorm ;
  }

  CacheElem* cache = getProjCache(normSet,intSet,0) ; // WVE rangename here?
  updateCoefficients(*cache,normSet) ;

  // Calculate the current value of this object  
  Double_t value(0) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  Double_t snormVal ;

  //cout << "ROP::aIWN updateCoefCache with rangeName = " << (rangeName?rangeName:"<null>") << endl ;
  RooArgList* snormSet = (cache->_suppNormList.getSize()>0) ? &cache->_suppNormList : 0 ;
  for (int i = 0; i < _pdfList.getSize(); ++i ) {
    auto pdf = static_cast<const RooAbsPdf*>(_pdfList.at(i));

    if (_coefCache[i]) {
      snormVal = snormSet ? ((RooAbsReal*) cache->_suppNormList.at(i))->getVal() : 1.0 ;
      
      // WVE swap this?
      Double_t val = pdf->analyticalIntegralWN(subCode[i],normSet,rangeName) ;
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
  Double_t expectedTotal(0.0);

  cxcoutD(Caching) << "RooAddPdf::expectedEvents(" << GetName() << ") calling getProjCache with nset = " << (nset?*nset:RooArgSet()) << endl ;
  CacheElem* cache = getProjCache(nset) ;
  updateCoefficients(*cache,nset) ;

  if (cache->_rangeProjList.getSize()>0) {

    RooFIter iter1 = cache->_refRangeProjList.fwdIterator() ;
    RooFIter iter2 = cache->_rangeProjList.fwdIterator() ;
    RooFIter iter3 = _pdfList.fwdIterator() ;

    if (_allExtendable) {
      
      RooAbsPdf* pdf ;
      while ((pdf=(RooAbsPdf*)iter3.next())) {      
	RooAbsReal* r1 = (RooAbsReal*)iter1.next() ;
	RooAbsReal* r2 = (RooAbsReal*)iter2.next() ;
	expectedTotal += (r2->getVal()/r1->getVal()) * pdf->expectedEvents(nset) ; 
      }    

    } else {

      RooFIter citer = _coefList.fwdIterator() ;
      RooAbsReal* coef ;
      while((coef=(RooAbsReal*)citer.next())) {
	Double_t ncomp = coef->getVal(nset) ;
	RooAbsReal* r1 = (RooAbsReal*)iter1.next() ;
	RooAbsReal* r2 = (RooAbsReal*)iter2.next() ;
	expectedTotal += (r2->getVal()/r1->getVal()) * ncomp ; 
      }

    }



  } else {

    if (_allExtendable) {

      RooFIter iter = _pdfList.fwdIterator() ;
      RooAbsPdf* pdf ;
      while((pdf=(RooAbsPdf*)iter.next())) {
	expectedTotal += pdf->expectedEvents(nset) ; 
      }

    } else {

      RooFIter citer = _coefList.fwdIterator() ;
      RooAbsReal* coef ;
      while((coef=(RooAbsReal*)citer.next())) {
	Double_t ncomp = coef->getVal(nset) ;
	expectedTotal += ncomp ;      
      }

    }

  }
  return expectedTotal ;  
}



////////////////////////////////////////////////////////////////////////////////
/// Interface function used by test statistics to freeze choice of observables
/// for interpretation of fraction coefficients

void RooAddPdf::selectNormalization(const RooArgSet* depSet, Bool_t force) 
{

  if (!force && _refCoefNorm.getSize()!=0) {
    return ;
  }

  if (!depSet) {
    fixCoefNormalization(RooArgSet()) ;
    return ;
  }

  RooArgSet* myDepSet = getObservables(depSet) ;
  fixCoefNormalization(*myDepSet) ;
  delete myDepSet ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface function used by test statistics to freeze choice of range
/// for interpretation of fraction coefficients

void RooAddPdf::selectNormalizationRange(const char* rangeName, Bool_t force) 
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
					const RooArgSet* auxProto, Bool_t verbose) const 
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
  list<Double_t>* sumHint = 0 ;
  Bool_t needClean(kFALSE) ;

  // Loop over components pdf
  for (const auto arg : _pdfList) {
    auto pdf = static_cast<const RooAbsPdf*>(arg);

    list<Double_t>* pdfHint = pdf->plotSamplingHint(obs,xlo,xhi) ;

    // Process hint
    if (pdfHint) {
      if (!sumHint) {

	// If this is the first hint, then just save it
	sumHint = pdfHint ;

      } else {

	list<Double_t>* newSumHint = new list<Double_t>(sumHint->size()+pdfHint->size()) ;

	// Merge hints into temporary array
	merge(pdfHint->begin(),pdfHint->end(),sumHint->begin(),sumHint->end(),newSumHint->begin()) ;

	// Copy merged array without duplicates to new sumHintArrau
	delete sumHint ;
	sumHint = newSumHint ;
	needClean = kTRUE ;
	
      }
    }
  }
  if (needClean) {
    list<Double_t>::iterator new_end = unique(sumHint->begin(),sumHint->end()) ;
    sumHint->erase(new_end,sumHint->end()) ;
  }

  return sumHint ;
}


////////////////////////////////////////////////////////////////////////////////
/// Loop over components for plot sampling hints and merge them if there are multiple

std::list<Double_t>* RooAddPdf::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const 
{
  list<Double_t>* sumBinB = 0 ;
  Bool_t needClean(kFALSE) ;
  
  // Loop over components pdf
  for (auto arg : _pdfList) {
    auto pdf = static_cast<const RooAbsPdf *>(arg);
    list<Double_t>* pdfBinB = pdf->binBoundaries(obs,xlo,xhi) ;

    // Process hint
    if (pdfBinB) {
      if (!sumBinB) {

	// If this is the first hint, then just save it
	sumBinB = pdfBinB ;

      } else {
	
	list<Double_t>* newSumBinB = new list<Double_t>(sumBinB->size()+pdfBinB->size()) ;

	// Merge hints into temporary array
	merge(pdfBinB->begin(),pdfBinB->end(),sumBinB->begin(),sumBinB->end(),newSumBinB->begin()) ;

	// Copy merged array without duplicates to new sumBinBArrau
	delete sumBinB ;
	delete pdfBinB ;
	sumBinB = newSumBinB ;
	needClean = kTRUE ;	
      }
    }
  }

  // Remove consecutive duplicates
  if (needClean) {    
    list<Double_t>::iterator new_end = unique(sumBinB->begin(),sumBinB->end()) ;
    sumBinB->erase(new_end,sumBinB->end()) ;
  }

  return sumBinB ;
}


////////////////////////////////////////////////////////////////////////////////
/// If all components that depend on obs are binned that so is the product

Bool_t RooAddPdf::isBinnedDistribution(const RooArgSet& obs) const 
{
  for (const auto arg : _pdfList) {
    auto pdf = static_cast<const RooAbsPdf*>(arg);
    if (pdf->dependsOn(obs) && !pdf->isBinnedDistribution(obs)) {
      return kFALSE ;
    }
  }
  
  return kTRUE  ;  
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
  Bool_t first(kTRUE) ;
    
  if (_coefList.getSize() != 0) {
    for (int i = 0; i < _pdfList.getSize(); ++i ) {
      const RooAbsArg * coef = _coefList.at(i);
      const RooAbsArg * pdf  = _pdfList.at(i);
      if (!first) {
        os << " + " ;
      } else {
        first = kFALSE ;
      }

      if (i < _coefList.getSize()) {
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
        first = kFALSE ;
      }
      os << pdf->GetName() ; 
    }  
  }

  os << " " ;    
}
