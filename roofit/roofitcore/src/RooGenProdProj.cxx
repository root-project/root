/// \cond ROOFIT_INTERNAL

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
\file RooGenProdProj.cxx
\class RooGenProdProj
\ingroup Roofitcore


RooGenProdProj is an auxiliary class for RooProdPdf that calculates
a general normalised projection of a product of non-factorising PDFs, e.g.
\f[
 P_{x,xy} = \frac{\int ( P1 * P2 * \ldots) \mathrm{d}x}{\int ( P1 * P2 * \ldots ) \mathrm{d}x \mathrm{d}y}
\f]

Partial integrals, which factorise and can be calculated, are calculated
analytically. Remaining non-factorising observables are integrated numerically.
**/


#include "Riostream.h"
#include <cmath>

#include "RooGenProdProj.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooProduct.h"


////////////////////////////////////////////////////////////////////////////////
/// Constructor for a normalization projection of the product of p.d.f.s _prodSet
/// integrated over _intSet in range isetRangeName while normalized over _normSet

RooGenProdProj::RooGenProdProj(const char *name, const char *title, const RooArgSet& _prodSet, const RooArgSet& _intSet,
                const RooArgSet& _normSet, const char* isetRangeName, const char* normRangeName, bool doFactorize) :
  RooAbsReal(name, title),
  _compSetN("compSetN","Set of integral components owned by numerator",this,false),
  _compSetD("compSetD","Set of integral components owned by denominator",this,false),
  _intList("intList","List of integrals",this,true)
{
  // Set expensive object cache to that of first item in prodSet
  setExpensiveObjectCache(_prodSet.first()->expensiveObjectCache()) ;

  // Create owners of components created in constructor
  _compSetOwnedN = std::make_unique<RooArgSet>();
  _compSetOwnedD = std::make_unique<RooArgSet>();

  RooAbsReal* numerator = makeIntegral("numerator",_prodSet,_intSet,*_compSetOwnedN,isetRangeName,doFactorize) ;
  RooAbsReal* denominator = makeIntegral("denominator",_prodSet,_normSet,*_compSetOwnedD,normRangeName,doFactorize) ;

//   std::cout << "RooGenProdPdf::ctor(" << GetName() << ") numerator = " << numerator->GetName() << std::endl ;
//   numerator->printComponentTree() ;
//   std::cout << "RooGenProdPdf::ctor(" << GetName() << ") denominator = " << denominator->GetName() << std::endl ;
//   denominator->printComponentTree() ;

  // Copy all components in (non-owning) set proxy
  _compSetN.add(*_compSetOwnedN) ;
  _compSetD.add(*_compSetOwnedD) ;

  _intList.add(*numerator) ;
  if (denominator) {
    _intList.add(*denominator) ;
    _haveD = true ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooGenProdProj::RooGenProdProj(const RooGenProdProj &other, const char *name)
   : RooAbsReal(other, name),
     _compSetN("compSetN", "Set of integral components owned by numerator", this),
     _compSetD("compSetD", "Set of integral components owned by denominator", this),
     _intList("intList", "List of integrals", this),
     _haveD(other._haveD)
{
  // Copy constructor
  _compSetOwnedN = std::make_unique<RooArgSet>();
  other._compSetN.snapshot(*_compSetOwnedN);
  _compSetN.add(*_compSetOwnedN) ;

  _compSetOwnedD = std::make_unique<RooArgSet>();
  other._compSetD.snapshot(*_compSetOwnedD);
  _compSetD.add(*_compSetOwnedD) ;

  for (RooAbsArg * arg : *_compSetOwnedN) {
    arg->setOperMode(_operMode) ;
  }
  for (RooAbsArg * arg : *_compSetOwnedD) {
    arg->setOperMode(_operMode) ;
  }

  // Fill _intList

  _intList.add(*_compSetN.find(other._intList.at(0)->GetName())) ;
  if (other._haveD) {
    _intList.add(*_compSetD.find(other._intList.at(1)->GetName())) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Utility function to create integral for product over certain observables.
/// \param[in] name Name of integral to be created.
/// \param[in] compSet All components of the product.
/// \param[in] intSet Observables to be integrated.
/// \param[out] saveSet All component objects needed to represent the product integral are added as owned members to saveSet.
/// \note The set owns new components that are created for the integral.
/// \param[in] isetRangeName Integral range.
/// \param[in] doFactorize
///
/// \return A RooAbsReal object representing the requested integral. The object is owned by `saveSet`.
///
/// The integration is factorized into components as much as possible and done analytically as far as possible.
RooAbsReal* RooGenProdProj::makeIntegral(const char* name, const RooArgSet& compSet, const RooArgSet& intSet,
                RooArgSet& saveSet, const char* isetRangeName, bool doFactorize)
{
  RooArgSet anaIntSet;
  RooArgSet numIntSet;

  // First determine subset of observables in intSet that are factorizable
  for (const auto arg : intSet) {
    auto count = std::count_if(compSet.begin(), compSet.end(), [arg](const RooAbsArg* pdfAsArg){
      auto pdf = static_cast<const RooAbsPdf*>(pdfAsArg);
      return (pdf->dependsOn(*arg));
    });

    if (count==1) {
      anaIntSet.add(*arg) ;
    }
  }

  // Determine which of the factorizable integrals can be done analytically
  RooArgSet prodSet ;
  numIntSet.add(intSet) ;

  // The idea of the RooGenProdProj is that we divide two integral objects each
  // created with this makeIntegral() function to get the normalized integral of
  // a product. Therefore, we don't need to normalize the numerater and
  // denominator integrals themselves. Doing the normalization would be
  // expensive and it would cancel out anyway. However, if we don't specify an
  // explicit normalization integral in createIntegral(), the last-used
  // normalization set might be used to normalize the pdf, resulting in
  // redundant computations.
  //
  // For this reason, the normalization set of the integrated pdfs is fixed to
  // an empty set in this case. Note that in RooFit, a nullptr normalization
  // set and an empty normalization set is not equivalent. The former implies
  // taking the last-used normalization set, and the latter means explicitly no
  // normalization.
  RooArgSet emptyNormSet{};

  RooArgSet keepAlive;

  for (const auto pdfAsArg : compSet) {
    auto pdf = static_cast<const RooAbsPdf*>(pdfAsArg);

    if (doFactorize && pdf->dependsOn(anaIntSet)) {
      RooArgSet anaSet ;
      Int_t code = pdf->getAnalyticalIntegralWN(anaIntSet,anaSet,nullptr,isetRangeName) ;
      if (code!=0) {
        // Analytical integral, create integral object
        std::unique_ptr<RooAbsReal> pai{pdf->createIntegral(anaSet,emptyNormSet,isetRangeName)};
        pai->setOperMode(_operMode) ;

        // Add to integral to product
        prodSet.add(*pai) ;

        // Remove analytically integratable observables from numeric integration list
        numIntSet.remove(anaSet) ;

        // Keep integral alive until the prodSet is cloned later
        keepAlive.addOwned(std::move(pai));
      } else {
        // Analytic integration of factorizable observable not possible, add straight pdf to product
        prodSet.add(*pdf) ;
      }
    } else {
      // Non-factorizable observables, add straight pdf to product
      prodSet.add(*pdf) ;
    }
  }

  // Create product of (partial) analytical integrals
  TString prodName ;
  if (isetRangeName) {
    prodName = Form("%s_%s_Range[%s]",GetName(),name,isetRangeName) ;
  } else {
    prodName = Form("%s_%s",GetName(),name) ;
  }

  // Create clones of the elements in prodSet. These need to be cloned
  // because when caching optimisation lvl 2 is activated, pre-computed
  // values are side-loaded into the elements.
  // Those pre-cached values already contain normalisation constants, so
  // the integral comes out wrongly. Therefore, we create here nodes that
  // don't participate in any caching, which are used to compute integrals.
  RooArgSet prodSetClone;
  prodSet.snapshot(prodSetClone, false);

  auto prod = std::make_unique<RooProduct>(prodName, "product", prodSetClone);
  prod->setExpensiveObjectCache(expensiveObjectCache()) ;
  prod->setOperMode(_operMode) ;

  // Create integral performing remaining numeric integration over (partial) analytic product
  std::unique_ptr<RooAbsReal> integral{prod->createIntegral(numIntSet,emptyNormSet,isetRangeName)};
  integral->setOperMode(_operMode) ;
  auto ret = integral.get();

  // Declare ownership of prodSet, product, and integral
  saveSet.addOwned(std::move(prodSetClone));
  saveSet.addOwned(std::move(prod));
  saveSet.addOwned(std::move(integral)) ;


  // Caller owners returned master integral object
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate and return value of normalization projection

double RooGenProdProj::evaluate() const
{
  RooArgSet const* nset = _intList.nset();

  double nom = static_cast<RooAbsReal*>(_intList.at(0))->getVal(nset);

  if (!_haveD) return nom ;

  double den = static_cast<RooAbsReal*>(_intList.at(1))->getVal(nset);

  //cout << "RooGenProdProj::eval(" << GetName() << ") nom = " << nom << " den = " << den << std::endl ;

  return nom / den ;
}



////////////////////////////////////////////////////////////////////////////////
/// Intercept cache mode operation changes and propagate them to the components

void RooGenProdProj::operModeHook()
{
  // WVE use cache manager here!

  for(RooAbsArg * arg : *_compSetOwnedN) {
    arg->setOperMode(_operMode) ;
  }

  for(RooAbsArg * arg : *_compSetOwnedD) {
    arg->setOperMode(_operMode) ;
  }

  _intList.at(0)->setOperMode(_operMode) ;
  if (_haveD) _intList.at(1)->setOperMode(Auto) ; // Denominator always stays in Auto mode (normalization integral)
}

/// \endcond
