/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROO_GEN_PROD_PROJ
#define ROO_GEN_PROD_PROJ

#include <RooAbsReal.h>
#include <RooSetProxy.h>
#include <RooListProxy.h>

/// General form of projected integral of product of PDFs, utility class for RooProdPdf.
class RooGenProdProj : public RooAbsReal {
public:
   RooGenProdProj(const char *name, const char *title, const RooArgSet &_prodSet, const RooArgSet &_intSet,
                  const RooArgSet &_normSet, const char *isetRangeName, const char *normRangeName = nullptr,
                  bool doFactorize = true);

   RooGenProdProj(const RooGenProdProj &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooGenProdProj(*this, newname); }

private:
   RooAbsReal *makeIntegral(const char *name, const RooArgSet &compSet, const RooArgSet &intSet, RooArgSet &saveSet,
                            const char *isetRangeName, bool doFactorize);

   void operModeHook() override;

   double evaluate() const override;
   std::unique_ptr<RooArgSet> _compSetOwnedN; ///< Owner of numerator components
   std::unique_ptr<RooArgSet> _compSetOwnedD; ///< Owner of denominator components
   RooSetProxy _compSetN;                     ///< Set proxy for numerator components
   RooSetProxy _compSetD;                     ///< Set proxy for denominator components
   RooListProxy _intList;                     ///< Master integrals representing numerator and denominator
   bool _haveD = false;                       ///< Do we have a denominator term?
};

#endif

/// \endcond
