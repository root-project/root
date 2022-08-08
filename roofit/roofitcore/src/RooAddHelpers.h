/*
 * Project: RooFit
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooFitCore_RooAddHelpers_h
#define RooFit_RooFitCore_RooAddHelpers_h

#include <RooAbsCacheElement.h>
#include <RooArgList.h>

class RooAbsPdf;
class RooArgSet;

class AddCacheElem : public RooAbsCacheElement {
public:
   AddCacheElem(RooAbsPdf const &addPdf, RooArgList const &pdfList, RooArgList const &coefList, const RooArgSet *nset,
                const RooArgSet *iset, const char *rangeName, bool projectCoefs, RooArgSet const &refCoefNorm,
                TNamed const *refCoefRangeName, int verboseEval);

   RooArgList containedArgs(Action) override;

   RooArgList _suppNormList; ///< Supplemental normalization list
   bool _needSupNorm;        ///< Does the above list contain any non-unit entries?

   RooArgList _projList;         ///< Projection integrals to be multiplied with coefficients
   RooArgList _suppProjList;     ///< Projection integrals to multiply with coefficients for supplemental normalization
   RooArgList _refRangeProjList; ///< Range integrals to be multiplied with coefficients (reference range)
   RooArgList _rangeProjList;    ///< Range integrals to be multiplied with coefficients (target range)
};

#endif
