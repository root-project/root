/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_ConstraintHelpers_h
#define RooFit_ConstraintHelpers_h

#include <RooAbsReal.h>

std::unique_ptr<RooAbsReal> createConstraintTerm(std::string const &name, RooAbsPdf const &pdf, RooAbsData const &data,
                                                 RooArgSet const *constrainedParameters,
                                                 RooArgSet const *externalConstraints,
                                                 RooArgSet const *globalObservables, const char *globalObservablesTag,
                                                 bool takeGlobalObservablesFromData);

#endif

/// \endcond
