/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOT_ROOFIT_TESTSTATISTICS_MINIMIZER_TYPE_H
#define ROOT_ROOFIT_TESTSTATISTICS_MINIMIZER_TYPE_H

#include <string>

namespace RooFit {
namespace TestStatistics {
enum class MinimizerType { Minuit, Minuit2 };

std::string minimizer_type(MinimizerType type);
} // namespace TestStatistics
} // namespace RooFit

#endif // ROOFIT_MINIMIZER_TYPE_H
