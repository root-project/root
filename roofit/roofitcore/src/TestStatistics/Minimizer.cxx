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

#include <TestStatistics/Minimizer.h>

namespace RooFit {
namespace TestStatistics {

// the non-template functions in the MinimizerGenericPtr class

ROOT::Fit::Fitter *MinimizerGenericPtr::fitter() const
{
   return val->fitter();
}

MinimizerGenericPtr::operator TObject *() const
{
   return val->get_ptr();
}

} // namespace TestStatistics
} // namespace RooFit