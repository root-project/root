/*
 * Project: RooFit
 * Authors:
 *   Lydia Brenner (lbrenner@cern.ch), Carsten Burgard (cburgard@cern.ch)
 *   Katharina Ecker (kecker@cern.ch), Adam Kaluza      (akaluza@cern.ch)
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef roofit_roofitcore_RooFit_Floats_h
#define roofit_roofitcore_RooFit_Floats_h

#ifdef USE_UBLAS
#include <boost/multiprecision/cpp_dec_float.hpp>
#endif

#include <limits>

namespace RooFit {

#ifdef USE_UBLAS
typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<100>> SuperFloat;
typedef std::numeric_limits<SuperFloat> SuperFloatPrecision;
#else
typedef double SuperFloat;
typedef std::numeric_limits<double> SuperFloatPrecision;
#endif

} // namespace RooFit

#endif
