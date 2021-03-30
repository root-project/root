/* -*- mode: c++ -*- *********************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * authors:                                                                  *
 *  Lydia Brenner (lbrenner@cern.ch), Carsten Burgard (cburgard@cern.ch)     *
 *  Katharina Ecker (kecker@cern.ch), Adam Kaluza      (akaluza@cern.ch)     *
 *****************************************************************************/

#ifndef ROO_LAGRANGIAN_MORPH_FLOAT
#define ROO_LAGRANGIAN_MORPH_FLOAT

#ifdef USE_UBLAS
#include <boost/multiprecision/cpp_dec_float.hpp>
#endif

#include <limits>
#ifdef USE_UBLAS
  typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<100> > SuperFloat;
  typedef std::numeric_limits< SuperFloat > SuperFloatPrecision;
#else
   typedef double SuperFloat;
   typedef std::numeric_limits<double> SuperFloatPrecision;
#endif

#endif
