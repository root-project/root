// @(#)root/mathcore:$Id$
// Author: Fons Rademakers,    12/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Functions for the evaluation of the Chebyshev polynomials and the    //
// ChebyshevPol class which can be used for creating a TF1.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Math/ChebyshevPol.h"

namespace ROOT {
   namespace Math {
      namespace Chebyshev {
         template<> double T<0> (double ) { return 1;}
         template<> double T<1> (double x) { return x;}
         template<> double T<2> (double x) { return 2.0*x*x -1;}
         template<> double T<3> (double x) { return 4.0*x*x*x -3.0*x;}
      
         template<> double Eval<0> (double , const double *c) { return c[0];}
         template<> double Eval<1> (double x, const double *c) { return c[1]*x + c[0];}
         template<> double Eval<2> (double x, const double *c) { return c[2]*Chebyshev::T<2>(x) + c[1]*x + c[0];}
         template<> double Eval<3> (double x, const double *c) { return c[3]*Chebyshev::T<3>(x) + Eval<2>(x,c); }
      }
   }
}
