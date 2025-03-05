// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnParabola
#define ROOT_Minuit2_MnParabola

#include <cmath>

namespace ROOT {

namespace Minuit2 {

/**

This class defines a parabola of the form a*x*x + b*x + c

@author Fred James and Matthias Winkler; comments added by Andras Zsenei
and Lorenzo Moneta

@ingroup Minuit

 */

class MnParabola {

public:
   /// Constructor that initializes the parabola with its three parameters.
   ///
   /// @param a the coefficient of the quadratic term.
   /// @param b the coefficient of the linear term.
   /// @param c the constant.
   MnParabola(double a, double b, double c) : fA(a), fB(b), fC(c) {}

   /// Evaluates the parabola a the point x.
   double Y(double x) const { return (fA * x * x + fB * x + fC); }

   /// Calculate the x coordinate of the Minimum of the parabola.
   double Min() const { return -fB / (2. * fA); }

   /// Calculate the y coordinate of the Minimum of the parabola.
   double YMin() const { return (-fB * fB / (4. * fA) + fC); }

   /// Get the coefficient of the quadratic term.
   double A() const { return fA; }

   /// Get the coefficient of the linear term.
   double B() const { return fB; }

   /// Get the coefficient of the constant term.
   double C() const { return fC; }

private:
   double fA;
   double fB;
   double fC;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnParabola
