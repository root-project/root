// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnPoint
#define ROOT_Minuit2_MnPoint

namespace ROOT {

namespace Minuit2 {

/**

A point in x-y.

@author Fred James and Matthias Winkler; comments added by Andras Zsenei
and Lorenzo Moneta

@ingroup Minuit

 */

class MnPoint {

public:
   /// Initializes the point with its coordinates.
   ///
   /// @param x the x (first) coordinate of the point.
   /// @param y the y (second) coordinate of the point.
   MnPoint(double x, double y) : fX(x), fY(y) {}

   /// Get the x (first) coordinate.
   double X() const { return fX; }

   /// Get the y (second) coordinate.
   double Y() const { return fY; }

private:
   double fX;
   double fY;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnPoint
