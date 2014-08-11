// @(#)root/mathmore:$Id$
// Authors: B. List 29.4.2010


 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  * This library is free software; you can redistribute it and/or      *
  * modify it under the terms of the GNU General Public License        *
  * as published by the Free Software Foundation; either version 2     *
  * of the License, or (at your option) any later version.             *
  *                                                                    *
  * This library is distributed in the hope that it will be useful,    *
  * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
  * General Public License for more details.                           *
  *                                                                    *
  * You should have received a copy of the GNU General Public License  *
  * along with this library (see file COPYING); if not, write          *
  * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
  * 330, Boston, MA 02111-1307 USA, or contact the author.             *
  *                                                                    *
  **********************************************************************/

// Implementation file for class Vavilov
//
// Created by: blist  at Thu Apr 29 11:19:00 2010
//
// Last update: Thu Apr 29 11:19:00 2010
//


#include "Math/Vavilov.h"
#include "Math/VavilovAccurate.h"
#include "Math/SpecFuncMathCore.h"
#include "Math/SpecFuncMathMore.h"

#include <cassert>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>
#include <sstream>


namespace ROOT {
namespace Math {

static const double eu = 0.577215664901532860606;      // Euler's constant

Vavilov::Vavilov()
{
}

Vavilov::~Vavilov()
{
   // desctructor (clean up resources)
}


double Vavilov::Mode() const {
   double x = -4.22784335098467134e-01-std::log(GetKappa())-GetBeta2();
   if (x>-0.223172) x = -0.223172;
   double eps = 0.01;
   double dx;

   do {
      double p0 = Pdf (x - eps);
      double p1 = Pdf (x);
      double p2 = Pdf (x + eps);
      double y1 = 0.5*(p2-p0)/eps;
      double y2 = (p2-2*p1+p0)/(eps*eps);
      dx = - y1/y2;
      x += dx;
      if (fabs(dx) < eps) eps = 0.1*fabs(dx);
   } while (fabs(dx) > 1E-5);
   return x;
}

double Vavilov::Mode(double kappa, double beta2) {
   SetKappaBeta2 (kappa, beta2);
   return Mode();
}

double Vavilov::Mean() const {
   return Mean (GetKappa(), GetBeta2());
}

double Vavilov::Mean(double kappa, double beta2) {
   return eu-1-std::log(kappa)-beta2;
}

double Vavilov::Variance() const {
   return Variance (GetKappa(), GetBeta2());
}

double Vavilov::Variance(double kappa, double beta2) {
   return (1-0.5*beta2)/kappa;
}

double Vavilov::Skewness() const {
   return Skewness (GetKappa(), GetBeta2());
}

double Vavilov::Skewness(double kappa, double beta2) {
   return (0.5-beta2/3)/(kappa*kappa) * std::pow ((1-0.5*beta2)/kappa, -1.5);
}


double Vavilov::Kurtosis() const {
   return Kurtosis (GetKappa(), GetBeta2());
}

double Vavilov::Kurtosis(double kappa, double beta2) {
   return (1./3-0.25*beta2)*pow (1-0.5*beta2, -2)/kappa;
}


} // namespace Math
} // namespace ROOT
