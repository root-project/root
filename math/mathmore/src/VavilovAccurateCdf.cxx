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

// Implementation file for class VavilovAccurateCdf
//
// Created by: blist  at Thu Apr 29 11:19:00 2010
//
// Last update: Thu Apr 29 11:19:00 2010
//

#include "Math/VavilovAccurateCdf.h"
#include "Math/VavilovAccurate.h"

namespace ROOT {
namespace Math {


VavilovAccurateCdf::VavilovAccurateCdf() {
   fP[0] = 1;
   fP[1] = 0;
   fP[2] = 1;
   fP[3] = 1;
   fP[4] = 1;
}

VavilovAccurateCdf::VavilovAccurateCdf(const double *p) {
   if (p)
      for (int i = 0; i < 5; ++i)
         fP[i] = p[i];
   else {
      fP[0] = 1;
      fP[1] = 0;
      fP[2] = 1;
      fP[3] = 1;
      fP[4] = 1;
   }
}

VavilovAccurateCdf::~VavilovAccurateCdf ()
{}

const double * VavilovAccurateCdf::Parameters() const {
   return fP;
}

void VavilovAccurateCdf::SetParameters(const double * p ) {
   if (p)
      for (int i = 0; i < 5; ++i)
         fP[i] = p[i];
}

unsigned int VavilovAccurateCdf::NPar() const {
  return 5;
}

std::string VavilovAccurateCdf::ParameterName(unsigned int i) const {
   switch (i) {
      case 0: return "Norm"; break;
      case 1: return "x0"; break;
      case 2: return "xi"; break;
      case 3: return "kappa"; break;
      case 4: return "beta2"; break;
   }
   return "???";
}

double VavilovAccurateCdf::DoEval(double x) const {
   VavilovAccurate v(fP[3], fP[4]);
   return fP[0]*v.Cdf ((x-fP[1])/fP[2]);
}

double VavilovAccurateCdf::DoEvalPar(double x, const double * p) const {
   if (!p) return 0;
   // p[0]: norm, p[1]: x0, p[2]: width, p[3]: kappa, p[4]: beta2
   VavilovAccurate v(p[3], p[4]);
   return p[0]*v.Cdf ((x-p[1])/p[2]);
}

IBaseFunctionOneDim * VavilovAccurateCdf::Clone() const {
   return new VavilovAccurateCdf (*this);
}

} // namespace Math
} // namespace ROOT
