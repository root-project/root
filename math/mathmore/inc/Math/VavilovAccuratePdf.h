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

// Header file for class VavilovAccuratePdf
//
// Created by: blist  at Thu Apr 29 11:19:00 2010
//
// Last update: Thu Apr 29 11:19:00 2010
//
#ifndef ROOT_Math_VavilovAccuratePdf
#define ROOT_Math_VavilovAccuratePdf


#include "Math/IParamFunction.h"
#include "Math/VavilovAccurate.h"

#include <string>

namespace ROOT {
namespace Math {

//____________________________________________________________________________
/**
   Class describing the Vavilov pdf.

   The probability density function of the Vavilov distribution
   is given by:
  \f[ p(\lambda; \kappa, \beta^2) =
  \frac{1}{2 \pi i}\int_{c-i\infty}^{c+i\infty} \phi(s) e^{\lambda s} ds\f]
   where \f$\phi(s) = e^{C} e^{\psi(s)}\f$
   with  \f$ C = \kappa (1+\beta^2 \gamma )\f$
   and \f[\psi(s) = s \ln \kappa + (s+\beta^2 \kappa)
               \cdot \left ( \int \limits_{0}^{1}
               \frac{1 - e^{\frac{-st}{\kappa}}}{t} \, dt- \gamma \right )
               - \kappa \, e^{\frac{-s}{\kappa}}\f].
   \f$ \gamma = 0.5772156649\dots\f$ is Euler's constant.

   The parameters are:
   - 0: Norm: Normalization constant
   - 1: x0:   Location parameter
   - 2: xi:   Width parameter
   - 3: kappa: Parameter \f$\kappa\f$ of the Vavilov distribution
   - 4: beta2: Parameter \f$\beta^2\f$ of the Vavilov distribution

   Benno List, June 2010

   @ingroup StatFunc
 */


class VavilovAccuratePdf: public IParametricFunctionOneDim {
   public:

      /**
         Default constructor
      */
      VavilovAccuratePdf();

      /**
         Constructor with parameter values
         @param p vector of doubles containing the parameter values (Norm, x0, xi, kappa, beta2).
      */
      VavilovAccuratePdf (const double *p);

      /**
         Destructor
      */
      ~VavilovAccuratePdf () override;

      /**
         Access the parameter values
      */
      const double * Parameters() const override;

      /**
         Set the parameter values

         @param p vector of doubles containing the parameter values (Norm, x0, xi, kappa, beta2).

      */
      void SetParameters(const double * p ) override;

      /**
         Return the number of Parameters
      */
      unsigned int NPar() const override;

      /**
         Return the name of the i-th parameter (starting from zero)
       */
      std::string ParameterName(unsigned int i) const override;

      /**
         Evaluate the function

       @param x The Landau parameter \f$x = \lambda_L\f$
       */
      double DoEval(double x) const override;

      /**
         Evaluate the function, using parameters p

       @param x The Landau parameter \f$x = \lambda_L\f$
         @param p vector of doubles containing the parameter values (Norm, x0, xi, kappa, beta2).
       */
      double DoEvalPar(double x, const double * p) const override;

      /**
         Return a clone of the object
       */
      IBaseFunctionOneDim  * Clone() const override;

   private:
      double fP[5];

};


} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_VavilovAccuratePdf */
