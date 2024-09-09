// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei, E.G.P. Bos   2003-2017

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FCNGradientBase
#define ROOT_Minuit2_FCNGradientBase

#include "Minuit2/FCNBase.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

//________________________________________________________________________
/** Extension of the FCNBase for providing the analytical Gradient of the
    function.
    The size of the output Gradient vector must be equal to the size of the
    input Parameter vector.
 */

enum class GradientParameterSpace {
  External, Internal
};

class FCNGradientBase : public FCNBase {

public:
   ~FCNGradientBase() override {}

   virtual std::vector<double> Gradient(std::span<const double> ) const = 0;
   virtual std::vector<double> GradientWithPrevResult(std::span<const double> parameters, double * /*previous_grad*/,
                                                      double * /*previous_g2*/, double * /*previous_gstep*/) const
   {
      return Gradient(parameters);
   };

   virtual GradientParameterSpace gradParameterSpace() const {
      return GradientParameterSpace::External;
   };

   /// return second derivatives (diagonal of the Hessian matrix)
   virtual std::vector<double> G2(std::span<const double> ) const { return std::vector<double>();}

   /// return Hessian
   virtual std::vector<double> Hessian(std::span<const double> ) const { return std::vector<double>();}

   virtual bool HasHessian() const { return false; }

   virtual bool HasG2() const { return false; }


};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_FCNGradientBase
