// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei, E.G.P. Bos   2003-2017

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 * Copyright (c) 2017 Patrick Bos, Netherlands eScience Center        *
 *                                                                    *
 **********************************************************************/

#include <vector>
#include "Minuit2/ExternalInternalGradientCalculator.h"
#include "Minuit2/FCNGradientBase.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MinimumParameters.h"

namespace ROOT {
  namespace Minuit2 {

    FunctionGradient ExternalInternalGradientCalculator::operator()(const MinimumParameters& par) const {
      // evaluate analytical gradient. take care of parameter transformations
      std::vector<double> par_vec;
      par_vec.resize(par.Vec().size());
      for (std::size_t ix = 0; ix < par.Vec().size(); ++ix) {
        par_vec[ix] = par.Vec()(ix);
      }

      std::vector<double> grad = fGradCalc.Gradient(par_vec);
      assert(grad.size() == fTransformation.Parameters().size());

      MnAlgebraicVector v(par.Vec().size());
      for(unsigned int i = 0; i < par.Vec().size(); i++) {
        unsigned int ext = fTransformation.ExtOfInt(i);
        v(i) = grad[ext];
      }

#ifdef DEBUG
      std::cout << "User given gradient in Minuit2" << v << std::endl;
#endif

      // check for 2nd derivative and step-size from the external gradient
      // function and use them if present
      // N.B.: for the time being we only allow both at the same time, since
      //       FunctionGradient only has ctors for two cases: 1. gradient only,
      //       2. grad, g2 & gstep.
      if (fGradCalc.hasG2ndDerivative() && fGradCalc.hasGStepSize()) {
        std::vector<double> g2 = fGradCalc.G2ndDerivative(par_vec);
        std::vector<double> gstep = fGradCalc.GStepSize(par_vec);

        MnAlgebraicVector vg2(par.Vec().size());
        MnAlgebraicVector vgstep(par.Vec().size());
        for(unsigned int i = 0; i < par.Vec().size(); i++) {
          unsigned int ext = fTransformation.ExtOfInt(i);
          vg2(i) = g2[ext];
          vgstep(i) = gstep[ext];
        }

        return FunctionGradient(v, vg2, vgstep);
      } else {
        return FunctionGradient(v);
      }
    }

    FunctionGradient ExternalInternalGradientCalculator::operator()(const MinimumParameters& par, const FunctionGradient&) const {
      // needed from base class
      return (*this)(par);
    }

  }  // namespace Minuit2
}  // namespace ROOT
