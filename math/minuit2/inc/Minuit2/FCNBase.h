// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FCNBase
#define ROOT_Minuit2_FCNBase

#include "Minuit2/MnConfig.h"

#include <ROOT/RSpan.hxx>

#include <vector>

namespace ROOT::Minuit2 {

/// \defgroup Minuit Minuit2 Minimization Library
///
/// New Object-oriented implementation of the MINUIT minimization package.
/// More information is available at the home page of the \ref Minuit2Page "Minuit2" minimization package".
///
/// \ingroup Math

enum class GradientParameterSpace {
   External,
   Internal
};

/// Interface (abstract class) defining the function to be minimized, which has to be implemented by the user.
///
/// \ingroup Minuit

class FCNBase {

public:
   virtual ~FCNBase() = default;

   /// The meaning of the vector of parameters is of course defined by the user,
   /// who uses the values of those parameters to calculate their function Value.
   /// The order and the position of these parameters is strictly the one specified
   /// by the user when supplying the starting values for minimization. The starting
   /// values must be specified by the user, either via an std::vector<double> or the
   /// MnUserParameters supplied as input to the MINUIT minimizers such as
   /// VariableMetricMinimizer or MnMigrad. Later values are determined by MINUIT
   /// as it searches for the Minimum or performs whatever analysis is requested by
   /// the user.
   ///
   /// @param v function parameters as defined by the user.
   ///
   /// @return the Value of the function.
   ///
   /// @see MnUserParameters
   /// @see VariableMetricMinimizer
   /// @see MnMigrad

   virtual double operator()(std::vector<double> const &v) const = 0;

   /// Error definition of the function. MINUIT defines Parameter errors as the
   /// change in Parameter Value required to change the function Value by up. Normally,
   /// for chisquared fits it is 1, and for negative log likelihood, its Value is 0.5.
   /// If the user wants instead the 2-sigma errors for chisquared fits, it becomes 4,
   /// as Chi2(x+n*sigma) = Chi2(x) + n*n.
   ///
   /// Comment a little bit better with links!!!!!!!!!!!!!!!!!

   virtual double ErrorDef() const { return Up(); }

   /// Error definition of the function. MINUIT defines Parameter errors as the
   /// change in Parameter Value required to change the function Value by up. Normally,
   /// for chisquared fits it is 1, and for negative log likelihood, its Value is 0.5.
   /// If the user wants instead the 2-sigma errors for chisquared fits, it becomes 4,
   /// as Chi2(x+n*sigma) = Chi2(x) + n*n.
   ///
   /// \todo Comment a little bit better with links!!!!!!!!!!!!!!!!! Idem for ErrorDef()

   virtual double Up() const = 0;

   /// add interface to set dynamically a new error definition
   /// Re-implement this function if needed.
   virtual void SetErrorDef(double) {};

   virtual bool HasGradient() const { return false; }

   /// Return the gradient vector of the function at the given parameter point.
   ///
   /// By default, returns an empty vector (no analytic gradient provided).
   /// Override this method if an analytic gradient is available.
   ///
   /// @param v Parameter vector.
   /// @return Gradient vector with respect to the parameters.
   virtual std::vector<double> Gradient(std::vector<double> const &) const { return {}; }

   /// \warning Not meant to be overridden! This is a requirement for an
   /// internal optimization in RooFit that might go away with any refactoring.
   virtual std::vector<double> GradientWithPrevResult(std::vector<double> const &parameters, double * /*previous_grad*/,
                                                      double * /*previous_g2*/, double * /*previous_gstep*/) const
   {
      return Gradient(parameters);
   };

   /// \warning Not meant to be overridden! This is a requirement for an
   /// internal optimization in RooFit that might go away with any refactoring.
   virtual GradientParameterSpace gradParameterSpace() const { return GradientParameterSpace::External; };

   /// Return the diagonal elements of the Hessian (second derivatives).
   ///
   /// By default, returns an empty vector. Override this method if analytic second derivatives
   /// (per-parameter curvature) are available.
   ///
   /// @param v Parameter vector.
   /// @return Vector of second derivatives with respect to each parameter.
   virtual std::vector<double> G2(std::vector<double> const &) const { return {}; }

   /// Return the full Hessian matrix of the function.
   ///
   /// By default, returns an empty vector. Override this method if the full analytic Hessian
   /// (matrix of second derivatives) is available.
   ///
   /// @param v Parameter vector.
   /// @return Flattened Hessian matrix.
   virtual std::vector<double> Hessian(std::vector<double> const &) const { return {}; }

   virtual bool HasHessian() const { return false; }

   virtual bool HasG2() const { return false; }
};

} // namespace ROOT::Minuit2

#endif // ROOT_Minuit2_FCNBase
