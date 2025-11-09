// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FCNAdapter
#define ROOT_Minuit2_FCNAdapter

#include "Minuit2/FCNBase.h"

#include <ROOT/RSpan.hxx>

#include <vector>
#include <functional>

namespace ROOT::Minuit2 {

/// Adapter class to wrap user-provided functions into the FCNBase interface.
///
/// This class allows users to supply their function (and optionally its gradient,
/// diagonal second derivatives, and Hessian) in the form of `std::function` objects.
/// It adapts these functions so that they can be used transparently with the MINUIT
/// minimizers via the `FCNBase` interface.
///
/// Typical usage:
/// - Pass the function to minimize to the constructor.
/// - Optionally set the gradient, G2 (second derivative diagonal), or Hessian
///   functions using the provided setter methods.
/// - MINUIT will then query these functions if available, or fall back to numerical
///   approximations if they are not provided.
///
/// \ingroup Minuit
class FCNAdapter : public FCNBase {

public:
   /// Construct an adapter around a user-provided function.
   ///
   /// @param f   Function to minimize. It must take a pointer to the parameter array
   ///            (`double const*`) and return the function value.
   /// @param up  Error definition parameter (defaults to 1.0).
   FCNAdapter(std::function<double(double const *)> f, double up = 1.) : fUp(up), fFunc(std::move(f)) {}

   /// Indicate whether an analytic gradient has been provided.
   ///
   /// @return `true` if a gradient function was set, otherwise `false`.
   bool HasGradient() const override { return bool(fGradFunc); }

   /// Indicate whether analytic second derivatives (diagonal of the Hessian) are available.
   ///
   /// @return `true` if a G2 function or a Hessian function has been set, otherwise `false`.
   bool HasG2() const override { return bool(fG2Func); }

   /// Indicate whether an analytic Hessian has been provided.
   ///
   /// @return `true` if a Hessian function was set, otherwise `false`.
   bool HasHessian() const override { return bool(fHessianFunc); }

   /// Evaluate the function at the given parameter vector.
   ///
   /// @param v Parameter vector.
   /// @return Function value at the specified parameters.
   double operator()(std::vector<double> const &v) const override { return fFunc(v.data()); }

   /// Return the error definition parameter (`up`).
   ///
   /// @return Current error definition value.
   double Up() const override { return fUp; }

   /// Evaluate the gradient of the function at the given parameter vector.
   ///
   /// @param v Parameter vector.
   /// @return Gradient vector (∂f/∂xᵢ) at the specified parameters.
   std::vector<double> Gradient(std::vector<double> const &v) const override
   {
      std::vector<double> output(v.size());
      fGradFunc(v.data(), output.data());
      return output;
   }

   /// Return the diagonal elements of the Hessian (second derivatives).
   ///
   /// If a G2 function is set, it is used directly. If only a Hessian function
   /// is available, the diagonal is extracted from the full Hessian.
   ///
   /// @param x Parameter vector.
   /// @return Vector of second derivatives (one per parameter).
   std::vector<double> G2(std::vector<double> const &x) const override
   {
      std::vector<double> output;
      if (fG2Func)
         return fG2Func(x);
      if (fHessianFunc) {
         std::size_t n = x.size();
         output.resize(n);
         if (fHessian.empty())
            fHessian.resize(n * n);
         fHessianFunc(x, fHessian.data());
         if (!fHessian.empty()) {
            // Extract diagonal elements of Hessian
            for (unsigned int i = 0; i < n; i++)
               output[i] = fHessian[i * n + i];
         }
      }
      return output;
   }

   /// Return the full Hessian matrix.
   ///
   /// If a Hessian function is available, it is used to fill the matrix.
   /// If the Hessian function fails, it is cleared and not used again.
   ///
   /// @param x Parameter vector.
   /// @return Flattened Hessian matrix in row-major order.
   std::vector<double> Hessian(std::vector<double> const &x) const override
   {
      std::vector<double> output;
      if (fHessianFunc) {
         std::size_t n = x.size();
         output.resize(n * n);
         bool ret = fHessianFunc(x, output.data());
         if (!ret) {
            output.clear();
            fHessianFunc = nullptr;
         }
      }

      return output;
   }

   /// Set the analytic gradient function.
   ///
   /// @param f Gradient function of type `void(double const*, double*)`.
   ///          The first argument is the parameter array, the second is
   ///          the output array for the gradient values.
   void SetGradientFunction(std::function<void(double const *, double *)> f) { fGradFunc = std::move(f); }

   /// Set the function providing diagonal second derivatives (G2).
   ///
   /// @param f Function taking a parameter vector and returning the
   ///          diagonal of the Hessian matrix as a vector.
   void SetG2Function(std::function<std::vector<double>(std::vector<double> const &)> f) { fG2Func = std::move(f); }

   /// Set the function providing the full Hessian matrix.
   ///
   /// @param f Function of type `bool(std::vector<double> const&, double*)`.
   ///          The first argument is the parameter vector, the second is
   ///          the output buffer (flattened matrix). The return value
   ///          should be `true` on success, `false` on failure.
   void SetHessianFunction(std::function<bool(std::vector<double> const &, double *)> f)
   {
      fHessianFunc = std::move(f);
   }

   /// Update the error definition parameter.
   ///
   /// @param up New error definition value.
   void SetErrorDef(double up) override { fUp = up; }

private:
   using Function = std::function<double(double const *)>;
   using GradFunction = std::function<void(double const *, double *)>;
   using G2Function = std::function<std::vector<double>(std::vector<double> const &)>;
   using HessianFunction = std::function<bool(std::vector<double> const &, double *)>;

   double fUp = 1.;                      ///< Error definition parameter.
   mutable std::vector<double> fHessian; ///< Storage for intermediate Hessian values.

   Function fFunc;                       ///< Wrapped function to minimize.
   GradFunction fGradFunc;               ///< Optional gradient function.
   G2Function fG2Func;                   ///< Optional diagonal second-derivative function.
   mutable HessianFunction fHessianFunc; ///< Optional Hessian function.
};

} // namespace ROOT::Minuit2

#endif // ROOT_Minuit2_FCNAdapter
