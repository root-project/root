// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnStrategy
#define ROOT_Minuit2_MnStrategy

namespace ROOT {

namespace Minuit2 {

/**
    API class for defining four levels of strategies: low (0), medium (1),
    high (2), very high (>=3);
    acts on: Migrad (behavioural),
             Minos (lowers strategy by 1 for Minos-own minimization),
             Hesse (iterations),
             Numerical2PDerivative (iterations)

   The strategy level controls the trade-off between the speed and the reliability of
   the minimization and error calculation. A higher level spends more function
   evaluations to refine the gradient and the Hessian.

   The table below documents every tunable that the strategy governs, together with
   its default value for each strategy level. Strategy 3 shares strategy 2's Migrad
   iteration parameters, but skips the expensive full initial Hessian at the seed
   stage (**ComputeInitialHessian**) and applies additional changes to the Hesse
   calculation that aim at a more accurate Hessian and allow returning the covariance
   matrix without forcing it to be positive-definite.

   <table>
     <tr>
       <th rowspan="2">Name and effect</th>
       <th rowspan="2">Type</th>
       <th colspan="4">Value for strategy *n*</th>
     </tr>
     <tr>
       <th>0</th>
       <th>1</th>
       <th>2</th>
       <th>3</th>
     </tr>
     <tr>
       <td>**ComputeInitialHessian**</td>
       <td><code>bool</code></td>
       <td rowspan="2" colspan="2">false</td> <td rowspan="2">true</td> <td rowspan="2"><span style="color:red;">false</span></td>
     </tr>
     <tr>
       <td colspan="2">
         Compute full initial Hessian for the seed state, which can be quite expensive for many parameters.

         Usually, the initial approximation that leaves the off-diagonal elements at zero is good enough.
       </td>
     </tr>
     <tr>
       <td>**RefineGradientInHessian**</td>
       <td><code>bool</code></td>
       <td rowspan="2">false</td> <td rowspan="2" colspan="3">true</td>
     </tr>
     <tr>
       <td colspan="2">
         After the diagonal second derivatives have been computed, recompute the
      first derivatives at the current point with the more
      accurate **HessianGradientCalculator** (which uses up to **HessianGradientNCycles**
      refinement cycles) instead of reusing the gradient from the last
      minimization step.
       </td>
     </tr>
     <tr>
       <td>**GradientNCycles**</td>
       <td><code>unsigned int</code></td>
       <td rowspan="2">2</td> <td rowspan="2">3</td> <td rowspan="2" colspan="2">5</td>
     </tr>
     <tr>
       <td colspan="2">
         Maximum number of cycles the numerical gradient calculator spends refining
      the finite-difference step size and the first-derivative estimate of each
      parameter. More cycles yield a more accurate gradient at the price of more
      function evaluations.

      A cycle is stopped early once either **GradientStepTolerance** or **GradientTolerance** is met.
       </td>
     </tr>
     <tr>
       <td>**GradientStepTolerance**</td>
       <td><code>double</code></td>
       <td rowspan="2">0.5</td> <td rowspan="2">0.3</td> <td rowspan="2" colspan="2">0.1</td>
     </tr>
     <tr>
       <td colspan="2">
         Stop the numerical gradient refinement cycle early once the finite-difference
      step size has stabilized.

      The parameter refers to the change in step size relative to the new step size.

      This is the partner of the **GradientTolerance** parameter.
       </td>
     </tr>
     <tr>
       <td>**GradientTolerance**</td>
       <td><code>double</code></td>
       <td rowspan="2">0.1</td> <td rowspan="2">0.05</td> <td rowspan="2" colspan="2">0.02</td>
     </tr>
     <tr>
       <td colspan="2">
         Stop the numerical gradient refinement cycle early once the first-derivative
      estimate itself is stable.

      The parameter refers to the change in the first-derivative estimate relative to
      the new estimate.

      This is the partner of the **GradientStepTolerance** parameter.
       </td>
     </tr>
     <tr>
       <td>**HessianCentralFDMixedDerivatives**</td>
       <td><code>unsigned int</code></td>
       <td rowspan="2" colspan="3">0</td> <td rowspan="2">1</td>
     </tr>
     <tr>
       <td colspan="2">
         Central finite difference is used for mixed partial derivatives (the off-diagonal terms of the Hessian).

      This requires 3 extra function evaluations per derivative, but is
      necessary in the case of minima where there is high curvature (in
      the case of high stats) and the forward finite difference (default)
      behaviour leads incorrectly to a non-positive-definite covariance
      matrix.
       </td>
     </tr>
     <tr>
       <td>**HessianForcePosDef**</td>
       <td><code>unsigned int</code></td>
       <td colspan="3" rowspan="2">1</td> <td rowspan="2"><span style="color:red;">0</span></td>
     </tr>
     <tr>
       <td colspan="2">
         Force Hessian / covariance matrix to be positive-definite.

         It can be useful to return the uncorrected covariance matrix, even if it is not positive
      definite.

      One use case is to check just how far from positive-definiteness the matrix is by being able to examine the eigenvalues.
       </td>
     </tr>
     <tr>
       <td>**HessianG2Tolerance**</td>
       <td><code>double</code></td>
       <td rowspan="2">0.1</td> <td rowspan="2">0.05</td> <td rowspan="2">0.02</td> <td rowspan="2">zero</td>
     </tr>
     <tr>
       <td colspan="2">
         Stop the Hessian diagonal refinement cycle early if the second derivative estimate itself is stable.

         The parameter refers to the change in the second derivative estimate relative to the new estimate.

         This is the partner of the **HessianStepTolerance** parameter.

         In some cases, it can help to set it to zero. This was found to be
      necessary in cases where Asimov datasets were used for the
      minimization and there were very few iterations for the approximate
      covariance to be determined from.
       </td>
     </tr>
     <tr>
       <td>**HessianGradientNCycles**</td>
       <td><code>unsigned int</code></td>
       <td rowspan="2">1</td> <td rowspan="2">2</td> <td rowspan="2" colspan="2">6</td>
     </tr>
     <tr>
       <td colspan="2">
         Maximum number of refinement cycles used when the first derivatives are
      recomputed together with the Hessian (see **RefineGradientInHessian**).

      This is the analogue of **GradientNCycles**, but for the more accurate gradient
      evaluated during the Hessian calculation.
       </td>
     </tr>
     <tr>
       <td>**HessianNCycles**</td>
       <td><code>unsigned int</code></td>
       <td rowspan="2">3</td> <td rowspan="2">5</td> <td rowspan="2" colspan="2">7</td>
     </tr>
     <tr>
       <td colspan="2">
         Maximum number of cycles used to compute each diagonal element of the Hessian
      (the second derivative with respect to a single parameter). More cycles refine
      the finite-difference step and give a more accurate second derivative.

      A cycle is stopped early once either **HessianStepTolerance** or **HessianG2Tolerance** is met.
       </td>
     </tr>
     <tr>
       <td>**HessianRecomputeThreshold**</td>
       <td><code>double</code></td>
       <td rowspan="2">inf</td> <td rowspan="2">0.05</td> <td rowspan="2" colspan="2">-inf</td>
     </tr>
     <tr>
       <td colspan="2">
         After a Migrad minimization, a full Hesse recomputation of the error matrix is
      triggered when the relative change of the covariance matrix in the last step
      (`Dcovar`) exceeds this threshold.

      A value of *inf* means the Hessian is never recomputed and the approximate
      covariance accumulated during Migrad is kept, while *-inf* means it is always
      recomputed.
       </td>
     </tr>
     <tr>
       <td>**HessianStepTolerance**</td>
       <td><code>double</code></td>
       <td rowspan="2">0.5</td> <td rowspan="2">0.3</td> <td rowspan="2">0.1</td> <td rowspan="2">zero</td>
     </tr>
     <tr>
       <td colspan="2">
         Stop the Hessian diagonal refinement cycle early if the estimated optimal finite-difference step size has stabilized.

         The parameter refers to the change in step size relative to the new step size.

         This is the partner of the **HessianG2Tolerance** parameter.

         Just like with that parameter, it can make sense to set the tolerance to zero to ensure the most accurate Hessians.
       </td>
     </tr>
     <tr>
       <td>**StorageLevel**</td>
       <td><code>int</code></td>
       <td rowspan="2" colspan="4">1</td>
     </tr>
     <tr>
       <td colspan="2">
         How much of the minimization history is kept: 1 stores the full sequence of
      iterations, while 0 keeps only the last iterations.

      This does not affect the minimization result, only the trace that can be
      inspected afterwards.
       </td>
     </tr>
   </table>
 */

class MnStrategy {

public:
   // default strategy
   MnStrategy();

   // user defined strategy (0, 1, 2, >=3)
   explicit MnStrategy(unsigned int);

   unsigned int GradientNCycles() const { return fGradNCyc; }
   double GradientStepTolerance() const { return fGradTlrStp; }
   double GradientTolerance() const { return fGradTlr; }

   unsigned int HessianNCycles() const { return fHessNCyc; }
   double HessianStepTolerance() const { return fHessTlrStp; }
   double HessianG2Tolerance() const { return fHessTlrG2; }
   unsigned int HessianGradientNCycles() const { return fHessGradNCyc; }
   unsigned int HessianCentralFDMixedDerivatives() const { return fHessCFDG2; }
   unsigned int HessianForcePosDef() const { return fHessForcePosDef; }

   int StorageLevel() const { return fStoreLevel; }

   bool RefineGradientInHessian() const { return fStrategy > 0; }

   bool ComputeInitialHessian() const { return fStrategy == 2; }

   double HessianRecomputeThreshold() const;

   void SetGradientNCycles(unsigned int n) { fGradNCyc = n; }
   void SetGradientStepTolerance(double stp) { fGradTlrStp = stp; }
   void SetGradientTolerance(double toler) { fGradTlr = toler; }

   void SetHessianNCycles(unsigned int n) { fHessNCyc = n; }
   void SetHessianStepTolerance(double stp) { fHessTlrStp = stp; }
   void SetHessianG2Tolerance(double toler) { fHessTlrG2 = toler; }
   void SetHessianGradientNCycles(unsigned int n) { fHessGradNCyc = n; }

   // 1 = calculate central finite difference mixed derivatives (involves 3 extra evaluations per derivative)
   // 0 = use forward finite difference (default)
   void SetHessianCentralFDMixedDerivatives(unsigned int flag) { fHessCFDG2 = flag; }

   // 1 = returned matrix from Hesse should be forced positive definite (default)
   // 0 = do not force matrix positive definite
   void SetHessianForcePosDef(unsigned int flag) { fHessForcePosDef = flag; }

   // set storage level of iteration quantities
   // 0 = store only last iterations 1 = full storage (default)
   void SetStorageLevel(unsigned int level) { fStoreLevel = level; }

private:
   friend class MnFunctionCross;
   friend class MnContours;
   MnStrategy NextLower() const;

   void SetLowStrategy();
   void SetMediumStrategy();
   void SetHighStrategy();
   void SetVeryHighStrategy();

   unsigned int fStrategy;

   unsigned int fGradNCyc;
   double fGradTlrStp;
   double fGradTlr;
   unsigned int fHessNCyc;
   double fHessTlrStp;
   double fHessTlrG2;
   unsigned int fHessGradNCyc;
   int fHessCFDG2;
   int fHessForcePosDef;
   int fStoreLevel;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnStrategy
