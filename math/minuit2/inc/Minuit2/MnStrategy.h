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

//_________________________________________________________________________
/**
    API class for defining four levels of strategies: low (0), medium (1),
    high (2), very high (>=3);
    acts on: Migrad (behavioural),
             Minos (lowers strategy by 1 for Minos-own minimization),
             Hesse (iterations),
             Numerical2PDerivative (iterations)
 */

class MnStrategy {

public:
   // default strategy
   MnStrategy();

   // user defined strategy (0, 1, 2, >=3)
   explicit MnStrategy(unsigned int);

   unsigned int Strategy() const { return fStrategy; }

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

   bool IsLow() const { return fStrategy == 0; }
   bool IsMedium() const { return fStrategy == 1; }
   bool IsHigh() const { return fStrategy == 2; }
   bool IsVeryHigh() const { return fStrategy >= 3; }

   void SetLowStrategy();
   void SetMediumStrategy();
   void SetHighStrategy();
   void SetVeryHighStrategy();

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
