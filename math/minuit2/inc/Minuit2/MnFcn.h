// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnFcn
#define ROOT_Minuit2_MnFcn

#include "Minuit2/FCNBase.h"
#include "Minuit2/MnMatrix.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

class MnUserTransformation;

class FCNBase;
/**
   Wrapper class to FCNBase interface used internally by Minuit.
   Apply conversion from calling the function from a Minuit Vector (MnAlgebraicVector) to a std::vector  for
   the function coordinates.
   The class counts also the number of function calls. By default counter start from zero, but a different value
   might be given if the class is  instantiated later on, for example for a set of different minimizaitons
   Normally the derived class MnUserFCN should be instantiated with performs in addition the transformatiopn
   internal-> external parameters
 */
class MnFcn {

public:
   explicit MnFcn(const FCNBase &fcn, int ncall = 0) : fFCN(fcn), fNumCall(ncall) {}
   explicit MnFcn(const FCNBase &fcn, const MnUserTransformation &trafo, int ncall = 0)
      : fFCN(fcn), fNumCall(ncall), fTransform(&trafo)
   {
   }

   unsigned int NumOfCalls() const { return fNumCall; }

   double ErrorDef() const
   {
      return fFCN.Up();
   }

   double Up() const
   {
      return fFCN.Up();
   }

   const FCNBase &Fcn() const { return fFCN; }

   // Access the parameter transformations.
   // For internal use in the Minuit2 implementation.
   const MnUserTransformation *Trafo() const { return fTransform; }

   double CallWithTransformedParams(std::vector<double> const &vpar) const;
   double CallWithoutDoingTrafo(const MnAlgebraicVector &) const;

private:
   const FCNBase &fFCN;
   mutable int fNumCall;
   const MnUserTransformation *fTransform = nullptr;
};

// Helper class to call the MnFcn, caching the transformed parameters if necessary.
class MnFcnCaller {
public:
   MnFcnCaller(const MnFcn &mfcn);

   double operator()(const MnAlgebraicVector &v);

private:
   MnFcn const &fMfcn;
   bool fDoInt2ext = false;
   std::vector<double> fLastInput;
   std::vector<double> fVpar;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnFcn
