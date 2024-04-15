// @(#)root/mathcore:$Id

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class GeneticMinimizer

#ifndef ROOT_Math_GeneticMinimizer
#define ROOT_Math_GeneticMinimizer

#include "Math/Minimizer.h"

#include "RtypesCore.h"

#include <vector>
#include <string>

namespace TMVA {
   class IFitterTarget;
   class Interval;
}

namespace ROOT {
   namespace Math {


//_______________________________________________________________________________
/*
  structure containing the parameters of the genetic minimizer
 */
struct GeneticMinimizerParameters {

   Int_t fPopSize;
   Int_t fNsteps;
   Int_t fCycles;
   Int_t fSC_steps;
   Int_t fSC_rate;
   Double_t fSC_factor;
   Double_t fConvCrit;
   Int_t fSeed;


   // constructor with default value
   GeneticMinimizerParameters();
};



//_______________________________________________________________________________
/**
   GeneticMinimizer

   Minimizer class based on the Gentic algorithm implemented in TMVA

   @ingroup MultiMin
*/
class GeneticMinimizer: public ROOT::Math::Minimizer {

public:

   //GeneticMinimizer (int = 0);
   GeneticMinimizer (int i = 0);
   ~GeneticMinimizer () override;

   void Clear() override;
   using ROOT::Math::Minimizer::SetFunction;
   void SetFunction(const ROOT::Math::IMultiGenFunction & func) override;

   bool SetLimitedVariable(unsigned int , const std::string& , double , double , double, double) override;
   bool SetVariable(unsigned int ivar, const std::string & name, double val, double step) override;
   bool SetFixedVariable(unsigned int ivar  , const std::string & name , double val) override;

    bool Minimize() override;
   double MinValue() const override;
   double Edm() const override;
   const double *  X() const override;
   const double *  MinGradient() const override;
   unsigned int NCalls() const override;

   unsigned int NDim() const override;
   unsigned int NFree() const override;

   bool ProvidesError() const override;
   const double * Errors() const override;

   double CovMatrix(unsigned int i, unsigned int j) const override;

   void SetParameters(const GeneticMinimizerParameters & params );

   void SetRandomSeed(int seed) { fParameters.fSeed = seed; }

   const GeneticMinimizerParameters & MinimizerParameters() const { return fParameters; }

   ROOT::Math::MinimizerOptions Options() const override;

   virtual void SetOptions(const ROOT::Math::MinimizerOptions & opt);

protected:

   void GetGeneticOptions(ROOT::Math::MinimizerOptions & opt) const;

   std::vector<TMVA::Interval*> fRanges;
   TMVA::IFitterTarget* fFitness;
   double fMinValue;
   std::vector<double> fResult;

   GeneticMinimizerParameters fParameters;

};


   } // end namespace Math
} // end namespace ROOT

#endif /* ROOT_Math_GeneticMinimizer */
