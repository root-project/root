// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Sep 22 15:06:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
// Header file for class TFoamSampler

#ifndef ROOT_TFoamSampler
#define ROOT_TFoamSampler


#include "Math/DistSampler.h"


namespace ROOT {

   namespace Fit {

      class DataRange;
      class BinData;
      class UnBinData;
   }

   namespace Math {
   }
}

class TFoamIntegrand;


class TRandom;
class TF1;
class TFoam;

class TFoamSampler : public ROOT::Math::DistSampler {

public:

   /// default constructor
   TFoamSampler();


   /// virtual destructor
   ~TFoamSampler() override;


   using DistSampler::SetFunction;

   /// set the parent function distribution to use for random sampling (one dim case)
   void SetFunction(const ROOT::Math::IGenFunction & func) override  {
      fFunc1D = &func;
      SetFunction<const ROOT::Math::IGenFunction>(func, 1);
   }

   /// set the Function using a TF1 pointer
   void SetFunction(TF1 * pdf);


   /**
      initialize the generators with the default options
   */
   bool Init(const char * = "") override;

   /**
      initialize the generators with the given options
   */
   bool Init(const ROOT::Math::DistSamplerOptions & opt ) override;

   /**
       Set the random engine to be used
       Needs to be called before Init to have effect
   */
   void SetRandom(TRandom * r) override;

   /**
       Set the random seed for the TRandom instances used by the sampler
       classes
       Needs to be called before Init to have effect
   */
   void SetSeed(unsigned int seed) override;


   /**
      Get the random engine used by the sampler
    */
   TRandom * GetRandom() override;


   /**
      sample one event in multi-dimension by filling the given array
      return false if sampling failed
   */
   bool Sample(double * x) override;

   /**
      sample one bin given an estimated of the pdf in the bin
      (this can be function value at the center or its integral in the bin
      divided by the bin width)
      By default do not do random sample, just return the function values
    */
   bool SampleBin(double prob, double & value, double *error = 0) override;



protected:


private:

//    bool                              fOneDim;      // flag to indicate if the function is 1 dimension
//    bool                              fHasMode;     // flag to indicate if a mode is set
//    bool                              fHasArea;     // flag to indicate if a area is set
//    double                            fMode;        // mode of dist
//    double                            fArea;        // area of dist
   const ROOT::Math::IGenFunction *  fFunc1D;      // 1D function pointer
   TFoam *                           fFoam;        // foam engine class
   TFoamIntegrand *                  fFoamDist;    // foam distribution interface

   //ClassDefOverride(TFoamSampler,1)  //Distribution sampler class based on FOAM

};



#endif /* ROOT_TFoamSampler */
