// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Sep 22 15:06:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
// Header file for class TUnuranSampler

#ifndef ROOT_TUnuranSampler
#define ROOT_TUnuranSampler


#ifndef ROOT_Math_DistSampler
#include "Math/DistSampler.h"
#endif


namespace ROOT {

   namespace Fit {

      class DataRange;
      class BinData;
      class UnBinData;
   }

   namespace Math {
   }
}



//_______________________________________________________________________________
/**
   TUnuranSampler class
   class implementing  the ROOT::Math::DistSampler interface using the UNU.RAN
   package for sampling distributions.


*/

class TRandom;
class TF1;
class TUnuran;

class TUnuranSampler : public ROOT::Math::DistSampler {

public:

   /// default constructor
   TUnuranSampler();


   /// virtual destructor
   virtual ~TUnuranSampler();


   using DistSampler::SetFunction;

   /// set the parent function distribution to use for random sampling (one dim case)
   void SetFunction(const ROOT::Math::IGenFunction & func)  {
      fFunc1D = &func;
      SetFunction<const ROOT::Math::IGenFunction>(func, 1);
   }

   /// set the Function using a TF1 pointer
   void SetFunction(TF1 * pdf);


   /**
      initialize the generators with the given algorithm
      If no algorithm is passed used the default one for the type of distribution
   */
   bool Init(const char * algo ="");


   /**
      initialize the generators with the given algorithm
      If no algorithm is passed used the default one for the type of distribution
   */
   bool Init(const ROOT::Math::DistSamplerOptions & opt );

   /**
       Set the random engine to be used
       Needs to be called before Init to have effect
   */
   void SetRandom(TRandom * r);

   /**
       Set the random seed for the TRandom instances used by the sampler
       classes
       Needs to be called before Init to have effect
   */
   void SetSeed(unsigned int seed);

   /**
      Set the print level
      (if level=-1 use default)
    */
   void SetPrintLevel(int level) {fLevel = level;}

   /*
      set the mode
    */
   void SetMode(double mode) {
      fMode = mode;
      fHasMode = true;
   }

   /*
     set the area
    */
   void SetArea(double area) {
      fArea = area;
      fHasArea = true;
   }

   /**
      Get the random engine used by the sampler
    */
   TRandom * GetRandom();

   /**
      sample one event in one dimension
      better implementation could be provided by the derived classes
   */
   double Sample1D();//  {
//       return fUnuran->Sample();
//    }

   /**
      sample one event in multi-dimension by filling the given array
      return false if sampling failed
   */
   bool Sample(double * x);
//  {
//       if (!fOneDim) return fUnuran->SampleMulti(x);
//       x[0] = Sample1D();
//       return true;
//    }

   /**
      sample one bin given an estimated of the pdf in the bin
      (this can be function value at the center or its integral in the bin
      divided by the bin width)
      By default do not do random sample, just return the function values
    */
   bool SampleBin(double prob, double & value, double *error = 0);



protected:

   //initialization for 1D distributions
   bool DoInit1D(const char * algo);
   //initialization for 1D discrete distributions
   bool DoInitDiscrete1D(const char * algo);
   //initialization for multi-dim distributions
   bool DoInitND(const char * algo);


private:

   // private member
   bool                              fOneDim;      // flag to indicate if the function is 1 dimension
   bool                              fDiscrete;    // flag to indicate if the function is discrete
   bool                              fHasMode;     // flag to indicate if a mode is set
   bool                              fHasArea;     // flag to indicate if a area is set
   int                               fLevel;       // debug level
   double                            fMode;        // mode of dist
   double                            fArea;        // area of dist
   const ROOT::Math::IGenFunction *  fFunc1D;      // 1D function pointer
   TUnuran *                         fUnuran;      // unuran engine class

   //ClassDef(TUnuranSampler,1)  //Distribution sampler class based on UNU.RAN

};



#endif /* ROOT_TUnuranSampler */
