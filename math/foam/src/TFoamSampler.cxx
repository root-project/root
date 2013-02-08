// @(#)root/unuran:$Id$
// Authors: L. Moneta,  Dec 2010

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2010  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TFoamSampler

#include "TFoamSampler.h"
#include "Math/DistSamplerOptions.h"

#include "TFoam.h"
#include "TFoamIntegrand.h"
#include "Math/OneDimFunctionAdapter.h"
#include "Math/IOptions.h"
#include "Fit/DataRange.h"

#include "TRandom.h"
#include "TError.h"

#include "TF1.h"
#include <cassert>
#include <cmath>

class FoamDistribution : public TFoamIntegrand { 

public:

   FoamDistribution(const ROOT::Math::IMultiGenFunction & f, const ROOT::Fit::DataRange & range) : 
      fFunc(f), 
      fX(std::vector<double>(f.NDim() ) ), 
      fMinX(std::vector<double>(f.NDim() ) ),
      fDeltaX(std::vector<double>(f.NDim() ) )
   {
      assert(f.NDim() == range.NDim() );
      std::vector<double> xmax(f.NDim() );
      for (unsigned int i = 0; i < range.NDim(); ++i) {
         if (range.Size(i) == 0)
            Error("FoamDistribution","Range is not set for coordinate dim %d",i);
         else if (range.Size(i)>1) 
            Warning("FoamDistribution","Using only first range in coordinate dim %d",i);

         std::pair<double,double> r = range(i); 
         fMinX[i] = r.first;
         fDeltaX[i] = r.second - r.first; 
      }
   } 
   // in principle function does not need to be cloned

   virtual double Density(int ndim, double * x) {
      assert(ndim == (int) fFunc.NDim() );
      for (int i = 0; i < ndim; ++i)
         fX[i] = fMinX[i] + x[i] * fDeltaX[i]; 

      return (fFunc)(&fX[0]);
   }

   double  MinX(unsigned int i) { return fMinX[i]; }
   double  DeltaX(unsigned int i) { return fDeltaX[i]; }
              
private:

   const ROOT::Math::IMultiGenFunction & fFunc;
   std::vector<double> fX; 
   std::vector<double> fMinX; 
   std::vector<double> fDeltaX; 

};



ClassImp(TFoamSampler)


//_______________________________________________________________________________
/**
   TFoamSampler class
   class implementing  the ROOT::Math::DistSampler interface using FOAM
   for sampling arbitrary distributions. 


*/
TFoamSampler::TFoamSampler() : ROOT::Math::DistSampler(), 
//    fOneDim(false), 
//    fDiscrete(false),
//    fHasMode(false), fHasArea(false),
//    fMode(0), fArea(0),
   fFunc1D(0),
   fFoam(new TFoam("FOAM")  ),
   fFoamDist(0)
{}

TFoamSampler::~TFoamSampler() {
   assert(fFoam != 0);
   delete fFoam; 
   if (fFoamDist) delete fFoamDist; 
}

bool TFoamSampler::Init(const char *) { 

   // initialize using default options
   ROOT::Math::DistSamplerOptions opt(0);
   ROOT::Math::IOptions * foamOpt  = ROOT::Math::DistSamplerOptions::FindDefault("Foam"); 
   if (foamOpt) opt.SetExtraOptions(*foamOpt); 
   return Init(opt);
}

bool TFoamSampler::Init(const ROOT::Math::DistSamplerOptions & opt) { 
   // initialize foam classes using the given algorithm
   assert (fFoam != 0 );
   if (NDim() == 0)  {
      Error("TFoamSampler::Init","Distribution function has not been set ! Need to call SetFunction first.");
      return false;
   }

   // initialize the foam 
   fFoam->SetkDim(NDim() );

   // initialize random number 
   if (!GetRandom()) SetRandom(gRandom);

   // create TFoamIntegrand class 
   if (fFoamDist) delete fFoamDist; 
   fFoamDist = new FoamDistribution(ParentPdf(),PdfRange());

   fFoam->SetRho(fFoamDist);
   // set print level
   fFoam->SetChat(opt.PrintLevel());

   // get extra options 
   ROOT::Math::IOptions * fopt = opt.ExtraOptions(); 
   if (fopt) { 
      int nval = 0; 
      double fval = 0;
      if (fopt->GetIntValue("nCells", nval) ) fFoam->SetnCells(nval);
      if (fopt->GetIntValue("nCell1D", nval) && NDim() ==1) fFoam->SetnCells(nval);
      if (fopt->GetIntValue("nCellND", nval) && NDim()  >1) fFoam->SetnCells(nval);
      if (fopt->GetIntValue("nCell2D", nval) && NDim() ==2) fFoam->SetnCells(nval);
      if (fopt->GetIntValue("nCell3D", nval) && NDim() ==3) fFoam->SetnCells(nval);

      if (fopt->GetIntValue("nSample", nval) ) fFoam->SetnSampl(nval);
      if (fopt->GetIntValue("nBin", nval) ) fFoam->SetnBin(nval);
      if (fopt->GetIntValue("OptDrive",nval) ) fFoam->SetOptDrive(nval);
      if (fopt->GetIntValue("OptRej",nval) ) fFoam->SetOptRej(nval);
      if (fopt->GetRealValue("MaxWtRej",fval) ) fFoam->SetMaxWtRej(fval);


      if (fopt->GetIntValue("chatLevel", nval) ) fFoam->SetChat(nval);
   }
   fFoam->Initialize();

   return true;
      
}


void TFoamSampler::SetFunction(TF1 * pdf) { 
   // set function from a TF1 pointer 
   SetFunction<TF1>(*pdf, pdf->GetNdim());
} 

void TFoamSampler::SetRandom(TRandom * r) { 
   // set random generator (must be called before Init to have effect)
   fFoam->SetPseRan(r); 
} 

void TFoamSampler::SetSeed(unsigned int seed) { 
   // set random generator seed (must be called before Init to have effect)
   TRandom * r = fFoam->GetPseRan();
   if (r) r->SetSeed(seed);
} 

TRandom * TFoamSampler::GetRandom() { 
   // get random generator used 
   return  fFoam->GetPseRan(); 
} 

// double TFoamSampler::Sample1D() { 
//    // sample 1D distributions
//    return (fDiscrete) ? (double) fFoam->SampleDiscr() : fFoam->Sample(); 
// }

bool TFoamSampler::Sample(double * x) { 
   // sample multi-dim distributions

   fFoam->MakeEvent();
   fFoam->GetMCvect(x);
   // adjust for the range   
   for (unsigned int i = 0; i < NDim(); ++i) 
      x[i] = ( (FoamDistribution*)fFoamDist)->MinX(i) + ( ( (FoamDistribution*) fFoamDist)->DeltaX(i))*x[i];

   return true; 
} 


bool TFoamSampler::SampleBin(double prob, double & value, double *error) {
   // sample a bin according to Poisson statistics

   TRandom * r = GetRandom(); 
   if (!r) return false; 
   value = r->Poisson(prob); 
   if (error) *error = std::sqrt(value);
   return true; 
}
