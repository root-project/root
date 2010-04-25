// @(#)root/unuran:$Id$
// Authors: L. Moneta, J. Leydold Wed Feb 28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2010  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TUnuranSampler
#include "TUnuranSampler.h"

#include "TUnuranContDist.h"
#include "TUnuranMultiContDist.h"
#include "TUnuran.h"
#include "Math/OneDimFunctionAdapter.h"
#include "Fit/DataRange.h"
//#include "Math/WrappedTF1.h"

#include "TRandom.h"
#include "TError.h"

#include "TF1.h"
#include <cassert>
#include <cmath>

ClassImp(TUnuranSampler)

TUnuranSampler::TUnuranSampler() : ROOT::Math::DistSampler(), 
   fOneDim(false), 
   fFunc1D(0),
   fUnuran(new TUnuran()  )
{}

TUnuranSampler::~TUnuranSampler() {
   assert(fUnuran != 0);
   delete fUnuran; 
}

bool TUnuranSampler::Init(const char * algo) { 
   // initialize unuran classes using the given algorithm
   assert (fUnuran != 0 );
   if (NDim() == 0)  {
      Error("TUnuranSampler::Init","Distribution function has not been set ! Need to call SetFunction first.");
      return false;
   }
   if (NDim() == 1) return DoInit1D(algo); 
   else return DoInitND(algo); 
}

bool TUnuranSampler::DoInit1D(const char * method) { 
   // initilize for 1D sampling
   // need to create 1D interface from Multidim one 
   // (to do: use directly 1D functions ??)
   fOneDim = true; 
   TUnuranContDist dist;
   if (fFunc1D == 0) { 
      ROOT::Math::OneDimMultiFunctionAdapter<> function(ParentPdf() ); 
      dist = TUnuranContDist(function,0,false,true); 
   }
   else { 
      dist = TUnuranContDist(*fFunc1D); // no need to copy the function
   }
   // set range in distribution (support only one range)
   const ROOT::Fit::DataRange & range = PdfRange(); 
   if (range.Size(0) > 0) { 
      double xmin, xmax; 
      range.GetRange(0,xmin,xmax); 
      dist.SetDomain(xmin,xmax); 
   }
   if (method) return fUnuran->Init(dist, method);       
   return fUnuran->Init(dist);
}

bool TUnuranSampler::DoInitND(const char * method) { 
   // initilize for 1D sampling
   TUnuranMultiContDist dist(ParentPdf()); 
   // set range in distribution (support only one range)
   const ROOT::Fit::DataRange & range = PdfRange(); 
   if (range.IsSet()) { 
      std::vector<double> xmin(range.NDim() ); 
      std::vector<double> xmax(range.NDim() ); 
      range.GetRange(&xmin[0],&xmax[0]); 
      dist.SetDomain(&xmin.front(),&xmax.front()); 
   }
   fOneDim = false; 
   if (method) return fUnuran->Init(dist, method); 
   return fUnuran->Init(dist);
}

void TUnuranSampler::SetFunction(TF1 * pdf) { 
   // set function from a TF1 pointer 
   SetFunction<TF1>(*pdf, pdf->GetNdim());
} 

void TUnuranSampler::SetRandom(TRandom * r) { 
   // set random generator (must be called before Init to have effect)
   fUnuran->SetRandom(r); 
} 

void TUnuranSampler::SetSeed(unsigned int seed) { 
   // set random generator seed (must be called before Init to have effect)
   fUnuran->SetSeed(seed); 
} 

TRandom * TUnuranSampler::GetRandom() { 
   // get random generator used 
   return  fUnuran->GetRandom(); 
} 

double TUnuranSampler::Sample1D() { 
   // sample 1D distributions
   return fUnuran->Sample(); 
}

bool TUnuranSampler::Sample(double * x) { 
   // sample multi-dim distributions
   if (!fOneDim) return fUnuran->SampleMulti(x); 
   x[0] = Sample1D(); 
   return true; 
} 


bool TUnuranSampler::SampleBin(double prob, double & value, double *error) {
   // sample a bin according to Poisson statistics

   TRandom * r = fUnuran->GetRandom(); 
   if (!r) return false; 
   value = r->Poisson(prob); 
   if (error) *error = std::sqrt(value);
   return true; 
}
