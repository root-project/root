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
#include "TUnuranDiscrDist.h"
#include "TUnuranMultiContDist.h"
#include "TUnuran.h"
#include "Math/OneDimFunctionAdapter.h"
#include "Math/DistSamplerOptions.h"
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
   fDiscrete(false),
   fHasMode(false), fHasArea(false),
   fMode(0), fArea(0),
   fFunc1D(0),
   fUnuran(new TUnuran()  )
{
   fLevel = ROOT::Math::DistSamplerOptions::DefaultPrintLevel();
}

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

   if (fLevel < 0) fLevel =  ROOT::Math::DistSamplerOptions::DefaultPrintLevel();

   TString method(algo); 
   if (method.IsNull() ) { 
      if (NDim() == 1) method = ROOT::Math::DistSamplerOptions::DefaultAlgorithm1D();
      else  method = ROOT::Math::DistSamplerOptions::DefaultAlgorithmND();
   }
   method.ToUpper();

   bool ret = false; 
   if (NDim() == 1) { 
       // check if distribution is discrete by 
      // using first string in the method name is "D"
      if (method.First("D") == 0) { 
         if (fLevel>1) Info("TUnuranSampler::Init","Initialize one-dim discrete distribution with method %s",method.Data());
         ret =  DoInitDiscrete1D(method);
      }
      else {
         if (fLevel>1) Info("TUnuranSampler::Init","Initialize one-dim continous distribution with method %s",method.Data());
         ret =  DoInit1D(method); 
      }
   }
   else { 
      if (fLevel>1) Info("TUnuranSampler::Init","Initialize multi-dim continous distribution with method %s",method.Data());
      ret = DoInitND(method); 
   }
   // set print level in UNURAN (must be done after having initialized) -
   if (fLevel>0) { 
      //fUnuran->SetLogLevel(fLevel); ( seems not to work  disable for the time being) 
      if (ret) Info("TUnuranSampler::Init","Successfully initailized Unuran with method %s",method.Data() );
      else Error("TUnuranSampler::Init","Failed to  initailize Unuran with method %s",method.Data() );
      // seems not to work in UNURAN (cll only when level > 0 )
   }
   return ret; 
}


bool TUnuranSampler::Init(const ROOT::Math::DistSamplerOptions & opt ) { 
   // default initialization with algorithm name
   SetPrintLevel(opt.PrintLevel() );
   return Init(opt.Algorithm().c_str() );
}


bool TUnuranSampler::DoInit1D(const char * method) { 
   // initilize for 1D sampling
   // need to create 1D interface from Multidim one 
   // (to do: use directly 1D functions ??)
   fOneDim = true; 
   TUnuranContDist * dist = 0;
   if (fFunc1D == 0) { 
      ROOT::Math::OneDimMultiFunctionAdapter<> function(ParentPdf() ); 
      dist = new TUnuranContDist(function,0,false,true); 
   }
   else { 
      dist = new TUnuranContDist(*fFunc1D); // no need to copy the function
   }
   // set range in distribution (support only one range)
   const ROOT::Fit::DataRange & range = PdfRange(); 
   if (range.Size(0) > 0) { 
      double xmin, xmax; 
      range.GetRange(0,xmin,xmax); 
      dist->SetDomain(xmin,xmax); 
   }
   if (fHasMode) dist->SetMode(fMode);
   if (fHasArea) dist->SetPdfArea(fArea);

   bool ret = false; 
   if (method) ret =  fUnuran->Init(*dist, method);       
   else ret =  fUnuran->Init(*dist);
   delete dist; 
   return ret; 
}

bool TUnuranSampler::DoInitDiscrete1D(const char * method) { 
   // initilize for 1D sampling of discrete distributions
   fOneDim = true; 
   fDiscrete = true;
   TUnuranDiscrDist * dist = 0;
   if (fFunc1D == 0) { 
      // need to copy the passed function pointer in this case
      ROOT::Math::OneDimMultiFunctionAdapter<> function(ParentPdf() ); 
      dist = new TUnuranDiscrDist(function,true); 
   }
   else { 
      // no need to copy the function since fFunc1D is managed outside
      dist = new TUnuranDiscrDist(*fFunc1D, false); 
   }
   // set range in distribution (support only one range)
   // otherwise 0, inf is assumed
   const ROOT::Fit::DataRange & range = PdfRange(); 
   if (range.Size(0) > 0) { 
      double xmin, xmax; 
      range.GetRange(0,xmin,xmax);
      if (xmin < 0) { 
         Warning("DoInitDiscrete1D","range starts from negative values - set minimum to zero"); 
         xmin = 0; 
      }
      dist->SetDomain(int(xmin+0.1),int(xmax+0.1)); 
   }
   if (fHasMode) dist->SetMode(int(fMode+0.1));
   if (fHasArea) dist->SetProbSum(fArea);

   bool ret =  fUnuran->Init(*dist, method);       
   delete dist;
   return ret;
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
//       std::cout << " range is min = "; 
//       for (int j = 0; j < NDim(); ++j) std::cout << xmin[j] << "   "; 
//       std::cout << " max = "; 
//       for (int j = 0; j < NDim(); ++j) std::cout << xmax[j] << "   "; 
//       std::cout << std::endl;
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
   return (fDiscrete) ? (double) fUnuran->SampleDiscr() : fUnuran->Sample(); 
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
