#include "TH1.h"
#include "TF1.h"
#include "TFile.h"
#include "TRandom.h"
#include "Math/IParamFunction.h"
#include "TFitResult.h"
#include "TError.h"
#include "Math/MinimizerOptions.h"

bool compareResult(double v1, double v2, std::string s = "", double tol = 0.01)
{
   // compare v1 with reference v2
   //  // give 1% tolerance
   if (std::abs(v1 - v2) < tol * std::abs(v2)) return true;
   std::cerr << s << " Failed comparison of fit results \t chi2 = " << v1 << "   it should be = " << v2 << std::endl;
   return false;
}

template <class T>
T func(const T *data, const double *params)
{
   return params[0] * exp(-(*data + (-130.)) * (*data + (-130.)) / 2) +
          params[1] * exp(-(params[2] * (*data * (0.01)) - params[3] * ((*data) * (0.01)) * ((*data) * (0.01))));
}

int main()
{
   TF1 *f = new TF1("fvCore", func<double>, 100, 200, 4);
   f->SetParameters(1, 1000, 7.5, 1.5);

   // NBins not multiple of SIMD vector size, testing padding
   TH1D h1f("h1f", "Test random numbers", 12801, 100, 200);
   gRandom->SetSeed(1);
   h1f.FillRandom("fvCore", 1000000);

   std::cout << "\n **FIT: Chi2 **\n\n";
   auto r1 = h1f.Fit(f, "S");
   if ((Int_t)r1 != 0) {
      Error("testBinnedFitExecPolicy", "Sequential Chi2 Fit failed!");
      return -1;
   }

   std::cout << "\n **FIT: Binned Likelihood **\n\n";
   auto rL1 = h1f.Fit(f, "S L");
   if ((Int_t)rL1 != 0) {
      Error("testBinnedFitExecPolicy", "Sequential  Binned Likelihood Fit failed!");
      return -1;
   }

#ifdef R__USE_IMT
   std::cout << "\n **FIT: Multithreaded Chi2 **\n\n";
   f->SetParameters(1, 1000, 7.5, 1.5);
   auto r2 = h1f.Fit(f, "MULTITHREAD S");
   if ((Int_t)r2 != 0) {
      Error("testBinnedFitExecPolicy", "Multithreaded Chi2 Fit failed!");
      return -1;
   } else {
      if (!compareResult(r2->MinFcnValue(), r1->MinFcnValue(), "Mutithreaded Chi2 Fit: "))
         return 1;
   }

   std::cout << "\n **FIT: Multithreaded Binned Likelihood **\n\n";
   f->SetParameters(1, 1000, 7.5, 1.5);
   auto rL2 = h1f.Fit(f, "MULTITHREAD S L");
   if ((Int_t)rL2 != 0) {
      Error("testBinnedFitExecPolicy", "Multithreaded Binned Likelihood Fit failed!");
      return -1;
   } else {
      if (!compareResult(rL2->MinFcnValue(), rL1->MinFcnValue(), "Mutithreaded Binned Likelihood Fit (PoissonLogL): "))
         return 2;
   }
#endif

#ifdef R__HAS_VECCORE
   TF1 *fvecCore = new TF1("fvCore", func<ROOT::Double_v>, 100, 200, 4);

   std::cout << "\n **FIT: Vectorized Chi2 **\n\n";
   fvecCore->SetParameters(1, 1000, 7.5, 1.5);
   auto r3 = h1f.Fit(fvecCore, "S");
   if ((Int_t)r3 != 0) {
      Error("testBinnedFitExecPolicy", "Vectorized Chi2 Fit failed!");
      return -1;
   } else {
      if (!compareResult(r3->MinFcnValue(), r1->MinFcnValue(), "Vectorized Chi2 Fit: "))
         return 3;
   }

   std::cout << "\n **FIT: Vectorized Binned Likelihood **\n\n";
   fvecCore->SetParameters(1, 1000, 7.5, 1.5);
   auto rL3 = h1f.Fit(fvecCore, "S L");
   if ((Int_t)rL3 != 0) {
      Error("testBinnedFitExecPolicy", "Vectorized Binned Likelihood Fit failed!");
      return -1;
   } else {
      if (!compareResult(rL3->MinFcnValue(), rL1->MinFcnValue(),
                         "Vectorized Binned Likelihood Fit (PoissonLogL) Fit: "))
         return 4;
   }

#ifdef R__USE_IMT
   std::cout << "\n **FIT: Mutithreaded vectorized Chi2 **\n\n";
   auto r4 = h1f.Fit(fvecCore, "MULTITHREAD S");
   if ((Int_t)r4 != 0) {
      Error("testBinnedFitExecPolicy", "Mutithreaded vectorized Chi2 Fit failed!");
      return -1;
   } else {
      if (!compareResult(r4->MinFcnValue(), r1->MinFcnValue(), "Mutithreaded vectorized Chi2 Fit: "))
         return 5;
   }

   std::cout << "\n **FIT: Multithreaded and vectorized Binned Likelihood **\n\n";
   fvecCore->SetParameters(1, 1000, 7.5, 1.5);
   auto rL4 = h1f.Fit(fvecCore, "MULTITHREAD S L");
   if ((Int_t)rL4 != 0) {
      Error("testBinnedFitExecPolicy", "Multithreaded Binned Likelihood vectorized Fit failed!");
      return -1;
   } else {
      if (!compareResult(rL4->MinFcnValue(), rL1->MinFcnValue(),
                         "Mutithreaded vectorized Binned Likelihood Fit (PoissonLogL) Fit: "))
         return 6;
   }

#endif
#endif
   return 0;
}
