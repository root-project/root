#include "TH1.h"
#include "TF1.h"
#include "TFile.h"
#include "TRandom.h"
#include "Math/IParamFunction.h"
#include "TFitResult.h"
#include "TError.h"

int compareResult(double v1, double v2, std::string s = "", double tol = 0.01)
{
   // compare v1 with reference v2
   //  // give 1% tolerance
   if (std::abs(v1 - v2) < tol * std::abs(v2)) return 0;
   std::cerr << s << " Failed comparison of fit results \t chi2 = " << v1 << "   it should be = " << v2 << std::endl;
   return -1;
}

template<class T>
T func(const T *data, const double *params)
{
   return params[0] * exp(-(*data + (-130.)) * (*data + (-130.)) / 2) +
          params[1] * exp(-(params[2] * (*data * (0.01)) -
                            params[3] * ((*data) * (0.01)) * ((*data) * (0.01))));
}

int main()
{
   TF1 *f = new TF1("fvCore", func<double>, 100, 200, 4);
   f->SetParameters(1, 1000, 7.5, 1.5);
   TH1D h1f("h1f", "Test random numbers", 128000 , 100, 200);
   gRandom->SetSeed(1);
   h1f.FillRandom("fvCore", 1000000);

   auto r1 = h1f.Fit(f, "S");
   if ((Int_t) r1 != 0) {
      Error("testChi2ExecPolicy", "Sequential Fit failed!");
      return -1;
   }

#ifdef R__USE_IMT
   auto r2 = h1f.Fit(f, "MULTITHREAD S");
   if ((Int_t) r2 != 0) {
      Error("testChi2ExecPolicy", "Sequential Fit failed!");
      return -1;
   } else {
      compareResult(r2->Chi2(), r1->Chi2(), "Mutithreaded Chi2 Fit: ");
   }
#endif

#ifdef R__HAS_VECCORE
   TF1 *fvecCore = new TF1("fvCore", func<ROOT::Double_v>, 100, 200, 4);
   fvecCore->SetParameters(1, 1000, 7.5, 1.5);
   auto r3 = h1f.Fit(fvecCore, "S");
   if ((Int_t) r3 != 0) {
      Error("testChi2ExecPolicy", "Vectorized Fit failed!");
      return -1;
   } else {
      compareResult(r3->Chi2(), r1->Chi2(), "Vectorized Chi2 Fit: ");
   }

#ifdef R__USE_IMT
   auto r4 = h1f.Fit(fvecCore, "MULTITHREAD S");
   if ((Int_t) r4 != 0) {
      Error("testChi2ExecPolicy", "Multithreaded vectorized Fit failed!");
      return -1;
   } else {
      compareResult(r4->Chi2(), r1->Chi2(), "Mutithreaded vectorized Chi2 Fit: ");
   }

#endif
#endif
}
