#include "TH1.h"
#include "TF1.h"
#include "TRandom.h"
#include "TFitResult.h"
#include "TError.h"
#include "TROOT.h"
#include "Math/MinimizerOptions.h"
#include <chrono>

double tolerance = 0.01;

// Evaluating function
template <class T>
T func(const T *data, const double *params)
{
   return params[0] * exp(-(*data + (-130.)) * (*data + (-130.)) / 2) +
          params[1] * exp(-(params[2] * (*data * (0.01)) - params[3] * ((*data) * (0.01)) * ((*data) * (0.01))));
}

struct FitModelData {
   std::chrono::duration<double> refTime;
   TFitResultPtr refValue;
   std::string name;
   std::vector<double> speedups;
};

// Function that calls the fit, checks for numerical correctness and computes speed up
void benchmarkFit(TF1 *f, TH1D &h, const std::string &opt, const std::string &fit, FitModelData &model)
{
   std::chrono::time_point<std::chrono::system_clock> start, end;
   auto msg = fit + " " + model.name + " Fit";
   std::cout << "\n\n **" << msg << "**\n";
   f->SetParameters(1, 1000, 7.5, 1.5);
   start = std::chrono::system_clock::now();
   auto result = h.Fit(f, opt.c_str());
   end = std::chrono::system_clock::now();
   std::chrono::duration<double> duration = end - start;

   // Check if Fit succeeds
   if ((Int_t)result != 0) {
      Error("testBinnedFitExecPolicy", "%s failed!", msg.c_str());
      exit(-1);
   }

   if (fit == "Sequential") {
      // Save reference times and values
      model.refTime = end - start;
      model.refValue = result;
   } else {
      // Check if numerical result is within tolerance
      if (std::abs(result->MinFcnValue() - model.refValue->MinFcnValue()) >
          tolerance * std::abs(model.refValue->MinFcnValue())) {
         Error("testBinnedFitExecPolicy", "%s : Failed comparison of fit results \t FCN = %f, it should be = %f",
               msg.c_str(), result->MinFcnValue(), model.refValue->MinFcnValue());
         exit(-1);
      }
      // Compute speedup
      model.speedups.emplace_back(model.refTime.count() / duration.count() * result->NCalls() /
                                  model.refValue->NCalls());
   }

   std::cout << "Time for the " << msg << ": " << duration.count() << std::endl;
}

void printSpeedUps(std::vector<FitModelData> &models)
{
   std::cout.precision(2);
   std::string longestName =
      std::max_element(models.begin(), models.end(), [](const FitModelData &m1, const FitModelData &m2) {
         return m1.name.size() < m2.name.size();
      })->name;
   std::cout << std::endl << "\n   ***Speedups! (normalized to the number of calls)***" << std::endl;
   std::cout << std::string(longestName.size() + 1, ' ') << "|  MT    "
             << "|  VEC   "
             << "| MT+VEC " << std::endl;
   // Name field + value field + three bars + 1 extra
   std::cout << std::string(longestName.size() + 1 + 8 * 3 + 3 + 1, '-') << std::endl;
   for (auto const &m : models) {
      std::cout << m.name << std::string(longestName.size() - m.name.size() + 1, ' ');
      for (auto su : m.speedups)
         std::cout << std::fixed << "|  " << su << "  ";
      std::cout << std::endl;
   }
   std::cout << std::endl;
}

int main()
{

   ROOT::EnableImplicitMT();

   TF1 *f = new TF1("fvCore", func<double>, 100, 200, 4);
   f->SetParameters(1, 1000, 7.5, 1.5);

   // NBins not multiple of SIMD vector size, testing padding
   TH1D h1f("h1f", "Test random numbers", 12801, 100, 200);
   gRandom->SetSeed(1);
   h1f.FillRandom("fvCore", 1000000);
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");

   enum EModel { kChi2 = 0, kPoisson };
   std::vector<FitModelData> models(2);
   models[kChi2].name = "Chi2";
   models[kPoisson].name = "Binned Likelihood";

   // Start testing
   auto fit = "Sequential";
   benchmarkFit(f, h1f, "S SERIAL", fit, models[EModel::kChi2]);
   benchmarkFit(f, h1f, "SERIAL S L", fit, models[EModel::kPoisson]);

#ifdef R__USE_IMT
   fit = "Multithreaded";
   benchmarkFit(f, h1f, "S", fit, models[EModel::kChi2]);
   benchmarkFit(f, h1f, "S L", fit, models[EModel::kPoisson]);

#endif
#ifdef R__HAS_VECCORE
   TF1 *fvecCore = new TF1("fvCore", func<ROOT::Double_v>, 100, 200, 4);

   fit = "Vectorized";
   benchmarkFit(fvecCore, h1f, "SERIAL S", fit, models[EModel::kChi2]);
   benchmarkFit(fvecCore, h1f, "SERIAL S L", fit, models[EModel::kPoisson]);

#ifdef R__USE_IMT

   fit = "Mutithreaded and vectorized";
   benchmarkFit(fvecCore, h1f, "S", fit, models[EModel::kChi2]);
   benchmarkFit(fvecCore, h1f, "S L", fit, models[EModel::kPoisson]);

#endif
#endif
   printSpeedUps(models);
   return 0;
}
