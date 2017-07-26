#include "Fit/Fitter.h"
#include "Math/WrappedMultiTF1.h"
#include "TH1.h"
#include "TMath.h"
#include "TRandom.h"
#include "TROOT.h"

#include<iostream>
#include<chrono>

constexpr int paramSize = 6;

bool compareResult(double v1, double v2, std::string s = "", double tol = 0.01)
{
   // compare v1 with reference v2
   //  // give 1% tolerance
   if (std::abs(v1 - v2) < tol * std::abs(v2)) return true;
   std::cerr << s << " Failed comparison of fit results \t logl = " << v1 << "   it should be = " << v2 << std::endl;
   return false;
}

//Functor for a Higgs Fit normalized with analytical integral
template<class T>
class Func {
public:

   Func()
   {
      params.resize(paramSize);
   }
   T operator()(const T *data, const Double_t *p)
   {
      bool changed = false;
      for (unsigned i = 0; i < params.size(); i++) {
         R__LOCKGUARD(gROOTMutex);
         if (p[i] != params[i])
            if (!changed) {
               changed = true;
            }

      }

      if (changed) {
         R__LOCKGUARD(gROOTMutex);
         for (unsigned i = 0; i < paramSize; i++) {
            params[i] = p[i];
         }
         auto funcInt1 = [&](double x) {
            return 50.*TMath::Sqrt(TMath::Pi()) * exp((p[4] * p[4]) / (4 * p[5])) * TMath::Erf((0.5 * p[4] + p[5] * 0.01 * x) / (TMath::Sqrt(p[5])))
                   / (TMath::Sqrt(p[5]));
         };
         auto funcInt2 = [&](double x) {
            return -p[3] * TMath::Sqrt(TMath::Pi() / 2.) * TMath::Erf((p[2] - x) / (TMath::Sqrt(2.) * p[3]));
         };
         integral1 = funcInt1(200) - funcInt1(100);
         integral2 = funcInt2(200) - funcInt2(100);
      }

      auto f1 = params[0] * exp(-(*data + (-p[2])) * (*data + (-p[2])) / (2.*p[3] * p[3]));

      auto f2 = (1 - params[0]) * exp(-(params[4] * (*data * (0.01)) +
                                        params[5] * ((*data) * (0.01)) * ((*data) * (0.01))));

      f1 /= integral2 != 0 ? integral2 : 1;
      f2 /= integral1 != 0 ? integral1 : 1;
      return f1 + f2;
   }

private:

   double integral1 = 1.;
   double integral2 = 1.;
   std::vector<double> params;
};

//Test class with functions for each of the cases
class TestVector {
public:
   TestVector(unsigned nPoints)
   {
      fSeq =  new TF1("fseq", Func<double>(), 100, 200, paramSize);
      wfSeq =  new ROOT::Math::WrappedMultiTF1Templ<double>(*fSeq);

#ifdef R__HAS_VECCORE
      fVec = new TF1("fvCore", Func<ROOT::Double_v>(), 100, 200, paramSize);
      wfVec =  new ROOT::Math::WrappedMultiTF1Templ<ROOT::Double_v>(*fVec);
#endif

      dataSB = new ROOT::Fit::UnBinData(nPoints);

      fSeq->SetParameters(p);
      if (!filledData) {
         for (unsigned i = 0; i < nPoints; ++i) {
            double x = fSeq->GetRandom();
            dataSB->Add(x);
         }
         filledData = true;
      }
      fitter.Config().MinimizerOptions().SetPrintLevel(3);
      fitter.Config().SetMinimizer("Minuit2", "Migrad");
   }



   double testFitSeq()
   {
      std::cout << "\n////////////////////////////SEQUENTIAL TEST////////////////////////////" << std::endl << std::endl;
      fSeq->SetParameters(p);
      fitter.SetFunction(*wfSeq, false);
      fitter.Config().ParSettings(0).SetLimits(0, 1);
      fitter.Config().ParSettings(1).Fix();
      fitter.Config().ParSettings(3).SetLowerLimit(0);
      fitter.Config().ParSettings(4).SetLowerLimit(0);
      fitter.Config().ParSettings(5).SetLowerLimit(0);
      start = std::chrono::system_clock::now();
      bool ret = fitter.Fit(*dataSB);
      end =  std::chrono::system_clock::now();
      duration = end - start;
      std::cout << std::endl << "Time for the sequential test: " << duration.count() << std::endl;
      return  ret;
   }

   double testMTFit()
   {
      std::cout << "\n///////////////////////////////MT TEST////////////////////////////" << std::endl << std::endl;
      fSeq->SetParameters(p);
      fitter.SetFunction(*wfSeq, false);
      fitter.Config().ParSettings(0).SetLimits(0, 1);
      fitter.Config().ParSettings(1).Fix();
      fitter.Config().ParSettings(3).SetLowerLimit(0);
      fitter.Config().ParSettings(4).SetLowerLimit(0);
      fitter.Config().ParSettings(5).SetLowerLimit(0);
      start = std::chrono::system_clock::now();
      bool ret = fitter.Fit(*dataSB, 0, ROOT::Fit::kMultithread);
      end =  std::chrono::system_clock::now();
      duration = end - start;
      std::cout << "Time for the parallel test: " << duration.count() << std::endl;
      return ret;
   }

   double testMPFit()
   {
      std::cout << "\n///////////////////////////////MP TEST////////////////////////////\n\n";
      fSeq->SetParameters(p);
      fitter.SetFunction(*wfSeq, false);
      start = std::chrono::system_clock::now();
      bool ret = fitter.Fit(*dataSB, 0, ROOT::Fit::kMultiprocess);
      end =  std::chrono::system_clock::now();
      duration = end - start;
      std::cout << "Time for the multiprocess test:" << duration.count() << std::endl;
      return ret;
   }

#ifdef R__HAS_VECCORE
   double testFitVec()
   {
      std::cout << "\n////////////////////////////VECTOR TEST////////////////////////////" << std::endl << std::endl;
      fVec->SetParameters(p);
      fitter.SetFunction(*wfVec);
      fitter.Config().ParSettings(0).SetLimits(0, 1);
      fitter.Config().ParSettings(1).Fix();
      fitter.Config().ParSettings(3).SetLowerLimit(0);
      fitter.Config().ParSettings(4).SetLowerLimit(0);
      fitter.Config().ParSettings(5).SetLowerLimit(0);
      start = std::chrono::system_clock::now();
      bool ret = fitter.Fit(*dataSB);
      end =  std::chrono::system_clock::now();
      duration = end - start;
      std::cout << "Time for the vectorized test: " << duration.count() << std::endl;
      return ret;
   }

   double testMTFitVec()
   {
      std::cout << "\n///////////////////////////////MT+VEC TEST////////////////////////////\n\n";
      fVec->SetParameters(p);
      fitter.SetFunction(*wfVec);
      start = std::chrono::system_clock::now();
      bool ret = fitter.Fit(*dataSB, 0, ROOT::Fit::kMultithread);
      end =  std::chrono::system_clock::now();
      duration = end - start;
      std::cout << "Time for the parallel+vectorized test: " << duration.count() << std::endl;
      return ret;
   }

   double testMPFitVec()
   {
      std::cout << "\n///////////////////////////////MP+VEC TEST////////////////////////////\n\n";
      fVec->SetParameters(p);
      fitter.SetFunction(*wfVec);
      start = std::chrono::system_clock::now();
      bool ret = fitter.Fit(*dataSB, 0, ROOT::Fit::kMultiprocess);
      end =  std::chrono::system_clock::now();
      duration = end - start;
      std::cout << "Time for the multiprocess+vectorized test:" << duration.count() << std::endl;
      return ret;
   }
#endif
   const ROOT::Fit::Fitter &GetFitter()
   {
      return fitter;
   }

private:
   TF1 *fSeq;
   ROOT::Math::WrappedMultiTF1Templ<double> *wfSeq;
#ifdef R__HAS_VECCORE
   TF1 *fVec;
   ROOT::Math::WrappedMultiTF1Templ<ROOT::Double_v> *wfVec;
#endif
   std::chrono::time_point<std::chrono::system_clock> start, end;
   std::chrono::duration<double> duration;
   ROOT::Fit::Fitter fitter;
   ROOT::Fit::UnBinData *dataSB;
   bool filledData = false;
   double p[paramSize] = {0.1, 1000., 130., 2., 3.5, 1.5};
};


int main()
{
   TestVector test(200000);

   //Sequential
   if (!test.testFitSeq()) {
      Error("testLogLExecPolicy", "Fit failed!");
      return -1;
   }

#if defined(R__USE_IMT) || defined(R__HAS_VECCORE)
   auto seq = test.GetFitter().Result().MinFcnValue();
#endif

// #ifdef R__USE_IMT
//    //Multithreaded
//    if (!test.testMTFit()) {
//       Error("testLogLExecPolicy", "Multithreaded Fit failed!");
//       return -1;
//    }
//    auto seqMT = test.GetFitter().Result().MinFcnValue();
//    if (!compareResult(seqMT, seq, "Mutithreaded LogL Fit: "))
//       return 1;
// #endif

#ifdef R__HAS_VECCORE
   //Vectorized
   if (!test.testFitVec()) {
      Error("testLogLExecPolicy", "Vectorized Fit failed!");
      return -1;
   }
   auto vec = test.GetFitter().Result().MinFcnValue();
   if (!compareResult(vec, seq, "vectorized LogL Fit: "))
      return 2;
#endif

// #if defined(R__USE_IMT) && defined(R__HAS_VECCORE)
//    //Multithreaded and vectorized
//    if (!test.testMTFitVec()) {
//       Error("testLogLExecPolicy", "Multithreaded + vectorized Fit failed!");
//       return -1;
//    }
//    auto vecMT = test.GetFitter().Result().MinFcnValue();
//    if (!compareResult(vecMT, seq, "Mutithreaded + vectorized LogL Fit: "))
//       return 3;
// #endif
   return 0;
}
