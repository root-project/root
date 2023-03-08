#include "Math/Random.h"
#include "Math/TRandomEngine.h"
#include "Math/MersenneTwisterEngine.h"
#include "Math/MixMaxEngine.h"
#include "Math/RanluxppEngine.h"
#include "Math/StdEngine.h"
//#include "Math/MyMixMaxEngine.h"
//#include "Math/GSLRndmEngines.h"
#include "Math/GoFTest.h"
#include "Math/ProbFuncMathCore.h"
#include "TH1.h"
#include "TVirtualPad.h"

#include "TRandom1.h"
#include "TRandom2.h"
#include "TRandom3.h"
//#include "TRandomNew3.h"

#include "TStopwatch.h"
#include <iostream>

#include <random>

using namespace ROOT::Math;

const long long NR = 1E6;
double minPValue = 1.E-3;

bool showPlots = false;

inline void fillBits(int64_t *v, uint64_t x)
{
   for (int i = 0; i < 64; ++i) {
      v[i] += x & 1;
      x >>= 1;
   }
}

template <typename Engine>
bool testInt()
{
   Engine engine;
   bool ret = true;
   static_assert((~0ULL) >> (64 - Engine::kNumberOfBits) == Engine::MaxInt(), "MaxInt inconsistent with kNumberOfBits");
   static_assert(0 == Engine::MinInt(), "MinInt not 0");
#ifdef _WIN64
   static_assert(Engine::kNumberOfBits == _popcount64(Engine::MaxInt()), "MaxInt inconsistent with kNumberOfBits");
#else
   static_assert(Engine::kNumberOfBits == __builtin_popcountll(Engine::MaxInt()),
                 "MaxInt inconsistent with kNumberOfBits");
   static_assert(64 - Engine::kNumberOfBits == __builtin_clzll(Engine::MaxInt()),
                 "MaxInt inconsistent with kNumberOfBits");
#endif

   std::cout << "testing integer engine " << engine.Name() << std::endl;
   int N = 2 * 1000 * 1000;
   int64_t vr[64];
   for (int i = 0; i < 64; ++i)
      vr[i] = 0;
   for (int i = 0; i < N; ++i) {
      auto r = engine.IntRndm();
      fillBits(vr, r);
   }
   int64_t t = 1000 * 1000;
   for (uint32_t i = 0; i < 64; ++i) {
      if (i >= Engine::kNumberOfBits && vr[i] != 0) {
         std::cout << "not zero? " << i << ' ' << vr[i] << std::endl;
         ret = false;
      }
      if (i < Engine::kNumberOfBits && std::abs(vr[i] - t) > 3000) {
         std::cout << "off? " << i << ' ' << vr[i] << std::endl;
         ret = false;
      }
   }
   return ret;
}

bool testCompatibility(const std::vector<double> &x, const std::vector<double> &y, double xmin = 0, double xmax = 1)
{

   GoFTest gof(x.size(), x.data(), y.size(), y.data());

   bool ok = true;
   double pvalue = gof.KolmogorovSmirnov2SamplesTest();
   if (pvalue < minPValue) {
      std::cout << "KS Test failed with p-value " << pvalue << std::endl;
      ok = false;
   } else {
      std::cout << "KS Test  :  OK - pvalue = " << pvalue << std::endl;
   }

   if (NR < 10000) {
      pvalue = gof.AndersonDarling2SamplesTest();
      if (pvalue < minPValue) {
         std::cout << "AD Test failed with p-value " << pvalue << std::endl;
         ok = false;
      } else {
         std::cout << "AD Test  :  OK - pvalue = " << pvalue << std::endl;
      }
   }

   // do a chi2 binned test
   int nbin = TMath::Min(x.size(), y.size()) / 1000;
   TH1D *h1 = new TH1D("h1", "h1", nbin, xmin, xmax);
   TH1D *h2 = new TH1D("h2", "h2", nbin, xmin, xmax);
   h1->FillN(x.size(), x.data(), nullptr);
   h2->FillN(y.size(), y.data(), nullptr);

   pvalue = h1->Chi2Test(h2);
   if (pvalue < minPValue) {
      std::cout << "Chi2 Test failed with p-value " << pvalue << std::endl;
      // showPlots = true;
      ok = false;
   } else {
      std::cout << "Chi2 Test:  OK - pvalue = " << pvalue << std::endl;
   }
   if (showPlots) {
      h1->Draw();
      h2->SetLineColor(kRed);
      h2->Draw("SAME");
      if (gPad)
         gPad->Update();
   } else {
      delete h1;
      delete h2;
   }

   return ok;
}

template <class R1, class R2>
bool testUniform(R1 &r1, R2 &r2)
{

   std::vector<double> x(NR);
   std::vector<double> y(NR);

   TStopwatch w;
   w.Start();
   for (int i = 0; i < NR; ++i)
      x[i] = r1();

   w.Stop();
   std::cout << "time for uniform filled for " << typeid(r1).name();
   w.Print();

   w.Start();
   for (int i = 0; i < NR; ++i)
      y[i] = r2();

   w.Stop();
   std::cout << "time for uniform filled for " << typeid(r2).name();
   w.Print();

   return testCompatibility(x, y);
}

template <class R1, class R2>
bool testGauss(R1 &r1, R2 &r2)
{

   std::vector<double> x(NR);
   std::vector<double> y(NR);

   TStopwatch w;
   w.Start();
   for (int i = 0; i < NR; ++i)
      x[i] = ROOT::Math::normal_cdf(r1.Gaus(0, 1), 1);

   w.Stop();
   std::cout << "time for GAUS filled for " << typeid(r1).name();
   w.Print();

   w.Start();
   for (int i = 0; i < NR; ++i)
      y[i] = ROOT::Math::normal_cdf(r2.Gaus(0, 1), 1);

   w.Stop();
   std::cout << "time for GAUS filled for " << typeid(r2).name();
   w.Print();

   return testCompatibility(x, y);
}

bool test1()
{

   bool ret = true;
   std::cout << "\nTesting MT vs MIXMAX " << std::endl;

   Random<MersenneTwisterEngine> rmt;
   Random<MixMaxEngine240> rmx;
   ret &= testUniform(rmx, rmt);
   ret &= testGauss(rmx, rmt);
   return ret;
}

bool test2()
{

   bool ret = true;

   std::cout << "\nTesting MIXMAX240 vs MIXMAX256" << std::endl;

   Random<MixMaxEngine240> rmx1(1111);
   Random<MixMaxEngine<256, 2>> rmx2(2222);

   ret &= testUniform(rmx1, rmx2);
   ret &= testGauss(rmx1, rmx2);
   return ret;
}

bool test3()
{

   bool ret = true;

   std::cout << "\nTesting MIXMAX240 vs MIXMAX17" << std::endl;

   Random<MixMaxEngine240> rmx1(1111);
   Random<MixMaxEngine<17, 0>> rmx2(2222);

   ret &= testUniform(rmx1, rmx2);
   ret &= testGauss(rmx1, rmx2);
   return ret;
}

bool test4()
{

   bool ret = true;

   std::cout << "\nTesting MIXMAX240 vs MIXMAX240 using different seeds" << std::endl;

   Random<MixMaxEngine240> rmx1(1111);
   Random<MixMaxEngine240> rmx2(2222);

   ret &= testUniform(rmx1, rmx2);
   ret &= testGauss(rmx1, rmx2);
   return ret;
}

bool test5()
{

   bool ret = true;
   std::cout << "\nTesting RanLun++ vs MIXMAX " << std::endl;

   Random<RanluxppEngine2048> rmt;
   Random<MixMaxEngine240> rmx;
   ret &= testUniform(rmx, rmt);
   ret &= testGauss(rmx, rmt);
   return ret;
}

bool test6()
{

   bool ret = true;
   std::cout << "\nTesting std::mt19937_64  vs MIXMAX " << std::endl;

   Random<StdEngine<std::mt19937_64>> rmt;
   Random<MixMaxEngine240> rmx;
   ret &= testUniform(rmx, rmt);
   //    ret &= testGauss(rmx, rmt);
   return ret;
}

bool testMathRandom()
{

   bool ret = true;
   std::cout << "testing generating " << NR << " numbers " << std::endl;

   ret &= test1();
   ret &= test2();
   ret &= test3();
   ret &= test4();
   ret &= test5();
   ret &= test6();

   ret &= testInt<MersenneTwisterEngine>();
   ret &= testInt<RanluxppEngine2048>();
   ret &= testInt<MixMaxEngine17>();
   ret &= testInt<MixMaxEngine240>();
   ret &= testInt<StdEngine<std::mt19937_64>>();

   if (!ret)
      Error("testMathRandom", "Test Failed");
   else
      std::cout << "\nTestMathRandom:  OK \n";
   return ret;
}

int main()
{
   bool ret = testMathRandom();
   return (ret) ? 0 : -1;
}
