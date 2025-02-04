#include "xRooFit/xRooFit.h"

#include <cmath>
#include "Math/ProbFunc.h"
#include "Math/BrentRootFinder.h"
#include "Math/WrappedFunction.h"

#include "RooStats/RooStatsUtils.h"

BEGIN_XROOFIT_NAMESPACE

double xRooFit::Asymptotics::k(const IncompatFunc &compatRegions, double pValue, double poiVal, double poiPrimeVal,
                               double sigma, double low, double high)
{

   // determine the pll value corresponding to nSigma expected - i.e. where the altPValue equals e.g. 50% for nSigma=0,
   // find the solution (wrt x) of: FitManager::altPValue(x, var(poi), alt_val, _sigma_mu, _compatibilityFunction) -
   // targetPValue = 0
   double targetTailIntegral = pValue; // ROOT::Math::normal_cdf(*nSigma);

   // check how much of the alt distribution density is in the delta function @ 0
   // if more than 1 - target is in there, if so then pll must be 0
   double prob_in_delta = Phi_m(poiVal, poiPrimeVal, std::numeric_limits<double>::infinity(), sigma, compatRegions);
   // also get a contribution to the delta function for mu_hat < mu_L IF mu==mu_L
   if (poiVal == low) {
      // since mu_hat is gaussian distributed about mu_prime with std-dev = sigma
      // the integral is Phi( mu_L - mu_prime / (sigma) )
      double mu_L = low;
      prob_in_delta += ROOT::Math::normal_cdf((mu_L - poiPrimeVal) / sigma);
   }

   if (prob_in_delta > 1 - targetTailIntegral) {
      return 0;
   }

   struct TailIntegralFunction {
      TailIntegralFunction(double _poiVal, double _alt_val, double _sigma_mu, double _low, double _high,
                           IncompatFunc _compatibilityFunction, double _target)
         : poiVal(_poiVal),
           alt_val(_alt_val),
           sigma_mu(_sigma_mu),
           low(_low),
           high(_high),
           target(_target),
           cFunc(_compatibilityFunction)
      {
      }

      double operator()(double x) const
      {
         double val = PValue(cFunc, x, poiVal, alt_val, sigma_mu, low, high);
         if (val < 0)
            kInvalid = true;
         return val - target;
      }

      double poiVal, alt_val, sigma_mu, low, high, target;
      IncompatFunc cFunc;
      mutable bool kInvalid = false;
   };

   TailIntegralFunction f(poiVal, poiPrimeVal, sigma, low, high, compatRegions, targetTailIntegral);
   ROOT::Math::BrentRootFinder brf;
   ROOT::Math::WrappedFunction<TailIntegralFunction &> wf(f);

   auto tmpLvl = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kFatal;
   double _pll = 500.;
   double currVal(1.);
   int tryCount(0);
   double _prev_pll = _pll;
   do {
      currVal = wf(_pll);
      if (currVal > 1e-4) {
         _pll = 2. * (_pll + 1.); // goto bigger pll scale
      } else if (currVal < -1e-4) {
         _pll /= 2.; // goto smaller pll scale
      }
      // std::cout << "pll = " << _pll << " currVal = " << currVal << std::endl;
      brf.SetFunction(wf, 0, _pll);
      if (brf.Solve()) {
         _prev_pll = _pll;
         _pll = brf.Root();
      }
      if (f.kInvalid) { // happens if problem evaluating PValue (e.g. sigma was nan)
         _pll = std::numeric_limits<double>::quiet_NaN();
         break;
      }
      // std::cout << " -- " << brf.Root() << " " << FitManager::altPValue(_pll, mu, alt_val, sigma, pllModifier()) << "
      // >> " << wf(_pll) << std::endl;
      tryCount++;
      if (tryCount > 20) {
         gErrorIgnoreLevel = tmpLvl;
         Warning("Asymptotics::k", "Reached limit pValue=%g pll=%g", pValue, _pll);
         break;
      }
   } while (std::abs(wf(_pll)) > 1e-4 && std::abs(wf(_pll)) < std::abs(wf(_prev_pll)) * 0.99);
   gErrorIgnoreLevel = tmpLvl;
   return _pll;
}

double xRooFit::Asymptotics::PValue(const IncompatFunc &compatRegions, double k, double poiVal, double poi_primeVal,
                                    double sigma, double lowBound, double upBound)
{
   // uncapped test statistic is equal to onesidednegative when k is positive, and equal to 1.0 - difference between
   // twosided and onesidednegative when k is negative ...
   if (compatRegions == IncompatibilityFunction(Uncapped, poiVal)) {
      // if(k==0) return 0.5;
      if (k > 0)
         return PValue(OneSidedNegative, k, poiVal, poi_primeVal, sigma, lowBound, upBound);
      return 1. - (PValue(TwoSided, -k, poiVal, poi_primeVal, sigma, lowBound, upBound) -
                   PValue(OneSidedNegative, -k, poiVal, poi_primeVal, sigma, lowBound, upBound));
   }

   // if(k<0) return 1.;
   if (k <= 0) {
      if (compatRegions == IncompatibilityFunction(OneSidedNegative, poiVal) && std::abs(poiVal - poi_primeVal) < 1e-9)
         return 0.5; // when doing discovery (one-sided negative) use a 0.5 pValue
      return 1.;     // case to catch the delta function that ends up at exactly 0 for the one-sided tests
   }

   if (sigma == 0) {
      if (lowBound != -std::numeric_limits<double>::infinity() || upBound != std::numeric_limits<double>::infinity()) {
         return -1;
      } else if (std::abs(poiVal - poi_primeVal) > 1e-12) {
         return -1;
      }
   }

   // get the poi value that defines the test statistic, and the poi_prime hypothesis we are testing
   // when setting limits, these are often the same value

   double Lambda_y = 0;
   if (std::abs(poiVal - poi_primeVal) > 1e-12)
      Lambda_y = (poiVal - poi_primeVal) / sigma;

   if (std::isnan(Lambda_y))
      return -1;

   double k_low = (lowBound == -std::numeric_limits<double>::infinity()) ? std::numeric_limits<double>::infinity()
                                                                         : pow((poiVal - lowBound) / sigma, 2);
   double k_high = (upBound == std::numeric_limits<double>::infinity()) ? std::numeric_limits<double>::infinity()
                                                                        : pow((upBound - poiVal) / sigma, 2);

   double out = Phi_m(poiVal, poi_primeVal, std::numeric_limits<double>::infinity(), sigma, compatRegions) - 1;

   double out2 = 0; // use to hold small corrections (terms in "out" are O(1), which ruins precision)

   if (out < -1) {
      // compatibility function is unsupported, return negative
      return -2;
   }

   // go through the 4 'regions' ... only two of which will apply
   if (k <= k_high) {
      out2 += ROOT::Math::gaussian_cdf_c(sqrt(k) + Lambda_y);
      out += 1.0 -
             Phi_m(poiVal, poi_primeVal, Lambda_y + sqrt(k), sigma, compatRegions);
   } else {
      double Lambda_high = (poiVal - upBound) * (poiVal + upBound - 2. * poi_primeVal) / (sigma * sigma);
      double sigma_high = 2. * (upBound - poiVal) / sigma;
      out2 += ROOT::Math::gaussian_cdf_c((k - Lambda_high) / sigma_high);
      out += 1.0 -
             Phi_m(poiVal, poi_primeVal, (k - Lambda_high) / sigma_high, sigma, compatRegions);
   }

   if (k <= k_low) {
      out2 += ROOT::Math::gaussian_cdf_c(sqrt(k) - Lambda_y);
      out += 1.0 +
             Phi_m(poiVal, poi_primeVal, Lambda_y - sqrt(k), sigma, compatRegions);
   } else {
      double Lambda_low = (poiVal - lowBound) * (poiVal + lowBound - 2. * poi_primeVal) / (sigma * sigma);
      double sigma_low = 2. * (poiVal - lowBound) / sigma;
      out2 += ROOT::Math::gaussian_cdf_c((k - Lambda_low) / sigma_low);
      out += 1.0 +
             Phi_m(poiVal, poi_primeVal, (Lambda_low - k) / sigma_low, sigma, compatRegions);
      /*out +=  ROOT::Math::gaussian_cdf((k-Lambda_low)/sigma_low) +
           2*Phi_m(poiVal,poi_primeVal,(Lambda_low - k_low)==0 ? 0 : ((Lambda_low -
         k_low)/sigma_low),sigma,compatRegions)
           - Phi_m(poiVal,poi_primeVal,(Lambda_low - k)/sigma_low,sigma,compatFunc);
*/

      // handle case where poiVal = lowBound (e.g. testing mu=0 when lower bound is mu=0).
      // sigma_low will be 0 and gaussian_cdf will end up being 1, but we need it to converge instead
      // to 0.5 so that pValue(k=0) converges to 1 rather than 0.5.
      // handle this by 'adding' back in the lower bound
      // TODO: Think more about this?
      /*if (sigma_low == 0) {
          out -= 0.5;
      }*/
   }

   out = 1.0 - out;
   return out2 + out;
}

double
xRooFit::Asymptotics::Phi_m(double /*mu*/, double mu_prime, double a, double sigma, const IncompatFunc &compatRegions)
{

   if (sigma == 0)
      sigma = 1e-100; // avoid nans if sigma is 0

   // want to evaluate gaussian integral in regions where IncompatFunc = 0

   double out = 0;
   double lowEdge = std::numeric_limits<double>::quiet_NaN();
   for (auto &transition : compatRegions) {
      if (transition.first >= (a * sigma + mu_prime))
         break;
      if (transition.second == 0 && std::isnan(lowEdge)) {
         lowEdge = transition.first;
      } else if (!std::isnan(lowEdge)) {
         out += ROOT::Math::gaussian_cdf((transition.first - mu_prime) / sigma) -
                ROOT::Math::gaussian_cdf((lowEdge - mu_prime) / sigma);
         lowEdge = std::numeric_limits<double>::quiet_NaN();
      }
   }
   if (!std::isnan(lowEdge)) { // also catches case where last transition is before a
      out += ROOT::Math::gaussian_cdf(a) - ROOT::Math::gaussian_cdf((lowEdge - mu_prime) / sigma);
   }

   return out;
}

int xRooFit::Asymptotics::CompatFactor(const IncompatFunc &func, double mu_hat)
{
   if (std::isnan(mu_hat))
      return 1; // nan is never compatible
   int out = 1;
   for (auto &transition : func) {
      if (transition.first > mu_hat)
         break;
      out = transition.second;
   }
   return out;
}

// RooRealVar xRooFit::Asymptotics::FindLimit(TGraph* pVals, double target_pVal) {
//     auto target_sig = RooStats::PValueToSignificance(target_pVal);
//     TGraph sig;
//     for(int i=0;i<pVals->GetN();i++) {
//         sig.SetPoint(i,pVals->GetPointX(i),RooStats::PValueToSignificance(pVals->GetPointY(i))-target_sig);
//     }
//     sig.Sort(); // ensure points are in x order
//     // now loop over and find where function crosses zero
//     for(int i=0;i<sig.GetN();i++) {
//         if (sig.GetPointY(i)>=0) {
//             if (i==0) return RooRealVar("limit","limit",std::numeric_limits<double>::quiet_NaN());
//             double prev_x = sig.GetPointX(i-1);
//             double next_x = sig.GetPointX(i);
//             double prev_y = sig.GetPointY(i-1);
//             double next_y = sig.GetPointY(i);
//             return RooRealVar("limit","limit", prev_x + (next_x-prev_x)*(-prev_y)/(next_y-prev_y));
//         }
//     }
//     return RooRealVar("limit","limit",std::numeric_limits<double>::quiet_NaN());
// }

END_XROOFIT_NAMESPACE
