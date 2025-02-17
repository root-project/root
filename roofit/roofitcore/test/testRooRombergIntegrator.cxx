// Tests for the RooRombergIntegrator.
// Authors: Stephan Hageboeck, CERN  05/2020
//          Jonas Rembser, CERN  08/2023

#include <RooRealBinding.h>
#include <RooRealVar.h>
#include <RooFormulaVar.h>
#include <RooNumIntConfig.h>
#include <RooHelpers.h>
#include <Math/ProbFuncMathCore.h>
#include <Math/SpecFuncMathCore.h>
#include <Math/Integrator.h>
#include <TMath.h>
#include <TStopwatch.h>

#include "gtest/gtest.h"

#include <numeric>
#include <algorithm>
#include <iomanip>

#include "../src/RooRombergIntegrator.h"

TEST(Roo1DIntegrator, RunFormulaVar_Trapezoid)
{

   RooRealVar x("x", "x", -100, 100);
   RooRealVar a("a", "a", 0.2, -100, 100);
   RooRealVar b("b", "b", 0.3, -100, 100);
   RooFormulaVar formula("formula", "0.1 + x*(a + b*x)", {x, a, b});
   auto solution = [&x](double aVal, double bVal) {
      auto indefInt = [=](double y) { return y * (0.1 + y * (1. / 2. * aVal + 1. / 3. * bVal * y)); };
      return indefInt(x.getMax()) - indefInt(x.getMin());
   };
   RooRealBinding binding(formula, {x, a, b});
   std::vector<double> yvals{a.getVal(), b.getVal()};

   // The integrators will warn, since we let them run until maxSteps
   RooHelpers::HijackMessageStream hijack(RooFit::WARNING, RooFit::Integration);

   // Test the recursion anchors of the Romberg integration
   {
      RooRombergIntegrator oneStep(binding, RooRombergIntegrator::Trapezoid, 1, 1.E-15);
      EXPECT_DOUBLE_EQ(oneStep.integral(yvals.data()), 0.5 * 200. * (2 * 0.1 + 2. * 0.3 * 10000.));
      x = -100.;
      const double left = formula.getVal();
      x = 100.;
      const double right = formula.getVal();
      // Run integral again, also to make sure that setting x has no effect:
      EXPECT_DOUBLE_EQ(oneStep.integral(yvals.data()), 0.5 * 200. * (left + right));

      RooRombergIntegrator twoStep(binding, RooRombergIntegrator::Trapezoid, 2, 1.E-15);
      x = 0.;
      const double middle = formula.getVal();
      EXPECT_DOUBLE_EQ(twoStep.integral(yvals.data()), 0.25 * 200. * (left + right) + 0.5 * 200. * middle);
   }

   // Now run many steps
   {
      constexpr unsigned int nSteps = 25;
      constexpr double relEps = 1.E-50;
      RooRombergIntegrator integrator(binding, RooRombergIntegrator::Trapezoid, nSteps, relEps);
      double inputs[] = {1., 3.123};
      double target = solution(inputs[0], inputs[1]);
      EXPECT_LT(std::abs(integrator.integral(inputs) - target) / target, 1.E-14);

      inputs[0] = a.getVal();
      inputs[1] = b.getVal();
      target = solution(inputs[0], inputs[1]);
      EXPECT_LT(std::abs(integrator.integral(inputs) - target) / target, 1.E-14);

      inputs[0] = 4.;
      inputs[1] = 5.;
      target = solution(inputs[0], inputs[1]);
      EXPECT_LT(std::abs(integrator.integral(inputs) - target) / target, 1.E-14);
   }
}

TEST(RooIntegrator1D, RunQuarticFormulaVar)
{
   constexpr unsigned int nSteps = 25;
   constexpr double relEps = 1.E-16;
   RooRealVar x("x", "x", -50, 50);
   RooRealVar a("a", "a", 0.2, -100, 100);
   RooRealVar b("b", "b", 0.3, -100, 100);
   RooRealVar c("c", "c", 0.4, -100, 100);
   RooRealVar d("d", "d", 0.5, -100, 100);
   RooFormulaVar formula("formula", "0.1 + x*(a + x*(b + x*(c + d * x)))", {x, a, b, c, d});
   auto solution = [&x](double aVal, double bVal, double cVal, double dVal) {
      auto indefInt = [=](double yVal) {
         return yVal * (0.1 + yVal * (1. / 2. * aVal +
                                      yVal * (1. / 3. * bVal + yVal * (1. / 4. * cVal + 1. / 5. * dVal * yVal))));
      };
      return indefInt(x.getMax()) - indefInt(x.getMin());
   };
   RooRealBinding binding(formula, {x});
   RooRombergIntegrator integrator(binding, RooRombergIntegrator::Trapezoid, nSteps, relEps);

   double target = solution(0.2, 0.3, 0.4, 0.5);
   EXPECT_LT(std::abs(integrator.integral() - target) / target, 1.E-13);
}

/// Run numeric integrations using RooRombergIntegrator and ROOT's adaptive integrator. Ensure that
/// they reach the requested precision.
void testConvergenceSettings(const RooFormulaVar &formula, const RooArgSet &observables,
                             const RooArgSet &funcParameters,
                             std::function<double(const double *pars, unsigned int nPar)> solution,
                             const std::string &name)
{
   constexpr unsigned int nSteps = 25;
   constexpr bool printDetails = false;
   SCOPED_TRACE(std::string("Function to integrate: ") + name + "\t" + formula.GetTitle());

   for (double relEps : {1.E-3, 1.E-6, 1.E-8}) {
      SCOPED_TRACE(std::string("relEps=" + std::to_string(relEps)));
      TStopwatch stopNew;
      TStopwatch stopRoot;

      RooRealVar &x = dynamic_cast<RooRealVar &>(*observables[0ul]);
      RooArgSet variables(observables);
      variables.add(funcParameters);
      RooRealBinding binding(formula, variables);
      RooRombergIntegrator integrator(binding, RooRombergIntegrator::Trapezoid, nSteps, relEps);

      std::vector<double> errors;
      std::vector<double> errorsRootInt;

      RooArgSet initialParameters;
      funcParameters.snapshot(initialParameters);

      constexpr unsigned int nRun = 10000;
      for (unsigned int i = 0; i < nRun; ++i) {
         std::vector<double> pars;
         for (const auto p : initialParameters) {
            auto par = static_cast<RooRealVar *>(p);
            pars.push_back(par->getVal() + (par->getMax() - par->getVal()) / nRun * i);
         }

         const double target = solution(pars.data(), pars.size());

         stopNew.Start(false);
         const double integral = integrator.integral(pars.data());
         stopNew.Stop();
         errors.push_back(std::abs((integral - target) / target));

         {
            std::vector<double> vars(1);
            std::copy(pars.begin(), pars.end(), std::back_inserter(vars));
            auto bindingROOTInt = [&vars, &binding](double xVal) {
               vars[0] = xVal;
               return binding(vars.data());
            };
            ROOT::Math::IntegratorOneDim rootIntegrator(bindingROOTInt, ROOT::Math::IntegrationOneDim::kADAPTIVE,
                                                        1.E-20, relEps, 100);

            stopRoot.Start(false);
            const double rootIntegral = rootIntegrator.Integral(x.getMin(), x.getMax());
            stopRoot.Stop();
            errorsRootInt.push_back(std::abs((rootIntegral - target) / target));
         }
      }

      auto mean = [](const std::vector<double> &vec) { return TMath::Mean(vec.begin(), vec.end()); };
      auto median = [&](const std::vector<double> &vec) { return TMath::Median(vec.size(), vec.data()); };
      auto q95_99 = [&](const std::vector<double> &vec) -> std::pair<double, double> {
         std::vector<double> q(2);
         std::vector<double> p(1, 0.95);
         p.push_back(0.99);
         TMath::Quantiles(vec.size(), q.size(), const_cast<double *>(vec.data()), q.data(), p.data(), false);
         return {q[0], q[1]};
      };

      const auto q95_99_int1D = q95_99(errors);
      const auto q95_99_intROOT = q95_99(errorsRootInt);

      if (printDetails) {
         std::cout << "Integrating " << name << ", relEps = " << relEps;
         std::cout << "\n\t    \t"
                   << "mean         \tmedian  \tq95     \tq99     \tmax";
         const std::vector<double> *vec = &errors;
         std::cout << "\n\tnew:\t" << mean(*vec) << "\t" << median(*vec) << "\t" << q95_99_int1D.first << "\t"
                   << q95_99_int1D.second << "\t"
                   << "\tt=" << stopNew.CpuTime();
         vec = &errorsRootInt;
         std::cout << "\n\tROOT:\t" << mean(*vec) << "\t" << median(*vec) << "\t" << q95_99_intROOT.first << "\t"
                   << q95_99_intROOT.second << "\t"
                   << "\tt=" << stopRoot.CpuTime();
         std::cout << std::endl;
      }

      // Depending on the function, the integrator precision doesn't reach the
      // actual precisiosn parameter, so we give it some headroom.
      const double mult = 100.0;

      EXPECT_LT(mean(errors), mult * relEps) << "RooRombergIntegrator should reach target precision.";
      EXPECT_LT(mean(errorsRootInt), mult * relEps) << "ROOT integrator should reach target precision.";

      EXPECT_LT(q95_99_int1D.first, mult * relEps) << "95% quantile of errors exceeds relEpsilon";
      EXPECT_LT(q95_99_int1D.second, 2. * mult * relEps) << "99% quantile of errors exceeds 2.*relEpsilon";

      EXPECT_LT(q95_99_intROOT.first, mult * relEps) << "95% quantile of errors exceeds relEpsilon";
      EXPECT_LT(q95_99_intROOT.second, 2. * mult * relEps) << "95% quantile of errors exceeds relEpsilon";
   }
}

/// Disabled because the integrator doesn't reach the asked precision. If this
/// behavior gets changed, this can be enabled.
TEST(RooIntegrator1D, ConvergenceSettings_log)
{
   RooRealVar x("x", "x", 0.1, 50);
   RooRealVar a("a", "a", 0.2, -100, 1.E5);
   RooFormulaVar formula("formula", "log(a*x)", {x, a});
   testConvergenceSettings(
      formula, {x}, {a},
      [&x](const double *pars, unsigned int nPar) {
         const double aVal = pars[0];
         if (nPar != 1)
            throw std::logic_error("Need npar == 1");

         auto indefInt = [=](double y) { return 1. / aVal * (y * log(y) - y); };
         return indefInt(aVal * x.getMax()) - indefInt(aVal * x.getMin());
      },
      "log(a*x)");
}

TEST(RooIntegrator1D, ConvergenceSettings_pol4)
{
   RooRealVar x2("x", "x", -10, 10);
   RooRealVar a2("a", "a", 0.2, -1., 1);
   RooRealVar b2("b", "b", -0.1, -1., 1);
   RooRealVar c2("c", "c", 0.02, -0.1, 0.1);
   RooRealVar d2("d", "d", -0., -0.01, 0.01);
   RooFormulaVar formula("formula", "0.1 + x*(a + x*(b + x*(c + d * x)))", {x2, a2, b2, c2, d2});
   testConvergenceSettings(
      formula, {x2}, {a2, b2, c2, d2},
      [&x2](const double *pars, unsigned int nPar) {
         const double a = pars[0];
         const double b = pars[1];
         const double c = pars[2];
         const double d = pars[3];
         if (nPar != 4)
            throw std::logic_error("Need npar == 4");

         auto indefInt = [=](double y) {
            return y * (0.1 + y * (1. / 2. * a + y * (1. / 3. * b + y * (1. / 4. * c + 1. / 5. * d * y))));
         };
         return indefInt(x2.getMax()) - indefInt(x2.getMin());
      },
      "Polynomial 4th order");
}

/// Disabled because the integrator doesn't reach the asked precision. If this
/// behavior gets changed, this can be enabled.
TEST(RooIntegrator1D, ConvergenceSettings_breitWig)
{
   RooRealVar x3("x", "x", 0., 100.);
   RooRealVar a3("a", "a", 10., 100.);
   RooRealVar b3("b", "b", 3., 10.);
   RooFormulaVar formula("breitwigner", "ROOT::Math::breitwigner_pdf(x, b, a)", {x3, a3, b3});
   testConvergenceSettings(
      formula, {x3}, {a3, b3},
      [&x3](const double *pars, unsigned int nPar) {
         const double a = pars[0];
         const double b = pars[1];
         if (nPar != 2)
            throw std::logic_error("Need npar == 2");

         return ROOT::Math::breitwigner_cdf(x3.getMax(), b, a) - ROOT::Math::breitwigner_cdf(x3.getMin(), b, a);
      },
      "Breit-Wigner distribution");
}

TEST(RooIntegrator1D, ConvergenceSettings_Erf)
{
   RooRealVar x("x", "x", -10., 10.);
   RooRealVar m("m", "m", -5., 5.);
   RooRealVar s("s", "s", 0.5, 10.);
   const std::string funcStr = "ROOT::Math::gaussian_pdf(x, s, m)";
   RooFormulaVar formula("gaus", funcStr.c_str(), {x, m, s});
   testConvergenceSettings(
      formula, {x}, {m, s},
      [&x](const double *pars, unsigned int nPar) {
         const double mVal = pars[0];
         const double sVal = pars[1];
         if (nPar != 2)
            throw std::logic_error("Need npar == 2");

         return ROOT::Math::gaussian_cdf(x.getMax(), sVal, mVal) - ROOT::Math::gaussian_cdf(x.getMin(), sVal, mVal);
      },
      "Gaussian distribution");
}

TEST(RooIntegrator1D, RunErf_NStep)
{
   RooHelpers::HijackMessageStream hijack(RooFit::WARNING, RooFit::Integration);
   constexpr double sigma = 1.3;
   constexpr double mean = 0.2;

   for (auto limits : std::initializer_list<std::pair<double, double>>{{0.1, 1.5}, {-0.3, 2.}, {2.5, 4.5}}) {
      const double min = limits.first;
      const double max = limits.second;
      RooRealVar theX("theX", "x", min, max);
      std::string funcStr =
         std::string("ROOT::Math::gaussian_pdf(theX, ") + std::to_string(sigma) + ", " + std::to_string(mean) + ")";
      RooFormulaVar gaus("gaus", funcStr.c_str(), theX);
      RooRealBinding binding(gaus, theX);

      double targetError = 0.;
      for (unsigned int nSteps = 4; nSteps < 24; ++nSteps) {
         RooRombergIntegrator integrator(binding, RooRombergIntegrator::Trapezoid, nSteps, 1.E-17);
         const double integral = integrator.integral();

         const double error = std::abs(
            integral - (ROOT::Math::gaussian_cdf(max, sigma, mean) - ROOT::Math::gaussian_cdf(min, sigma, mean)));

         if (nSteps == 4) {
            targetError = error;
         } else {
            // Error should go down faster than 0.5^nSteps because the integrator uses series acceleration,
            // so test if it goes down faster than 0.333^nSteps
            targetError /= 3.;
            // But cannot be better than double precision
            EXPECT_LT(error, std::max(targetError, 1.E-16)) << "For step " << nSteps << " with integral=" << integral;
         }
         if (nSteps > 10) {
            EXPECT_LT(error / integral, 1.E-4) << "For step " << nSteps << " with integral=" << integral;
         }
         if (nSteps > 15) {
            EXPECT_LT(error / integral, 1.E-6) << "For step " << nSteps << " with integral=" << integral;
         }
         if (nSteps > 21) {
            EXPECT_LT(error / integral, 1.E-8) << "For step " << nSteps << " with integral=" << integral;
         }
      }
   }
}

TEST(RooIntegrator1D, RunErf_Midpoint)
{
   const double min = 0;
   const double max = 1;
   RooRealVar theX("theX", "x", min, max);
   RooFormulaVar gaus("gaus", "ROOT::Math::gaussian_pdf(theX, 1, 0)", theX);
   RooRealBinding binding(gaus, theX);
   double targetError = 0.;

   RooHelpers::HijackMessageStream hijack(RooFit::WARNING, RooFit::Integration);

   for (unsigned int nSteps = 4; nSteps < 20; ++nSteps) {
      RooRombergIntegrator integrator(binding, RooRombergIntegrator::Midpoint, nSteps, 1.E-16);
      const double integral = integrator.integral();
      const double error =
         std::abs(integral - (ROOT::Math::gaussian_cdf(max, 1, 0) - ROOT::Math::gaussian_cdf(min, 1, 0)));
      if (nSteps == 4) {
         targetError = error;
      } else {
         // Error should go down faster than 2^nSteps because of series acceleration.
         targetError /= 3.;
         // But cannot be better than double precision
         EXPECT_LT(error, std::max(targetError, 1.E-16)) << "For step " << nSteps << " with integral=" << integral;
      }
      if (nSteps > 10) {
         EXPECT_LT(error / integral, 1.E-4) << "For step " << nSteps << " with integral=" << integral;
      }
      if (nSteps > 15) {
         EXPECT_LT(error / integral, 1.E-6) << "For step " << nSteps << " with integral=" << integral;
      }
   }
}

double testIntegrationMethod(int ndim, std::string const &label)
{
   constexpr bool verbose = false;

   RooRealVar x{"x", "x", 0, 10};
   RooRealVar y{"y", "y", 0, 10};

   std::string funcName = std::string("func") + label;
   RooFormulaVar func{funcName.c_str(), "x*std::sqrt(x) + y*std::sqrt(y) + x*y", {x, y}};

   if (verbose) {
      std::cout << label << ":" << std::endl;
   }

   RooNumIntConfig cfg(*func.getIntegratorConfig());

   if (ndim == 2) {
      cfg.method2D().setLabel(label.c_str());
   }
   if (ndim == 1) {
      cfg.method1D().setLabel(label.c_str());
   }

   RooArgSet iset{x};
   if (ndim > 1) {
      iset.add(y);
   }
   std::unique_ptr<RooAbsReal> integ{func.createIntegral(iset, RooFit::NumIntConfig(cfg))};
   double val = integ->getVal();
   if (verbose) {
      std::cout << std::setprecision(15) << val << std::endl;
      std::cout << std::endl;
   }

   return val;
}

TEST(RooRombergIntegrator, CompareToReference)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   std::vector<std::string> methods1d{"RooIntegrator1D", "RooSegmentedIntegrator1D"};
   std::vector<std::string> methods2d{"RooAdaptiveIntegratorND", "RooIntegrator2D", "RooSegmentedIntegrator2D"};

   // These are reference values obtained with ROOT 6.28.04
   // We compare the results exactly to these references to ensure that the
   // implementation has not changed accidentally. These values were printed
   // out with setprecision(15), but for the comparisons we only use
   // EXPECT_FLOAT_EQ because we are not bothered by foating point precision
   // problems here.
   std::vector<double> references1d{488.294986988088, 488.294680086881};
   std::vector<double> references2d{5029.82213550336, 5029.84276464679, 5029.82506801992};

   for (std::size_t i = 0; i < methods1d.size(); ++i) {
      double res = testIntegrationMethod(1, methods1d[i]);
      EXPECT_FLOAT_EQ(res, references1d[i]) << methods1d[i];
   }

   for (std::size_t i = 0; i < methods2d.size(); ++i) {
      double res = testIntegrationMethod(2, methods2d[i]);
      EXPECT_FLOAT_EQ(res, references2d[i]) << methods2d[i];
   }
}
