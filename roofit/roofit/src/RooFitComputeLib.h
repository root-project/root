#ifndef ROOFITCOMPUTELIB_H
#define ROOFITCOMPUTELIB_H

#include "RooFitComputeInterface.h"

namespace RooFitCompute {

  namespace RF_ARCH {
    class RooFitComputeClass : RooFitComputeInterface {
      public:
        RooFitComputeClass();
        ~RooFitComputeClass() override {}
        RooSpan<double> computeArgusBG(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> m, RooSpan<const double> m0, RooSpan<const double> c, RooSpan<const double> p) override;
        void computeBernstein(size_t batchSize, double * __restrict output, const double * __restrict const xData, double xmin, double xmax, std::vector<double> coef) override;
        RooSpan<double> computeBifurGauss(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> mean, RooSpan<const double> sigmaL, RooSpan<const double> sigmaR) override;
        RooSpan<double> computeBukin(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> Xp, RooSpan<const double> sigp, RooSpan<const double> xi, RooSpan<const double> rho1, RooSpan<const double> rho2) override;
        RooSpan<double> computeBreitWigner(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> mean, RooSpan<const double> width) override;
        RooSpan<double> computeCBShape(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> m, RooSpan<const double> m0, RooSpan<const double> sigma, RooSpan<const double> alpha, RooSpan<const double> n) override;
        void computeChebychev(size_t batchSize, double * __restrict output, const double * __restrict const xData, double xmin, double xmax, std::vector<double> coef) override;
        RooSpan<double> computeChiSquare(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> ndof) override;
        RooSpan<double> computeDstD0BG(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> dm, RooSpan<const double> dm0, RooSpan<const double> C, RooSpan<const double> A, RooSpan<const double> B) override;
        RooSpan<double> computeExponential(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> c) override;
        RooSpan<double> computeGamma(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> gamma, RooSpan<const double> beta, RooSpan<const double> mu) override;
        RooSpan<double> computeGaussian(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> mean, RooSpan<const double> sigma) override;
        RooSpan<double> computeJohnson(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> mass, RooSpan<const double> mu, RooSpan<const double> lambda, RooSpan<const double> gamma, RooSpan<const double> delta, double massThreshold) override;
        RooSpan<double> computeLandau(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> mean, RooSpan<const double> sigma) override;
        RooSpan<double> computeLognormal(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> m0, RooSpan<const double> k) override;
        RooSpan<double> computeNovosibirsk(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> peak, RooSpan<const double> width, RooSpan<const double> tail) override;
        RooSpan<double> computePoisson(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> mean, bool protectNegative, bool noRounding) override;
        void computePolynomial(size_t batchSize, double* __restrict output, const double* __restrict const xData, int lowestOrder, std::vector<BatchHelpers::BracketAdapterWithMask> &coef) override;
        RooSpan<double> computeVoigtian(const RooAbsReal*, BatchHelpers::RunContext&, RooSpan<const double> x, RooSpan<const double> mean, RooSpan<const double> width, RooSpan<const double> sigma) override;
      };
  };
} // end namespace RooFitCompute

#endif
