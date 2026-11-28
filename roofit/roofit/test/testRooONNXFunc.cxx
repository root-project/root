// Tests for the RooONNXFunc
// Authors: Jonas Rembser, CERN 2026

#include <RooONNXFunc.h>
#include <RooProduct.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooEvaluatorWrapper.h>
#include <RooHelpers.h>
#include <RooWorkspace.h>

#include <TFile.h>

#include "SofieLoader.h"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

// RooONNXFunc needs SOFIE, which is loaded at runtime
#define SKIP_IF_NO_SOFIE()                      \
   if (!RooFit::Detail::IsSofieAvailable()) {   \
      GTEST_SKIP() << "SOFIE is not available"; \
   }

namespace {

std::vector<double> readDoublesFromFile(const std::string &filename)
{
   std::vector<double> values;
   std::ifstream file(filename);

   if (!file) {
      std::cerr << "Error: Could not open file " << filename << "\n";
      return values;
   }

   double x;
   while (file >> x) {
      values.push_back(x);
   }

   return values;
}

void fillArgs(RooArgList &args, int n, double value = 0.1, std::string const &prefix = "")
{
   for (int i = 0; i < n; ++i) {
      auto v = std::make_unique<RooRealVar>((prefix + std::to_string(i)).c_str(), "", value, -10.0, 10.0);
      args.addOwned(std::move(v));
   }
}

} // namespace

/// Basic test for the evaluation of a RooONNXFunc with a single input
/// vector.
TEST(RooONNXFunc, Basic_1Tensor)
{
   SKIP_IF_NO_SOFIE();

   double refPred = readDoublesFromFile("regression_mlp_pred.txt")[0];

   RooArgList args;
   fillArgs(args, 10);

   RooONNXFunc roo_func{"func", "", {args}, "regression_mlp.onnx"};

   EXPECT_NEAR(roo_func.getVal(), refPred, 1e-5);
}

TEST(RooONNXFunc, Basic_2Tensors)
{
   SKIP_IF_NO_SOFIE();

   double refPred = readDoublesFromFile("regression_mlp_two_input_pred.txt")[0];

   RooArgList args0;
   fillArgs(args0, 10, 0.1, "a");
   RooArgList args1;
   fillArgs(args1, 5, 0.2, "b");

   RooONNXFunc roo_func{"func", "", {args0, args1}, "regression_mlp_two_input.onnx"};

   EXPECT_NEAR(roo_func.getVal(), refPred, 1e-5);
}

// Test the serialization to RooWorkspace. The ONNX payload will be embedded in
// the RooWorkspace as a binary blob.
TEST(RooONNXFunc, Basic_RooWorkspace)
{
   SKIP_IF_NO_SOFIE();

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::ObjectHandling, true};

   // Write to RooWorkspace
   {
      RooArgList args;
      fillArgs(args, 10);

      RooONNXFunc roo_func{"func", "", {args}, "regression_mlp.onnx"};
      RooWorkspace ws{"ws"};
      ws.import(roo_func);
      ws.writeToFile("RooONNXFunc_Basic.root");
   }

   // Read back and validate
   std::unique_ptr<TFile> file{TFile::Open("RooONNXFunc_Basic.root")};
   RooWorkspace *ws = dynamic_cast<RooWorkspace *>(file->Get("ws"));
   auto *roo_func = dynamic_cast<RooONNXFunc *>(ws->function("func"));

   double refPred = readDoublesFromFile("regression_mlp_pred.txt")[0];
   EXPECT_NEAR(roo_func->getVal(), refPred, 1e-5);
}

#ifdef ROOFIT_CLAD
/// Basic test for getting the analytic gradient of a RooONNXFunc with a
/// single input vector.
TEST(RooONNXFunc, Basic_CodegenAD)
{
   SKIP_IF_NO_SOFIE();

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::Fitting, true};

   double refPred = readDoublesFromFile("regression_mlp_pred.txt")[0];
   std::vector<double> refGrad = readDoublesFromFile("regression_mlp_grad_0.txt");

   RooArgList args;
   fillArgs(args, 10);

   RooONNXFunc roo_func{"func", "", {args}, "regression_mlp.onnx"};

   RooDataSet data("data", "data", {});

   RooFit::Experimental::RooEvaluatorWrapper roo_final{roo_func, &data, false, "", nullptr, false};

   EXPECT_NEAR(roo_final.getVal(), refPred, 1e-5);

   roo_final.generateGradient();

   std::vector<double> output_vec(10);

   roo_final.gradient(output_vec.data());
   roo_final.setUseGeneratedFunctionCode(true);
   // For debugging
   // roo_final.writeDebugMacro("codegen");

   for (int i = 0; i < 10; ++i) {
      EXPECT_NEAR(output_vec[i], refGrad[i], 1e-5);
   }

   // Zero out gradient output buffer and recalculate, just to check that no
   // internal state in not reset.
   for (int i = 0; i < 10; ++i) {
      output_vec[i] = 0.;
   }

   roo_final.gradient(output_vec.data());

   for (int i = 0; i < 10; ++i) {
      EXPECT_NEAR(output_vec[i], refGrad[i], 1e-5);
   }
}

/// Test the analytic gradient of a RooONNXFunc with two input tensors.
TEST(RooONNXFunc, Basic_CodegenAD_2Tensors)
{
   SKIP_IF_NO_SOFIE();

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::Fitting, true};

   double refPred = readDoublesFromFile("regression_mlp_two_input_pred.txt")[0];
   std::vector<double> refGrad0 = readDoublesFromFile("regression_mlp_two_input_grad_0.txt");
   std::vector<double> refGrad1 = readDoublesFromFile("regression_mlp_two_input_grad_1.txt");

   RooArgList args0;
   fillArgs(args0, 10, 0.1, "a");
   RooArgList args1;
   fillArgs(args1, 5, 0.2, "b");

   RooONNXFunc roo_func{"func", "", {args0, args1}, "regression_mlp_two_input.onnx"};

   RooDataSet data("data", "data", {});

   RooFit::Experimental::RooEvaluatorWrapper roo_final{roo_func, &data, false, "", nullptr, false};

   EXPECT_NEAR(roo_final.getVal(), refPred, 1e-5);

   roo_final.generateGradient();

   const std::size_t nTotal = 10 + 5;
   std::vector<double> output_vec(nTotal);

   roo_final.gradient(output_vec.data());
   roo_final.setUseGeneratedFunctionCode(true);

   for (int i = 0; i < 10; ++i) {
      EXPECT_NEAR(output_vec[i], refGrad0[i], 1e-5);
   }
   for (int i = 0; i < 5; ++i) {
      EXPECT_NEAR(output_vec[10 + i], refGrad1[i], 1e-5);
   }

   // Zero out gradient output buffer and recalculate, just to check that no
   // internal state is reset.
   for (std::size_t i = 0; i < nTotal; ++i) {
      output_vec[i] = 0.;
   }

   roo_final.gradient(output_vec.data());

   for (int i = 0; i < 10; ++i) {
      EXPECT_NEAR(output_vec[i], refGrad0[i], 1e-5);
   }
   for (int i = 0; i < 5; ++i) {
      EXPECT_NEAR(output_vec[10 + i], refGrad1[i], 1e-5);
   }
}

namespace {

/// Checks the value and the Clad Hessian of `func` with `n` parameters against
/// the references, and the gradient too if `refGrad` is not empty.
void checkCodegenHessian(RooAbsReal &func, std::size_t n, double refVal, std::vector<double> const &refHess,
                         std::vector<double> const &refGrad = {})
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::Fitting, true};

   RooDataSet data("data", "data", {});
   RooFit::Experimental::RooEvaluatorWrapper roo_final{func, &data, false, "", nullptr, false};

   EXPECT_NEAR(roo_final.getVal(), refVal, 1e-5);

   if (!refGrad.empty()) {
      roo_final.generateGradient();
      std::vector<double> grad(n);
      roo_final.gradient(grad.data());
      for (std::size_t i = 0; i < n; ++i) {
         EXPECT_NEAR(grad[i], refGrad[i], 1e-5);
      }
   }

   roo_final.generateHessian();
   std::vector<double> hess(n * n);
   roo_final.hessian(hess.data());
   for (std::size_t i = 0; i < n * n; ++i) {
      EXPECT_NEAR(hess[i], refHess[i], 1e-5);
   }
}

} // namespace

/// Validate the Clad Hessian of a RooONNXFunc against a PyTorch reference.
TEST(RooONNXFunc, CodegenHessian)
{
   SKIP_IF_NO_SOFIE();

   RooArgList args;
   fillArgs(args, 10);

   RooONNXFunc roo_func{"func", "", {args}, "regression_mlp_tanh.onnx"};

   checkCodegenHessian(roo_func, 10, readDoublesFromFile("regression_mlp_tanh_pred.txt")[0],
                       readDoublesFromFile("regression_mlp_tanh_hessian.txt"),
                       readDoublesFromFile("regression_mlp_tanh_grad_0.txt"));
}

/// Clad Hessian with two input tensors, including the mixed blocks.
TEST(RooONNXFunc, CodegenHessian_2Tensors)
{
   SKIP_IF_NO_SOFIE();

   RooArgList args0;
   fillArgs(args0, 10, 0.1, "a");
   RooArgList args1;
   fillArgs(args1, 5, 0.2, "b");

   RooONNXFunc roo_func{"func", "", {args0, args1}, "regression_mlp_two_input_tanh.onnx"};

   checkCodegenHessian(roo_func, 10 + 5, readDoublesFromFile("regression_mlp_two_input_tanh_pred.txt")[0],
                       readDoublesFromFile("regression_mlp_two_input_tanh_hessian.txt"));
}

/// Clad Hessian of the square of a RooONNXFunc, which makes the pushforward pullback's
/// primal-value adjoint d_y.value nonzero (it stays zero for a top-level RooONNXFunc).
TEST(RooONNXFunc, CodegenHessianCompound)
{
   SKIP_IF_NO_SOFIE();

   const std::size_t n = 10;
   double f = readDoublesFromFile("regression_mlp_tanh_pred.txt")[0];
   std::vector<double> g = readDoublesFromFile("regression_mlp_tanh_grad_0.txt");
   std::vector<double> hess = readDoublesFromFile("regression_mlp_tanh_hessian.txt");
   for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
         hess[i * n + j] = 2. * (f * hess[i * n + j] + g[i] * g[j]);
      }
   }

   RooArgList args;
   fillArgs(args, n);

   RooONNXFunc roo_func{"func", "", {args}, "regression_mlp_tanh.onnx"};
   RooProduct square{"square", "", {roo_func, roo_func}};

   checkCodegenHessian(square, n, f * f, hess);
}

/// Clad Hessian for inputs u = x^2, which makes the pushforward pullback's tangent
/// adjoints d_out_d_input nonzero (they stay zero if the parameters are the inputs).
TEST(RooONNXFunc, CodegenHessianNonlinearInputs)
{
   SKIP_IF_NO_SOFIE();

   const std::size_t n = 10;
   const double x = std::sqrt(0.1); // the reference inputs are u = 0.1
   std::vector<double> g = readDoublesFromFile("regression_mlp_tanh_grad_0.txt");
   std::vector<double> hess = readDoublesFromFile("regression_mlp_tanh_hessian.txt");
   for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
         hess[i * n + j] = 4. * x * x * hess[i * n + j] + (i == j ? 2. * g[i] : 0.);
      }
   }

   RooArgList xs;
   fillArgs(xs, n, x);
   RooArgList us;
   for (std::size_t i = 0; i < n; ++i) {
      auto &xi = xs[i];
      us.addOwned(std::make_unique<RooProduct>(("u" + std::to_string(i)).c_str(), "", RooArgList{xi, xi}));
   }

   RooONNXFunc roo_func{"func", "", {us}, "regression_mlp_tanh.onnx"};

   checkCodegenHessian(roo_func, n, readDoublesFromFile("regression_mlp_tanh_pred.txt")[0], hess);
}
#endif
